"""LightningModule + DataModule for the ELF pixel-vocab variant.

`ELFPixelLitModule` subclasses `ELFLitModule` so it reuses, byte-for-byte, the
parent's EMA, optimizer/LR schedule, checkpoint plumbing, and `_log_running`.
Only two things change:

  * `__init__` — no frozen T5 encoder; build an `ELFPixel` model instead. `x0`
    is produced deterministically by rendering text to a glyph strip and
    patchifying it (the glyph atlas IS the "frozen encoder").
  * `training_step` — the *minimal baseline*: a Bernoulli(decoder_prob) branch
    select between the pixel velocity-MSE denoiser and the OCR cross-entropy
    decoder. No self-conditioning / CFG / label-drop (those knobs stay off; the
    model + flow utils already support turning them on later).

Everything in `utils/sampling_utils.py` is reused: `add_noise`, `sample_timesteps`,
and `net_out_to_v_x` operate on the (B, N, P) patch sequence exactly as they do
on T5 (B, L, D) embeddings.
"""

from typing import Dict

import lightning as L
import torch
import torch.nn.functional as F

from lightning_module import ELFLitModule
from utils.sampling_utils import add_noise, net_out_to_v_x, sample_timesteps

from pixel.glyph_dataset import (
    ContinuousDocumentStreamDataset, load_text_dataset, make_glyph_collate_with_labels,
)
from pixel.glyph_vocab import IGNORE_INDEX, GlyphAtlasVocab
from pixel.model import ELFPixel_models, patchify


class ELFPixelLitModule(ELFLitModule):
    def __init__(self, config, vocab_size: int):
        # Skip ELFLitModule.__init__ (it builds a T5 encoder we don't want);
        # replicate the small field setup and build the pixel model instead.
        L.LightningModule.__init__(self)
        self.cfg = config
        self.vocab_size = vocab_size
        self.automatic_optimization = False
        self._train_generator = None
        self._ema = None
        self._loss_running: Dict[str, list] = {"loss": [], "l2": [], "ce": []}

        model_fn = ELFPixel_models[config.model]
        self.model = model_fn(
            patch_h=config.patch_h, patch_w=config.patch_w, char_w=config.char_w,
            in_channels=1, vocab_size=vocab_size, max_length=config.max_length,
            attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
            bottleneck_dim=config.bottleneck_dim,
            decode_bottleneck=config.decode_bottleneck,
            num_time_tokens=config.num_time_tokens,
            num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
            num_model_mode_tokens=config.num_model_mode_tokens,
            self_cond_input=(config.self_cond_prob > 0),
            use_flash=config.use_flash,
        )

    # --- decoder-branch lambda: per char-cell, expand to patch-pixel layout ---
    def _lambda_to_patch(self, lam_cells: torch.Tensor) -> torch.Tensor:
        """(B, max_chars) -> (B, N, P): each char cell's lambda fills its
        char_w-wide column block over the full patch height."""
        m = self.model
        B = lam_cells.shape[0]
        lam = lam_cells.reshape(B, m.max_length, m.chars_per_token)   # (B, N, cpt)
        lam = lam.repeat_interleave(m.char_w, dim=2)                  # (B, N, patch_w)
        lam = lam.unsqueeze(2).expand(B, m.max_length, m.patch_h, m.patch_w)
        return lam.reshape(B, m.max_length, m.patch_pixels)           # (B, N, P)

    def training_step(self, batch, batch_idx):
        cfg = self.cfg
        gen = self._train_generator
        device = batch["pixel_values"].device

        is_opt_step = (batch_idx + 1) % cfg.grad_accum_steps == 0
        lr = self._lr_at_step(self._my_opt_step)
        opts = self.optimizers()
        if not isinstance(opts, (list, tuple)):   # single optimizer (adamw) -> wrap
            opts = [opts]
        for opt in opts:
            for g in opt.param_groups:
                g["lr"] = lr

        # x0: glyph strip (B,1,H,W) in [0,1] -> patches (B,N,P) in [-1,1]
        img = batch["pixel_values"].to(device) * 2.0 - 1.0
        x0 = patchify(img, cfg.patch_h, cfg.patch_w)
        char_indices = batch["char_indices"].to(device)
        B = x0.shape[0]
        V = self.vocab_size

        decoder_step_active = bool(
            (torch.rand((), generator=gen, device=device) < cfg.decoder_prob).item())

        if decoder_step_active:
            # --- OCR decoder branch (cross-entropy) ---
            lam_cells = torch.sigmoid(
                torch.randn(B, self.model.max_chars, generator=gen, device=device)
                * cfg.decoder_p_std + cfg.decoder_p_mean)
            lam = self._lambda_to_patch(lam_cells).to(x0.dtype)
            noise = (torch.randn(x0.shape, generator=gen, device=device, dtype=x0.dtype)
                     * cfg.decoder_noise_scale)
            decoder_z = lam * x0 + (1.0 - lam) * noise
            t1 = torch.ones(B, device=device)
            _, logits = self.model(decoder_z, t1, decoder_step_active=True)
            ce_loss = F.cross_entropy(logits.reshape(-1, V).float(),
                                      char_indices.reshape(-1), ignore_index=IGNORE_INDEX)
            loss = ce_loss
            l2_loss = torch.zeros((), device=device)
        else:
            # --- denoiser branch (velocity MSE) ---
            t = sample_timesteps(gen, B, device=device,
                                 P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std,
                                 time_schedule=cfg.time_schedule)
            noise = torch.randn(x0.shape, generator=gen, device=device, dtype=x0.dtype)
            z = add_noise(x0, noise, t, cfg)
            v_target = (x0 - z) / torch.clamp(1.0 - t.view(-1, 1, 1), min=cfg.t_eps)
            net_out = self.model(z, t, decoder_step_active=False)
            v_pred, _ = net_out_to_v_x(net_out, z, t, cfg.t_eps)
            l2_loss = ((v_pred - v_target) ** 2).mean()
            loss = l2_loss
            ce_loss = torch.zeros((), device=device)

        self.manual_backward(loss / cfg.grad_accum_steps)
        self._loss_running["loss"].append(loss.detach())
        self._loss_running["l2"].append(l2_loss.detach())
        self._loss_running["ce"].append(ce_loss.detach())

        if is_opt_step:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            for opt in opts:
                opt.step()
                opt.zero_grad(set_to_none=True)
            self._ema.update(self.model)
            self._my_opt_step += 1
            if self._my_opt_step % cfg.log_freq == 0:
                self._log_running(lr)

        return loss.detach()


# --------------------------------------------------------------------------- #
# DataModule
# --------------------------------------------------------------------------- #
class PixelGlyphDataModule(L.LightningDataModule):
    def __init__(self, config, vocab: GlyphAtlasVocab):
        super().__init__()
        self.cfg = config
        self.vocab = vocab
        self.max_chars = config.max_length * (config.patch_w // config.char_w)
        self._train_dataset = None

    def setup(self, stage=None):
        if self._train_dataset is None:
            cfg = self.cfg
            source = load_text_dataset(cfg.dataset, split=cfg.lm1b_split,
                                       cache_dir=cfg.lm1b_cache_dir)
            chars_per_window = cfg.img_width // cfg.char_w
            self._train_dataset = ContinuousDocumentStreamDataset(
                source, chars_per_window=chars_per_window,
                limit_documents=cfg.limit_documents, drop_last=True, wrap=False)

    def train_dataloader(self):
        cfg = self.cfg
        collate = make_glyph_collate_with_labels(self.vocab, self.max_chars)
        return torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=cfg.global_batch_size // self.trainer.world_size,
            shuffle=True, collate_fn=collate, num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
            drop_last=True)
