"""Generation + per-epoch eval for the ELF pixel variant.

Generation reuses the modality-agnostic ELF sampler (`generate_samples`) and
decode head (`dlm_decode_batch`) verbatim — the only pixel-specific bits are the
initial-noise shape (B, N, P) and turning the decoded glyph indices into text via
the atlas inverse map (instead of `tokenizer.decode`).

`PixelGenEvalCallback` mirrors `callbacks.PerEpochGenEvalCallback`: swap EMA in →
generate → decode to text → gen-PPL (gpt2-large) + unigram entropy → dump a few
glyph-strip PNGs + decoded.txt → restore weights.
"""

import json
import os

import lightning as L
import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback
from tqdm import tqdm

from configs.config import SamplingConfig
from utils.generation_utils import dlm_decode_batch, generate_samples
from utils.metrics_utils import Metrics as PPLMetrics
from utils.sampling_utils import get_sampling_steps

from pixel.glyph_vocab import build_index_to_char, indices_to_text
from pixel.model import unpatchify

_STRIP_W = 2048
_GAP = 4


@torch.no_grad()
def generate_pixel_samples(model, cfg, *, num: int, steps: int, method: str,
                           sde_gamma: float, device, generator,
                           self_cond_cfg_scale: float = 1.0):
    """Return (images (num,1,H,W) in [-1,1], glyph_idx (num, max_chars)).

    `self_cond_cfg_scale` is the distilled-guidance scale used at inference; only
    has effect when the model was trained with `num_self_cond_cfg_tokens > 0`.
    """
    P = model.text_encoder_dim                       # per-patch pixel dim
    N = model.max_length
    sc = SamplingConfig(sampling_method=method, num_sampling_steps=[steps],
                        cfgs=[1], self_cond_cfg_scales=[self_cond_cfg_scale], sde_gamma=sde_gamma,
                        time_schedule="logit_normal")
    t_steps = get_sampling_steps(generator, n_steps=steps, device=device,
                                 time_schedule="logit_normal",
                                 P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std)
    z = torch.randn((num, N, P), generator=generator, device=device) * cfg.denoiser_noise_scale
    latent = generate_samples(model, z, t_steps, cond_seq=None, cond_seq_mask=None,
                              config=cfg, sampling_config=sc, cfg_scale=1.0,
                              self_cond_cfg_scale=self_cond_cfg_scale, generator=generator)
    glyph_idx = dlm_decode_batch(model, latent, config=cfg,
                                 self_cond_cfg_scale=self_cond_cfg_scale,
                                 t_final_val=float(t_steps[-1].item()))
    images = unpatchify(latent, cfg.patch_h, cfg.patch_w, channels=1)   # (num,1,H,W)
    return images, glyph_idx


def _to_strip_png(img_u8: np.ndarray) -> np.ndarray:
    """Wrap a tall-but-wide (H, W) uint8 glyph strip into stacked rows for viewing."""
    H, W = img_u8.shape
    if W <= _STRIP_W:
        return img_u8
    n = (W + _STRIP_W - 1) // _STRIP_W
    canvas = np.zeros((n * (H + _GAP) - _GAP, _STRIP_W), dtype=np.uint8)
    for i in range(n):
        x0, x1 = i * _STRIP_W, min((i + 1) * _STRIP_W, W)
        y0 = i * (H + _GAP)
        canvas[y0:y0 + H, :x1 - x0] = img_u8[:, x0:x1]
    return canvas


def _save_png(path: str, img_u8: np.ndarray) -> None:
    """Save an already-wrapped (H, W) uint8 strip; no-op if PIL is missing."""
    try:
        from PIL import Image
        Image.fromarray(img_u8).save(path)
    except Exception as e:                            # PIL missing / write error -> skip quietly
        print(f"[pixel-eval] PNG dump skipped ({e})")


class PixelGenEvalCallback(Callback):
    """Per-(eval_freq)-epoch generation + glyph decode + gen-PPL for pixel ELF."""

    def __init__(self, *, vocab, output_dir: str, num_samples: int,
                 num_sampling_steps: int, sample_method: str, sde_gamma: float,
                 num_images: int, eval_freq: int, self_cond_cfg_scale: float,
                 eval_ppl_model: str, eval_ppl_batch_size: int, eval_ppl_max_length: int):
        super().__init__()
        self.index_to_char = build_index_to_char(vocab)
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.num_sampling_steps = num_sampling_steps
        self.sample_method = sample_method
        self.sde_gamma = sde_gamma
        self.num_images = num_images
        self.eval_freq = max(1, eval_freq)
        self.self_cond_cfg_scale = self_cond_cfg_scale
        self.eval_ppl_model = eval_ppl_model
        self.eval_ppl_batch_size = eval_ppl_batch_size
        self.eval_ppl_max_length = eval_ppl_max_length
        self._ppl_metrics = None

    @torch.no_grad()
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.eval_freq != 0:
            return
        cfg = pl_module.cfg
        device = pl_module.device
        rank = trainer.global_rank
        world_size = trainer.world_size

        backup = pl_module._ema.swap_in(pl_module.model)
        try:
            pl_module.model.eval()
            per_rank = (self.num_samples + world_size - 1) // world_size
            batch = cfg.global_batch_size // max(1, world_size)
            num_batches = (per_rank + batch - 1) // batch

            texts, first_images = [], []
            for bi in tqdm(range(num_batches), desc=f"[ep{epoch+1}] pixel-gen",
                           disable=(rank != 0)):
                cur = min(batch, per_rank - bi * batch)
                if cur <= 0:
                    break
                seed = cfg.seed * 1000003 + epoch * 991 + bi * 97 + rank
                gen = torch.Generator(device=device).manual_seed(seed)
                images, glyph_idx = generate_pixel_samples(
                    pl_module.model, cfg, num=cur, steps=self.num_sampling_steps,
                    method=self.sample_method, sde_gamma=self.sde_gamma,
                    device=device, generator=gen,
                    self_cond_cfg_scale=self.self_cond_cfg_scale)
                rows = [indices_to_text(glyph_idx[i].cpu(), self.index_to_char)
                        for i in range(cur)]
                gathered = self._all_gather_obj(rows, world_size)
                texts.extend(gathered)
                if bi == 0 and rank == 0:
                    first_images = images[:self.num_images].float().cpu()

            if rank == 0:
                out_dir = os.path.join(self.output_dir, f"pixel_epoch_{epoch+1:03d}")
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "decoded.txt"), "w", encoding="utf-8") as f:
                    for i, t in enumerate(texts):
                        f.write(f"[{i:05d}]\n{t}\n\n")
                strips = []                                   # (uint8 strip, decoded text)
                for i in range(first_images.shape[0]):
                    u8 = np.round(((first_images[i, 0] + 1) / 2).clamp(0, 1).numpy() * 255
                                  ).astype(np.uint8)
                    u8 = _to_strip_png(u8)
                    _save_png(os.path.join(out_dir, f"{i:05d}.png"), u8)
                    strips.append((u8, texts[i] if i < len(texts) else ""))
                self._log_samples_to_wandb(trainer, epoch + 1, strips)

                nonempty = [s for s in texts if isinstance(s, str) and s.strip()]
                if nonempty:
                    if self._ppl_metrics is None:
                        self._ppl_metrics = PPLMetrics(
                            gen_ppl_eval_model_name_or_path=self.eval_ppl_model,
                            eval_ppl_batch_size=self.eval_ppl_batch_size,
                            eval_context_size=self.eval_ppl_max_length, device=str(device))
                    res = self._ppl_metrics.record_generative_perplexity(
                        text_samples=nonempty, max_length=self.eval_ppl_max_length,
                        retokenize=True)
                    pl_module.log("eval/gen_ppl", float(res["ppl"]), rank_zero_only=True)
                    pl_module.log("eval/sample_entropy", float(res["mean_entropy"]),
                                  rank_zero_only=True)
                    with open(os.path.join(out_dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps({"epoch": epoch + 1, "step": trainer.global_step,
                                            "gen_ppl": float(res["ppl"]),
                                            "sample_entropy": float(res["mean_entropy"])}) + "\n")
                    print(f"[pixel-eval] ep{epoch+1}: gen_ppl={res['ppl']:.2f} "
                          f"entropy={res['mean_entropy']:.2f} | sample0: {nonempty[0][:120]!r}")

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        finally:
            pl_module._ema.restore(pl_module.model, backup)
            pl_module.model.train()

    @staticmethod
    def _all_gather_obj(local_list, world_size):
        if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
            return local_list
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_list)
        return [x for sub in gathered for x in sub]

    @staticmethod
    def _log_samples_to_wandb(trainer, epoch, strips):
        """Log generated glyph strips + decoded text as a wandb.Table (rank-0 only)."""
        logger = getattr(trainer, "logger", None)
        exp = getattr(logger, "experiment", None)
        if exp is None or not strips:
            return
        try:
            import wandb
            table = wandb.Table(columns=["epoch", "idx", "image", "decoded"])
            for i, (u8, text) in enumerate(strips):
                table.add_data(epoch, i, wandb.Image(u8), text)
            exp.log({"samples": table, "epoch": epoch})
        except Exception as e:                    # wandb missing / logger not wandb -> skip
            print(f"[pixel-eval] wandb sample log skipped ({e})")
