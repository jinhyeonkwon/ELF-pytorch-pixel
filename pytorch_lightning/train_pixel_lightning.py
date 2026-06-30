#!/usr/bin/env python
"""Lightning entry point for the ELF pixel-vocab variant.

The flow runs over glyph-strip patches rendered from LM1B text via a fixed glyph
atlas (see `pixel/README.md`). Reuses the ELF trainer / EMA / optimizer / sampler;
only the model I/O, data, and decode are pixel-specific.

Usage (4-GPU DDP):
  cd pytorch_lightning/
  torchrun --nproc_per_node=4 --master_port=29503 train_pixel_lightning.py \
      --config configs/training_configs/train_lm1b_pixel_ELF-B.yml
"""

import argparse
import datetime
import logging
import math
import os
import re
import sys

import torch
import yaml

# Share DataLoader-worker tensors via the file-system strategy instead of the
# default file-descriptor one. On shared nodes the fd strategy can hit
# "could not unlink the shared memory file ... No such file or directory" when an
# external /dev/shm cleaner (or systemd RemoveIPC) removes the unlinked-fd shm
# file out from under a worker; that kills the worker, hangs the rank, and trips
# the NCCL collective-timeout on the other ranks. file_system avoids that race.
torch.multiprocessing.set_sharing_strategy("file_system")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from configs.config import SamplingConfig, apply_config_overrides, load_config_from_yaml
from pixel.eval import PixelGenEvalCallback
from pixel.glyph_vocab import GlyphAtlasVocab
from pixel.lit_module import ELFPixelLitModule, PixelGlyphDataModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--config_override", action="append", default=[])
    return p.parse_args()


def _resolve_precision(p: str) -> str:
    return {"fp32": "32", "32": "32", "bf16": "bf16-mixed", "bf16-mixed": "bf16-mixed",
            "fp16": "16-mixed", "16-mixed": "16-mixed"}.get(p, "32")


def main():
    args = parse_args()
    logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)], level=logging.INFO, force=True)

    cfg = load_config_from_yaml(args.config)
    if args.config_override:
        cfg = apply_config_overrides(cfg, args.config_override)

    # Sanity: geometry must be consistent so patchify/decode align.
    assert cfg.patch_h == cfg.img_height, "patch_h must equal img_height (single patch row)"
    assert cfg.img_width % cfg.patch_w == 0, "img_width must be a multiple of patch_w"
    assert cfg.patch_w % cfg.char_w == 0, "patch_w must be a multiple of char_w"
    expected_tokens = cfg.img_width // cfg.patch_w
    if cfg.max_length != expected_tokens:
        cfg.max_length = expected_tokens   # #patch tokens is derived from geometry

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    L.seed_everything(cfg.seed, workers=True)

    # Glyph vocab (the "frozen encoder"): atlas + LUT + meta. Resolve relative to repo root.
    vocab_dir = cfg.vocab_dir if os.path.isabs(cfg.vocab_dir) else os.path.join(REPO_ROOT, cfg.vocab_dir)
    vocab = GlyphAtlasVocab(vocab_dir, load_atlas=True)
    if vocab.char_h != cfg.img_height or vocab.char_w != cfg.char_w:
        raise ValueError(f"atlas cell ({vocab.char_h}x{vocab.char_w}) != "
                         f"img_height({cfg.img_height}) x char_w({cfg.char_w})")

    model = ELFPixelLitModule(cfg, vocab_size=vocab.vocab_size)
    datamodule = PixelGlyphDataModule(cfg, vocab=vocab)

    logger = False
    if cfg.use_wandb:
        # Stable run id (sanitized from the run name) + resume="allow" so resuming
        # training continues the SAME wandb run instead of starting a new one.
        run_id = (re.sub(r"[^A-Za-z0-9_.-]", "-", cfg.wandb_run_name)
                  if cfg.wandb_run_name else None)
        logger = WandbLogger(project=cfg.wandb_project, entity=cfg.wandb_entity,
                             name=cfg.wandb_run_name, id=run_id, resume="allow",
                             save_dir="/tmp",
                             tags=cfg.wandb_tag.split(",") if cfg.wandb_tag else None)

    # eval-time distilled-guidance scale (only effective if trained with cfg tokens)
    eval_sc_cfg_scale = (cfg.epoch_eval_self_cond_cfg_scale
                         if cfg.num_self_cond_cfg_tokens > 0 else 1.0)

    callbacks = [ModelCheckpoint(
        dirpath=cfg.output_dir, filename="checkpoint_epoch{epoch:02d}_step{step:08d}",
        every_n_epochs=int(cfg.save_freq) if cfg.save_freq >= 1 else 1,
        save_top_k=-1, save_last=True, auto_insert_metric_name=False)]
    if cfg.online_eval and cfg.eval_freq >= 1:
        callbacks.append(PixelGenEvalCallback(
            vocab=vocab, output_dir=cfg.output_dir,
            num_samples=cfg.epoch_eval_num_samples,
            num_sampling_steps=cfg.epoch_eval_num_sampling_steps,
            sample_method=cfg.pixel_sample_method, sde_gamma=cfg.epoch_eval_sde_gamma,
            num_images=cfg.pixel_eval_num_images, eval_freq=int(cfg.eval_freq),
            self_cond_cfg_scale=eval_sc_cfg_scale,
            eval_ppl_model=cfg.eval_ppl_model, eval_ppl_batch_size=cfg.eval_ppl_batch_size,
            eval_ppl_max_length=cfg.eval_ppl_max_length))

    # Raise the NCCL collective timeout well above the 30-min default: the
    # per-epoch eval (long ODE/SDE generation + gpt2-large gen-PPL on rank 0)
    # can leave other ranks waiting at the next collective; 30 min could abort a
    # multi-hour run. 2h is comfortable headroom.
    strategy = DDPStrategy(find_unused_parameters=True, broadcast_buffers=False,
                           timeout=datetime.timedelta(hours=2))
    trainer = L.Trainer(
        max_epochs=cfg.epochs, accelerator="gpu", devices=-1, strategy=strategy,
        precision=_resolve_precision(cfg.precision),
        accumulate_grad_batches=cfg.grad_accum_steps, logger=logger, callbacks=callbacks,
        log_every_n_steps=cfg.log_freq, default_root_dir=cfg.output_dir,
        use_distributed_sampler=True)

    ckpt_path = cfg.resume
    if ckpt_path is None:
        last = os.path.join(cfg.output_dir, "last.ckpt")
        if os.path.exists(last):
            ckpt_path = last

    snap = {k: ([vars(sc) for sc in v]
                if isinstance(v, list) and v and isinstance(v[0], SamplingConfig) else v)
            for k, v in vars(cfg).items()}
    if trainer.is_global_zero:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "config.yml"), "w") as f:
            yaml.dump(snap, f, default_flow_style=False, sort_keys=False)
    # Log the full resolved config to the wandb run (Overview/Config tab).
    if logger:
        logger.log_hyperparams(snap)

    # Build the windowing index once so the LR schedule knows steps/epoch.
    datamodule.setup()
    steps_per_epoch = max(1, math.ceil(len(datamodule._train_dataset) / cfg.global_batch_size))
    model._steps_per_epoch = steps_per_epoch
    model._num_optimizer_steps = steps_per_epoch * cfg.epochs // max(1, cfg.grad_accum_steps)

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
