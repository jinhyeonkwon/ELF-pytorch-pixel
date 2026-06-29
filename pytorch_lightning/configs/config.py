"""Config dataclass — same fields as `pytorch/configs/config.py` plus the
optimization knobs added in this update."""

import os
from typing import List, Optional

import yaml


class SamplingConfig:
    sampling_method: str = "ode"
    num_sampling_steps: list = [50]
    cfgs: list = [1]
    self_cond_cfg_scales: list = [1.0]
    time_schedule: str = "logit_normal"
    sde_gamma: float = 0.0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        fields = {k: getattr(self, k, None) for k in self.__class__.__annotations__}
        fields.update({k: v for k, v in vars(self).items() if not k.startswith("_")})
        return f"SamplingConfig({', '.join(f'{k}={v!r}' for k,v in fields.items())})"


class Config:
    # Dataset
    data_path: str = None
    eval_data_path: str = None
    max_length: int = 128
    max_input_length: int = None
    pad_token: str = "pad"
    tokenizer_name: str = None

    # Encoder
    encoder_model_name: str = "t5-small"
    encoder_checkpoint: str = None
    latent_mean: float = 0.0
    latent_std: float = 1.0

    # Model architecture
    model: str = "ELF-B"
    bottleneck_dim: int = 128
    num_time_tokens: int = 4
    num_self_cond_cfg_tokens: int = 4
    num_model_mode_tokens: int = 4
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0

    # Denoiser objective
    denoiser_p_mean: float = 0.8
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 1.0
    t_eps: float = 5e-2
    time_schedule: str = "logit_normal"

    # Decoder objective
    decoder_prob: float = 0.5
    decoder_noise_scale: float = 1.0
    decoder_p_mean: float = 0.8
    decoder_p_std: float = 0.8

    # Conditioning / CFG
    label_drop_prob: float = 0.0
    self_cond_prob: float = 0.5
    self_cond_cfg_min: float = 0.5
    self_cond_cfg_max: float = 5.0

    # Training
    epochs: int = 200
    warmup_epochs: float = None
    warmup_steps: int = 5000
    batch_size: int = None
    global_batch_size: int = 512
    lr: float = None
    blr: float = 5e-5
    min_lr: float = 0.0
    lr_schedule: str = "constant"
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    grad_accum_steps: int = 1

    # EMA
    ema_decay1: float = 0.9999

    # Sampling
    sampling_configs_path: str = None
    sampling_configs: list = [SamplingConfig()]
    num_samples: int = 100

    # PPL eval
    online_eval: bool = True
    eval_ppl_model: str = "gpt2-large"
    eval_ppl_batch_size: int = 64
    eval_ppl_max_length: int = 1024

    # Logging / checkpointing
    log_freq: int = 100
    eval_freq: int = 10
    save_freq: float = 100

    # Output
    output_dir: str = "./output_dir"
    resume: str = None

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "ELF"
    wandb_entity: str = None
    wandb_run_name: str = None
    wandb_tag: str = None

    # Misc
    seed: int = 0
    num_workers: int = 8           # Task 5: 8 per rank by default.
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # Optimization knobs (this update)
    use_flash: bool = True         # Task 2: FlashAttention path in ELF.
    precision: str = "fp32"        # Lightning precision: "fp32" | "bf16-mixed" | "16-mixed"
    compile: bool = False

    # Per-epoch eval callback (Task 4)
    epoch_eval_num_samples: int = 256
    epoch_eval_num_sampling_steps: int = 32
    epoch_eval_sde_gamma: float = 1.5
    epoch_eval_self_cond_cfg_scale: float = 3.0

    # --- Pixel-vocab variant (glyph-strip flow; see pixel/README.md) ---------
    # Only used by train_pixel_lightning.py / eval_pixel_lightning.py. The flow
    # runs over a sequence of glyph-strip patches instead of T5 embeddings, so
    # the encoder fields above are ignored in pixel mode.
    img_height: int = 16            # glyph strip height (== atlas char_h)
    img_width: int = 1280           # glyph strip width (chars_per_window * char_w)
    patch_h: int = 16               # patch height (== img_height; single patch row)
    patch_w: int = 64               # patch width -> chars_per_token = patch_w // char_w
    char_w: int = 8                 # glyph cell width (== atlas char_w)
    decode_bottleneck: int = 256    # OCR decode-head bottleneck
    vocab_dir: str = "pixel/assets/bert-base-uncased"   # glyph atlas + LUT + meta
    lm1b_cache_dir: str = None      # where LM1B is downloaded/extracted (or $DATA_DIR/lm1b)
    lm1b_split: str = "train"
    limit_documents: int = None     # cap #documents (None == full corpus)
    pixel_sample_method: str = "ode"   # "ode" | "sde" for generation/eval
    pixel_eval_num_images: int = 4  # glyph-strip PNGs dumped per epoch eval


def load_config_from_yaml(path: Optional[str]) -> Config:
    cfg = Config()
    if not path or not os.path.isfile(path):
        return cfg
    with open(path, "r") as f:
        d = yaml.safe_load(f) or {}
    for k, v in d.items():
        if k == "sampling_configs":
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    if cfg.sampling_configs_path:
        cfg.sampling_configs = load_sampling_configs(cfg.sampling_configs_path)
    return cfg


def _coerce(value: str, target_type) -> object:
    if target_type is bool:
        return value.lower() in ("true", "1", "yes")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def apply_config_overrides(config: Config, overrides: List[str]) -> Config:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override: '{override}'")
        name, val = override.split("=", 1)
        name, val = name.strip(), val.strip()
        if not hasattr(config, name):
            raise ValueError(f"No config field '{name}'")
        if val.lower() == "none":
            setattr(config, name, None)
            continue
        cur = getattr(config, name)
        target_type = type(cur) if cur is not None else config.__annotations__.get(name, str)
        setattr(config, name, _coerce(val, target_type))
    return config


def load_sampling_configs(path: str) -> List[SamplingConfig]:
    with open(path, "r") as f:
        entries = yaml.safe_load(f)
    return [SamplingConfig(**e) for e in entries]
