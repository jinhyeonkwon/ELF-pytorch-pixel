# ELF pixel-vocab variant — flow-matching over glyph-strip pixels

A self-contained re-implementation, inside this ELF-pytorch repo and in its
PyTorch-Lightning idiom, of the **pixel-defined-vocabulary** setting: instead of
running ELF's flow over T5 token embeddings, we run it over the **raw pixels of a
rendered glyph strip**. Text is rendered to a fixed monospaced image via a glyph
*atlas* (one Unicode char → one fixed 16×8 glyph), the flow generates such an
image, and an OCR decode head reads it back to characters.

> **Status:** pixel velocity-MSE **denoiser** + OCR cross-entropy **decoder**,
> with **self-conditioning** and **CFG guidance-distillation** implemented and
> toggled by config (`self_cond_prob`, `num_self_cond_cfg_tokens`; 0 = off).
> In-context conditioning (time/mode/cfg prefix tokens) is the design, as in ELF.
> The code reuses ELF's flow math, sampler, EMA, optimizer, and trainer verbatim.
> See [§7](#7-feature-toggles--whats-still-open) for the toggles.

---

## 0. The one-paragraph idea

ELF is flow-matching over a sequence of continuous vectors `x0 ∈ (B, L, D)` with a
DiT-style transformer that (a) denoises in that space and (b) has a decode head
that reads a (near-clean) latent back to token ids. **Everything in ELF that is
not the encoder, the data, or the in/out projections is agnostic to what `D`
*means*.** The pixel variant sets `x0` = a glyph strip cut into `L = N` patches of
`D = P` pixels each (`P = patch_h·patch_w`). There is no learned encoder — the
glyph atlas is a frozen, deterministic, *lossless* "encoder" (a discriminator
stamp at build time makes every glyph bit-distinct, so a clean image decodes back
to the exact characters). The flow runs over `(B, N, P)`; the denoiser head
reconstructs pixels, the decode head reads glyph indices.

```
 text ──render(atlas)──► glyph strip (B,1,16,1280) ──patchify──► x0 (B, 20, 1024)   ∈ [-1,1]
                                                                      │
                                                         ┌── flow-matching (ELF, unchanged) ──┐
                                                         │   z=t·x0+(1-t)·ε ; v=(x0-z)/(1-t)   │
                                                         ▼                                     │
                                              ELFPixel transformer  ──► denoiser head ► x̂ (B,20,1024)
                                                         │             └► OCR head ► (B,160,V) logits
 generate ◄── ODE/SDE sampler (ELF, unchanged) ◄─────────┘                     │ argmax
                                                                               ▼
                            glyph indices (B,160) ──atlas⁻¹──► text ──► gen-PPL (gpt2-large)
```

---

## 1. Geometry — fully parametric

Nothing is hardcoded to "20 patches". Everything is **derived from config** and the
code is verified at N = 20 / 128 / 1024 and at `chars_per_token` = 8 / 4 / 1:

```
N (patch tokens, = transformer seq len)  = img_width // patch_w
P (per-patch flow dim, "text_encoder_dim") = in_channels · patch_h · patch_w
chars_per_token (cpt)                     = patch_w // char_w
max_chars (OCR target length)             = N · cpt   (= img_width // char_w)
```

Invariants (asserted in `train_pixel_lightning.py`): `patch_h == img_height` (the
glyph strip is a **single patch row**, always 16 px tall = atlas `char_h`);
`patch_w` divides `img_width`; `char_w` divides `patch_w` (so `cpt ≥ 1`). The
transformer's RoPE length tracks `max_length = N` automatically — large N just
means a longer sequence, no code change.

| config | `img_width` | `patch_w` | → N | cpt | chars | use |
|---|---|---|---|---|---|---|
| **demo (this YAML)** | 1280 | 64 | **20** | 8 | 160 | quick LM1B runs |
| **LM1B big** | 8192 | 64 | **128** | 8 | 1024 | scaled LM1B |
| **OWT** | 65536 | 64 | **1024** | 8 | 8192 | OpenWebText |
| ELF-isomorphic | 1280 | 8 | 160 | 1 | 160 | 1 char/token (like ELF token unembed) |

To get more patches you either widen the image (more chars) **or** shrink `patch_w`
(finer patches, fewer chars/token, smaller OCR head). Minimum `patch_w = char_w = 8`
(cpt = 1). Just set `img_width` / `patch_w` in the YAML — `max_length` is
recomputed from `img_width // patch_w` (the entry overrides any stale YAML value).

Fixed across all configs: strip height 16, glyph cell `char_h=16` / `char_w=8`,
vocab `V = 1000` (bert-base-uncased atlas: 998 chars + UNK + EOS), backbone
ELFPixel-B (depth 12, hidden 768, 12 heads; ~89M params at the demo geometry — the
denoiser/OCR head sizes scale with `P` and `cpt`).

---

## 2. What is reused vs. new

The clean seam is: **flow math + sampling + EMA + optimizer + trainer are
modality-agnostic; encoder, data, in/out projections, and decode are pixel-specific.**

**Reused from the repo, untouched:**
- `utils/sampling_utils.py` — `add_noise`, `sample_timesteps`, `net_out_to_v_x`,
  `ode_step`/`sde_step` (pure `(B,L,C)` tensor math).
- `utils/generation_utils.py` — `generate_samples`, `dlm_decode_batch`.
- `utils/metrics_utils.py::Metrics` — gen-PPL + unigram entropy (gpt2-large).
- `lightning_module.py::{EMA, ELFLitModule.configure_optimizers, _lr_at_step,
  _log_running, on_save/load_checkpoint}` — inherited by subclassing.
- `modules/layers.py` — `ELFBlock`'s `Attention` (RoPE, qk-norm, flash), `RMSNorm`,
  `TextRotaryEmbeddingFast`, `TimestepEmbedder`, `BottleneckTextProj`, `FinalLayer`.
- `L.Trainer` + `DDPStrategy(find_unused_parameters=True, broadcast_buffers=False)`
  + `ModelCheckpoint` (set up identically in `train_pixel_lightning.py`).

**New, pixel-specific (this `pixel/` package):**

| file | role |
|---|---|
| `glyph_vocab.py` | `GlyphAtlasVocab` (atlas + codepoint→index LUT + meta), `build_index_to_char`, `indices_to_text`. The "frozen encoder". |
| `glyph_dataset.py` | `load_lm1b` (downloads/extracts LM1B), `ContinuousDocumentStreamDataset` (fixed-char windows over `doc+EOS+doc…`), `make_glyph_collate_with_labels` (render strip + per-cell OCR labels). |
| `model.py` | `ELFPixel` — ELF transformer with Linear patch-in (`P→hidden`), `FinalLayer` pixel head (`hidden→P`), and a multi-char OCR decode head (`hidden→cpt·V`). `patchify`/`unpatchify`. |
| `lit_module.py` | `ELFPixelLitModule(ELFLitModule)` (overrides `__init__` + `training_step`), `PixelGlyphDataModule`. |
| `eval.py` | `generate_pixel_samples`, `PixelGenEvalCallback` (per-epoch gen → glyph-decode → gen-PPL + PNG dump). |
| `assets/bert-base-uncased/` | the three vocab files (atlas `.npy`, LUT `.npy`, meta `.json`) — ~5 MB, committed so the variant is self-contained. |

Entry points: `train_pixel_lightning.py` and
`configs/training_configs/train_lm1b_pixel_ELF-B.yml`.

---

## 3. The model — `pixel/model.py::ELFPixel`

Same trunk as `modules/model.py::ELF` (prefix time/mode/cfg tokens with identity
RoPE, `ELFBlock` stack with qk-norm + SwiGLU + RoPE, FA4/SDPA), with three swaps:

- **patch-in**: `BottleneckTextProj(P → hidden)` consumes the flattened patch
  pixels directly (the conv patch-embed of the source FLM model is equivalent to a
  Linear over a non-overlapping patch). For self-cond the input is `(B,N,2P)` and a
  `self_cond_proj(2P→P)` collapses it (as ELF's does for `2D`).
- **denoiser head**: `FinalLayer(hidden, patch_size=1, out_channels=P)` →
  `(B,N,P)` **x-prediction** (zero-init, ELF's adaLN-zero-style output). `unpatchify`
  turns it back into a `(B,1,16,img_width)` image for visualization.
- **OCR decode head**: `RMSNorm → Linear(hidden→decode_bottleneck) → GELU →
  Linear(→ cpt·V)` reshaped to `(B, N·cpt, V) = (B, max_chars, V)`. Char alignment
  is exact because rendering is monospaced at `char_w` and `patch_w` is a multiple
  of it.

`forward(x, t, attention_mask, self_cond_cfg_scale, decoder_step_active)` keeps
ELF's exact contract `→ (x_pred, decoder_logits|None)`, so `net_out_to_v_x`,
`generate_samples`, and `dlm_decode_batch` work **unchanged**.

> **Init caveat.** The source FLM model collapsed to flat-gray output when its
> denoiser head was zero-initialized (because a later xavier sweep clobbered the
> rest, leaving the trunk gradient-starved). We **do not** inherit that pathology:
> ELFPixel uses ELF's own init scheme (xavier kernels / zero bias, zero-init
> `FinalLayer`, `N(0,0.02²)` prefix tokens), which trains fine on ELF. Do not add a
> blanket `_init_weights` xavier sweep on top.

---

## 4. The objective — `pixel/lit_module.py::ELFPixelLitModule.training_step`

Per optimizer step, a `Bernoulli(decoder_prob)` coin selects **one** branch
(ELF-faithful, not a weighted sum):

- **Denoiser** (prob `1-decoder_prob`): logit-normal `t`, interpolant
  `z = add_noise(x0, ε, t, cfg)` (= `t·x0 + (1-t)·ε·denoiser_noise_scale`),
  velocity target `v = (x0-z)/clamp(1-t, t_eps)`, **MSE**. (Reuses the repo's
  `add_noise` / `net_out_to_v_x`.)
- **Decoder/OCR** (prob `decoder_prob`): per-**char-cell** logit-normal `λ`
  expanded to the patch-pixel layout (`_lambda_to_patch`), low-noise latent
  `z = λ·x0 + (1-λ)·ε·decoder_noise_scale` read at `t=1`, **cross-entropy** over the
  glyph vocab with `ignore_index=-100` on empty (white) cells.

**Self-conditioning** (when `self_cond_prob > 0`): a stop-grad forward on
`[z, 0]` produces `x̂_init`; it is fed back as the 2nd input channel
(`[z, x̂_init·mask]`, zeros for non-selected samples) on the gradient forward —
same as ELF, on `(B,N,P)` instead of `(B,L,D)`.

**CFG guidance distillation** (when `num_self_cond_cfg_tokens > 0`): two extra
stop-grad forwards (uncond hint `[z,0]`, then cond hint `[z, x̂_uncond]`, both
carrying a per-sample log-uniform scale `w` in the cfg prefix tokens) build a
guided target `v_target + (1-1/w)·(v_cond-v_uncond)` the gradient forward distills
into. At inference, a single `self_cond_cfg_scale` (default `epoch_eval_self_cond_cfg_scale=3.0`)
selects the baked-in guidance level.

The conditional-prefix masking ELF uses (`cond_seq_mask`/`restore_cond`) is the
identity here (unconditional generation), so it is dropped. EMA, grad-clip(1.0),
manual optimizer step, the LR schedule, and checkpoint plumbing are inherited from
`ELFLitModule` unchanged. `x0` is `patchify(pixel_values·2-1)` instead of
`encode_text(...)` — that one line is the whole "no T5 encoder" difference.

---

## 5. The data — `pixel/glyph_dataset.py`

1. `load_text_dataset(cfg.dataset, …)` returns an HF `Dataset` with a `text`
   column. `dataset: lm1b` (validated) downloads + extracts the One Billion Word
   Benchmark once (statmt.org tarball); `dataset: openwebtext` (forward-compat,
   untested here — large download) loads `Skylion007/openwebtext`. Adding another
   corpus = one loader returning a `text` column; everything below is
   dataset-agnostic.
2. `ContinuousDocumentStreamDataset` builds a char-prefix index over
   `doc + EOS + next_doc + EOS …` and yields fixed-length windows of
   `img_width // char_w` characters — `{"text": <window>}`. The window length
   scales with the geometry (160 / 1024 / 8192 chars for the N = 20 / 128 / 1024
   configs).
3. `make_glyph_collate_with_labels` BERT-normalizes each window, maps codepoints
   through the LUT, gathers glyph cells from the in-memory atlas (no rendered image
   is ever cached), and emits `pixel_values (B,1,16,img_width)` +
   `char_indices (B, max_chars)`.

> Building the window index scans `limit_documents` documents up front (set
> `limit_documents` for a fast smoke run; `null` = the full corpus, which is slow
> to index once, exactly as in the source experiment).

---

## 6. Running it

Easiest — the launcher `pixel/train_pixel.sh` (sets GPUs, LM1B/HF caches, auto-resume):

```bash
cd pytorch_lightning/
bash pixel/train_pixel.sh                       # all visible GPUs, full config
GPUS=0,1,2,3 bash pixel/train_pixel.sh          # pick GPUs
SMOKE=1 bash pixel/train_pixel.sh               # tiny fast sanity run (no eval/wandb)
bash pixel/train_pixel.sh --config_override epochs=30 --config_override lr=1e-4
```

Or call the entry directly:

```bash
torchrun --standalone --nproc_per_node=4 train_pixel_lightning.py \
    --config configs/training_configs/train_lm1b_pixel_ELF-B.yml
```

> **`use_flash` portability:** the FA4 (CuTeDSL) kernel only exists on Hopper/
> Blackwell. On Ampere (A100) the model auto-falls back to PyTorch SDPA in bf16,
> so `use_flash: true` runs everywhere; set `use_flash: false` to force fp32 SDPA.

Outputs land in `output_dir`: `last.ckpt` + per-epoch checkpoints, `config.yml`
snapshot, and (when `online_eval`) `pixel_epoch_NNN/{decoded.txt, *.png,
metrics.jsonl}`.

**W&B logging** (when `use_wandb: true`; entity/project/run from `wandb_entity`/
`wandb_project`/`wandb_run_name`):
- **scalars** every `log_freq` opt steps: `train/loss`, `train/l2_loss`,
  `train/ce_loss` (each un-biased by its branch probability), `train/lr`; and per
  eval `eval/gen_ppl`, `eval/sample_entropy`.
- **config**: the full resolved `Config` is logged to the run (Overview/Config tab)
  via `logger.log_hyperparams(...)` in the entry.
- **samples**: a `wandb.Table` (`samples`) with the generated glyph-strip image +
  its decoded text, logged each eval by `PixelGenEvalCallback`.
- **resume**: the run `id` is derived from `wandb_run_name` with `resume="allow"`,
  so resuming training (auto-detected `last.ckpt`, or `--config_override resume=…`)
  continues the **same** W&B run instead of starting a new one.

**Key hyperparameters** (`train_lm1b_pixel_ELF-B.yml`): AdamW lr 2e-4 constant,
β(0.9,0.95), wd 0, batch 512, 60 epochs, bf16-mixed, EMA 0.9999; denoiser
`p_mean −0.8 / p_std 0.8 / noise_scale 1.0 / t_eps 0.05`; decoder `prob 0.2 /
p_mean 0.8 / p_std 0.8 / noise_scale 2.0 / bottleneck 256`; eval = 256-step ODE,
gen-PPL via gpt2-large.

> **Deps beyond the base repo:** `transformers` (atlas normalizer + gpt2-large —
> already required), `datasets` (LM1B — already required), and `Pillow` for PNG
> dumps (optional; the callback skips image writing if it's missing).

---

## 7. Feature toggles & what's still open

Self-conditioning and CFG distillation are **implemented** (§4) and switched by
config — no code change needed:

- **Self-conditioning**: `self_cond_prob: 0.5` (0 = off). Builds the 2-channel
  patch input + `self_cond_proj`; training does the stop-grad hint forward;
  sampling threads `x̂_prev` across steps.
- **CFG / guidance distillation**: `num_self_cond_cfg_tokens: 4` (0 = off) with
  `self_cond_cfg_min/max` (train scale range) and `epoch_eval_self_cond_cfg_scale`
  (inference scale, default 3.0). Builds the cfg prefix tokens + embedder.
- **In-context conditioning** is the design (time/mode/cfg are prefix tokens, not
  adaLN), matching ELF; `num_model_mode_tokens` signals decode mode.

> **Config constraint:** CFG tokens require the 2-channel input, so the model is
> built with `self_cond_input = (self_cond_prob>0 or num_self_cond_cfg_tokens>0)`.
> In practice use CFG together with self-cond (both > 0), as the shipped config does.

Still open (not yet ported from the source experiment): the **staged warmup**
schedule (introduce self-cond at epoch N, CFG at epoch M — `flm_ours` did this via
epoch-gated switches), the E4 **hint clamp / guidance t-gate** stabilizers, and the
training-free **GlyphCNN/nearest-glyph** decode path (here only the model's own OCR
head decodes). The architecture supports all of these as `training_step` / eval
edits, not structural changes.

---

## 8. Provenance

Ported from the source experiment in the sibling `flm_ours` repo
(`pixel_predict_mse/flm_based/elf_like/`, branch `jinhyeonkwon/elf-like`): the
glyph atlas + renderer (`glyph_render.py`), the dual-head pixel flow model
(`model.py`/`denoiser.py`), and the LM1B windowing dataset. Re-expressed here in
ELF-pytorch's Lightning architecture so the flow math, sampler, EMA, optimizer,
and trainer are shared with the T5-embedding ELF rather than reimplemented.
