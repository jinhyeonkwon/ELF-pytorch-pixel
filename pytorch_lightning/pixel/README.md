# ELF pixel-vocab variant — flow-matching over glyph-strip pixels

A self-contained re-implementation, inside this ELF-pytorch repo and in its
PyTorch-Lightning idiom, of the **pixel-defined-vocabulary** setting: instead of
running ELF's flow over T5 token embeddings, we run it over the **raw pixels of a
rendered glyph strip**. Text is rendered to a fixed monospaced image via a glyph
*atlas* (one Unicode char → one fixed 16×8 glyph), the flow generates such an
image, and an OCR decode head reads it back to characters.

> **Status:** minimal baseline — pixel velocity-MSE **denoiser** + OCR
> cross-entropy **decoder**. Self-conditioning / CFG-distillation / in-context
> conditioning tokens are *wired but disabled by default* (see
> [§7 Extending](#7-extending-to-self-cond--cfg)). The code reuses ELF's flow
> math, sampler, EMA, optimizer, and trainer verbatim.

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

## 1. Geometry (defaults, faithful to the source experiment)

| quantity | value | where |
|---|---|---|
| image | 16 × 1280, grayscale, `[-1,+1]` (−1 ink, +1 white bg) | `img_height/img_width` |
| glyph cell | `char_h=16`, `char_w=8` | atlas meta (must match `img_height`,`char_w`) |
| patch | 16 × 64 → **N = 20** patch tokens | `patch_h/patch_w` |
| per-patch flow dim | **P = 16·64 = 1024** | derived (`model.text_encoder_dim`) |
| chars / patch token | `64/8 = 8` | derived (`chars_per_token`) |
| OCR target length | `20·8 = 160` chars | derived (`max_chars`) |
| vocab | bert-base-uncased glyph atlas, **V = 1000** (998 chars + UNK + EOS) | `pixel/assets/…` |
| backbone | ELFPixel-B: depth 12, hidden 768, 12 heads (~89M params) | `model` |

`max_length` (the transformer sequence length) is **derived** as
`img_width // patch_w`; the train entry overrides the YAML value if inconsistent.

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
  turns it back into a `(B,1,16,1280)` image for visualization.
- **OCR decode head**: `RMSNorm → Linear(hidden→decode_bottleneck) → GELU →
  Linear(→ cpt·V)` reshaped to `(B, N·cpt, V) = (B,160,V)`. Char alignment is exact
  because rendering is monospaced at `char_w` and `patch_w` is a multiple of it.

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

EMA, grad-clip(1.0), manual optimizer step, the LR schedule, and checkpoint
plumbing are all inherited from `ELFLitModule` unchanged. `x0` is produced by
`patchify(pixel_values·2-1)` instead of `encode_text(...)` — that single line is
the whole "no T5 encoder" difference.

---

## 5. The data — `pixel/glyph_dataset.py`

1. `load_lm1b(split)` downloads + extracts the One Billion Word Benchmark once
   (statmt.org tarball) and exposes it as an HF `Dataset` with a `text` column.
2. `ContinuousDocumentStreamDataset` builds a char-prefix index over
   `doc + EOS + next_doc + EOS …` and yields fixed-length windows of
   `img_width // char_w = 160` characters — `{"text": <window>}`.
3. `make_glyph_collate_with_labels` BERT-normalizes each window, maps codepoints
   through the LUT, gathers glyph cells from the in-memory atlas (no rendered image
   is ever cached), and emits `pixel_values (B,1,16,1280)` + `char_indices (B,160)`.

> Building the window index scans `limit_documents` documents up front (set
> `limit_documents` for a fast smoke run; `null` = the full corpus, which is slow
> to index once, exactly as in the source experiment).

---

## 6. Running it

```bash
cd pytorch_lightning/

# 4-GPU DDP, full config
torchrun --nproc_per_node=4 --master_port=29503 train_pixel_lightning.py \
    --config configs/training_configs/train_lm1b_pixel_ELF-B.yml

# fast smoke (few docs, tiny eval) — single GPU
torchrun --nproc_per_node=1 train_pixel_lightning.py \
    --config configs/training_configs/train_lm1b_pixel_ELF-B.yml \
    --config_override limit_documents=200000 --config_override epochs=2 \
    --config_override online_eval=false
```

Outputs land in `output_dir`: `last.ckpt` + per-epoch checkpoints, `config.yml`
snapshot, and (when `online_eval`) `pixel_epoch_NNN/{decoded.txt, *.png,
metrics.jsonl}` plus `eval/gen_ppl` + `eval/sample_entropy` to W&B.

**Key hyperparameters** (`train_lm1b_pixel_ELF-B.yml`): AdamW lr 2e-4 constant,
β(0.9,0.95), wd 0, batch 512, 60 epochs, bf16-mixed, EMA 0.9999; denoiser
`p_mean −0.8 / p_std 0.8 / noise_scale 1.0 / t_eps 0.05`; decoder `prob 0.2 /
p_mean 0.8 / p_std 0.8 / noise_scale 2.0 / bottleneck 256`; eval = 256-step ODE,
gen-PPL via gpt2-large.

> **Deps beyond the base repo:** `transformers` (atlas normalizer + gpt2-large —
> already required), `datasets` (LM1B — already required), and `Pillow` for PNG
> dumps (optional; the callback skips image writing if it's missing).

---

## 7. Extending to self-cond / CFG

The model and the reused flow utilities already support the full ELF feature set;
the minimal baseline just keeps the knobs off. To enable, in the YAML:

- **Self-conditioning**: set `self_cond_prob: 0.5`. The model is built with
  `self_cond_input=True` (2-channel patch input + `self_cond_proj`), and the
  training step needs the self-cond pre-forward added back (mirror
  `ELFLitModule.training_step` lines ~226-235, which already implement it for the
  `(B,L,D)` case — it transfers directly to `(B,N,P)`).
- **CFG / guidance distillation**: set `num_self_cond_cfg_tokens: 4` and
  `self_cond_cfg_min/max`. The cfg prefix tokens + embedder exist in `ELFPixel`;
  reuse the parent's guidance-distillation block (lines ~243-258) and pass an
  inference scale through `generate_pixel_samples`.
- **In-context conditioning** is already the design here (time/mode/cfg are prefix
  tokens, not adaLN), matching ELF; `num_model_mode_tokens` signals decode mode.

Because the only modality-specific surface is the patch-in / pixel-out / OCR head,
turning these on is a `training_step` edit, not an architecture change.

---

## 8. Provenance

Ported from the source experiment in the sibling `flm_ours` repo
(`pixel_predict_mse/flm_based/elf_like/`, branch `jinhyeonkwon/elf-like`): the
glyph atlas + renderer (`glyph_render.py`), the dual-head pixel flow model
(`model.py`/`denoiser.py`), and the LM1B windowing dataset. Re-expressed here in
ELF-pytorch's Lightning architecture so the flow math, sampler, EMA, optimizer,
and trainer are shared with the T5-embedding ELF rather than reimplemented.
