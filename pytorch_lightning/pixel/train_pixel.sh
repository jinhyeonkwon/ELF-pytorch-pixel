#!/bin/bash
# Launcher for the ELF pixel-vocab variant (glyph-strip flow) on LM1B.
#
#   bash pixel/train_pixel.sh                      # full run, all visible GPUs
#   GPUS=0,1,2,3 bash pixel/train_pixel.sh         # pick GPUs
#   SMOKE=1 bash pixel/train_pixel.sh              # fast sanity run (few docs, 2 ep, no eval)
#   bash pixel/train_pixel.sh --config_override epochs=30 --config_override lr=1e-4
#
# Any extra args are forwarded verbatim to train_pixel_lightning.py (use
# --config_override key=value to tweak the YAML). Resume is automatic: if
# $OUTPUT_DIR/last.ckpt exists it continues that run (same W&B run too).
set -euo pipefail

# --- run from pytorch_lightning/ so configs/ and pixel/assets resolve ---------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../pytorch_lightning/pixel
PL_DIR="$(dirname "$SCRIPT_DIR")"                            # .../pytorch_lightning
cd "$PL_DIR"

# --- knobs (override via env) -------------------------------------------------
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export CUDA_VISIBLE_DEVICES="$GPUS"
NUM_GPUS="$(awk -F',' '{print NF}' <<< "$GPUS")"

CONFIG="${CONFIG:-configs/training_configs/train_lm1b_pixel_ELF-B.yml}"
TORCHRUN="${TORCHRUN:-/home/work/miniconda3/envs/jit_jh/bin/torchrun}"   # env with torch+lightning+flash_attn
OUTPUT_DIR="${OUTPUT_DIR:-outputs/elf_pixel_lm1b}"

# --- caches: reuse the already-downloaded LM1B + HF models ---------------------
export DATA_DIR="${DATA_DIR:-/home/work/RADAR/workspace/KAIST/pixel_lm/jinhyeon/data}"  # -> $DATA_DIR/lm1b
export HF_HOME="${HF_HOME:-/home/work/.cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# W&B: needs a prior `wandb login`. Set WANDB_MODE=offline to log locally only,
# or pass --config_override use_wandb=false to disable.

mkdir -p "$OUTPUT_DIR" logs
LOG_FILE="logs/$(basename "$OUTPUT_DIR").log"

# --- SMOKE: tiny, fast end-to-end sanity (overrides win over the YAML) --------
EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA+=(--config_override limit_documents=20000
          --config_override epochs=2
          --config_override global_batch_size="$((8 * NUM_GPUS))"
          --config_override online_eval=false
          --config_override save_freq=1
          --config_override use_wandb=false)
  echo "[train_pixel] SMOKE mode: tiny LM1B slice, 2 epochs, no eval/wandb."
fi

echo "[train_pixel] GPUs=$CUDA_VISIBLE_DEVICES (nproc=$NUM_GPUS) | config=$CONFIG | out=$OUTPUT_DIR"
echo "[train_pixel] log -> $LOG_FILE"

"$TORCHRUN" --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    train_pixel_lightning.py \
    --config "$CONFIG" \
    --config_override output_dir="$OUTPUT_DIR" \
    "${EXTRA[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"
