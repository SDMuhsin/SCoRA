#!/usr/bin/env bash
# [CLM PILOT] -- the calibration run, at the CORRECTED protocol (per-block BOS, TF32).
#
# Answers exactly two questions before anything is preregistered:
#   1. Does fine-tuning an adapter-only causal LM move perplexity enough to RANK arms?
#   2. What is the seed spread, i.e. how big must an effect be to survive it?
#
# ⛔ ONE SEQUENTIAL DRIVER (PROCESS.md 6). Cells in separate queues are not comparable
#    even on the same box, and an 8th concurrent job inflates every running cell 2.2x.
# ⛔ cuda:0 only -- shared box, cuda:1 stays free (memory: shared-server-cuda0-only).
# ⚠ These are CALIBRATION cells. They inform design; they are NOT study cells and a
#   prereg that wants them must say so explicitly (PROCESS.md 1.1).
set -u
cd "$(dirname "$0")/.."
PY=env/bin/python
COMMON="--model_name_or_path google/gemma-2b --dtype float32 --tf32 \
 --adapter_target_modules q_proj,o_proj --block_size 512 \
 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
 --num_train_epochs 3 --warmup_ratio 0.05983 --weight_decay 0.01 \
 --seeds 42,43,44 --log_every 150 --name pilot --cell_dir scratchpad/clm"

# SCoRA at the frozen gemma-2b proxy lr (GEMMA_HP_PROXY.md; same backbone, same
# targets, same k, so the raw lr carries -- the atom is unchanged).
CUDA_VISIBLE_DEVICES=0 $PY -u src/train_clm.py $COMMON \
  --optimizer adamw-slr --slr_rank 1 --slr_s 128 --slr_init zero --slr_seed 777 \
  --learning_rate 0.0451498 >> logs/clm_pilot_scora.log 2>&1

# FourierFT (merged), the comparator the letter's headline is against.
CUDA_VISIBLE_DEVICES=0 $PY -u src/train_clm.py $COMMON \
  --optimizer adamw-fourierftmerged --fourierftmerged_k 256 --fourierftmerged_seed 777 \
  --fourierftmerged_scaling 400 --learning_rate 1.5 >> logs/clm_pilot_fftm.log 2>&1

echo "[pilot] done"
