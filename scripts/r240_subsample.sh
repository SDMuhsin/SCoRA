#!/bin/bash
# [R.240] Is STANDING #2's k=64 margin a SMALL-DATA effect?
# CoLA truncated to N_train = 2490 (RTE's exact size => RTE's exact 2340 steps and,
# at the shared warmup 140, RTE's exact 5.98% warmup ratio).  ONE knob vs [R.191]'s
# CoLA column: --max_train_samples 2490.
#
# Prereg: llmdocs/R240_subsample_prereg.md (frozen, NO CALL declared).
# Motivation: llmdocs/R239_margin_confounded_with_dataset_size.md
# Gate: scripts/r239_gate_subsample.py 7/7 (G2 fails on pre-fix code).
# Reader: scripts/r240_read.py
#
# ⛔ SELF-GATING: [R.209] measured an 8th concurrent job inflating every running cell
# ~2.2x, and [R.237] (147 cells, user-prioritised) is live.  This driver WAITS until
# the box drops below 6 training processes before claiming its first cell, and gates
# on a DEADLINE as well as a condition ([R.210 7]: never a count alone).
set -u
cd /workspace/lora_research_signal || exit 1
D=scratchpad/phaseR/r240; mkdir -p "$D"/{csv,logs,done,claim,failed}

njobs() { pgrep -af 'src/train_glue.py' | grep -c '^[0-9]\+ env/bin/python'; }

THRESH=${R240_THRESH:-6}
DEADLINE=$(( $(date +%s) + ${R240_MAX_WAIT:-86400} ))
echo "[r240] waiting for box load < $THRESH (now $(njobs)) $(date +%F' '%T)"
while [ "$(njobs)" -ge "$THRESH" ]; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "[r240] deadline reached with box still loaded; exiting WITHOUT running." >&2
    exit 3
  fi
  sleep 120
done
echo "[r240] box at $(njobs) -- starting $(date +%F' '%T)"

COMMON="--model_name_or_path roberta-base --task_name cola --dtype float32 \
 --weight_decay 0.01 --adapter_target_modules query,value \
 --per_device_train_batch_size 32 --num_train_epochs 30 \
 --learning_rate 5e-2 --classifier_lr 5e-3 --num_warmup_steps 140 \
 --max_train_samples 2490"

FFT="--optimizer adamw-fourierftmerged --fourierftmerged_seed 777 --fourierftmerged_scaling 150.0 \
 --fourierftmerged_support scattered --fourierftmerged_target_modules query,value"
SCO="--optimizer adamw-slr --slr_rank 1 --slr_init zero --slr_seed 777 --slr_target_modules query,value"

# k=256 first (the comparator rung), then k=64 -- so a partial run still yields the
# budget-cost asymmetry X4 needs, and [R.191]'s CoLA column is the full-N mirror.
for ARM in fft-k256 scora-k256 fft-k64 scora-k64; do
  case "$ARM" in
    fft-k256)   E="$FFT --fourierftmerged_k 256";;
    fft-k64)    E="$FFT --fourierftmerged_k 64";;
    scora-k256) E="$SCO --slr_s 128";;
    scora-k64)  E="$SCO --slr_s 32";;
  esac
  for S in 41 42 43 44 45; do
    name="$ARM-s$S"
    [ -f "$D/done/$name" ] && { echo "[r240] skip $name"; continue; }
    mkdir "$D/claim/$name" 2>/dev/null || { echo "[r240] claimed $name"; continue; }
    t0=$(date +%s); n0=$(njobs)
    GLUE_SEEDS=$S GLUE_RESULTS_FILE="$D/csv/$name.csv" \
      env/bin/python -u src/train_glue.py $COMMON $E --name "$name" > "$D/logs/$name.log" 2>&1
    rc=$?; dt=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ] && [ -s "$D/csv/$name.csv" ]; then
      touch "$D/done/$name"
      echo "[r240] OK   $name ${dt}s njobs_at_start=$n0 $(date +%T)"
    else
      echo "rc=$rc dt=${dt}s" > "$D/failed/$name"; rmdir "$D/claim/$name" 2>/dev/null
      echo "[r240] FAIL $name rc=$rc ${dt}s $(date +%T)" >&2
    fi
  done
done
echo "[r240] DRIVER EXIT done=$(ls $D/done | wc -l)/20 $(date +%F' '%T)"
