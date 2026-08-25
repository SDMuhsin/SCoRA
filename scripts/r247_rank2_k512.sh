#!/bin/bash
# [R.247] RANK REALLOCATION AT THE k=512 BUDGET.
#
# At 512 params/module -- the ONE budget where SCoRA loses to FourierFT -- the rank
# axis has NEVER been touched: [verified, 229 SCoRA rows in the whole pool] only
# r=1,s=256 exists there.  [R.94]'s ladder was at 256 params, is NON-MONOTONE
# (r=1 > r=4 > r=2) and was not cost-matched, so it does not license a kill at 512.
#
# THE ONE KNOB vs r208_k512.sh:  --slr_rank 1 --slr_s 256  ->  --slr_rank 2 --slr_s 128
# Same 512 params/module.  s=128 is the largest subspace with a MEASURED 0% collapse
# rate ([R.223]: 0% at s=32/128, 20% at s=256, 100% at s=500/768), and the r=1,s=256
# arm's 1-in-5 dead seed is exactly that predicted 20%.
#
# Prereg + audit: llmdocs/R246_k512_sign_unstable_and_rank_never_reallocated.md 5
# Comparator: ALREADY MEASURED -- r208.csv FourierFT k=512, median 0.7617, 0 dead runs.
#   => only 5 new cells.
#
# ⚠️ COST REGRESSION IS PART OF THE RESULT, not a footnote: [derived] r=2/s=128 costs
#    6,936 flops/token vs r=1/s=256's 3,936 (1.76x dearer) and FourierFT k=512's 41,414
#    (still 6.0x cheaper).  A win here weakens the headline cost ratio 10.5x -> 6.0x.
#
# ⛔ SELF-GATING on box load, same rule as r240 ([R.209]): [R.237] is user-prioritised.
set -u
cd /workspace/lora_research_signal || exit 1
D=scratchpad/phaseR/r247; mkdir -p "$D"/{csv,logs,done,claim,failed}

njobs() { pgrep -af 'src/train_glue.py' | grep -c '^[0-9]\+ env/bin/python'; }
# ⛔ ORCHESTRATION BUG FIXED 2026-08-20: r240 and r247 were both frozen on the SAME
# "box load < 6" trigger, so they would have fired SIMULTANEOUSLY and taken the box to
# 8 -- exactly [R.209]'s measured 2.2x-inflation configuration, defeating the purpose
# of the gate and taxing [R.237] (user-prioritised).  Neither FROZEN RULE is changed:
# both still require load < 6.  This adds SERIALISATION only -- an orchestration fix,
# the same class as [R.200]'s silent-null fix, not an edit to an experimental rule.
# Anchored match per [R.194 5]: an unanchored pgrep matches this script's own shell.
r240_alive() { pgrep -af 'r240_subsample.sh' | grep -c '^[0-9]\+ bash scripts/'; }
THRESH=${R247_THRESH:-6}
DEADLINE=$(( $(date +%s) + ${R247_MAX_WAIT:-86400} ))
echo "[r247] waiting for box load < $THRESH (now $(njobs)) AND r240 to finish (alive=$(r240_alive)) $(date +%F' '%T)"
while [ "$(njobs)" -ge "$THRESH" ] || [ "$(r240_alive)" -gt 0 ]; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "[r247] deadline reached with box still loaded; exiting WITHOUT running." >&2; exit 3
  fi
  sleep 120
done
echo "[r247] box at $(njobs) -- starting $(date +%F' '%T)"

# COMMON copied VERBATIM from r208_k512.sh:5-7 (PROCESS.md 1.2, flag-by-flag)
COMMON="--model_name_or_path roberta-base --task_name rte --dtype float32 --weight_decay 0.01 \
 --adapter_target_modules query,value --per_device_train_batch_size 32 --num_train_epochs 30 \
 --learning_rate 5e-2 --classifier_lr 5e-3 --num_warmup_steps 140"
SLR2="--optimizer adamw-slr --slr_rank 2 --slr_s 128 --slr_seed 777 --slr_init zero \
 --slr_target_modules query,value"

for S in ${R247_SEEDS:-41 42 43 44 45}; do
  name="slr-r2s128-s$S"
  [ -f "$D/done/$name" ] && { echo "[r247] skip $name"; continue; }
  mkdir "$D/claim/$name" 2>/dev/null || { echo "[r247] claimed $name"; continue; }
  t0=$(date +%s); n0=$(njobs)
  GLUE_SEEDS=$S GLUE_RESULTS_FILE="$D/csv/$name.csv" \
    env/bin/python -u src/train_glue.py $COMMON $SLR2 --name "$name" > "$D/logs/$name.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ] && [ -s "$D/csv/$name.csv" ]; then
    touch "$D/done/$name"; echo "[r247] OK   $name ${dt}s njobs_at_start=$n0 $(date +%T)"
  else
    echo "rc=$rc dt=${dt}s" > "$D/failed/$name"; rmdir "$D/claim/$name" 2>/dev/null
    echo "[r247] FAIL $name rc=$rc ${dt}s $(date +%T)" >&2
  fi
done
echo "[r247] DRIVER EXIT done=$(ls $D/done | wc -l)/5 $(date +%F' '%T)"
