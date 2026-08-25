#!/bin/bash
# [R.249] LoCA alpha-ladder EXTENSION -- alpha = 8.0 at all five LR rungs, 5 cells.
#
# WHY: [R.244 8] measured BOTH complete LoCA rungs putting their argmax at alpha = 4.0,
# the TOP of the swept ladder {0.5,1,2,4}, with a monotone rise across it.  A boundary
# argmax does not establish an optimum, and PROCESS.md 5 test 5 requires a baseline at
# its OWN optimum -- so the grid as frozen could only ever have reported "best within
# the swept range" for this arm.
#
# ⛔ SCOPE, per the user (2026-08-20): "feel free to increase the ladder" AND
#    "don't feel obligated to chase a win for the baselines."
#    => this extension exists to BRACKET the optimum, not to maximise LoCA.
#    ONE rung is added (alpha=8), not a sweep to alpha=16/32.
#
# ## STOPPING RULE, FROZEN BEFORE ANY CELL RUNS
#    * alpha=8 LOSES to alpha=4 at the plane's argmax  -> the optimum is BRACKETED.
#      The LoCA arm reports a genuine interior optimum.  DONE.
#    * alpha=8 WINS at the plane's argmax               -> the optimum is NOT bracketed.
#      Report the value as a LOWER BOUND and label it as such.  ⛔ DO NOT extend to
#      alpha=16.  Chasing a baseline's maximum is explicitly out of scope.
#
# The cells are ALSO appended to r237/jobs.tsv so the grid's totals and
# scripts/r237_read.py stay consistent; they write into r237's own csv/ and done/
# dirs, so the reader picks them up with no change.  mkdir-claim makes it safe if an
# r237 worker reaches them independently.
set -u
cd /workspace/lora_research_signal || exit 1
D=scratchpad/phaseR/r237                      # SAME dirs as the grid, by design
JOBS=$D/ext_alpha8.tsv

njobs() { pgrep -af 'src/train_glue.py' | grep -c '^[0-9]\+ env/bin/python'; }
THRESH=${R249_THRESH:-7}                      # one extra worker alongside r237's three
DEADLINE=$(( $(date +%s) + ${R249_MAX_WAIT:-172800} ))

COMMON="--model_name_or_path roberta-base --task_name rte --dtype float32 \
 --adapter_target_modules query,value --per_device_train_batch_size 32 \
 --num_train_epochs 30 --num_warmup_steps 140"

while IFS=$'\t' read -r label args; do
  [ -z "${label:-}" ] && continue
  [ -f "$D/done/$label" ] && { echo "[r249] skip $label"; continue; }
  while [ "$(njobs)" -ge "$THRESH" ]; do
    [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "[r249] deadline; exiting" >&2; exit 3; }
    sleep 120
  done
  mkdir "$D/claim/$label" 2>/dev/null || { echo "[r249] claimed $label"; continue; }
  t0=$(date +%s); n0=$(njobs)
  GLUE_SEEDS=41 GLUE_RESULTS_FILE="$D/csv/$label.csv" \
    env/bin/python -u src/train_glue.py $COMMON $args --name "$label" > "$D/logs/$label.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ] && [ -s "$D/csv/$label.csv" ]; then
    touch "$D/done/$label"; echo "[r249] OK   $label ${dt}s njobs_at_start=$n0 $(date +%T)"
  else
    echo "rc=$rc dt=${dt}s" > "$D/failed/$label"; rmdir "$D/claim/$label" 2>/dev/null
    echo "[r249] FAIL $label rc=$rc ${dt}s $(date +%T)" >&2
  fi
done < "$JOBS"
echo "[r249] DRIVER EXIT $(date +%F' '%T)"
