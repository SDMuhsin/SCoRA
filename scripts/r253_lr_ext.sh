#!/bin/bash
# [R.253] FourierFT LR-ladder EXTENSION -- lr = 5e-1 at all four scalings, 4 cells.
#
# WHY: [R.252 6] FourierFT's plane argmax sits at lr = 1.5e-1, the TOP of the swept
# ladder, so its tuned level (0.7942, +0.0397 over the [R.68] banked config on the
# same seed) is a LOWER BOUND, not an optimum.  FourierFT is THE comparator in
# STANDING #1 and #2, so leaving it unbracketed would understate the very baseline
# our own margins are measured against.
#
# The scaling ladder is ALREADY bracketed at that rung (argmax s=100, interior in
# {50,100,150,300}), so ONLY lr is extended.  One rung.  4 cells, ~1 h.
#
# ⛔ SCOPE, per the user: "feel free to increase the ladder" AND "don't feel
#    obligated to chase a win for the baselines."  This BRACKETS; it does not chase.
#
# ## STOPPING RULE, FROZEN BEFORE ANY CELL RUNS
#    * lr 5e-1 LOSES to 1.5e-1 at the argmax -> BRACKETED.  Report a genuine optimum.
#    * lr 5e-1 WINS                          -> NOT bracketed.  Report the tuned
#      FourierFT level as a LOWER BOUND and label it so.  ⛔ DO NOT extend to 1.5.
#
# The cells are ALSO appended to r237/jobs.tsv so the grid's totals and
# scripts/r237_read.py stay consistent; they write into r237's own csv/ and done/
# dirs, so the reader picks them up with no change.  mkdir-claim makes it safe if an
# r237 worker reaches them independently.
set -u
cd /workspace/lora_research_signal || exit 1
D=scratchpad/phaseR/r237                      # SAME dirs as the grid, by design
JOBS=$D/ext_lr5e1.tsv

njobs() { pgrep -af 'src/train_glue.py' | grep -c '^[0-9]\+ env/bin/python'; }
THRESH=${R253_THRESH:-7}                      # one extra worker alongside r237's three
DEADLINE=$(( $(date +%s) + ${R253_MAX_WAIT:-172800} ))

COMMON="--model_name_or_path roberta-base --task_name rte --dtype float32 \
 --adapter_target_modules query,value --per_device_train_batch_size 32 \
 --num_train_epochs 30 --num_warmup_steps 140"

while IFS=$'\t' read -r label args; do
  [ -z "${label:-}" ] && continue
  [ -f "$D/done/$label" ] && { echo "[r253] skip $label"; continue; }
  while [ "$(njobs)" -ge "$THRESH" ]; do
    [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "[r253] deadline; exiting" >&2; exit 3; }
    sleep 120
  done
  mkdir "$D/claim/$label" 2>/dev/null || { echo "[r253] claimed $label"; continue; }
  t0=$(date +%s); n0=$(njobs)
  GLUE_SEEDS=41 GLUE_RESULTS_FILE="$D/csv/$label.csv" \
    env/bin/python -u src/train_glue.py $COMMON $args --name "$label" > "$D/logs/$label.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ] && [ -s "$D/csv/$label.csv" ]; then
    touch "$D/done/$label"; echo "[r253] OK   $label ${dt}s njobs_at_start=$n0 $(date +%T)"
  else
    echo "rc=$rc dt=${dt}s" > "$D/failed/$label"; rmdir "$D/claim/$label" 2>/dev/null
    echo "[r253] FAIL $label rc=$rc ${dt}s $(date +%T)" >&2
  fi
done < "$JOBS"
echo "[r253] DRIVER EXIT $(date +%F' '%T)"
