#!/bin/bash
# =============================================================================
# [R.311-ablation] ORCHESTRATOR -- the multi-task (r, s) rank ladder, resumable.
#
#   env/bin/python scripts/r311ab_plan.py --selftest       # MUST pass first
#   nohup scripts/r311ab_run.sh > scratchpad/phaseR/r311ab/run.log 2>&1 &
#
# ⭐ It REUSES `scripts/r310_drive.sh` -- the worker pool that ran the 315-cell
# grid -- via its `R310_DIR` parameter, rather than shipping a second driver.
# The traps that driver is built around (T1 seed-in-the-CSV-name, T5 mkdir
# claiming, T6 completion = rc0 AND non-empty CSV) are the same traps here, and
# a copy of it would be a second place for them to be got wrong.
#
# ⭐ TASKS RUN CHEAPEST FIRST (cola, stsb, boolq).  Every intermediate stopping
# point is a COMPLETE ladder on the tasks finished so far -- so if the 24 h limit
# bites, what exists is a usable 2-task discharge, not 3 partial ones.
#
# Resume: re-run it.  Idempotent -- `done` markers and the planner's manifest.
# Restrict:  R311AB_TASKS="cola" scripts/r311ab_run.sh
# =============================================================================
set -u
cd /workspace/lora_research_signal || exit 1
D=scratchpad/phaseR/r311ab
PLAN="env/bin/python scripts/r311ab_plan.py"
mkdir -p "$D"

# CONTEXT §4.6: this box is SHARED.  cuda:1 is another user's job; use cuda:0.
export CUDA_VISIBLE_DEVICES=0
export R310_DIR="$D"
export R310_WORKERS=${R311AB_WORKERS:-3}

DEFAULT_TASKS=$($PLAN --tasks) || { echo "[r311ab] cannot read task list"; exit 1; }
TASKS=${R311AB_TASKS:-$DEFAULT_TASKS}

$PLAN --selftest || { echo "[r311ab] SELFTEST FAILED -- refusing to spend GPU"; exit 1; }

env/bin/python scripts/r310_reap.py "$D" || exit 1

echo "[r311ab] ===== START $(date +%F' '%T) ====="
echo "[r311ab] task order: $TASKS | workers $R310_WORKERS | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
for t in $TASKS; do
  echo "[r311ab] ---- task $t begins $(date +%F' '%T) ----"
  $PLAN --generate "$t" || exit 1
  [ -s "$D/jobs_${t}.tsv" ] || { echo "[r311ab] task $t empty, skipping"; continue; }
  scripts/r310_drive.sh "$t"
  $PLAN --status || true
done
echo "[r311ab] ===== ALL TASKS DONE $(date +%F' '%T) ====="
$PLAN --report || true
