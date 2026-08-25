#!/bin/bash
# =============================================================================
# [R.310] ORCHESTRATOR -- the seven-task camera-ready grid, resumable.
#
#   env/bin/python scripts/r310_plan.py --selftest        # MUST pass first
#   nohup scripts/r310_run.sh > scratchpad/phaseR/r310/run.log 2>&1 &
#
# ⭐ TASKS RUN CHEAPEST FIRST (cb, mrpc, stsb, cola, boolq, sst2, mnli).  The
# whole grid is ~1,238 h at 3 workers, so it cannot be treated as atomic: this
# ordering means the five small tasks are complete in ~3 days and EVERY
# intermediate stopping point is a usable, complete-per-task table.  Stopping
# after `boolq` yields 6 of 8 columns rather than 8 partial ones.
#
# Resume: re-run it.  mkdir claims and `done` markers make every task idempotent
# and the planner never re-emits a label already in the manifest.
#
# Restrict the run:  R310_TASKS="cb mrpc" scripts/r310_run.sh
# =============================================================================
set -u
cd /workspace/lora_research_signal || exit 1
D=scratchpad/phaseR/r310
PLAN="env/bin/python scripts/r310_plan.py"
mkdir -p "$D"

# ⛔ Nothing in this file decides anything -- it only sequences.  The task list
# itself comes from the planner, so the shell cannot silently drop a task the
# user asked for (the planner's selftest asserts all seven are present).
DEFAULT_TASKS=$($PLAN --tasks) || { echo "[r310] cannot read task list"; exit 1; }
TASKS=${R310_TASKS:-$DEFAULT_TASKS}

$PLAN --selftest || { echo "[r310] SELFTEST FAILED -- refusing to spend GPU"; exit 1; }

# ⛔ REAP STALE CLAIMS -- see scripts/r310_reap.py for what this is and for the
# duplicate-run bug the previous inline shell version caused.  It is a gated
# instrument, not shell, because its failure mode is SILENT in both directions:
# not reaping loses cells forever, over-reaping runs two workers into one CSV.
env/bin/python scripts/r310_reap.py "$D" || exit 1

echo "[r310] ===== START $(date +%F' '%T) ====="
echo "[r310] task order: $TASKS"
for t in $TASKS; do
  echo "[r310] ---- task $t begins $(date +%F' '%T) ----"
  $PLAN --generate "$t" || exit 1
  [ -s "$D/jobs_${t}.tsv" ] || { echo "[r310] task $t empty, skipping"; continue; }
  scripts/r310_drive.sh "$t"
  # A task that did not fully complete is NOT a reason to stop: the remaining
  # tasks are independent columns, and a failed cell is re-run on resume.
  env/bin/python scripts/r310_read.py --task "$t" || true
done
echo "[r310] ===== ALL TASKS DONE $(date +%F' '%T) ====="
env/bin/python scripts/r310_read.py || true
