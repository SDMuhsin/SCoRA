#!/bin/bash
# =============================================================================
# [R.313] ORCHESTRATOR -- the multi-task basis control, resumable.
#
#   env/bin/python scratchpad/phaseR/r313_gate.py        # MUST pass first
#   env/bin/python scripts/r313_plan.py --selftest       # MUST pass first
#   nohup scripts/r313_run.sh > scratchpad/phaseR/r313/run.log 2>&1 &
#
# It REUSES `scripts/r310_drive.sh` via `R310_DIR`, as [R.311-ablation] does,
# rather than shipping a second worker pool.
#
# TASKS RUN CHEAPEST FIRST (cola, stsb, boolq), so every intermediate stopping
# point is a COMPLETE control on the tasks finished so far.
#
# Resume: re-run it. Idempotent -- `done` markers and the planner's manifest.
# =============================================================================
set -u
cd /workspace/lora_research_signal || exit 1
D=scratchpad/phaseR/r313
PLAN="env/bin/python scripts/r313_plan.py"
mkdir -p "$D"

# CONTEXT: this box is SHARED. cuda:1 is another user's job; use cuda:0.
export CUDA_VISIBLE_DEVICES=0
export R310_DIR="$D"
export R310_WORKERS=${R313_WORKERS:-3}

DEFAULT_TASKS=$($PLAN --tasks) || { echo "[r313] cannot read task list"; exit 1; }
TASKS=${R313_TASKS:-$DEFAULT_TASKS}

env/bin/python scratchpad/phaseR/r313_gate.py || { echo "[r313] GATE FAILED -- refusing to spend GPU"; exit 1; }
$PLAN --selftest || { echo "[r313] SELFTEST FAILED -- refusing to spend GPU"; exit 1; }

env/bin/python scripts/r310_reap.py "$D" || exit 1

echo "[r313] ===== START $(date +%F' '%T) ====="
echo "[r313] task order: $TASKS | workers $R310_WORKERS | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
for t in $TASKS; do
  echo "[r313] ---- task $t begins $(date +%F' '%T) ----"
  $PLAN --generate "$t" || exit 1
  [ -s "$D/jobs_${t}.tsv" ] || { echo "[r313] task $t empty, skipping"; continue; }
  scripts/r310_drive.sh "$t"
  $PLAN --status || true
done
echo "[r313] ===== ALL TASKS DONE $(date +%F' '%T) ====="
$PLAN --report || true
