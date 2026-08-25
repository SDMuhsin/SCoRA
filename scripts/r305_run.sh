#!/bin/bash
# =============================================================================
# [R.305] ORCHESTRATOR -- A -> B(xN) -> C -> D, resumable, one command.
#
#   env/bin/python scripts/r305_plan.py --selftest   # MUST pass first
#   nohup scripts/r305_run.sh > scratchpad/phaseR/r305/run.log 2>&1 &
#
# Each stage's job list is GENERATED FROM THE PREVIOUS STAGE'S RESULTS by
# scripts/r305_plan.py, whose rules are frozen and fixture-tested (704
# assertions) BEFORE any cell ran -- PROCESS.md 5.1.  Nothing in this file
# decides anything; it only sequences.
#
# Resume: re-run it.  Claims and `done` markers make every stage idempotent,
# and the planner never re-emits a label already in the manifest.
# =============================================================================
set -u
cd /workspace/lora_research_signal || exit 1
D=scratchpad/phaseR/r305
PLAN="env/bin/python scripts/r305_plan.py"

$PLAN --selftest || { echo "[r305] SELFTEST FAILED -- refusing to spend GPU"; exit 1; }

run_stage() {
  local st=$1
  $PLAN --generate "$st" || exit 1
  [ -s "$D/jobs_${st}.tsv" ] || { echo "[r305] stage $st empty, skipping"; return 0; }
  scripts/r305_drive.sh "$st"
}

echo "[r305] ===== START $(date +%F' '%T) ====="
run_stage A
# Stage B is iterative: each round can reveal a NEW edge.  The planner caps it
# at MAX_EXTENSION_ROUNDS per arm and MAX_EXTENSION_CELLS overall, so this loop
# terminates on the planner's own preregistered rule, not on a hand judgement.
for round in 1 2; do
  before=$(wc -l < "$D/jobs_B.tsv" 2>/dev/null || echo 0)
  run_stage B
  after=$(wc -l < "$D/jobs_B.tsv" 2>/dev/null || echo 0)
  [ "$after" -eq "$before" ] && { echo "[r305] stage B converged after $((round-1)) round(s)"; break; }
done
run_stage C
run_stage D
echo "[r305] ===== ALL STAGES DONE $(date +%F' '%T) ====="
env/bin/python scripts/r305_read.py || true
