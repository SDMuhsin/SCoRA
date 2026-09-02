#!/bin/bash
# =============================================================================
# FP16 FULL-FT BASELINE -- orchestrator.  Two stages, cheapest task first.
#
#   env/bin/python scripts/fp16_baseline_plan.py --selftest    # MUST pass first
#   nohup scripts/fp16_baseline_run.sh > scratchpad/phaseR/fp16base/run.log 2>&1 &
#
# STAGE 1 `search` -- the full lr x batch ladder at ONE seed (84 cells).
# STAGE 2 `final`  -- the winning config at the other four seeds (24 cells).
# ⭐ Stage 2 reads stage 1's winner off disk, so the two are ONE resumable run and
#   the selection seed is never re-trained.
#
# Resume: re-run it.  Same mkdir-claim + `done`-marker machinery as [R.310] -- the
# driver is literally r310_drive.sh, pointed at a different state dir by R310_DIR.
# ⛔ SEPARATE STATE DIR ON PURPOSE: a shared one would put baseline labels into
#   [R.310]'s manifest and its reader's arm list.
# =============================================================================
set -u
cd /workspace/lora_research_signal || exit 1
D=scratchpad/phaseR/fp16base
PLAN="env/bin/python scripts/fp16_baseline_plan.py"
mkdir -p "$D"

TASKS=${FP16_TASKS:-$($PLAN --tasks)} || { echo "[fp16base] cannot read task list"; exit 1; }
$PLAN --selftest > "$D/selftest.log" 2>&1 || {
    echo "[fp16base] SELFTEST FAILED -- refusing to spend GPU"; tail -20 "$D/selftest.log"; exit 1; }

env/bin/python scripts/r310_reap.py "$D" || exit 1

echo "[fp16base] ===== START $(date +%F' '%T) ====="
echo "[fp16base] tasks: $TASKS"
SEL=$($PLAN --selection-task) || exit 1

# ⭐ STAGE 1 sweeps the SELECTION TASK ONLY -- one ladder, not six.
echo "[fp16base] ========== STAGE search (selection task: $SEL) =========="
$PLAN --generate "$SEL" --stage search || exit 1
[ -s "$D/jobs_${SEL}.tsv" ] || { echo "[fp16base] no search cells -- aborting"; exit 1; }
R310_DIR="$D" scripts/r310_drive.sh "$SEL"
$PLAN --read || true

# ⛔ THE PROXY MUST EXIST BEFORE ANY FINAL CELL RUNS. If the sweep did not complete,
#   `--proxy` exits non-zero and we stop here rather than carrying a winner chosen
#   from a partial ladder.
PROXY=$($PLAN --proxy) || {
    echo "[fp16base] ⛔ the sweep on $SEL produced no winner -- refusing to run the"
    echo "           final stage. Re-run this script; it resumes the sweep."
    exit 1; }
echo "[fp16base] ========== STAGE final (proxy carried: lr/bs = $PROXY) =========="
for t in $TASKS; do
  echo "[fp16base] ---- final / $t  $(date +%F' '%T) ----"
  $PLAN --generate "$t" --stage final || exit 1
  [ -s "$D/jobs_${t}.tsv" ] || { echo "[fp16base] $t empty, skipping"; continue; }
  R310_DIR="$D" scripts/r310_drive.sh "$t"
done
$PLAN --read || true
echo "[fp16base] ===== DONE $(date +%F' '%T) ====="
$PLAN --read || true
