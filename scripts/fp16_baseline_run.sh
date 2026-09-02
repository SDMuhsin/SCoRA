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
for stage in search final; do
  echo "[fp16base] ========== STAGE $stage =========="
  for t in $TASKS; do
    echo "[fp16base] ---- $stage / $t  $(date +%F' '%T) ----"
    $PLAN --generate "$t" --stage "$stage" || exit 1
    [ -s "$D/jobs_${t}.tsv" ] || { echo "[fp16base] $t empty, skipping"; continue; }
    R310_DIR="$D" scripts/r310_drive.sh "$t"
  done
  $PLAN --read || true
done
echo "[fp16base] ===== DONE $(date +%F' '%T) ====="
$PLAN --read || true
