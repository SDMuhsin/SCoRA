#!/bin/bash
# [R.310] failure watcher.  Emits ONLY lines worth acting on:
#   * a cell that failed             * a task boundary
#   * the driver dying while work remains      * a stall
# Silence is NOT success here -- the driver-death and stall checks are what make
# a quiet stream mean "still running" rather than "died and nobody noticed".
#
# ⭐ The stall threshold is PER TASK, because a single MNLI cell legitimately runs
# ~66 h and a flat 90-minute threshold (which is what [R.305] used on RTE) would
# cry wolf on every one of them.  It is derived from the planner's own projected
# h/cell, so it cannot drift away from the schedule.
set -u
cd /workspace/lora_research_signal || exit 1
D=scratchpad/phaseR/r310
touch "$D/.seen_marks"
seen_fail=$(ls "$D/failed" 2>/dev/null | wc -l)
last_done=-1; last_change=$(date +%s)

while true; do
  nf=$(ls "$D/failed" 2>/dev/null | wc -l)
  if [ "$nf" -gt "$seen_fail" ]; then
    for f in $(ls -t "$D/failed" | head -n $((nf - seen_fail))); do
      echo "FAIL $f :: $(cat "$D/failed/$f") :: $(grep -iE 'error|Traceback|assert|CUDA' "$D/logs/$f.log" 2>/dev/null | tail -1)"
    done
    seen_fail=$nf
  fi

  grep -E '^\[r310\] (task .* (begins|EXIT)|ALL TASKS DONE|SELFTEST FAILED)' "$D/run.log" 2>/dev/null \
    | comm -13 "$D/.seen_marks" - 2>/dev/null
  grep -E '^\[r310\] (task .* (begins|EXIT)|ALL TASKS DONE|SELFTEST FAILED)' "$D/run.log" 2>/dev/null > "$D/.seen_marks"

  nd=$(ls "$D/done" 2>/dev/null | wc -l)
  now=$(date +%s)
  [ "$nd" -ne "$last_done" ] && { last_done=$nd; last_change=$now; }

  if ! pgrep -f r310_run.sh > /dev/null; then
    if ! grep -q 'ALL TASKS DONE' "$D/run.log" 2>/dev/null; then
      echo "DRIVER GONE with work remaining -- ${nd} cells done, $(ls "$D/failed" 2>/dev/null | wc -l) failed"
    fi
    exit 0
  fi

  # stall budget = 3x the projected cell time of the task currently running
  CUR=$(grep -oE '^\[r310\] ---- task [a-z2]+ begins' "$D/run.log" 2>/dev/null | tail -1 | awk '{print $4}')
  BUDGET=$(env/bin/python - "$CUR" <<'PY' 2>/dev/null || echo 5400
import sys, os
sys.path.insert(0, "scripts")
import r310_plan as PL
t = sys.argv[1] if len(sys.argv) > 1 else ""
if t not in PL.TASKS:
    print(5400); raise SystemExit
print(int(max(5400, 3 * PL.total_steps(t) * 0.65)))
PY
)
  if [ $((now - last_change)) -gt "$BUDGET" ]; then
    echo "STALL -- no new completed cell in $((BUDGET/3600))h on task ${CUR:-?} (done=${nd}, live=$(pgrep -fc 'src/train_glue.py'))"
    last_change=$now
  fi
  sleep 300
done
