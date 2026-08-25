#!/bin/bash
# [R.305] failure watcher.  Emits ONLY lines worth acting on:
#   * a cell that failed          * a stage boundary
#   * the driver dying while work remains   * a stall (no new cell for 90 min)
# Silence is NOT success here -- the driver-death and stall checks are what make
# a quiet stream mean "still running" rather than "died and nobody noticed".
set -u
cd /workspace/lora_research_signal || exit 1
D=scratchpad/phaseR/r305
seen_fail=0; last_done=-1; last_change=$(date +%s)
while true; do
  nf=$(ls "$D/failed" 2>/dev/null | wc -l)
  if [ "$nf" -gt "$seen_fail" ]; then
    for f in $(ls -t "$D/failed" | head -n $((nf - seen_fail))); do
      echo "FAIL $f :: $(cat "$D/failed/$f") :: $(grep -iE 'error|Traceback|assert' "$D/logs/$f.log" 2>/dev/null | tail -1)"
    done
    seen_fail=$nf
  fi
  grep -E '^\[r305\] stage .* EXIT|ALL STAGES DONE|SELFTEST FAILED|stage B converged' "$D/run.log" 2>/dev/null \
    | comm -13 "$D/.seen_marks" - 2>/dev/null
  grep -E '^\[r305\] stage .* EXIT|ALL STAGES DONE|SELFTEST FAILED|stage B converged' "$D/run.log" 2>/dev/null > "$D/.seen_marks"
  nd=$(ls "$D/done" 2>/dev/null | wc -l)
  now=$(date +%s)
  [ "$nd" -ne "$last_done" ] && { last_done=$nd; last_change=$now; }
  if ! pgrep -f r305_run.sh > /dev/null; then
    if ! grep -q 'ALL STAGES DONE' "$D/run.log" 2>/dev/null; then
      echo "DRIVER GONE with work remaining -- ${nd} cells done, $(ls $D/failed 2>/dev/null|wc -l) failed"
    fi
    exit 0
  fi
  if [ $((now - last_change)) -gt 5400 ]; then
    echo "STALL -- no new completed cell in 90 min (done=${nd}, live=$(pgrep -fc 'src/train_glue.py'))"
    last_change=$now
  fi
  sleep 300
done
