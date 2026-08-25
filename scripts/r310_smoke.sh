#!/bin/bash
# =============================================================================
# [R.310] SMOKE TEST -- PROCESS §2: *"'Generated' is not 'runnable'"*.
#
# Runs ONE real `train_glue.py` invocation per (arm x task-shape) at a tiny
# budget, using THE EXACT ARG STRINGS the planner emits (minus the budget), so
# that a flag an arm does not accept, or a task whose data path is untested,
# fails in ~6 minutes instead of 20 hours in.
#
# Task shapes covered -- one per distinct code path, not one per task:
#   stsb  regression (`is_regression`, pearson)      <- the riskiest path
#   cb    3-class + super_glue loader + macro f1
#   boolq super_glue binary, long passages
#   cola  matthews_correlation
# mrpc/sst2/mnli are the same glue classification path as RTE, already proven.
#
# Results land in scratchpad/phaseR/r310/smoke and are NEVER read as data:
# 64 examples for 1 epoch is not a measurement.
# =============================================================================
set -u
cd /workspace/lora_research_signal || exit 1
D=scratchpad/phaseR/r310/smoke
rm -rf "$D"; mkdir -p "$D"/{csv,logs}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
WORKERS=${R310_WORKERS:-3}

env/bin/python - > "$D/jobs.tsv" <<'PY'
import sys
sys.path.insert(0, "scripts")
import r310_plan as PL
sel = PL.selected_args()
S = PL.sizes()
TINY = ("--max_train_samples 64 --max_eval_samples 64 --num_train_epochs 1 "
        "--num_warmup_steps 1")
for task in ("stsb", "cb", "boolq", "cola"):
    common = PL.common(task, S)
    # strip the real budget; everything else is exactly what will run
    toks, out, i = common.split(), [], 0
    while i < len(toks):
        if toks[i] in ("--num_train_epochs", "--num_warmup_steps"):
            i += 2; continue
        out.append(toks[i]); i += 1
    for arm in PL.ARM_NAMES:
        print(f"smoke-{task}-{arm}\t{' '.join(out)} {TINY} {sel[arm]}")
PY

N=$(wc -l < "$D/jobs.tsv")
echo "[r310-smoke] $N cells, $WORKERS workers, $(date +%T)"

worker() {
  while IFS=$'\t' read -r label args; do
    [ -z "${label:-}" ] && continue
    mkdir "$D/.claim-$label" 2>/dev/null || continue
    GLUE_SEEDS=41 GLUE_RESULTS_FILE="$D/csv/$label.csv" \
      env/bin/python -u src/train_glue.py $args --name "$label" \
      > "$D/logs/$label.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -s "$D/csv/$label.csv" ]; then
      echo "  OK   $label"
    else
      echo "  FAIL $label  rc=$rc :: $(grep -iE 'error|Traceback|unrecognized|assert' "$D/logs/$label.log" | tail -1)" >&2
    fi
  done < "$D/jobs.tsv"
}
for w in $(seq 1 "$WORKERS"); do worker & done
wait

# ⛔ train_glue writes a sibling `<name>.csv.lock`; counting the DIRECTORY
# double-counts every cell and made this gate report 72/36.
OK=$(ls "$D"/csv/*.csv 2>/dev/null | wc -l)
echo "[r310-smoke] $OK/$N produced a results row  $(date +%T)"
[ "$OK" -eq "$N" ] || { echo "[r310-smoke] ⛔ NOT ALL ARMS RUN -- do not launch"; exit 1; }
# ⛔ A CSV is not enough: the reader must be able to SCORE it on the task's own
# metric.  A run that writes NaN into `matthews_correlation` would pass the file
# check and vanish from the table later.
env/bin/python - "$D/csv" <<'PY'
import sys
sys.path.insert(0, "scripts")
import r310_read as R
res = R.load(sys.argv[1])
bad = [l for l in sorted(res) if res[l]["val"] != res[l]["val"]]
print(f"[r310-smoke] reader scored {len(res)} cells on their own metric; unscorable: {bad}")
import os
missing = [f for f in os.listdir(sys.argv[1])
           if f.endswith(".csv") and f[:-4] not in res]
if missing:
    print(f"[r310-smoke] ⛔ {len(missing)} CSVs the reader could NOT score: {missing}")
    raise SystemExit(1)
for l in sorted(res):
    print(f"    {l:28s} {res[l]['metric']:22s} {res[l]['val']:+.4f}")
PY
