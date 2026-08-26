#!/bin/bash
# ============================================================================
# 04_hp_sweep.sh — the MRPC hyperparameter search for FourierFT on gemma-2b.
# ============================================================================
#   bash sbatch/fir/04_hp_sweep.sh --canary 2      # ⭐ ALWAYS DO THIS FIRST
#   bash sbatch/fir/04_hp_sweep.sh --time 01:30:00 # then the rest, sized from it
#   bash sbatch/fir/04_hp_sweep.sh --status        # what is done / failed / left
#
# ONE Slurm ARRAY, one cell per task, 160 cells (2 arms x 5 lr x 4 scaling x
# 4 classifier_lr), MRPC, 1 seed, 5 epochs.  Grid: scripts/fir_hp_plan.py.
#
# ⛔⛔ IT IS BUILT ON THE ASSUMPTION THAT CELLS WILL FAIL.  [user, 2026-08-26:
#    "don't make the same mistake of assuming all jobs will succeed"]
#      * every cell is INDEPENDENT -- one OOM cannot take the grid with it
#      * a cell writes `done/<id>` only after exit 0, so a re-run RESUMES and
#        never redoes finished work
#      * a failure writes `fail/<id>` with the exit code and the log tail, so the
#        reader can say WHICH cells are missing and WHY, instead of a short table
#        that looks complete
#      * ⛔ re-running this script is the recovery procedure. It is idempotent.
#
# ⭐ WHY A CANARY IS NOT OPTIONAL.  The per-cell wall-clock on an H100 has NEVER
#   been measured for this backbone; the 8-step preflight cells are
#   startup-dominated and say nothing about it.  --time is a HARD kill on fir: too
#   low and every cell dies at the wall having written nothing.  So measure two
#   cells, read the elapsed time out of `done/`, and size the array from it.
# ============================================================================
set -uo pipefail
FIR_SELF="$(readlink -f "$0")"
cd "$(dirname "$FIR_SELF")/../.." || exit 1
source sbatch/fir/fir_env.sh
# ⛔ NO ./logs TRANSCRIPT FOR AN ARRAY TASK. fir_log_to names its file by the
#   SECOND, and 160 array tasks start in bursts: two tasks in the same second
#   write the SAME file and interleave. The array's own
#   $SWEEP_ROOT/logs/slurm_%A_%a.out already captures each task separately, and
#   the per-cell train_glue log is written beside it. Only the submit/status
#   paths (one process, on a login node) get a transcript.
case " $* " in *" --run-one "*) FIR_LOGGING=1 ;; esac
fir_log_to fir_hp_sweep "$@"

P_TIME="${P_TIME:-02:00:00}"        # generous for the canary; SIZE IT after
P_CONCURRENT="${P_CONCURRENT:-8}"   # array throttle (%N)
P_CANARY=0
STATUS=false
LOCAL_ONE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --time) P_TIME="$2"; shift 2 ;;
        --concurrent) P_CONCURRENT="$2"; shift 2 ;;
        --canary) P_CANARY="$2"; shift 2 ;;
        --status) STATUS=true; shift ;;
        --run-one) LOCAL_ONE="$2"; shift 2 ;;   # internal: the array task body
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done

SWEEP_ROOT="$FIR_RUN_ROOT/hpsweep"
mkdir -p "$SWEEP_ROOT"/{csv,logs,done,fail}

# ===========================================================================
# THE ARRAY TASK BODY (one cell).  Runs on a compute node.
# ===========================================================================
if [ -n "$LOCAL_ONE" ]; then
    cid="$LOCAL_ONE"
    echo "### cell $cid  node=$(hostname)  job=${SLURM_JOB_ID:-none}"
    if [ -f "$SWEEP_ROOT/done/$cid" ]; then
        echo "### already done -- skipping (this is how a re-run resumes)"
        exit 0
    fi
    fir_load_modules_gpu || exit 1
    # shellcheck disable=SC1091
    source "$FIR_VENV/bin/activate" || exit 1
    fir_export_offline
    PY_BIN="$FIR_VENV/bin/python"          # ⚠ never bare `python` (a moved venv)
    [ -x "$PY_BIN" ] || { echo "FAIL: no interpreter at $PY_BIN"; exit 1; }
    rm -f "$SWEEP_ROOT/fail/$cid"
    t0=$SECONDS
    "$PY_BIN" scripts/fir_hp_run_cell.py --cell "$cid" --run-root "$SWEEP_ROOT"
    rc=$?
    el=$((SECONDS - t0))
    if [ $rc -eq 0 ]; then
        echo "$el" > "$SWEEP_ROOT/done/$cid"
        echo "### cell OK in ${el}s"
    else
        { echo "exit=$rc elapsed=${el}s node=$(hostname)"
          tail -25 "$SWEEP_ROOT/logs/$cid.log" 2>/dev/null; } > "$SWEEP_ROOT/fail/$cid"
        echo "### cell FAILED rc=$rc after ${el}s -- recorded in fail/$cid"
    fi
    exit $rc
fi

# ===========================================================================
# STATUS
# ===========================================================================
env/bin/python scripts/fir_hp_plan.py --list > "$SWEEP_ROOT/cells.txt" || exit 1
TOTAL=$(wc -l < "$SWEEP_ROOT/cells.txt")
NDONE=$(find "$SWEEP_ROOT/done" -type f 2>/dev/null | wc -l)
NFAIL=$(find "$SWEEP_ROOT/fail" -type f 2>/dev/null | wc -l)
if $STATUS; then
    fir_print_provenance
    echo "sweep root: $SWEEP_ROOT"
    echo "cells: $TOTAL   done: $NDONE   failed: $NFAIL   remaining: $((TOTAL - NDONE))"
    if [ "$NDONE" -gt 0 ]; then
        echo "--- measured per-cell wall-clock (seconds) ---"
        cat "$SWEEP_ROOT"/done/* 2>/dev/null | sort -n | awk '
            {a[NR]=$1; s+=$1}
            END {printf "  n=%d  min=%d  median=%d  max=%d  mean=%.0f\n",
                        NR, a[1], a[int((NR+1)/2)], a[NR], s/NR}'
        echo "  ⚠ size --time from the MAX, not the median: --time is a hard kill."
    fi
    [ "$NFAIL" -gt 0 ] && { echo "--- failures ---"; for f in "$SWEEP_ROOT"/fail/*; do
        echo "  $(basename "$f"): $(head -1 "$f")"; done; }
    echo
    echo "read: env/bin/python scripts/fir_hp_read.py --run-root $SWEEP_ROOT"
    exit 0
fi

# ===========================================================================
# SUBMIT
# ===========================================================================
fir_print_provenance
echo "--- login-node gate ---"
fir_assert_env cpu 02 || { echo "environment not sane — refusing to submit"; exit 1; }
for g in fir_arms fir_plan fir_hp_plan; do
    env/bin/python "scripts/$g.py" --selftest >/dev/null || { echo "FAIL: $g selftest"; exit 1; }
done
env/bin/python scripts/fir_hp_run_cell.py --selftest >/dev/null || { echo "FAIL: run_cell selftest"; exit 1; }
echo "  instrument selftests: OK"
echo
env/bin/python scripts/fir_hp_plan.py --show
echo
echo "  sweep root : $SWEEP_ROOT"
echo "  done       : $NDONE / $TOTAL   (failed so far: $NFAIL)"

# ⭐ THE ARRAY INDEXES cells.txt, and cells.txt is REGENERATED from the planner
#   every submit. The planner's order is deterministic and selftested, so an index
#   means the same cell on every submission.
if [ "$P_CANARY" -gt 0 ]; then
    LAST=$((P_CANARY - 1))
    # ⚠ the canary takes ONE CELL PER ARM, not the first N lines: the first 80 are
    #   all fftm, so a naive head -2 would never exercise the stock-PEFT path --
    #   and that is the arm whose build differs (modules_to_save, no module-count
    #   log). A canary that cannot see one of the two paths is not a canary.
    IDX=$(env/bin/python - <<'PY'
import sys; sys.path.insert(0, "scripts")
import fir_hp_plan as H
ids = [H.cell_id(c) for c in H.cells()]
pick = []
for arm in H.ARMS:                      # the DERIVED-ish point for each arm
    want = f"{H.TASK}-{arm}-{H.TARGETS}-lr{H._fmt(0.5)}-sc{H._fmt(142)}-clr{H._fmt(5e-3)}-seed{H.SEED}"
    pick.append(ids.index(want))
print(",".join(str(i) for i in pick))
PY
)
    ARRAY_SPEC="$IDX"
    echo "  CANARY: array=$ARRAY_SPEC  (one cell per arm, at lr 0.5 / scaling 142 / clf 5e-3)"
else
    # ⛔ RESUME MUST NOT RE-QUEUE FINISHED CELLS. The task body skips a done cell in
    #   seconds -- but only AFTER Slurm has allocated it a whole H100. On a 160-cell
    #   grid where 150 are done that is 150 pointless GPU allocations. Submit only
    #   the indices that have no `done` marker, collapsed into ranges.
    ARRAY_SPEC=$(env/bin/python - "$SWEEP_ROOT" <<'PY'
import os, sys
sys.path.insert(0, "scripts")
import fir_hp_plan as H
root = sys.argv[1]
ids = [H.cell_id(c) for c in H.cells()]
todo = [i for i, c in enumerate(ids) if not os.path.exists(os.path.join(root, "done", c))]
if not todo:
    print("")
    raise SystemExit(0)
runs, start, prev = [], todo[0], todo[0]
for i in todo[1:]:
    if i == prev + 1:
        prev = i; continue
    runs.append((start, prev)); start = prev = i
runs.append((start, prev))
print(",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs))
PY
)
    if [ -z "$ARRAY_SPEC" ]; then
        echo "  nothing to submit: every cell already has a done marker."
        echo "  read: env/bin/python scripts/fir_hp_read.py --run-root $SWEEP_ROOT"
        exit 0
    fi
    ARRAY_SPEC="$ARRAY_SPEC%$P_CONCURRENT"
    echo "  FULL  : array=$ARRAY_SPEC   (only cells with no done marker)"
fi
echo "  time/cell  : $P_TIME   account $FIR_ACCOUNT_GPU   gres $FIR_GPU_FULL   mem $FIR_GPU_MEM"
echo

jid=$(sbatch --parsable <<SB
#!/bin/bash
#SBATCH --job-name=lrs_hp_mrpc
#SBATCH --account=$FIR_ACCOUNT_GPU
#SBATCH --gpus=$FIR_GPU_FULL
#SBATCH --cpus-per-task=8
#SBATCH --mem=$FIR_GPU_MEM
#SBATCH --time=$P_TIME
#SBATCH --array=$ARRAY_SPEC
#SBATCH --output=$SWEEP_ROOT/logs/slurm_%A_%a.out
cd "\$SLURM_SUBMIT_DIR" || exit 1
set -uo pipefail
cid=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$SWEEP_ROOT/cells.txt")
[ -n "\$cid" ] || { echo "FAIL: no cell at index \$SLURM_ARRAY_TASK_ID"; exit 1; }
bash sbatch/fir/04_hp_sweep.sh --run-one "\$cid"
SB
)
echo "submitted array job $jid"
echo "watch:   squeue -j $jid"
echo "status:  bash sbatch/fir/04_hp_sweep.sh --status"
echo
if [ "$P_CANARY" -gt 0 ]; then
    echo "⛔ NEXT: when these two finish, run --status. It prints the MEASURED"
    echo "   per-cell seconds. Size --time from the MAX plus headroom, then submit"
    echo "   the full array. Do NOT submit 160 cells against an unmeasured wall."
fi
