#!/bin/bash
# ============================================================================
# 04_hp_sweep.sh — the MRPC hyperparameter search on gemma-2b.  ONE script, MANY
# grids: FIR_HP_GRID selects which.  ⛔ It is NOT forked per arm on purpose — this
# is the one file in the tree with a local self-test (scripts/fir_shell_gates.py),
# and five copies of it would be five copies of the resume, wall-kill-marker,
# receipt and dry-run machinery, four of them untested.
# ============================================================================
#   bash sbatch/fir/04_hp_sweep.sh --canary 2      # ⭐ ALWAYS DO THIS FIRST
#   bash sbatch/fir/04_hp_sweep.sh --time 01:30:00 # then the rest, sized from it
#   bash sbatch/fir/04_hp_sweep.sh --status        # what is done / failed / left
#   bash sbatch/fir/04_hp_sweep.sh --dry-run       # print the array spec, submit nothing
#
#   FIR_HP_GRID=w1 bash sbatch/fir/04_hp_sweep.sh --canary 2    # ⭐ the WaveFT grid
#
# GRIDS (scripts/fir_hp_plan.py owns every ladder and its justification):
#   g1 g2      FourierFT  — DONE, 142 cells/arm
#   w1 w2      WaveFT     — DONE, 144 cells/arm     (`wave` = a READING VIEW of both)
#   loca       LoCA       144   qwha  QWHA   144    lyra  LYRA  192
#   scora      SCoRA      36    scora2 SCoRA-2 140  (ours — deliberately the smallest)
#   wref       WaveFT at its own PUBLISHED point — 4 REF cells, no canary (no centre)
#   locax qwhax lyrax scora2x   ⭐ EDGE PROBES — 2 cells each, no canary (every cell
#              is one step PAST an edge, so none of them is central). Submit whole.
#              Each holds its base grid's WINNING cell fixed and steps ONE axis past
#              the edge the reader flagged; read with fir_hp_read.py, which scores
#              them as DELTAS against that anchor. ⚠ scora2x is OUR arm and uses its
#              last 2 cells of budget headroom (140+2 = the comparator's 142).
#
# ONE Slurm ARRAY, one cell per task.  MRPC, q_proj+o_proj, 1 seed (42), 5 epochs.
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
DRY=false
LOCAL_ONE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --time) P_TIME="$2"; shift 2 ;;
        --concurrent) P_CONCURRENT="$2"; shift 2 ;;
        --canary) P_CANARY="$2"; shift 2 ;;
        --status) STATUS=true; shift ;;
        --dry-run) DRY=true; shift ;;   # compute the array spec and STOP. see below
        --run-one) LOCAL_ONE="$2"; shift 2 ;;   # internal: the array task body
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done

GRID_NAME="${FIR_HP_GRID:-$(env/bin/python -c "import sys;sys.path.insert(0,'scripts');import fir_hp_plan as H;print(H.GRID_NAME)")}"
export FIR_HP_GRID="$GRID_NAME"
# ⚠ ONE sweep root for every grid, deliberately: a cell id is a pure function of its
#   knobs, so cells shared by two grids reuse their CSV and their `done` marker and
#   cost nothing on the re-run. Do NOT split the root per grid.
SWEEP_ROOT="$FIR_RUN_ROOT/hpsweep"
mkdir -p "$SWEEP_ROOT"/{csv,logs,done,fail,started,plans}

# ⛔⛔ THE CELL LIST IS A PER-SUBMISSION SNAPSHOT, NOT A SHARED FILE. [2026-08-28]
#   It used to be ONE `$SWEEP_ROOT/cells.txt`, rewritten by every submit. Five
#   canaries submitted 40 s apart therefore ALL read the LAST writer's list: four
#   arrays looked up their index in `scora2`'s cells and ran a scora2 cell id under
#   a loca / qwha / lyra / scora grid pin. `parse_cell_id` refused every one of
#   them (fail closed, and it is the reason nothing wrong was measured) -- but four
#   H100 allocations, and the four wall-clock measurements they existed to produce,
#   were lost.
#   ⭐ THE LESSON: an array task resolves its work at RUN time from a path chosen at
#     SUBMIT time. Anything that path points at must be IMMUTABLE from that moment
#     on. A name that is unique per submission is the whole fix.
#   ⚠ The `--run-one` body must NOT compute this (it does not enumerate the grid).
PLAN_FILE="$SWEEP_ROOT/plans/${GRID_NAME}-$(date -u +%Y%m%dT%H%M%SZ)-$$.txt"

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
    # ⛔⛔ A CELL KILLED AT THE --time WALL RECORDS NOTHING.
    #   Slurm SIGKILLs the task, so the `fail/` writer below never runs: `--status`
    #   would show done=0 failed=0 -- IDENTICAL to "never started". On a first-ever
    #   sweep against an UNMEASURED wall-clock that is the single most likely
    #   outcome, and it would look like nothing happened. A start marker makes
    #   "started and vanished" a distinguishable state.
    echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -u +%FT%TZ)" \
        > "$SWEEP_ROOT/started/$cid"
    t0=$SECONDS
    "$PY_BIN" scripts/fir_hp_run_cell.py --cell "$cid" --run-root "$SWEEP_ROOT"
    rc=$?
    el=$((SECONDS - t0))
    # ⛔ rc=5 IS "THIS CELL IS NOT IN THIS GRID" -- the plan/grid mismatch above.
    #   NOTHING TRAINED, so the start marker is a LIE and must be withdrawn: leaving
    #   it makes the cell look like it ran under a grid it does not belong to, and
    #   `--status` would report a failure for a cell whose grid never submitted it.
    #   That is what the 2026-08-28 canaries left behind on three scora2 cells.
    if [ $rc -eq 5 ]; then
        rm -f "$SWEEP_ROOT/started/$cid"
        { echo "exit=5 PLAN/GRID MISMATCH -- nothing ran. node=$(hostname)"
          echo "cell id '$cid' is not in grid '${FIR_HP_GRID:-?}'."
          echo "The array read a plan file written for a DIFFERENT grid."
          echo "⇒ re-submit this grid; the plan file is now per-submission."
        } > "$SWEEP_ROOT/fail/$cid"
        echo "### ⛔ PLAN/GRID MISMATCH after ${el}s -- start marker WITHDRAWN"
        exit $rc
    fi
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
env/bin/python scripts/fir_hp_plan.py --list > "$PLAN_FILE" || exit 1
TOTAL=$(wc -l < "$PLAN_FILE")
# ⛔ COUNT ONLY THE MARKERS THAT BELONG TO *THIS* GRID. One sweep root holds every
#   grid's cells (deliberately -- shared cells resume for free), so a bare count of
#   done/ reported `done: 160 / 140` the first time g2 was submitted: more than
#   100% complete, before a single g2 cell had run. The ARRAY SPEC was right all
#   along -- it filters by cell id -- but a status line that cannot be true is a
#   status line nobody can use.
NDONE_ALL=$(find "$SWEEP_ROOT/done" -type f 2>/dev/null | wc -l)
NDONE=$(ls "$SWEEP_ROOT/done" 2>/dev/null | grep -Fxf "$PLAN_FILE" 2>/dev/null | wc -l)
NFAIL=$(ls "$SWEEP_ROOT/fail" 2>/dev/null | grep -Fxf "$PLAN_FILE" 2>/dev/null | wc -l)
NOTHER=$((NDONE_ALL - NDONE))
if $STATUS; then
    fir_print_provenance
    echo "grid: $GRID_NAME    sweep root: $SWEEP_ROOT"
    echo "cells: $TOTAL   done: $NDONE   failed: $NFAIL   remaining: $((TOTAL - NDONE))"
    [ "$NOTHER" -gt 0 ] && echo "  (+$NOTHER done markers from other grids in this root -- not counted above)"
    if [ "$NDONE" -gt 0 ]; then
        echo "--- measured per-cell wall-clock (seconds, EVERY grid in this root) ---"
        cat "$SWEEP_ROOT"/done/* 2>/dev/null | sort -n | awk '
            {a[NR]=$1; s+=$1}
            END {printf "  n=%d  min=%d  median=%d  max=%d  mean=%.0f\n",
                        NR, a[1], a[int((NR+1)/2)], a[NR], s/NR}'
        echo "  ⚠ size --time from the MAX, not the median: --time is a hard kill."
    fi
    if [ "$NFAIL" -gt 0 ]; then
        echo "--- failures (exit != 0, or a receipt check that refused the cell) ---"
        for f in "$SWEEP_ROOT"/fail/*; do
            echo "  ⛔ $(basename "$f")"
            sed 's/^/       /' "$f" | head -26
        done
    fi
    # started, but neither done nor failed => it was KILLED (the --time wall, an
    # OOM kill, a node fault). Nothing else in this tree can see that state.
    DIED=""
    for m in "$SWEEP_ROOT"/started/*; do
        [ -e "$m" ] || continue
        c="$(basename "$m")"
        [ -f "$SWEEP_ROOT/done/$c" ] && continue
        [ -f "$SWEEP_ROOT/fail/$c" ] && continue
        DIED="$DIED $c"
    done
    if [ -n "$DIED" ]; then
        echo "--- ⛔ STARTED AND NEVER FINISHED (killed: --time wall / OOM / node fault) ---"
        for c in $DIED; do
            echo "  $c   [$(cat "$SWEEP_ROOT/started/$c")]"
            echo "     slurm log: $SWEEP_ROOT/logs/slurm_*_*.out ; cell log: $SWEEP_ROOT/logs/$c.log"
        done
        echo "  ⚠ If these hit the wall, RAISE --time. A killed cell records no exit code,"
        echo "    which is exactly why the start marker exists."
    fi
    # ⭐ COPY THE EVIDENCE INTO ./logs SO IT TRAVELS. The array's own output lives
    #   on /scratch (correct: /project is inode-bound), but /scratch does NOT get
    #   scp'd back, so a failure was invisible to anyone reading ./logs -- the only
    #   channel this project actually has. Bounded: failed + died cells only.
    # ⛔⛔ OVERRIDABLE, AND THAT IS NOT A NICETY. This path was HARDCODED, so
    #   fir_shell_gates.py -- which runs --status against a temp sweep root -- still
    #   collected into the REAL ./logs/hpsweep, and then cleaned it up afterwards.
    #   Running the test suite therefore DELETED the canary logs and CSVs a user had
    #   just scp'd there. A test must never be able to reach a real artifact
    #   directory; the fixture now points this somewhere disposable.
    COLLECT="${FIR_COLLECT_DIR:-./logs/hpsweep}"; mkdir -p "$COLLECT"
    n_c=0
    for c in $DIED $(cd "$SWEEP_ROOT/fail" 2>/dev/null && ls 2>/dev/null); do
        [ -f "$SWEEP_ROOT/logs/$c.log" ] && { cp "$SWEEP_ROOT/logs/$c.log" "$COLLECT/"; n_c=$((n_c+1)); }
    done
    for f in "$SWEEP_ROOT"/logs/slurm_*.out; do
        [ -e "$f" ] || continue
        cp "$f" "$COLLECT/" 2>/dev/null && n_c=$((n_c+1))
    done
    echo "--- collected $n_c file(s) into $COLLECT (scp ./logs as usual) ---"
    echo
    echo "read: env/bin/python scripts/fir_hp_read.py --run-root $SWEEP_ROOT"
    exit 0
fi

# ===========================================================================
# SUBMIT
# ===========================================================================
fir_print_provenance
# ⭐ --dry-run: everything the submit path computes -- the grid, the selftests, the
#   array spec -- WITHOUT sbatch. It exists so scripts/fir_shell_gates.py can
#   exercise the canary picker and the resume spec on the DEV BOX, which is the one
#   part of this file that used to be checkable only by submitting a job on fir.
# ⛔ IT IS NOT A SUBMIT-READINESS CHECK: it SKIPS the login-node environment gate,
#   because that gate demands the fir venv and would fail everywhere else. A green
#   dry run says the PLAN is right, never that the CLUSTER is.
if $DRY; then
    echo "--- ⚠ DRY RUN: the login-node environment gate is SKIPPED ---"
else
echo "--- login-node gate ---"
fir_assert_env cpu 02 || { echo "environment not sane — refusing to submit"; exit 1; }
fi
for g in fir_arms fir_plan fir_hp_plan; do
    env/bin/python "scripts/$g.py" --selftest >/dev/null || { echo "FAIL: $g selftest"; exit 1; }
done
env/bin/python scripts/fir_hp_run_cell.py --selftest >/dev/null || { echo "FAIL: run_cell selftest"; exit 1; }
echo "  instrument selftests: OK"
echo
env/bin/python scripts/fir_hp_plan.py --show
echo
# ⛔⛔ THE WRONG-CHECKOUT WARNING.  [2026-08-29]
#   fir_env derives FIR_SCRATCH_ROOT from `basename $(pwd)` -- deliberately, so two
#   experiments cannot share one venv/cache/runs.  The consequence is that the sweep
#   root FOLLOWS THE DIRECTORY YOU ARE STANDING IN, and running from a second
#   checkout does not fail: it silently starts a second sweep from zero, in a root
#   fir_hp_read.py will never look at.  A table built from either would be partial
#   and would look complete.  ⭐ Warn, do not block: a genuinely fresh root is legal.
if [ "$NDONE_ALL" -eq 0 ]; then
    OTHER=""
    # ⛔ SIBLINGS OF *THIS* SCRATCH ROOT, not $SCRATCH/$USER rebuilt from scratch.
    #   The first version expanded `${SCRATCH:-/scratch/$USER}` unconditionally, and
    #   `$USER` is not set in every environment this script must survive (`set -u`
    #   made it fatal) -- so a WARNING about being in the wrong place took out every
    #   local gate. fir_env has already resolved FIR_SCRATCH_ROOT; its parent is the
    #   only base this check needs, and it exists wherever the script runs at all.
    for d in "$(dirname "$FIR_SCRATCH_ROOT")"/*/runs/hpsweep; do
        [ -d "$d/done" ] || continue
        [ "$d" = "$SWEEP_ROOT" ] && continue
        n=$(find "$d/done" -type f 2>/dev/null | wc -l)
        [ "$n" -gt 0 ] && OTHER="$OTHER
     $n done markers in $d"
    done
    if [ -n "$OTHER" ]; then
        echo "  ⛔⛔ THIS SWEEP ROOT IS EMPTY, BUT ANOTHER ONE IS NOT:$OTHER"
        echo "     The root is derived from \`basename \$(pwd)\` = '$FIR_REPO_NAME'."
        echo "     You are probably in the WRONG CHECKOUT. Submitting from here starts a"
        echo "     SECOND sweep from zero; the reader only ever looks at one root."
        echo "     ⇒ cd to the checkout that owns those markers, or set FIR_SCRATCH_ROOT."
    fi
fi
echo "  sweep root : $SWEEP_ROOT   (derived from basename \$(pwd) = '$FIR_REPO_NAME')"
echo "  done       : $NDONE / $TOTAL   (failed so far: $NFAIL)"
[ "$NOTHER" -gt 0 ] && echo "               +$NOTHER cells done under ANOTHER grid in this root"

# ⭐ THE ARRAY INDEXES THE PLAN FILE, a per-submission snapshot of the planner's
#   output. The planner's order is deterministic and selftested, so an index means
#   the same cell on every submission -- and because the snapshot is unique to this
#   submit, a LATER submit of another grid cannot move it under a queued array.
if [ "$P_CANARY" -gt 0 ]; then
    LAST=$((P_CANARY - 1))
    # ⚠ the canary takes ONE CELL PER ARM, not the first N lines: the first 80 are
    #   all fftm, so a naive head -2 would never exercise the stock-PEFT path --
    #   and that is the arm whose build differs (modules_to_save, no module-count
    #   log). A canary that cannot see one of the two paths is not a canary.
    # ⛔ THE PICKER LIVES IN THE PLANNER (H.canary_indices), not here. Two earlier
    #   versions of it were hardcoded against a grid -- first literal knob values,
    #   then `max(SCALINGS)` -- and both broke or degraded the day a new grid
    #   landed. The shell asks; it does not decide.
    IDX=$(env/bin/python -c "
import sys; sys.path.insert(0, 'scripts')
import fir_hp_plan as H
print(','.join(str(i) for i in H.canary_indices()))") || exit 1
    ARRAY_SPEC="$IDX"
    echo "  CANARY: array=$ARRAY_SPEC  (one cell per arm, CENTRAL on every axis)"
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

# ⛔⛔ RENDER THE ARRAY BODY BEFORE DECIDING WHETHER TO SUBMIT IT.
#   It used to be built INSIDE the `sbatch <<SB` heredoc, which --dry-run never
#   reached -- so the single line that decides WHICH CELL EACH TASK RUNS was the
#   one line no local gate could see. That is the 2026-08-28 defect's own lesson
#   turned on the check: A CHECK MUST RUN WHAT THE JOB RUNS. Rendering it here
#   means --dry-run inspects the exact text sbatch would receive.
BODY_FILE="${PLAN_FILE%.txt}.sbatch"
cat > "$BODY_FILE" <<SB
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
# ⚠ PIN THE GRID EXPLICITLY. The plan file was generated by the submitter under
#   this grid; if the array task resolved a DIFFERENT one, parse_cell_id refuses the
#   id (fail closed) -- but relying on sbatch's --export default to carry an env var
#   that decides WHICH EXPERIMENT RUNS is not something to leave implicit.
# ⭐ AND THE PLAN PATH IS BAKED IN, unique to this submission, so a LATER submit of
#   a different grid cannot change what this array reads. That is exactly how the
#   2026-08-28 canaries lost four cells.
export FIR_HP_GRID="$GRID_NAME"
cid=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$PLAN_FILE")
[ -n "\$cid" ] || { echo "FAIL: no cell at index \$SLURM_ARRAY_TASK_ID"; exit 1; }
bash sbatch/fir/04_hp_sweep.sh --run-one "\$cid"
SB

if $DRY; then
    echo "DRY RUN: would submit array=$ARRAY_SPEC  (grid $GRID_NAME, time $P_TIME)"
    echo "         plan  : $PLAN_FILE"
    echo "         script: $BODY_FILE"
    echo "--- the array body sbatch would receive ---"
    sed 's/^/    | /' "$BODY_FILE"
    echo "         nothing was submitted."
    exit 0
fi

jid=$(sbatch --parsable < "$BODY_FILE")
echo "submitted array job $jid"
echo "script:  $BODY_FILE"
echo "watch:   squeue -j $jid"
echo "status:  bash sbatch/fir/04_hp_sweep.sh --status"
echo
if [ "$P_CANARY" -gt 0 ]; then
    echo "⛔ NEXT: when the canary finishes, run --status. It prints the MEASURED"
    echo "   per-cell seconds. Size --time from the MAX plus headroom, then submit"
    echo "   the full array. Do NOT submit the grid against an unmeasured wall."
    echo "   ⛔ A MEASUREMENT FROM ANOTHER ARM IS NOT A MEASUREMENT — and neither is"
    echo "      a PREDICTION from one. I predicted a WaveFT cell at ~1.7x a"
    echo "      FourierFT one, extrapolating [R.307]'s 6.7x per-module latency"
    echo "      ratio; [measured, 96 cells] it is 1.03x. ⭐ A per-module latency"
    echo "      ratio measured on ANOTHER backbone does not give a per-cell"
    echo "      wall-clock on this one — the adapter SHARE it multiplies is"
    echo "      backbone-dependent. loca and qwha are structurally heavier and"
    echo "      UNMEASURED here; size each grid from its OWN canary."
    echo "   ⚠ And a 2-cell canary sizes a WALL, it does not estimate a"
    echo "      DISTRIBUTION: the WaveFT canary said 1.22x, 96 cells said 1.03x."
fi
