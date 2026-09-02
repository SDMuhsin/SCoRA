#!/bin/bash
# ============================================================================
# 06_baseline.sh — THE FP16 FULL-FINE-TUNING BASELINE on gemma-2b.
# hyperparameters (llmdocs/GEMMA_HP_PROXY.md).  ONE ARRAY PER TASK.
# ============================================================================
#   FIR_BASE_TASK=rte bash sbatch/fir/06_baseline.sh --canary 9   # ⭐ ALWAYS FIRST
#   FIR_BASE_TASK=rte bash sbatch/fir/06_baseline.sh --time HH:MM:SS --concurrent 16
#   FIR_BASE_TASK=rte bash sbatch/fir/06_baseline.sh --status
#   FIR_BASE_TASK=all bash sbatch/fir/06_baseline.sh --status      # every task
#   bash sbatch/fir/06_baseline.sh --dry-run                        # plan, submit nothing
#
#   9 arms x 6 tasks x 5 seeds = 270 cells.  ONE cell = one (arm, task, seed), so
#   the SEED axis and the TASK axis are both parallel, as asked.
#
# ⛔⛔ WHY THIS IS A SECOND FILE AND NOT A FLAG ON 04.  04's header says a stage
#   must not be forked, and that stands -- it was written against forking 04 PER
#   ARM, which would have made five copies of one job shape.  This is a DIFFERENT
#   job shape: one array PER TASK (because --time is per-array and the per-task
#   wall-clock spans ~6x), a different planner, a different run root, and a
#   canary that must cover two axes at once.  ⭐ THE PRICE IS PAID THE ONLY WAY IT
#   CAN BE: scripts/fir_shell_gates.py runs the SAME battery against this file as
#   against 04 -- canary picker, resume spec, plan snapshot, and the end-to-end
#   "every array index resolves to a cell of this stage" check.
#   ⭐ And the part that MUST NOT be duplicated is not duplicated: the cell body
#   calls `fir_hp_run_cell.py --planner baseline`, so the RECEIPT CHECK -- the one
#   thing standing between "the adapter attached to nothing" and a plausible
#   number -- exists once.
#
# ⛔ ONE ARRAY PER TASK IS NOT A STYLE CHOICE.  --time is a HARD kill and it is set
#   per array; [predicted] an RTE cell is ~900 s and an SST-2 cell ~5,900 s.  A
#   single wall for both either kills every SST-2 cell or wastes 6x the queue
#   priority on every RTE one.
#
# ⛔⛔ IT IS BUILT ON THE ASSUMPTION THAT CELLS WILL FAIL, exactly as 04 is:
#   independent cells, `done/<id>` only after exit 0, `fail/<id>` with the exit
#   code and log tail, `started/<id>` so a cell KILLED AT THE WALL (which records
#   nothing and otherwise looks identical to "never started") is visible.
#   ⛔ Re-running this script IS the recovery procedure.  It is idempotent.
#
# ⭐ WHY THE CANARY IS 9 CELLS AND COVERS BOTH AXES.  Two different things are
#   unproven and they fail in different places: an ARM's flags at a NEW task's step
#   count, and a TASK's data/metric/collapse path on this backbone (every gemma
#   cell to date is MRPC).  The planner picks a covering set -- every arm exactly
#   once, every task at least once -- so one canary per task measures that task's
#   wall AND smokes an arm.  `H.canary_indices()` decides; this file only asks.
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
fir_log_to fir_baseline "$@"

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

GRID_NAME="${FIR_BASE_TASK:-all}"
export FIR_BASE_TASK="$GRID_NAME"
# ⛔ A SEPARATE ROOT FROM THE SEARCH. These cells share the (task, arm, targets,
#   seed) shape with the search's, and only the `-final-epN` in the id keeps them
#   apart. Two roots means a mistake in that id can never overwrite 68 GPU-h of
#   search CSVs. ⚠ ONE root for all six TASKS, though -- a cell id names its task,
#   so `--status` can be asked about one task or all of them from the same place.
# ⛔⛔ `all` IS A READING VIEW, NOT A RUN TARGET -- and the refusal is not pedantry.
#   --time is set PER ARRAY and the per-task wall-clock spans ~6x, so one array over
#   all six tasks either kills every SST-2 cell at an RTE-sized wall or wastes 6x the
#   queue priority on every RTE one. `--status` over `all` is exactly right and is
#   allowed; anything that SUBMITS is refused, including --dry-run, which exists to
#   exercise the submit path. Same shape as the `wave` union view in 04.
if [ "$GRID_NAME" = "all" ] && [ -z "$LOCAL_ONE" ] && ! $STATUS; then
    echo "⛔ FIR_BASE_TASK=all is a READING VIEW, not a run target."
    echo "   --time is per-array and the per-task cost spans ~6x, so the six tasks"
    echo "   are six submissions. Pick one:"
    echo "     FIR_BASE_TASK=rte bash sbatch/fir/06_baseline.sh --canary 9"
    echo "   (--status over 'all' IS supported and is the way to see the whole table.)"
    exit 1
fi
SWEEP_ROOT="$FIR_RUN_ROOT/baseline"
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
PLAN_FILE="$SWEEP_ROOT/plans/baseline-${GRID_NAME}-$(date -u +%Y%m%dT%H%M%SZ)-$$.txt"

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
    "$PY_BIN" scripts/fir_hp_run_cell.py --planner baseline --cell "$cid" --run-root "$SWEEP_ROOT"
    rc=$?
    el=$((SECONDS - t0))
    # ⛔ rc=5 IS "THIS CELL IS NOT IN THIS PLAN" -- the plan/selector mismatch above.
    #   NOTHING TRAINED, so the start marker is a LIE and must be withdrawn: leaving
    #   it makes the cell look like it ran under a grid it does not belong to, and
    #   `--status` would report a failure for a cell whose grid never submitted it.
    #   That is what the 2026-08-28 canaries left behind on three scora2 cells.
    if [ $rc -eq 5 ]; then
        rm -f "$SWEEP_ROOT/started/$cid"
        { echo "exit=5 PLAN/GRID MISMATCH -- nothing ran. node=$(hostname)"
          echo "cell id '$cid' is not in task view '${FIR_BASE_TASK:-?}'."
          echo "The array read a plan file written for a DIFFERENT task."
          echo "⇒ re-submit this task; the plan file is per-submission."
        } > "$SWEEP_ROOT/fail/$cid"
        echo "### ⛔ PLAN/TASK MISMATCH after ${el}s -- start marker WITHDRAWN"
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
STAGE="${FIR_BASE_STAGE:-search}"
# ⛔ THE STAGE IS PART OF THE PLAN FILE'S IDENTITY. `final` emits ZERO cells
#   until fir_baseline_plan.WINNERS is filled in from the search, so a
#   premature `final` submit produces an EMPTY plan and is refused below
#   rather than silently submitting the whole ladder at five seeds.
env/bin/python scripts/fir_baseline_plan.py --list --stage "$STAGE" > "$PLAN_FILE" || exit 1
TOTAL=$(wc -l < "$PLAN_FILE")
# ⛔⛔ AN EMPTY PLAN IS A REFUSAL, NOT A NO-OP. `--stage final` emits zero cells
#   until fir_baseline_plan.WINNERS is filled in from the search. Without this guard
#   the array spec (computed separately, below) still produced 0-11 and sbatch would
#   have allocated 12 H100s to read an EMPTY plan file -- the 2026-08-28 failure
#   shape exactly: an array index resolving out of a list nobody checked.
if [ "$TOTAL" -eq 0 ]; then
    echo "⛔ REFUSING: the plan for grid '$GRID_NAME' stage '$STAGE' is EMPTY."
    if [ "$STAGE" = "final" ]; then
        echo "   The final stage runs the SEARCH's winner at the remaining seeds, and"
        echo "   fir_baseline_plan.WINNERS is still empty. Run and READ the search first,"
        echo "   then write the measured (lr, batch) per task into WINNERS."
    fi
    exit 1
fi
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
    echo "task view: $GRID_NAME    run root: $SWEEP_ROOT"
    echo "cells: $TOTAL   done: $NDONE   failed: $NFAIL   remaining: $((TOTAL - NDONE))"
    [ "$NOTHER" -gt 0 ] && echo "  (+$NOTHER done markers from OTHER TASKS in this root -- not counted above)"
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
    COLLECT="${FIR_COLLECT_DIR:-./logs/baseline}"; mkdir -p "$COLLECT"
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
    echo "read: (stage 06 has no dedicated reader yet -- csv/ holds one file per cell)"
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
# ⛔⛔ PRINT THE FAILURE, DO NOT JUST NAME IT. [2026-08-30] this line swallowed the
#   selftest output with `>/dev/null`, so six refused canaries said only
#   "FAIL: <module> selftest" -- the ONE line that could not say WHICH check
#   failed or why. Two round trips were spent guessing at a message the cluster
#   already had. ⭐ REPORT THE RECEIPT, NOT THE FLAG; PRINT THE LOCATION OF AN
#   ERROR, NOT JUST ITS LAST LINE (CONTEXT §4.4.3).
for g in fir_arms fir_plan fir_hp_plan fir_baseline_plan fir_hp_run_cell; do
    if ! _out=$(env/bin/python "scripts/$g.py" --selftest 2>&1); then
        echo "FAIL: $g selftest -- its own output follows:"
        printf '%s\n' "$_out" | grep -E "⛔|Error|error|Traceback|selftest:" | tail -25 \
            | sed 's/^/    | /'
        echo "    reproduce: env/bin/python scripts/$g.py --selftest"
        exit 1
    fi
done
env/bin/python scripts/fir_hp_run_cell.py --selftest >/dev/null || { echo "FAIL: run_cell selftest"; exit 1; }
echo "  instrument selftests: OK"
echo
env/bin/python scripts/fir_baseline_plan.py --show
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
    for d in "$(dirname "$FIR_SCRATCH_ROOT")"/*/runs/baseline; do
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
[ "$NOTHER" -gt 0 ] && echo "               +$NOTHER cells done under ANOTHER TASK in this root"

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
import fir_baseline_plan as H
print(','.join(str(i) for i in H.canary_indices(stage=__import__('os').environ.get('FIR_BASE_STAGE','search'))))") || exit 1
    ARRAY_SPEC="$IDX"
    echo "  CANARY: array=$ARRAY_SPEC  (the covering set for this task: every arm"
    echo "          exactly once ACROSS the six tasks, every task at least once)"
else
    # ⛔ RESUME MUST NOT RE-QUEUE FINISHED CELLS. The task body skips a done cell in
    #   seconds -- but only AFTER Slurm has allocated it a whole H100. On a 160-cell
    #   grid where 150 are done that is 150 pointless GPU allocations. Submit only
    #   the indices that have no `done` marker, collapsed into ranges.
    # ⛔⛔ THE ARRAY SPEC IS DERIVED FROM THE PLAN FILE, NOT FROM A SECOND
    #   ENUMERATION OF THE PLANNER. Those are two sources of truth for "which cell is
    #   index N", and [2026-09-02, caught by --dry-run] they DISAGREED: `--stage final`
    #   wrote a 0-line plan while a bare `H.cells()` re-enumerated the SEARCH stage and
    #   returned 12, so the spec said 0-11 against an empty file. The plan file is what
    #   the array body actually seds into, so the plan file is the authority.
    ARRAY_SPEC=$(env/bin/python - "$SWEEP_ROOT" "$PLAN_FILE" <<'PY'
import os, sys
root, plan = sys.argv[1], sys.argv[2]
ids = [l.strip() for l in open(plan) if l.strip()]
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
        echo "  read: rsync the root back, then FIR_BASE_TASK=<t> bash sbatch/fir/06_baseline.sh --status"
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
#SBATCH --job-name=lrs_base_${GRID_NAME}
#SBATCH --account=$FIR_ACCOUNT_GPU
#SBATCH --gpus=$FIR_GPU_FULL
#SBATCH --cpus-per-task=8
#SBATCH --mem=$FIR_GPU_MEM
#SBATCH --time=$P_TIME
#SBATCH --array=$ARRAY_SPEC
#SBATCH --output=$SWEEP_ROOT/logs/slurm_%A_%a.out
cd "\$SLURM_SUBMIT_DIR" || exit 1
set -uo pipefail
# ⚠ PIN THE TASK EXPLICITLY. The plan file was generated by the submitter under
#   this task view; if the array task resolved a DIFFERENT one, parse_cell_id
#   refuses the id (fail closed) -- but relying on sbatch's --export default to
#   carry an env var that decides WHICH EXPERIMENT RUNS is not left implicit.
# ⭐ AND THE PLAN PATH IS BAKED IN, unique to this submission, so a LATER submit of
#   a different grid cannot change what this array reads. That is exactly how the
#   2026-08-28 canaries lost four cells.
export FIR_BASE_TASK="$GRID_NAME"
export FIR_BASE_STAGE="$STAGE"
cid=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$PLAN_FILE")
[ -n "\$cid" ] || { echo "FAIL: no cell at index \$SLURM_ARRAY_TASK_ID"; exit 1; }
bash sbatch/fir/06_baseline.sh --run-one "\$cid"
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
echo "status:  FIR_BASE_TASK=$GRID_NAME bash sbatch/fir/06_baseline.sh --status"
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
