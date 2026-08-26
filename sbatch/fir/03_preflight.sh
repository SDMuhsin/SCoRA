#!/bin/bash
# ============================================================================
# 03_preflight.sh — ONE 1-GPU job that proves the environment before any spend.
# ============================================================================
#   bash sbatch/fir/03_preflight.sh [--targets q_o] [--port-mode derived] [--local]
#
#   --local  run the body HERE instead of submitting (use inside salloc)
#
# WHAT IT PROVES, IN ORDER, CHEAPEST FIRST:
#   A. the env gate, on a GPU node
#   B. ⛔⛔ THE BIT-IDENTITY GATES, UNDER FIR'S peft.  This is the stage that
#      carries the cost of the fir-native pin decision. See below.
#   C. the port table reproduces on this stack (a cross-stack check on the
#      adapter layers themselves)
#   D. ALL NINE ARMS actually train, on the real backbone, for a handful of steps
#      — with RECEIPTS, not flags.
#
# ⛔⛔ WHY B IS NOT A FORMALITY.
#   `src/qwha_adapter.py:14`: every FourierFT number in this repo is gated as
#   BIT-IDENTICAL to the INSTALLED `peft.tuners.fourierft`.  The dev box runs peft
#   0.13.2; fir runs 0.18.1 (user decision, 2026-08-25).  If PEFT moved that layer,
#   the fir `fftstock` arm is a DIFFERENT comparator from every dev-box number, and
#   the whole point of `fftstock` (the stock-PEFT control) changes meaning.
#   ⇒ a failure here is LOUD and STOPS the chain.  It is a finding, not a nuisance.
#
# ⭐ D IS THE CHECK THAT CANNOT BE REPLACED BY A FLAG.  A RoBERTa module name on a
#   decoder matches NOTHING: the adapter attaches to zero modules, the classifier
#   head trains alone, the run succeeds, and the row looks entirely plausible.
#   So D parses each arm's OWN log for the module count and the trainable-parameter
#   count and ENUMERATES THEM ACROSS EVERY ARM — [FIR_SETUP G1]: identical numbers
#   across unrelated methods mean a shared setting is missing, and a number that
#   differs between arms at a matched budget means one of them did not attach.
# ============================================================================
set -uo pipefail
FIR_SELF="$(readlink -f "$0")"
cd "$(dirname "$FIR_SELF")/../.." || exit 1
source sbatch/fir/fir_env.sh
fir_log_to fir_preflight "$@"

P_TARGETS="${P_TARGETS:-q_o}"
P_PORT_MODE="${P_PORT_MODE:-derived}"
P_TASK="${P_TASK:-rte}"
P_STEPS="${P_STEPS:-8}"          # training samples per arm = P_STEPS * batch
LOCAL=false
while [ $# -gt 0 ]; do
    case "$1" in
        --targets) P_TARGETS="$2"; shift 2 ;;
        --port-mode) P_PORT_MODE="$2"; shift 2 ;;
        --task) P_TASK="$2"; shift 2 ;;
        --local) LOCAL=true; shift ;;
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done

# ---- fail on the LOGIN node before asking for a GPU -------------------------
if ! $LOCAL && [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "--- login-node gate (fail here, not 40 minutes into an allocation) ---"
    fir_assert_env cpu 02 || { echo "environment not sane — refusing to submit"; exit 1; }
    env/bin/python scripts/fir_arms.py --selftest   >/dev/null || { echo "FAIL: fir_arms selftest"; exit 1; }
    env/bin/python scripts/fir_plan.py --selftest   >/dev/null || { echo "FAIL: fir_plan selftest"; exit 1; }
    echo "  instrument selftests: OK"
    # ⭐ --dry-run must PRINT AND VALIDATE what it is about to run.  An inherited
    #   exported variable silently surviving a switch is an INVOCATION hazard that
    #   a dry-run whose whole job is to catch it once showed without comment.
    echo
    echo "--- what the preflight will run ---"
    echo "  model      : $FIR_MODEL"
    echo "  targets    : $P_TARGETS      port-mode: $P_PORT_MODE      task: $P_TASK"
    echo "  arms       : $(env/bin/python -c "import sys;sys.path.insert(0,'scripts');import fir_arms as F;print(' '.join(F.ARM_ORDER))")"
    echo "  account    : $FIR_ACCOUNT_GPU     gres: $FIR_GPU_FULL     mem: $FIR_GPU_MEM"
    echo
    jid=$(sbatch --parsable <<SB
#!/bin/bash
#SBATCH --job-name=lrs_preflight
#SBATCH --account=$FIR_ACCOUNT_GPU
#SBATCH --gpus=$FIR_GPU_FULL
#SBATCH --cpus-per-task=8
#SBATCH --mem=$FIR_GPU_MEM
#SBATCH --time=2:00:00
#SBATCH --output=./logs/preflight_%j.out
cd "\$SLURM_SUBMIT_DIR" || exit 1
# ⚠⚠ THE SUBMITTER'S SHELL OPTIONS DO NOT REACH THIS NEW SHELL. Without this line
#    a Python traceback and a clean run are indistinguishable through a pipe.
set -uo pipefail
export P_TARGETS="$P_TARGETS" P_PORT_MODE="$P_PORT_MODE" P_TASK="$P_TASK" P_STEPS="$P_STEPS"
bash sbatch/fir/03_preflight.sh --local --targets "$P_TARGETS" --port-mode "$P_PORT_MODE" --task "$P_TASK"
SB
)
    echo "submitted preflight job $jid"
    echo "watch:  squeue -j $jid ; tail -f logs/preflight_$jid.out"
    exit 0
fi

# =========================== THE JOB BODY ===================================
echo "############ PREFLIGHT — $(date -u +%FT%TZ) ############"
# ⚠ PRINT THE NODE. Without it an intermittent per-node failure cannot be
#   attributed, and one bad node ate 17 cells on the sibling project before the
#   pattern became visible.
echo "node=$(hostname)  JOB=${SLURM_JOB_ID:-none}  NODELIST=${SLURM_JOB_NODELIST:-none}"
fir_load_modules_gpu || exit 1
# shellcheck disable=SC1091
source "$FIR_VENV/bin/activate" || exit 1
fir_export_offline
nvidia-smi --query-gpu=name,memory.total,compute_mode --format=csv 2>&1 | head -3

rc=0

echo; echo "=== A. environment gate (GPU) ==="
fir_assert_env gpu || rc=1

# ⚠⚠ CALL THE VENV INTERPRETER EXPLICITLY, NEVER BARE `python`.
# A venv is not relocatable: bin/activate hardcodes an absolute VIRTUAL_ENV, so
# after the venv directory is moved, `activate` succeeds, puts a NONEXISTENT dir
# on PATH, and bare `python` is the MODULE python -- no torch, no peft. On fir
# 2026-08-26 that made all four verifiers die on ModuleNotFoundError while the
# env gate PASSED, because the gate calls "$FIR_VENV/bin/python" explicitly.
# FIR_SETUP E6 says the same thing for the login-node report jobs.
PY_BIN="$FIR_VENV/bin/python"
[ -x "$PY_BIN" ] || { echo "FAIL: no interpreter at $PY_BIN"; exit 1; }
echo "interpreter: $PY_BIN  ($("$PY_BIN" -V 2>&1))"
echo; echo "=== B. BIT-IDENTITY GATES under peft $("$PY_BIN" -c 'import peft;print(peft.__version__)') ==="
echo "    ⛔ These decide whether the fir comparator IS the dev-box comparator."
# ⛔⛔ RECORD THE OUTCOME, NEVER PREDICT IT (FIR_SETUP G5).
#    The first version of this block treated ANY non-zero exit as "the comparator
#    moved under peft 0.18.1" -- the failure it was written expecting. On fir
#    2026-08-25 that produced a FALSE headline: verify_loca_adapter died on a
#    `SyntaxError` inside its own generated driver (so it compared NOTHING) and
#    verify_fourierft_fast died on `CUDA error: invalid device ordinal` (an
#    environment fault), yet both were reported as bit-identity failures.
#    "The verifier crashed" and "the verifier found a difference" are OPPOSITE
#    meanings, and a check that cannot tell them apart is worse than no check.
#    ⇒ three outcomes, read out of each verifier's OWN output:
#       MISMATCH — it ran and the numbers differ   => the pin decision's real bill
#       ERROR    — it could not run at all         => says NOTHING about numerics
#       OK       — it ran and matched
b_mismatch=""; b_error=""
for v in verify_merged_fourierft verify_qwha_adapter verify_loca_adapter verify_fourierft_fast; do
    echo "--- $v ---"
    # ⚠ --device EXPLICITLY. verify_fourierft_fast defaulted to `cuda:1`, a dev-box
    #   artifact; a Slurm job with --gpus=h100:1 has only cuda:0 and it raised
    #   "invalid device ordinal". The default is resolved now, but pass it anyway:
    #   gate on what we hand the tool, not on its default (FIR_SETUP Law 2).
    vdev=""
    case "$v" in verify_fourierft_fast|verify_coset_adapter) vdev="--device cuda:0" ;; esac
    vout="$("$PY_BIN" "src/$v.py" $vdev 2>&1)"; vrc=$?
    echo "$vout" | tail -25
    if [ $vrc -eq 0 ]; then
        echo "  $v: OK"
        continue
    fi
    # An interpreter-level fault means the harness never got to compare anything.
    if echo "$vout" | grep -qE "SyntaxError|ModuleNotFoundError|ImportError|CUDA error|AcceleratorError|FileNotFoundError|No such file|Permission denied"; then
        echo "  ⚠ $v: ERROR — the verifier could not RUN. This is NOT a numerical finding."
        b_error="$b_error $v"
    else
        echo "  ⛔ $v: MISMATCH — it ran and the numbers differ."
        b_mismatch="$b_mismatch $v"
    fi
done
if [ -n "$b_error" ]; then
    echo
    echo "⚠⚠ VERIFIER(S) FAILED TO RUN:$b_error"
    echo "   These say NOTHING about whether the comparator moved. The bit-identity"
    echo "   question is UNANSWERED for them until they execute. Fix the harness, then"
    echo "   re-run -- do not read this as either a pass or a fail."
    rc=1
fi
if [ -n "$b_mismatch" ]; then
    echo
    echo "⛔⛔ BIT-IDENTITY MISMATCH:$b_mismatch"
    echo "   These RAN and disagreed. The FourierFT/LoCA/QWHA comparator on fir is NOT"
    echo "   the one every dev-box number was measured against. This is a RESULT, not a"
    echo "   bug to route around: report it, and do not quote a fir table alongside a"
    echo "   dev-box table until it is understood. (peft $FIR_PIN_PEFT vs dev box 0.13.2.)"
    rc=1
fi

echo; echo "=== C. port table reproduces on this stack ==="
# ⭐ The adapters are pure torch, so a difference here means torch/transformers/peft
#   moved a layer. Tolerance is float32-reduction-aware and its firing is gated.
"$PY_BIN" scripts/fir_backbone_port.py --model "$FIR_MODEL" --verify-emit || rc=1

echo; echo "=== D. ALL NINE ARMS TRAIN, WITH RECEIPTS ==="
"$PY_BIN" scripts/fir_preflight_arms.py \
    --targets "$P_TARGETS" --port-mode "$P_PORT_MODE" --task "$P_TASK" \
    --steps "$P_STEPS" --run-root "$FIR_RUN_ROOT/preflight" || rc=1

echo
if [ $rc -eq 0 ]; then
    echo "############ PREFLIGHT OK ############"
    echo "next: bash sbatch/fir/04_pilot_cell.sh --targets $P_TARGETS --port-mode $P_PORT_MODE"
else
    echo "############ PREFLIGHT FAILED ############"
    echo "Read the FAIL lines above. Nothing larger should be submitted."
fi
exit $rc
