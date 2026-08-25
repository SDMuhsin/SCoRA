#!/bin/bash
# ============================================================================
# 00c_probe_deps.sh — WILL OUR EXACT PINS RESOLVE ON FIR? LOGIN NODE. READ-ONLY.
# ============================================================================
#   bash sbatch/fir/00c_probe_deps.sh
#
# ⛔ RUN THIS BEFORE 01_setup_venv.sh. It downloads NOTHING (`--dry-run`), takes
#    a couple of minutes, and answers the one question `00` could not.
#
# WHY `00` §6 IS NOT ENOUGH
# ------------------------
# `avail_wheels` reports only the wheelhouse **DEFAULTS**. On fir [measured
# 2026-08-25] that is torch 2.13.0 / transformers 5.14.1 / peft 0.19.1 /
# datasets 5.0.0 — none of which is what we pin. The OLDER pins are generally in
# the wheelhouse too, each resolving as `<version>+computecanada`, but
# `avail_wheels` does not show them, so §6 looks alarming and proves nothing.
#
# ⚠ AND `pip index versions` IS UNRELIABLE — it is experimental and misreports
#   flat PEP-503 indexes; on the sibling project it reported "No matching
#   distribution found for torch" for a package that installs fine.
#   ⇒ ONLY `pip install --dry-run` may be used to conclude anything.
#
# ⛔ NO `--index-url` ANYWHERE. Alliance's pip config adds `find-links` to the
#   CVMFS wheelhouse with `prefer-binary = true`, so the local wheel wins
#   regardless; on the sibling project `--index-url ...cu128` and even
#   `--isolated` both still resolved to `+computecanada`.
# ============================================================================
set -uo pipefail
FIR_SELF="$(readlink -f "$0")"
cd "$(dirname "$FIR_SELF")/../.." || exit 1
source sbatch/fir/fir_env.sh
fir_log_to fir_probe_deps "$@"

echo "############ DEP PROBE — $(date -u +%FT%TZ) ############"
fir_load_modules_gpu || { echo "FAIL: module load"; exit 1; }
echo "python : $(python -V 2>&1) at $(command -v python)"
echo "pip    : $(python -m pip -V 2>&1)"
echo
echo "PIP_CONFIG_FILE : ${PIP_CONFIG_FILE:-<unset>}"
if [ -n "${PIP_CONFIG_FILE:-}" ] && [ -f "${PIP_CONFIG_FILE}" ]; then
    echo "--- contents (this is what silently redirects every install) ---"
    sed 's/^/    /' "${PIP_CONFIG_FILE}"
fi

# ⚠ Resolve in a THROWAWAY venv rather than the module python, so the probe cannot
#   leave anything behind and cannot be polluted by an earlier attempt.
#   ⛔ IT IS CREATED --system-site-packages ON PURPOSE, AND THE EARLIER COMMENT HERE
#      CLAIMED THE OPPOSITE. 01_setup_venv.sh's venv is --system-site-packages
#      (REQUIRED: numpy comes from the scipy-stack module and `import datasets`
#      dies without it), so a probe without it would resolve a DIFFERENT
#      environment from the one the job runs -- which is the whole failure class
#      FIR_SETUP Law 1 is about. Fidelity beats isolation here.
TMPV="${FIR_SCRATCH_ROOT}/.depprobe_venv"
echo
echo "--- throwaway resolver venv at $TMPV ---"
mkdir -p "$(dirname "$TMPV")"
rm -rf "$TMPV" "$CONS"
python -m venv --system-site-packages "$TMPV" >/dev/null 2>&1 || {
    echo "FAIL: could not create the probe venv"; exit 1; }
PY="$TMPV/bin/python"
"$PY" -m pip install -q --upgrade pip >/dev/null 2>&1
echo "  $("$PY" -V 2>&1)"

rc=0
probe() {   # probe <label> <pip-spec...>
    local label="$1"; shift
    echo
    echo "=== $label ==="
    echo "    pip install --dry-run --no-deps $*"
    local out
    out=$("$PY" -m pip install --dry-run --no-deps "$@" 2>&1); local prc=$?
    # the line that matters names the exact artifact pip WOULD install
    echo "$out" | grep -iE "Would install|Processing|Downloading|Using cached|ERROR|No matching" \
        | sed 's/^/    /' | head -8
    if [ $prc -ne 0 ]; then
        echo "    ⛔ DOES NOT RESOLVE"
        rc=1
    else
        echo "    ✅ resolves"
    fi
}

# --- the pins the experiment is defined by. A failure here is a STOP, not a
#     prompt to substitute a neighbour: a different stack is a different experiment.
probe "torch  == $FIR_PIN_TORCH"         "torch==$FIR_PIN_TORCH"
probe "transformers == $FIR_PIN_TRANSFORMERS" "transformers==$FIR_PIN_TRANSFORMERS"
probe "datasets == $FIR_PIN_DATASETS"    "datasets==$FIR_PIN_DATASETS"
probe "peft == $FIR_PIN_PEFT"            "peft==$FIR_PIN_PEFT"
probe "accelerate == $FIR_PIN_ACCELERATE" "accelerate==$FIR_PIN_ACCELERATE"
probe "evaluate == $FIR_PIN_EVALUATE"    "evaluate==$FIR_PIN_EVALUATE"

# --- ⚠ triton: NOT transitive on every torch build. Same version string,
#     different dependency graph — no version check can see it.
probe "triton (explicit; NOT transitive on +computecanada)" "triton"

# --- ⚠ the two the wheelhouse showed EMPTY in 00 §6. Both are module-scope
#     imports in train_glue.py, so a miss kills EVERY arm, not just their own.
#     `adapters` requires transformers~=4.51.3, which REINFORCES our pin;
#     `adapter-transformers` is a deprecated 4.0.0 STUB that installs cleanly and
#     provides no `adapters` module — do not let anyone "fix" a failure with it.
probe "adapters (wheelhouse showed EMPTY -> must come from PyPI)" "adapters"
probe "galore-torch (wheelhouse showed EMPTY -> must come from PyPI)" "galore-torch"
probe "lion-pytorch" "lion-pytorch"
probe "sentencepiece (gemma tokenizer; NOT in requirements.txt)" "sentencepiece"

# --- and the whole requirements file, resolved as a SET under our constraints.
#     ⭐ This is the real question: the pins above can each resolve alone and
#     still conflict together. Resolving the set is what 01 will actually do.
CONS="${FIR_SCRATCH_ROOT}/.depprobe_constraints.txt"
{
    echo "torch==$FIR_PIN_TORCH"
    echo "transformers==$FIR_PIN_TRANSFORMERS"
    echo "datasets==$FIR_PIN_DATASETS"
    echo "peft==$FIR_PIN_PEFT"
    echo "accelerate==$FIR_PIN_ACCELERATE"
    echo "evaluate==$FIR_PIN_EVALUATE"
} > "$CONS"
echo
echo "=== requirements.txt AS A SET, under the constraints file ==="
echo "--- constraints ---"; sed 's/^/    /' "$CONS"
echo "    pip install --dry-run -r requirements.txt -c $CONS"
out=$("$PY" -m pip install --dry-run -r requirements.txt -c "$CONS" 2>&1); prc=$?
echo "$out" | grep -iE "Would install|ERROR|No matching|conflict|incompatible" \
    | sed 's/^/    /' | head -20
if [ $prc -ne 0 ]; then
    echo "    ⛔ THE SET DOES NOT RESOLVE — this is the finding. Report it; do not"
    echo "       relax a pin to make it pass. A different stack is a different experiment."
    rc=1
else
    echo "    ✅ the whole set resolves"
    # ⭐ name the BUILD, not just the version: 2.10.0+computecanada != 2.10.0+cu128,
    #   and peak memory is allocator- and kernel-sensitive.
    # ⚠ anchor on a word boundary: an earlier version used `torch-[0-9][^ ]*` and
    #   matched `galore-torch-1.0`, printing a second, bogus line
    #   "torch build: torch-1.0-py3-none-any.whl.metadata". A status line that
    #   lies is worse than none.
    echo "$out" | grep -oE "(^|[^-[:alnum:]])torch-[0-9][^ ]*" | sed 's/^[^t]*//' \
        | head -1 | sed 's/^/    torch build: /'
fi

rm -rf "$TMPV"
echo
if [ $rc -eq 0 ]; then
    echo "############ DEPS RESOLVE — safe to run 01_setup_venv.sh ############"
else
    echo "############ DEPS DO NOT RESOLVE — STOP ############"
    echo "Send this transcript. Do NOT run 01 and do NOT substitute versions."
fi
exit $rc
