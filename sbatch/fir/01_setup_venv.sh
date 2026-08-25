#!/bin/bash
# ============================================================================
# 01_setup_venv.sh — build the venv on a FIR LOGIN NODE. Internet required.
# ============================================================================
#   bash sbatch/fir/01_setup_venv.sh [--fresh]
#   (a full transcript lands in ./logs/ automatically — do not add `| tee`)
#
# Idempotent: re-running verifies and repairs rather than rebuilding.
#
# WHY THIS FIGHTS THE WHEELHOUSE
# ------------------------------
# `avail_wheels` on fir shows only the DEFAULTS (torch 2.13, transformers 5.x,
# datasets 5.x).  The older pins are in the wheelhouse too — each resolves as
# `<version>+computecanada`.  ⛔ NO `--index-url` ANYWHERE: Alliance's pip config
# adds `find-links` to the CVMFS wheelhouse with `prefer-binary = true`, so the
# local wheel wins regardless; `--index-url ...cu128` and even `--isolated` both
# still resolved to `+computecanada` on the sibling project.
#
# ⛔⛔ USER DECISION 2026-08-25 — READ THIS BEFORE CHANGING A PIN.
#   fir runs the FIR-NATIVE stack (torch 2.10.0 / transformers 4.51.3 /
#   datasets 4.5.0 / peft 0.18.1), NOT this repo's dev-box stack (torch 2.5.1+cu121
#   / transformers 4.45.2 / peft 0.13.2).  The price is named in fir_env.sh and
#   re-checked by 03_preflight.sh: `src/qwha_adapter.py:14` gates every FourierFT
#   number in this repo as BIT-IDENTICAL to the INSTALLED `peft.tuners.fourierft`,
#   and peft 0.13 -> 0.18 can move that layer.  If the preflight's bit-identity
#   check fails, the fir comparator is a DIFFERENT comparator and that fact must
#   travel with every table quoting both.
#
# ⚠ RECORD THE BUILD, NOT JUST THE VERSION: fir installs `2.10.0+computecanada`;
#   the dev box measured `2.10.0+cu128`.  Same upstream version, different compile.
#   Peak memory is allocator- and kernel-sensitive => fir numbers are internally
#   comparable and must NOT be quoted interchangeably with dev-box tables.
#
# Each stage VERIFIES before the next begins, so a failure costs a login-node
# minute rather than a GPU allocation.
# ============================================================================
set -uo pipefail

# FIR_SELF must be resolved BEFORE the cd: $0 is relative to the invocation
# directory, and fir_log_to re-execs this script from the repo root.
FIR_SELF="$(readlink -f "$0")"
cd "$(dirname "$FIR_SELF")/../.." || exit 1          # repo root
source sbatch/fir/fir_env.sh
fir_log_to fir_setup_venv "$@"

FRESH=false
for a in "$@"; do
    case "$a" in
        --fresh) FRESH=true ;;
        *) echo "unknown option: $a"; echo "usage: $0 [--fresh]"; exit 1 ;;
    esac
done

echo "############ fir venv setup — $(date -u +%FT%TZ) ############"
echo "repo (on /project): $(pwd)"

# --- 0. modules FIRST, in the load-bearing order (cudnn is only visible after cuda)
fir_load_modules_gpu || { echo "FAIL: module load '$FIR_MODULES_GPU'"; exit 1; }
echo "python: $(python -V 2>&1) at $(command -v python)"

# --- 1. scratch targets + symlinks (venv/data must NOT consume /project inodes)
echo; echo "--- linking venv/data/temp onto /scratch ---"
fir_link_scratch || exit 1
# --- 2. create the venv
# ⚠ A HALF-BUILT VENV MUST BE DETECTED, NOT INHERITED.
# CPython's venv runs `_setup_pip` (ensurepip) BEFORE `setup_scripts` (which writes
# bin/activate), so a Ctrl-C during ensurepip — which is exactly what happens when
# the slow Lustre/CVMFS create looks hung — leaves `bin/python` present and
# `bin/activate` absent. Checking only for bin/python then "finds" a venv, skips
# creation, and every later step dies on `./env/bin/activate: No such file`.
# Observed on fir 2026-08-11. Validate by RUNNING the interpreter, not by stat.
venv_is_healthy() {
    [ -x "$FIR_VENV/bin/python" ] && [ -f "$FIR_VENV/bin/activate" ] \
        && "$FIR_VENV/bin/python" -c "import sys, ensurepip" >/dev/null 2>&1
}

if $FRESH && [ -d "$FIR_VENV_REAL" ]; then
    echo "--fresh: removing $FIR_VENV_REAL"; rm -rf "$FIR_VENV_REAL"
fi
if ! venv_is_healthy; then
    if [ -e "$FIR_VENV_REAL/bin" ]; then
        echo; echo "--- found a BROKEN/partial venv at $FIR_VENV_REAL — removing and rebuilding ---"
        echo "    (bin/python:  $([ -x "$FIR_VENV/bin/python" ] && echo present || echo missing))"
        echo "    (bin/activate:$([ -f "$FIR_VENV/bin/activate" ] && echo present || echo missing))"
        rm -rf "$FIR_VENV_REAL"
    fi
    mkdir -p "$FIR_VENV_REAL"
    echo; echo "--- creating venv at $FIR_VENV_REAL ---"
    echo "    NOTE: MEASURED on fir — virtualenv ~10s, python -m venv ~15s. Not minutes."
    echo "    If it sits far longer, something IS wrong. Re-running repairs a partial venv."
    # --system-site-packages is REQUIRED, not stylistic: numpy comes from the
    # scipy-stack module, and the venv is built against it. Without it
    # `import datasets` dies with "ModuleNotFoundError: No module named 'numpy'"
    # (the rorqual scripts carry the same note). Our pins still shadow the system copy.
    #
    # ⚠ CREATION IS A CASCADE, because both seeders fail in different ways on fir.
    #
    #   1. `virtualenv --no-download` is Alliance's recommended path and is the
    #      fastest, BUT its via_app_data seeder extracts bundled pip/setuptools
    #      wheels that live in CVMFS, and CVMFS returns EIO on a cold/transient
    #      read. Observed on fir 2026-08-11:
    #        RuntimeError: failed to build image setuptools, pip because:
    #        OSError: [Errno 5] Input/output error   (in zip_ref.extractall)
    #      That is a filesystem hiccup, not a broken environment, so it is worth
    #      one retry before giving up on the fast path.
    #   2. `python -m venv` runs ensurepip instead. It is SLOW here (1-3 min on
    #      Lustre) but it does not read CVMFS zips, so it survives exactly the
    #      failure above. This is the path that was already working before it was
    #      interrupted for looking hung.
    #
    # App-data is redirected onto /scratch so virtualenv's cache never touches
    # /home (48 GiB) or /project (inode-bound).
    export VIRTUALENV_OVERRIDE_APP_DATA="$FIR_SCRATCH_ROOT/.virtualenv_appdata"
    mkdir -p "$VIRTUALENV_OVERRIDE_APP_DATA"

    created=false
    if command -v virtualenv >/dev/null 2>&1; then
        for attempt in 1 2; do
            echo "    [$attempt/2] virtualenv --no-download --system-site-packages"
            if virtualenv --no-download --system-site-packages "$FIR_VENV_REAL"; then
                created=true; break
            fi
            echo "    virtualenv attempt $attempt failed (CVMFS EIO is the known cause)"
            rm -rf "$FIR_VENV_REAL" "$VIRTUALENV_OVERRIDE_APP_DATA"
            mkdir -p "$FIR_VENV_REAL" "$VIRTUALENV_OVERRIDE_APP_DATA"
            sleep 5
        done
    fi
    if ! $created; then
        echo "    falling back to: python -m venv --system-site-packages (~15s, measured)"
        echo "    (ensurepip instead of the CVMFS app-data zip extract — survives the EIO above)"
        rm -rf "$FIR_VENV_REAL"; mkdir -p "$FIR_VENV_REAL"
        python -m venv --system-site-packages "$FIR_VENV_REAL" || {
            echo "FAIL: both virtualenv and python -m venv failed to create $FIR_VENV_REAL"
            echo "  If this was another Errno 5, the CVMFS/Lustre mount is unhealthy right now —"
            echo "  wait a few minutes and re-run; nothing here is corrupted."
            exit 1; }
    fi
    venv_is_healthy || { echo "FAIL: venv still not healthy after creation"; exit 1; }
    echo "    venv created OK"
fi
# shellcheck disable=SC1091
source "$FIR_VENV/bin/activate" || exit 1
echo "venv python: $(python -V 2>&1) at $(command -v python)"

python -m pip install -q --upgrade pip setuptools wheel packaging ninja || exit 1

# ⚠ THE TORCH GUARD. A pip stage can silently DOWNGRADE torch by pulling a
# dependency that pins an older one — which is exactly what happened on fir
# 2026-08-11: flash-attn's Alliance wheel is `2.8.3+torch29.computecanada`, it
# requires torch 2.9.x, and installing it without --no-deps took torch from the
# pinned 2.10.0 down to 2.9.1. Every earlier stage had already verified and
# passed, so nothing complained until the preflight job failed on a compute node.
# Re-assert the pin after EVERY stage so the culprit is named at the moment it acts.
assert_torch_pin() {
    local where="$1"
    python - <<PY || { echo "FAIL: torch pin broken by: $where"; exit 1; }
import sys, torch
want, got = "${FIR_PIN_TORCH}", torch.__version__
if not got.startswith(want):
    print(f"  ⚠⚠ TORCH DOWNGRADED to {got} (pinned {want}) by: $where")
    print( "     Some dependency pinned an older torch. Re-install that package with")
    print( "     --no-deps, or constrain it. Continuing would measure a DIFFERENT stack.")
    sys.exit(1)
PY
}

stage() {   # stage <label> <import-check> -- <pip args...>
    local label="$1" check="$2"; shift 3
    echo; echo "--- $label ---"
    python -m pip install "$@" || { echo "FAIL: pip install for $label"; exit 1; }
    python -c "$check" || { echo "FAIL: post-install verification for $label"; exit 1; }
    assert_torch_pin "$label"
}

# ===========================================================================
# THE STAGES.  Order is load-bearing: the pinned stages run FIRST and decide
# versions; requirements.txt comes last under a CONSTRAINTS file so its loose
# bounds (`transformers>=4.30.0` would pull 5.x) cannot override them.
# ===========================================================================

# --- 3. torch FIRST. Everything else compiles or links against it.
#     ⛔ NO --index-url. See the header.
stage "torch==$FIR_PIN_TORCH" \
      "import torch; print('  torch', torch.__version__)" \
      -- -q "torch==$FIR_PIN_TORCH"

# --- 4. the pinned HF stack, installed TOGETHER so pip resolves them as a set
#     rather than letting each one re-solve and bump a neighbour.
stage "pinned HF stack" \
      "import transformers, datasets, peft, accelerate, evaluate; \
print('  transformers', transformers.__version__, '| datasets', datasets.__version__, \
'| peft', peft.__version__, '| accelerate', accelerate.__version__, '| evaluate', evaluate.__version__)" \
      -- -q "transformers==$FIR_PIN_TRANSFORMERS" "datasets==$FIR_PIN_DATASETS" \
            "peft==$FIR_PIN_PEFT" "accelerate==$FIR_PIN_ACCELERATE" \
            "evaluate==$FIR_PIN_EVALUATE"

# --- 4b. ⚠⚠ TRITON — EXPLICIT, AND THAT IS THE WHOLE POINT.
#     `triton` is a transitive dependency of `torch 2.10.0+cu128` (the dev-box
#     build) and NOT of `torch 2.10.0+computecanada` (fir's).  Same version
#     string, DIFFERENT DEPENDENCY GRAPH — invisible to every version check.
#     On the sibling project this silently removed triton and killed an entire
#     artifact at import time.  This repo does not import triton directly, but
#     torch.compile and several fused paths reach for it, and the cost of the
#     stage is seconds.  ⛔ DO NOT TIDY THIS AWAY AS REDUNDANT.
stage "triton (NOT transitive on every torch build — see comment)" \
      "import triton; print('  triton', triton.__version__)" \
      -- -q triton

# --- 4c. ⚠⚠ THE REPO'S OWN requirements.txt, UNDER A CONSTRAINTS FILE.
#     HAND-LISTING PACKAGES CANNOT CONVERGE.  On the sibling project three
#     preflight jobs died on three DIFFERENT missing packages — galore_torch,
#     then lion_pytorch, with `adapters` already queued behind them — and every
#     one was declared in requirements.txt all along.  train_glue.py has ~48
#     UNGUARDED module-scope imports, so a package no arm uses still kills every
#     arm.  Curating a list IS the bug; installing what the repo DECLARES is the
#     fix.
#     ⚠ PEP 440: `torch==2.10.0` DOES match `2.10.0+computecanada` — a local
#       version label is ignored unless the specifier carries one.  So the
#       constraints file pins without fighting the wheelhouse build.
CONSTRAINTS="$FIR_SCRATCH_ROOT/fir_constraints.txt"
{
    echo "torch==$FIR_PIN_TORCH"
    echo "transformers==$FIR_PIN_TRANSFORMERS"
    echo "datasets==$FIR_PIN_DATASETS"
    echo "peft==$FIR_PIN_PEFT"
    echo "accelerate==$FIR_PIN_ACCELERATE"
    echo "evaluate==$FIR_PIN_EVALUATE"
} > "$CONSTRAINTS"
echo; echo "--- constraints file ($CONSTRAINTS) ---"; cat "$CONSTRAINTS"

# ⚠ requirements.txt:6 says `adapters`.  On the sibling repo the equivalent line
#   said `adapter-transformers`, which now resolves to a DEPRECATED 4.0.0 STUB
#   whose entire content is "use adapters instead": it installs cleanly, provides
#   no `adapters` module, and the import still fails.  A package that installs is
#   not a package that imports — which is why the check below imports it.
stage "requirements.txt (under constraints)" \
      "import adapters, galore_torch, lion_pytorch, sklearn, scipy, pandas, filelock; \
print('  requirements leaf packages import OK')" \
      -- -q -r requirements.txt -c "$CONSTRAINTS"

# --- 4d. sentencepiece: gemma's tokenizer is SentencePiece-backed.  It is NOT in
#     requirements.txt (the repo's history is RoBERTa/BPE), so it would have been
#     the next "one more missing package" round trip.  Named explicitly, with the
#     reason, rather than discovered on a compute node.
stage "sentencepiece (gemma tokenizer) + protobuf" \
      "import sentencepiece; print('  sentencepiece', sentencepiece.__version__)" \
      -- -q sentencepiece protobuf

# ===========================================================================
# --- 5. FINAL GATE
# ===========================================================================
# ⚠ FIR_ASSERT_SKIP_TEMP=1 IS A REAL DEPENDENCY ORDER, NOT AN ESCAPE HATCH.
#   01c_stage_repos.sh clones the authors' LoCA/QWHA trees and needs this venv,
#   so it can only run AFTER this script.  Without the flag, a COMPLETELY CORRECT
#   fresh setup could never pass its own closing gate and would always end
#   "SETUP INCOMPLETE" — observed on fir 2026-08-14.  01c and 03_preflight do NOT
#   set it, so nothing reaches a GPU unchecked.
#
# `cpu` because this is a LOGIN NODE: a GPU-dependent check here would condemn a
#   working environment.  Verification must be no more privileged than the node
#   it runs on.
echo
if FIR_ASSERT_SKIP_TEMP=1 fir_assert_env cpu; then
    echo
    echo "############ SETUP OK ############"
    echo "next:  bash sbatch/fir/01c_stage_repos.sh     # authors' clones for the bit-identity gates"
    echo "then:  bash sbatch/fir/02_download_cache.sh   # LOGIN node — compute nodes have no internet"
else
    echo
    echo "############ SETUP INCOMPLETE ############"
    echo "Read the FAIL lines above. Do not proceed to 02/03 — they will fail further in."
    exit 1
fi
