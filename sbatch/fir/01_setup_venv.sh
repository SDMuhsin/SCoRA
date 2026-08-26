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
fir_print_provenance

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
        && "$FIR_VENV/bin/python" -c "import sys, ensurepip" >/dev/null 2>&1 \
        && venv_not_relocated
}

# ⛔⛔ A MOVED VENV IS A BROKEN VENV, AND EVERY EXPLICIT-PATH CHECK MISSES IT.
#    bin/activate hardcodes VIRTUAL_ENV as an absolute path fixed at creation.
#    Move the directory and activate still "succeeds" -- it just prepends a
#    NONEXISTENT dir to PATH, so bare `python` becomes the module python with no
#    torch and no peft. Observed on fir 2026-08-26 after the venv was moved from
#    .../lora_research_signal to .../SCoRA (a migration this project suggested).
#    01 reported SETUP OK because venv_is_healthy tested only the explicit
#    "$FIR_VENV/bin/python" path -- exactly the check that cannot see it.
#    ⇒ treat relocation as UNHEALTHY so a plain re-run repairs it.
venv_not_relocated() {
    local act="$FIR_VENV/bin/activate" stamp real
    [ -f "$act" ] || return 1
    real="$(readlink -f "$FIR_VENV" 2>/dev/null)"
    stamp="$(sed -n 's/^VIRTUAL_ENV=["'"'"']\{0,1\}\([^"'"'"']*\)["'"'"']\{0,1\}$/\1/p' "$act" | head -1)"
    [ -n "$stamp" ] || return 0
    [ "$(readlink -f "$stamp" 2>/dev/null)" = "$real" ] && return 0
    echo "    venv was MOVED: activate says $stamp, actually at $real -> rebuilding"
    return 1
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
# ⚠⚠ CALL THE VENV INTERPRETER EXPLICITLY FROM HERE DOWN, NEVER BARE `python`.
#   `activate` only PREPENDS to PATH, and it prepends the absolute path baked in
#   at creation time. If the venv directory was ever moved, that path does not
#   exist, activate still succeeds, and bare `python` is the MODULE python -- so
#   `python -m pip install` would install into the module environment (or fail),
#   and every check below would test the wrong interpreter. venv_not_relocated()
#   above makes that impossible here, but the SAME line in 02_download_cache.sh
#   had no such guard, so this is enumerated rather than argued (FIR_SETUP Law 4).
VPY="$FIR_VENV/bin/python"
echo "venv python: $("$VPY" -V 2>&1) at $VPY"
echo "  PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-<unset>}  (must be 1, or pip may no-op against ~/.local)"

"$VPY" -m pip install -q --upgrade pip setuptools wheel packaging ninja || exit 1

# ⚠ THE TORCH GUARD. A pip stage can silently DOWNGRADE torch by pulling a
# dependency that pins an older one — which is exactly what happened on fir
# 2026-08-11: flash-attn's Alliance wheel is `2.8.3+torch29.computecanada`, it
# requires torch 2.9.x, and installing it without --no-deps took torch from the
# pinned 2.10.0 down to 2.9.1. Every earlier stage had already verified and
# passed, so nothing complained until the preflight job failed on a compute node.
# Re-assert the pin after EVERY stage so the culprit is named at the moment it acts.
assert_torch_pin() {
    local where="$1"
    "$VPY" - <<PY || { echo "FAIL: torch pin broken by: $where"; exit 1; }
import sys, torch
want, got = "${FIR_PIN_TORCH}", torch.__version__
if not got.startswith(want):
    print(f"  ⚠⚠ TORCH DOWNGRADED to {got} (pinned {want}) by: $where")
    print( "     Some dependency pinned an older torch. Re-install that package with")
    print( "     --no-deps, or constrain it. Continuing would measure a DIFFERENT stack.")
    sys.exit(1)
PY
}

# ⛔⛔ THE DEFECT THAT COST THIS SESSION A ROUND TRIP, fir 2026-08-26.
#   `python -m pip install torch==2.10.0` into a FRESHLY CREATED, EMPTY venv
#   printed "Requirement already satisfied" and installed NOTHING — because the
#   venv is --system-site-packages, ~/.local holds torch 2.10.0 / transformers
#   4.51.3 / datasets 4.5.0 / peft 0.18.1 (the SAME versions we pin), and pip
#   counts those as satisfying the requirement. The stage's own post-install
#   check then did `import torch; print(torch.__version__)`, imported ~/.local,
#   printed `2.10.0`, and PASSED. Four stages in a row passed against packages
#   that were never installed. 01 declared the venv built; the next compute-node
#   job, where PYTHONNOUSERSITE=1 removes ~/.local, died on `import transformers`.
#
#   Two fixes, and BOTH are needed:
#     1. fir_env.sh now exports PYTHONNOUSERSITE=1 at source time, so pip cannot
#        see ~/.local and actually installs. That is the CAUSE fix.
#     2. this one — every stage now asserts its packages resolve INSIDE the venv
#        directory. That is the CONTROL, and it is the part that can fail. A
#        version check cannot distinguish "installed here" from "already present
#        somewhere else"; only the path can. ⛔ Do not delete it as redundant with
#        (1): it is what will catch the NEXT way the venv ends up empty.
assert_in_venv() {   # assert_in_venv <label> "<mod> <mod> ..."
    local label="$1" mods="$2" vreal
    [ -n "$mods" ] || return 0
    vreal="$(readlink -f "$FIR_VENV_REAL")"
    FIR_CHECK_MODS="$mods" FIR_CHECK_VENV="$vreal" "$VPY" - <<'PY' || {
import importlib, os, sys
venv = os.path.realpath(os.environ["FIR_CHECK_VENV"])
bad = []
for m in os.environ["FIR_CHECK_MODS"].split():
    try:
        mod = importlib.import_module(m)
    except Exception as e:
        print(f"  ⛔ {m}: import FAILED right after install: {type(e).__name__}: {str(e)[:200]}")
        bad.append(m); continue
    f = os.path.realpath(getattr(mod, "__file__", "") or "")
    if not f.startswith(venv + os.sep):
        print(f"  ⛔ {m} resolves OUTSIDE the venv -> {f or '<no __file__>'}")
        bad.append(m)
if bad:
    print(f"     expected everything under: {venv}")
    print("     pip reported success and the version check passed, but the packages")
    print("     are not IN the venv. This is the --system-site-packages + ~/.local")
    print("     'already satisfied' no-op. PYTHONNOUSERSITE should have prevented it;")
    print("     check it is exported, then re-run. Manual escape hatch:")
    print("       python -m pip install --ignore-installed <pkgs>")
    sys.exit(1)
print(f"  location: all in venv ({len(os.environ['FIR_CHECK_MODS'].split())} modules)")
PY
        echo "FAIL: packages for '$label' are not installed in the venv"; exit 1; }
}

stage() {   # stage <label> <import-check> <modules-that-must-be-in-the-venv> -- <pip args...>
    local label="$1" check="$2" mods="$3"; shift 4
    echo; echo "--- $label ---"
    "$VPY" -m pip install "$@" || { echo "FAIL: pip install for $label"; exit 1; }
    "$VPY" -c "$check" || { echo "FAIL: post-install verification for $label"; exit 1; }
    assert_in_venv "$label" "$mods"
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
      "torch" \
      -- -q "torch==$FIR_PIN_TORCH"

# --- 4. the pinned HF stack, installed TOGETHER so pip resolves them as a set
#     rather than letting each one re-solve and bump a neighbour.
stage "pinned HF stack" \
      "import transformers, datasets, peft, accelerate, evaluate; \
print('  transformers', transformers.__version__, '| datasets', datasets.__version__, \
'| peft', peft.__version__, '| accelerate', accelerate.__version__, '| evaluate', evaluate.__version__)" \
      "transformers datasets peft accelerate evaluate huggingface_hub" \
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
      "triton" \
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
      "adapters galore_torch lion_pytorch" \
      -- -q -r requirements.txt -c "$CONSTRAINTS"

# --- 4d. sentencepiece: gemma's tokenizer is SentencePiece-backed.  It is NOT in
#     requirements.txt (the repo's history is RoBERTa/BPE), so it would have been
#     the next "one more missing package" round trip.  Named explicitly, with the
#     reason, rather than discovered on a compute node.
stage "sentencepiece (gemma tokenizer) + protobuf" \
      "import sentencepiece; print('  sentencepiece', sentencepiece.__version__)" \
      "sentencepiece" \
      -- -q sentencepiece protobuf

# ===========================================================================
# --- 5. FINAL GATE
# ===========================================================================
# ⚠⚠ `fir_assert_env cpu 01` — THE STAGE ARGUMENT IS LOAD-BEARING.
#   Two of the gate's checks need artifacts a LATER stage creates: the temp/
#   clones come from 01c, and the offline HF cache comes from 02.  Run
#   unconditionally, this gate demands what cannot exist yet and a COMPLETELY
#   CORRECT FRESH SETUP ALWAYS ENDS "SETUP INCOMPLETE" — observed on fir
#   2026-08-25, and on the sibling project in 2026-08 before that.
#   Passing `01` says "only stage 01 has run"; later-stage checks are SKIPPED
#   WITH THEIR REASON.  03_preflight passes no stage, so every check is enforced
#   before anything reaches a GPU.
#
# `cpu` because this is a LOGIN NODE: a GPU-dependent check here would condemn a
#   working environment.  Verification must be no more privileged than the node
#   it runs on.
echo
if fir_assert_env cpu 01; then
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
