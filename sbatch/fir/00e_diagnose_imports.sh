#!/bin/bash
# ============================================================================
# 00e_diagnose_imports.sh — WHICH package does not run on THIS CPU? READ-ONLY.
# ============================================================================
#   bash sbatch/fir/00e_diagnose_imports.sh
#   bash sbatch/narval/00e_diagnose_imports.sh
#
# ⛔ WHY THIS EXISTS (narval, 2026-09-08).
#    01_setup_venv.sh's post-install check for requirements.txt died as
#        line 255: 812333 Illegal instruction  (core dumped)
#    It imports SEVEN modules in ONE process, so the transcript proved only that
#    one of seven native wheels uses a CPU instruction this host lacks.
#    ⚠ A SIGNAL IS NOT AN EXCEPTION: there is no traceback, no `except` can catch
#      it, and nothing identifies the module. FIR_SETUP Law 12 -- a log that
#      cannot identify its own failure is not evidence. The ONLY way to attribute
#      it is to import each module in ITS OWN PROCESS and see which one dies.
#
# ⚠ WHY A WHEEL CAN INSTALL PERFECTLY AND STILL SIGILL. Alliance builds wheels per
#   micro-arch tier (x86-64-v3 / v4) while PyPI ships generic ones. A wheel
#   compiled for a newer tier installs without complaint and dies on first use, so
#   `pip install` succeeding says nothing about whether the code RUNS here.
#
# ⚠⚠ IT CONCLUDES NOTHING AND CHANGES NOTHING. No installs, no venv edits, no
#    submission. It prints where each package came from and which one dies, so the
#    fix is chosen from a measurement. In particular it does NOT reinstall: a
#    LOGIN node and a COMPUTE node can be different CPUs, and reinstalling a wheel
#    that was fine leaves the real fault in place.
# ============================================================================
set -uo pipefail
FIR_SELF="$(readlink -f "$0")"
cd "$(dirname "$FIR_SELF")/../.." || exit 1
source "${LRS_ENV:-sbatch/fir/fir_env.sh}"
fir_log_to fir_diagnose_imports "$@"

echo "############ IMPORT DIAGNOSIS — $(date -u +%FT%TZ) ############"
echo "This changes NOTHING. It installs nothing and submits nothing."
echo

# --- 0. provenance: Law 12, a log must identify the code that produced it ------
echo "=================== 0. PROVENANCE ==================="
echo "commit : $(git rev-parse --short HEAD 2>/dev/null || echo '?') on $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
echo "uncommitted files: $(git status --porcelain 2>/dev/null | wc -l)"
echo "cluster: ${CC_CLUSTER:-<unset>}   stage dir: $LRS_STAGE_DIR"

# --- 1. the host, because the answer is about THIS CPU -------------------------
echo
echo "=================== 1. THIS HOST ==================="
echo "hostname     : $(hostname)"
echo "kind         : $([ -n "${SLURM_JOB_ID:-}" ] && echo "COMPUTE node (job ${SLURM_JOB_ID})" || echo "LOGIN node")"
echo "uname -m     : $(uname -m)"
echo "cpu          : $(sed -n 's/^model name[ \t]*: //p' /proc/cpuinfo | head -1)"
echo "wheel tier   : ${RSNT_ARCH:-<unset>}   (CC_ARCH=${CC_ARCH:-<unset>})"
echo "PIP_CONFIG   : ${PIP_CONFIG_FILE:-<unset>}"
# ⭐ The CPU FLAGS are the actual question: a wheel built for x86-64-v4 needs
#   avx512f, and no amount of version-matching substitutes for the flag being there.
FLAGS=$(sed -n 's/^flags[ \t]*: //p' /proc/cpuinfo | head -1)
for f in avx avx2 avx512f avx512dq fma bmi2; do
    case " $FLAGS " in *" $f "*) printf '  ✅ %-10s\n' "$f" ;;
                       *)        printf '  ⛔ %-10s ABSENT\n' "$f" ;; esac
done
echo "  ⇒ x86-64-v3 needs avx2+fma+bmi2; x86-64-v4 additionally needs avx512f."

# --- 2. the venv -------------------------------------------------------------
echo
echo "=================== 2. THE VENV ==================="
VPY="$FIR_VENV/bin/python"
if [ ! -x "$VPY" ]; then
    echo "⛔ no interpreter at $VPY"
    echo "   Run $LRS_STAGE_DIR/01_setup_venv.sh first."
    exit 1
fi
echo "python  : $VPY"
echo "version : $("$VPY" -V 2>&1)"
echo "PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-<unset>}  (must be 1)"

# --- 3. THE BISECT -----------------------------------------------------------
# ⛔ Each module in its OWN process. That is the entire point: a module that
#   SIGILLs takes its interpreter with it, so it can never be reported by a
#   process that also imported anything else.
echo
echo "=================== 3. ⭐ ONE MODULE, ONE PROCESS ==================="
# Import names, not distribution names: `pip install scikit-learn` gives `sklearn`,
# `galore-torch` gives `galore_torch`. Getting this wrong reports a false failure.
MODS="torch transformers datasets peft accelerate evaluate triton
      adapters galore_torch lion_pytorch bitsandbytes sentencepiece
      numpy scipy sklearn pandas filelock safetensors tokenizers
      huggingface_hub tqdm requests yaml jinja2 regex packaging psutil"
CULPRITS=""; MISSING=""
for m in $MODS; do
    "$VPY" -c "import $m" >/dev/null 2>&1; rc=$?
    if [ "$rc" -eq 0 ]; then
        printf '  ✅ %-16s ok\n' "$m"
    elif [ "$rc" -gt 128 ]; then
        # 128+N. 132=SIGILL 133=SIGTRAP 134=SIGABRT 136=SIGFPE 139=SIGSEGV
        case "$rc" in
            132) sig="SIGILL — an instruction this CPU does not have" ;;
            133) sig="SIGTRAP" ;; 134) sig="SIGABRT" ;;
            136) sig="SIGFPE"  ;; 139) sig="SIGSEGV — segfault" ;;
            *)   sig="signal $((rc - 128))" ;;
        esac
        printf '  ⛔ %-16s KILLED: exit %s = %s\n' "$m" "$rc" "$sig"
        CULPRITS="$CULPRITS $m"
    else
        printf '  ⚠ %-16s import failed (exit %s) — an ordinary exception:\n' "$m" "$rc"
        "$VPY" -c "import $m" 2>&1 | tail -2 | sed 's/^/       /'
        MISSING="$MISSING $m"
    fi
done

# --- 4. where the guilty wheels came from ------------------------------------
echo
echo "=================== 4. PROVENANCE OF THE NATIVE PACKAGES ==================="
# ⚠ A version number does NOT identify a wheel: 2.10.0, 2.10.0+cu128 and
#   2.10.0+computecanada are three different builds. The WHEEL metadata records the
#   tag it was BUILT for, which is the fact that decides whether it runs here.
"$VPY" - <<'PY'
import importlib.metadata as md, os, re
want = ["numpy","scipy","scikit-learn","pandas","bitsandbytes","torch",
        "sentencepiece","tokenizers","safetensors","triton","psutil","regex"]
for name in want:
    try:
        dist = md.distribution(name)
    except Exception:
        print(f"  {name:16s} NOT INSTALLED"); continue
    # ⚠ `dist.read_text(name)` resolves RELATIVE TO THE .dist-info DIRECTORY.
    #   Passing the full path returned by dist.files silently yields None, so every
    #   row printed "?" -- the column that carries the entire answer read as
    #   "unknown" for reasons that had nothing to do with the packages.
    tags = []
    try:
        txt = dist.read_text("WHEEL") or ""
        tags = [m.group(1).strip() for m in re.finditer(r"^Tag:\s*(.+)$", txt, re.M)]
    except Exception:
        pass
    tag = ", ".join(tags) if tags else "? (no WHEEL metadata — built from sdist?)"
    loc = str(getattr(dist, "_path", "")) or "?"
    print(f"  {name:16s} {dist.version:24s} built-for: {tag}")
    print(f"  {'':16s} {loc}")
PY

# --- 5. what to do -----------------------------------------------------------
echo
echo "=================== 5. WHAT THIS MEANS ==================="
if [ -z "$CULPRITS" ] && [ -z "$MISSING" ]; then
    echo "✅ EVERY module imports on this host. Nothing here is broken."
    echo "   If 01_setup_venv.sh still fails, the failure is NOT a CPU/wheel"
    echo "   mismatch and this diagnosis does not explain it — send the transcript."
    exit 0
fi
[ -n "$MISSING" ] && {
    echo "⚠ NOT INSTALLED / import error:$MISSING"
    echo "   These are ordinary failures with tracebacks above, NOT CPU faults."
}
[ -n "$CULPRITS" ] && {
    echo "⛔ KILLED BY A SIGNAL:$CULPRITS"
    echo
    echo "   ⚠⚠ DO NOT REINSTALL YET. A LOGIN node and a COMPUTE node can be"
    echo "     different CPUs, and this ran on a $([ -n "${SLURM_JOB_ID:-}" ] && echo COMPUTE || echo LOGIN) node."
    echo "     Settle that first — it decides whether the wheel is wrong or only"
    echo "     this host is too old, and those have opposite fixes:"
    echo
    echo "       salloc --account=$FIR_ACCOUNT_CPU --time=0:20:00 --mem=8G --cpus-per-task=2"
    echo "       # then, in the shell it gives you:"
    echo "       bash $LRS_STAGE_DIR/00e_diagnose_imports.sh"
    echo
    echo "     If it imports there, the wheel is FINE and stage 01's check simply"
    echo "     ran on a host the code never runs on. If it dies there too, the"
    echo "     wheel genuinely does not match this cluster's CPUs."
}
exit 1
