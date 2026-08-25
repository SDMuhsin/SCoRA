#!/bin/bash
# ============================================================================
# 00d_probe_runtime.sh — WHICH PACKAGES DOES A JOB ACTUALLY GET? READ-ONLY.
# ============================================================================
#   bash sbatch/fir/00d_probe_runtime.sh
#
# ⛔ WHY THIS EXISTS (fir, 2026-08-25).
#    03_preflight's tracebacks all resolved to
#        /home/sdmuhsin/.local/lib/python3.11/site-packages/{torch,transformers}
#    NOT to the venv we built. If user-site packages shadow the venv, then the
#    pinned stack is NOT what ran, and EVERY result from that job is about a
#    different environment than the one we think we configured.
#
#    ⚠ And the env gate could not see it: it asserts
#        torch.__version__.startswith("2.10.0")
#      which passes for 2.10.0, 2.10.0+cu128 AND 2.10.0+computecanada alike --
#      even though this project's own notes say the BUILD is what matters
#      (peak memory is allocator- and kernel-sensitive). A version check cannot
#      distinguish two different wheels with the same version number.
#
# This script CONCLUDES NOTHING. It prints where every package actually comes
# from, so the diagnosis is read off measurements instead of guessed.
# ============================================================================
set -uo pipefail
FIR_SELF="$(readlink -f "$0")"
cd "$(dirname "$FIR_SELF")/../.." || exit 1
source sbatch/fir/fir_env.sh
fir_log_to fir_probe_runtime "$@"

echo "############ RUNTIME PROBE — $(date -u +%FT%TZ) ############"
echo "node: $(hostname)"
fir_load_modules_gpu || exit 1

report() {   # report <label>  -- runs with whatever env the caller set up
    echo
    echo "=================== $1 ==================="
    "$FIR_VENV/bin/python" - <<'PY'
import os, sys, site, importlib
print(f"  sys.executable      : {sys.executable}")
print(f"  sys.prefix          : {sys.prefix}")
print(f"  base_prefix         : {sys.base_prefix}")
print(f"  in a venv           : {sys.prefix != sys.base_prefix}")
print(f"  site.ENABLE_USER_SITE: {site.ENABLE_USER_SITE}")
try:
    print(f"  user site dir       : {site.getusersitepackages()}")
except Exception as e:
    print(f"  user site dir       : <{type(e).__name__}>")
print(f"  PYTHONNOUSERSITE    : {os.environ.get('PYTHONNOUSERSITE', '<unset>')}")
print(f"  PYTHONPATH          : {os.environ.get('PYTHONPATH', '<unset>')}")
print("  --- sys.path, in order (the FIRST match wins) ---")
for i, p in enumerate(sys.path):
    tag = ""
    if ".local" in p:       tag = "   <-- USER SITE (~/.local): shadows the venv if it comes first"
    elif "/env/" in p or p.rstrip('/').endswith('/env'): tag = "   <-- our venv"
    elif "cvmfs" in p:      tag = "   <-- module stack"
    print(f"   [{i}] {p}{tag}")
print("  --- where each pinned package RESOLVES ---")
# ⭐ the FILE is the answer, not the version. Two wheels can share a version
#   string and be different builds; only the path says which one loaded.
for m in ["torch", "transformers", "datasets", "peft", "accelerate",
          "evaluate", "numpy", "huggingface_hub"]:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        f = getattr(mod, "__file__", "?") or "?"
        where = ("USER-SITE ~/.local" if ".local" in f
                 else "venv" if "/env/" in f
                 else "module stack" if "cvmfs" in f
                 else "?")
        print(f"   {m:16} {v:28} [{where}]")
        print(f"   {'':16} {f}")
    except Exception as e:
        print(f"   {m:16} IMPORT FAILED: {type(e).__name__}: {str(e)[:120]}")
# ⚠ the LOCAL VERSION LABEL is the build. This is the thing the pin check misses.
try:
    import torch
    lbl = torch.__version__.split("+", 1)
    print(f"  torch version={lbl[0]}  build_label={'+' + lbl[1] if len(lbl) > 1 else '<NONE>'}")
    if len(lbl) == 1:
        print("   ⚠ NO build label. The CVMFS wheel reports '+computecanada'.")
        print("     A bare '2.10.0' is a PyPI-style wheel -- i.e. NOT the one 01 installed.")
    print(f"  torch.version.cuda  : {torch.version.cuda}")
    print(f"  cuda available      : {torch.cuda.is_available()}  device_count={torch.cuda.device_count()}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
except Exception as e:
    print(f"  torch probe failed: {type(e).__name__}: {str(e)[:160]}")
PY
}

echo
echo "--- does ~/.local even exist, and what is in it? ---"
if [ -d "$HOME/.local/lib" ]; then
    echo "  $HOME/.local/lib exists:"
    for d in "$HOME"/.local/lib/python*/site-packages; do
        [ -d "$d" ] || continue
        echo "    $d  ($(find -L "$d" -maxdepth 1 -type d | wc -l) entries)"
        for p in torch transformers peft datasets huggingface_hub accelerate; do
            [ -e "$d/$p" ] && echo "        ⚠ $p PRESENT in user site"
        done
    done
else
    echo "  $HOME/.local/lib does not exist -- user-site shadowing is NOT the story"
fi

# 1. exactly as 03_preflight sets things up
( fir_export_offline; report "AS 03_PREFLIGHT RUNS IT (fir_export_offline)" )

# 2. with user-site explicitly disabled, to see whether that is the difference
( fir_export_offline; export PYTHONNOUSERSITE=1
  report "WITH PYTHONNOUSERSITE=1 (does this change what resolves?)" )

echo
echo "############ RUNTIME PROBE COMPLETE — send the whole transcript ############"
echo "⛔ This script diagnoses nothing by itself. Compare the two blocks above:"
echo "   if they differ, user-site packages were shadowing the venv."
