#!/bin/bash
# ============================================================================
# FIR PROBE — run this on a FIR LOGIN NODE. It changes NOTHING.
# ============================================================================
#
#   bash sbatch/fir/00_probe_fir.sh 2>&1 | tee fir_probe.txt
#
# Then send back `fir_probe.txt`.
#
# WHY THIS EXISTS
# ---------------
# The rorqual scripts encode facts that are true OF RORQUAL, not of Alliance
# clusters in general: the module set `gcc arrow scipy-stack cuda cudnn` IN THAT
# ORDER, the absence of an explicit `module load python`, the GPU strings
# `h100:1` / `h100_3g.40gb:1`, and the account `def-seokbum`. Every one of those
# is a guess on fir until measured. Previous HPC work burned significant compute
# on exactly this class of mistake, so this script measures instead of assuming.
#
# READ-ONLY: no installs, no venv creation, no job submission, no writes outside
# the probe's own output. Safe to run repeatedly.
# ============================================================================

echo "############ FIR PROBE — $(date -u +%Y-%m-%dT%H:%M:%SZ) ############"

hr() { echo; echo "=================== $* ==================="; }

# ---------------------------------------------------------------------------
hr "1. IDENTITY / SITE"
echo "hostname       : $(hostname -f 2>/dev/null || hostname)"
echo "user           : $USER"
echo "CC_CLUSTER     : ${CC_CLUSTER:-<unset>}"
echo "CC_ARCH        : ${CC_ARCH:-<unset>}"
echo "RSNT_ARCH      : ${RSNT_ARCH:-<unset>}"
echo "EBVERSIONGENTOO: ${EBVERSIONGENTOO:-<unset>}"
echo "cpu            : $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2-)"
echo "pwd            : $(pwd)"
echo "repo present   : $([ -d ./src ] && echo yes || echo 'NO — run this from the repo root')"

# ---------------------------------------------------------------------------
hr "2. MODULE SYSTEM + DEFAULT ENVIRONMENT"
if ! command -v module >/dev/null 2>&1; then
    echo "!!! no Lmod on this host — everything below is meaningless"
else
    echo "--- currently loaded (before we touch anything) ---"
    module list 2>&1 | head -30
    echo
    echo "--- StdEnv versions available ---"
    module avail StdEnv 2>&1 | head -20
fi

# ---------------------------------------------------------------------------
hr "3. DO RORQUAL'S MODULES EXIST ON FIR? (exact names, in rorqual's order)"
# The ORDER below is rorqual's, deliberately preserved. If fir needs a different
# order, that is a finding — do not silently reorder.
for m in gcc arrow scipy-stack cuda cudnn python rust; do
    echo "--- module avail $m ---"
    module avail "$m" 2>&1 | grep -v '^$' | head -12
    echo
done

# ---------------------------------------------------------------------------
hr "4. THE RORQUAL LOAD LINE, TRIED VERBATIM"
echo "\$ module load gcc arrow scipy-stack cuda cudnn"
( module load gcc arrow scipy-stack cuda cudnn 2>&1 ) | head -20
echo "exit=$?"
echo
echo "--- what that actually loaded ---"
( module load gcc arrow scipy-stack cuda cudnn >/dev/null 2>&1; module list 2>&1 | head -30 )
echo
echo "--- python that results ---"
( module load gcc arrow scipy-stack cuda cudnn >/dev/null 2>&1
  echo "which python : $(command -v python || echo none)"
  echo "python -V    : $(python -V 2>&1)"
  echo "which python3: $(command -v python3 || echo none)"
  echo "python3 -V   : $(python3 -V 2>&1)"
  echo "numpy from system site-packages (scipy-stack): "
  python -c "import numpy, sys; print('  numpy', numpy.__version__, 'from', numpy.__file__)" 2>&1 | head -3 )

# ---------------------------------------------------------------------------
hr "5. THE DOWNLOAD-NODE LOAD LINE (no cuda/cudnn), AS IN download_cache.sh"
( module load gcc arrow scipy-stack 2>&1 | head -10; echo "exit=$?" )

# ---------------------------------------------------------------------------
hr "6. ALLIANCE WHEELHOUSE — what versions can we actually install offline?"
# `avail_wheels` is the Alliance-specific tool; PyPI is generally NOT reachable
# from compute nodes and the local wheelhouse is what pip resolves against.
if command -v avail_wheels >/dev/null 2>&1; then
    # THIS repo's package set. No deepspeed / flash-attn / unsloth: src/train_glue.py
    # is a single-process trainer with no distributed or fused-kernel arms.
    # `adapters` and `galore-torch` / `lion-pytorch` are module-scope imports in
    # train_glue.py (48 of them) -- see requirements.txt, which is what 01 installs.
    for pkg in torch triton transformers peft datasets accelerate evaluate \
               adapters galore_torch lion_pytorch bitsandbytes scikit_learn \
               filelock sentencepiece pandas; do
        echo "--- avail_wheels $pkg ---"
        avail_wheels "$pkg" 2>&1 | head -12
        echo
    done
else
    echo "!!! avail_wheels not found — report this; it changes the venv strategy entirely"
fi

# ---------------------------------------------------------------------------
hr "7. INTERNET REACHABILITY FROM THE LOGIN NODE"
# The download step needs HuggingFace; the venv build may need PyPI.
for host in pypi.org huggingface.co; do
    printf "%-20s " "$host"
    if command -v curl >/dev/null 2>&1; then
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 12 "https://$host" 2>/dev/null)
        echo "HTTP $code"
    else
        echo "(no curl)"
    fi
done

# ---------------------------------------------------------------------------
hr "8. GPUS AVAILABLE TO SLURM — what may follow '--gpus='"
echo "--- gres/partitions ---"
sinfo -o "%P %.6D %G %N" 2>&1 | head -30
echo
echo "--- distinct Gres strings ---"
sinfo -h -o "%G" 2>&1 | tr ',' '\n' | sed 's/(.*//' | sort -u | head -25
echo
echo "--- does rorqual's MIG slice name exist here? ---"
sinfo -h -o "%G" 2>&1 | grep -o 'h100[^ ,]*' | sort -u | head
echo "(rorqual used: h100:1 and h100_3g.40gb:1)"

# ---------------------------------------------------------------------------
hr "9. ACCOUNTS / ALLOCATIONS (rorqual used def-seokbum)"
if command -v sacctmgr >/dev/null 2>&1; then
    sacctmgr -nP show user "$USER" withassoc format=Account,Partition 2>&1 | sort -u | head -20
fi
echo "--- sshare ---"
sshare -U -u "$USER" 2>&1 | head -15

# ---------------------------------------------------------------------------
hr "10. FILESYSTEMS AND QUOTA (venv + HF cache are large: PG-19 alone is ~7 GB)"
if command -v diskusage_report >/dev/null 2>&1; then
    diskusage_report 2>&1 | head -20
else
    df -h "$HOME" "${SCRATCH:-$HOME}" "${PROJECT:-$HOME}" 2>&1 | head -10
fi
echo "HOME=$HOME  SCRATCH=${SCRATCH:-<unset>}  PROJECT=${PROJECT:-<unset>}"
echo "SLURM_TMPDIR (only set inside a job): ${SLURM_TMPDIR:-<unset as expected on login>}"

# ---------------------------------------------------------------------------
hr "11. EXISTING REPO STATE ON THIS CLUSTER"
echo "./env exists     : $([ -d ./env ] && echo yes || echo no)"
if [ -d ./env ]; then
    echo "./env python     : $(./env/bin/python -V 2>&1)"
    echo "--- key package versions in the existing venv ---"
    ./env/bin/python - <<'PY' 2>&1 | head -20
import importlib
for m in ["torch","triton","transformers","peft","datasets","accelerate",
          "evaluate","adapters","galore_torch","lion_pytorch","bitsandbytes",
          "sklearn","scipy","pandas","filelock"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:14} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m:14} MISSING ({type(e).__name__})")
PY
    echo "--- torch CUDA build (if importable) ---"
    ./env/bin/python -c "import torch;print('  torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())" 2>&1 | head -3
fi
echo "./data exists    : $([ -d ./data ] && echo "yes ($(du -sh ./data 2>/dev/null | cut -f1))" || echo no)"
echo "requirements.txt : $([ -f requirements.txt ] && echo yes || echo no)"

# ---------------------------------------------------------------------------
hr "12. WHAT THE EXPERIMENTS NEED (check feasibility against §6)"
cat <<'NEEDS'
  The runner is src/train_glue.py -- ONE process, ONE GPU, no distributed backend.
  It has ~48 UNGUARDED module-scope imports, so a single missing package kills
  every arm, including arms that never use it.  ⇒ 01_setup_venv.sh installs
  `-r requirements.txt` under a constraints file rather than a hand list; see
  llmdocs/FIR_SETUP.md C11 ("hand-listing packages cannot converge").

  ⛔ THE PIN DECISION (user, 2026-08-25): fir runs the FIR-NATIVE stack
        torch 2.10.0 · transformers 4.51.3 · datasets 4.5.0 · peft 0.18.1
     NOT this repo's dev-box stack (torch 2.5.1+cu121 / transformers 4.45.2 /
     peft 0.13.2).  §6 above must show every one of those resolving; if any does
     not, STOP and report rather than substituting a neighbour.

  ⛔⛔ THE PRICE, so it is never lost: src/qwha_adapter.py:14 -- every FourierFT
     number in this repo is gated BIT-IDENTICAL to the INSTALLED
     peft.tuners.fourierft.  peft 0.13.2 -> 0.18.1 may move that layer.
     03_preflight.sh re-runs the bit-identity verifiers ON FIR and a failure is
     LOUD, not a warning.

  Backbone: google/gemma-2b -- a GATED repo (5.0 GB).  §13 checks the token.
NEEDS

# ---------------------------------------------------------------------------
hr "13. HUGGINGFACE TOKEN + GATED-MODEL ACCESS (google/gemma-2b)"
# ⚠⚠ MEASURED TRAP (dev box, 2026-08-25): setting HF_HOME ALSO RELOCATES where
#    huggingface_hub looks for the token ($HF_HOME/token).  An access check in a
#    plain shell PASSES and the identical download under the job's own environment
#    FAILS with "Access to model google/gemma-2b is restricted".  So this probe
#    reports BOTH locations rather than whichever happens to be in scope.
echo "HF_TOKEN env         : ${HF_TOKEN:+present}${HF_TOKEN:-<unset>}"
echo "./.hf_token (repo root, searched FIRST) : $([ -s ./.hf_token ] && echo present || echo ABSENT)"
echo "~/.cache/huggingface/token : $([ -s "$HOME/.cache/huggingface/token" ] && echo present || echo ABSENT)"
echo "./data/token (HF_HOME=./data) : $([ -s ./data/token ] && echo present || echo ABSENT)"
echo
echo "--- can we actually reach the gated repo? (metadata only, no download) ---"
TOK="${HF_TOKEN:-$(cat ./.hf_token 2>/dev/null || cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"
TOK="$(printf '%s' "$TOK" | tr -d '[:space:]')"
if [ -n "$TOK" ] && command -v curl >/dev/null 2>&1; then
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 \
           -H "Authorization: Bearer $TOK" \
           "https://huggingface.co/google/gemma-2b/resolve/main/config.json" 2>/dev/null)
    echo "  GET config.json with token -> HTTP $code   (200 = access granted; 401/403 = accept the licence at https://huggingface.co/google/gemma-2b)"
else
    echo "  no token available on this node -> 02_download_cache.sh CANNOT fetch gemma-2b"
fi

# ---------------------------------------------------------------------------
hr "14. LUSTRE flock SEMANTICS (decides whether a shared CSV writer is safe)"
# ⚠ filelock uses fcntl.flock.  On Lustre mounted `-o localflock` the lock is
#   NODE-LOCAL: two array tasks on different nodes both "hold" it, interleave a
#   read-modify-write, and the second os.replace SILENTLY DISCARDS the first row.
#   Nothing raises; the table just comes up short.
#   fir measured `flock` (cluster-wide) on /scratch in 2026-08.  RE-CHECK, do not
#   inherit -- and note this repo writes ONE CSV PER CELL anyway (the upsert key
#   omits seed; scripts/r304_upsert_gate.py enforces it), so it does not depend on
#   the answer.  Report it regardless: it is a property of the cluster.
# ⚠ `$PROJECT` is UNSET on fir [measured 2026-08-25], so an earlier version of this
#   loop fell back to "${PROJECT:-$HOME}" and probed /home TWICE while never probing
#   /project at all -- the very filesystem the repo lives on. Resolve it from the cwd
#   instead, which is where the repo actually is.
FIR_PROJECT_GUESS="$(readlink -f "$(pwd)" 2>/dev/null)"
for m in "${SCRATCH:-/scratch/$USER}" "$HOME" "${PROJECT:-$FIR_PROJECT_GUESS}"; do
    echo "--- $m ---"
    findmnt -o TARGET,SOURCE,FSTYPE,OPTIONS --target "$m" 2>&1 | head -4
done

echo
echo "############ PROBE COMPLETE — send back the whole transcript ############"
