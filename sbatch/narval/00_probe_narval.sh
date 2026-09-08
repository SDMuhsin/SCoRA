#!/bin/bash
# ============================================================================
# 00_probe_narval.sh — MEASURE narval. Run on a NARVAL LOGIN NODE.
# ============================================================================
#   bash sbatch/narval/00_probe_narval.sh
#
# It is READ-ONLY except for ONE file it writes: sbatch/narval/measured.sh.
# No installs, no venv, no job submission, no downloads. Safe to re-run.
#
# ⛔⛔ WHY THIS SCRIPT WRITES A FILE INSTEAD OF JUST PRINTING ONE.
#   fir's probe printed a fact dump and a HUMAN then copied values into
#   fir_env.sh. Four of the values in that file were never copied from a
#   measurement at all -- they were inherited from rorqual -- and one of them
#   (`--gpus=h100_3g.40gb:1`) does not error on a cluster that lacks it:
#   ⛔ IT QUEUES FOREVER, which is indistinguishable from a busy scheduler.
#   The transcription step IS the defect. So the probe now writes the values
#   itself, and sbatch/narval/narval_env.sh REFUSES TO RUN without the file.
#   There is nothing to transcribe, so nothing can be transcribed wrongly.
#
# ⛔ IT WRITES A KEY ONLY WHEN IT MEASURED IT. A fact it could not determine is
#   OMITTED, and the env file then refuses and names this script. FIR_SETUP Law 3:
#   fail closed. An absent value stops the chain; a guessed one spends an
#   allocation and looks like a scheduler problem.
#
# ⚠ WHAT IT CANNOT ANSWER, BY CONSTRUCTION: how much memory the GPU has. A login
#   node has no GPU. That is measured later, on a GPU node, by
#   sbatch/narval/03b_probe_gpu_memory.sh. This script does not invent it from the
#   model name.
# ============================================================================
set -uo pipefail
NARVAL_SELF="$(readlink -f "$0")"
cd "$(dirname "$NARVAL_SELF")/../.." || exit 1

OUT="sbatch/narval/measured.sh"
TMP="$(mktemp)"                 # built up as we measure; moved into place at the end
mkdir -p ./logs
LOG="./logs/narval_probe_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee "$LOG") 2>&1

# ⚠ $USER is NOT set in every shell this must survive, and `set -u` makes that
#   FATAL rather than empty. It fired in the local gates, on the very first run.
NARVAL_USER="${USER:-$(id -un 2>/dev/null || echo unknown)}"

hr() { echo; echo "=================== $* ==================="; }
FOUND=""; MISSING=""
put() {   # put KEY VALUE   -- record a MEASURED fact
    printf '%s=%q\n' "$1" "$2" >> "$TMP"; FOUND="$FOUND $1"
    printf '  ✅ %-24s = %s\n' "$1" "$2"
}
nomeasure() {  # nomeasure KEY WHY  -- record that we could NOT measure it
    MISSING="$MISSING $1"
    printf '  ⛔ %-24s NOT MEASURED: %s\n' "$1" "$2"
}

echo "############ NARVAL PROBE — $(date -u +%FT%TZ) ############"
echo "This changes nothing except $OUT."

# ---------------------------------------------------------------------------
hr "0. PROVENANCE — which code produced this transcript"
# FIR_SETUP Law 12: a log that cannot identify its own code is not evidence about
# that code. A fix pushed to the wrong ref once produced output byte-identical to
# the broken version, and diagnosing it cost a full round trip.
echo "commit : $(git rev-parse --short HEAD 2>/dev/null || echo '<not a git checkout>') on $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
echo "uncommitted files: $(git status --porcelain 2>/dev/null | wc -l)"
echo "transcript: $LOG"

# ---------------------------------------------------------------------------
hr "1. IDENTITY / SITE"
echo "hostname       : $(hostname -f 2>/dev/null || hostname)"
echo "user           : $NARVAL_USER"
echo "CC_CLUSTER     : ${CC_CLUSTER:-<unset>}"
echo "CC_ARCH        : ${CC_ARCH:-<unset>}"
echo "RSNT_ARCH      : ${RSNT_ARCH:-<unset>}"
echo "cpu            : $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2-)"
echo "repo present   : $([ -d ./src ] && echo yes || echo 'NO — run from the repo root')"
[ -d ./src ] || { echo "⛔ not in the repo root. Nothing measured."; exit 1; }

# ⛔ REFUSE TO WRITE narval FACTS FROM A NON-narval HOST. Running this on fir would
#   produce a measured.sh full of fir values under narval names -- the inheritance
#   bug, automated. ⚠ CC_CLUSTER is the site's own variable; if it is unset we say
#   so and continue, because a probe that refuses to run cannot be used to find out
#   why it refuses.
if [ -n "${CC_CLUSTER:-}" ] && [ "${CC_CLUSTER}" != "narval" ]; then
    echo
    echo "⛔⛔ REFUSING: \$CC_CLUSTER says this host is '${CC_CLUSTER}', not narval."
    echo "   Writing $OUT from here would record ${CC_CLUSTER}'s values under narval's"
    echo "   names -- which is exactly the cross-cluster inheritance this tree exists"
    echo "   to prevent. Run this on a narval login node."
    exit 1
fi
if [ -z "${CC_CLUSTER:-}" ]; then
    echo "  ⚠ \$CC_CLUSTER is unset. Cannot confirm this is narval; continuing, but"
    echo "    check the hostname above before trusting $OUT."
fi
put NARVAL_PROBE_UTC  "$(date -u +%FT%TZ)"
put NARVAL_PROBE_HOST "$(hostname -f 2>/dev/null || hostname)"

# ---------------------------------------------------------------------------
hr "2. MODULES — WHICH ONES EXIST, AND DOES THE ORDER WORK"
# ⚠ ORDER IS LOAD-BEARING on Alliance clusters: `module avail cudnn` returns
#   "No module(s) found" on its own, because cudnn is CUDA-dependent in the Lmod
#   hierarchy and only becomes visible AFTER cuda is loaded. Any check that
#   probes cudnn first concludes it is absent. So we TRY THE LINE, in order, and
#   record it only if it loads.
if ! command -v module >/dev/null 2>&1; then
    nomeasure NARVAL_MODULES_CPU "no Lmod on this host"
    nomeasure NARVAL_MODULES_GPU "no Lmod on this host"
    # ⚠ AND THE PYTHON VERSION, which is only derivable from a loaded module line.
    #   Omitting it here made the closing "Missing:" list UNDER-REPORT: the key was
    #   simply absent from measured.sh and the failure surfaced one stage later,
    #   against the env file, instead of in the transcript that gets sent back.
    #   A status line that lies is worse than none.
    nomeasure NARVAL_PYTHON_VERSION "no Lmod on this host — cannot resolve the module python"
else
    echo "--- loaded by default, before we touch anything ---"
    module list 2>&1 | head -20

    # ⛔ A CHECK MUST RUN WHAT THE JOB RUNS (FIR_SETUP Law 1). `module load` exiting
    #   0 is NOT the property we need -- the property is that the line leaves a
    #   usable `python` on PATH, which is what every stage then calls. Lmod can also
    #   report success while a hierarchy dependency silently supplied nothing.
    # ⚠ `module purge` first, so a line is judged on its own and not on what the
    #   site happened to load by default. That is also why "StdEnv/2023 ..." is a
    #   candidate: after a purge, `gcc` may not be visible without it.
    try_line() {   # try_line "<modules>"  -> 0 if the line loads AND yields python
        ( module purge >/dev/null 2>&1
          module load $1 >/dev/null 2>&1 || exit 1
          command -v python >/dev/null 2>&1 || exit 1
          python -c 'import sys' >/dev/null 2>&1 ) >/dev/null 2>&1
    }
    # Candidates in preference order. The FIRST that loads wins; we record what
    # actually worked, never what we expected to work.
    CPU_OK=""; GPU_OK=""
    for line in "gcc arrow scipy-stack" "StdEnv/2023 gcc arrow scipy-stack" "arrow scipy-stack" "scipy-stack"; do
        echo "--- trying CPU line: module load $line"
        if try_line "$line"; then echo "    ✅ loads"; CPU_OK="$line"; break; else echo "    ⛔ fails"; fi
    done
    for line in "gcc arrow scipy-stack cuda cudnn" "StdEnv/2023 gcc arrow scipy-stack cuda cudnn" \
                "gcc arrow scipy-stack cuda" "arrow scipy-stack cuda cudnn"; do
        echo "--- trying GPU line: module load $line"
        if try_line "$line"; then echo "    ✅ loads"; GPU_OK="$line"; break; else echo "    ⛔ fails"; fi
    done
    [ -n "$CPU_OK" ] && put NARVAL_MODULES_CPU "$CPU_OK" || nomeasure NARVAL_MODULES_CPU "no candidate CPU module line loaded — see the attempts above"
    [ -n "$GPU_OK" ] && put NARVAL_MODULES_GPU "$GPU_OK" || nomeasure NARVAL_MODULES_GPU "no candidate GPU module line loaded — see the attempts above"

    if [ -n "$GPU_OK" ]; then
        echo "--- what that line actually loads, and the python it gives ---"
        ( module purge >/dev/null 2>&1; module load $GPU_OK >/dev/null 2>&1
          module list 2>&1 | head -20
          echo "  which python : $(command -v python || echo none)"
          echo "  python -V    : $(python -V 2>&1)"
          python -c "import numpy;print('  numpy',numpy.__version__,'from',numpy.__file__)" 2>&1 | head -2 )
        PYV=$( module purge >/dev/null 2>&1; module load $GPU_OK >/dev/null 2>&1; python -V 2>&1 | awk '{print $2}' )
        # ⚠ THE PYTHON VERSION DECIDES THE WHEEL ABI. fir is 3.11.5 and the dev box
        #   3.10.12; a venv is not portable across that boundary.
        [ -n "$PYV" ] && put NARVAL_PYTHON_VERSION "$PYV" || nomeasure NARVAL_PYTHON_VERSION "module line loaded but python -V produced nothing"
    else
        nomeasure NARVAL_PYTHON_VERSION "no GPU module line loaded"
    fi
fi

# ---------------------------------------------------------------------------
hr "3. GPUS THE SCHEDULER WILL ACCEPT — the value that QUEUES FOREVER if wrong"
# ⛔ FIR_SETUP A1. This is THE value to get right, and the only one whose failure
#   mode is silence. We read it from the scheduler itself and never from a table.
echo "--- partitions and their Gres ---"
sinfo -o "%P %.6D %G %N" 2>&1 | head -40
echo
echo "--- distinct Gres strings on this cluster ---"
sinfo -h -o "%G" 2>&1 | tr ',' '\n' | sed 's/(.*//' | sed 's/^gpu://' | sort -u | grep -v '^$' | head -30
echo
# The gres TYPE names, e.g. `a100`, `a100_3g.20gb`. Pick the plain (non-MIG) one:
# a MIG slice has a `_Ng.` in its name and is far too small for stage 06 anyway.
GRES_TYPES=$(sinfo -h -o "%G" 2>&1 | tr ',' '\n' | sed 's/(.*//' \
             | grep '^gpu:' | sed 's/^gpu://' | sed 's/:[0-9]*$//' | sort -u | grep -v '^$')
echo "--- candidate GPU type names ---"; echo "$GRES_TYPES" | sed 's/^/    /'
FULL=$(echo "$GRES_TYPES" | grep -v '_[0-9]g\.' | grep -v '^[0-9]*$' | head -1)
if [ -n "$FULL" ]; then
    put NARVAL_GPU_FULL "${FULL}:1"
    put NARVAL_GPU_NAME "$FULL"
    echo "  ⚠ the MIG-looking types below are NOT recorded and nothing here submits to one:"
    echo "$GRES_TYPES" | grep '_[0-9]g\.' | sed 's/^/      /' || true
else
    nomeasure NARVAL_GPU_FULL "sinfo reported no plain (non-MIG) gpu type — read the Gres dump above"
    nomeasure NARVAL_GPU_NAME "as above"
fi
echo
echo "⛔ NOTE: the GPU's MEMORY is NOT measurable here (a login node has no GPU),"
echo "   and this probe does not infer it from the model name. It is measured by"
echo "   sbatch/narval/03b_probe_gpu_memory.sh, on a GPU node, after stage 02."

# ---------------------------------------------------------------------------
hr "4. ACCOUNTS — fir splits them _gpu/_cpu; rorqual did not. MEASURE IT."
echo "--- sacctmgr associations ---"
ACCTS=""
if command -v sacctmgr >/dev/null 2>&1; then
    sacctmgr -nP show user "$NARVAL_USER" withassoc format=Account 2>/dev/null | sort -u | tee /dev/stderr > /tmp/.n_accts.$$
    ACCTS=$(cat /tmp/.n_accts.$$ 2>/dev/null); rm -f /tmp/.n_accts.$$
fi
echo "--- sshare ---"
sshare -U -u "$NARVAL_USER" 2>&1 | head -15
A_GPU=$(echo "$ACCTS" | grep -- '_gpu$' | head -1)
A_CPU=$(echo "$ACCTS" | grep -- '_cpu$' | head -1)
if [ -n "$A_GPU" ]; then
    put NARVAL_ACCOUNT_GPU "$A_GPU"
else
    # No _gpu suffix anywhere => this cluster does not split accounts by resource.
    A_PLAIN=$(echo "$ACCTS" | grep -v '_gpu$\|_cpu$' | grep -v '^$' | head -1)
    if [ -n "$A_PLAIN" ]; then
        echo "  ⚠ no '_gpu' account exists; this cluster does not split accounts by resource."
        put NARVAL_ACCOUNT_GPU "$A_PLAIN"
    else
        nomeasure NARVAL_ACCOUNT_GPU "sacctmgr listed no account for $NARVAL_USER — read the dump above"
    fi
fi
if [ -n "$A_CPU" ]; then
    put NARVAL_ACCOUNT_CPU "$A_CPU"
else
    A_PLAIN=$(echo "$ACCTS" | grep -v '_gpu$\|_cpu$' | grep -v '^$' | head -1)
    [ -n "$A_PLAIN" ] && put NARVAL_ACCOUNT_CPU "$A_PLAIN" \
        || nomeasure NARVAL_ACCOUNT_CPU "no CPU account found"
fi
echo "  ⚠ if more than one account is listed above, CHECK the one recorded is the"
echo "    allocation you mean to spend. Re-run with NARVAL_FORCE_ACCOUNT=<name> to override."
if [ -n "${NARVAL_FORCE_ACCOUNT:-}" ]; then
    echo "  [override] NARVAL_FORCE_ACCOUNT=$NARVAL_FORCE_ACCOUNT"
    put NARVAL_ACCOUNT_GPU "$NARVAL_FORCE_ACCOUNT"
fi

# ---------------------------------------------------------------------------
hr "5. FILESYSTEMS — space AND INODES (the binding constraint on fir was inodes)"
# ⚠⚠ On fir, /project was at 486K of a 500K FILE quota. A venv is tens of thousands
#   of files and the HF cache tens of thousands more; materialising either there
#   puts the WHOLE ALLOCATION over quota, at which point every write on /project
#   fails for every job in it -- not just this repo's. Hence env/ data/ temp/ on
#   scratch, behind symlinks.
if command -v diskusage_report >/dev/null 2>&1; then diskusage_report 2>&1 | head -25
else df -h "$HOME" "${SCRATCH:-$HOME}" 2>&1 | head -10; fi
echo "HOME=$HOME  SCRATCH=${SCRATCH:-<unset>}  PROJECT=${PROJECT:-<unset>}"
if [ -n "${SCRATCH:-}" ] && [ -d "${SCRATCH}" ]; then
    put NARVAL_SCRATCH "$SCRATCH"
elif [ -d "/scratch/$NARVAL_USER" ]; then
    put NARVAL_SCRATCH "/scratch/$NARVAL_USER"
else
    nomeasure NARVAL_SCRATCH "\$SCRATCH is unset and /scratch/$NARVAL_USER does not exist"
fi
# ⚠ --mem is a per-job cap; too low is an OOM-kill, too high hurts scheduling. Take
#   a fraction of what a real node has rather than carrying fir's 64000M.
NODE_MEM=$(sinfo -h -o "%m" 2>/dev/null | sort -n | tail -1)
if [ -n "$NODE_MEM" ] && [ "$NODE_MEM" -gt 0 ] 2>/dev/null; then
    echo "  largest RealMemory on any node: ${NODE_MEM} MiB"
    # fir used 64000M against 80 GiB GPUs; keep the same shape but never exceed the node.
    WANT=64000; [ "$NODE_MEM" -lt 70000 ] && WANT=$((NODE_MEM / 4 * 3))
    put NARVAL_GPU_MEM "${WANT}M"
else
    nomeasure NARVAL_GPU_MEM "sinfo reported no RealMemory"
fi

# ---------------------------------------------------------------------------
hr "6. LUSTRE flock SEMANTICS (decides whether a shared-CSV writer is safe)"
# ⚠ `filelock` uses fcntl.flock. On Lustre mounted `-o localflock` the lock is
#   NODE-LOCAL: two array tasks on different nodes both "hold" it, interleave a
#   read-modify-write, and the second os.replace SILENTLY DISCARDS the first row.
#   Nothing raises. ⚠ RE-CHECK PER CLUSTER, never inherit.
#   ⭐ This repo writes ONE CSV PER CELL (the upsert key omits seed), so it does not
#     depend on the answer -- but it is a property of the cluster and is recorded.
for m in "${SCRATCH:-/scratch/$NARVAL_USER}" "$HOME" "$(readlink -f "$(pwd)")"; do
    echo "--- $m ---"
    findmnt -o TARGET,SOURCE,FSTYPE,OPTIONS --target "$m" 2>&1 | head -4
done

# ---------------------------------------------------------------------------
hr "7. INTERNET FROM THE LOGIN NODE (compute nodes have NO route)"
# ⚠ github.com is here because 01c CLONES the authors' LoCA and QWHA trees. On fir
#   this list originally tested only pypi and HF, so the one host a whole stage
#   depends on was never probed.
for host in pypi.org huggingface.co github.com; do
    printf "  %-20s " "$host"
    if command -v curl >/dev/null 2>&1; then
        echo "HTTP $(curl -s -o /dev/null -w '%{http_code}' --max-time 12 "https://$host" 2>/dev/null)"
    else echo "(no curl)"; fi
done

# ---------------------------------------------------------------------------
hr "8. THE WHEELHOUSE — can narval serve OUR pins? (00c settles it; this is a look)"
# ⚠ `avail_wheels` shows only the DEFAULTS, so this section proves NOTHING on its
#   own and looks alarming when the defaults are newer than our pins -- which they
#   were on fir. ⛔ Only `pip install --dry-run` can conclude: that is stage 00c.
# ⚠⚠ narval is an AMD (EPYC) site and fir is too, but the gentoo micro-arch tier
#   (x86-64-v3 vs v4) selects a DIFFERENT wheelhouse path. A wheel present on fir
#   is NOT automatically present here. 00c is not optional.
if command -v avail_wheels >/dev/null 2>&1; then
    for pkg in torch triton transformers peft datasets accelerate evaluate \
               adapters galore_torch lion_pytorch bitsandbytes sentencepiece; do
        echo "--- avail_wheels $pkg ---"; avail_wheels "$pkg" 2>&1 | head -8; echo
    done
else
    echo "!!! avail_wheels not found — report this; it changes the venv strategy entirely"
fi
echo "PIP_CONFIG_FILE : ${PIP_CONFIG_FILE:-<unset>}"
[ -n "${PIP_CONFIG_FILE:-}" ] && [ -f "${PIP_CONFIG_FILE}" ] && sed 's/^/    /' "${PIP_CONFIG_FILE}"

# ---------------------------------------------------------------------------
hr "9. HUGGINGFACE TOKEN + GATED-MODEL ACCESS (google/gemma-2b)"
# ⚠⚠ THE TOKEN TRAP: setting HF_HOME ALSO RELOCATES where huggingface_hub reads the
#   token ($HF_HOME/token). An access check in a plain shell PASSES and the identical
#   download under the job's own environment FAILS with "Access to model ... is
#   restricted" -- FIR_SETUP Law 1 verbatim. So report BOTH locations.
echo "  HF_TOKEN env               : ${HF_TOKEN:+present}${HF_TOKEN:-<unset>}"
echo "  ./.hf_token (searched 1st) : $([ -s ./.hf_token ] && echo present || echo ABSENT)"
echo "  ~/.cache/huggingface/token : $([ -s "$HOME/.cache/huggingface/token" ] && echo present || echo ABSENT)"
echo "  ./data/token (HF_HOME)     : $([ -s ./data/token ] && echo present || echo ABSENT)"
TOK="${HF_TOKEN:-$(cat ./.hf_token 2>/dev/null || cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"
TOK="$(printf '%s' "$TOK" | tr -d '[:space:]')"
if [ -n "$TOK" ] && command -v curl >/dev/null 2>&1; then
    echo "  GET gemma-2b config.json with token -> HTTP $(curl -s -o /dev/null -w '%{http_code}' --max-time 15 \
         -H "Authorization: Bearer $TOK" "https://huggingface.co/google/gemma-2b/resolve/main/config.json" 2>/dev/null)"
    echo "     (200 = access granted; 401/403 = accept the licence at https://huggingface.co/google/gemma-2b)"
else
    echo "  ⛔ no token on this node -> 02_download_cache.sh CANNOT fetch the gated model."
fi

# ---------------------------------------------------------------------------
hr "10. WRITING $OUT"
if [ -n "$MISSING" ]; then
    echo "⛔⛔ NOT ALL REQUIRED FACTS WERE MEASURED. Missing:$MISSING"
    echo
    echo "   $OUT is written anyway, with ONLY the keys that were measured, so"
    echo "   re-running the probe after fixing the cause is additive. Every stage"
    echo "   will refuse until the missing keys appear -- which is the point:"
    echo "   a stage that refuses costs a login-node second, a stage that runs on a"
    echo "   guessed value costs an allocation and looks like a busy scheduler."
fi
# ⭐ Which measured values coincide with fir's defaults? Compare, do not assume:
#   the module lines very probably DO match (both are StdEnv/2023 Alliance sites)
#   while the gres string certainly does not.
ALLOW_SAME=""
fir_default_of() { sed -n "s/^$1=\"\${$1:-\([^}]*\)}\".*/\1/p" sbatch/fir/fir_env.sh | head -1; }
_same_check() {   # _same_check <NARVAL_KEY> <FIR_VAR>
    local nv fv
    eval "nv=\${$1:-}"
    fv="$(fir_default_of "$2")"
    [ -n "$nv" ] && [ "$nv" = "$fv" ] && ALLOW_SAME="$ALLOW_SAME $2"
    return 0
}
# ⚠ the values are in $TMP (KEY=value, one per line), not in this shell. Source them.
# shellcheck disable=SC1090
. "$TMP" 2>/dev/null || true
_same_check NARVAL_MODULES_CPU  FIR_MODULES_CPU
_same_check NARVAL_MODULES_GPU  FIR_MODULES_GPU
_same_check NARVAL_GPU_FULL     FIR_GPU_FULL
_same_check NARVAL_GPU_MEM      FIR_GPU_MEM
_same_check NARVAL_ACCOUNT_GPU  FIR_ACCOUNT_GPU
_same_check NARVAL_ACCOUNT_CPU  FIR_ACCOUNT_CPU
ALLOW_SAME="${ALLOW_SAME# }"
[ -n "$ALLOW_SAME" ] && echo "  ⚠ measured values that MATCH fir's defaults (declared, not assumed):$ALLOW_SAME"

{
    echo "# sbatch/narval/measured.sh — MEASURED ON NARVAL. DO NOT EDIT BY HAND."
    echo "#"
    echo "# ⛔ Every line below was written by sbatch/narval/00_probe_narval.sh from"
    echo "#   something it actually observed on this cluster. If a value is wrong, the"
    echo "#   fix is to fix the PROBE and re-run it -- not to edit this file. A"
    echo "#   hand-edited value here is indistinguishable from a measured one, and"
    echo "#   that indistinguishability is the whole defect this file exists to remove."
    echo "#"
    echo "# written : $(date -u +%FT%TZ)"
    echo "# host    : $(hostname -f 2>/dev/null || hostname)"
    echo "# commit  : $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
    echo "# probe log: $LOG"
    echo
    cat "$TMP"
    echo
    echo "# ⭐ Values narval genuinely SHARES with fir, computed by comparing each"
    echo "#   MEASUREMENT above against the default written into sbatch/fir/fir_env.sh."
    echo "#   narval_env.sh's audit refuses any FIR_* variable that still holds fir's"
    echo "#   value, because that is what a forgotten binding looks like -- and a wrong"
    echo "#   gres string QUEUES FOREVER instead of erroring. This list is what lets it"
    echo "#   tell 'measured and identical' apart from 'never set'."
    echo "#   ⛔ It is DERIVED FROM A MEASUREMENT. Adding a name here by hand is exactly"
    echo "#     the exemption that would let an unmeasured value through."
    echo "NARVAL_ALLOW_SAME=\"$ALLOW_SAME\""
} > "$OUT"
rm -f "$TMP"
echo
echo "--- $OUT ---"
sed 's/^/    /' "$OUT"

hr "11. WHAT TO DO NEXT"
if [ -n "$MISSING" ]; then
    echo "⛔ STOP. Send back this transcript ($LOG)."
    echo "   Missing:$MISSING"
    exit 1
fi
cat <<'NEXT'
✅ Every required fact was measured. In order, and DO NOT SKIP A STAGE
   (every stage-skip in this lineage's history cost a GPU allocation):

     bash sbatch/narval/00c_probe_deps.sh      # do the pins resolve HERE? read-only
     bash sbatch/narval/01_setup_venv.sh
     bash sbatch/narval/01c_stage_repos.sh
     bash sbatch/narval/02_download_cache.sh
     bash sbatch/narval/03_preflight.sh        # 1 GPU job; wait for PREFLIGHT OK
     bash sbatch/narval/03b_probe_gpu_memory.sh # 1 GPU job; measures if stage 06 FITS

   ⛔ Stage 06 is the ONLY science left to run. Stages 04 and 05 are COMPLETE on
     fir (656+8 search cells, 270 final cells) and MUST NOT be re-run --
     sbatch/narval/README.md holds the ledger and the narval tree ships no 04/05.
NEXT
echo
echo "############ PROBE COMPLETE — transcript: $LOG ############"
