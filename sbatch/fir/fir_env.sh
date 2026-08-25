#!/bin/bash
# =============================================================================
# sbatch/fir/fir_env.sh — THE SINGLE SOURCE OF TRUTH for running THIS repo
# (LYRA / lora_research_signal) on fir (Alliance Canada).
#
# Nothing else in sbatch/fir/ defines a module line, an offline flag, a GPU
# string, an account, a path or a pin. Every value here is either MEASURED by
# 00_probe_fir.sh or inherited from llmdocs/FIR_SETUP.md, which is the compiled
# record of ~40 defects paid for on this cluster by ../CompAct.
#
# ⛔ READ llmdocs/FIR_SETUP.md BEFORE EDITING THIS FILE. Several lines below look
#    redundant or untidy and are load-bearing; each carries the incident that
#    bought it.
#
# ⚠ THIS REPO IS NOT CompAct. Differences that matter:
#     * the runner is `src/train_glue.py` (a single-process trainer), NOT
#       `run_production.py`; there is no fused block, no DeepSpeed/ALST/StreamBP,
#       and therefore none of CompAct's F2/F4/G26 distributed hazards.
#     * `temp/` holds ONLY the two authors' clones used by the bit-identity
#       VERIFIERS (LoCA, QWHA). Training never reaches into temp/. See §TEMP.
#     * the results CSV writer is `train_glue._upsert_result`, whose key OMITS
#       `seed` — one CSV per (config, seed), seed in the FILENAME. Gated by
#       scripts/r304_upsert_gate.py. Do not point two cells at one CSV.
# =============================================================================

# --- MODULES -----------------------------------------------------------------
# ⚠ ORDER IS LOAD-BEARING. `module avail cudnn` returns "No module(s) found" on
#   its own: cudnn is CUDA-dependent in the Lmod hierarchy and only becomes
#   visible AFTER cuda is loaded. Any reorder putting cudnn first fails.
#   Loading gcc/arrow/scipy-stack explicitly is a no-op today (fir loads them by
#   default) and protects against a future default change. DO NOT TIDY.
FIR_MODULES_CPU="${FIR_MODULES_CPU:-gcc arrow scipy-stack}"
FIR_MODULES_GPU="${FIR_MODULES_GPU:-gcc arrow scipy-stack cuda cudnn}"

# --- GPUS / ACCOUNTS ---------------------------------------------------------
# ⚠⚠ A WRONG GRES STRING QUEUES FOREVER; IT DOES NOT ERROR (FIR_SETUP A1).
#    Every MIG slice on fir lives on `gpubase_interac` (one node, fc11020).
#    A BATCH job must request a FULL H100 — batch partitions carry h100:4 only.
FIR_GPU_FULL="${FIR_GPU_FULL:-h100:1}"                              # batch: use this
FIR_GPU_MIG40="${FIR_GPU_MIG40:-nvidia_h100_80gb_hbm3_3g.40gb:1}"   # INTERACTIVE ONLY
FIR_GPU_MEM="${FIR_GPU_MEM:-64000M}"

# ⚠ fir splits the account by resource; rorqual did not. Wrong name => rejected
#   at submit. The CPU-only report/aggregation job MUST use the CPU account.
FIR_ACCOUNT_GPU="${FIR_ACCOUNT_GPU:-def-seokbum_gpu}"
FIR_ACCOUNT_CPU="${FIR_ACCOUNT_CPU:-def-seokbum_cpu}"

# --- PATHS -------------------------------------------------------------------
# ⚠⚠ THE BINDING CONSTRAINT ON /project IS INODES, NOT SPACE (486K / 500K files
#    when last measured). A venv is tens of thousands of files and the HF cache
#    tens of thousands more. Materialising either on /project puts the ALLOCATION
#    over its file quota, at which point EVERY WRITE ON /project FAILS FOR EVERY
#    JOB IN THE ALLOCATION — not just this repo's.
#    => code on /project; env/, data/, temp/ on /scratch behind SYMLINKS, so
#       every existing `./env/bin/python` and `$(pwd)/data` reference in the repo
#       keeps working with no edit.
# ⚠⚠ DERIVED FROM THE REPO DIRECTORY, NOT HARDCODED.
#    This was literally `.../lora_research_signal` while the checkout on fir is
#    named `SCoRA` [observed 2026-08-25]. Harmless on its own -- but the SCoRA
#    branch carries LYRA's full history, so BOTH repos can plausibly be checked
#    out on the same account, and with a hardcoded name they would SHARE ONE
#    /scratch root: one venv, one HF cache, one runs/ directory, silently.
#    Two different experiments writing into one environment is precisely the
#    class of collision this file exists to prevent, and it fails silently.
#    Override with FIR_SCRATCH_ROOT if you deliberately want them shared.
FIR_REPO_NAME="${FIR_REPO_NAME:-$(basename "$(readlink -f "$(pwd)")")}"
FIR_SCRATCH_ROOT="${FIR_SCRATCH_ROOT:-${SCRATCH:-/scratch/$USER}/$FIR_REPO_NAME}"
FIR_VENV_REAL="${FIR_VENV_REAL:-$FIR_SCRATCH_ROOT/env}"
FIR_DATA_REAL="${FIR_DATA_REAL:-$FIR_SCRATCH_ROOT/data}"
FIR_TEMP_REAL="${FIR_TEMP_REAL:-$FIR_SCRATCH_ROOT/temp}"
FIR_VENV="${FIR_VENV:-./env}"                       # symlink -> $FIR_VENV_REAL
FIR_DATA="${FIR_DATA:-$(pwd)/data}"                 # symlink -> $FIR_DATA_REAL
FIR_TEMP="${FIR_TEMP:-$(pwd)/temp}"                 # symlink -> $FIR_TEMP_REAL

# Where every fir run of this repo writes its per-cell CSVs and markers.
FIR_RUN_ROOT="${FIR_RUN_ROOT:-$FIR_SCRATCH_ROOT/runs}"

fir_link_scratch() {
    # idempotent; REFUSES to clobber a real directory rather than deleting one.
    mkdir -p "$FIR_VENV_REAL" "$FIR_DATA_REAL" "$FIR_TEMP_REAL" "$FIR_RUN_ROOT" || return 1
    local pair link target
    for pair in "./env:$FIR_VENV_REAL" "./data:$FIR_DATA_REAL" "./temp:$FIR_TEMP_REAL"; do
        link="${pair%%:*}"; target="${pair#*:}"
        if [ -L "$link" ]; then
            [ "$(readlink -f "$link")" = "$(readlink -f "$target")" ] || {
                echo "FAIL: $link is a symlink pointing somewhere else:"
                echo "      $(readlink -f "$link")  !=  $(readlink -f "$target")"; return 1; }
        elif [ -e "$link" ]; then
            echo "FAIL: $link exists and is NOT a symlink."
            echo "      Move it aside by hand — refusing to delete data."; return 1
        else
            ln -s "$target" "$link" || return 1
        fi
        # ⚠ `find ./temp -type f` reports 0 here: find does NOT follow a symlink
        #   ARGUMENT, and ./temp is one, so it matches only the link (type l).
        #   Use `find -L`. A status line that lies is worse than none (FIR_SETUP B4).
        echo "  $link -> $(readlink -f "$link")  ($(find -L "$link" -type f 2>/dev/null | wc -l) files)"
    done
}

# --- PINS --------------------------------------------------------------------
# ⚠⚠ USER DECISION 2026-08-25: fir runs the FIR-NATIVE stack (the one ../CompAct
#    proved installs on fir's python 3.11 wheelhouse), NOT this repo's dev-box
#    stack (torch 2.5.1+cu121 / transformers 4.45.2 / peft 0.13.2).
#
# ⛔⛔ THE PRICE OF THAT DECISION, RECORDED SO IT IS NEVER FORGOTTEN:
#    `src/qwha_adapter.py:14` — every FourierFT number in this repo is gated as
#    BIT-IDENTICAL to the INSTALLED `peft.tuners.fourierft`. peft 0.13.2 -> 0.18.1
#    can change that layer. => 03_preflight.sh RE-RUNS the bit-identity verifiers
#    ON FIR and a failure is LOUD. If they fail, the fir comparator is a DIFFERENT
#    comparator from every dev-box number, and that fact must travel with any table
#    quoting both. This is not a formality: it is the one scientific risk the pin
#    choice buys.
#
# Record the BUILD, not just the version: fir installs `2.10.0+computecanada`,
# the dev box measured `2.10.0+cu128`. Same upstream version, DIFFERENT COMPILE.
# Peak memory is allocator- and kernel-sensitive => fir numbers are internally
# comparable and must NOT be quoted interchangeably with dev-box tables.
FIR_PIN_TORCH="${FIR_PIN_TORCH:-2.10.0}"
FIR_PIN_TRANSFORMERS="${FIR_PIN_TRANSFORMERS:-4.51.3}"
FIR_PIN_DATASETS="${FIR_PIN_DATASETS:-4.5.0}"
FIR_PIN_PEFT="${FIR_PIN_PEFT:-0.18.1}"
FIR_PIN_ACCELERATE="${FIR_PIN_ACCELERATE:-1.12.0}"
FIR_PIN_EVALUATE="${FIR_PIN_EVALUATE:-0.4.6}"

# --- THE BACKBONE ------------------------------------------------------------
# USER DECISION 2026-08-25: google/gemma-2b.
# ⚠ GATED REPO. See fir_export_online() for the token trap.
# ⚠ MQA: num_attention_heads 8, head_dim 256, num_key_value_heads 1, hidden 2048
#   => q_proj/o_proj are 2048x2048 but k_proj/v_proj are 256x2048. This is the
#   [phase-m2] GQA init defect verbatim; see llmdocs/FIR_GEMMA_TARGETING.md for
#   the measured per-module perturbation and the derived scaling.
FIR_MODEL="${FIR_MODEL:-google/gemma-2b}"

fir_load_modules_cpu() { module load $FIR_MODULES_CPU; }
fir_load_modules_gpu() { module load $FIR_MODULES_GPU; }

# --- LOGGING -----------------------------------------------------------------
# NOT a convenience. A fir diagnosis has already had to be made from hand-pasted
# scrollback TRUNCATED MID-LINE at exactly the point the outcome would have
# appeared, because the `2>&1 | tee` convention in a usage header was not used.
# A script that only prints is a script whose failure cannot be handed to anyone.
#
# Usage — the caller MUST resolve FIR_SELF BEFORE its `cd` ($0 is relative to the
# invocation directory and this re-execs from the repo root):
#     FIR_SELF="$(readlink -f "$0")"
#     cd "$(dirname "$FIR_SELF")/../.." || exit 1
#     source sbatch/fir/fir_env.sh
#     fir_log_to <tag> "$@"
fir_log_to() {
    [ -n "${FIR_LOGGING:-}" ] && return 0
    local tag="$1"; shift
    [ -n "${FIR_SELF:-}" ] || { echo "fir_log_to: FIR_SELF unset — NOT logging"; return 0; }
    mkdir -p ./logs
    local f="./logs/${tag}_$(date -u +%Y%m%dT%H%M%SZ).log"
    export FIR_LOGGING=1
    echo "### transcript -> $f"
    # ⚠⚠ RE-EXEC THROUGH bash, NOT THE PATH DIRECTLY.
    #    `"$FIR_SELF" "$@"` requires the EXECUTE BIT, but every usage line in this
    #    tree says `bash sbatch/fir/<script>.sh`, which does not. So the mode bit
    #    silently became load-bearing: 02_download_cache.sh and 03_preflight.sh
    #    were committed 100644 and 02 died `Permission denied` (exit 126) on fir
    #    2026-08-25 -- AFTER the user had correctly invoked it with `bash`.
    #    Re-execing through bash makes the bit irrelevant and matches the
    #    documented invocation. (The local test that "verified" fir_log_to used
    #    chmod +x on its fixture, so it could only ever exercise the executable
    #    case -- a test that cannot see the failure it is meant to catch.)
    bash "$FIR_SELF" "$@" 2>&1 | tee "$f"
    local rc=${PIPESTATUS[0]}          # ⚠ preserve the SCRIPT's status, not tee's
    echo "### $tag exit=$rc" | tee -a "$f"
    echo "### transcript: $f"
    exit $rc
}

# --- CACHES A LIBRARY WILL OTHERWISE PUT ON $HOME ----------------------------
# /home is 48 GiB and quota-bound, and it is a network filesystem. triton's JIT
# cache defaulting to $HOME/.triton produced `OSError: [Errno 5] Input/output
# error` four times in one fir job. virtualenv's app-data cache defaults there too.
fir_export_caches() {
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$FIR_SCRATCH_ROOT/.triton}"
    export VIRTUALENV_OVERRIDE_APP_DATA="${VIRTUALENV_OVERRIDE_APP_DATA:-$FIR_SCRATCH_ROOT/.virtualenv_appdata}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$FIR_SCRATCH_ROOT/.cache}"
    mkdir -p "$TRITON_CACHE_DIR" "$VIRTUALENV_OVERRIDE_APP_DATA" "$XDG_CACHE_HOME"
}

# --- ONLINE (LOGIN NODE ONLY) ------------------------------------------------
# ⚠⚠ THE TOKEN TRAP, MEASURED ON THE DEV BOX 2026-08-25 AND IT WILL FIRE HERE.
#    Setting HF_HOME ALSO RELOCATES WHERE huggingface_hub READS THE TOKEN
#    ($HF_HOME/token). google/gemma-2b is a GATED repo. An access check run in a
#    plain shell PASSES (it reads ~/.cache/huggingface/token) and the identical
#    download then FAILS under fir_export_online with
#        "Access to model google/gemma-2b is restricted ... Please log in."
#    — i.e. exactly FIR_SETUP Law 1: a check that did not run what the job runs.
#    => resolve the token from the DEFAULT location and export it EXPLICITLY.
fir_export_online() {
    export HF_HOME="$FIR_DATA"
    export TORCH_HOME="$FIR_DATA"
    export HF_HUB_DISABLE_XET=1     # the xet backend has produced stalled/partial pulls here
    if [ -z "${HF_TOKEN:-}" ]; then
        local t
        # ⚠ `./.hf_token` (repo root) is FIRST because it is where this project
        #   actually keeps it. It was added 2026-08-25 after the token was copied
        #   there and this function would not have found it -- stage 02 would have
        #   reported "token ABSENT" while the credential sat in the working dir.
        #   ⛔ It is gitignored (.gitignore); NEVER commit it.
        for t in "./.hf_token" "$HOME/.cache/huggingface/token" "$HF_HOME/token"; do
            [ -s "$t" ] && { HF_TOKEN="$(tr -d '[:space:]' < "$t")"; export HF_TOKEN; break; }
        done
    fi
    if [ -n "${HF_TOKEN:-}" ]; then
        export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"    # older hub versions read this name
        echo "  HF token: present (${#HF_TOKEN} chars)"
    else
        echo "  ⚠ HF token: ABSENT — a GATED repo ($FIR_MODEL) CANNOT be downloaded."
        echo "    Put it at ~/.cache/huggingface/token or export HF_TOKEN before 02."
    fi
    fir_export_caches
    mkdir -p "$HF_HOME"
}

# --- OFFLINE (COMPUTE NODE) --------------------------------------------------
# Compute nodes have NO ROUTE TO THE INTERNET.
fir_export_offline() {
    export HF_HOME="$FIR_DATA"
    export TORCH_HOME="$FIR_DATA"
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    # ⚠⚠ MANDATORY AND NOT REDUNDANT: `evaluate` IGNORES HF_HUB_OFFLINE. Without
    #    this, evaluate.load() probes a Hub it cannot reach and STALLS ~44 MINUTES
    #    PER SEED. A cold cache offline does not fail fast — it HANGS. This single
    #    line has cost the sibling project more compute than anything else.
    export HF_EVALUATE_OFFLINE=1
    # torch >= 2.9 renamed this and warns on every process start; older torch knows
    # only the old name. Set BOTH so the setting applies and the logs stay clean.
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    fir_export_caches
    # ⚠⚠ APPEND, NEVER ASSIGN. numpy is supplied by the scipy-stack MODULE, which
    #    puts it on PYTHONPATH. `PYTHONPATH=/some/dir` DELETES NUMPY on Alliance
    #    clusters. This never reproduces on a dev box, where numpy is in
    #    site-packages. Applies at RUN time too, inside job scripts.
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
    mkdir -p "$HF_HOME"
}

# --- THE GATE ----------------------------------------------------------------
# Fail on the LOGIN NODE, not 40 minutes into an allocation.
#   fir_assert_env <cpu|gpu> [stage]
#     cpu / gpu   — whether a CUDA check is legal on this node type
#     stage       — the stage that has JUST COMPLETED: 01 | 01c | 02 | all
#                   (default `all`: every check runs)
#
# ⛔⛔ WHY THE `stage` ARGUMENT EXISTS — FIR_SETUP C15, WHICH I REPRODUCED.
#    Some checks here need an artifact that a LATER stage creates:
#        the temp/ clones      are made by 01c
#        the offline HF cache  is made by 02
#    ...but this gate is 01's closing gate. Run unconditionally, it demands
#    artifacts that cannot exist yet, so a COMPLETELY CORRECT FRESH SETUP CAN
#    NEVER PASS ITS OWN GATE and always ends "SETUP INCOMPLETE".
#
#    That happened on fir 2026-08-25. The `temp/` half had been guarded by an
#    ad-hoc FIR_ASSERT_SKIP_TEMP flag; the offline-cache half had not, because
#    the first instance was fixed and the CLASS was never enumerated -- the exact
#    mistake FIR_SETUP Law 4 is about ("enumerate a control across every arm the
#    moment it fails on one").
#
#    ⇒ Ad-hoc per-check flags are the bug. EVERY check now DECLARES the stage its
#      precondition comes from, via `_need <stage>`, and a check whose stage has
#      not run yet is SKIPPED WITH ITS REASON rather than failing. Adding a new
#      check forces you to declare a stage; you cannot silently omit one.
#      ⛔ Nothing reaches a GPU unchecked: 03_preflight calls this with no stage
#        argument, so every check is enforced there.
_fir_stage_num() {
    case "$1" in
        01)  echo 1 ;;
        01c) echo 2 ;;
        02)  echo 3 ;;
        all|03|"") echo 99 ;;
        *)   echo "fir_assert_env: unknown stage '$1'" >&2; echo -1 ;;
    esac
}

fir_assert_env() {
    local want="${1:-gpu}" stage="${2:-all}" rc=0
    local _have; _have="$(_fir_stage_num "$stage")"
    [ "$_have" -lt 0 ] && return 1
    echo "--- fir_assert_env ($want, after stage $stage) ---"
    # _need <stage> -> 0 if that stage has run (so the check should execute)
    _need() { local n; n="$(_fir_stage_num "$1")"; [ "$_have" -ge "$n" ]; }
    [ -d ./src ] || { echo "FAIL: not in repo root (no ./src)"; return 1; }
    [ -x "$FIR_VENV/bin/python" ] || { echo "FAIL: no venv at $FIR_VENV — run 01_setup_venv.sh"; return 1; }

    # (a) FLOOR CHECK ONLY. The real check is (b). ⛔ DO NOT GROW THIS LIST — a
    #     hand-curated set of "core" packages cannot track train_glue.py's
    #     module-scope imports, and curating it IS the bug (FIR_SETUP C11).
    "$FIR_VENV/bin/python" - <<PY || rc=1
import importlib, sys
bad = []
for m in ["numpy", "torch", "transformers", "peft", "datasets", "accelerate",
          "evaluate", "pandas", "filelock", "sklearn", "scipy"]:
    try:
        importlib.import_module(m)
    except Exception as e:
        # ⚠ PRINT THE MESSAGE, NOT JUST THE TYPE. "peft (RuntimeError)" is
        #   unactionable and cost a round trip on fir.
        bad.append(m); print(f"  IMPORT FAILED {m}: {type(e).__name__}: {str(e)[:400]}")
print("  floor imports:", "OK" if not bad else "MISSING -> " + ", ".join(bad))
if bad: sys.exit(1)
import torch, transformers, datasets, peft
pins = {"torch": ("$FIR_PIN_TORCH", torch.__version__),
        "transformers": ("$FIR_PIN_TRANSFORMERS", transformers.__version__),
        "datasets": ("$FIR_PIN_DATASETS", datasets.__version__),
        "peft": ("$FIR_PIN_PEFT", peft.__version__)}
drift = {k: v for k, v in pins.items() if v[0] and not v[1].startswith(v[0])}
for k, (w, g) in drift.items():
    print(f"  ⚠⚠ {k}: pinned {w}, installed {g} — this is a DIFFERENT EXPERIMENT.")
if drift: sys.exit(1)
print(f"  pinned stack OK  (torch build: {torch.__version__})")
PY

    # (b) ⚠⚠ IMPORT WHAT THE JOB IMPORTS. This is the check that ends the
    #     missing-module round trips, and it REPLACES curating a package list.
    #     ~6 s on a login node instead of a GPU allocation.
    #     ⚠ PYTHONPATH, not `cd src`: $FIR_VENV is the RELATIVE ./env, so a cd
    #       into src makes $FIR_VENV/bin/python a nonexistent path and the check
    #       fails for a reason unrelated to the environment. And APPEND (numpy).
    echo "  entry-point imports (what the job actually imports):"
    ( PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}" "$FIR_VENV/bin/python" - <<'PY' || exit 1
import importlib, sys
# train_glue is THE runner and pulls in every adapter it dispatches to; the rest
# are imported here so a broken one is named directly instead of surfacing as a
# 2,700-line traceback inside train_glue's module scope.
mods = ["train_glue", "slr_adapter", "merged_fourierft", "loca_adapter",
        "loca_dct_utils", "qwha_adapter", "qwha_hadamard", "spectral_adapter",
        "bwht_adapter", "fourierft_fast", "sparse_adapter", "commonsense_mc"]
bad = []
for m in mods:
    try:
        importlib.import_module(m); print(f"    {m}: OK")
    except Exception as e:
        bad.append(m); print(f"    {m}: FAILED -> {type(e).__name__}: {str(e)[:300]}")
if bad:
    print("    -> a module the run needs is missing or broken. Every cell importing")
    print("       it will die on a compute node. Fix here, not in the sweep.")
    sys.exit(1)
PY
    ) || rc=1

    # (c) GPU only where a GPU exists (FIR_SETUP C7/C8: verification must be no
    #     more privileged than the node it runs on).
    if [ "$want" = "gpu" ]; then
        "$FIR_VENV/bin/python" - <<'PY' || rc=1
import torch, sys
if not torch.cuda.is_available():
    print("  FAIL: no CUDA device visible"); sys.exit(1)
print(f"  torch {torch.__version__} cuda={torch.version.cuda} "
      f"dev={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} "
      f"bf16={torch.cuda.is_bf16_supported()}")
PY
    fi

    # (TEMP) the two AUTHORS' clones. ⚠ TRAINING DOES NOT NEED THESE — only the
    #   bit-identity VERIFIERS do (src/verify_loca_adapter.py:41,
    #   src/verify_qwha_adapter.py:33 exec the authors' files in a subprocess).
    #   Missing clones do not break a cell; they silently remove the ONE check
    #   that the peft 0.13->0.18 pin change did not move our comparator, which is
    #   the worse failure. Probe the EXACT FILE the code opens, not the directory:
    #   a clone that fetched but left its pin unchecked-out passes a bare -d test.
    #   ⚠ FIR_ASSERT_SKIP_TEMP exists for a REAL DEPENDENCY ORDER, not as an
    #     escape hatch: 01c needs the venv, so it can only run after 01, and
    #     without the flag a correct fresh setup always ends "SETUP INCOMPLETE".
    #     01 sets it; 01c and 03_preflight do NOT.
    if ! _need 01c; then
        echo "  temp/ author clones: SKIPPED — created by stage 01c, which has not run yet"
    else
        local miss="" probe
        for probe in "LoCA/peft/src/peft/tuners/loca/dct_utils.py" \
                     "qwha/peft/src/peft/tuners/qwha/hadamard.py"; do
            [ -e "./temp/$probe" ] || miss="$miss ${probe%%/*}"
        done
        if [ -n "$miss" ]; then
            echo "  ⚠ temp/ author clones MISSING ->$miss"
            echo "    temp/ is gitignored (.gitignore), so a git pull does NOT carry it."
            echo "    -> bash sbatch/fir/01c_stage_repos.sh"
            rc=1
        else
            echo "  temp/ author clones: OK"
        fi
    fi

    # (d) AN OFFLINE LOAD, EXACTLY AS A COMPUTE NODE WILL DO IT. In a subshell so
    #     the offline exports do not leak into the caller.
    if ! _need 02; then
        echo "  offline model cache: SKIPPED — populated by stage 02, which has not run yet"
    else
    ( fir_export_offline
      "$FIR_VENV/bin/python" - <<PY || exit 1
import sys
from transformers import AutoConfig, AutoTokenizer
try:
    c = AutoConfig.from_pretrained("$FIR_MODEL")
    AutoTokenizer.from_pretrained("$FIR_MODEL")
    print(f"  offline model cache OK: $FIR_MODEL "
          f"(hidden={c.hidden_size} layers={c.num_hidden_layers} kv={getattr(c,'num_key_value_heads','?')})")
except Exception as e:
    print(f"  FAIL offline model cache: {type(e).__name__}: {str(e)[:200]}")
    print("    -> run 02_download_cache.sh on a LOGIN node first")
    sys.exit(1)
PY
    ) || rc=1
    fi

    [ $rc -eq 0 ] && echo "--- fir_assert_env PASSED ---" || echo "--- fir_assert_env FAILED ---"
    return $rc
}
