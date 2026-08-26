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

# --- ⛔⛔ USER-SITE SHADOWING — EXPORTED AT SOURCE TIME, LOGIN NODE INCLUDED ----
# `~/.local/lib/python3.11/site-packages` on this account holds a FULL torch /
# transformers / peft / datasets / accelerate / huggingface_hub stack [MEASURED,
# 00d_probe_runtime, 2026-08-26]. The venv is built --system-site-packages
# (numpy comes from the scipy-stack module), so ~/.local is on sys.path unless
# this is set.
#
# ⛔⛔ IT IS NOT ONLY AN IMPORT HAZARD. IT IS AN *INSTALL* HAZARD, AND THAT IS THE
#    ONE THAT ACTUALLY FIRED, ON FIR 2026-08-26:
#      01_setup_venv.sh rebuilt the venv with this variable UNSET. pip then saw
#      ~/.local's torch 2.10.0 / transformers 4.51.3 / datasets 4.5.0 /
#      peft 0.18.1 — THE SAME VERSIONS WE PIN — said "Requirement already
#      satisfied", and installed NOTHING INTO THE NEW VENV. Every post-install
#      check then imported ~/.local, printed exactly the pinned version, and
#      PASSED. The empty venv only surfaced on the next compute-node job, where
#      fir_export_offline sets this and `import transformers` died.
#    ⇒ setting it only on the compute node made the login node and the compute
#      node resolve packages DIFFERENTLY — FIR_SETUP Law 1 verbatim. It belongs
#      here, at source time, so every stage, check and job sees one sys.path.
#
# ⚠ It is a no-op when ~/.local is empty, and it does NOT hide the module stack:
#   --system-site-packages and PYTHONPATH (numpy, scipy) are unaffected. Only
#   ~/.local is removed.
# ⚠ If a LOGIN-NODE TOOL ever lived in ~/.local this would break it. Measured:
#   pip and virtualenv both come from CVMFS, not ~/.local; and 01's creation
#   cascade falls back to stdlib `python -m venv` if virtualenv fails at all.
export PYTHONNOUSERSITE=1

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

# --- WHICH COMMIT IS THIS? -----------------------------------------------------
# ⛔⛔ 2026-08-26: a fix was committed here, pushed to the WRONG remote branch
#   (`git push scora HEAD` writes refs/heads/scora; the fir checkout follows
#   `main`), and the next fir log came back showing the OLD behaviour. Diagnosing
#   that cost a full round trip, and every byte needed to spot it in one second
#   was missing from the log: it never said which commit produced it.
#   ⇒ every stage prints its provenance FIRST. A log that cannot identify the code
#     that wrote it cannot be used as evidence about that code.
fir_print_provenance() {
    local sha branch dirty
    sha="$(git rev-parse --short HEAD 2>/dev/null || echo '<not a git checkout>')"
    branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
    dirty="$(git status --porcelain 2>/dev/null | wc -l)"
    echo "commit: $sha on $branch  (uncommitted files: $dirty)"
    echo "        ⚠ compare against the sha that was pushed. If it is older, the pull"
    echo "          did not land -- check WHICH REF you are on, that is how it went"
    echo "          wrong before:  git fetch --all && git log --oneline -3 @{u}"
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
    # ⚠⚠ USER-SITE PACKAGES SHADOW THE VENV. On fir 2026-08-25 every traceback in
    #    03_preflight resolved to ~/.local/lib/python3.11/site-packages -- torch,
    #    transformers and huggingface_hub were NOT the pinned ones we installed.
    #    A venv is not automatically insulated: whether ~/.local is on sys.path
    #    depends on how the venv was created. Setting this makes it explicit and
    #    is a no-op when ~/.local is empty.
    #    ⛔ Diagnose with sbatch/fir/00d_probe_runtime.sh before assuming this is
    #      THE cause -- it prints where every package actually resolves from.
    #    ⚠ KEPT DELIBERATELY, THOUGH fir_env.sh NOW EXPORTS IT AT SOURCE TIME (see
    #      the top of this file). Having it ONLY here is what let stage 01 install
    #      into nothing; having it in BOTH places costs nothing and survives
    #      someone calling fir_export_offline from a shell that never sourced the
    #      header.
    export PYTHONNOUSERSITE=1
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
    # ⚠ THE OVERRIDE EXISTS SO THE CONTROL BELOW CAN BE MADE TO FIRE, and for no
    #   other reason: scripts/fir_shell_gates.py points it at a nonexistent root
    #   to prove the check is not vacuous. Unset (always, on fir) it is the venv.
    local _vreal; _vreal="$(readlink -f "${FIR_ASSERT_VENV_ROOT_OVERRIDE:-$FIR_VENV}" 2>/dev/null)"
    # ⛔ FAIL CLOSED. `readlink -f` prints NOTHING when a parent component does not
    #   exist, and an empty root made os.path.realpath("") resolve to the CWD --
    #   under which every package trivially "resolved inside" and the whole check
    #   silently passed. Caught by its own negative control in fir_shell_gates.py,
    #   which is exactly the job of a negative control.
    [ -n "$_vreal" ] || { echo "FAIL: cannot resolve the venv root ($FIR_VENV)"; return 1; }
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
# ⚠⚠ A VERSION CHECK CANNOT SEE A DIFFERENT BUILD, AND THAT IS WHAT MATTERS.
#    2.10.0, 2.10.0+cu128 and 2.10.0+computecanada all pass
#    startswith("2.10.0") -- yet they are different wheels, and peak memory is
#    allocator- and kernel-sensitive. On fir 2026-08-25 the gate printed
#    "torch build: 2.10.0" (no label) and PASSED while every traceback in the
#    same job resolved to ~/.local, i.e. the venv was being shadowed and the
#    check was structurally blind to it.
#    ⇒ report the build label AND where torch actually loaded from.
import torch as _t
_lbl = _t.__version__.split("+", 1)
print(f"  torch build label   : {'+' + _lbl[1] if len(_lbl) > 1 else '<NONE>'}")
# ⛔ RETRACTED 2026-08-26: an earlier note here reasoned "no build label => not the
#    computecanada wheel". WRONG, and measured on fir: torch imported FROM THE VENV
#    reports '2.10.0' with build_label=<NONE>. The Alliance wheel does not carry a
#    local version label at runtime, so the label proves nothing either way.
#    The PATH is the evidence, not the version string.
#
# ⛔⛔ A VERSION IS NOT A LOCATION, AND ONLY THE LOCATION IS EVIDENCE.
#    Fired on fir 2026-08-26: a freshly rebuilt venv contained NONE of the pinned
#    packages (pip saw the same versions in ~/.local and no-op'd), yet every
#    version check passed because it imported ~/.local. And checking only torch,
#    only for the substring ".local", was the same mistake one layer in: it is a
#    two-name allow-list against one specific shadow. ⇒ assert that EVERY pinned
#    package resolves INSIDE THIS VENV, and print where it came from when it does
#    not. Anything outside is a different experiment, whatever its version says.
#    ⚠ numpy/scipy/pandas are deliberately ABSENT here: they legitimately come
#      from the scipy-stack MODULE (--system-site-packages), so demanding they be
#      in the venv would fail a correct environment.
import importlib, os
_venv = os.path.realpath("$_vreal")
_outside = []
for _m in ["torch", "transformers", "datasets", "peft", "accelerate", "evaluate",
           "huggingface_hub"]:
    _f = os.path.realpath(getattr(importlib.import_module(_m), "__file__", "") or "")
    if not _f.startswith(_venv + os.sep):
        _outside.append((_m, _f))
if _outside:
    print(f"  ⛔⛔ NOT RESOLVING FROM THE VENV ({_venv}):")
    for _m, _f in _outside:
        print(f"      {_m:16} -> {_f or '<no __file__>'}")
    print("      The pinned stack is NOT what will run. Either ~/.local is shadowing")
    print("      the venv (fir_env.sh exports PYTHONNOUSERSITE=1 -- check it survived),")
    print("      or the venv is EMPTY because pip reported 'already satisfied' against")
    print("      ~/.local. Rebuild: bash sbatch/fir/01_setup_venv.sh --fresh")
    print("      Diagnose: bash sbatch/fir/00d_probe_runtime.sh")
    sys.exit(1)
print(f"  pinned packages     : all 7 resolve inside {_venv}")
for k, (w, g) in drift.items():
    print(f"  ⚠⚠ {k}: pinned {w}, installed {g} — this is a DIFFERENT EXPERIMENT.")
if drift: sys.exit(1)
print(f"  pinned stack OK  (torch build: {torch.__version__})")
PY

    # (a2) ⚠⚠ IS THE VENV ACTUALLY ON $PATH? A venv is NOT RELOCATABLE: bin/activate
    #      hardcodes VIRTUAL_ENV as an ABSOLUTE path fixed at creation time. Moving
    #      the venv directory leaves activate pointing at the OLD path, which it
    #      prepends to PATH -- so bare `python` silently falls through to the module
    #      python with no torch and no peft.
    #      ⛔ THIS EXACT THING HAPPENED ON FIR 2026-08-26, after the venv was moved
    #        from .../lora_research_signal to .../SCoRA. Every check here passed,
    #        because they all call "$FIR_VENV/bin/python" EXPLICITLY through the
    #        symlink -- and every stage that used bare `python` then died on
    #        ModuleNotFoundError. A gate that only tests the explicit path cannot
    #        see the PATH the job actually uses.
    local _act="$FIR_VENV/bin/activate" _venv_real
    _venv_real="$(readlink -f "$FIR_VENV" 2>/dev/null)"
    if [ -f "$_act" ]; then
        local _stamp
        _stamp="$(sed -n 's/^VIRTUAL_ENV=["'"'"']\{0,1\}\([^"'"'"']*\)["'"'"']\{0,1\}$/\1/p' "$_act" | head -1)"
        if [ -n "$_stamp" ] && [ "$(readlink -f "$_stamp" 2>/dev/null)" != "$_venv_real" ]; then
            echo "  ⛔ RELOCATED VENV: bin/activate says VIRTUAL_ENV=$_stamp"
            echo "     but the venv really lives at $_venv_real"
            echo "     activate will put a NONEXISTENT dir on PATH and bare 'python'"
            echo "     will NOT be this venv. Rebuild: bash sbatch/fir/01_setup_venv.sh --fresh"
            rc=1
        else
            echo "  venv is not relocated (activate stamp matches)"
        fi
    fi

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
