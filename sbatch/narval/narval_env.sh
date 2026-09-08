#!/bin/bash
# =============================================================================
# sbatch/narval/narval_env.sh — the narval env. IT CONTAINS NO CLUSTER VALUES.
# =============================================================================
# ⛔⛔ READ THIS FIRST. IT IS THE WHOLE DESIGN, AND IT IS DELIBERATELY DIFFERENT
#    FROM sbatch/fir/fir_env.sh.
#
# fir_env.sh carries fir's account names, gres strings, module order and quotas as
# LITERALS. Those literals were originally INHERITED FROM RORQUAL and four of them
# were wrong (FIR_SETUP §2.3) -- including `--gpus=h100_3g.40gb:1`, which does not
# error on a cluster that lacks it: ⛔ IT QUEUES FOREVER, WHICH LOOKS EXACTLY LIKE
# A BUSY SCHEDULER. That single failure mode is what turned the fir bring-up into
# a multi-day round trip, because a guess and a correct value are indistinguishable
# until a job either starts or does not.
#
# ⇒ THIS FILE DEFINES NO CLUSTER VALUE AT ALL. Every one comes from
#   `sbatch/narval/measured.sh`, which `00_probe_narval.sh` WRITES from what it
#   actually measured on a narval login node. If that file is absent, or is missing
#   a key, EVERY STAGE REFUSES TO RUN. There is nothing to inherit, so there is
#   nothing to inherit WRONGLY, and the failure moves from "a job that never starts"
#   to "a message on the login node naming the stage that produces the value".
#
# WHAT THIS FILE DOES, IN ORDER:
#   1. declares the required measured keys (and nothing else);
#   2. loads measured.sh, or refuses with the command that creates it;
#   3. asserts every required key is present and non-empty  (FAIL CLOSED);
#   4. exports them under the FIR_* names the SHARED implementation reads;
#   5. sources sbatch/fir/fir_env.sh for its FUNCTIONS only -- every value it would
#      otherwise default is already set, and step 6 proves it;
#   6. AUDITS: re-reads fir_env.sh's own defaults and refuses if any of them
#      survived into a variable we were supposed to have measured.
#
# ⚠ WHY THE VARIABLES ARE STILL CALLED FIR_*. They are the parameter names of the
#   shared implementation (sbatch/fir/*.sh, scripts/fir_*.py), which narval reuses
#   UNCHANGED. Renaming them across ~10,000 lines to make one directory read nicely
#   would fork the protocol, and `two copies of a protocol are two protocols` is the
#   rule this whole tree is built on. Read FIR_ as "the cluster's", not "fir's".
#   ⭐ Step 6 is what makes that safe: it proves no fir VALUE leaked in with the name.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. THE CONTRACT. Every key here MUST be set by measured.sh. Adding a knob to
#    the tree means adding it here, so it cannot be silently omitted.
# ---------------------------------------------------------------------------
NARVAL_REQUIRED_KEYS="
NARVAL_MODULES_CPU
NARVAL_MODULES_GPU
NARVAL_GPU_FULL
NARVAL_GPU_MEM
NARVAL_ACCOUNT_GPU
NARVAL_ACCOUNT_CPU
NARVAL_PYTHON_VERSION
NARVAL_SCRATCH
NARVAL_PROBE_UTC
NARVAL_PROBE_HOST
"

# ⚠ GPU MODEL AND VRAM ARE **NOT** IN THE REQUIRED SET, AND THAT IS DELIBERATE.
#   A login node has no GPU, so `00_probe_narval.sh` CANNOT measure VRAM -- it can
#   only read the gres NAME out of the scheduler. Making it required would either
#   block stages 00c/01/02 (which do not need it) or push the probe into inventing
#   the number from the model name, which is precisely the class of inherited guess
#   this file exists to eliminate.
#   ⇒ VRAM is measured by `sbatch/narval/03b_probe_gpu_memory.sh`, ON A GPU NODE,
#     and appended to measured.sh with NARVAL_GPU_MIB_SOURCE=measured. Only stage 06
#     needs it, and `narval_assert_stage06_fits` refuses without it.
NARVAL_OPTIONAL_KEYS="NARVAL_GPU_NAME NARVAL_GPU_MIB NARVAL_GPU_MIB_SOURCE"

# ⛔⛔ THERE IS NO `die()` HELPER HERE, AND THAT IS NOT AN OVERSIGHT.
#   The first version of this file had one:
#       _narval_die() { echo "$@" >&2; return 1 2>/dev/null || exit 1; }
#   ⛔ IT DID NOT STOP ANYTHING. `return` inside a FUNCTION returns from the
#   FUNCTION, not from the sourced file, so every refusal below printed its message
#   in full and then execution carried straight on -- into sourcing a file that does
#   not exist, into checks against unset keys, and ultimately into a submit path
#   with no measured values at all. The message looked exactly like a working gate.
#   ⭐ Caught by scripts/narval_shell_gates.py on its first run, which is the whole
#     argument for Law 11: this is a bug that a green transcript hides and only an
#     adversarial local harness finds.
#   ⇒ every refusal is written INLINE, at FILE scope, where `return` really does
#     leave the sourced file. Verbose, and load-bearing.

# ---------------------------------------------------------------------------
# 2. LOAD THE MEASURED FACTS, OR REFUSE.
# ---------------------------------------------------------------------------
NARVAL_MEASURED="${NARVAL_MEASURED:-sbatch/narval/measured.sh}"
if [ ! -f "$NARVAL_MEASURED" ]; then
    echo "⛔⛔ NO MEASURED CLUSTER FACTS: $NARVAL_MEASURED does not exist."
    echo
    echo "   This tree deliberately carries NO cluster values. On fir, four values"
    echo "   inherited from another cluster were wrong, and one of them (a gres"
    echo "   string) does not error -- it QUEUES FOREVER. So nothing here is"
    echo "   guessed; it is measured once, on a login node, by:"
    echo
    echo "       bash sbatch/narval/00_probe_narval.sh"
    echo
    echo "   That is READ-ONLY apart from writing this one file. Run it, read what"
    echo "   it found, THEN re-run this stage."
    return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
. "$NARVAL_MEASURED" || { echo "⛔ $NARVAL_MEASURED exists but could not be sourced" >&2
                          return 1 2>/dev/null || exit 1; }

# The GPU-node half of the measurements, written by 03b_probe_gpu_memory.sh. It is
# a SEPARATE file for one reason: a login node cannot produce it, so its absence is
# a normal early state and must not look like a broken measured.sh. Sourced SECOND
# so a real GPU measurement always wins over anything in the login-node file.
NARVAL_MEASURED_GPU="${NARVAL_MEASURED_GPU:-sbatch/narval/measured_gpu.sh}"
if [ -f "$NARVAL_MEASURED_GPU" ]; then
    # shellcheck disable=SC1090
    . "$NARVAL_MEASURED_GPU" || { echo "⛔ $NARVAL_MEASURED_GPU could not be sourced" >&2
                                 return 1 2>/dev/null || exit 1; }
fi

# ---------------------------------------------------------------------------
# 3. FAIL CLOSED ON A MISSING OR EMPTY KEY.
#    ⚠ EMPTY, not just unset. FIR_SETUP Law 3: an empty scope variable turns a
#      check into a rubber stamp, and `readlink -f` on a nonexistent path prints
#      the empty string -- this project has already shipped one gate that passed
#      vacuously because a variable was "" rather than absent.
# ---------------------------------------------------------------------------
_narval_missing=""
for _k in $NARVAL_REQUIRED_KEYS; do
    eval "_v=\${$_k:-}"
    [ -n "$_v" ] || _narval_missing="$_narval_missing $_k"
done
if [ -n "$_narval_missing" ]; then
    echo "⛔⛔ $NARVAL_MEASURED is INCOMPLETE. Missing or empty:$_narval_missing"
    echo "   The probe writes a key ONLY when it actually measured it, so a missing"
    echo "   key means the probe could not determine that fact -- read its transcript"
    echo "   (./logs/narval_probe_*.log) for the reason. ⛔ Do NOT hand-write the"
    echo "   value in: an unmeasured value here is the fir failure verbatim."
    return 1 2>/dev/null || exit 1
fi
unset _narval_missing _k _v

# ---------------------------------------------------------------------------
# 4. BIND THE MEASURED FACTS TO THE SHARED IMPLEMENTATION'S PARAMETER NAMES.
#    ⚠ These are EXPORTS, not defaults: fir_env.sh writes every one of its values
#      as ${X:-<fir literal>}, so a value already set here WINS and the fir
#      literal is never reached. Step 6 verifies that claim rather than trusting it.
# ---------------------------------------------------------------------------
export LRS_CLUSTER="narval"
export LRS_STAGE_DIR="sbatch/narval"

export FIR_MODULES_CPU="$NARVAL_MODULES_CPU"
export FIR_MODULES_GPU="$NARVAL_MODULES_GPU"
export FIR_GPU_FULL="$NARVAL_GPU_FULL"
export FIR_GPU_MEM="$NARVAL_GPU_MEM"
export FIR_ACCOUNT_GPU="$NARVAL_ACCOUNT_GPU"
export FIR_ACCOUNT_CPU="$NARVAL_ACCOUNT_CPU"
# ⚠ NO MIG SLICE IS CARRIED. fir's FIR_GPU_MIG40 is an INTERACTIVE-only string that
#   exists on exactly one fir node. If narval has MIG partitions the probe records
#   them in its transcript for a human to read; nothing in this tree submits to one,
#   because stage 06 needs more memory than any MIG slice has (see §MEMORY below).
export FIR_GPU_MIG40="__UNSET_ON_NARVAL__"

# The pinned stack is a property of THE EXPERIMENT, not of the cluster, so it is
# NOT measured -- it is asserted, and 00c proves narval's wheelhouse can serve it.
# ⛔ If a pin does not resolve on narval, that is a FINDING and a STOP. Substituting
#   a neighbouring version is a DIFFERENT EXPERIMENT, and every FourierFT number in
#   this repo is gated bit-identical to the INSTALLED peft (src/qwha_adapter.py:14).
export FIR_PIN_TORCH="${FIR_PIN_TORCH:-2.10.0}"
export FIR_PIN_TRANSFORMERS="${FIR_PIN_TRANSFORMERS:-4.51.3}"
export FIR_PIN_DATASETS="${FIR_PIN_DATASETS:-4.5.0}"
export FIR_PIN_PEFT="${FIR_PIN_PEFT:-0.18.1}"
export FIR_PIN_ACCELERATE="${FIR_PIN_ACCELERATE:-1.12.0}"
export FIR_PIN_EVALUATE="${FIR_PIN_EVALUATE:-0.4.6}"
export FIR_MODEL="${FIR_MODEL:-google/gemma-2b}"

# Scratch root: same rule as fir (derived from the repo directory name, never
# hardcoded, so two checkouts cannot silently share one venv/cache/runs root).
export FIR_SCRATCH_ROOT="${FIR_SCRATCH_ROOT:-$NARVAL_SCRATCH/$(basename "$(readlink -f "$(pwd)")")}"

# ---------------------------------------------------------------------------
# 5. THE SHARED FUNCTIONS.
# ---------------------------------------------------------------------------
# shellcheck disable=SC1091
. sbatch/fir/fir_env.sh || { echo "⛔ could not source the shared implementation" >&2
                             return 1 2>/dev/null || exit 1; }

# ---------------------------------------------------------------------------
# 6. ⭐⭐ THE AUDIT — THE CONTROL THAT MAKES REUSING FIR'S FILE SAFE.
#    A variable we forgot to set in step 4 would silently take fir's literal and
#    submit narval jobs with `--gpus=h100:1`, which QUEUES FOREVER. So do not trust
#    step 4: parse fir_env.sh for the defaults it WOULD have supplied and refuse if
#    any of them is what we ended up with.
#    ⚠ It compares VALUES, not whether we assigned. Assigning the fir value on
#      purpose is still an error here -- and if narval genuinely shares a value
#      with fir, measured.sh will contain it and NARVAL_ALLOW_SAME lists it.
# ---------------------------------------------------------------------------
narval_audit_no_fir_defaults() {
    local rc=0 v fir_default ours
    # ⚠ read the defaults out of the REAL file, so this cannot drift from it.
    for v in FIR_GPU_FULL FIR_ACCOUNT_GPU FIR_ACCOUNT_CPU FIR_MODULES_CPU FIR_MODULES_GPU FIR_GPU_MEM; do
        fir_default="$(sed -n "s/^${v}=\"\${${v}:-\([^}]*\)}\".*/\1/p" sbatch/fir/fir_env.sh | head -1)"
        eval "ours=\${$v:-}"
        [ -n "$fir_default" ] || continue
        if [ "$ours" = "$fir_default" ]; then
            case " ${NARVAL_ALLOW_SAME:-} " in
                *" $v "*)
                    echo "  ⚠ $v == fir's value ('$ours') -- declared identical in measured.sh" ;;
                *)
                    echo "  ⛔⛔ $v is still FIR'S DEFAULT: '$ours'"
                    echo "     It was not set from a narval measurement. On a wrong gres string"
                    echo "     the job QUEUES FOREVER instead of failing (FIR_SETUP A1)."
                    echo "     ⇒ re-run: bash sbatch/narval/00_probe_narval.sh"
                    echo "     (If narval genuinely shares this value, the probe records it and"
                    echo "      adds it to NARVAL_ALLOW_SAME -- do not add it by hand.)"
                    rc=1 ;;
            esac
        fi
    done
    return $rc
}

# ---------------------------------------------------------------------------
# §MEMORY — ⛔⛔ THE ONE NUMBER THAT DECIDES WHETHER STAGE 06 CAN RUN HERE.
#
# Stage 06 is a FULL fine-tune of google/gemma-2b (2.506 B params) with fp32
# master weights and fp32 AdamW state. That state is irreducible:
#     params 9,560 MiB + grads 9,560 + exp_avg 9,560 + exp_avg_sq 9,560
#   = 38,241 MiB  ( ~37.3 GiB )  BEFORE a single activation.
#
# ⛔ [MEASURED 2026-09-08, dev box, NVIDIA A40 44.4 GiB, google/gemma-2b, MRPC,
#    --dtype float32 --mixed_precision fp16 --max_length 128]
#      torch's DEFAULT AdamW (the `foreach` multi-tensor path) OOMs at 44.4 GiB
#      in `_multi_tensor_adamw` -> `torch._foreach_sqrt`, which allocates a
#      FULL-SIZE temporary copy of the second-moment buffer. It died on the THIRD
#      optimizer step in ALL FOUR of {bs 16, bs 32} x {gradient checkpointing on,
#      off} -- ⭐ batch size and checkpointing are IRRELEVANT here, because the
#      activations are not what is large.
#      With `--adam_impl single` (foreach=False) the identical cell COMPLETES,
#      peak 42,258 MiB.
#
# ⇒ The floor for this stage is ~42 GiB, i.e. it needs a GPU with MORE THAN 40 GiB.
#   `narval_assert_stage06_fits` below refuses on the LOGIN NODE if the measured
#   GPU is too small, instead of discovering it after a queue wait and an OOM.
# ---------------------------------------------------------------------------
# Measured peak, in MiB, for the cheapest configuration that actually completes.
NARVAL_STAGE06_PEAK_MIB="${NARVAL_STAGE06_PEAK_MIB:-42258}"
# Headroom for a different torch build, a different allocator and fragmentation.
# ⚠ The dev box ran torch 2.5.1+cu121; narval will run 2.10.0+computecanada.
#   Peak memory is allocator- and kernel-sensitive, so this is a MEASUREMENT ON A
#   DIFFERENT BUILD, not a guarantee. Treat it as a lower bound.
NARVAL_STAGE06_HEADROOM_MIB="${NARVAL_STAGE06_HEADROOM_MIB:-1500}"

narval_assert_stage06_fits() {
    local need=$((NARVAL_STAGE06_PEAK_MIB + NARVAL_STAGE06_HEADROOM_MIB))
    echo "--- stage 06 memory feasibility (measured, not assumed) ---"
    # ⛔ FAIL CLOSED ON AN UNMEASURED GPU. "$NARVAL_GPU_MIB" is empty until a job has
    #   actually run on a narval GPU. An empty value in an arithmetic comparison is
    #   0 under `set -u`-less shells, i.e. it would read as "0 MiB" and refuse for
    #   the wrong reason -- or, with a default, silently PASS. Neither is acceptable.
    if [ -z "${NARVAL_GPU_MIB:-}" ] || [ "${NARVAL_GPU_MIB_SOURCE:-}" != "measured" ]; then
        echo "  ⛔ THE GPU'S MEMORY HAS NOT BEEN MEASURED ON THIS CLUSTER."
        echo "     NARVAL_GPU_MIB='${NARVAL_GPU_MIB:-<unset>}'  source='${NARVAL_GPU_MIB_SOURCE:-<unset>}'"
        echo "     A login node has no GPU, so stage 00 cannot answer this and does"
        echo "     not pretend to. Run the 3-minute GPU job that measures it:"
        echo "         bash sbatch/narval/03b_probe_gpu_memory.sh"
        echo "     It also runs 12 steps of the REAL stage-06 cell, so the answer is"
        echo "     this cluster's own peak on this cluster's own torch build --"
        echo "     not a number carried from another machine."
        return 1
    fi
    echo "  GPU measured on a narval node : $NARVAL_GPU_NAME, ${NARVAL_GPU_MIB} MiB"
    echo "  stage 06 needs            : ${NARVAL_STAGE06_PEAK_MIB} MiB measured"
    echo "                              + ${NARVAL_STAGE06_HEADROOM_MIB} MiB headroom = ${need} MiB"
    if [ "${NARVAL_GPU_MIB:-0}" -ge "$need" ]; then
        echo "  ✅ fits"
        return 0
    fi
    echo "  ⛔⛔ DOES NOT FIT. Stage 06 CANNOT run on this GPU."
    echo
    echo "  This is not a tuning problem. gemma-2b full fine-tuning holds"
    echo "  params + grads + two AdamW moments in fp32 = ~37.3 GiB of state before"
    echo "  any activation, so NEITHER a smaller batch NOR gradient checkpointing"
    echo "  moves it -- [measured] all four combinations OOM identically."
    echo
    echo "  The options, and every one of them is the USER'S CALL, not this"
    echo "  script's, because each changes what the row means:"
    echo "    (a) run stage 06 on an >=80 GiB GPU (fir's H100 is where it was built"
    echo "        for, and stages 04/05 already ran there);"
    echo "    (b) change the optimizer state precision (8-bit Adam, Adafactor)"
    echo "        -- ⛔ a DIFFERENT EXPERIMENT: the full-FT row every LoRA-family"
    echo "        paper reports is fp32 AdamW, and stage 05's nine arms all used it;"
    echo "    (c) shard the optimizer across GPUs -- src/train_glue.py is a"
    echo "        single-process trainer, so this is a real code change, not a flag."
    echo
    echo "  ⛔ Nothing in this tree will pick one. Report the measurement."
    return 1
}
