#!/bin/bash
# ============================================================================
# 03b_probe_gpu_memory.sh — DOES STAGE 06 FIT ON THIS CLUSTER'S GPU?
# ============================================================================
#   bash sbatch/narval/03b_probe_gpu_memory.sh            # submits ONE short GPU job
#   bash sbatch/narval/03b_probe_gpu_memory.sh --local    # run the body here (in salloc)
#   bash sbatch/narval/03b_probe_gpu_memory.sh --status   # read the result
#
# ⛔⛔ WHY THIS STAGE EXISTS AT ALL, AND WHY IT IS NOT OPTIONAL.
#   Stage 06 is a FULL fine-tune of google/gemma-2b (2.506 B params) with fp32
#   master weights and fp32 AdamW state. That state is irreducible:
#       params 9,560 MiB + grads 9,560 + exp_avg 9,560 + exp_avg_sq 9,560
#     = 38,241 MiB (~37.3 GiB) BEFORE A SINGLE ACTIVATION.
#   Stages 04 and 05 ran on fir's 80 GiB H100s, where that is comfortable. It is
#   NOT comfortable on a 40 GiB card, and nothing about the previous stages'
#   success says anything about this one.
#
#   ⛔ [MEASURED 2026-09-08, dev box, NVIDIA A40 44.4 GiB, torch 2.5.1+cu121,
#      google/gemma-2b, MRPC, --dtype float32 --mixed_precision fp16 --max_length 128]
#
#        adam_impl        grad ckpt   batch   peak MiB     fits 40 GiB?
#        auto (foreach)   no          32      OOM >45,486      no
#        auto (foreach)   yes         32      OOM >45,486      no
#        auto (foreach)   no          16      OOM >45,486      no
#        auto (foreach)   yes         16      OOM >45,486      no
#        single           no          32      44,090           no
#        single           yes         32      42,258           no
#        fused            no          32      44,090           no
#        fused            yes         32      38,311           ✅
#        fused            yes         16      38,270           ✅
#
#   ⭐ THREE THINGS THAT TABLE SETTLES, AND EACH WOULD OTHERWISE HAVE COST AN
#     ALLOCATION AND A ROUND TRIP:
#     1. torch's DEFAULT AdamW cannot run this cell on anything under ~46 GiB. It
#        dies inside `_multi_tensor_adamw` -> `torch._foreach_sqrt`, which
#        allocates a FULL-SIZE temporary copy of the second-moment buffer.
#     2. BATCH SIZE AND GRADIENT CHECKPOINTING DO NOT RESCUE IT. All four
#        combinations OOM identically, because the activations are not what is
#        large. Anyone debugging this by lowering the batch size would conclude
#        the cluster was broken.
#     3. `--adam_impl fused` + `--gradient_checkpointing` fits 40 GiB with ~2.6 GiB
#        to spare, and BOTH are required -- fused alone is 44,090 MiB.
#
#   ⚠ THAT IS A MEASUREMENT ON A DIFFERENT GPU AND A DIFFERENT torch BUILD
#     (2.5.1+cu121 vs narval's 2.10.0+computecanada). Peak memory is allocator-
#     and kernel-sensitive; this repo's own notes say so. So it is a PREDICTION
#     here, and this stage exists to replace it with narval's own number.
#     FIR_SETUP's most expensive recurring lesson is exactly this: an
#     extrapolation is not a measurement, and a measurement from another machine
#     is not a measurement either.
# ============================================================================
set -uo pipefail
NARVAL_SELF="$(readlink -f "$0")"
cd "$(dirname "$NARVAL_SELF")/../.." || exit 1
export LRS_ENV="sbatch/narval/narval_env.sh"
export LRS_STAGE_DIR="sbatch/narval"
# shellcheck disable=SC1090
source "$LRS_ENV" || exit 1
FIR_SELF="$NARVAL_SELF"
fir_log_to fir_probe_gpumem "$@"

OUTFILE="sbatch/narval/measured_gpu.sh"
LOCAL=false; STATUS=false
for a in "$@"; do
    case "$a" in
        --local) LOCAL=true ;;
        --status) STATUS=true ;;
        --run-body) LOCAL=true ;;
        *) echo "unknown option: $a"; exit 1 ;;
    esac
done

if $STATUS; then
    if [ -f "$OUTFILE" ]; then echo "--- $OUTFILE ---"; sed 's/^/  /' "$OUTFILE"
    else echo "⛔ $OUTFILE does not exist yet — the probe job has not produced a result."; exit 1; fi
    exit 0
fi

# ===========================================================================
# THE BODY — runs on a GPU node.
# ===========================================================================
if [ -n "${SLURM_JOB_ID:-}" ] || $LOCAL; then
    echo "### node=$(hostname)  job=${SLURM_JOB_ID:-none}"
    fir_load_modules_gpu || exit 1
    fir_export_offline
    PY="$FIR_VENV/bin/python"
    [ -x "$PY" ] || { echo "FAIL: no interpreter at $PY — run 01_setup_venv.sh"; exit 1; }

    echo "--- 1. what GPU is this, really ---"
    "$PY" - <<'PY' > /tmp/.narval_gpu.$$ || { echo "FAIL: could not query the GPU"; exit 1; }
import torch, sys
if not torch.cuda.is_available():
    print("ERR no CUDA device visible"); sys.exit(1)
p = torch.cuda.get_device_properties(0)
# ⚠ total_memory is the DEVICE's, in bytes. Report MiB, the unit every other
#   memory number in this repo uses, so the two can be compared without a
#   conversion step nobody re-checks.
print(f"OK {p.name}|{p.total_memory // (1024*1024)}|{p.major}.{p.minor}|{torch.__version__}")
PY
    read -r RES < /tmp/.narval_gpu.$$; rm -f /tmp/.narval_gpu.$$
    case "$RES" in OK\ *) ;; *) echo "FAIL: $RES"; exit 1 ;; esac
    RES="${RES#OK }"
    GPU_NAME="${RES%%|*}"; R2="${RES#*|}"
    GPU_MIB="${R2%%|*}";  R3="${R2#*|}"
    GPU_CC="${R3%%|*}";   TORCH_V="${R3#*|}"
    echo "  device : $GPU_NAME"
    echo "  memory : $GPU_MIB MiB"
    echo "  cc     : $GPU_CC   torch: $TORCH_V"

    echo
    echo "--- 2. ⛔ THE CONTROL: does torch's DEFAULT AdamW actually fail here? ---"
    # ⛔ A CHECK THAT CANNOT FAIL IS NOT A CHECK (FIR_SETUP Law 7). If we only ran
    #   the configuration we expect to work, a green result would not distinguish
    #   "fused rescued it" from "this GPU was always big enough and the flags are
    #   pointless". Run the default too, and EXPECT it to OOM on a 40 GiB card.
    #   ⚠ Its failure is INFORMATION, not an error: the stage does not abort on it.
    # ⚠ NO `set +e`/`set -e` PAIR HERE. `-e` was never on (the header is
    #   `set -uo pipefail`), so switching it ON after the control would have armed
    #   errexit for the REST of the script -- and the very next thing is a `grep -q`
    #   whose non-zero result is a normal outcome. The stage would have exited
    #   silently, at the point it is supposed to report a finding.
    GLUE_SEEDS=42 GLUE_RESULTS_FILE="/tmp/.narval_ctl_$$.csv" \
    "$PY" -u src/train_glue.py --model_name_or_path "$FIR_MODEL" --task_name mrpc \
        --dtype float32 --mixed_precision fp16 --per_device_train_batch_size 32 \
        --num_train_epochs 1 --max_train_steps 12 --num_warmup_steps 1 \
        --max_length 128 --optimizer adamw --learning_rate 3e-5 --weight_decay 0.1 \
        --gradient_checkpointing --name "narval-gpumem-control" > /tmp/.narval_ctl_$$.log 2>&1
    CTL_RC=$?
    if [ $CTL_RC -eq 0 ]; then
        CTL="COMPLETED (this GPU is big enough for the DEFAULT AdamW)"
        echo "  ⚠ the default AdamW COMPLETED. On a 40 GiB card it does not."
        echo "    That is fine and is itself the measurement -- but it means the"
        echo "    fused/checkpointing flags are not load-bearing HERE, and the"
        echo "    reason they are in the recipe must still be recorded."
    elif grep -q "OutOfMemoryError" /tmp/.narval_ctl_$$.log; then
        CTL="OOM (as predicted from the A40 measurements)"
        echo "  ✅ the default AdamW OOMs here, exactly as the dev-box table predicts."
        grep -m1 "OutOfMemoryError" /tmp/.narval_ctl_$$.log | cut -c1-200 | sed 's/^/     /'
    else
        CTL="FAILED rc=$CTL_RC, NOT an OOM"
        echo "  ⛔ the control failed for a reason that is NOT memory. Read the log;"
        echo "     do not proceed on the assumption that this stage measured memory."
        tail -20 /tmp/.narval_ctl_$$.log | sed 's/^/     /'
    fi

    echo
    echo "--- 3. THE RECIPE: --adam_impl fused --gradient_checkpointing ---"
    GLUE_SEEDS=42 GLUE_RESULTS_FILE="/tmp/.narval_fit_$$.csv" \
    "$PY" -u src/train_glue.py --model_name_or_path "$FIR_MODEL" --task_name mrpc \
        --dtype float32 --mixed_precision fp16 --per_device_train_batch_size 32 \
        --num_train_epochs 1 --max_train_steps 12 --num_warmup_steps 1 \
        --max_length 128 --optimizer adamw --adam_impl fused --gradient_checkpointing \
        --learning_rate 3e-5 --weight_decay 0.1 --name "narval-gpumem-recipe" \
        > /tmp/.narval_fit_$$.log 2>&1
    FIT_RC=$?
    PEAK=$("$PY" - "/tmp/.narval_fit_$$.csv" <<'PY'
import csv, sys
try:
    rows = list(csv.DictReader(open(sys.argv[1])))
    print(rows[-1].get("peak_mem_mib", ""))
except Exception:
    print("")
PY
)
    if [ $FIT_RC -ne 0 ]; then
        echo "  ⛔⛔ THE RECIPE ITSELF FAILED (rc=$FIT_RC). STAGE 06 CANNOT RUN HERE."
        tail -25 /tmp/.narval_fit_$$.log | sed 's/^/     /'
        echo
        echo "  ⇒ Report this. The options are all the USER'S CALL and each changes"
        echo "    what the row means: run stage 06 on an >=80 GiB GPU; change the"
        echo "    optimizer state precision (a DIFFERENT experiment); or shard the"
        echo "    optimizer (a real code change to a single-process trainer)."
        exit 1
    fi
    echo "  ✅ completed. peak = ${PEAK:-?} MiB on a ${GPU_MIB} MiB device."

    # ⛔ WRITE THE RECEIPT, NOT THE FLAG. The file records what was MEASURED here,
    #   including the control's outcome, so a later reader can tell whether the
    #   fused/checkpointing flags were doing anything on this cluster.
    {
        echo "# sbatch/narval/measured_gpu.sh — MEASURED ON A NARVAL GPU NODE."
        echo "# ⛔ DO NOT EDIT BY HAND. Written by sbatch/narval/03b_probe_gpu_memory.sh."
        echo "# written : $(date -u +%FT%TZ)"
        echo "# node    : $(hostname)"
        echo "# job     : ${SLURM_JOB_ID:-local}"
        echo "# torch   : $TORCH_V"
        echo "# control (torch's DEFAULT AdamW, same cell): $CTL"
        echo
        printf 'NARVAL_GPU_NAME=%q\n' "$GPU_NAME"
        printf 'NARVAL_GPU_MIB=%q\n' "$GPU_MIB"
        echo 'NARVAL_GPU_MIB_SOURCE=measured'
        printf 'NARVAL_STAGE06_PEAK_MIB=%q\n' "${PEAK%%.*}"
        echo "# the peak above is THIS cluster's own, so the stage-06 gate no longer"
        echo "# relies on the dev-box A40 number it was seeded with."
    } > "$OUTFILE"
    echo
    echo "--- wrote $OUTFILE ---"; sed 's/^/  /' "$OUTFILE"
    rm -f /tmp/.narval_ctl_$$.log /tmp/.narval_fit_$$.log /tmp/.narval_ctl_$$.csv /tmp/.narval_fit_$$.csv
    echo
    echo "############ GPU MEMORY PROBE OK ############"
    exit 0
fi

# ===========================================================================
# SUBMIT
# ===========================================================================
fir_print_provenance
echo "--- login-node gate (fail here, not inside an allocation) ---"
fir_assert_env cpu 02 || { echo "environment not sane — refusing to submit"; exit 1; }
narval_audit_no_fir_defaults || exit 1
echo
echo "  account : $FIR_ACCOUNT_GPU     gres: $FIR_GPU_FULL     mem: $FIR_GPU_MEM"
echo "  ⚠ 25 minutes is generous for ~24 optimizer steps plus two model loads."
jid=$(sbatch --parsable <<SB
#!/bin/bash
#SBATCH --job-name=lrs_gpumem
#SBATCH --account=$FIR_ACCOUNT_GPU
#SBATCH --gpus=$FIR_GPU_FULL
#SBATCH --cpus-per-task=8
#SBATCH --mem=$FIR_GPU_MEM
#SBATCH --time=00:25:00
#SBATCH --output=$(pwd)/logs/slurm_gpumem_%j.out
cd "\$SLURM_SUBMIT_DIR" || exit 1
bash sbatch/narval/03b_probe_gpu_memory.sh --run-body
SB
) || { echo "⛔ sbatch refused the submission — read the message above"; exit 1; }
echo "submitted job $jid"
echo "watch : squeue -j $jid"
echo "read  : bash sbatch/narval/03b_probe_gpu_memory.sh --status"
echo
echo "⛔ Stage 06 REFUSES until this job has written sbatch/narval/measured_gpu.sh."
