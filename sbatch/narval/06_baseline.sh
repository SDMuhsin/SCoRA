#!/bin/bash
# ============================================================================
# sbatch/narval/06_baseline.sh — narval's entry point for STAGE 06.
# ============================================================================
# ⛔⛔ THERE IS NO SECOND COPY OF THIS STAGE. The implementation is
#   sbatch/fir/06_baseline.sh, shared by both clusters; this file says WHICH
#   ENVIRONMENT it runs under and adds the ONE check that is narval-specific.
#
# ⭐ THE CHECK: DOES STAGE 06 EVEN FIT ON THIS CLUSTER'S GPU?
#   Stage 06 is a FULL fine-tune of gemma-2b -- params + grads + two fp32 AdamW
#   moments = 37.3 GiB of state before any activation. Stages 04 and 05 ran on
#   fir's 80 GiB H100s and say NOTHING about a 40 GiB card. [measured on an A40]
#   torch's default AdamW OOMs at 44 GiB on every batch size and with or without
#   gradient checkpointing; the recipe this stage now submits (`--adam_impl fused
#   --gradient_checkpointing`, baked into fir_baseline_plan.cell_cmd) peaks at
#   38,311 MiB.
#   ⇒ 2.6 GiB of headroom on a 40 GiB device is not a margin to discover after a
#     queue wait. `narval_assert_stage06_fits` refuses HERE, on the login node,
#     against a number MEASURED on a narval GPU by 03b_probe_gpu_memory.sh.
#
# ⛔ THE GATE IS SKIPPED FOR --status AND --run-one, AND ONLY THOSE.
#   --status must work when the answer is "it did not fit" (that is when a status
#   read matters most), and --run-one is the array body: it is already inside an
#   allocation, the decision was made at submit time, and re-deciding there would
#   let a compute node reach a different conclusion from the submitter -- which is
#   FIR_SETUP Law 1's failure shape.
# ============================================================================
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
[ -f sbatch/narval/narval_env.sh ] || { echo "⛔ sbatch/narval/narval_env.sh is missing"; exit 1; }
[ -f sbatch/fir/06_baseline.sh ]   || { echo "⛔ the shared implementation sbatch/fir/06_baseline.sh is missing"; exit 1; }
export LRS_ENV="sbatch/narval/narval_env.sh"
export LRS_STAGE_DIR="sbatch/narval"

_skip_gate=false
for a in "$@"; do
    case "$a" in --status|--run-one) _skip_gate=true ;; esac
done
if ! $_skip_gate; then
    # shellcheck disable=SC1090
    source "$LRS_ENV" || exit 1
    narval_audit_no_fir_defaults || {
        echo "⛔ refusing to submit against un-measured cluster values"; exit 1; }
    narval_assert_stage06_fits   || {
        echo
        echo "⛔ REFUSING TO SUBMIT STAGE 06 ON THIS GPU."
        echo "   Nothing was queued. --status still works."
        exit 1; }
    echo
fi
exec bash "sbatch/fir/06_baseline.sh" "$@"
