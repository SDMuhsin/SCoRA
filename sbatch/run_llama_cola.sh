#!/bin/bash
# ============================================================================
# LLaMA-7B CoLA Experiments - Spectral vs FourierFT
# ============================================================================
#
# Validates the Spectral Adapter on LLaMA-7B (decoder-only, 4096-dim)
# against FourierFT on CoLA (Matthews Correlation Coefficient).
#
# Mixed-precision: base model in float16, adapter computation in float32.
# All methods target Q+V (q_proj, v_proj) = 64 modules on LLaMA-7B.
#
# Usage:
#   ./sbatch/run_llama_cola.sh
#   ./sbatch/run_llama_cola.sh --account def-myprof
#   ./sbatch/run_llama_cola.sh --local    # Run locally (no SLURM)
#
# ============================================================================

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

ACCOUNT=""
LOCAL_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --local)
            LOCAL_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--account SLURM_ACCOUNT] [--local]"
            exit 1
            ;;
    esac
done

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL="huggyllama/llama-7b"
TASK="cola"
DTYPE="bfloat16"   # float16 causes gradient underflow through 32 LLaMA layers

# --- Shared hyperparameters ---
BATCH_SIZE=4
EVAL_BATCH_SIZE=8
GRAD_ACCUM=8           # Effective batch size = 4 * 8 = 32
WEIGHT_DECAY=0.01
LR_SCHEDULER="linear"
GRAD_CLIP=1.0
EPOCHS=10              # CoLA has 8.5K train samples
TARGET_MODULES="q_proj,v_proj"

# --- FourierFT configurations ---
# Scaling analysis: BERT effective = scaling * 1/(768^2) = 150 * 1.7e-6 = 2.5e-4
# LLaMA effective = scaling * 1/(4096^2) = scaling * 6e-8
# Match BERT: scaling ~ 4267. Try range: 150, 1000, 4000.
FOURIERFT_N=256        # 64 modules x 256 = 16,384 adapter params
FOURIERFT_LR="5e-2"

# --- Spectral configurations ---
# Dense p=q=16: 64 modules x 256 = 16,384 adapter params (matches FourierFT n=256)
# Factored p=q=32, r=4: 64 modules x 256 = 16,384 adapter params (same count)
SPECTRAL_LR="2e-2"

# ============================================================================
# EXPERIMENTS
# ============================================================================

job_count=0
mkdir -p ./logs ./results

run_experiment() {
    local name=$1
    local cmd=$2
    local time_limit=${3:-"12:00:00"}
    local gpu_mem=${4:-"40000M"}

    if [[ "$LOCAL_MODE" == true ]]; then
        echo "========================================"
        echo "Running locally: $name"
        echo "Command: $cmd"
        echo "========================================"
        eval "$cmd"
        return
    fi

    account_line=""
    if [[ -n "$ACCOUNT" ]]; then
        account_line="#SBATCH --account=$ACCOUNT"
    fi

    sbatch_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=llama_${name}
#SBATCH --output=./logs/llama_${name}_%j.out
#SBATCH --error=./logs/llama_${name}_%j.err
#SBATCH --time=$time_limit
#SBATCH --gres=gpu:1
#SBATCH --mem=$gpu_mem
#SBATCH --cpus-per-task=4
$account_line

module load gcc arrow scipy-stack cuda cudnn
source ./env/bin/activate

export HF_HOME=\$(pwd)/data
export HF_DATASETS_CACHE=\$(pwd)/data
export TRANSFORMERS_CACHE=\$(pwd)/data
export TORCH_HOME=\$(pwd)/data
mkdir -p \$HF_HOME

echo '========================================'
echo "Job: llama_${name}"
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Started: \$(date)"
echo '========================================'
nvidia-smi
export PYTHONPATH=\$PYTHONPATH:\$(pwd)/src
$cmd
echo '========================================'
echo "Finished: \$(date)"
echo '========================================'
EOF
)
    echo "  [$sbatch_id] llama_${name}"
    ((job_count++))
}

COMMON="python src/train_glue.py --model_name_or_path $MODEL --task_name $TASK"
COMMON+=" --per_device_train_batch_size $BATCH_SIZE --per_device_eval_batch_size $EVAL_BATCH_SIZE"
COMMON+=" --gradient_accumulation_steps $GRAD_ACCUM"
COMMON+=" --weight_decay $WEIGHT_DECAY --lr_scheduler_type $LR_SCHEDULER"
COMMON+=" --grad_clipping $GRAD_CLIP --dtype $DTYPE"
COMMON+=" --gradient_checkpointing"
COMMON+=" --adapter_target_modules $TARGET_MODULES"

# ============================================================================
# Step 1: Smoke Test (1 epoch, verify no OOM)
# ============================================================================

echo "=== Step 1: Smoke Test ==="

run_experiment "smoke_spectral" \
    "$COMMON --optimizer adamw-spectral --learning_rate 2e-2 --num_train_epochs 1 --spectral_p 16 --spectral_q 16 --spectral_scaling 1.0 --spectral_d_initial 0.01 --name llama_smoke_spectral" \
    "2:00:00" "40000M"

# ============================================================================
# Step 2: FourierFT Baseline Sweep
# ============================================================================

echo ""
echo "=== Step 2: FourierFT Baseline ==="

# Scaling sweep at lr=5e-2
for scaling in 150 1000 4000; do
    run_experiment "fourierft_s${scaling}" \
        "$COMMON --optimizer adamw-fourierft --learning_rate $FOURIERFT_LR --num_train_epochs $EPOCHS --fourierft_n_frequency $FOURIERFT_N --fourierft_scaling ${scaling}.0 --name llama_fourierft_n${FOURIERFT_N}_s${scaling}" \
        "24:00:00" "40000M"
done

# LR sweep at scaling=150 (BERT default)
for lr in 1e-1 2e-1; do
    run_experiment "fourierft_lr${lr}" \
        "$COMMON --optimizer adamw-fourierft --learning_rate $lr --num_train_epochs $EPOCHS --fourierft_n_frequency $FOURIERFT_N --fourierft_scaling 150.0 --name llama_fourierft_n${FOURIERFT_N}_lr${lr}" \
        "24:00:00" "40000M"
done

# ============================================================================
# Step 3: Spectral Configurations
# ============================================================================

echo ""
echo "=== Step 3: Spectral Configurations ==="

# Dense p=q=16 (BERT-winning config transferred)
run_experiment "spectral_p16_d01" \
    "$COMMON --optimizer adamw-spectral --learning_rate $SPECTRAL_LR --num_train_epochs $EPOCHS --spectral_p 16 --spectral_q 16 --spectral_scaling 1.0 --spectral_d_initial 0.01 --name llama_spectral_p16_d01" \
    "24:00:00" "40000M"

# Factored p=q=32, r=4, learn_scaling (BERT CoLA-winning config)
run_experiment "spectral_f32r4_ls" \
    "$COMMON --optimizer adamw-spectral --learning_rate $SPECTRAL_LR --num_train_epochs $EPOCHS --spectral_p 32 --spectral_q 32 --spectral_factored_rank 4 --spectral_scaling 1.0 --spectral_d_initial 0.01 --spectral_learn_scaling --name llama_spectral_f32r4_ls_d01" \
    "24:00:00" "40000M"

# Factored p=q=32, r=4, learn_scaling, d_initial=0.07
run_experiment "spectral_f32r4_ls_d07" \
    "$COMMON --optimizer adamw-spectral --learning_rate $SPECTRAL_LR --num_train_epochs $EPOCHS --spectral_p 32 --spectral_q 32 --spectral_factored_rank 4 --spectral_scaling 1.0 --spectral_d_initial 0.07 --spectral_learn_scaling --name llama_spectral_f32r4_ls_d07" \
    "24:00:00" "40000M"

# Dense p=q=16, scaling=2.0
run_experiment "spectral_p16_s2" \
    "$COMMON --optimizer adamw-spectral --learning_rate $SPECTRAL_LR --num_train_epochs $EPOCHS --spectral_p 16 --spectral_q 16 --spectral_scaling 2.0 --spectral_d_initial 0.01 --name llama_spectral_p16_s2_d01" \
    "24:00:00" "40000M"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================"
echo "Total experiments submitted: $job_count"
echo "Results CSV: ./results/mo53_glue.csv"
echo "Logs: ./logs/"
echo "============================================"
