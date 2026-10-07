#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --partition=sm3090el8
#SBATCH --job-name=mace_mh1
#SBATCH --output=mace_mh1_%j.out
#SBATCH --error=mace_mh1_%j.err
#SBATCH --time=7-00:00:00      # 2 days
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
##SBATCH --exclusive
#SBATCH --mem-per-gpu=64G #use this to run on sm3090, set batch size = 16.

module purge

__conda_setup="$('/home/energy/rogle/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/energy/rogle/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/energy/rogle/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/energy/rogle/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate mace
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

set -euo pipefail

echo "Job started on $(hostname)"
echo "Working directory: $(pwd)"

nvidia-smi


RUN_NAME="sac_test"

# FOUNDATION_MODEL="/home/energy/emeol/high_entropy_oxides/mace/new_model_MH/mace-mh-1.model"
# PT_TRAIN_FILE="/home/energy/chipa/HEO_mace/train1/fms/mp_traj_filtered_HEO.extxyz"
# TRAIN_FILE="/home/energy/emeol/high_entropy_oxides/mace/new_data/mace_extxyz/train_RLX.extxyz"
TRAIN_FILE="/home/energy/rogle/asm_orr_rxn/mlip/mace_mh1/gen_test_train/split/train.extxyz"
TEST_FILE="/home/energy/rogle/asm_orr_rxn/mlip/mace_mh1/gen_test_train/split/test.extxyz"

LOG_DIR="logs_mh1"
MODEL_DIR="models_mh1"
CHECKPOINTS_DIR="checkpoints_mh1"
RESULTS_DIR="results_mh1"

mkdir -p "$LOG_DIR" "$MODEL_DIR" "$CHECKPOINTS_DIR" "$RESULTS_DIR"

# -----------------------------
# Sanity checks
# -----------------------------
echo "Checking environment..."
which python
which mace_run_train

python - <<'PY'
import mace
import torch

print("MACE version:", getattr(mace, "__version__", "unknown"))
print("Torch version:", torch.__version__)
print("CUDA available inside job:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Checking model/data files..."
# ls -lh "$FOUNDATION_MODEL"
# ls -lh "$PT_TRAIN_FILE"
ls -lh "$TRAIN_FILE"

module use /home/energy/modules/modules/all
module load CUDA/12.6.0

# -----------------------------
# Training
# -----------------------------
# --foundation_model="$FOUNDATION_MODEL" \
#     --E0s='{56:-0.031, 20:-0.026, 27:-1.845, 29:-0.243, 26:-3.099, 57:-0.318, 28:-0.575, 8:-1.548, 59:-0.157, 38:-0.028, 30:-0.011, 40:-1.971}'  \
#         --pt_train_file="$PT_TRAIN_FILE" \
    # --multiheads_finetuning=True \

# NO TEST SET?
# how to gen extxyz
# Do i have the correct energies etccccc compared to your extxyz!!


mace_run_train \
    --E0s='{1: -3.57, 6: -7.87, 7: -6.97, 8: -3.33, 21: -3.62, 24: -7.17, 25: -6.45, 26: -5.85, 27: -4.29, 28: -2.53, 29: -0.84, 30: 2.06, 42: -8.17, 44: -6.26, 45: -4.24, 46: -1.96, 47: 0.55, 50: -1.18, 51: -2.09, 74: -0.43, 75: -9.59, 76: -8.28, 77: -5.82, 78: -3.13, 79: -0.25, 83: -1.19}' \
    --name="$RUN_NAME" \
    --foundation_model="../mace-mh-1.model" \
    --seed=111 \
    --log_dir="$LOG_DIR" \
    --model_dir="$MODEL_DIR" \
    --checkpoints_dir="$CHECKPOINTS_DIR" \
    --results_dir="$RESULTS_DIR" \
    --train_file="$TRAIN_FILE" \
    --test_file="$TEST_FILE" \
    --valid_fraction=0.05 \
    --energy_weight=1.0 \
    --forces_weight=100.0 \
    --loss='universal' \
    --compute_stress=False \
    --eval_interval=1 \
    --error_table='PerAtomMAE' \
    --lr=0.001 \
    --weight_decay=1e-8 \
    --ema \
    --ema_decay=0.995 \
    --swa \
    --swa_energy_weight=10.0 \
    --swa_forces_weight=100.0 \
    --scheduler_patience=5 \
    --batch_size=4 \
    --valid_batch_size=4 \
    --pair_repulsion \
    --max_num_epochs=600 \
    --patience=10 \
    --amsgrad \
    --device=cuda \
    --clip_grad=100 \
    --keep_checkpoints \
    --save_all_checkpoints \
    --restart_latest \
    --default_dtype="float64" \
    --num_workers=8 \
    --save_cpu \
    --enable_cueq=True \
    --plot=True \
