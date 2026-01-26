#!/usr/bin/env bash
set -x

# Configuration
GPUS=${GPUS:-1}
PORT=${PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PRETRAINED_WEIGHTS="${PROJECT_ROOT}/pretrained_weights/swin-large_pretrain.pth"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/train_run_$(date +%Y%m%d_%H%M%S)}"

# Dataset and model configuration
DATASET_FILE="${DATASET_FILE:-ytvos}"
BACKBONE="${BACKBONE:-swin_l_p4w7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-10}"
LR_DROP="${LR_DROP:-3 5}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Additional arguments from command line
PY_ARGS="${@}"

echo "============================================"
echo "Training Configuration:"
echo "  GPUs: ${GPUS}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Pretrained weights: ${PRETRAINED_WEIGHTS}"
echo "  Dataset: ${DATASET_FILE}"
echo "  Backbone: ${BACKBONE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS}"
echo "============================================"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run training
PYTHONPATH="${PROJECT_ROOT}":$PYTHONPATH \
python3 -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${PORT} \
    --use_env \
    ${PROJECT_ROOT}/main.py \
    --dataset_file ${DATASET_FILE} \
    --backbone ${BACKBONE} \
    --with_box_refine \
    --freeze_text_encoder \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr_drop ${LR_DROP} \
    --num_workers ${NUM_WORKERS} \
    --output_dir ${OUTPUT_DIR} \
    --pretrained_weights ${PRETRAINED_WEIGHTS} \
    ${PY_ARGS}

echo "============================================"
echo "Training completed!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "============================================"
