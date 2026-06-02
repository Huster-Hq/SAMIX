#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ROOT="${CONDA_ROOT:-/data/huqiang/anaconda3}"
ENV_NAME="${ENV_NAME:-SAMIX}"

MANIFEST="${MANIFEST:-${PROJECT_ROOT}/Data/Polyp/train_annotation_manifest.json}"
SAM2_ROOT="${SAM2_ROOT:-${PROJECT_ROOT}/external/sam2}"
SAM2_CKPT="${SAM2_CKPT:-${PROJECT_ROOT}/external/sam2/checkpoints/sam2.1_hiera_tiny.pt}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_t.yaml}"
TEST_ROOT="${TEST_ROOT:-/memory/huqiang/Polyp/images/Public_Dataset/PraNet_241217/TestDataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train_cotrain}"

IMAGE_SIZE="${IMAGE_SIZE:-512}"
SHOTS="${SHOTS:-1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
JOINT_EPOCHS="${JOINT_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
SA_DEVICE="${SA_DEVICE:-cuda:0}"
SEG_DEVICE="${SEG_DEVICE:-cuda:1}"
SUPPORT_SELECTION="${SUPPORT_SELECTION:-nearest}"
SUPPORT_INDEX_BATCH_SIZE="${SUPPORT_INDEX_BATCH_SIZE:-16}"
WARMUP_LR="${WARMUP_LR:-5e-5}"
JOINT_LR="${JOINT_LR:-5e-5}"
SAVE_EVERY="${SAVE_EVERY:-1}"
EVAL_EVERY="${EVAL_EVERY:-1}"
LOG_IMAGES_EVERY="${LOG_IMAGES_EVERY:-200}"

source "${CONDA_ROOT}/bin/activate" "${ENV_NAME}"

cd "${PROJECT_ROOT}"

python scripts/train_cotrain.py \
  --manifest "${MANIFEST}" \
  --sam2-root "${SAM2_ROOT}" \
  --sam2-ckpt "${SAM2_CKPT}" \
  --sam2-config "${SAM2_CONFIG}" \
  --test-root "${TEST_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --image-size "${IMAGE_SIZE}" \
  --shots "${SHOTS}" \
  --warmup-epochs "${WARMUP_EPOCHS}" \
  --joint-epochs "${JOINT_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --sa-device "${SA_DEVICE}" \
  --seg-device "${SEG_DEVICE}" \
  --support-selection "${SUPPORT_SELECTION}" \
  --support-index-batch-size "${SUPPORT_INDEX_BATCH_SIZE}" \
  --warmup-lr "${WARMUP_LR}" \
  --joint-lr "${JOINT_LR}" \
  --save-every "${SAVE_EVERY}" \
  --eval-every "${EVAL_EVERY}" \
  --log-images-every "${LOG_IMAGES_EVERY}" \
  "$@"
