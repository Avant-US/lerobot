#!/usr/bin/env bash
# =============================================================================
# 使用 gcloud CLI 提交 Vertex AI 分布式训练作业
#
# 适用场景：不想安装 Python SDK，直接用 gcloud 命令行提交。
#
# 用法:
#   # 单节点 8xA100（默认配置）
#   bash bt/vtx_1/submit_vertex_gcloud.sh
#
#   # 2 节点 x 8xA100
#   NUM_NODES=2 bash bt/vtx_1/submit_vertex_gcloud.sh
#
#   # 自定义所有参数
#   GCP_PROJECT=my-proj GCP_REGION=us-central1 NUM_NODES=4 GPU_COUNT=8 \
#   STEPS=50000 BATCH_SIZE=64 \
#     bash bt/vtx_1/submit_vertex_gcloud.sh
#
#   # dry-run（只打印，不执行）
#   DRY_RUN=1 bash bt/vtx_1/submit_vertex_gcloud.sh
# =============================================================================

set -euo pipefail

# --- GCP 配置 ---
GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
GCP_REGION="${GCP_REGION:-us-central1}"
REPO_NAME="${REPO_NAME:-lerobot-training}"
IMAGE_NAME="${IMAGE_NAME:-str-groot-vertex}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

IMAGE_URI="${IMAGE_URI:-${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}}"

# --- 计算资源 ---
NUM_NODES="${NUM_NODES:-1}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-ultragpu-8g}"
GPU_TYPE="${GPU_TYPE:-NVIDIA_A100_80GB}"
GPU_COUNT="${GPU_COUNT:-8}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-200}"

# --- 训练参数 ---
DATASET_REPO="${DATASET_REPO:-HuggingFaceVLA/libero}"
STARVLA_CKPT="${STARVLA_CKPT:-StarVLA/Qwen3VL-GR00T-Bridge-RT-1}"
STATE_INDICES="${STATE_INDICES:-0 1 2 3 4 5 7}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
LOG_FREQ="${LOG_FREQ:-100}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-/gcs/output/str_groot_vertex}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-str-groot}"
GCS_OUTPUT="${GCS_OUTPUT:-}"

# --- WandB ---
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-str_groot_vertex}"
WANDB_KEY="${WANDB_API_KEY:-}"

# --- 其他 ---
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
DRY_RUN="${DRY_RUN:-0}"

# --- 作业名 ---
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
JOB_DISPLAY_NAME="${JOB_NAME_PREFIX}-${NUM_NODES}x${GPU_COUNT}gpu-${TIMESTAMP}"

# --- 构建容器参数 ---
CONTAINER_ARGS=(
    "--dataset-repo=${DATASET_REPO}"
    "--starvla-checkpoint=${STARVLA_CKPT}"
    "--steps=${STEPS}"
    "--batch-size=${BATCH_SIZE}"
    "--num-workers=${NUM_WORKERS}"
    "--lr=${LR}"
    "--log-freq=${LOG_FREQ}"
    "--save-freq=${SAVE_FREQ}"
    "--output-dir=${OUTPUT_DIR}"
    "--job-name=${JOB_NAME_PREFIX}"
    "--freeze-vlm"
)

# state indices
for idx in ${STATE_INDICES}; do
    CONTAINER_ARGS+=("--state-indices" "${idx}")
done

if [[ -n "${GCS_OUTPUT}" ]]; then
    CONTAINER_ARGS+=("--gcs-output=${GCS_OUTPUT}")
fi

if [[ "${WANDB_ENABLE}" == "true" ]]; then
    CONTAINER_ARGS+=("--wandb" "--wandb-project=${WANDB_PROJECT}")
fi

# --- 构建 worker pool spec JSON ---
# Vertex AI gcloud 使用 --worker-pool-spec 格式
# 格式: machine-type=X,accelerator-type=Y,accelerator-count=Z,replica-count=N,container-image-uri=URI

# Primary worker pool (workerpool0)
PRIMARY_SPEC="machine-type=${MACHINE_TYPE}"
PRIMARY_SPEC+=",accelerator-type=${GPU_TYPE}"
PRIMARY_SPEC+=",accelerator-count=${GPU_COUNT}"
PRIMARY_SPEC+=",replica-count=1"
PRIMARY_SPEC+=",container-image-uri=${IMAGE_URI}"

# --- 构建 gcloud 命令 ---
CMD=(
    gcloud ai custom-jobs create
    --project="${GCP_PROJECT}"
    --region="${GCP_REGION}"
    --display-name="${JOB_DISPLAY_NAME}"
    --worker-pool-spec="${PRIMARY_SPEC}"
)

# 多节点时添加额外 worker pool
if [[ "${NUM_NODES}" -gt 1 ]]; then
    WORKER_SPEC="machine-type=${MACHINE_TYPE}"
    WORKER_SPEC+=",accelerator-type=${GPU_TYPE}"
    WORKER_SPEC+=",accelerator-count=${GPU_COUNT}"
    WORKER_SPEC+=",replica-count=$((NUM_NODES - 1))"
    WORKER_SPEC+=",container-image-uri=${IMAGE_URI}"
    CMD+=(--worker-pool-spec="${WORKER_SPEC}")
fi

if [[ -n "${SERVICE_ACCOUNT}" ]]; then
    CMD+=(--service-account="${SERVICE_ACCOUNT}")
fi

# 添加容器参数（通过 --args 传递）
ARGS_STR=""
for arg in "${CONTAINER_ARGS[@]}"; do
    if [[ -n "${ARGS_STR}" ]]; then
        ARGS_STR+=","
    fi
    ARGS_STR+="${arg}"
done
CMD+=(--args="${ARGS_STR}")

# --- 打印配置 ---
echo "============================================"
echo "  Vertex AI StrGroot 训练作业"
echo "============================================"
echo "项目:       ${GCP_PROJECT}"
echo "区域:       ${GCP_REGION}"
echo "镜像:       ${IMAGE_URI}"
echo "作业名:     ${JOB_DISPLAY_NAME}"
echo "节点数:     ${NUM_NODES}"
echo "机器类型:   ${MACHINE_TYPE}"
echo "GPU:        ${GPU_COUNT} x ${GPU_TYPE} / node"
echo "总 GPU 数:  $((NUM_NODES * GPU_COUNT))"
echo "训练步数:   ${STEPS}"
echo "Batch size: ${BATCH_SIZE} (per node)"
echo ""
echo "命令:"
echo "  ${CMD[*]}"
echo "============================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo ""
    echo "DRY RUN — 不执行。"
    exit 0
fi

# --- 提交作业 ---
echo ""
echo "提交作业..."
"${CMD[@]}"

echo ""
echo "作业已提交: ${JOB_DISPLAY_NAME}"
echo ""
echo "查看状态:  gcloud ai custom-jobs list --region=${GCP_REGION} --project=${GCP_PROJECT} --filter=\"displayName:${JOB_NAME_PREFIX}\""
echo "查看日志:  gcloud ai custom-jobs stream-logs JOB_ID --region=${GCP_REGION} --project=${GCP_PROJECT}"
