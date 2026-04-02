#!/usr/bin/env bash
# =============================================================================
# 构建并推送 LeRobot StrGroot 训练 Docker 镜像到 Google Artifact Registry
#
# 用法:
#   # 仅构建本地镜像
#   bash bt/vtx_1/build_docker.sh
#
#   # 构建并推送到 Artifact Registry
#   bash bt/vtx_1/build_docker.sh --push
#
#   # 自定义项目/区域/镜像名
#   GCP_PROJECT=my-project GCP_REGION=us-central1 IMAGE_NAME=my-image \
#     bash bt/vtx_1/build_docker.sh --push
# =============================================================================

set -euo pipefail

# --- 配置（可通过环境变量覆盖）---
GCP_PROJECT="${GCP_PROJECT:-your-gcp-project-id}"
GCP_REGION="${GCP_REGION:-us-central1}"
REPO_NAME="${REPO_NAME:-lerobot-training}"
IMAGE_NAME="${IMAGE_NAME:-str-groot-vertex}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# 完整镜像 URI
LOCAL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
REMOTE_IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "============================================"
echo "  LeRobot StrGroot Vertex AI Docker 构建"
echo "============================================"
echo "项目根目录: ${PROJECT_ROOT}"
echo "本地镜像:   ${LOCAL_IMAGE}"
echo "远程镜像:   ${REMOTE_IMAGE}"
echo ""

# --- 构建镜像 ---
echo "[1/3] 构建 Docker 镜像..."
cd "${PROJECT_ROOT}"

docker build \
    -f bt/vtx_1/Dockerfile \
    -t "${LOCAL_IMAGE}" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

echo "[1/3] 构建完成: ${LOCAL_IMAGE}"

# --- 推送到 Artifact Registry ---
if [[ "${1:-}" == "--push" ]]; then
    echo ""
    echo "[2/3] 配置 Docker 认证..."
    gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

    # 确保 Artifact Registry 仓库存在
    echo "[2/3] 确保 Artifact Registry 仓库存在..."
    gcloud artifacts repositories describe "${REPO_NAME}" \
        --project="${GCP_PROJECT}" \
        --location="${GCP_REGION}" \
        --format="value(name)" 2>/dev/null || \
    gcloud artifacts repositories create "${REPO_NAME}" \
        --project="${GCP_PROJECT}" \
        --location="${GCP_REGION}" \
        --repository-format=docker \
        --description="LeRobot training images"

    echo "[3/3] 推送镜像到 Artifact Registry..."
    docker tag "${LOCAL_IMAGE}" "${REMOTE_IMAGE}"
    docker push "${REMOTE_IMAGE}"

    echo ""
    echo "推送完成!"
    echo "镜像 URI: ${REMOTE_IMAGE}"
else
    echo ""
    echo "跳过推送。使用 --push 参数推送到 Artifact Registry。"
fi

echo ""
echo "============================================"
echo "  本地测试命令:"
echo "  docker run --gpus all ${LOCAL_IMAGE} --local-test --steps 2 --batch-size 1"
echo "============================================"
