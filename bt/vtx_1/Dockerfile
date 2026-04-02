# =============================================================================
# LeRobot StrGroot Vertex AI 训练镜像
#
# 基于 NVIDIA PyTorch NGC 镜像，包含 CUDA、cuDNN、NCCL 等。
# 构建命令见 build_docker.sh
# =============================================================================

# --- Stage 1: 基础环境 ---
FROM nvcr.io/nvidia/pytorch:24.12-py3

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Google Cloud SDK（用于 gsutil 上传 checkpoint）
RUN curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/opt
ENV PATH="/opt/google-cloud-sdk/bin:${PATH}"

WORKDIR /app

# --- Stage 2: Python 依赖 ---

# 先复制 pyproject.toml 和 setup 文件，利用 Docker 缓存
COPY pyproject.toml setup.py* README.md LICENSE ./
COPY src/lerobot/__init__.py src/lerobot/__init__.py

# 安装 lerobot 核心依赖 + groot extra（包含 flash-attn 等）
# 使用 --no-build-isolation 加速 flash-attn 编译
RUN pip install --no-cache-dir -e ".[groot]" 2>&1 | tail -20

# 安装额外依赖
RUN pip install --no-cache-dir \
    google-cloud-aiplatform>=1.60.0 \
    google-cloud-storage>=2.18.0 \
    wandb>=0.18.0

# --- Stage 3: 复制完整项目代码 ---
COPY . .

# 重新安装以确保所有模块可用（使用 editable mode）
RUN pip install --no-cache-dir -e ".[groot]" 2>&1 | tail -5

# --- Stage 4: 入口 ---

# 将训练入口脚本复制到容器根目录便于调用
COPY bt/vtx_1/train_vertex.py /app/train_vertex.py

# 默认入口：运行 Vertex AI 训练脚本
ENTRYPOINT ["python", "/app/train_vertex.py"]

# 默认参数（可被 Vertex AI CustomJob args 覆盖）
CMD ["--steps=30000", "--batch-size=32", "--freeze-vlm"]
