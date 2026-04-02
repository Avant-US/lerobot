# =============================================================================
# LeRobot StrGroot Vertex AI 分布式训练镜像
#
# 基于 NVIDIA PyTorch NGC 镜像 (含 CUDA, cuDNN, NCCL)
# 安装链路: lerobot[groot] → starVLA (bt260402) → 额外依赖
#
# 构建命令见 build_docker.sh
# =============================================================================

FROM nvcr.io/nvidia/pytorch:24.12-py3

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ---- 系统依赖 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- Google Cloud SDK (gsutil 上传 checkpoint) ----
RUN curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/opt
ENV PATH="/opt/google-cloud-sdk/bin:${PATH}"

WORKDIR /app

# ---- 安装 lerobot[groot] ----
# 先复制 pyproject.toml 等元文件, 利用 Docker 层缓存
COPY pyproject.toml setup.py* README.md LICENSE ./
COPY src/lerobot/__init__.py src/lerobot/__init__.py

RUN pip install --no-cache-dir -e ".[groot]" 2>&1 | tail -20

# ---- 克隆并安装 starVLA (bt260402 分支) ----
RUN git clone --depth 1 -b bt260402 https://github.com/Avant-US/starVLA.git /app/starVLA

# 安装 starVLA 的依赖
RUN pip install --no-cache-dir -r /app/starVLA/requirements.txt 2>&1 | tail -20

# 安装 starVLA 包本身 (editable mode)
RUN pip install --no-cache-dir -e /app/starVLA 2>&1 | tail -10

# ---- 额外依赖 ----
RUN pip install --no-cache-dir \
    qwen-vl-utils \
    omegaconf \
    google-cloud-aiplatform>=1.60.0 \
    google-cloud-storage>=2.18.0 \
    wandb>=0.18.0

# ---- 复制完整 lerobot 项目代码 ----
COPY . .

# 重新安装以确保所有模块可用
RUN pip install --no-cache-dir -e ".[groot]" 2>&1 | tail -5

# ---- 验证 starVLA 导入 ----
RUN python -c "from starVLA.model.framework.QwenGR00T import Qwen_GR00T; print('starVLA import OK')"

# ---- 入口 ----
COPY bt/vtx_1/train_vertex.py /app/train_vertex.py

ENTRYPOINT ["python", "/app/train_vertex.py"]

# 默认参数 (可被 Vertex AI CustomJob args 覆盖)
CMD ["--steps=30000", "--batch-size=32", "--freeze-vlm"]
