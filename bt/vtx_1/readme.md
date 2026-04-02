# vtx_1 — LeRobot StrGroot 在 Vertex AI 上的分布式训练

## 概述

将 `str_groot`（StarVLA Qwen3-VL + GR00T FlowMatching）的训练流水线打包为 Docker 镜像，
部署到 Google Cloud Vertex AI 进行单节点多卡或多节点多卡分布式训练。

StarVLA 从 `https://github.com/Avant-US/starVLA` 的 `bt260402` 分支获取安装。

## 文件说明

| 文件 | 说明 |
|---|---|
| `train_vertex.py` | Vertex AI 训练入口脚本，自动解析集群拓扑，调用 `accelerate launch` + `lerobot-train` |
| `Dockerfile` | 基于 NGC PyTorch 镜像，安装 LeRobot + starVLA(bt260402) + GCS 工具 |
| `build_docker.sh` | 构建 Docker 镜像并推送到 Artifact Registry |
| `submit_vertex_job.py` | 使用 Vertex AI Python SDK 提交分布式训练作业 |
| `submit_vertex_gcloud.sh` | 使用 `gcloud` CLI 提交分布式训练作业 |
| `vertex_ai.md` | Vertex AI 官方参考文档链接 |
| `vertex_plan_1.md` | 本次开发的实施计划 |

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                    Vertex AI CustomJob                  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Node 0      │  │  Node 1      │  │  Node N      │  │
│  │  (Primary)   │  │  (Worker)    │  │  (Worker)    │  │
│  │              │  │              │  │              │  │
│  │ train_vertex │  │ train_vertex │  │ train_vertex │  │
│  │      ↓       │  │      ↓       │  │      ↓       │  │
│  │ accelerate   │  │ accelerate   │  │ accelerate   │  │
│  │   launch     │  │   launch     │  │   launch     │  │
│  │      ↓       │  │      ↓       │  │      ↓       │  │
│  │ lerobot-train│  │ lerobot-train│  │ lerobot-train│  │
│  │  (8x GPU)    │  │  (8x GPU)    │  │  (8x GPU)    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │     NCCL AllReduce / Ring          │          │
│         └──────────────┼────────────────────┘          │
│                        ↓                                │
│                   GCS (checkpoint)                      │
└─────────────────────────────────────────────────────────┘
```

### 关键设计

- **集群拓扑自动发现**：`train_vertex.py` 解析 Vertex AI 注入的 `CLUSTER_SPEC` / `TF_CONFIG` 环境变量，
  自动设置 `MASTER_ADDR`、`MASTER_PORT`、`node_rank` 等，无需手动配置。
- **accelerate 启动**：多卡/多节点时自动使用 `accelerate launch --multi_gpu`，
  与 LeRobot 标准训练流水线完全兼容。
- **starVLA 安装**：Docker 构建时从 git 克隆 `bt260402` 分支并 `pip install -e`。
- **GCS 上传**：训练完成后自动将 checkpoint 上传到指定的 GCS 路径。

## 快速开始

### 前置条件

```bash
# 安装 gcloud CLI 并登录
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 安装 Vertex AI SDK (仅 submit_vertex_job.py 需要)
pip install google-cloud-aiplatform>=1.60.0
```

### Step 1: 构建并推送 Docker 镜像

```bash
# 设置 GCP 项目
export GCP_PROJECT="your-gcp-project-id"
export GCP_REGION="us-central1"

# 构建并推送
bash bt/vtx_1/build_docker.sh --push
```

### Step 2: 提交训练作业

#### 方式 A: Python SDK (推荐)

```bash
# 单节点 8xA100_80GB
python bt/vtx_1/submit_vertex_job.py \
  --project your-gcp-project-id \
  --steps 30000 \
  --batch-size 32

# 2 节点 x 8xA100_80GB 分布式训练
python bt/vtx_1/submit_vertex_job.py \
  --project your-gcp-project-id \
  --num-nodes 2 \
  --steps 30000 \
  --batch-size 64

# dry-run (查看配置, 不提交)
python bt/vtx_1/submit_vertex_job.py --dry-run
```

#### 方式 B: gcloud CLI

```bash
# 单节点
GCP_PROJECT=your-project bash bt/vtx_1/submit_vertex_gcloud.sh

# 2 节点
GCP_PROJECT=your-project NUM_NODES=2 bash bt/vtx_1/submit_vertex_gcloud.sh

# dry-run
DRY_RUN=1 bash bt/vtx_1/submit_vertex_gcloud.sh
```

### Step 3: 监控训练

```bash
# 查看作业列表
gcloud ai custom-jobs list --region=us-central1 --filter="displayName:str-groot"

# 流式查看日志
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1

# 在 Vertex AI Console 查看
# https://console.cloud.google.com/vertex-ai/training/custom-jobs
```

## 本地测试

```bash
# 构建镜像后本地测试 (单卡)
docker run --gpus all str-groot-vertex:latest \
  --local-test --steps 2 --batch-size 1 --starvla-checkpoint ""

# 不使用 Docker, 直接本地测试
python bt/vtx_1/train_vertex.py --local-test --steps 2 --batch-size 1 --starvla-checkpoint ""
```

## 参数速查

### train_vertex.py 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dataset-repo` | `HuggingFaceVLA/libero` | 数据集 repo |
| `--starvla-checkpoint` | `StarVLA/Qwen3VL-GR00T-Bridge-RT-1` | 预训练权重 |
| `--state-indices` | `0 1 2 3 4 5 7` | 状态维度索引 (跳过 pad 维度 6) |
| `--freeze-vlm` | `true` | 冻结 VLM backbone |
| `--steps` | `30000` | 训练步数 |
| `--batch-size` | `32` | 每节点 batch size |
| `--lr` | `1e-4` | 学习率 |
| `--gcs-output` | `""` | 训练完成后上传到 GCS |
| `--wandb` | `false` | 启用 WandB |
| `--local-test` | `false` | 本地单卡测试 |

### submit_vertex_job.py 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--project` | gcloud 默认 | GCP 项目 ID |
| `--region` | `us-central1` | 训练区域 |
| `--num-nodes` | `1` | 节点数 |
| `--gpu-type` | `A100_80GB` | GPU 类型 |
| `--gpu-count` | `8` | 每节点 GPU 数 |
| `--dry-run` | `false` | 只打印配置 |

## Docker 镜像安装链路

```
nvcr.io/nvidia/pytorch:24.12-py3  (CUDA + cuDNN + NCCL)
  ↓
pip install -e ".[groot]"          (lerobot + flash-attn + transformers + peft + ...)
  ↓
git clone -b bt260402 starVLA      (从 github.com/Avant-US/starVLA)
pip install -r starVLA/requirements.txt
pip install -e starVLA
  ↓
pip install qwen-vl-utils omegaconf google-cloud-aiplatform wandb
```

## 常见问题

### Q: 首次构建 Docker 镜像很慢?
A: flash-attn 需要编译，首次构建约 20-30 分钟。后续利用 Docker layer cache 会快很多。

### Q: 多节点训练时 NCCL 连接超时?
A: 确保所有节点在同一 VPC 网络中，且防火墙允许 NCCL 端口 (默认 29500)。
可通过 `--network` 参数指定 VPC 网络。

### Q: 如何使用 GCS FUSE 挂载数据集?
A: Vertex AI 支持 GCS FUSE 自动挂载。将数据集放在 GCS bucket 中，
在 `--output-dir` 使用 `/gcs/your-bucket/...` 路径即可。

### Q: WandB 在 Vertex AI 中如何使用?
A: 通过 `--wandb-api-key` 传入 API key，或在 submit 时设置 `WANDB_API_KEY` 环境变量。
