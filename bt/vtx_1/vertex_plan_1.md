# Plan: bt/vtx_1/ — StrGroot Vertex AI 分布式训练（从零创建）

## Context

将 `str_groot`（StarVLA Qwen3-VL + GR00T FlowMatching action head）训练流水线打包为 Docker 镜像，部署到 Google Cloud Vertex AI 进行分布式训练。StarVLA 需从 `https://github.com/Avant-US/starVLA` 的 `bt260402` 分支获取安装。

## 需要创建的文件（共 6 个，全部在 bt/vtx_1/）

### 文件 1: `bt/vtx_1/Dockerfile`

基于 `nvcr.io/nvidia/pytorch:24.12-py3` (包含 CUDA、cuDNN、NCCL)。

安装顺序：
1. 系统依赖：git, curl, ffmpeg, libgl1-mesa-glx 等
2. Google Cloud SDK（gsutil 用于上传 checkpoint 到 GCS）
3. 复制 lerobot 项目代码，安装 `pip install -e ".[groot]"`（包含 flash-attn, transformers, peft, timm 等）
4. **克隆 starVLA**：`git clone --depth 1 -b bt260402 https://github.com/Avant-US/starVLA.git /app/starVLA`
5. 安装 starVLA 依赖：`pip install -r /app/starVLA/requirements.txt`
6. 安装 starVLA 包：`pip install -e /app/starVLA`
7. 额外依赖：`pip install qwen-vl-utils omegaconf google-cloud-aiplatform wandb`
8. ENTRYPOINT 设为 `python /app/train_vertex.py`

关键参考：
- `bt/str_groot_1/setup_env.sh` — starVLA 安装流程
- `src/lerobot/policies/str_groot/modeling_str_groot.py:35` — `from starVLA.model.framework.QwenGR00T import Qwen_GR00T`

### 文件 2: `bt/vtx_1/train_vertex.py`

Vertex AI 分布式训练入口脚本，功能：

1. **解析 Vertex AI 集群拓扑**：读取 `CLUSTER_SPEC` 或 `TF_CONFIG` 环境变量，提取 `master_addr`、`master_port`、`node_rank`、`num_nodes`
2. **获取当前节点 GPU 数量**：`torch.cuda.device_count()`
3. **构建 lerobot-train 参数**：与 `bt/str_groot_1/train_str_groot_libero.py` 和 `bt/str_groot_1/readme.md` 中的 CLI 参数保持一致
   - `--policy.type=str_groot`
   - `--policy.starvla_checkpoint=StarVLA/Qwen3VL-GR00T-Bridge-RT-1`
   - `--policy.freeze_vlm=true`, `--policy.tune_vlm=false`, `--policy.tune_action_head=true`
   - `--policy.state_indices=[0,1,2,3,4,5,7]`
   - `--policy.attn_implementation=flash_attention_2`
   - `--dataset.repo_id=HuggingFaceVLA/libero`
   - `--batch_size`, `--steps`, `--save_freq`, `--log_freq`, `--eval_freq=0`
   - 可选 `--wandb.enable=true`
4. **决定启动方式**：
   - 多卡/多节点 → `accelerate launch --multi_gpu --num_processes=N --num_machines=M --machine_rank=R --main_process_ip=IP --main_process_port=PORT -m lerobot.scripts.lerobot_train`
   - 单卡 → `python -m lerobot.scripts.lerobot_train`
   - 本地测试 → 直接调用
5. **训练完成后上传 GCS**（仅主节点，用 `gsutil -m rsync -r`）

CLI 参数：`--dataset-repo`, `--starvla-checkpoint`, `--state-indices`, `--freeze-vlm`, `--steps`, `--batch-size`, `--lr`, `--gcs-output`, `--wandb`, `--local-test` 等

### 文件 3: `bt/vtx_1/build_docker.sh`

构建并推送 Docker 镜像到 Google Artifact Registry：

1. 可配置环境变量：`GCP_PROJECT`, `GCP_REGION`, `REPO_NAME`, `IMAGE_NAME`, `IMAGE_TAG`
2. `docker build -f bt/vtx_1/Dockerfile -t IMAGE .`
3. `--push` 参数时：
   - `gcloud auth configure-docker`
   - 确保 Artifact Registry 仓库存在（不存在则创建）
   - `docker tag + docker push`

### 文件 4: `bt/vtx_1/submit_vertex_job.py`

使用 Vertex AI Python SDK (`google.cloud.aiplatform`) 提交 CustomJob：

1. 构建 `worker_pool_specs`：
   - workerpool0（primary）：1 replica
   - workerpool1（workers）：`num_nodes - 1` replicas（多节点时）
2. 每个 pool 指定：机器类型、GPU 类型/数量、容器镜像 URI、容器参数
3. 支持的 GPU 类型：T4, V100, A100, A100_80GB, L4, H100
4. 自动推断机器类型（如 A100_80GB → a2-ultragpu-8g）
5. `--dry-run` 模式只打印不提交
6. 传递训练参数给容器（`--steps`, `--batch-size`, `--freeze-vlm` 等）

### 文件 5: `bt/vtx_1/submit_vertex_gcloud.sh`

使用 `gcloud ai custom-jobs create` 提交训练作业：

1. 环境变量配置：`GCP_PROJECT`, `NUM_NODES`, `GPU_TYPE`, `GPU_COUNT` 等
2. 构建 `--worker-pool-spec` 参数字符串
3. 多节点时添加第二个 worker pool
4. `DRY_RUN=1` 只打印不执行

### 文件 6: `bt/vtx_1/readme.md`

文档包含：架构图、文件说明、快速开始（3 步：构建→提交→监控）、参数速查、本地测试命令、FAQ

## 关键设计决策

1. **starVLA 安装方式**：在 Docker 构建时从 git 克隆 bt260402 分支并 `pip install -e`，不从本地复制
2. **训练启动方式**：通过 `accelerate launch` + `lerobot-train` 而非自定义训练循环，与 LeRobot 标准流水线完全兼容
3. **集群拓扑发现**：解析 Vertex AI 的 `CLUSTER_SPEC` 环境变量自动配置分布式参数

## 关键文件参考

| 参考文件 | 用途 |
|---|---|
| `bt/str_groot_1/setup_env.sh` | starVLA 安装流程（git clone + pip install 顺序） |
| `bt/str_groot_1/train_str_groot_libero.py` | 训练参数基准 |
| `bt/str_groot_1/readme.md` | lerobot-train CLI 完整用法，单卡/多卡/resume 示例 |
| `bt/str_groot_1/requirements.txt` | 额外依赖：qwen-vl-utils, omegaconf |
| `src/lerobot/policies/str_groot/modeling_str_groot.py` | starVLA 导入点 (line 35) |
| `src/lerobot/policies/str_groot/configuration_str_groot.py` | StrGrootConfig 所有参数 |
| `bt/vtx_1/vertex_ai.md` | Vertex AI 官方文档参考 |

## 验证方法

1. **构建 Docker 镜像**：`bash bt/vtx_1/build_docker.sh`
2. **容器内测试**：`docker run --gpus all str-groot-vertex:latest --local-test --steps 2 --batch-size 1 --starvla-checkpoint ""`
3. **Dry-run 提交**（Python SDK）：`python bt/vtx_1/submit_vertex_job.py --dry-run`
4. **Dry-run 提交**（gcloud）：`DRY_RUN=1 bash bt/vtx_1/submit_vertex_gcloud.sh`
