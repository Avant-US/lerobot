# DM0 R1-Pro 两阶段训练（freeze ViT → LoRA on ViT）

把 `dexbotic/playground/benchmarks/real/r1_pro_dm0_freeze_lora.py` 移植到 LeRobot。

- 入口：`bt/dm0/train_dm0_r1_pro.py`
- 一键脚本：`bt/dm0/run_dm0_freeze_vit.sh`（设好 HF cache、自动跑一次 norm_stats，再 launch 训练）
- 图像增强：`bt/dm0/dm0_augmentations.py`（1:1 复刻 dexbotic `policy_dm0` / `policy_color_dm0`）
- Policy：`lerobot.policies.dm0.DM0Policy` / `DM0Config`
- norm_stats：`examples/dm0/compute_norm_stats.py`（生成 dexbotic 风格 q01/q99 归一化）

## 训练阶段

| `--task` | 含义 | 大致策略 |
|----------|------|----------|
| `train` | phase-1：冻结 ViT，微调 LLM (+ projector / action head) | `DM0Config(freeze_vision_encoder=True)` |
| `lora_train` | phase-2：冻结整模型，在 ViT 的 `out_proj` 上挂 LoRA | `TrainPipelineConfig.peft = PeftConfig(method_type="LORA", r=16)` + `DM0Policy.wrap_with_peft` |

LoRA 的 `target_modules`、`lora_alpha=32`、`lora_dropout=0.05` 写在
`DM0Policy._get_default_peft_targets`，跟 dexbotic 配方一致；想改就改那个方法
（`PeftConfig` 没暴露 alpha / dropout 字段）。

## 准备工作

1. **DM0 base 权重**（dexbotic 原生格式）：默认放 `./checkpoints/DM0-base`，可用 `--dm0-base` 覆盖。
2. **LeRobotDataset**：HuggingFace repo id（`namespace/repo_name`）**或**本地数据目录。
   本地数据要同时给两个参数：
   - `--dataset-repo=local/<任意名>`（占位符，必须满足 `namespace/name` 形式）
   - `--dataset-root=/绝对/路径/到/数据集`
3. **HF cache 环境变量**：必须 `export` 而不是只 set，并且目录必须真的存在可写——
   不然 `datasets` 下载/解析 parquet 时会报 `FileNotFoundError`，错误信息看起来像
   `RepositoryNotFoundError` 让人摸不着头脑。`run_dm0_freeze_vit.sh` 已经处理：

   ```bash
   export HF_HOME=/path/to/hf_cache
   export HF_DATASETS_CACHE="${HF_HOME}/datasets"
   export HF_HUB_CACHE="${HF_HOME}/hub"
   mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"
   ```

4. **`norm_stats.json`**：dexbotic 用的是 q01/q99 quantile 归一化（按 delta action 计算），
   先跑一次：

   ```bash
   python examples/dm0/compute_norm_stats.py \
     --repo-id /绝对/路径/到/数据集 \
     --output ./norm_stats/r1_pro_chassis_v3.json \
     --chunk-size 50 --non-delta-mask 14 15 \
     --max-action-dim 32 --max-state-dim 32
   ```

   生成的 JSON 路径作为 `--norm-stats-path` 传给训练入口（phase-1 必填，phase-2
   一般继承自 phase-1 ckpt，也可显式覆盖）。

## 快速开始

### 推荐：一键脚本（已经把 HF cache / norm_stats / 多卡 / W&B 都串好了）

```bash
bash lerobot/bt/dm0/run_dm0_freeze_vit.sh
# 默认 8 卡，要改：NUM_GPUS=4 bash lerobot/bt/dm0/run_dm0_freeze_vit.sh
```

里面要按你环境改的硬编码：`HF_HOME`、`WANDB_API_KEY`、`DATASET_PATH`、`DM0_BASE`、`NORM_STATS`。

### 手动 phase-1：freeze ViT

```bash
accelerate launch bt/dm0/train_dm0_r1_pro.py \
  --task=train \
  --dataset-repo=local/r1_pro_chassis_v3 \
  --dataset-root=/path/to/dataset \
  --norm-stats-path=./norm_stats/r1_pro_chassis_v3.json \
  --dm0-base=./checkpoints/DM0-base
```

输出在 `outputs/bt/dm0/train-<YYYYMMDD_HHMMSS>/`，最新 ckpt 在
`checkpoints/last/pretrained_model/`。

### 手动 phase-2：LoRA on ViT

```bash
accelerate launch bt/dm0/train_dm0_r1_pro.py \
  --task=lora_train \
  --dataset-repo=local/r1_pro_chassis_v3 \
  --dataset-root=/path/to/dataset \
  --phase1-ckpt=outputs/bt/dm0/train-<YYYYMMDD_HHMMSS>/checkpoints/last/pretrained_model
```

`wrap_with_peft` 会把 base 模型的所有参数都冻住，所以 phase-2 不需要再指定
`--policy.freeze_vision_encoder` 之类的标志；只暴露 `--lora-r`（默认 16）。

### dry-run（只组装配置，不启训练）

```bash
python bt/dm0/train_dm0_r1_pro.py --task=train \
  --dataset-repo=local/r1_pro_chassis_v3 \
  --dataset-root=/path/to/dataset \
  --norm-stats-path=./norm_stats/r1_pro_chassis_v3.json \
  --dry-run
```

## 图像增强（默认开启）

跟 dexbotic 的 `aug_policy=["dm0", "color_dm0", "color_dm0"]` **逐字节一致**：

| camera 类型 | 选用 policy | albumentations 链 |
|-------------|-------------|-------------------|
| head（含 `head` 子串，或第一台 cam） | `policy_dm0` | PadToSquare → RandomResizedCrop(728, scale=0.95) → Rotate(±5°) → ColorJitter |
| wrist / hand（含 `wrist`/`hand` 子串，或后续 cam） | `policy_color_dm0` | PadToSquare → Resize(728) → ColorJitter |

输出统一是 `(3, 728, 728) float[0,1]`，下游 ViT 的 image processor 会再缩放到模型期望尺寸。

实现是一层薄包装 `DM0AugmentedDataset`，套在 `make_dataset(cfg)` 返回的 `LeRobotDataset` 外面，**不动 LeRobot 库本体**——通过 monkey-patch
`lerobot.scripts.lerobot_train.make_dataset` 完成。

相关 CLI：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--no-aug` | off（即增强默认开） | 关闭增强，做对照实验 |
| `--aug-prob` | `0.5` | ColorJitter 的应用概率 `p` |
| `--aug-size` | `728` | 增强后图像尺寸 |
| `--aug-head-cam-substr` | `head` | 命中此子串（不区分大小写）的 cam 走 `policy_dm0`；填空串则不按子串识别，退化为「第一台 cam = head」 |
| `--aug-wrist-cam-substrs` | `wrist hand` | 命中其一的 cam 走 `policy_color_dm0` |

匹配规则按上面表的优先级：head substring → wrist substring → 位置 fallback
（`["dm0","color_dm0","color_dm0"]`）。也就是说，如果你给 cam key 起了
`observation.images.head_rgb` / `..._left_wrist_rgb` / `..._right_wrist_rgb`
这种 dexbotic 风格命名，就什么都不用调。

> 关闭增强：`--no-aug`
> 调小色彩抖动概率：`--aug-prob 0.3`

## 默认值（CLI）

> ⚠️ 注意：这里是 LeRobot 入口的 CLI 默认值，**不完全等于 dexbotic 原配方**。
> 主要差异是把 dexbotic 的 `per_device_train_batch_size=4 + grad_accum=4`（等效 16）
> 换成了 `--batch-size=16 + grad_accum=1`（等效 16，但 per-step 没有微批平均，
> 所以 wandb 上的 loss 曲线肉眼会更抖一点）。

| 参数 | 默认值 | dexbotic 对应 |
|------|--------|---------------|
| `--num-images` | 3 | `DM0DataConfig` ✓ |
| `--original-action-dim` | 23 | `DM0ActionConfig` ✓ |
| `--non-delta-mask` | `14 15` | `DM0ActionConfig`（gripper 维度保持绝对值）✓ |
| `--chunk-size` | 50 | `DM0ActionConfig` ✓ |
| `--lr` | `1e-4` | dexbotic = `2.5e-5`（差 4 倍） |
| `--decay-lr` | `1e-5` | dexbotic = `2.5e-6`（差 10 倍） |
| `--warmup-steps` | 1000 | ✓ |
| `--steps` | 30000 | dexbotic = 6000 |
| `--save-freq` | 2500 | ✓ |
| `--batch-size` | 16 | dexbotic per_device=4 × grad_accum=4 = 16（等效一致） |
| `--num-workers` | 4 | ✓ |
| `--lora-r` | 16 | ✓ |
| `lora_alpha` | 32（写死） | ✓ |
| `lora_dropout` | 0.05（写死） | ✓ |
| 图像增强 | head=`policy_dm0`, wrist=`policy_color_dm0`, p=0.5, size=728 | ✓ |

要严格复现 dexbotic：`--lr 2.5e-5 --decay-lr 2.5e-6 --steps 6000`，
（per-device batch 维持 dexbotic 的 4 的话，`--batch-size 4` 然后用 `accelerate config`
设 `gradient_accumulation_steps=4`）。

`bf16` / 多卡 / `gradient_accumulation_steps` 走 `accelerate config`，不在这里管。

## 多卡 / W&B / 后台训练示例

```bash
export HF_HOME=/path/to/hf_cache
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"
export WANDB_API_KEY=...

export BTPRJNAME=dm0_r1_pro
export BTJOBNAME=phase1

nohup accelerate launch \
  --multi_gpu --num_processes=8 --mixed_precision=bf16 \
  bt/dm0/train_dm0_r1_pro.py \
  --task=train \
  --dataset-repo=local/r1_pro_chassis_v3 \
  --dataset-root=/path/to/dataset \
  --norm-stats-path=./norm_stats/r1_pro_chassis_v3.json \
  --dm0-base=./checkpoints/DM0-base \
  --batch-size=16 --steps=30000 --save-freq=2500 --log-freq=10 \
  --output-root=/mnt/g/CKPT/${BTPRJNAME} \
  --job-name=${BTJOBNAME} \
  --wandb --wandb-project=${BTPRJNAME} \
  > /mnt/g/CKPT/${BTPRJNAME}_${BTJOBNAME}.log 2>&1 &
echo $! > /mnt/g/CKPT/${BTPRJNAME}_${BTJOBNAME}.pid
```

Phase-2 把 `--task=train` 换成 `--task=lora_train`，并加
`--phase1-ckpt=.../checkpoints/last/pretrained_model` 即可。

## 全部 CLI 参数

```bash
python bt/dm0/train_dm0_r1_pro.py --help
```

## 排错小抄

- **`HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`**：把本地路径塞给了 `--dataset-repo`。`--dataset-repo` 必须是 Hub 形式
  （本地数据用 `local/<任意名>` 占位即可），本地路径走 `--dataset-root`。
- **`RepositoryNotFoundError: 404` / 启动时 hang 在「downloading」**：八成是 HF cache 目录无效。
  确认 `HF_HOME` / `HF_DATASETS_CACHE` / `HF_HUB_CACHE` **都 export 了**，且 `mkdir -p` 过、目录可写。
- **loss 看起来比 dexbotic 抖**：见上面默认值表格，dexbotic 的 wandb step 是 4 个 micro-batch
  累积出来的（`grad_accum=4`），相当于每个 step 写的是 4 mini-batch loss 的均值；
  本入口默认 `grad_accum=1`，记录的是单 mini-batch loss，肉眼上更抖但训练是等价的。
  可以用 `accelerate config` 把 `gradient_accumulation_steps` 调到 4，并把
  `--batch-size` 调到 4，复现 dexbotic 的曲线观感。

## 相关文件

- `bt/dm0/train_dm0_r1_pro.py`：本入口（推荐使用）。
- `bt/dm0/run_dm0_freeze_vit.sh`：phase-1 一键脚本（HF cache + norm_stats + accelerate launch）。
- `bt/dm0/dm0_augmentations.py`：dexbotic 风格图像增强（`policy_dm0` / `policy_color_dm0`），
  并提供 `DM0AugmentedDataset` 包装。
- `examples/dm0/compute_norm_stats.py`：生成 `norm_stats.json` 的离线脚本。
- `examples/dm0/train_phase1_freeze_vit.sh` / `train_phase2_lora_vit.sh`：纯 shell 调用 `lerobot-train` 的等价版本（不走本 .py 入口，**不带图像增强**）。
- `src/lerobot/policies/dm0/`：DM0 policy 本体（`configuration_dm0.py` / `modeling_dm0.py` / `processor_dm0.py` / `dm0_transforms.py`）。
- `dexbotic/playground/benchmarks/real/r1_pro_dm0_freeze_lora.py`：原始 dexbotic 训练脚本，本入口的对照参考。
- `dexbotic/dexbotic/data/dataset/augmentations.py`：原始 dexbotic 增强实现，`dm0_augmentations.py` 的对照参考。
