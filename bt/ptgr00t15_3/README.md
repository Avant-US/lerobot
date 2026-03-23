# GR00T N1.5 小规模预训练（libero 随机37条）

脚本：

- 训练：`bt/ptgr00t15_3/train_groot_n15_libero_small.py`
- 评估：`bt/ptgr00t15_3/eval_groot_n15_dataset.py`

## 功能

- 使用 `HuggingFaceVLA/libero` 数据集
- 随机抽取 `37` 个 episode（可通过 `--sample-size` 修改）
- 使用已安装的 `lerobot` 训练 `GR00T N1.5`
- 默认关闭 wandb、关闭推送 Hub，做本地小规模预训练

## 快速运行

```bash
python bt/ptgr00t15_3/train_groot_n15_libero_small.py --steps 1 --batch-size 1
```

## 常用参数

- `--sample-seed`：抽样随机种子
- `--episodes-file`：抽样结果 JSON 输出路径
- `--dataset-root`：本地缓存目录
- `--output-root`：训练输出根目录
- `--tune-diffusion-model`：是否训练 diffusion 部分（默认关闭）
- `--no-use-bf16`：禁用 bf16

## LIBERO评估

```bash
python bt/ptgr00t15_3/eval_groot_n15_dataset.py \
  --policy-path nvidia/GR00T-N1.5-3B \
  --n-episodes 1 \
  --batch-size 1
```

评估脚本使用 `groot2` policy，并打印：

- LIBERO-Spatial
- LIBERO-Object
- LIBERO-Goal
- LIBERO-Long
- 四项平均分

默认评估四个 suite 的全部 task。可用 `--max-tasks-per-suite` 做快速调试。

```shell
lerobot-eval \
  --policy.path="eagle2hg-processor-groot-n1p5" \
  --env.type=libero \
  --env.task=libero_object,libero_spatial,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10
```
```
lerobot-eval \
  --output_dir=/logs/ \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.type=groot2   \
  --policy.n_action_steps=10 \
  --output_dir=./eval_logs/ \
  --env.max_parallel_tasks=1
```


