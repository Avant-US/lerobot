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

```
lerobot-eval \
  --output_dir=/logs/ \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.path=lerobot/pi05_libero_finetuned   \
  --policy.n_action_steps=10 \
  --output_dir=./eval_logs/ \
  --env.max_parallel_tasks=1
```
```
lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.type=groot2   \
  --policy.n_action_steps=10 \
  --output_dir=./outputs_eval/ \
  --env.max_parallel_tasks=1
```

训练
```bash
# 后台 + 持久：nohup 忽略挂起信号，关掉 VS Code/终端后训练仍继续。
# 原控制台日志（含 accelerate / 各 rank 的 logging）写入 /mnt/g/lrb.log。
# 查看：tail -f /mnt/g/lrb.log   结束：kill $(cat /mnt/g/lrb_train.pid)
export NCCL_DEBUG=TRACE
export HF_DEBUG=1
export TRANSFORMERS_VERBOSITY=debug
export HF_HUB_VERBOSITY=debug
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export HF_HOME="/home/Luogang/hfhome/"
export BTPRJNAME="lrb_grt_1"
export BTJOBNAME="r2"
nohup accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  $(which lerobot-train) \
  --output_dir=/mnt/g/CKPT/VLA/Libero/${BTPRJNAME}_${BTJOBNAME}/ \
  --save_checkpoint=true \
  --batch_size=32 \
  --steps=30000 \
  --save_freq=500 \
  --log_freq=50 \
  --policy.push_to_hub=false \
  --policy.type=groot2 \
  --policy.repo_id=nvidia/GR00T-N1.5-3B \
  --policy.tune_diffusion_model=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --wandb.enable=true \
  --wandb.project=${BTPRJNAME} \
  --wandb.disable_artifact=true \
  --job_name=${BTJOBNAME} \
  > /mnt/g/CKPT/VLA/Libero/${BTPRJNAME}_${BTJOBNAME}.log 2>&1 &
echo $! > /mnt/g/CKPT/VLA/Libero/${BTPRJNAME}_${BTJOBNAME}.pid
echo "训练已在后台启动，PID=$(cat /mnt/g/CKPT/VLA/Libero/${BTPRJNAME}_${BTJOBNAME}.pid)，日志：/mnt/g/CKPT/VLA/Libero/${BTPRJNAME}_${BTJOBNAME}.log"




accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  $(which lerobot-train) \
  --output_dir=/mnt/g/outputs_lrb_grt_1/ \
  --save_checkpoint=true \
  --batch_size=8 \
  --steps=30000 \
  --save_freq=8 \
  --log_freq=5 \
  --policy.push_to_hub=false \
  --policy.type=groot2 \
  --policy.repo_id=nvidia/GR00T-N1.5-3B \
  --policy.tune_diffusion_model=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --wandb.enable=true \
  --wandb.project=lrb_grt_1 \
  --wandb.disable_artifact=true \
  --job_name=btgrt_lrb02
```
eval时不要设置n_action_steps,以便使其的chunk_size与chekpoint中的一致.而因为lerobot-train时没指定,所以应是默认值50.
batch_size > 1 会出问题.
```bash
export HF_DEBUG=1
export TRANSFORMERS_VERBOSITY=debug
export HF_HUB_VERBOSITY=debug
export TORCH_DISTRIBUTED_DEBUG=DETAIL


export CUDA_VISIBLE_DEVICES=7
export BTPRJNAME="lrb_grt_1"
export BTJOBNAME="r2"
export MUJOCO_GL=egl
lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.obs_type=pixels_agent_pos \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --policy.path=/mnt/g/CKPT/VLA/Libero/${BTPRJNAME}_${BTJOBNAME}/checkpoints/030000/pretrained_model \
  --policy.device=cuda \
  --policy.use_amp=false \
  --seed=1000 \
  --output_dir=/mnt/g/CKPT/VLA/Libero/${BTPRJNAME}_${BTJOBNAME}$/eval/spatial030000/


export CUDA_VISIBLE_DEVICES=3
export BTPRJNAME="lrb_grt_1"
export BTJOBNAME="r2"
export MUJOCO_GL=egl
lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --policy.path=/mnt/g/CKPT/VLA/Libero/${BTPRJNAME}_${BTJOBNAME}/checkpoints/030000/pretrained_model \
  --policy.device=cuda \
  --policy.use_amp=false \
  --seed=1000 \
  --output_dir=/mnt/g/CKPT/VLA/Libero/${BTPRJNAME}_${BTJOBNAME}$/eval/spatial030000_2/



export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.obs_type=pixels_agent_pos \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --policy.path=/mnt/g/CKPT/VLA/Libero/lrb_grt_1_r1/checkpoints/030000/pretrained_model \
  --policy.device=cuda \
  --policy.use_amp=false \
  --seed=1000 \
  --output_dir=/mnt/g/CKPT/VLA/Libero/lrb_grt_1_r1/eval/spatial030000/


export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=3
lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --policy.path=/mnt/g/CKPT/VLA/Libero/lrb_grt_1_r1/checkpoints/030000/pretrained_model \
  --policy.device=cuda \
  --policy.use_amp=false \
  --seed=1000 \
  --output_dir=/mnt/g/CKPT/VLA/Libero/lrb_grt_1_r1/eval/spatial030000_2/

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=1
lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.obs_type=pixels_agent_pos \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --policy.path=/mnt/g/CKPT/VLA/Libero/lrb_grt_1_r1/checkpoints/017000/pretrained_model \
  --policy.device=cuda \
  --policy.use_amp=false \
  --seed=1000 \
  --output_dir=/mnt/g/CKPT/VLA/Libero/lrb_grt_1_r1/eval/spatial017000/

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=5
lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.obs_type=pixels_agent_pos \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --policy.path=/mnt/g/CKPT/VLA/Libero/lrb_grt_1_r1/checkpoints/024000/pretrained_model \
  --policy.device=cuda \
  --policy.use_amp=false \
  --seed=1000 \
  --output_dir=/mnt/g/CKPT/VLA/Libero/lrb_grt_1_r1/eval/spatial024000/

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=2
lerobot-eval \
  --env.type=libero \
  --env.task=libero_object \
  --env.obs_type=pixels_agent_pos \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --policy.path=/home/Luogang/SRC/Robot/lerobot/outputs/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --policy.use_amp=false \
  --seed=1000 \
  --output_dir=./outputs_evalobj/

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=4
lerobot-eval \
  --env.type=libero \
  --env.task=libero_goal \
  --env.obs_type=pixels_agent_pos \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --policy.path=/home/Luogang/SRC/Robot/lerobot/outputs/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --policy.use_amp=false \
  --seed=1000 \
  --output_dir=./outputs_evalgoal/


export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=6
lerobot-eval \
  --env.type=libero \
  --env.task=libero_10 \
  --env.obs_type=pixels_agent_pos \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --policy.path=/home/Luogang/SRC/Robot/lerobot/outputs/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --policy.use_amp=false \
  --seed=1000 \
  --output_dir=./outputs_eval10/
```
  # 强制重新编译安装flash-attn
  ```
cd /home/Luogang/SRC/Robot/lerobot && \
export PATH="/home/Luogang/SRC/Robot/lerobot/lerobot-venv/bin:$PATH" && \
export CUDA_HOME=/usr/local/cuda && \
export FLASH_ATTENTION_FORCE_BUILD=TRUE && \
export FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE && \
export MAX_JOBS=32 && \
pip uninstall -y flash-attn && \
pip cache remove flash_attn 2>/dev/null || true && \
pip install 'flash-attn>=2.5.9,<3.0.0' --no-build-isolation --no-cache-dir
```