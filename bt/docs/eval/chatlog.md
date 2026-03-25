Exported on 3/25/2026 at 14:09:57 GMT+8 from Cursor (2.6.21)
---



# LIBERO或其它,Train的时候是否要「在仿真里交互或对齐分布」

**User**

Train 的情况下,什么时候才需要「在仿真里交互或对齐分布」

---

**Cursor**


在 **Train** 场景里，“要在仿真里交互”和“要对齐分布”通常是两件事，但经常一起出现。

## 一、什么时候需要「在仿真里交互」

指训练过程中**策略（或别的模型）要在仿真里 stepping、拿新转移**，而不只是读磁盘上的 HDF5。

常见情况包括：

1. **在线/在环算法**  
   - **DAgger / 迭代 BC**：用当前策略 rollout，再用专家（或修正动作）补数据。  
   - **RL、RLHF、带环境奖励的微调**：必须和环境交互。  
   - **世界模型 / 动力学模型**：用 rollout 拟合下一步状态或奖励。

2. **在仿真里造数据或增强**  
   - 域随机化、改相机/光照/物体位姿再采轨迹。  
   - 用仿真生成「几乎不可能全靠人类录」的大量变体。

3. **训练循环里周期性做「仿真验证」**  
   - 严格说算监控/早停，不是必须用仿真梯度；但很多 pipeline 会把「在 sim 里跑几步成功率」写进训练脚本。

**不需要仿真交互的典型情况**：  
纯 **离线模仿学习**——只从已有 HDF5（或 LeRobot 数据集）里 sample 批次、算监督损失，**从不** `env.step()`。

---

## 二、什么时候需要「对齐分布」

这里指：**训练时见到的状态/任务分布**和**你真正关心的分布**（通常是 **eval 基准**或 **真机**）一致或尽量接近。

常见情况包括：

1. **和 LIBERO 官方评估协议对齐**  
   - Eval 用 `init_states` 里那套 **`.pt` 复位分布**。  
   - 若你在仿真里 **on-policy 采数据**（例如 DAgger），往往也会刻意用**同一套 init**，这样新数据才和测试分布一致。

2. **缓解协变量偏移（covariate shift）**  
   - 纯离线 BC：训练分布 ≈ **专家轨迹**上的分布；策略执行后会漂到 **自己 induced 的状态分布**，误差会累积。  
   - 需要「对齐」时，常见做法是：**DAgger 式补数据**、**过滤/加权离策略样本**、或 **噪声/初始化随机化** 让策略别过拟合演示流形上的一小块。

3. **Sim2Real 或跨仿真**  
   - 对齐的是 **仿真统计量 ↔ 真机**（延迟、摩擦、相机噪声等），属于更广义的「分布对齐」。

**不一定刻意对齐的情况**：  
只做 **标准离线 BC**，且你只关心「在固定 HDF5 划分上 loss 低」；此时对齐的是 **演示数据本身的分布**，未必和 **eval 的 init_states 分布**逐项一致（论文里有时会因此和官方数字有差距）。

---

## 三、和 LIBERO / LeRobot 怎么对应

| 训练方式 | 一般要仿真？ | 分布对齐在关心什么 |
|----------|--------------|-------------------|
| 只读 `datasets` HDF5 或 `HuggingFaceVLA/libero` 做监督 | **否** | 与 **演示数据** 对齐（隐含） |
| DAgger / RL / 仿真里续标数据 | **是** | 与 **eval 的 init / 任务** 对齐很重要 |
| 仅用仿真做「训练中途 eval」 | 评估时用 sim，**梯度仍可不来自 sim** | 与 **基准测试协议** 对齐 |

**简短结论**：  
- **要仿真交互** → 只要训练更新依赖 **当前策略在环境里 rollout 出来的新数据或奖励**。  
- **要对齐分布** → 当你在乎 **和官方 eval 一致**、或 **减轻离线 IL 的分布偏移**、或 **Sim2Real** 时；纯离线 HDF5 训练默认只对齐「演示集」，不自动对齐「测试用 init_states 分布」，除非你的数据或增广本来就是按同一协议采的。

## LIBERO 上一般的Train的例子是否有「在仿真里交互」和「对齐分布」
---

**User**

基于当前的lerobot, 若我训练时使用这段命令 : @bt/ptgr00t15_3/README.md:76-92 . 需要「在仿真里交互」和「对齐分布」吗? 实际上运行的代码里又有没有用到「在仿真里交互」和「对齐分布」?
例程如下:
``` bash
# Using a multi-GPU setup
accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  $(which lerobot-train) \
  --output_dir=./outputs \
  --save_checkpoint=true \
  --batch_size=32 \
  --steps=30000 \
  --save_freq=1000 \
  --log_freq=100 \
  --policy.push_to_hub=false \
  --policy.type=groot2 \
  --policy.repo_id=nvidia/GR00T-N1.6-3B \
  --policy.tune_diffusion_model=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --wandb.enable=false \
  --job_name=btgrt_lrb01
```

---

**Cursor**


正在追踪 `lerobot-train` 在 `groot2` + `HuggingFaceVLA/libero` 配置下是否会实例化仿真环境。




结论可以直接说清楚：你这条 **`lerobot-train` 命令既没有配置 `--env`，也没有在训练循环里做「主动分布对齐」**；训练就是 **离线读 `HuggingFaceVLA/libero`**。

---

### 1. 需不需要「在仿真里交互」？

**按这条命令：不需要。**  
训练更新只来自 **DataLoader 里的固定数据集**，没有 `env.step()` 参与反向传播。

依据在 `lerobot_train.py`：

- 只有同时满足 **`eval_freq > 0` 且 `cfg.env is not None`** 时，主进程才会 `make_env`（用于训练中周期性 eval）：

```230:236:/home/Luogang/SRC/Robot/lerobot/src/lerobot/scripts/lerobot_train.py
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
```

- 你的 README 里没有 `--env.type=libero` 等参数；`TrainPipelineConfig` 里 **`env` 默认是 `None`**：

```37:39:/home/Luogang/SRC/Robot/lerobot/src/lerobot/configs/train.py
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
```

因此 **`eval_env` 一直是 `None`**，不会在训练里建 LIBERO。

- 主循环明确是 **offline training**，每步只有 `next(dl_iter)` + `update_policy`：

```407:408:/home/Luogang/SRC/Robot/lerobot/src/lerobot/scripts/lerobot_train.py
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
```

```411:416:/home/Luogang/SRC/Robot/lerobot/src/lerobot/scripts/lerobot_train.py
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
```

- 训练中 eval 也 **只有在 `cfg.env` 非空** 时才会跑：

```477:478:/home/Luogang/SRC/Robot/lerobot/src/lerobot/scripts/lerobot_train.py
        if cfg.env and is_eval_step:
```

所以你 **不会**在 `lerobot-train` 里触发「在仿真里交互式训练」；默认的 `eval_freq=20000` 也不会生效，因为 **`cfg.env` 为 `None`**。

---

### 2. 需不需要「对齐分布」？

**这条命令没有单独的「对齐 eval 分布」步骤。**

含义拆开说：

- **训练分布**：由 **`HuggingFaceVLA/libero` 里已经写好的帧/episode** 决定；代码只是采样 mini-batch，**没有**再读 LIBERO 的 `init_states`、也没有 DAgger/再采集去对齐 `lerobot-eval` 的复位分布。
- **和以后 `lerobot-eval` 的关系**：若你事后用 LIBERO 仿真 eval，那是 **另一条管线**；离线训练本身 **不保证** 与 eval 时 `init_states` 分布一致，除非你的策略与数据/增广恰好覆盖得好——这不是当前 `lerobot_train.py` 里显式做的模块。

所以：  
- **不需要你额外准备「对齐分布」的流程** 才能跑这条命令；  
- **也没有**在代码里实现「专门对齐 LIBERO eval 协议」的逻辑。

---

### 3. 简要对照表

| 项目 | 你的这条 `lerobot-train` 命令 |
|------|-----------------------------|
| 在仿真里交互（用于学习） | **无**（无 `env`，只有 DataLoader） |
| 训练中周期性仿真 eval | **无**（`cfg.env is None`，`if cfg.env and is_eval_step` 不成立） |
| 显式「对齐 eval 分布」 | **无**（仅离线监督学习） |
| 实际用的数据 | **`--dataset.repo_id=HuggingFaceVLA/libero`**（Hub 上的 LeRobot 格式） |

若将来希望在 **训练过程中** 也在 LIBERO 里 rollout 看成功率，需要额外加例如 `--env.type=libero` 以及相应的 `task` 等，并接受主进程会建仿真环境、在 `eval_freq` 步上做 eval（仍不是用仿真梯度训练，除非你再改代码接 RL/DAgger）。
---

# LIBERO上的eval是否要「对齐分布」

**User**

那在做evaluation前是不是要「对齐分布」?基于现在的lerobot应该怎样做

---

**Cursor**


**不需要**在 evaluation 前单独跑一步名叫「对齐分布」的流程。  
评估时重要的是：**让仿真里的观测/动作接口与训练时一致**（有时叫 *protocol alignment*），否则不是「分布没对齐」的玄学问题，而是 **输入输出不匹配** 会直接拉低成功率。

下面按 **当前 LeRobot** 说明该怎么做。

---

## 1. 概念上：eval 前要「对齐」什么？

| 要对齐的 | 含义 | 不做会怎样 |
|----------|------|------------|
| **观测空间** | 键名、相机数量、图像尺寸、state 维度与训练一致 | 策略吃错输入或 shape 报错 |
| **动作空间** | 维度、相对/绝对控制、是否 chunk（`n_action_steps` 等）与训练一致 | 动作语义错了，成功率差 |
| **任务语义** | 语言/任务描述若模型用了，eval 侧要能提供同类信息（若有） | VLA 类模型可能懵 |
| **LIBERO 资源** | BDDL、init_states、assets 齐全 | 环境起不来或复位不对 |

这些 **都不是** 在 eval 前额外算一个「分布」再喂给模型，而是 **配置与 checkpoint 一致**。

---

## 2. 基于现在 LeRobot，eval 前具体怎么做？

**（1）用训练保存的整套配置**

```bash
lerobot-eval \
  --policy.path=/path/to/.../pretrained_model \
  ...
```

`pretrained_model` 里的 `config.json`（及 processor 等）会带上训练时的策略设定；**优先以 checkpoint 为准**，不要和训练时矛盾的 CLI 覆盖（除非你知道在改什么）。

**（2）环境侧与训练数据对齐（LIBERO 示例）**

训练用的是 `HuggingFaceVLA/libero` 时，数据里是 **双相机 + state + 7D action** 等（见数据集 card）。eval 时建议显式对齐，例如：

- `--env.obs_type=pixels_agent_pos`（若训练数据含 `observation.state` / 机器人状态）
- `--env.camera_name=...` 与数据里 `image` / `image2` 的语义一致（LeRobot 的 `LiberoProcessorStep` 会映射到 `observation.images.*`）
- `--env.control_mode=relative` 或 `absolute`：**必须和演示/训练时一致**（LIBERO 常见是相对）
- `--env.observation_height` / `--env.observation_width`：若与训练分辨率差很多，视觉策略会 domain shift；尽量与训练一致或接受性能下降

你 README 里已有 `--policy.n_action_steps=10`，需与 **训练时 `n_action_steps`（或 chunk）** 一致，否则动作执行方式不对。

**（3）`rename_map`（若训练用过）**

训练时若用了 `--rename_map=...` 把某些 key 映到策略期望的名字，eval 也要 **同样传入**，否则观测 key 对不上。检查 `train_config.json` 或当时命令行。

**（4）LIBERO 磁盘/配置**

- 保证 `~/.libero/config.yaml` 已写好，或你用过的 `--env.libero_dataset_path` / `LEROBOT_LIBERO_DATASET_PATH`（我们之前加的非交互初始化）。
- **Eval 不读 HDF5 演示**，但仍要 **assets + bddl + init_states** 能访问到。

**（5）设备与 AMP**

- `--policy.device=cuda` 等与训练/推理习惯一致；`--policy.use_amp` 若训练关了，eval 也建议一致，避免数值差异（通常不是主因）。

---

## 3. 怎样自检「有没有对齐好」？

1. **先能跑通**：无 shape / key 报错，能完整 rollout。  
2. **对比单步**：若有疑问，可在 sim 里打印一帧预处理后的 observation 结构，与 `lerobot-dataset-viz` / 训练 dataloader 里 batch 的 key 对比。  
3. **弱基线**：同一 checkpoint 在 **与训练相同的 task 子集** 上 eval，再扩展到全 benchmark。

---

## 4. 一句话

- **Eval 前没有单独的「分布对齐」步骤**；要做的是 **策略配置 + LIBERO 环境配置 + 观测/动作/分辨率与控制模式** 与 **训练时一致**。  
- **做法**：用 **`--policy.path=.../pretrained_model`** 加载训练配置，再按上表把 **`--env.*` / `--policy.n_action_steps` / `rename_map`** 对齐训练记录（`train_config.json` 或当时命令）。这样就是在当前 LeRobot 下最实在的「对齐」。
---

# 关于 n_action_steps


**User**

之前提到的 lerobot-train 命令中好像没设置 n_action_steps, 此情况下, 它的n_action_steps是多少呢? 与lerobot-eval时的是否一致?

---

**Cursor**


结论先说：**类上的默认值是 `n_action_steps=50`（且 `chunk_size=50`）**。若训练命令里没写，实际以 **当时加载进来的策略配置** 为准（常见是预训练 `config.json` + CLI 覆盖）；**eval 里你写的 `--policy.n_action_steps=10` 若与 checkpoint 里不一致，就和训练不一致。**

---

## 1. 代码里的默认值（groot2）

```31:34:/home/Luogang/SRC/Robot/lerobot/src/lerobot/policies/groot2/configuration_groot.py
    # Basic policy settings
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50
```

也就是说：**在只谈 `GrootConfig` 类默认值时，`n_action_steps` 是 50**，且必须满足 `n_action_steps <= chunk_size`。

---

## 2. 训练时「没写 `n_action_steps`」实际是多少？

`lerobot-train` 里策略一般来自 **`--policy.path=...`** 的 `from_pretrained`（你 README 里若最终也是指向某个 Hub/本地目录），会 **先读该目录下的 `config.json`**，再叠 **CLI 的 `--policy.*`**。

因此：

- **没写 `--policy.n_action_steps`** → 用 **预训练/ checkpoint 里保存的值**；若那份也没有改过，则往往是 **Hub 模型自带配置**。
- 若某次训练完全是 **仅靠 `GrootConfig` 默认构造**（没有从 hub 合并），才会是上面的 **50**。

**最可靠做法**：看你这次 run 的  
`outputs/.../checkpoints/.../pretrained_model/config.json`  
里的 `"n_action_steps"`（以及 `"chunk_size"`），那就是训练时写进 checkpoint 的值。

---

## 3. 和 `lerobot-eval` 是否一致？

- **Eval** 用 `--policy.path=.../pretrained_model` 时，会先加载 **checkpoint 里的 `config.json`**。
- 若你再写 **`--policy.n_action_steps=10`**，属于 **用 CLI 覆盖** checkpoint 里的 `n_action_steps`。
- 因此：
  - **若训练时实际是 50，eval 强行改成 10 → 不一致**（每步执行的动作步数不同，行为会变）。
  - **要一致**：eval **不要改** `n_action_steps`，或显式设为 **与 `pretrained_model/config.json` 相同**。

---

## 4. 简要对照

| 场景 | `n_action_steps` |
|------|------------------|
| `GrootConfig` 类默认 | **50** |
| 训练未传 `--policy.n_action_steps` | **以 `policy.path` 的 config + 合并结果为准**（常为预训练里的值，需打开 `config.json` 确认） |
| Eval 传 `--policy.n_action_steps=10` | **10（覆盖 checkpoint）**；若与训练不同则 **不一致** |

**建议**：以 **`pretrained_model/config.json`** 为准；eval 要么不传 `n_action_steps`，要么与其中数值相同。
---

# lerobot-eval 的计算性能优化


**User**

该机子上有8张Nvidia H200 GPU, 在调用 lerobot-eval 命令时, 为了能最大化使用GPU的算力和显存, 更快地完成evaluation, 想一想 应设置些什么选项? 怎样设置? 为什么?

---

**Cursor**


查看 `lerobot-eval` 的并行模型与可调参数。




下面按 **当前 `lerobot-eval` 的实现**（单进程、策略默认占 **一张** GPU）来说明怎么尽量榨干算力/显存，以及 8×H200 时怎么分工。

---

## 1. 先认清瓶颈：`lerobot-eval` 在干什么

- **策略推理**：`policy.select_action` 在 **一个** PyTorch device 上跑（`--policy.device=cuda`，一般是 `cuda:0`）。
- **LIBERO**：MuJoCo 仿真主要在 **CPU**；`eval.batch_size` 是 **单个 (suite, task_id)** 上的 **VectorEnv 并行环境数**，决定 **一次 forward 的 batch 维有多大**。
- **多任务并行**：`--env.max_parallel_tasks>1` 时用 **线程池** 同时跑多个 task，**多个线程共用一个 GPU** 上的同一份 policy，CUDA 上往往会 **串行排队**，不一定更快，还可能和仿真线程抢资源。

因此：**单卡 H200 上，优先用「大 `eval.batch_size`」把 GPU 吃满**；**8 张卡要靠「多进程/多任务拆分」**，脚本本身没有内置 8-GPU 数据并行 eval。

---

## 2. 建议优先调的选项（单卡把 GPU 吃满）

### （1）`--eval.batch_size`（最重要）

- **作用**：同一 task 上 **一次 policy forward 处理多少条并行环境**，直接增大 **推理 batch**，提高 Tensor Core 利用率、摊薄 kernel 启动开销。
- **怎么设**：从 **16 → 32 → 64 → …** 往上加，直到 **OOM** 或收益变平；H200 显存大，GROOT 这类模型往往可以开到 **较大 batch**（具体上限取决于分辨率、`n_action_steps`、是否保存视频等）。
- **约束**：`EvalConfig` 要求 **`eval.n_episodes >= eval.batch_size`**，否则会报错；且若 `batch_size` 远大于实际用到的 episode 数，会白建环境、变慢（见 `EvalConfig.__post_init__` 的说明）。

示例（单 task 或你接受「每 task 至少跑满一整批」）：

```bash
--eval.batch_size=64 \
--eval.n_episodes=64   # 至少 ≥ batch_size；全 benchmark 时 episode 总数会按 task 重复计
```

你现在是 `n_episodes=10、batch_size=1`，GPU 几乎总在 **batch=1** 上跑，**最浪费 H200**。

### （2）`--policy.device=cuda`

- **作用**：确保用 GPU 推理（可写 `cuda:0` 指定卡）。
- **为什么**：CPU 推理对大模型 eval 会慢几个数量级。

### （3）`--policy.use_amp=true`（在可用时）

- **作用**：`eval_main` 里对 `device_type=cuda` 走 **`torch.autocast`**，常用 **bf16/fp16** 降低显存、提高吞吐（视策略与算子支持而定）。
- **注意**：若某策略/算子不稳定，再关回 `false`。

### （4）`--env.max_parallel_tasks`：建议 **先 1，再 profile**

- **作用**：多个 **不同 task** 用多线程并行跑。
- **为什么不一定越大越好**：**多线程共享单 GPU** 时，forward 往往 **不能真正 8 路并行**，还可能增加 **线程切换与显存峰值**；仿真若在 CPU 上很重，有时 **略大于 1** 能掩盖一点 I/O，但对 **3B 级 GPU 推理**，常见最优是 **单线程任务 + 大 `eval.batch_size`**。
- **建议**：先在 **`max_parallel_tasks=1`** 下把 **`eval.batch_size` 拉高**；若 profiling 显示 GPU 很空、CPU 很满，再试 **小幅度**增加 `max_parallel_tasks`（例如 2～4），对比 wall-clock。

### （5）`--eval.use_async_envs=true`（可选）

- **作用**：用 **多进程** VectorEnv，仿真与主进程 **解耦**，在 **batch 较大、CPU 核多** 时有时能减轻 GIL/CPU 瓶颈。
- **代价**：进程多、内存占用高、调试更难；需实测。

### （6）视频与日志

- **作用**：`lerobot_eval` 里 `max_episodes_rendered=10` 写死在 `eval_main`，会 **编码/写盘**，占 CPU/IO。
- **为什么**：若你只关心成功率、要极致速度，需要 **改代码** 关掉或减小渲染 episode 数（当前 CLI 若未暴露，只能从脚本改）。这是 **除 batch 外**常见的耗时项。

---

## 3. 8×H200 怎么「都用上」

**现状**：一次 `lerobot-eval` **默认只密集用一张卡**（一个 policy 在一个 device 上）。

**可行做法**：

1. **数据并行（推荐思路）**：把 task 划分成 8 份，起 **8 个进程**，例如：
   - `CUDA_VISIBLE_DEVICES=0 lerobot-eval ... --env.task=libero_spatial --env.task_ids=0,1,...`
   - 每张卡一个进程，各跑一部分 `task_ids` 或不同 suite。  
   最后合并 `eval_info.json`（需自己写小脚本或手动汇总）。

2. **不要指望** `max_parallel_tasks=8` 自动用满 8 张卡——它仍是 **单进程单默认 GPU**。

---

## 4. 和你当前命令的对比（为何慢）

| 当前 | 问题 |
|------|------|
| `eval.batch_size=1` | 策略几乎总是 batch=1，**GPU 利用率极低** |
| `max_parallel_tasks=8` | 8 线程抢 **一张** GPU，**往往不如** 单线程 + 大 batch |
| `n_episodes=10` | 与「拉大 batch」一起时，注意必须 **`n_episodes >= batch_size`** |

---

## 5. 推荐的一条「单卡先优化」示例方向

```bash
lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=32 \
  --eval.n_episodes=32 \
  --policy.path=.../pretrained_model \
  --policy.device=cuda \
  --policy.use_amp=true \
  --output_dir=./outputs_eval/ \
  ...
```

再按显存把 `eval.batch_size` 往上扫；确认 **`n_episodes` 与评测协议**（每个 task 要多少条 episode）是否允许改成 32/64。若必须每个 task 只评 10 条，则 **`batch_size` 最大只能 10**，此时 GPU 仍较难吃满，这是 **协议约束** 不是 H200 问题。

---

**总结**：  
- **最大化单卡**：主要靠 **`eval.batch_size` + `policy.use_amp` + `policy.device=cuda`**，并慎用 **`max_parallel_tasks`**。  
- **最大化 8 卡**：需 **多进程 + `CUDA_VISIBLE_DEVICES` 拆分任务**，因为 **当前 eval 没有内置多 GPU 模型并行**。
- **注意**, 大部分时候我们都的计算性能优化都是基于evaluation能真实反映训练出来的模型的算法效果的,而不能仅仅只为计算性能,比如:如果训练用全精度的,eval时就不能用半精度,use_amp就必须falsee
  - 因此,请接着看下面**lerobot-eval 忠实度为约束的计算性能优化**
---

# lerobot-eval 忠实度为约束的计算性能优化


**User**

如果既要最大化利用显卡以及加快evaluation速度, 同时也要保证evaluation符合模型 /home/Luogang/SRC/Robot/lerobot/outputs/checkpoints/last/pretrained_model 的训练配置, 以evaluation的真实性与精度优先, 应该怎么做?

---

**Cursor**


查看该 checkpoint 的 `config.json` 与训练配置，以便给出与训练一致且可加速的 eval 建议。




结合你这份 checkpoint 里的 **`pretrained_model/config.json`** 和 **`train_config.json`**，下面按「**真实性 / 与训练一致优先**」说明该怎么设，以及哪些项可以**只加速、不该动推理语义**。

---

## 1. 必须先与 checkpoint 一致（否则 eval 不等价于训练时的策略）

### （1）**不要**再写 `--policy.n_action_steps=10`

Checkpoint 里是：

```45:46:/home/Luogang/SRC/Robot/lerobot/outputs/checkpoints/last/pretrained_model/config.json
    "chunk_size": 50,
    "n_action_steps": 50,
```

训练配置里同样是 **50**。你 eval 用 **10** 等于改了「一次预测多少步、队列里执行多少步」，**和训练时的动作执行方式不一致**，成功率对比会失真。

**做法**：删掉该 CLI 项，让 eval 从 `pretrained_model/config.json` 加载 **`n_action_steps=50`**。

### （2）**精度 / 数值路径：`use_amp` 与 checkpoint 一致**

Checkpoint：

```36:37:/home/Luogang/SRC/Robot/lerobot/outputs/checkpoints/last/pretrained_model/config.json
    "device": "cuda",
    "use_amp": false,
```

`lerobot-eval` 里 `use_amp=true` 会走 `torch.autocast`，可能带来与训练时略有不同的数值路径。你要求 **真实性优先** 时，建议：

**做法**：显式 **`--policy.use_amp=false`**（或不传，若加载后仍是 false），**不要为了加速擅自开 AMP**。

（`use_bf16: true` 是 GROOT 侧训练相关字段；eval 仍以 checkpoint + `eval_main` 实际用的 autocast/精度为准，**不要**用「训练快」的理由在 eval 上乱开混合精度。）

### （3）**随机种子与训练一致（可复现性）**

`train_config.json` 里 **`seed`: 1000**。eval 里 `start_seed` 会用到 pipeline 的 seed。

**做法**：**`--seed=1000`**，与训练一致，便于复现；换 seed 不改「训练一致性」，但会改 eval 随机性。

### （4）**环境观测与数据对齐（LIBERO + 该数据集）**

训练数据是 **`HuggingFaceVLA/libero`**，特征为双图 **256×256** + **8 维 state**（见 checkpoint `input_features`）。  
LeRobot 里 `LiberoEnv` 默认渲染有时是 **360×360**，若与 **256** 不一致，会带来**分布偏移**（虽后面可能再 resize 到 `image_size` 224，但源头分辨率仍可能影响）。

**做法（推荐）**：

- **`--env.obs_type=pixels_agent_pos`**（与 8 维 `observation.state` 一致）
- **`--env.observation_height=256 --env.observation_width=256`**，与 checkpoint 里视觉特征形状一致  
- 相机名保持默认 **`agentview_image,robot0_eye_in_hand_image`**（与数据里 `image` / `image2` 语义一致）
- **`--env.control_mode=relative`**（LIBERO 演示与常见设置；若你曾用 absolute 训练再改）

`train_config` 里 **`rename_map` 为空**，eval 也不要乱加 `rename_map`。

### （5）**指标「精度」：episode 数**

训练配置里 eval 预设是 **`n_episodes=50`**（见 `train_config.json` 的 `eval` 段）。你现在 **10** 条方差大，**统计意义上更不准**。

**做法**：在算力允许下 **提高 `eval.n_episodes`**（例如 50 或与论文一致）；这与「模型前向是否一致」无关，但影响 **评估结论的可信度**。

---

## 2. 可以动、且**不改变**策略数学含义的加速手段

这些只影响「**一次并行多少条环境、调度方式**」，**不改变**单步 `select_action` 的输入输出定义：

| 项 | 建议 | 原因 |
|----|------|------|
| **`--eval.batch_size`** | 在 **`eval.n_episodes ≥ batch_size`** 前提下尽量加大（如 32、50、64…） | 同一 policy，只是把多条 env 拼 batch 做推理，**语义与 batch=1 一致**，GPU 利用率更高。 |
| **`--env.max_parallel_tasks`** | **真实性优先时先用 1**；若 profiling 确认 GPU 仍很空再试 2～4 | 多线程共单卡易造成排队与非确定性；**不提高「单步推理忠实度」**。 |
| **`--eval.use_async_envs`** | 可选 **true**，在 CPU 仿真重时有时更快 | 一般**不改变**策略输入输出；若遇奇怪问题再关回 false。 |
| **多卡** | 8 个进程 + **`CUDA_VISIBLE_DEVICES`** 拆 `task` / **`--env.task_ids`**，**同一 `policy.path`** | 每进程仍是**同一 checkpoint、同一超参**，只是**任务分片**，不牺牲单模型忠实度。 |

---

## 3. 不推荐为提速牺牲「与训练一致」的项

- 改 **`n_action_steps` / `chunk_size`**（应用 checkpoint 的 50）  
- 改 **`use_amp`**（与 checkpoint 的 false 对齐）  
- 改 **控制模式、观测类型、分辨率** 而不对照训练数据与 checkpoint  
- 为省事关掉与 **state / 归一化** 相关的处理（你这里是 **MEAN_STD**，需与训练时同一套统计；通常由 checkpoint 的 processor 处理，**不要随意覆盖**）

---

## 4. 一条「忠实 + 尽量快」的示例命令骨架

```bash
lerobot-eval \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.obs_type=pixels_agent_pos \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=50 \
  --eval.n_episodes=50 \
  --eval.use_async_envs=false \
  --policy.path=/home/Luogang/SRC/Robot/lerobot/outputs/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --policy.use_amp=false \
  --seed=1000 \
  --output_dir=./outputs_eval/
```

说明：

- **未写 `n_action_steps`** → 使用 checkpoint 的 **50**。  
- **`batch_size=50` 且 `n_episodes=50`**：与 `train_config.json` 里 eval 默认值一致，且满足 `n_episodes ≥ batch_size`，**无浪费 env**，GPU batch 也大。  
- 若 **50 OOM**，在保持 **`n_episodes ≥ batch_size`** 前提下把两者**同比减小**（例如 32/32），**不要**单独把 `n_action_steps` 改成 10。

---

**一句话**：  
**真实性优先** = **严格用 checkpoint 里的 `n_action_steps`/`chunk_size`、`use_amp`、观测与控制与 `HuggingFaceVLA/libero` + `input_features` 对齐，并提高 `n_episodes`；**加速** = 主要靠 **合法的 `eval.batch_size`、可选 async env、多进程多卡拆任务**，而不是改动作步数或混合精度。

## 最后采用的命令
由于:
- eval时不要设置n_action_steps,以便使其的chunk_size与chekpoint中的一致.而因为lerobot-train时没指定,所以应是默认值50.
- batchSz>1时模拟环境的同步会出问题;
- 不同task在不同GPU上跑;
**所以,最终采用如下命令:**
```bash
export HF_DEBUG=1
export TRANSFORMERS_VERBOSITY=debug
export HF_HUB_VERBOSITY=debug
export TORCH_DISTRIBUTED_DEBUG=DETAIL

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
  --policy.path=/home/Luogang/SRC/Robot/lerobot/outputs/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --policy.use_amp=false \
  --seed=1000 \
  --output_dir=./outputs_evalspatial/



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


---

# 解释 env.obs_type & env.control_mode


**User**

`env.obs_type=pixels_agent_pos`是什么意思?env.obs_type还有其它选项吗,分别有什么不同? `env.control_mode=relative`是什么意思?env.control_mode还有其它选项吗,分别有什么不同?

---

**Cursor**


下面只针对 **LeRobot 里的 `LiberoEnv`**（`--env.type=libero`）说明，依据 `configs.py` 和 `libero.py` 的实现。

---

## `env.obs_type=pixels_agent_pos` 是什么意思？

`obs_type` 决定 **仿真返回给策略的观测里包含哪些字段**（再经 `LiberoProcessorStep` 等转成 `observation.images.*` / `observation.state`）。

**`pixels_agent_pos`** 表示：

- **图像**：`agentview` + `eye_in_hand` 两路（默认映射成 `image` / `image2`）。
- **机器人状态**：末端位姿（位置、四元数、旋转矩阵）、夹爪、关节位置/速度等，打包在 `robot_state` 里；策略侧会对应到 `observation.state` 一类特征（你 checkpoint 里是 **8 维**，由预处理拼成固定维度）。

底层 `_format_raw_obs` 在 `pixels_agent_pos` 下会校验 **eef_pos、eef_quat、gripper_qpos** 必须存在：

```281:289:/home/Luogang/SRC/Robot/lerobot/src/lerobot/envs/libero.py
        if self.obs_type == "pixels_agent_pos":
            # Validate required fields are present
            if eef_pos is None or eef_quat is None or gripper_qpos is None:
                raise ValueError(
                    f"Missing required robot state fields in raw observation. "
```

所以：**既要视觉又要低维本体感觉**时用 `pixels_agent_pos`，与 **HuggingFaceVLA/libero** 那种「双图 + state」训练数据一致。

---

## `env.obs_type` 还有哪些选项？区别是什么？

在 **`LiberoEnv` 里实际可用**的只有两种（配置里写死分支，其它值会报错）：

| 取值 | 观测内容 | 典型用途 |
|------|-----------|----------|
| **`pixels_agent_pos`**（默认） | 双相机图像 + **完整 `robot_state`** | 与 **图像+状态** 训练的数据集对齐（如你的 GROOT + libero）。 |
| **`pixels`** | **仅**双相机图像，**不包含** `robot_state` | 纯视觉策略；若模型/训练仍要 `observation.state`，会对不上。 |

另外：

- 代码里曾出现 **`state` 纯状态** 分支，但会 **`NotImplementedError`**，不能用作 `obs_type`：

```170:174:/home/Luogang/SRC/Robot/lerobot/src/lerobot/envs/libero.py
        if self.obs_type == "state":
            raise NotImplementedError(
                "The 'state' observation type is not supported in LiberoEnv. "
                "Please switch to an image-based obs_type (e.g., 'pixels', 'pixels_agent_pos')."
```

`configs.py` 里也只接受 **`pixels`** 和 **`pixels_agent_pos`**，否则 `ValueError`。

---

## `env.control_mode=relative` 是什么意思？

`control_mode` 控制 **LIBERO 里机械臂控制器把 7 维动作解释成「增量」还是「绝对目标」**。

在 `reset()` 里设置 `robot.controller.use_delta`：

```310:317:/home/Luogang/SRC/Robot/lerobot/src/lerobot/envs/libero.py
        if self.control_mode == "absolute":
            for robot in self._env.robots:
                robot.controller.use_delta = False
        elif self.control_mode == "relative":
            for robot in self._robots:
                robot.controller.use_delta = True
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")
```

- **`relative`**：`use_delta=True` → 动作为 **相对当前位姿的增量（delta）**，与常见 LIBERO 演示/模仿学习设定一致。
- **`absolute`**：`use_delta=False` → 动作为 **绝对目标**（在同一套 7D 接口下语义不同）。

**必须与训练数据收集时一致**；换模式等于换动力学接口，成功率会明显不对。

---

## `env.control_mode` 还有哪些选项？

**只有两个合法值**（见上）：

| 取值 | 含义 |
|------|------|
| **`relative`** | 增量控制（默认）。 |
| **`absolute`** | 绝对控制。 |

其它字符串会直接 **`ValueError: Invalid control mode`**。

---

**小结**：  
- **`pixels_agent_pos`** = 图 + 机器人状态，适合你的 **8 维 state + 双图** 模型。  
- **`pixels`** = 只要图，不要状态。  
- **`relative` / `absolute`** = 动作是增量还是绝对目标，需与 **训练数据标注方式** 一致。

---

# 解释 n_episodes
**User**

n_episodes 是什么? 训练时需要设置吗? 其值的大小对算法精度和计算性能有什么影响?

---

**Cursor**


分场景说明 `n_episodes` 在 LeRobot 里的含义。




## `n_episodes` 是什么？

在 LeRobot 里，**`eval.n_episodes`**（`lerobot-eval` 或 `lerobot-train` 里带 `eval.` 前缀）表示：**做一次评估时，每个被评的 task 要跑多少条完整 episode（从 reset 到结束）**。

在 LIBERO 多 suite、多 `task_id` 时，`eval_policy_all` 会对 **每个 (suite, task_id)** 各跑 **`n_episodes` 条**，用来算成功率、回报等；**不是**「全 benchmark 一共只跑 N 条」。

---

## 训练时要不要设？

分两种：

### 1. 离线训练（你常用的 `lerobot-train` + 数据集）

- **不依赖 `eval.n_episodes` 来训练**：优化用的是 **DataLoader 按 step 采样**，数据量由 **数据集里有多少 episode / 多少帧** 和 **`--steps`** 决定。
- **`eval.n_episodes` 只在「训练过程中开仿真 eval」时才有意义**：即配置了 **`--env.type=libero`（等）** 且 **`eval_freq > 0`** 时，周期性在环境里 rollout，此时才会用 **`eval.n_episodes`** 决定每个 task 评几条。
- 若 **没有 `--env`**（你之前的 groot 训练就是 `env: null`），**训练过程里根本不会用到 `eval.n_episodes`**，设不设都不影响训练更新。

### 2. 单独跑 `lerobot-eval`

- **必须关心 `eval.n_episodes`**：它直接决定 **评估规模**和 **指标可信度**。

**结论**：纯离线训练可以**不设** `eval.n_episodes`（用默认即可）；只有 **train 里要开 env 做中途 eval** 或 **单独 eval** 时，才需要按需求设。

---

## 取值对「算法精度」的影响

这里要分清两层含义：

| 含义 | `n_episodes` 的影响 |
|------|---------------------|
| **训练得到的权重是否更准** | 离线训练时 **不**由 `eval.n_episodes` 决定；由 **数据、steps、lr** 等决定。中途 sim eval 的 `n_episodes` 只影响 **你看到的监控曲线噪声**，一般不直接当「精度」定义。 |
| **评估指标是否可信** | **越大**，成功率/均回报等 **方差越小**，越接近「真实」期望；**越小**，随机性大，**容易误判**好坏。 |

所以：**`n_episodes` 主要影响的是「评估的统计精度」**，不是改算法公式本身。

---

## 取值对「计算性能」的影响

- **几乎与 `n_episodes` 成正比**：每条 episode 要仿真很多步，episode 越多 **wall time 越长**、**CPU/GPU 占用时间越久**。
- 和 **`eval.batch_size` 的关系**：在实现里会按 batch 切多轮 rollout；在 **`n_episodes` 固定**时，适当增大 `batch_size` 可减少 **policy forward 的批次数**，但 **总仿真步数**仍随 `n_episodes` 增加。

---

## 小结

- **`n_episodes`**：每个 eval **task** 跑几条完整 rollout。  
- **训练**：离线训练 **不必**为「学得好」单独设它；只有 **带 `--env` 的中途 eval** 才会用到。  
- **更大**：评估 **更稳、更可信**，但 **更慢**。  
- **更小**：**更快**，但 **指标更抖**，不利于公平对比模型。

