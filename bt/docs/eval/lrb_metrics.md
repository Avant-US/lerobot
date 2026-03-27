# # Lerobot Eval 有些什么Metrics

**User**

请参考该lerobot项目的代码和docs文档,同时参考官方网站 https://huggingface.co/docs/lerobot/index , 官方论文: https://arxiv.org/abs/2602.22818 , 以及网上其它相关的文章, 说一下 lerobot-eval做评估时都提供些什么metrics指标? 分别是什么含义?

---

**Cursor**


正在从代码库、文档及官方资料中查找 `lerobot-eval` 的评估指标。




下面根据本仓库里 `lerobot-eval` 的实现（`src/lerobot/scripts/lerobot_eval.py`）说明：**官方文档首页**和 **arXiv:2602.22818** 主要介绍 LeRobot 整体定位，**并未逐项定义 eval 指标**；指标含义以代码为准。

---

## 1. 指标从哪里来

评估时会对每个环境调用 `env.step()`，得到逐步的 **`reward`**；成功信号来自 Gymnasium **向量化环境**在 episode 结束时的 **`info["final_info"]["is_success"]`**（若本步没有环境结束，则该步全为 `False`）：

```193:204:src/lerobot/scripts/lerobot_eval.py
        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available if none of the envs finished.
        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs
```

因此：**reward 的具体数值、success 的判定规则都取决于具体环境**（例如 LIBERO、Pusht 等各自实现不同）。

---

## 2. 每条 episode 上算什么（`per_episode`）

对每条轨迹，在**首次 `done` 及之前**的时间步上做掩码，然后统计：

| 字段 | 含义 |
|------|------|
| **`sum_reward`** | 该 episode 内（到结束为止）每步 **reward 的和**（累积回报）。 |
| **`max_reward`** | 该 episode 内（到结束为止）**单步 reward 的最大值**。在稀疏奖励里有时能反映“是否出现过成功那一步的高奖励”；在稠密奖励里信息量可能接近过程质量。 |
| **`success`** | 该 episode 内是否**任意一步**出现过 `is_success == True`（逻辑上是 `any`）。 |
| **`seed`** | 若启动时提供了 `start_seed`，则为该条 episode 使用的随机种子；否则可能为 `None`。 |
| **`episode_ix`** | 从 0 起的 episode 序号。 |

聚合计算见：

```351:366:src/lerobot/scripts/lerobot_eval.py
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
```

---

## 3. 单次 `eval_policy` 的汇总（`aggregated`）

| 指标 | 含义 |
|------|------|
| **`avg_sum_reward`** | 所有被统计的 episode 上 **`sum_reward` 的算术平均**。 |
| **`avg_max_reward`** | 所有 episode 上 **`max_reward` 的算术平均**。 |
| **`pc_success`** | 成功率（**百分比**）：对 `success` 取平均再 **×100**，即“成功 episode 占比 × 100”。 |
| **`eval_s`** | 本次 `eval_policy` 的**墙上时钟总耗时**（秒）。 |
| **`eval_ep_s`** | **平均每集耗时**：`eval_s / n_episodes`（这里的 `n_episodes` 是本次评估请求的 episode 数）。 |

```442:448:src/lerobot/scripts/lerobot_eval.py
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
```

---

## 4. 多任务 / 多 suite 时（`eval_policy_all`，`lerobot-eval` 默认路径）

`lerobot-eval` 调用 `eval_policy_all`，结果会写入 **`eval_info.json`**，结构包括：

- **`overall`**：把所有 task 的 episode **摊平后**再汇总。
- **`per_group`**：按 `task_group`（例如某个 benchmark 名）分别汇总。
- **`per_task`**：每个 `(task_group, task_id)` 的原始列表指标（`sum_rewards`、`max_rewards`、`successes` 等）。

在 **`overall` / `per_group`** 里，除上述 `avg_sum_reward`、`avg_max_reward`、`pc_success` 外，还有：

| 字段 | 含义 |
|------|------|
| **`n_episodes`** | 参与汇总的 **episode 条数**（多任务时是各 task 的 episode 数之和）。 |
| **`video_paths`** | 若开启了渲染保存，收集到的 **评测视频路径列表**（CLI 里对前若干条 episode 写视频）。 |
| **`eval_s`** | 整次 `eval_policy_all` 的**总耗时**（秒）。 |
| **`eval_ep_s`** | **总耗时 / 参与汇总的 episode 总数**（注意：是多任务下“全局平均每集”，不是单 task 的 wall time）。 |

```777:797:src/lerobot/scripts/lerobot_eval.py
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
    }
```

---

## 5. 控制台里的 “running_success_rate”

`rollout` / `eval_policy` 的进度条里会显示 **当前已跑完的 episode 的即时成功率**，这是**交互显示**，不是 `eval_info.json` 里单独命名的字段；最终仍以 `pc_success` 为准。

---

## 6. 与论文 / 官网的关系

- **[Hugging Face LeRobot 文档首页](https://huggingface.co/docs/lerobot/index)**：概述库的定位与能力，**没有**展开 `lerobot-eval` 的指标表。  
- **[arXiv:2602.22818](https://arxiv.org/abs/2602.22818)**：介绍 LeRobot 作为端到端开源库的论文，摘要侧重数据、训练与部署栈，**不替代**具体 eval 脚本里的指标定义。  

**结论**：`lerobot-eval` 在实现层面提供的**核心数值指标**就是 **回报类**（`sum_reward` / `avg_sum_reward`）、**单步奖励峰值类**（`max_reward` / `avg_max_reward`）、**任务成功类**（`success` / `pc_success`），以及 **计时与视频路径**；其中 reward 与 success 的**语义完全由所接的 Gymnasium 环境**决定。若你需要对 LIBERO 等环境做“和论文表格一致”的对比，还需要对照该环境包装器里 **reward  shaping** 和 **`is_success` 何时为 True** 的说明。

---

# Lerobot Eval 有些什么Metrics (few-shot版)

**User**

请参考该lerobot项目的代码和docs文档,同时参考官方网站 https://huggingface.co/docs/lerobot/index , 官方论文: https://arxiv.org/abs/2602.22818 , 以及网上其它相关的文章, 说一下 lerobot-eval做评估时都提供些什么metrics指标? 分别是什么含义?

---

**Cursor**


正在搜索 LeRobot 评测代码与文档中定义的指标。




下面依据本仓库里 **`lerobot-eval` 的实现**（[`src/lerobot/scripts/lerobot_eval.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_eval.py)）说明指标；官方文档 [LeRobot Docs](https://huggingface.co/docs/lerobot/index) 与论文 [arXiv:2602.22818](https://arxiv.org/abs/2602.22818) 侧重库的定位与整体能力，**具体 eval 数值字段以代码为准**。

---

## 1. 评测输出里有哪些层次

- **单环境 / 单任务**：`eval_policy()` 返回 `per_episode` + `aggregated`（可选 `video_paths`、`episodes`）。
- **多任务（如 LIBERO 多 `task_id`）**：`eval_policy_all()` 额外汇总为：
  - **`per_task`**：每个 `(task_group, task_id)` 一坨原始列表；
  - **`per_group`**：按 `task_group`（suite）聚合；
  - **`overall`**：所有 episode 跨任务总体聚合。

结果会写入 **`eval_info.json`**（与 `eval_main` 一致）。

---

## 2. 每个 episode 上算什么（per_episode / TaskMetrics）

每条 rollout 在**首次 `done` 之前（含该步）**的有效步上统计（见 `rollout` + mask 逻辑）：

| 字段 | 含义 |
|------|------|
| **`sum_reward`** | 该 episode **累计奖励**：每步 `env` 返回的 `reward` 之和（截断在 episode 结束处）。 |
| **`max_reward`** | 该 episode 内**单步奖励的最大值**。对稀疏奖励（很多步为 0、成功时 1）常与“是否出现过成功信号”一致。 |
| **`success`** | 该 episode 是否**成功**：来自 Gymnasium 的 `info["final_info"]["is_success"]`（环境结束时才可能出现 `True`）；实现里对有效步做 `any`，即整局只要出现过成功标记即为 `True`。 |
| **`seed`** | 该 episode 使用的随机种子（若配置了 `start_seed`）；否则可能为 `None`。 |

多任务时，这些会落成列表：**`sum_rewards`、`max_rewards`、`successes`**，以及可选的 **`video_paths`**。

---

## 3. 聚合指标（aggregated / overall / per_group）

| 指标 | 含义 |
|------|------|
| **`avg_sum_reward`** | 所有（参与汇总的）episode 的 **`sum_reward` 的算术平均**。 |
| **`avg_max_reward`** | 所有 episode 的 **`max_reward` 的算术平均**。 |
| **`pc_success`** | **成功率（百分比）**：`success` 为布尔值时先取平均再 **`× 100`**，因此 **4.0 表示 4%**，**不是 0–1 的概率标量**（你之前日志里 `7.000000000000001` 是浮点误差）。 |
| **`n_episodes`** | 参与聚合的 episode 数量（`eval_policy_all` 的 per_group / overall 里出现）。 |
| **`eval_s`** | **整次评测墙钟时间**（秒），从评测开始计时到聚合完成。 |
| **`eval_ep_s`** | **平均每 episode 耗时**：`eval_s / n_episodes`（overall 里 `n_episodes` 为总 episode 数）。 |
| **`video_paths`** | 保存的评测录像路径列表（当开启渲染保存时）。 |

单环境 `eval_policy` 的 **`aggregated`** 与多任务 **`overall`** 在核心标量上对齐（success 比例、平均奖励、时间）；多任务时 **`per_group`** 是对同一 suite 内所有 episode 再算一遍上述聚合。

---

## 4. 和官方文档/论文的关系

- [Hugging Face LeRobot 文档](https://huggingface.co/docs/lerobot/index) 介绍的是库的目标与组件，**不替代**上述字段级定义。  
- [LeRobot 论文](https://arxiv.org/abs/2602.22818) 描述的是开源库的整体设计与机器人学习栈，**eval 的具体 metric 名称与公式以 `lerobot_eval.py` 为准**。

---

## 5. 简要结论

**`lerobot-eval` 提供的核心 metrics 是：**  
**`avg_sum_reward`、`avg_max_reward`、`pc_success`（成功率百分比）、`eval_s`、`eval_ep_s`**，以及 **`n_episodes`、`video_paths`**；底层 per-episode 还有 **`sum_reward`、`max_reward`、`success`、`seed`**。  
奖励与是否成功**完全由具体 Gym 环境**的 `reward` 和 `final_info["is_success"]` 决定，因此不同环境（PushT、LIBERO 等）数值尺度可能不同，但**统计方式相同**。

---

# 对 Eval 日志的分析例子

## libero_spatial eval log

**User**

@\home\Luogang\.cursor\projects\home-Luogang-SRC-Robot-lerobot\terminals\1.txt:457-696 这是lerobot对libero_spatial任务做eval的日志，解析它的含义，说说它eval指标的分数是多少。

---

**Cursor**

根据你贴的这段 **LIBERO Spatial** 评测日志，含义和分数可以概括如下。

### 日志在说什么

这是一次在 **`libero_spatial` 任务组** 上的离线评测：

- **10 个子任务**（`task_id` 0–9），每个子任务跑 **10 个 episode**，一共 **100 个 episode**（`n_episodes: 100`）。
- 每个 episode 会记录：**累计奖励** `sum_rewards`、**单步最大奖励** `max_rewards`、是否**成功** `successes`，以及对应 **录像路径** `video_paths`。
- 末尾还有 **整次评测耗时**：总时间约 **12986 s**，平均每 episode 约 **130 s**（`eval_s` / `eval_ep_s`）。

从 `per_task` 里可以看出：成功时一般是 `sum_rewards` / `max_rewards` 里出现 **1.0**，失败多为 **0.0**（和你贴的列表一致）。

### 主要评测指标（分数）

**整体（Overall / per_group / overall 三块里核心数字一致）：**

| 指标 | 数值 | 含义 |
|------|------|------|
| **`pc_success`** | **4.0** | 成功率 **4%**（100 个 episode 里 **4 次** `successes=True`） |
| **`avg_sum_reward`** | **0.04** | 平均每个 episode 的累计奖励 ≈ 4/100（与 4 次成功、每次成功给 1 一致） |
| **`avg_max_reward`** | **0.04** | 平均每个 episode 的“最大单步奖励”统计结果也是 **0.04**（本日志里与 sum 一致） |

**按任务拆开（你贴的数据）：**

- **task_id 0**：10 里成功 **1** 次（第 9 个 episode）  
- **task_id 1**：成功 **1** 次（第 1 个 episode）  
- **task_id 5**：成功 **1** 次（第 4 个 episode）  
- **task_id 8**：成功 **1** 次（第 2 个 episode）  
- **task_id 2,3,4,6,7,9**：**0** 次成功  

合计 **4/100 → 4%**，与 `pc_success: 4.0` 一致。

**一句话**：这次 eval 的“主分数”可以记为 **成功率 4%**（`pc_success`），配套的平均奖励约为 **0.04**（`avg_sum_reward` / `avg_max_reward`）。

---

## libero_object eval log

**User**

@\home\Luogang\.cursor\projects\home-Luogang-SRC-Robot-lerobot\terminals\3.txt:654-890 这是lerobot对libero_object任务做eval的日志，解析它的含义，说说它eval指标的分数是多少。

---

**Cursor**

这段日志和之前的 **libero_spatial** 是同一套评测格式，只是任务组换成了 **`libero_object`**（LIBERO Object 基准）。

### 日志含义

- **任务组**：`libero_object`  
- **设置**：10 个子任务（`task_id` 0–9），每个任务 **10 个 episode**，共 **100 个 episode**（`n_episodes: 100`）。  
- **每条 episode**：记录 `sum_rewards`、`max_rewards`、`successes`，以及录像路径（`outputs_evalobj/videos/...`）。  
- **耗时**：整次评测约 **12611 s**（`eval_s`），平均每个 episode 约 **126 s**（`eval_ep_s`）。

成功时通常 `sum_rewards` / `max_rewards` 为 **1.0**，失败多为 **0.0**。

### 评测指标（分数）

| 指标 | 数值 | 说明 |
|------|------|------|
| **`pc_success`** | **7%**（日志里写成 `7.000000000000001` 是浮点误差） | 100 次里 **7 次**成功 |
| **`avg_sum_reward`** | **0.07** | 平均每 episode 累计奖励 = 7/100 |
| **`avg_max_reward`** | **0.07** | 平均每 episode 的最大单步奖励统计，与上式一致 |

### 按任务的成功次数（与 7 次总成功一致）

- **task 0**：1 次（episode 8）  
- **task 2**：1 次（episode 3）  
- **task 4**：3 次（episode 7、8、9）  
- **task 7**：1 次（episode 9）  
- **task 8**：1 次（episode 0）  
- **task 1、3、5、6、9**：0 次  

**一句话**：这次 **libero_object** eval 的主结果是 **成功率 7%**（`pc_success`），平均奖励 **0.07**（`avg_sum_reward` / `avg_max_reward`）。

---

## libero_goal eval log

**User**

@\home\Luogang\.cursor\projects\home-Luogang-SRC-Robot-lerobot\terminals\4.txt:273-504 这是lerobot对libero_goal任务做eval的日志，解析它的含义，说说它eval指标的分数是多少。

---

**Cursor**

这段日志对应 **LIBERO Goal** 任务组上的评测，格式与 `libero_spatial` / `libero_object` 相同。

### 日志含义

- **任务组**：`libero_goal`  
- **规模**：10 个子任务（`task_id` 0–9），每个 **10 个 episode**，共 **100** 个 episode（`n_episodes: 100`）。  
- **每条 episode**：`sum_rewards`、`max_rewards`、`successes`，以及录像（`outputs_evalgoal/videos/...`）。  
- **耗时**：总评测约 **12942 s**（`eval_s`），平均每 episode 约 **129 s**（`eval_ep_s`）。

### 评测指标（分数）

这里的 **`pc_success` 表示成功率（百分比数值）**，不是 0–1 的归一化“满分 1”。

| 指标 | 数值 | 含义 |
|------|------|------|
| **`pc_success`** | **1.0** | **1%** 成功率：100 次里 **1 次**成功 |
| **`avg_sum_reward`** | **0.01** | 平均每 episode 累计奖励 = 1/100 |
| **`avg_max_reward`** | **0.01** | 平均每 episode 的最大单步奖励统计，与上式一致 |

### 成功来自哪条任务

`per_task` 里只有 **`task_id: 2`** 在 **第 5 个 episode**（列表下标 4）成功：`sum_rewards` / `max_rewards` 为 1.0，`successes` 为 `True`。其余 9 个子任务共 90 个 episode 全部失败。

**一句话**：这次 **libero_goal** eval 的主结果是 **成功率 1%**（`pc_success: 1.0`），平均奖励 **0.01**（`avg_sum_reward` / `avg_max_reward`）。

---

## libero_10 eval log

**User**

@\home\Luogang\.cursor\projects\home-Luogang-SRC-Robot-lerobot\terminals\5.txt:735-953 这是lerobot对libero_10任务做eval的日志，解析它的含义，说说它eval指标的分数是多少。

---

**Cursor**

这段日志对应 **LIBERO 的 `libero_10` 任务组**（长程/10 任务子集，命名与路径里的 `libero_10_*` 一致），评测格式与前面几个 suite 相同。

### 日志含义

- **任务组**：`libero_10`  
- **规模**：10 个子任务（`task_id` 0–9），每个 **10 个 episode**，共 **100** 个 episode（`n_episodes: 100`）。  
- **每条 episode**：`sum_rewards`、`max_rewards`、`successes`，录像在 `outputs_eval10/videos/libero_10_*/eval_episode_*.mp4`。  
- **耗时**：总评测约 **13490 s**（`eval_s`），平均每 episode 约 **135 s**（`eval_ep_s`）。

### 评测指标（分数）

| 指标 | 数值 | 含义 |
|------|------|------|
| **`pc_success`** | **0.0** | **0%** 成功率：100 次里 **0 次**成功 |
| **`avg_sum_reward`** | **0.0** | 平均每 episode 累计奖励为 0 |
| **`avg_max_reward`** | **0.0** | 平均每 episode 的最大单步奖励统计也为 0 |

`per_task` 里 **10 个子任务** 的 `successes` 全部为 `False`，`sum_rewards` / `max_rewards` 全为 **0.0**，与上述汇总一致。

**一句话**：这次 **libero_10** eval 的结果是 **成功率 0%**，平均奖励 **0**（`pc_success`、`avg_sum_reward`、`avg_max_reward` 均为 0）。

