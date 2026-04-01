# bt/robotwin2 整合设计方案

## Context

当前分支 `bt0.5at260311` 已对 LeRobot 原生代码做了若干修改（新增 Groot2/StrGroot 策略、SmolVLA 调整、WandB 配置等）。本设计方案**基于 bt0.5at260311 分支现有状态**继续推进，在 `bt/robotwin2/` 中实现与 RobotTwin2 的完整整合。

目标：
1. **在 `bt/robotwin2/` 中实现与 RobotTwin2 的完整整合**，尽量不再额外修改 LeRobot 原生代码
2. 整合范围覆盖：数据（合成/采集/转换）、训练、评测
3. 保留分支已有的 Groot2/StrGroot 等策略修改

核心原则：`bt/robotwin2` 作为 companion layer，复用 LeRobot 公共 API 和扩展点，不把 RobotTwin2 逻辑塞进 `src/lerobot/`。

### 当前分支已有的 LeRobot 修改（保留不动）

| 文件 | 修改内容 | 状态 |
|------|----------|------|
| `src/lerobot/policies/factory.py` | 新增 groot2/str_groot 策略分支 | 保留 |
| `src/lerobot/policies/groot2/` | Groot2 策略实现 | 保留 |
| `src/lerobot/policies/str_groot/` | StrGroot (StarVLA/Qwen) 策略 | 保留 |
| `src/lerobot/policies/smolvla/modeling_smolvla.py` | SmolVLA 调整 | 保留 |
| `src/lerobot/configs/default.py` | WandB 配置调整 | 保留 |
| `src/lerobot/rl/wandb_utils.py` | WandB tag 逻辑调整 | 保留 |

---

## Phase 1: 总体架构设计

### 1.1 组件图

```
┌─────────────────────────────────────────────────────────────────┐
│                        bt/robotwin2                             │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   common/    │  │    data/     │  │      record/         │  │
│  │              │  │              │  │                      │  │
│  │ canonical_   │──│ generate.py  │  │ run_record.py        │  │
│  │ adapter.py   │  │ convert.py   │  │ action_sources.py    │  │
│  │ schema.py    │  │ split_merge  │  │ episode_controller   │  │
│  │ profiles.py  │  │ validate.py  │  │                      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│         │    ┌────────────┴──────────────────────┘              │
│         │    │                                                  │
│  ┌──────┴────┴──┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │    train/    │  │    eval/     │  │     configs/         │  │
│  │              │  │              │  │                      │  │
│  │ stage_runner │  │ env_config   │  │ data/*.yaml          │  │
│  │ curriculum   │  │ gym_env.py   │  │ record/*.yaml        │  │
│  │ offline_val  │  │ gym_reg.py   │  │ train/*.yaml         │  │
│  │ ckpt_eval    │  │ benchmark    │  │ eval/*.yaml          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────┬───────────────────────┬───────────────────┘
                      │  复用公共 API          │  插件注册
                      ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LeRobot (bt0.5at260311 分支)                    │
│                                                                 │
│  LeRobotDataset    PreTrainedPolicy    EnvConfig                │
│  .create()         .from_pretrained()  .register_subclass()     │
│  .add_frame()      .forward()          make_env()               │
│  .save_episode()   .select_action()    plugin discovery         │
│  .finalize()                                                    │
│                                                                 │
│  已有策略: ACT, Diffusion, SmolVLA, Groot2, StrGroot, ...       │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RobotTwin2                                  │
│                                                                 │
│  SAPIEN runtime    expert planner    task configs                │
│  CuRobo IK        collect_data.sh   domain randomization        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 总体工作流

```
RobotTwin2 原始数据 ──► convert ──► LeRobotDataset v3
                                         │
仿真录制 ──► run_record ──► LeRobotDataset v3
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                         split/merge           validate
                              │
                              ▼
                    stage_runner (多阶段训练)
                    调用 lerobot-train
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              offline_val         lerobot-eval
              (held-out)          (RobotTwin2 gym env)
                                        │
                                        ▼
                                   benchmark
```

### 1.3 关键设计：零改动 LeRobot 的可行性分析

| 功能 | LeRobot 扩展点 | bt/robotwin2 实现 | 是否需要改 LeRobot |
|------|---------------|-------------------|-------------------|
| 数据转换 | `LeRobotDataset.create()` API | `data/convert.py` 调用 | **否** |
| 仿真录制 | `LeRobotDataset.add_frame()` API | `record/run_record.py` | **否** |
| 训练 | `lerobot-train` CLI | `train/stage_runner.py` 子进程调用 | **否** |
| 评测环境 | `@EnvConfig.register_subclass()` + plugin discovery | `eval/env_config.py` 注册 | **否** |
| Gym 环境 | `gymnasium.register()` + `importlib.import_module()` | `eval/gym_registration.py` | **否** |
| Benchmark | `lerobot-eval` CLI | `eval/benchmark.py` 批量调用 | **否** |

**结论：所有 RobotTwin2 整合功能均可在不额外修改 LeRobot 原生代码的情况下实现。**

### 1.4 LeRobot 扩展机制详解

#### 1.4.1 插件发现机制 (`src/lerobot/configs/parser.py`)

LeRobot 提供 `discover_packages_path` CLI 参数，在配置解析前自动导入外部包：

```python
# LeRobot 内部实现（不需要修改）
PLUGIN_DISCOVERY_SUFFIX = "discover_packages_path"

def load_plugin(plugin_path: str) -> None:
    """导入 plugin_path 及其所有子模块，触发 @register_subclass() 注册。"""
    importlib.import_module(plugin_path)
```

**使用方式**：CLI 传入 `--env.discover_packages_path=bt.robotwin2.eval`，LeRobot 自动导入该包，触发 `@EnvConfig.register_subclass("robotwin2_eval")` 装饰器注册。

#### 1.4.2 EnvConfig 注册模式 (`src/lerobot/envs/configs.py`)

```python
# LeRobot 基类（不需要修改）
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature]      # 特征定义
    features_map: dict[str, str]            # 原始 obs key → canonical key
    gym_kwargs: dict                         # 传给 gym.make() 的参数
```

外部包只需用 `@EnvConfig.register_subclass("name")` 装饰器即可注册新环境类型。

#### 1.4.3 Gym 环境动态导入 (`src/lerobot/envs/factory.py`)

```python
# LeRobot 内部实现（不需要修改）
def make_env(cfg: EnvConfig, ...):
    if cfg.gym_id not in gym_registry:
        importlib.import_module(cfg.package_name)  # 自动导入外部 gym 包
```

只要 `RobotTwinEvalEnvConfig.package_name` 返回正确的模块路径，LeRobot 会自动导入并完成 gym 环境注册。

---

## Phase 2: 核心模块详细设计

### 2.1 `common/canonical_adapter.py` — 统一映射层

**这是整个设计最关键的文件。** 所有 data/record/eval 模块都必须复用它，禁止在各模块中各自实现映射逻辑。

```python
"""统一 RobotTwin2 与 LeRobot 的 observation/action/task 映射。

所有 data/record/eval 模块都必须通过此 adapter 做映射，
禁止在各模块中各自实现映射逻辑。
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from lerobot.configs.types import FeatureType, PolicyFeature


# ── RobotTwin2 原始 key → LeRobot canonical key ──────────────
CAMERA_KEY_MAP: dict[str, str] = {
    "cam_high":        "observation.images.head",
    "cam_left_wrist":  "observation.images.left_wrist",
    "cam_right_wrist": "observation.images.right_wrist",
}

STATE_KEY = "observation.state"
ACTION_KEY = "action"
TASK_KEY = "task"


@dataclass
class RobotTwinDatasetProfile:
    """描述一种 RobotTwin2 → LeRobot 数据映射 profile。

    不同的任务/机器人可能有不同的维度和相机配置，
    通过 profile 参数化来支持多种场景。
    """
    action_mode: str = "qpos"                           # qpos | ee | hybrid
    camera_keys: tuple[str, ...] = ("head", "left_wrist", "right_wrist")
    state_dim: int = 14                                 # 双臂 7+7
    action_dim: int = 14
    image_shape: tuple[int, int, int] = (480, 640, 3)   # H, W, C
    include_endpose: bool = False
    instruction_strategy: str = "first_non_empty"        # first_non_empty | per_step | fixed


class RobotTwinCanonicalAdapter:
    """RobotTwin2 ↔ LeRobot 的统一转换器。

    职责：
    1. 构造 LeRobotDataset features（供 create() 使用）
    2. 将 RobotTwin2 原始观测转为 dataset frame（供 add_frame() 使用）
    3. 将 RobotTwin2 原始观测转为 policy 输入（供 eval gym env 使用）
    4. 将 policy 输出转为 env action（供 eval step 使用）
    """

    def __init__(self, profile: RobotTwinDatasetProfile | None = None):
        self.profile = profile or RobotTwinDatasetProfile()

    # ── Dataset Features ─────────────────────────────────────

    def build_dataset_features(self) -> dict[str, PolicyFeature]:
        """构造 LeRobotDataset.create() 所需的 features dict。"""
        features = {}

        # 状态
        features[STATE_KEY] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(self.profile.state_dim,),
        )

        # 动作
        features[ACTION_KEY] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(self.profile.action_dim,),
        )

        # 相机
        for cam_name in self.profile.camera_keys:
            key = f"observation.images.{cam_name}"
            features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=self.profile.image_shape,
            )

        return features

    # ── Raw Obs → Dataset Frame ──────────────────────────────

    def raw_obs_to_dataset_frame(
        self, raw_obs: dict, raw_action: np.ndarray, info: dict, task: str
    ) -> dict[str, Any]:
        """将 RobotTwin2 原始观测转为 LeRobotDataset 的一帧。

        用于 data/convert.py 和 record/run_record.py。
        """
        frame: dict[str, Any] = {}

        # 状态
        frame[STATE_KEY] = raw_obs["qpos"].astype(np.float32)

        # 动作
        frame[ACTION_KEY] = raw_action.astype(np.float32)

        # 相机图像
        for rt_key, lr_key in CAMERA_KEY_MAP.items():
            cam_short = lr_key.split(".")[-1]
            if cam_short in self.profile.camera_keys and rt_key in raw_obs:
                frame[lr_key] = raw_obs[rt_key]  # uint8 H×W×3

        # 任务文本
        frame[TASK_KEY] = task

        return frame

    # ── Raw Obs → Policy Input (eval 时用) ───────────────────

    def raw_obs_to_policy_obs(self, raw_obs: dict, info: dict) -> dict[str, Any]:
        """将 RobotTwin2 观测转为 policy 可接受的 dict（eval 时使用）。

        输出格式需与 EnvConfig.features_map 对应:
          pixels.head → observation.images.head
          agent_pos   → observation.state
        """
        obs: dict[str, Any] = {}
        obs["pixels"] = {}
        for rt_key, lr_key in CAMERA_KEY_MAP.items():
            cam_short = lr_key.split(".")[-1]
            if cam_short in self.profile.camera_keys and rt_key in raw_obs:
                obs["pixels"][cam_short] = raw_obs[rt_key]
        obs["agent_pos"] = raw_obs["qpos"]
        return obs

    # ── Policy Action → Env Action ──────────────────────────

    def policy_action_to_env_action(self, action: np.ndarray) -> np.ndarray:
        """policy 输出 → RobotTwin2 env.step() 所需格式。

        qpos 模式：直接透传
        ee 模式：需要做坐标系/单位转换（预留扩展点）
        """
        return action

    # ── Episode Metadata ────────────────────────────────────

    def extract_episode_meta(self, info: dict) -> dict[str, Any]:
        """提取 episode 级元数据（lineage 追踪用）。"""
        return {
            "task": info.get("task", ""),
            "success": info.get("success", False),
            "seed": info.get("seed"),
            "embodiment": info.get("embodiment", "aloha-agilex"),
            "scene_info": info.get("scene_info", {}),
        }
```

### 2.2 `common/schema.py` — 数据契约

```python
"""定义 RobotTwin2 数据集的结构契约和常量。

所有模块引用常量都应从此处导入，确保全局一致。
"""

# 默认 FPS
DEFAULT_FPS = 15

# 默认 robot_type 标识
ROBOT_TYPE = "robotwin2"

# Env 注册 ID（格式: namespace/EnvName-vN）
GYM_ENV_ID = "bt_robotwin2/RobotTwinEval-v0"

# 数据集 observation key 与 gym env observation key 的映射
# gym env 输出 pixels/agent_pos 结构，
# LeRobot 通过 features_map 将其映射为 canonical key
ENV_FEATURES_MAP: dict[str, str] = {
    "pixels.head":        "observation.images.head",
    "pixels.left_wrist":  "observation.images.left_wrist",
    "pixels.right_wrist": "observation.images.right_wrist",
    "agent_pos":          "observation.state",
}
```

### 2.3 `data/convert.py` — 离线数据转换

将 RobotTwin2 的原始数据（HDF5 / npy / mp4）转为 LeRobotDataset v3 格式。

```python
"""将 RobotTwin2 原始数据转为 LeRobotDataset v3。

用法:
  PYTHONPATH=. python -m bt.robotwin2.data.convert \
    --input_path /data/robotwin2/raw/beat_block_hammer \
    --output_root bt/robotwin2/output/datasets \
    --repo_id local/robotwin2_beat_block
"""

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from bt.robotwin2.common.canonical_adapter import (
    RobotTwinCanonicalAdapter,
    RobotTwinDatasetProfile,
)
from bt.robotwin2.common.schema import DEFAULT_FPS, ROBOT_TYPE


def iterate_robotwin_episodes(input_path: Path):
    """遍历 RobotTwin2 原始数据目录，yield 每个 episode 的路径。

    RobotTwin2 典型目录结构:
      input_path/task_name/episode_0001/
        cam_high.mp4 / cam_left_wrist.mp4 / cam_right_wrist.mp4
        qpos.npy / action.npy / metadata.json
    """
    for ep_dir in sorted(input_path.glob("*/episode_*")):
        yield ep_dir


def iterate_steps(episode_path: Path):
    """遍历单个 episode 的每一步，yield (raw_obs, raw_action, info)。"""
    import json
    import numpy as np

    qpos = np.load(episode_path / "qpos.npy")        # (T, state_dim)
    actions = np.load(episode_path / "action.npy")    # (T, action_dim)

    meta_path = episode_path / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    for t in range(len(qpos)):
        raw_obs = {"qpos": qpos[t]}

        # 加载当前帧图像（实际实现需按 RobotTwin2 格式适配）
        for cam_name in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
            img_path = episode_path / f"{cam_name}" / f"{t:06d}.png"
            if img_path.exists():
                import cv2
                raw_obs[cam_name] = cv2.imread(str(img_path))[..., ::-1]  # BGR→RGB

        info = {
            "task": meta.get("task_name", episode_path.parent.name),
            "success": meta.get("success", True),
            "seed": meta.get("seed"),
        }
        yield raw_obs, actions[t], info


def convert_robotwin_dataset(
    input_path: str | Path,
    output_root: str | Path,
    repo_id: str,
    profile: RobotTwinDatasetProfile | None = None,
    fps: int = DEFAULT_FPS,
    use_videos: bool = True,
) -> LeRobotDataset:
    """主转换入口。

    流程:
    1. CanonicalAdapter.build_dataset_features() → features dict
    2. LeRobotDataset.create(features) → 空数据集
    3. 遍历 episode → 逐帧 raw_obs_to_dataset_frame() → add_frame()
    4. save_episode() → finalize()
    """
    input_path = Path(input_path)
    output_root = Path(output_root)
    adapter = RobotTwinCanonicalAdapter(profile)

    features = adapter.build_dataset_features()

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_root / repo_id,
        robot_type=ROBOT_TYPE,
        features=features,
        use_videos=use_videos,
    )

    for ep_path in iterate_robotwin_episodes(input_path):
        for raw_obs, raw_action, info in iterate_steps(ep_path):
            task = info["task"]
            frame = adapter.raw_obs_to_dataset_frame(raw_obs, raw_action, info, task)
            dataset.add_frame(frame)
        dataset.save_episode(task=info["task"])

    dataset.finalize()
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--no_videos", action="store_true")
    args = parser.parse_args()
    convert_robotwin_dataset(
        args.input_path, args.output_root, args.repo_id,
        fps=args.fps, use_videos=not args.no_videos,
    )
```

### 2.4 `record/run_record.py` — 仿真录制

在 RobotTwin2 仿真中实时录制数据，直接生成 LeRobotDataset。

```python
"""在 RobotTwin2 仿真中直接录制 LeRobotDataset。

用法:
  PYTHONPATH=. python -m bt.robotwin2.record.run_record \
    --task_name=beat_block_hammer \
    --num_episodes=50 \
    --action_source_type=expert
"""

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from bt.robotwin2.common.canonical_adapter import (
    RobotTwinCanonicalAdapter,
    RobotTwinDatasetProfile,
)
from bt.robotwin2.common.schema import DEFAULT_FPS, ROBOT_TYPE


@dataclass
class RecordConfig:
    task_name: str = "beat_block_hammer"
    task_config: str = "demo_clean"
    embodiment: str = "aloha-agilex"
    num_episodes: int = 50
    max_steps_per_episode: int = 300
    action_source_type: str = "expert"   # expert | policy | teleop
    policy_path: str | None = None
    repo_id: str = "local/robotwin2_record"
    output_root: str = "bt/robotwin2/output/datasets"
    fps: int = DEFAULT_FPS
    use_videos: bool = True
    profile: RobotTwinDatasetProfile = field(default_factory=RobotTwinDatasetProfile)


def make_robotwin_runtime(cfg: RecordConfig):
    """创建 RobotTwin2 仿真 runtime。

    需要对接 RobotTwin2 的 SAPIEN runtime API。
    返回对象需支持 reset() -> (obs, info) 和 step(action) -> (obs, r, term, trunc, info)。
    """
    raise NotImplementedError("需要对接 RobotTwin2 runtime")


def make_action_source(cfg: RecordConfig):
    """创建动作源：expert / policy / teleop。"""
    if cfg.action_source_type == "expert":
        from bt.robotwin2.record.action_sources import ExpertActionSource
        return ExpertActionSource(cfg.task_name, cfg.task_config)
    elif cfg.action_source_type == "policy":
        from bt.robotwin2.record.action_sources import PolicyActionSource
        return PolicyActionSource(cfg.policy_path)
    elif cfg.action_source_type == "teleop":
        from bt.robotwin2.record.action_sources import TeleopActionSource
        return TeleopActionSource()
    raise ValueError(f"Unknown action_source_type: {cfg.action_source_type}")


def run_record(cfg: RecordConfig):
    """录制主循环。

    流程:
    1. 创建 adapter + runtime + action_source
    2. LeRobotDataset.create()
    3. 循环 episode:
       a. runtime.reset()
       b. 循环 step: action_source → runtime.step → adapter → dataset.add_frame()
       c. dataset.save_episode()
    4. dataset.finalize()
    """
    adapter = RobotTwinCanonicalAdapter(cfg.profile)
    runtime = make_robotwin_runtime(cfg)
    action_source = make_action_source(cfg)

    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_id,
        fps=cfg.fps,
        root=Path(cfg.output_root) / cfg.repo_id,
        robot_type=ROBOT_TYPE,
        features=adapter.build_dataset_features(),
        use_videos=cfg.use_videos,
    )

    for ep_idx in range(cfg.num_episodes):
        raw_obs, info = runtime.reset()
        action_source.reset()

        for step in range(cfg.max_steps_per_episode):
            env_action = action_source.next_action(raw_obs, info)
            next_obs, reward, terminated, truncated, info = runtime.step(env_action)

            frame = adapter.raw_obs_to_dataset_frame(
                next_obs, env_action, info, task=info.get("task", cfg.task_name)
            )
            dataset.add_frame(frame)

            raw_obs = next_obs
            if terminated or truncated:
                break

        dataset.save_episode(task=info.get("task", cfg.task_name))
        print(f"Episode {ep_idx + 1}/{cfg.num_episodes} saved "
              f"(success={info.get('success', 'N/A')})")

    dataset.finalize()
    print(f"Dataset finalized: {dataset.repo_id}")
```

### 2.5 `record/action_sources.py` — 动作源抽象

```python
"""动作源抽象层：统一 expert/policy/teleop 的接口。"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ActionSource(ABC):
    """动作源接口。所有动作源（expert、policy、teleop）统一实现此接口。"""

    @abstractmethod
    def reset(self) -> None:
        """episode 开始时重置内部状态。"""
        ...

    @abstractmethod
    def next_action(self, raw_obs: dict, info: dict) -> np.ndarray:
        """根据当前观测返回动作。"""
        ...


class ExpertActionSource(ActionSource):
    """RobotTwin2 内置 expert planner（CuRobo IK）。"""

    def __init__(self, task_name: str, task_config: str):
        self.task_name = task_name
        self.task_config = task_config
        self._planner = None

    def reset(self) -> None:
        # 重置 planner 状态
        if self._planner is not None:
            self._planner.reset()

    def next_action(self, raw_obs: dict, info: dict) -> np.ndarray:
        # 调用 RobotTwin2 expert planner
        raise NotImplementedError("需要对接 RobotTwin2 expert planner")


class PolicyActionSource(ActionSource):
    """从已训练的 LeRobot policy 推理动作。"""

    def __init__(self, policy_path: str | None):
        self.policy_path = policy_path
        self._policy = None
        self._adapter = None

    def reset(self) -> None:
        if self._policy is None:
            from lerobot.policies.factory import make_policy
            self._policy = make_policy(self.policy_path)
            self._policy.eval()

    def next_action(self, raw_obs: dict, info: dict) -> np.ndarray:
        from bt.robotwin2.common.canonical_adapter import RobotTwinCanonicalAdapter
        if self._adapter is None:
            self._adapter = RobotTwinCanonicalAdapter()
        obs = self._adapter.raw_obs_to_policy_obs(raw_obs, info)
        action = self._policy.select_action(obs)
        return self._adapter.policy_action_to_env_action(action.cpu().numpy())


class TeleopActionSource(ActionSource):
    """遥操作（键盘/手柄）动作源。"""

    def reset(self) -> None:
        pass

    def next_action(self, raw_obs: dict, info: dict) -> np.ndarray:
        raise NotImplementedError("需要对接遥操作设备")
```

### 2.6 `eval/env_config.py` — 环境配置注册

```python
"""注册 RobotTwin2 评测环境配置，供 lerobot-eval 使用。

使用方式:
  PYTHONPATH=. lerobot-eval \
    --env.discover_packages_path=bt.robotwin2.eval \
    --env.type=robotwin2_eval \
    --env.task=beat_block_hammer \
    --policy.path=/path/to/checkpoint

工作原理:
  1. lerobot-eval 解析 --env.discover_packages_path=bt.robotwin2.eval
  2. 自动 import bt.robotwin2.eval（触发本文件中 @register_subclass）
  3. --env.type=robotwin2_eval 匹配到 RobotTwinEvalEnvConfig
  4. make_env() 用 cfg.package_name 导入 gym_registration.py 完成 gym 注册
  5. gym.make(cfg.gym_id, **cfg.gym_kwargs) 创建环境实例
"""

from dataclasses import dataclass

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig

from bt.robotwin2.common.schema import DEFAULT_FPS, ENV_FEATURES_MAP, GYM_ENV_ID


@EnvConfig.register_subclass("robotwin2_eval")
@dataclass
class RobotTwinEvalEnvConfig(EnvConfig):
    task: str | None = None
    task_config: str = "demo_clean"
    embodiment: str = "aloha-agilex"
    fps: int = DEFAULT_FPS
    episode_length: int = 300

    @property
    def package_name(self) -> str:
        return "bt.robotwin2.eval.gym_registration"

    @property
    def gym_id(self) -> str:
        return GYM_ENV_ID

    @property
    def gym_kwargs(self) -> dict:
        return {
            "task": self.task,
            "task_config": self.task_config,
            "embodiment": self.embodiment,
            "fps": self.fps,
            "episode_length": self.episode_length,
        }

    @property
    def features(self) -> dict[str, PolicyFeature]:
        return {
            "observation.images.head": PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3)),
            "observation.images.left_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3)),
            "observation.images.right_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }

    @property
    def features_map(self) -> dict[str, str]:
        return ENV_FEATURES_MAP
```

### 2.7 `eval/gym_env.py` — Gym 环境适配

```python
"""将 RobotTwin2 runtime 包装为 gymnasium.Env。

关键契约：
  - observation 结构必须与 EnvConfig.features_map 匹配
  - 输出 pixels.{head,left_wrist,right_wrist} 和 agent_pos
  - LeRobot eval 通过 features_map 将其映射为 canonical key
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bt.robotwin2.common.canonical_adapter import CAMERA_KEY_MAP


class RobotTwinEvalEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task: str,
        task_config: str = "demo_clean",
        embodiment: str = "aloha-agilex",
        fps: int = 15,
        episode_length: int = 300,
    ):
        super().__init__()
        self.task = task
        self.task_config = task_config
        self.embodiment = embodiment
        self.fps = fps
        self.episode_length = episode_length

        # 初始化 RobotTwin2 runtime
        self.runtime = self._make_runtime()
        self._step_count = 0

        # Gym spaces（必须与 EnvConfig.features 维度一致）
        self.observation_space = spaces.Dict({
            "pixels": spaces.Dict({
                "head": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                "left_wrist": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                "right_wrist": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
            }),
            "agent_pos": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)

    def _make_runtime(self):
        """创建 RobotTwin2 仿真 runtime。

        需要对接 SAPIEN runtime:
          import sapien.core as sapien
          from robotwin2.envs import make_env as make_rt2_env
          return make_rt2_env(self.task, self.task_config)
        """
        # TODO: 实际对接 SAPIEN + CuRobo
        raise NotImplementedError

    def _raw_to_obs(self, raw_obs: dict) -> dict:
        """RobotTwin2 原始观测 → gym observation dict。

        注意: 这里输出的 key (pixels.head 等) 必须与
              EnvConfig.features_map 中的 key 完全匹配。
        """
        return {
            "pixels": {
                "head": raw_obs["cam_high"],
                "left_wrist": raw_obs["cam_left_wrist"],
                "right_wrist": raw_obs["cam_right_wrist"],
            },
            "agent_pos": raw_obs["qpos"].astype(np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        raw_obs, info = self.runtime.reset(seed=seed)
        self._step_count = 0
        info["task"] = self.task
        return self._raw_to_obs(raw_obs), info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.runtime.step(action)
        self._step_count += 1
        if self._step_count >= self.episode_length:
            truncated = True
        info["task"] = self.task
        return self._raw_to_obs(raw_obs), reward, terminated, truncated, info

    def render(self):
        return self.runtime.render()

    def task_description(self) -> str:
        """LeRobot eval 调用此方法获取 task 文本。"""
        return self.task

    def close(self):
        if hasattr(self, "runtime"):
            self.runtime.close()
```

### 2.8 `eval/gym_registration.py` — Gym 注册

```python
"""注册 RobotTwin2 评测环境到 gymnasium。

被 LeRobot 的 importlib.import_module(cfg.package_name) 自动加载。
导入此模块即完成 gym 环境注册，无需手动调用。
"""

import gymnasium as gym

from bt.robotwin2.common.schema import GYM_ENV_ID

gym.register(
    id=GYM_ENV_ID,
    entry_point="bt.robotwin2.eval.gym_env:RobotTwinEvalEnv",
)
```

### 2.9 `eval/__init__.py` — 插件入口

```python
"""bt.robotwin2.eval 插件入口。

被 --env.discover_packages_path=bt.robotwin2.eval 触发导入。
导入时自动完成:
  1. RobotTwinEvalEnvConfig 注册到 EnvConfig
  2. RobotTwinEvalEnv 注册到 gymnasium
"""

from bt.robotwin2.eval.env_config import RobotTwinEvalEnvConfig  # noqa: F401
from bt.robotwin2.eval import gym_registration  # noqa: F401
```

### 2.10 `train/stage_runner.py` — 多阶段训练编排

```python
"""用子进程编排多个 lerobot-train stage。不修改 lerobot-train 任何代码。

用法:
  PYTHONPATH=. python -m bt.robotwin2.train.stage_runner \
    --config bt/robotwin2/configs/train/curriculum_example.yaml

工作原理:
  每个 stage 都是一次独立的 lerobot-train 子进程调用，
  stage 之间通过 checkpoint 路径串联（上一个 stage 的输出作为下一个的 --policy.path）。
"""

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageConfig:
    """单个训练阶段的配置。"""
    name: str                              # stage 标识（如 synth_clean, synth_rand, real_ft）
    repo_id: str                           # 数据集 repo_id
    root: str                              # 数据集本地路径
    output_dir: str                        # 输出目录
    steps: int = 100_000                   # 训练步数
    batch_size: int = 8
    policy_type: str = "act"               # ACT, Diffusion, Groot2, StrGroot 等
    extra_args: list[str] = field(default_factory=list)  # 额外 CLI 参数


@dataclass
class CurriculumConfig:
    """多阶段训练课程的配置。"""
    stages: list[StageConfig] = field(default_factory=list)
    run_offline_val: bool = True           # 每个 stage 后做 held-out validation
    run_online_eval: bool = False          # 每个 stage 后做仿真 eval
    eval_task: str | None = None           # 仿真 eval 的任务名


def run_stage(stage: StageConfig, pretrained_path: str | None = None) -> Path:
    """运行单个训练 stage，返回 checkpoint 路径。"""
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--policy.type={stage.policy_type}",
        f"--dataset.repo_id={stage.repo_id}",
        f"--dataset.root={stage.root}",
        f"--output_dir={stage.output_dir}",
        f"--job_name={stage.name}",
        f"--steps={stage.steps}",
        f"--batch_size={stage.batch_size}",
    ]
    if pretrained_path:
        cmd.append(f"--policy.path={pretrained_path}")
    cmd.extend(stage.extra_args)

    print(f"[stage_runner] Running stage: {stage.name}")
    print(f"[stage_runner] Command: {' '.join(cmd)}")

    subprocess.run(cmd, check=True)

    # 返回最终 checkpoint 路径
    return Path(stage.output_dir) / "checkpoints" / "last" / "pretrained_model"


def run_curriculum(cfg: CurriculumConfig):
    """按顺序执行多个 stage，每个 stage 继承上一个的 checkpoint。"""
    pretrained_path = None

    for stage in cfg.stages:
        ckpt_path = run_stage(stage, pretrained_path)
        pretrained_path = str(ckpt_path)

        if cfg.run_offline_val:
            print(f"[stage_runner] Running offline validation for {stage.name}...")
            # from bt.robotwin2.train.offline_val import run_offline_val
            # run_offline_val(ckpt_path, stage.repo_id, stage.root)

        if cfg.run_online_eval and cfg.eval_task:
            print(f"[stage_runner] Running online eval for {stage.name}...")
            _run_online_eval(pretrained_path, cfg.eval_task)

    print(f"[stage_runner] Curriculum complete. Final checkpoint: {pretrained_path}")


def _run_online_eval(policy_path: str, task: str):
    """调用 lerobot-eval 做仿真评测。"""
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_eval",
        "--env.discover_packages_path=bt.robotwin2.eval",
        "--env.type=robotwin2_eval",
        f"--env.task={task}",
        f"--policy.path={policy_path}",
    ]
    subprocess.run(cmd, check=True)
```

### 2.11 `eval/benchmark.py` — 批量评测

```python
"""批量跨任务评测：对多个 RobotTwin2 任务运行 lerobot-eval。

用法:
  PYTHONPATH=. python -m bt.robotwin2.eval.benchmark \
    --policy_path /path/to/checkpoint \
    --tasks beat_block_hammer,pick_place_cube,pour_water
"""

import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    policy_path: str = ""
    tasks: list[str] = field(default_factory=lambda: [
        "beat_block_hammer",
        "pick_place_cube",
        "pour_water",
    ])
    n_episodes: int = 20
    n_envs: int = 1


def run_benchmark(cfg: BenchmarkConfig) -> dict[str, dict]:
    """对每个任务运行 eval，返回 {task: metrics}。"""
    results = {}
    for task in cfg.tasks:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {task}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "-m", "lerobot.scripts.lerobot_eval",
            "--env.discover_packages_path=bt.robotwin2.eval",
            "--env.type=robotwin2_eval",
            f"--env.task={task}",
            f"--policy.path={cfg.policy_path}",
            f"--eval.n_episodes={cfg.n_episodes}",
            f"--eval.batch_size={cfg.n_envs}",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        results[task] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    # 汇总输出
    print(f"\n{'='*60}")
    print("Benchmark Summary")
    print(f"{'='*60}")
    for task, res in results.items():
        status = "PASS" if res["returncode"] == 0 else "FAIL"
        print(f"  {task}: {status}")

    return results
```

---

## Phase 3: 序列图

### 3.1 数据转换序列图

```
User               convert.py          CanonicalAdapter    LeRobotDataset
 │                    │                      │                   │
 │── run convert ────►│                      │                   │
 │                    │── build_features() ─►│                   │
 │                    │◄─ features dict ─────│                   │
 │                    │                      │                   │
 │                    │── create(repo_id, features) ────────────►│
 │                    │◄─ dataset ──────────────────────────────│
 │                    │                      │                   │
 │                    │ loop episodes:       │                   │
 │                    │  loop steps:         │                   │
 │                    │── raw_obs_to_frame()─►│                  │
 │                    │◄─ frame ────────────│                   │
 │                    │── add_frame(frame) ─────────────────────►│
 │                    │                      │                   │
 │                    │── save_episode() ───────────────────────►│
 │                    │                      │                   │
 │                    │── finalize() ───────────────────────────►│
 │◄─ done ───────────│                      │                   │
```

### 3.2 仿真录制序列图

```
User      run_record   Runtime   ActionSource   Adapter    LeRobotDataset
 │           │           │           │             │            │
 │── start ─►│           │           │             │            │
 │           │── build_features() ──────────────►│            │
 │           │── create(features) ───────────────────────────►│
 │           │           │           │             │            │
 │           │ loop episodes:        │             │            │
 │           │── reset()─►│          │             │            │
 │           │◄─ obs ────│           │             │            │
 │           │── reset()────────────►│             │            │
 │           │           │           │             │            │
 │           │  loop steps:          │             │            │
 │           │── next_action(obs) ──►│             │            │
 │           │◄─ action ────────────│             │            │
 │           │── step(action) ──────►│             │            │
 │           │◄─ obs, reward, done ─│             │            │
 │           │── raw_obs_to_frame() ─────────────►│            │
 │           │◄─ frame ──────────────────────────│            │
 │           │── add_frame(frame) ────────────────────────────►│
 │           │           │           │             │            │
 │           │── save_episode() ──────────────────────────────►│
 │           │                                                 │
 │           │── finalize() ──────────────────────────────────►│
 │◄─ done ──│
```

### 3.3 评测序列图（最复杂的流程）

```
lerobot-eval     PluginLoader     EnvConfig        GymEnv       Runtime    Policy
    │                │               │               │            │          │
    │── parse CLI ──►│               │               │            │          │
    │                │── import bt.robotwin2.eval    │            │          │
    │                │   ├── env_config.py           │            │          │
    │                │   │   └── @register_subclass("robotwin2_eval")       │
    │                │   └── gym_registration.py     │            │          │
    │                │       └── gymnasium.register()│            │          │
    │                │               │               │            │          │
    │── --env.type=robotwin2_eval ──►│               │            │          │
    │◄─ RobotTwinEvalEnvConfig ─────│               │            │          │
    │                │               │               │            │          │
    │── make_env(cfg) ──────────────►│               │            │          │
    │                │               │── gym.make(   │            │          │
    │                │               │   gym_id,     │            │          │
    │                │               │   **kwargs) ─►│            │          │
    │                │               │               │── init() ─►│          │
    │◄─ vec_env ─────────────────────────────────────│            │          │
    │                │               │               │            │          │
    │── make_policy(policy.path) ──────────────────────────────────────────►│
    │◄─ policy ─────────────────────────────────────────────────────────────│
    │                │               │               │            │          │
    │ loop n_episodes:               │               │            │          │
    │── env.reset() ────────────────────────────────►│── reset()─►│          │
    │◄─ obs {pixels, agent_pos} ────────────────────│◄─ raw_obs─│          │
    │                │               │               │            │          │
    │── features_map: pixels.head → observation.images.head      │          │
    │── preprocess(obs) ──────────────────────────────────────────────────►│
    │── policy.select_action() ──────────────────────────────────────────►│
    │◄─ action ────────────────────────────────────────────────────────────│
    │                │               │               │            │          │
    │── env.step(action) ───────────────────────────►│── step() ─►│         │
    │◄─ obs, reward, done ──────────────────────────│◄───────────│          │
    │                │               │               │            │          │
    │── aggregate metrics ──────────►│               │            │          │
    │── log results                  │               │            │          │
```

### 3.4 多阶段训练序列图

```
User        stage_runner         lerobot-train      offline_val     lerobot-eval
 │              │                     │                 │               │
 │── start ────►│                     │                 │               │
 │              │                     │                 │               │
 │              │ stage0: synth_clean │                 │               │
 │              │── subprocess ──────►│                 │               │
 │              │   (--policy.type=   │                 │               │
 │              │    --dataset.repo_id│                 │               │
 │              │    --steps=...)     │                 │               │
 │              │◄─ checkpoint0 ─────│                 │               │
 │              │── validate ─────────────────────────►│               │
 │              │◄─ val_loss ────────────────────────│               │
 │              │                     │                 │               │
 │              │ stage1: synth_rand  │                 │               │
 │              │── subprocess ──────►│                 │               │
 │              │   (--policy.path=   │                 │               │
 │              │    checkpoint0)     │                 │               │
 │              │◄─ checkpoint1 ─────│                 │               │
 │              │── eval ──────────────────────────────────────────────►│
 │              │   (--env.discover_packages_path=bt.robotwin2.eval)   │
 │              │◄─ success_rate ──────────────────────────────────────│
 │              │                     │                 │               │
 │              │ stage2: real_ft     │                 │               │
 │              │── subprocess ──────►│                 │               │
 │              │   (--policy.path=   │                 │               │
 │              │    checkpoint1)     │                 │               │
 │              │◄─ checkpoint2 ─────│                 │               │
 │              │                     │                 │               │
 │◄─ final ckpt│                     │                 │               │
```

---

## Phase 4: 类图

```
┌──────────────────────────────────┐
│    RobotTwinDatasetProfile       │
├──────────────────────────────────┤
│ +action_mode: str                │
│ +camera_keys: tuple[str, ...]    │
│ +state_dim: int                  │
│ +action_dim: int                 │
│ +image_shape: tuple              │
│ +include_endpose: bool           │
│ +instruction_strategy: str       │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│    RobotTwinCanonicalAdapter     │
├──────────────────────────────────┤
│ +profile: RobotTwinDatasetProfile│
├──────────────────────────────────┤
│ +build_dataset_features()        │ → dict[str, PolicyFeature]
│ +raw_obs_to_dataset_frame()      │ → dict[str, Any]
│ +raw_obs_to_policy_obs()         │ → dict[str, Any]
│ +policy_action_to_env_action()   │ → np.ndarray
│ +extract_episode_meta()          │ → dict[str, Any]
└──────────┬───────────────────────┘
           │ 被以下模块复用 (单一数据契约)
     ┌─────┼─────────────────┐
     │     │                 │
     ▼     ▼                 ▼
┌────────┐ ┌──────────┐ ┌──────────────┐
│convert │ │run_record│ │RobotTwinEval │
│  .py   │ │  .py     │ │  Env         │
└────┬───┘ └────┬─────┘ └──────┬───────┘
     │          │              │
     ▼          ▼              │
┌─────────────────────┐        │
│ LeRobotDataset      │        │
│ (LeRobot 原生 API)   │        │
│ .create()            │        │
│ .add_frame()         │        │
│ .save_episode()      │        │
│ .finalize()          │        │
└─────────────────────┘        │
                               │
                               ▼
                  ┌──────────────────────┐
                  │ gymnasium.Env        │
                  │ (标准 gym 接口)       │
                  │ .reset()             │
                  │ .step()              │
                  │ .close()             │
                  └──────────────────────┘

┌─────────────────────────┐
│    ActionSource          │
│    <<abstract>>          │
├─────────────────────────┤
│ +reset()                │
│ +next_action(obs, info) │ → np.ndarray
└──────────┬──────────────┘
     ┌─────┼──────────┐
     │     │          │
     ▼     ▼          ▼
┌────────┐┌────────┐┌────────┐
│Expert  ││Policy  ││Teleop  │
│Action  ││Action  ││Action  │
│Source  ││Source  ││Source  │
└────────┘└────────┘└────────┘

┌──────────────────────────────────┐
│  RobotTwinEvalEnvConfig          │
│  @EnvConfig.register_subclass()  │
├──────────────────────────────────┤
│ +task: str                       │
│ +task_config: str                │
│ +embodiment: str                 │
│ +fps: int                        │
│ +episode_length: int             │
├──────────────────────────────────┤
│ +package_name → str              │  "bt.robotwin2.eval.gym_registration"
│ +gym_id → str                    │  "bt_robotwin2/RobotTwinEval-v0"
│ +gym_kwargs → dict               │  {task, task_config, embodiment, ...}
│ +features → dict                 │  {obs.images.head: VISUAL, ...}
│ +features_map → dict             │  {pixels.head → obs.images.head, ...}
└──────────────────────────────────┘
         │ extends
         ▼
┌──────────────────────────────────┐
│  EnvConfig (LeRobot 原生)         │
│  draccus.ChoiceRegistry          │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│  StageConfig                     │
├──────────────────────────────────┤
│ +name: str                       │
│ +repo_id: str                    │
│ +root: str                       │
│ +output_dir: str                 │
│ +steps: int                      │
│ +batch_size: int                 │
│ +policy_type: str                │
│ +extra_args: list[str]           │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│  CurriculumConfig                │
├──────────────────────────────────┤
│ +stages: list[StageConfig]       │
│ +run_offline_val: bool           │
│ +run_online_eval: bool           │
│ +eval_task: str | None           │
├──────────────────────────────────┤
│ run_curriculum()                 │  → 顺序执行 stages，串联 checkpoint
└──────────────────────────────────┘
```

---

## Phase 5: 精确目录树

```text
bt/robotwin2/
├── __init__.py
├── design.md                       # 已存在 (原始设计文档)
├── design2.md                      # 本文档 (详细实施设计)
│
├── common/                         # 统一数据契约层
│   ├── __init__.py
│   ├── canonical_adapter.py        # 核心: RobotTwinCanonicalAdapter
│   ├── schema.py                   # 常量: FPS, ROBOT_TYPE, GYM_ENV_ID, FEATURES_MAP
│   ├── profiles.py                 # 不同数据 profile (qpos/ee/hybrid)
│   ├── metadata.py                 # lineage / sidecar 元数据
│   └── paths.py                    # 输出目录规则
│
├── data/                           # 离线数据处理
│   ├── __init__.py
│   ├── generate.py                 # 调用 RobotTwin2 原生数据生成
│   ├── convert.py                  # HDF5/raw → LeRobotDataset v3
│   ├── split_merge.py              # train/val/test 切分
│   └── validate_dataset.py         # dataset smoke check
│
├── record/                         # 仿真实时录制
│   ├── __init__.py
│   ├── run_record.py               # 录制主入口
│   ├── action_sources.py           # ActionSource: Expert/Policy/Teleop
│   └── episode_controller.py       # episode 结束条件
│
├── train/                          # 训练编排
│   ├── __init__.py
│   ├── stage_runner.py             # 多 stage 编排 (子进程调 lerobot-train)
│   ├── curriculum.py               # curriculum 策略定义
│   ├── offline_val.py              # held-out validation
│   └── checkpoint_eval.py          # 训练后自动 eval
│
├── eval/                           # 评测插件
│   ├── __init__.py                 # 插件入口 (导入 env_config + gym_registration)
│   ├── env_config.py               # RobotTwinEvalEnvConfig (@register_subclass)
│   ├── gym_registration.py         # gymnasium.register()
│   ├── gym_env.py                  # RobotTwinEvalEnv(gym.Env)
│   ├── benchmark.py                # 批量跨任务评测
│   └── run_eval.py                 # CLI wrapper
│
├── configs/                        # YAML 配置模板
│   ├── data/
│   │   ├── generate_example.yaml
│   │   └── convert_example.yaml
│   ├── record/
│   │   ├── record_expert.yaml
│   │   └── record_policy.yaml
│   ├── train/
│   │   └── curriculum_example.yaml
│   └── eval/
│       ├── eval_task.yaml
│       └── benchmark.yaml
│
└── scripts/                        # Shell 脚本入口
    ├── convert.sh
    ├── record.sh
    ├── train_curriculum.sh
    └── eval_benchmark.sh
```

---

## Phase 6: 实施顺序

### Step 1: common/ 层 (基础)
- 实现 `canonical_adapter.py`、`schema.py`、`profiles.py`
- 这是后续所有模块的基础，必须先完成
- 验证：单元测试 adapter 的各方法输出格式

### Step 2: data/ 层 (数据转换)
- 实现 `convert.py`（最核心）
- 实现 `generate.py`（如果需要从头生成）、`split_merge.py`
- 验证：转换一份 RobotTwin2 数据，用 `lerobot-info` 检查输出

```bash
PYTHONPATH=. python -m bt.robotwin2.data.convert \
  --input_path /path/to/robotwin2/raw \
  --output_root bt/robotwin2/output/datasets \
  --repo_id local/test_convert

lerobot-info --repo_id local/test_convert \
  --root bt/robotwin2/output/datasets/local/test_convert
```

### Step 3: eval/ 层 (评测插件)
- 实现 `env_config.py`、`gym_registration.py`、`gym_env.py`
- 实现 `eval/__init__.py` 作为插件入口
- 验证：使用 lerobot-eval 加载插件

```bash
PYTHONPATH=. lerobot-eval \
  --env.discover_packages_path=bt.robotwin2.eval \
  --env.type=robotwin2_eval \
  --env.task=beat_block_hammer \
  --policy.path=/path/to/checkpoint
```

### Step 4: record/ 层 (仿真录制)
- 实现 `run_record.py`、`action_sources.py`
- 对接 RobotTwin2 runtime API
- 验证：在仿真中录制并用 `lerobot-info` 检查输出

### Step 5: train/ 层 (训练编排)
- 实现 `stage_runner.py`、`curriculum.py`、`offline_val.py`
- 验证：对转换后的数据集执行完整训练流程

```bash
# 单步训练验证
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=local/test_convert \
  --dataset.root=bt/robotwin2/output/datasets/local/test_convert \
  --batch_size=4 --steps=100

# 多阶段训练验证
PYTHONPATH=. python -m bt.robotwin2.train.stage_runner \
  --config bt/robotwin2/configs/train/curriculum_example.yaml
```

---

## 附录 A: 数据一致性保证

### A.1 核心不变量

整个设计中最重要的不变量是：**data/record/eval 三条路径产出的 observation/action key 和维度必须完全一致**。

```
                    ┌─────────────────────────┐
                    │  RobotTwinCanonicalAdapter │
                    │  (唯一映射定义点)           │
                    └──────────┬──────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
    data/convert.py      record/run_record.py  eval/gym_env.py
    raw_obs_to_frame()   raw_obs_to_frame()    _raw_to_obs()
          │                    │                    │
          ▼                    ▼                    ▼
    LeRobotDataset       LeRobotDataset        features_map →
    (Parquet + MP4)      (Parquet + MP4)        canonical keys
```

如果三条路径中的任何一条偏离了 adapter 的映射定义，就会出现 train/eval 不一致的 bug。因此：
- **禁止**在 convert.py / run_record.py / gym_env.py 中硬编码映射逻辑
- **所有映射**都必须经过 `RobotTwinCanonicalAdapter` 的方法

### A.2 维度一致性检查链

```
RobotTwinDatasetProfile.state_dim  ──► canonical_adapter.build_features()
                                         ──► LeRobotDataset features
                                         ──► EnvConfig.features (eval)
                                         ──► policy input shape
```

如果 profile 的 `state_dim=14`，则：
- `convert.py` 产出的 `observation.state` 列必须是 (14,)
- `gym_env.py` 的 `agent_pos` space 必须是 (14,)
- `env_config.py` 的 features 中 `observation.state` 必须是 (14,)

---

## 附录 B: 与已有策略的兼容性

本分支已有的策略（ACT, Diffusion, SmolVLA, Groot2, StrGroot）与 RobotTwin2 整合层的关系：

| 策略 | 所在位置 | 与 RobotTwin2 数据兼容性 | 说明 |
|------|---------|------------------------|------|
| ACT | `src/lerobot/policies/act/` (原生) | 直接兼容 | 支持任意 state/action/image 维度 |
| Diffusion | `src/lerobot/policies/diffusion/` (原生) | 直接兼容 | 支持任意维度 |
| SmolVLA | `src/lerobot/policies/smolvla/` (已修改) | 直接兼容 | VLA 策略，支持 language + image |
| Groot2 | `src/lerobot/policies/groot2/` (新增) | 需验证 | Eagle2 VLM，可能有图像尺寸要求 |
| StrGroot | `src/lerobot/policies/str_groot/` (新增) | 需验证 | StarVLA/Qwen，可能有特定输入格式 |

训练时通过 `--policy.type=act/diffusion/smolvla/groot2/str_groot` 选择策略，bt/robotwin2 的 `stage_runner.py` 只负责编排，不干涉策略内部逻辑。
