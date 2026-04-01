#!/usr/bin/env python
"""Headless smoke test for the minimal HIL simulation integration path.

This demo intentionally avoids changes under `src/` and only reuses the
existing LeRobot HIL pieces:

- `gym_hil` wrapped simulation env
- `GymHILAdapterProcessorStep`
- `InterventionActionProcessorStep`
- `LeRobotDataset`

The interactive keyboard/gamepad path still exists via
`bt/hil/interactive_keyboard_record.json`, but this script is the version that
can be executed and debugged in the current headless machine session.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
CACHE_DIR = THIS_DIR / ".cache"
MESA_CACHE_DIR = CACHE_DIR / "mesa"
HF_HOME_DIR = CACHE_DIR / "hf_home"
HF_DATASETS_CACHE_DIR = CACHE_DIR / "hf_datasets"

# Set headless rendering before importing gym_hil / mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MESA_SHADER_CACHE_DIR", str(MESA_CACHE_DIR))
os.environ.setdefault("HF_HOME", str(HF_HOME_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE_DIR))

CACHE_DIR.mkdir(parents=True, exist_ok=True)
MESA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_HOME_DIR.mkdir(parents=True, exist_ok=True)
HF_DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

import gymnasium as gym
import numpy as np
import torch
from gym_hil.wrappers.factory import make_env as make_gym_hil_env

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import GripperConfig, HILSerlProcessorConfig, HILSerlRobotEnvConfig, ResetConfig
from lerobot.processor import TransitionKey, create_transition
from lerobot.rl.gym_manipulator import DatasetConfig, GymManipulatorConfig, make_processors
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD


@dataclass(frozen=True)
class ScriptedTakeoverEvent:
    action: tuple[float, float, float, float]
    is_intervention: bool = True
    success: bool = False
    terminate_episode: bool = False
    rerecord_episode: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScriptedTakeoverEvent":
        action = tuple(float(x) for x in payload["action"])
        if len(action) != 4:
            raise ValueError(f"Expected 4 action values, got {len(action)} in {payload}")
        return cls(
            action=action,
            is_intervention=bool(payload.get("is_intervention", True)),
            success=bool(payload.get("success", False)),
            terminate_episode=bool(payload.get("terminate_episode", False)),
            rerecord_episode=bool(payload.get("rerecord_episode", False)),
        )

    def to_info_dict(self) -> dict[str, Any]:
        return {
            "teleop_action": {
                "delta_x": self.action[0],
                "delta_y": self.action[1],
                "delta_z": self.action[2],
                "gripper": self.action[3],
            },
            "is_intervention": self.is_intervention,
            TeleopEvents.IS_INTERVENTION: self.is_intervention,
            TeleopEvents.SUCCESS: self.success,
            TeleopEvents.TERMINATE_EPISODE: self.terminate_episode,
            TeleopEvents.RERECORD_EPISODE: self.rerecord_episode,
        }


class ScriptedInterventionWrapper(gym.Wrapper):
    """Inject scripted intervention signals into the env `info`.

    The returned `info` always represents the intervention state to be consumed
    *before the next action is produced*, matching the way
    `gym_manipulator.step_env_and_process_transition` uses transition info.
    """

    def __init__(self, env: gym.Env, schedule: list[ScriptedTakeoverEvent]):
        super().__init__(env)
        self.schedule = schedule
        self._event_index = 0

    def _current_event_info(self) -> dict[str, Any]:
        if self._event_index >= len(self.schedule):
            return {
                "is_intervention": False,
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        return self.schedule[self._event_index].to_info_dict()

    def reset(self, **kwargs):
        self._event_index = 0
        obs, info = self.env.reset(**kwargs)
        merged_info = dict(info)
        merged_info.update(self._current_event_info())
        return obs, merged_info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._event_index += 1
        merged_info = dict(info)
        merged_info.update(self._current_event_info())
        return obs, reward, terminated, truncated, merged_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bt/hil scripted takeover smoke test.")
    parser.add_argument(
        "--schedule",
        type=Path,
        default=THIS_DIR / "scripted_takeover_schedule.json",
        help="JSON file describing the scripted intervention schedule.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=THIS_DIR / "output" / "scripted_takeover_dataset",
        help="Where to write the demo dataset.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="How many episodes to record.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed used when resetting the environment.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="scripted_pick_cube",
        help="Task string stored inside the recorded dataset.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the previous output directory before running.",
    )
    return parser.parse_args()


def load_schedule(path: Path) -> list[ScriptedTakeoverEvent]:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError(f"Schedule at {path} must be a non-empty list.")
    return [ScriptedTakeoverEvent.from_dict(item) for item in payload]


def build_cfg(output_root: Path, episodes: int, task_name: str) -> GymManipulatorConfig:
    env_cfg = HILSerlRobotEnvConfig(
        task="PandaPickCubeBase-v0",
        fps=10,
        name="gym_hil",
        processor=HILSerlProcessorConfig(
            control_mode="scripted",
            gripper=GripperConfig(use_gripper=True, gripper_penalty=-0.02),
            reset=ResetConfig(
                fixed_reset_joint_positions=[0.0, 0.195, 0.0, -2.43, 0.0, 2.62, 0.785],
                reset_time_s=0.0,
                control_time_s=2.0,
                terminate_on_success=True,
            ),
        ),
    )
    dataset_cfg = DatasetConfig(
        repo_id="local/scripted_takeover_demo",
        root=str(output_root),
        task=task_name,
        num_episodes_to_record=episodes,
        replay_episode=None,
        push_to_hub=False,
    )
    return GymManipulatorConfig(
        env=env_cfg,
        dataset=dataset_cfg,
        mode="record",
        device="cpu",
    )


def make_env(schedule: list[ScriptedTakeoverEvent]) -> gym.Env:
    env = make_gym_hil_env(
        "gym_hil/PandaPickCubeBase-v0",
        image_obs=True,
        use_inputs_control=False,
        use_gripper=True,
        gripper_penalty=-0.02,
        reset_delay_seconds=0.0,
    )
    return ScriptedInterventionWrapper(env, schedule)


def build_dataset(output_root: Path, transition: dict, cfg: GymManipulatorConfig) -> LeRobotDataset:
    features = {
        ACTION: {
            "dtype": "float32",
            "shape": (4,),
            "names": ["delta_x", "delta_y", "delta_z", "gripper"],
        },
        REWARD: {"dtype": "float32", "shape": (1,), "names": None},
        DONE: {"dtype": "bool", "shape": (1,), "names": None},
        "complementary_info.discrete_penalty": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["discrete_penalty"],
        },
    }

    for key, value in transition[TransitionKey.OBSERVATION].items():
        squeezed = value.squeeze(0)
        if key == OBS_STATE:
            features[key] = {
                "dtype": "float32",
                "shape": tuple(squeezed.shape),
                "names": None,
            }
        elif "image" in key:
            features[key] = {
                "dtype": "video",
                "shape": tuple(squeezed.shape),
                "names": ["channels", "height", "width"],
            }

    return LeRobotDataset.create(
        repo_id=cfg.dataset.repo_id,
        fps=cfg.env.fps,
        root=output_root,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
        features=features,
    )


def neutral_action(use_gripper: bool) -> torch.Tensor:
    action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    if use_gripper:
        action = torch.cat([action, torch.tensor([0.0], dtype=torch.float32)])
    return action


def ensure_action_dim(
    action_numpy: np.ndarray,
    action_tensor: torch.Tensor,
    use_gripper: bool,
) -> tuple[np.ndarray, torch.Tensor]:
    """Pad scripted intervention actions to 4D when the current core path drops gripper."""
    if use_gripper and action_numpy.shape[0] == 3:
        action_numpy = np.concatenate([action_numpy, np.array([1.0], dtype=np.float32)])
        action_tensor = torch.cat([action_tensor, torch.tensor([1.0], dtype=torch.float32)])
    return action_numpy, action_tensor


def step_once(
    env: gym.Env,
    transition: dict,
    env_processor,
    action_processor,
    use_gripper: bool,
) -> tuple[dict, torch.Tensor]:
    transition = dict(transition)
    transition[TransitionKey.ACTION] = neutral_action(use_gripper)
    transition[TransitionKey.OBSERVATION] = (
        env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {}
    )

    processed_action_transition = action_processor(transition)
    executed_action = processed_action_transition[TransitionKey.ACTION]
    if isinstance(executed_action, np.ndarray):
        action_numpy = executed_action
        executed_action_tensor = torch.from_numpy(executed_action.copy())
    elif isinstance(executed_action, torch.Tensor):
        action_numpy = executed_action.detach().cpu().numpy()
        executed_action_tensor = executed_action.detach().cpu()
    else:
        raise TypeError(f"Unsupported action type: {type(executed_action)}")

    action_numpy, executed_action_tensor = ensure_action_dim(
        action_numpy=action_numpy,
        action_tensor=executed_action_tensor,
        use_gripper=use_gripper,
    )

    obs, reward, terminated, truncated, info = env.step(action_numpy)
    reward = float(reward) + float(processed_action_transition[TransitionKey.REWARD])
    terminated = bool(terminated) or bool(processed_action_transition[TransitionKey.DONE])
    truncated = bool(truncated) or bool(processed_action_transition[TransitionKey.TRUNCATED])

    complementary_data = dict(processed_action_transition[TransitionKey.COMPLEMENTARY_DATA])
    new_info = dict(processed_action_transition[TransitionKey.INFO])
    new_info.update(info)

    next_transition = create_transition(
        observation=obs,
        action=action_numpy,
        reward=reward,
        done=terminated,
        truncated=truncated,
        info=new_info,
        complementary_data=complementary_data,
    )
    next_transition = env_processor(next_transition)
    return next_transition, executed_action_tensor


def build_frame(transition: dict, executed_action: torch.Tensor, task_name: str) -> dict[str, Any]:
    observations = {
        k: v.squeeze(0).cpu()
        for k, v in transition[TransitionKey.OBSERVATION].items()
        if isinstance(v, torch.Tensor)
    }

    discrete_penalty = transition[TransitionKey.INFO].get("discrete_penalty", 0.0)
    frame = {
        **observations,
        ACTION: executed_action,
        REWARD: np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
        DONE: np.array(
            [bool(transition[TransitionKey.DONE]) or bool(transition[TransitionKey.TRUNCATED])],
            dtype=bool,
        ),
        "complementary_info.discrete_penalty": np.array([discrete_penalty], dtype=np.float32),
        "task": task_name,
    }
    return frame


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    schedule = load_schedule(args.schedule)
    output_root = args.output_root.resolve()

    if output_root.exists():
        if args.clean:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"{output_root} already exists. Re-run with --clean or choose another --output-root."
            )

    cfg = build_cfg(output_root=output_root, episodes=args.episodes, task_name=args.task_name)
    env = make_env(schedule)
    env_processor = None
    action_processor = None
    dataset = None
    episode_summaries: list[dict[str, Any]] = []

    try:
        env_processor, action_processor = make_processors(env, None, cfg.env, device=cfg.device)

        for episode_idx in range(cfg.dataset.num_episodes_to_record):
            obs, info = env.reset(seed=args.seed + episode_idx)
            transition = create_transition(observation=obs, info=info)
            transition = env_processor(transition)

            if dataset is None:
                dataset = build_dataset(output_root=output_root, transition=transition, cfg=cfg)

            steps = 0
            max_steps = max(len(schedule) + 5, cfg.env.fps * 4)
            while True:
                transition, executed_action = step_once(
                    env=env,
                    transition=transition,
                    env_processor=env_processor,
                    action_processor=action_processor,
                    use_gripper=cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper else True,
                )

                frame = build_frame(transition, executed_action, cfg.dataset.task)
                dataset.add_frame(frame)
                steps += 1

                if transition[TransitionKey.DONE] or transition[TransitionKey.TRUNCATED]:
                    dataset.save_episode(parallel_encoding=False)
                    episode_summaries.append(
                        {
                            "episode_index": episode_idx,
                            "steps": steps,
                            "final_reward": float(transition[TransitionKey.REWARD]),
                            "done": bool(transition[TransitionKey.DONE]),
                            "truncated": bool(transition[TransitionKey.TRUNCATED]),
                        }
                    )
                    break
                if steps >= max_steps:
                    raise RuntimeError(
                        f"Demo exceeded max_steps={max_steps} without finishing an episode. "
                        "Check the scripted takeover schedule."
                    )

        assert dataset is not None
        dataset.finalize()

        reloaded = LeRobotDataset(cfg.dataset.repo_id, root=output_root)
        first_sample = reloaded[0]
        action_sample = first_sample[ACTION].tolist()
        summary = {
            "output_root": str(output_root),
            "repo_id": cfg.dataset.repo_id,
            "num_episodes": reloaded.num_episodes,
            "num_frames": reloaded.num_frames,
            "first_action": action_sample,
            "episode_summaries": episode_summaries,
            "schedule_path": str(args.schedule.resolve()),
        }

        summary_path = output_root.parent / "scripted_takeover_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(json.dumps(summary, indent=2))
        return summary
    finally:
        if dataset is not None:
            dataset.finalize()
        env.close()


def main() -> None:
    args = parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
