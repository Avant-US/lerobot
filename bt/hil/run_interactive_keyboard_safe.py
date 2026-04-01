#!/usr/bin/env python
"""Safer interactive keyboard recording wrapper for gym_hil.

This wrapper avoids the two crash-prone parts seen in remote desktop sessions:

1. Parallel MP4 encoding on episode save.
2. Native MuJoCo viewer cleanup during interpreter shutdown.

To reduce risk, it records camera observations as PNG images instead of MP4
videos, finalizes the dataset explicitly, prints a summary, flushes output, and
then exits the Python process immediately with `os._exit(...)`.
"""

import contextlib
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from gym_hil.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from gym_hil.wrappers.hil_wrappers import DEFAULT_EE_STEP_SIZE, EEActionWrapper, GripperPenaltyWrapper, ResetDelayWrapper
from gym_hil.wrappers.viewer_wrapper import PassiveViewerWrapper
from lerobot.processor import TransitionKey, create_transition
from lerobot.rl.gym_manipulator import GymManipulatorConfig, make_processors, step_env_and_process_transition
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD
from lerobot.utils.robot_utils import precise_sleep


THIS_DIR = Path(__file__).resolve().parent
CACHE_DIR = THIS_DIR / ".cache"
HF_HOME_DIR = CACHE_DIR / "hf_home"
HF_DATASETS_CACHE_DIR = CACHE_DIR / "hf_datasets"

# Keep HF caches under the workspace so dataset reload/finalize does not depend
# on user-level directories with uncertain permissions.
os.environ.setdefault("HF_HOME", str(HF_HOME_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE_DIR))

HF_HOME_DIR.mkdir(parents=True, exist_ok=True)
HF_DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def hard_exit(code: int) -> None:
    """Flush stdio and skip fragile native object destructors."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


class SafeKeyboardController:
    """Minimal keyboard controller with explicit manual episode ending.

    Differences from gym_hil's built-in KeyboardController:
    - `Backspace` is used for failure, matching the on-screen prompt.
    - `ESC` aborts the current run instead of silently marking the episode as failure.
    - Episode completion is only emitted for `Enter` / `Backspace`.
    """

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.open_gripper_command = False
        self.close_gripper_command = False
        self.listener = None
        self.keyboard = None
        self.episode_end_status = None
        self.exit_requested = False
        self.key_states = {
            "forward_x": False,
            "backward_x": False,
            "forward_y": False,
            "backward_y": False,
            "forward_z": False,
            "backward_z": False,
            "intervention": False,
            "rerecord": False,
        }

    def start(self):
        from pynput import keyboard as pynput_keyboard

        self.keyboard = pynput_keyboard

        def char_of(key) -> str | None:
            value = getattr(key, "char", None)
            if value is None:
                return None
            return value.lower()

        def on_press(key):
            try:
                key_char = char_of(key)
                if key == self.keyboard.Key.up:
                    self.key_states["forward_x"] = True
                elif key == self.keyboard.Key.down:
                    self.key_states["backward_x"] = True
                elif key == self.keyboard.Key.left:
                    self.key_states["forward_y"] = True
                elif key == self.keyboard.Key.right:
                    self.key_states["backward_y"] = True
                elif key == self.keyboard.Key.shift:
                    self.key_states["backward_z"] = True
                elif key == self.keyboard.Key.shift_r:
                    self.key_states["forward_z"] = True
                elif key == self.keyboard.Key.ctrl_r:
                    self.open_gripper_command = True
                elif key == self.keyboard.Key.ctrl_l:
                    self.close_gripper_command = True
                elif key == self.keyboard.Key.enter:
                    self.episode_end_status = "success"
                elif key == self.keyboard.Key.backspace:
                    self.episode_end_status = "failure"
                elif key == self.keyboard.Key.esc:
                    self.exit_requested = True
                elif key == self.keyboard.Key.space:
                    self.key_states["intervention"] = not self.key_states["intervention"]
                elif key_char == "o":
                    self.open_gripper_command = True
                elif key_char == "c":
                    self.close_gripper_command = True
            except AttributeError:
                pass

        def on_release(key):
            try:
                key_char = char_of(key)
                if key == self.keyboard.Key.up:
                    self.key_states["forward_x"] = False
                elif key == self.keyboard.Key.down:
                    self.key_states["backward_x"] = False
                elif key == self.keyboard.Key.left:
                    self.key_states["forward_y"] = False
                elif key == self.keyboard.Key.right:
                    self.key_states["backward_y"] = False
                elif key == self.keyboard.Key.shift:
                    self.key_states["backward_z"] = False
                elif key == self.keyboard.Key.shift_r:
                    self.key_states["forward_z"] = False
                elif key == self.keyboard.Key.ctrl_r:
                    self.open_gripper_command = False
                elif key == self.keyboard.Key.ctrl_l:
                    self.close_gripper_command = False
                elif key_char == "o":
                    self.open_gripper_command = False
                elif key_char == "c":
                    self.close_gripper_command = False
            except AttributeError:
                pass

        self.listener = self.keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()
        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift and Shift_R: Move in Z axis")
        print("  Right Ctrl or O: Open gripper")
        print("  Left Ctrl or C: Close gripper")
        print("  Enter: End episode with SUCCESS")
        print("  Backspace: End episode with FAILURE")
        print("  Space: Start/Stop Intervention")
        print("  ESC: Abort current run without saving partial episode")

    def stop(self):
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def update(self):
        return None

    def get_deltas(self):
        delta_x = delta_y = delta_z = 0.0
        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size
        return delta_x, delta_y, delta_z

    def gripper_command(self):
        if self.open_gripper_command and not self.close_gripper_command:
            return "open"
        if self.close_gripper_command and not self.open_gripper_command:
            return "close"
        return "stay"

    def should_intervene(self):
        return self.key_states["intervention"]

    def get_episode_end_status(self):
        status = self.episode_end_status
        self.episode_end_status = None
        return status

    def consume_exit_request(self):
        requested = self.exit_requested
        self.exit_requested = False
        return requested

    def reset(self):
        for key in self.key_states:
            self.key_states[key] = False
        self.open_gripper_command = False
        self.close_gripper_command = False
        self.episode_end_status = None
        self.exit_requested = False


class ManualOnlyKeyboardControlWrapper(gym.Wrapper):
    """Keyboard control wrapper that only ends episodes on Enter/Backspace."""

    def __init__(
        self,
        env: gym.Env,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        use_gripper=True,
        auto_reset=False,
    ):
        super().__init__(env)
        self.controller = SafeKeyboardController(
            x_step_size=x_step_size,
            y_step_size=y_step_size,
            z_step_size=z_step_size,
        )
        self.controller.start()
        self.use_gripper = use_gripper
        self.auto_reset = auto_reset

    def _get_keyboard_action(self):
        self.controller.update()
        delta_x, delta_y, delta_z = self.controller.get_deltas()
        intervention_is_active = self.controller.should_intervene()
        action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)
        if self.use_gripper:
            gripper_command = self.controller.gripper_command()
            if gripper_command == "open":
                action = np.concatenate([action, [2.0]])
            elif gripper_command == "close":
                action = np.concatenate([action, [0.0]])
            else:
                action = np.concatenate([action, [1.0]])
        episode_end_status = self.controller.get_episode_end_status()
        exit_requested = self.controller.consume_exit_request()
        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        return intervention_is_active, action, terminate_episode, success, exit_requested

    def step(self, action):
        is_intervention, keyboard_action, terminate_episode, success, exit_requested = self._get_keyboard_action()
        if is_intervention:
            action = keyboard_action

        obs, reward, env_terminated, env_truncated, info = self.env.step(action)
        info = dict(info)

        if success:
            reward = 1.0

        info["is_intervention"] = is_intervention
        info["teleop_action"] = action
        info["manual_terminate"] = terminate_episode
        info["manual_success"] = success
        info["exit_requested"] = exit_requested
        info["env_terminated"] = bool(env_terminated)
        info["env_truncated"] = bool(env_truncated)

        terminated = bool(terminate_episode)
        truncated = False
        if terminated:
            info["next.success"] = success

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.controller.reset()
        return self.env.reset(**kwargs)

    def close(self):
        self.controller.stop()
        return self.env.close()


def resolve_output_root(root: str | None) -> Path:
    if root is None:
        root_path = THIS_DIR / "output" / "interactive_keyboard_safe_dataset"
    else:
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = (Path.cwd() / root_path).resolve()

    if not root_path.exists():
        return root_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root_path.with_name(f"{root_path.name}_{timestamp}")


def ensure_action_tensor(action: Any, use_gripper: bool) -> torch.Tensor:
    if isinstance(action, dict):
        values = [
            float(action.get("delta_x", 0.0)),
            float(action.get("delta_y", 0.0)),
            float(action.get("delta_z", 0.0)),
        ]
        if use_gripper:
            values.append(float(action.get("gripper", 1.0)))
        tensor = torch.tensor(values, dtype=torch.float32)
    elif isinstance(action, torch.Tensor):
        tensor = action.detach().cpu().to(dtype=torch.float32)
    elif isinstance(action, np.ndarray):
        tensor = torch.from_numpy(action.astype(np.float32, copy=True))
    else:
        raise TypeError(f"Unsupported action type: {type(action)}")

    if use_gripper and tensor.shape[0] == 3:
        tensor = torch.cat([tensor, torch.tensor([1.0], dtype=torch.float32)])
    return tensor


def create_safe_dataset(cfg: GymManipulatorConfig, transition: dict) -> LeRobotDataset:
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True
    features = {
        ACTION: {
            "dtype": "float32",
            "shape": (4,) if use_gripper else (3,),
            "names": ["delta_x", "delta_y", "delta_z", "gripper"] if use_gripper else ["delta_x", "delta_y", "delta_z"],
        },
        REWARD: {"dtype": "float32", "shape": (1,), "names": None},
        DONE: {"dtype": "bool", "shape": (1,), "names": None},
    }
    if use_gripper:
        features["complementary_info.discrete_penalty"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["discrete_penalty"],
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
            # Safe mode stores PNG images directly instead of encoding MP4 videos.
            features[key] = {
                "dtype": "image",
                "shape": tuple(squeezed.shape),
                "names": ["channels", "height", "width"],
            }

    return LeRobotDataset.create(
        repo_id=cfg.dataset.repo_id,
        fps=cfg.env.fps,
        root=cfg.dataset.root,
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
        features=features,
    )


def build_frame(transition: dict, cfg: GymManipulatorConfig) -> dict[str, Any]:
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True
    observations = {
        key: value.squeeze(0).cpu()
        for key, value in transition[TransitionKey.OBSERVATION].items()
        if isinstance(value, torch.Tensor)
    }

    raw_action = transition[TransitionKey.COMPLEMENTARY_DATA].get(
        "teleop_action",
        transition[TransitionKey.ACTION],
    )
    action_tensor = ensure_action_tensor(raw_action, use_gripper=use_gripper)
    frame = {
        **observations,
        ACTION: action_tensor,
        REWARD: np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
        DONE: np.array(
            [bool(transition[TransitionKey.DONE]) or bool(transition[TransitionKey.TRUNCATED])],
            dtype=bool,
        ),
        "task": cfg.dataset.task,
    }
    if use_gripper:
        discrete_penalty = (
            transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty")
            or transition[TransitionKey.INFO].get("discrete_penalty", 0.0)
        )
        frame["complementary_info.discrete_penalty"] = np.array([discrete_penalty], dtype=np.float32)
    return frame


def print_banner(cfg: GymManipulatorConfig, env, env_processor, action_processor) -> None:
    print("Safe keyboard demo: image recording + explicit finalize + hard exit")
    print(f"Output root: {cfg.dataset.root}")
    print("Environment observation space:", env.observation_space)
    print("Environment action space:", env.action_space)
    print("Environment processor:", env_processor)
    print("Action processor:", action_processor)
    print(f"Starting safe control loop at {cfg.env.fps} FPS")
    print("Controls:")
    print("- Focus the MuJoCo window before using the keyboard")
    print("- Space: Start/Stop Intervention")
    print("- Arrow keys / Shift: Move end effector")
    print("- Right Ctrl or O: Open gripper")
    print("- Left Ctrl or C: Close gripper")
    print("- Enter: End episode with SUCCESS")
    print("- Backspace: End episode with FAILURE")
    print("- ESC: Abort current run without saving the current partial episode")


def make_safe_keyboard_env(cfg: GymManipulatorConfig) -> gym.Env:
    base_env = PandaPickCubeGymEnv(
        control_dt=1.0 / cfg.env.fps,
        render_mode="human",
        image_obs=True,
    )
    env = base_env
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True
    gripper_penalty = cfg.env.processor.gripper.gripper_penalty if cfg.env.processor.gripper is not None else 0.0
    reset_delay = cfg.env.processor.reset.reset_time_s if cfg.env.processor.reset is not None else 0.0

    if use_gripper:
        env = GripperPenaltyWrapper(env, penalty=gripper_penalty)
    env = EEActionWrapper(env, ee_action_step_size=DEFAULT_EE_STEP_SIZE, use_gripper=use_gripper)
    env = ManualOnlyKeyboardControlWrapper(env, use_gripper=use_gripper, auto_reset=False)
    env = PassiveViewerWrapper(env)
    env = ResetDelayWrapper(env, delay_seconds=reset_delay)
    return env


def run_safe_record(cfg: GymManipulatorConfig) -> dict[str, Any]:
    if cfg.mode not in {None, "record"}:
        raise ValueError("run_interactive_keyboard_safe.py only supports record mode.")

    output_root = resolve_output_root(cfg.dataset.root)
    cfg = replace(
        cfg,
        mode="record",
        dataset=replace(cfg.dataset, root=str(output_root)),
    )

    env = None
    dataset = None
    exit_summary: dict[str, Any] | None = None
    aborted = False

    try:
        env = make_safe_keyboard_env(cfg)
        env_processor, action_processor = make_processors(env, None, cfg.env, cfg.device)
        print_banner(cfg, env, env_processor, action_processor)

        dt = 1.0 / cfg.env.fps
        use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True

        obs, info = env.reset()
        complementary_data = (
            {"raw_joint_positions": info.pop("raw_joint_positions")}
            if "raw_joint_positions" in info
            else {}
        )
        env_processor.reset()
        action_processor.reset()

        transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
        transition = env_processor(transition)
        dataset = create_safe_dataset(cfg, transition)

        episode_idx = 0
        episode_step = 0
        saved_episodes: list[dict[str, Any]] = []

        while episode_idx < cfg.dataset.num_episodes_to_record:
            step_start_time = time.perf_counter()

            neutral_action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
            if use_gripper:
                neutral_action = torch.cat([neutral_action, torch.tensor([0.0], dtype=torch.float32)])

            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=neutral_action,
                env_processor=env_processor,
                action_processor=action_processor,
            )
            terminated = bool(transition.get(TransitionKey.DONE, False))
            truncated = bool(transition.get(TransitionKey.TRUNCATED, False))
            if transition[TransitionKey.INFO].get("exit_requested", False):
                print("ESC received, aborting current run without saving the partial episode.")
                aborted = True
                if dataset.episode_buffer["size"] > 0:
                    dataset.clear_episode_buffer(delete_images=True)
                break

            dataset.add_frame(build_frame(transition, cfg))
            episode_step += 1

            if terminated or truncated:
                if transition[TransitionKey.INFO].get(TeleopEvents.RERECORD_EPISODE, False):
                    print(f"Re-recording episode {episode_idx}")
                    dataset.clear_episode_buffer()
                else:
                    dataset.save_episode(parallel_encoding=False)
                    saved_episodes.append(
                        {
                            "episode_index": episode_idx,
                            "steps": episode_step,
                            "reward": float(transition[TransitionKey.REWARD]),
                            "success": bool(
                                transition[TransitionKey.INFO].get(TeleopEvents.SUCCESS, False)
                            ),
                        }
                    )
                    episode_idx += 1

                episode_step = 0
                if episode_idx < cfg.dataset.num_episodes_to_record:
                    obs, info = env.reset()
                    env_processor.reset()
                    action_processor.reset()
                    transition = create_transition(observation=obs, info=info)
                    transition = env_processor(transition)

            precise_sleep(max(dt - (time.perf_counter() - step_start_time), 0.0))

        if dataset.episode_buffer["size"] > 0:
            # Defensive: avoid leaving an in-memory partial episode around.
            dataset.clear_episode_buffer(delete_images=True)

        dataset.finalize()
        exit_summary = {
            "output_root": str(output_root),
            "repo_id": cfg.dataset.repo_id,
            "num_episodes": 0,
            "num_frames": 0,
            "saved_episodes": saved_episodes,
            "aborted": aborted,
        }

        if saved_episodes:
            reloaded = LeRobotDataset(cfg.dataset.repo_id, root=output_root)
            exit_summary["num_episodes"] = reloaded.num_episodes
            exit_summary["num_frames"] = reloaded.num_frames
        else:
            # No complete episode means no usable dataset metadata such as tasks.parquet.
            # Clean up the incomplete output directory and report it clearly instead of
            # trying to reload and accidentally falling back to the Hub.
            if output_root.exists():
                shutil.rmtree(output_root, ignore_errors=True)
            exit_summary["output_root_removed"] = True
            if aborted:
                exit_summary["message"] = "Run aborted by ESC before any episode was saved."
            else:
                exit_summary["message"] = "Run ended without any completed episode being saved."

        summary_path = output_root.parent / "interactive_keyboard_safe_summary.json"
        with open(summary_path, "w") as f:
            json.dump(exit_summary, f, indent=2)
        print(json.dumps(exit_summary, indent=2))
        return exit_summary
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, finalizing safely...")
        if dataset is not None and dataset.episode_buffer["size"] > 0:
            with contextlib.suppress(Exception):
                dataset.clear_episode_buffer(delete_images=True)
        raise
    finally:
        if dataset is not None:
            with contextlib.suppress(Exception):
                dataset.finalize()


@parser.wrap()
def main(cfg: GymManipulatorConfig) -> None:
    try:
        run_safe_record(cfg)
    except KeyboardInterrupt:
        hard_exit(130)
    except Exception:
        traceback.print_exc()
        hard_exit(1)

    # Intentionally avoid `env.close()` / Python object destructors because the
    # MuJoCo passive viewer can segfault during shutdown in XRDP sessions.
    hard_exit(0)


if __name__ == "__main__":
    main()
