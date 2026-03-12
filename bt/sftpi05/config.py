"""
Configuration for PI05 fine-tuning.

Provides a dataclass-based config that maps to PI05Config and TrainPipelineConfig.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PI05SFTConfig:
    # Dataset
    dataset_repo_id: str = "lerobot/aloha_sim_insertion_scripted"
    dataset_root: str | None = None
    dataset_episodes: list[int] | None = None

    # Policy
    pretrained_path: str | None = "lerobot/pi05_base"
    policy_type: str = "pi05"
    dtype: str = "bfloat16"

    # Dimensions (auto-detected from dataset if not set)
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Training
    output_dir: str = "outputs/sft_pi05"
    job_name: str = "sft_pi05"
    batch_size: int = 2
    steps: int = 100
    lr: float = 2.5e-5
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    warmup_steps: int = 10
    decay_steps: int = 100
    decay_lr: float = 2.5e-6
    seed: int = 42
    num_workers: int = 0

    # Logging & saving
    log_freq: int = 10
    save_freq: int = 50
    save_checkpoint: bool = True

    # Model options
    gradient_checkpointing: bool = True
    compile_model: bool = False
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    # Normalization: QUANTILES (need q01/q99) or MEAN_STD
    normalization_mode: str = "MEAN_STD"

    # wandb
    wandb_enable: bool = False
    wandb_project: str = "sft_pi05"

    # Device
    device: str | None = None

    def auto_detect_device(self) -> str:
        if self.device:
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
