"""Configuration for StrGroot policy — StarVLA Qwen-GR00T adapted for LeRobot."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("str_groot")
@dataclass
class StrGrootConfig(PreTrainedConfig):
    """Wraps StarVLA's Qwen-GR00T (QwenVL + FlowMatching action head) as a LeRobot policy."""

    # --- LeRobot policy basics ---
    n_obs_steps: int = 1
    chunk_size: int = 8
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    image_size: tuple[int, int] = (224, 224)

    # --- VLM backbone ---
    base_vlm: str = "Qwen/Qwen3-VL-4B-Instruct"
    attn_implementation: str = "flash_attention_2"

    # --- Action model (DiT + FlowMatching) ---
    action_model_type: str = "DiT-B"
    action_dim: int = 7
    state_dim: int = 7
    action_hidden_dim: int = 1024
    future_action_window_size: int = 7
    past_action_window_size: int = 0
    max_seq_len: int = 1024
    num_target_vision_tokens: int = 32

    # When the dataset state dim differs from the checkpoint's state_dim,
    # specify which indices to keep.  E.g. for libero (8-dim) with a 7-dim
    # checkpoint, use (0,1,2,3,4,5,7) to drop the pad dim at index 6.
    state_indices: tuple[int, ...] | None = None

    dit_num_layers: int = 16
    dit_dropout: float = 0.2

    repeated_diffusion_steps: int = 8
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    cross_attention_dim: int = 2048

    # --- Training flags ---
    freeze_vlm: bool = False
    tune_vlm: bool | None = None
    tune_action_head: bool = True
    use_bf16: bool = True

    # --- StarVLA pretrained checkpoint (local path or HF repo id) ---
    starvla_checkpoint: str | None = None

    # --- Optimizer ---
    optimizer_lr: float = 1e-4
    optimizer_lr_action_head: float | None = None
    optimizer_lr_vlm: float | None = None
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    warmup_ratio: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        self.chunk_size = self.future_action_window_size + 1
        self.n_action_steps = self.chunk_size

        # Keep backward compatibility with old `freeze_vlm` while exposing
        # explicit fine-tuning controls used by `get_optim_params`.
        if self.tune_vlm is None:
            self.tune_vlm = not self.freeze_vlm
        self.freeze_vlm = not self.tune_vlm

        if self.optimizer_lr_action_head is None:
            self.optimizer_lr_action_head = self.optimizer_lr
        if self.optimizer_lr_vlm is None:
            self.optimizer_lr_vlm = self.optimizer_lr * 0.1

        if not self.tune_vlm and not self.tune_action_head:
            raise ValueError("Both tune_vlm and tune_action_head are False. No trainable parameters left.")

    def validate_features(self) -> None:
        image_features = [k for k, f in self.input_features.items() if f.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "StrGroot policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if OBS_STATE in self.input_features:
            actual_dim = self.input_features[OBS_STATE].shape[0]
            if self.state_indices is not None:
                if len(self.state_indices) != self.state_dim:
                    raise ValueError(
                        f"len(state_indices)={len(self.state_indices)} must equal "
                        f"state_dim={self.state_dim}"
                    )
                if max(self.state_indices) >= actual_dim:
                    raise ValueError(
                        f"state_indices contains index {max(self.state_indices)} "
                        f"but dataset state has only {actual_dim} dims"
                    )
            else:
                if actual_dim != self.state_dim:
                    self.state_dim = actual_dim
        else:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.state_dim,),
            )

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.action_dim,),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=int(10000 * self.warmup_ratio),
            num_decay_steps=10000,
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * 0.1,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
