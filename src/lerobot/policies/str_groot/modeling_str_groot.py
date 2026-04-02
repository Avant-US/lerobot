"""StrGroot Policy — adapts StarVLA's Qwen_GR00T as a LeRobot PreTrainedPolicy."""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import TypeVar

import torch
from torch import Tensor
from torchvision.transforms.functional import to_pil_image

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.str_groot.configuration_str_groot import StrGrootConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="StrGrootPolicy")


class StrGrootPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper around StarVLA's Qwen_GR00T."""

    name = "str_groot"
    config_class = StrGrootConfig

    def __init__(self, config: StrGrootConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        starvla_cfg = self._build_starvla_config()

        from starVLA.model.framework.QwenGR00T import Qwen_GR00T

        self._starvla_model = Qwen_GR00T(starvla_cfg)

        if config.starvla_checkpoint:
            self._load_starvla_checkpoint(config.starvla_checkpoint)

        if config.freeze_vlm:
            for p in self._starvla_model.qwen_vl_interface.parameters():
                p.requires_grad = False

        self.reset()

    # ------------------------------------------------------------------
    # StarVLA OmegaConf config builder
    # ------------------------------------------------------------------
    def _build_starvla_config(self):
        from omegaconf import OmegaConf

        c = self.config
        cfg_dict = {
            "framework": {
                "name": "QwenGR00T",
                "qwenvl": {
                    "base_vlm": c.base_vlm,
                    "attn_implementation": c.attn_implementation,
                },
                "action_model": {
                    "action_model_type": c.action_model_type,
                    "action_hidden_dim": c.action_hidden_dim,
                    "hidden_size": c.action_hidden_dim,
                    "add_pos_embed": True,
                    "max_seq_len": c.max_seq_len,
                    "action_dim": c.action_dim,
                    "state_dim": c.state_dim,
                    "future_action_window_size": c.future_action_window_size,
                    "action_horizon": c.future_action_window_size + 1,
                    "past_action_window_size": c.past_action_window_size,
                    "repeated_diffusion_steps": c.repeated_diffusion_steps,
                    "noise_beta_alpha": c.noise_beta_alpha,
                    "noise_beta_beta": c.noise_beta_beta,
                    "noise_s": c.noise_s,
                    "num_timestep_buckets": c.num_timestep_buckets,
                    "num_inference_timesteps": c.num_inference_timesteps,
                    "num_target_vision_tokens": c.num_target_vision_tokens,
                    "diffusion_model_cfg": {
                        "cross_attention_dim": c.cross_attention_dim,  # overridden by Qwen_GR00T.__init__
                        "dropout": c.dit_dropout,
                        "final_dropout": True,
                        "interleave_self_attention": True,
                        "norm_type": "ada_norm",
                        "num_layers": c.dit_num_layers,
                        "output_dim": c.action_hidden_dim,
                        "positional_embeddings": None,
                    },
                },
                "reduce_in_full_precision": True,
            },
            "trainer": {
                "repeated_diffusion_steps": c.repeated_diffusion_steps,
            },
            "datasets": {
                "vla_data": {
                    "image_size": list(c.image_size),
                },
            },
        }
        return OmegaConf.create(cfg_dict)

    # ------------------------------------------------------------------
    # StarVLA checkpoint loading
    # ------------------------------------------------------------------
    @staticmethod
    def _find_ckpt_in_dir(d: str):
        """Return the first checkpoint file path inside *d* (recursive), or ``None``."""
        _EXTS = (".pt", ".safetensors", ".bin")
        candidates = ["model.pt", "model.safetensors", "pytorch_model.bin"]
        for fname in candidates:
            p = os.path.join(d, fname)
            if os.path.exists(p):
                return p
        for root, _dirs, files in os.walk(d):
            matched = [f for f in files if f.endswith(_EXTS)]
            if matched:
                return os.path.join(root, matched[0])
        return None

    def _load_starvla_checkpoint(self, checkpoint_path: str) -> None:
        local_path = None

        if os.path.isfile(checkpoint_path):
            local_path = checkpoint_path
        elif os.path.isdir(checkpoint_path):
            local_path = self._find_ckpt_in_dir(checkpoint_path)

        if local_path is None and "/" in checkpoint_path:
            from huggingface_hub import snapshot_download

            logger.info("Downloading StarVLA checkpoint from HF: %s", checkpoint_path)
            local_dir = snapshot_download(repo_id=checkpoint_path)
            local_path = self._find_ckpt_in_dir(local_dir)

        if local_path is None:
            raise FileNotFoundError(
                f"No checkpoint file (.pt / .safetensors / .bin) found for: {checkpoint_path}"
            )

        logger.info("Loading StarVLA checkpoint: %s", local_path)

        if local_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(local_path)
        else:
            state_dict = torch.load(local_path, map_location="cpu", weights_only=False)

        prefixed = {f"_starvla_model.{k}": v for k, v in state_dict.items()}
        missing, unexpected = self.load_state_dict(prefixed, strict=False)
        if missing:
            logger.warning("Missing keys when loading StarVLA checkpoint: %d keys", len(missing))
        if unexpected:
            logger.warning("Unexpected keys when loading StarVLA checkpoint: %d keys", len(unexpected))

    # ------------------------------------------------------------------
    # Action queue (inference)
    # ------------------------------------------------------------------
    def reset(self):
        self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        return self.parameters()

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        examples = self._batch_to_examples(batch, inference=False)
        outputs = self._starvla_model.forward(examples)
        loss = outputs["action_loss"]
        return loss, {"loss": loss.item()}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        examples = self._batch_to_examples(batch, inference=True)
        device = next(self.parameters()).device

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16
        ):
            outputs = self._starvla_model.predict_action(examples)

        actions = torch.from_numpy(outputs["normalized_actions"]).to(
            device=device, dtype=torch.float32
        )
        actions = actions[:, :, : self.config.action_dim]
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    # ------------------------------------------------------------------
    # Batch ↔ StarVLA example conversion
    # ------------------------------------------------------------------
    def _batch_to_examples(
        self,
        batch: dict[str, Tensor],
        inference: bool = False,
    ) -> list[dict]:
        image_keys = sorted(k for k in batch if k.startswith("observation.images."))
        if not image_keys:
            raise ValueError(
                f"No image keys found in batch. Available keys: {sorted(batch.keys())}"
            )

        B = batch[image_keys[0]].shape[0]
        examples: list[dict] = []

        for i in range(B):
            images = []
            for key in image_keys:
                img_t = batch[key][i]  # (C, H, W)
                if img_t.is_floating_point():
                    img_t = (img_t.clamp(0, 1) * 255).to(torch.uint8)
                images.append(to_pil_image(img_t.cpu()))

            example: dict = {
                "image": images,
                "lang": batch["task"][i] if "task" in batch else "",
            }

            if not inference and "action" in batch:
                example["action"] = batch["action"][i].cpu().float().numpy()

            if "observation.state" in batch:
                state = batch["observation.state"][i].cpu().float().numpy()
                if self.config.state_indices is not None:
                    state = state[..., list(self.config.state_indices)]
                example["state"] = state.reshape(1, -1)

            examples.append(example)

        return examples
