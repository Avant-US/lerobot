"""StrGroot Policy — adapts StarVLA's Qwen_GR00T as a LeRobot PreTrainedPolicy."""

from __future__ import annotations

import logging
import os
import sys
import types
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import to_pil_image

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.str_groot.configuration_str_groot import StrGrootConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="StrGrootPolicy")

# ---------------------------------------------------------------------------
# Ensure StarVLA package is importable from the project root
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parents[4]  # .../src/lerobot/policies/str_groot -> project root
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# StarVLA's QwenGR00T.py does `from deployment.model_server.tools.image_tools
# import to_pil_preserve`.  That module lives inside the starVLA repo checkout
# which may not be on the path.  Provide a lightweight mock so the import
# succeeds without the full deployment package.
if "deployment" not in sys.modules:
    _d = types.ModuleType("deployment")
    _ms = types.ModuleType("deployment.model_server")
    _tl = types.ModuleType("deployment.model_server.tools")
    _it = types.ModuleType("deployment.model_server.tools.image_tools")

    def _to_pil_preserve(img):
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, np.ndarray):
            return Image.fromarray(img.astype(np.uint8))
        if isinstance(img, list):
            return [_to_pil_preserve(i) for i in img]
        return img

    _it.to_pil_preserve = _to_pil_preserve
    _tl.image_tools = _it
    _ms.tools = _tl
    _d.model_server = _ms
    for _name, _mod in [
        ("deployment", _d),
        ("deployment.model_server", _ms),
        ("deployment.model_server.tools", _tl),
        ("deployment.model_server.tools.image_tools", _it),
    ]:
        sys.modules[_name] = _mod

# Some StarVLA modules import `qwen_vl_utils` even when not using it in
# runtime paths required by StrGroot. Provide a small compatibility shim.
if "qwen_vl_utils" not in sys.modules:
    _qwen_vl_utils = types.ModuleType("qwen_vl_utils")

    def _process_vision_info(*args, **kwargs):
        return None, None

    _qwen_vl_utils.process_vision_info = _process_vision_info
    sys.modules["qwen_vl_utils"] = _qwen_vl_utils


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

        self._set_trainable_parameters()

        self.reset()

    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: StrGrootConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """Load from a LeRobot checkpoint without re-loading StarVLA base checkpoint."""
        if config is None:
            from lerobot.configs.policies import PreTrainedConfig

            loaded_config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
            if not isinstance(loaded_config, StrGrootConfig):
                raise TypeError(
                    f"Expected StrGrootConfig, but got {type(loaded_config).__name__} from pretrained config."
                )
            config = loaded_config

        # `model.safetensors` already contains the model parameters, so avoid
        # an extra (and potentially very large) StarVLA checkpoint load first.
        config = deepcopy(config)
        if config.starvla_checkpoint:
            logger.info(
                "Loading StrGroot from pretrained checkpoint: skipping `starvla_checkpoint=%s` bootstrap load.",
                config.starvla_checkpoint,
            )
            config.starvla_checkpoint = None

        return super().from_pretrained(
            pretrained_name_or_path=pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs,
        )

    def _set_trainable_parameters(self) -> None:
        vlm_trainable = bool(self.config.tune_vlm)
        action_trainable = bool(self.config.tune_action_head)

        for p in self._starvla_model.qwen_vl_interface.parameters():
            p.requires_grad = vlm_trainable

        for p in self._starvla_model.action_model.parameters():
            p.requires_grad = action_trainable

        n_vlm_trainable = sum(p.numel() for p in self._starvla_model.qwen_vl_interface.parameters() if p.requires_grad)
        n_action_trainable = sum(
            p.numel() for p in self._starvla_model.action_model.parameters() if p.requires_grad
        )
        logger.info(
            "StrGroot trainable params: vlm=%s action_head=%s",
            f"{n_vlm_trainable:,}",
            f"{n_action_trainable:,}",
        )

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
        vlm_params: list[Tensor] = []
        action_params: list[Tensor] = []
        other_params: list[Tensor] = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("_starvla_model.qwen_vl_interface"):
                vlm_params.append(param)
            elif name.startswith("_starvla_model.action_model"):
                action_params.append(param)
            else:
                other_params.append(param)

        if not vlm_params and not action_params and not other_params:
            raise ValueError("No trainable parameters found for StrGroot policy.")

        param_groups: list[dict[str, object]] = []
        if action_params:
            param_groups.append(
                {
                    "params": action_params,
                    "lr": float(self.config.optimizer_lr_action_head),
                }
            )
        if vlm_params:
            param_groups.append(
                {
                    "params": vlm_params,
                    "lr": float(self.config.optimizer_lr_vlm),
                }
            )
        if other_params:
            # Treat uncategorized trainable modules as action-head side modules.
            param_groups.append(
                {
                    "params": other_params,
                    "lr": float(self.config.optimizer_lr_action_head),
                }
            )

        return param_groups

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        examples = self._batch_to_examples(batch, inference=False)
        if reduction == "none":
            # StarVLA currently returns a reduced scalar loss. For RA-BC we need
            # per-sample loss, so we compute one forward pass per sample.
            per_sample_losses: list[Tensor] = []
            for ex in examples:
                outputs = self._starvla_model.forward([ex])
                loss = outputs["action_loss"]
                if not torch.is_tensor(loss):
                    loss = torch.as_tensor(loss, device=next(self.parameters()).device, dtype=torch.float32)
                per_sample_losses.append(loss.reshape(()))
            per_sample_loss = torch.stack(per_sample_losses, dim=0)
            return per_sample_loss, {"loss": per_sample_loss.mean().item()}

        if reduction != "mean":
            raise ValueError(f"Unsupported reduction '{reduction}'. Expected 'mean' or 'none'.")

        outputs = self._starvla_model.forward(examples)
        loss = outputs["action_loss"]
        return loss, {"loss": float(loss.item())}

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
