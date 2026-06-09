# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import builtins
import json
import logging
import shutil
from collections import deque
from pathlib import Path
from typing import Any

import packaging
import safetensors
import torch
import transformers
from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_INDEX_FILE, SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import load_model as load_model_as_safetensor
from torch import Tensor
from transformers import Qwen2Tokenizer

from dexbotic.model.dm0.dm0_arch import DM0Config as DexboticDM0ArchConfig
from dexbotic.model.dm0.dm0_arch import DM0ForCausalLM
from dexbotic.tokenization.process import DM0Tokenization

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.dm0.configuration_dm0 import DM0Config
from lerobot.policies.dm0.dm0_transforms import (
    compute_absolute,
    compute_delta,
    load_norm_stats,
    pad_to_dim,
    quantile_denormalize,
    quantile_normalize,
)
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)


def _patch_pe_self_attention_for_lora() -> None:
    """Make dexbotic PE ViT's ``SelfAttention.forward`` go through ``self.out_proj(x)``.

    Upstream dexbotic ``pe_model.SelfAttention.forward`` ends with::

        return F.linear(attn, self.out_proj.weight, self.out_proj.bias)

    which pulls the bare ``weight`` / ``bias`` parameters and sidesteps the module's
    ``__call__``. After PEFT wraps ``out_proj`` with a ``LoraLinear``, the LoRA branch
    (``base + lora_B(lora_A(x)) * scaling``) lives inside that module's ``forward``;
    the ``F.linear(..., m.weight, m.bias)`` form silently bypasses the LoRA contribution
    AND prevents autograd from registering ``lora_A`` / ``lora_B`` as forward dependencies,
    so their ``.grad`` stays ``None`` (manifesting as ``train/grad_norm == 0`` and a flat
    loss curve).

    Switching the last line to ``self.out_proj(attn)`` is mathematically identical when
    ``out_proj`` is a plain ``nn.Linear`` (``Linear.forward`` just calls
    ``F.linear(x, self.weight, self.bias)``), and routes correctly through PEFT when it
    isn't. Hence safe to apply unconditionally — phase-1 training, dexbotic-native
    inference, and any non-LoRA code path are all numerically unchanged.

    Applied at module-import time so any caller that touches this file (training,
    eval, third-party scripts) gets the fix automatically. Idempotent.
    """
    import torch.nn.functional as F  # noqa: WPS433 (deliberate local import; keeps module import cheap)
    from einops import rearrange  # noqa: WPS433
    from dexbotic.model.modules.mm_vision.pe import pe_model

    if getattr(pe_model.SelfAttention.forward, "_lora_patched", False):
        return

    def forward(self, x, grid_hw):
        _, _, embed_dim = x.shape
        proj = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        proj = (
            proj.unflatten(-1, (3, embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        q, k, v = proj[0], proj[1], proj[2]
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)
        q, k = self.rope(q, k, grid_hw=grid_hw)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale
        )
        attn = rearrange(attn, "b h s d -> b s (h d)")
        return self.out_proj(attn)

    forward._lora_patched = True  # type: ignore[attr-defined]
    pe_model.SelfAttention.forward = forward


def _patch_pe_transformer_grad_ckpt() -> None:
    """Wrap each PE ``ResidualAttentionBlock`` in ``torch.utils.checkpoint.checkpoint``.

    HF's ``PreTrainedModel.gradient_checkpointing_enable()`` (called on ``dm0_model`` in
    :class:`DM0Policy.__init__`) only propagates to submodules that inherit from
    ``PreTrainedModel`` and implement the corresponding hook. The dexbotic PE
    ``Transformer`` is a plain ``nn.Module`` with no built-in gradient-checkpointing
    support and no ``checkpoint(...)`` calls in its forward, so before this patch the
    ViT was never checkpointed.

    With phase-1 / fully-frozen ViT this was harmless (autograd sees no trainable params
    inside the ViT and skips the activation save entirely). With phase-2 LoRA — once
    :func:`_patch_pe_self_attention_for_lora` makes the LoRA branch actually participate
    in forward — every ViT layer's activations have to be retained for backward, which
    blows up memory by ~10× (e.g. ``num_images=3`` × ``728²/14² = 8112`` tokens per
    sample × 24 layers × ``4×width`` MLP intermediates). On 140 GB H200 this OOMs at the
    LoRA forward inside the very first batch.

    This patch wraps each resblock in non-reentrant checkpoint so block-internal
    activations are recomputed during backward instead of stored. ``use_reentrant=False``
    is required because the resblock's input typically has ``requires_grad=False`` (image
    features come from a non-leaf tensor that doesn't require grad in the outer graph)
    while the *inner* LoRA params do require grad — the older reentrant checkpointing
    sees no input requiring grad and prunes the whole subgraph from backward, which would
    nullify LoRA gradients again.

    Only enabled when ``self.training`` and grad mode is on, so ``select_action`` /
    ``inference_action`` (which run under ``torch.no_grad()``) don't pay the recompute
    cost. Trade-off vs. baseline: forward goes through the resblocks twice per step (once
    cached inputs + once recomputed during backward), so wallclock per step grows by
    roughly ``+ViT_forward_time / total_step_time`` (typically 20-30% in this setup).

    Idempotent. Applied at module-import time alongside the LoRA-targeting patch.
    """
    from torch.utils.checkpoint import checkpoint  # noqa: WPS433
    from dexbotic.model.modules.mm_vision.pe import pe_model

    if getattr(pe_model.Transformer.forward, "_grad_ckpt_patched", False):
        return

    def forward(self, x, grid_hw, layer_idx: int = -1):
        stop_idx = (self.layers + layer_idx) % self.layers
        # Only checkpoint when we actually need backward; saves the recompute pass on
        # eval / inference where ``select_action`` runs under ``torch.no_grad()``.
        use_ckpt = self.training and torch.is_grad_enabled()
        for i, r in enumerate(self.resblocks):
            if use_ckpt:
                x = checkpoint(r, x, grid_hw, use_reentrant=False)
            else:
                x = r(x, grid_hw=grid_hw)
            if i == stop_idx:
                break
        return x

    forward._grad_ckpt_patched = True  # type: ignore[attr-defined]
    pe_model.Transformer.forward = forward


_patch_pe_self_attention_for_lora()
_patch_pe_transformer_grad_ckpt()


DM0_ARCH_CONFIG_FILENAME = "dm0_arch_config.json"
_DEXBOTIC_DM0_MODEL_TYPE = "dexbotic_dm0"


def _is_arch_config_only_dir(root: Path) -> bool:
    """A LeRobot-saved DM0 checkpoint dir bundles ``dm0_arch_config.json`` to describe the dexbotic
    architecture without the original base weights. When this file is present we should *only* build
    the architecture; the policy-level safetensors will fill in the trained weights afterwards.
    """
    return root.is_dir() and (root / DM0_ARCH_CONFIG_FILENAME).is_file()


def _local_dm0_safetensors_present(root: Path) -> bool:
    """Single-file or sharded HF-style safetensors next to ``config.json``."""
    if not root.is_dir():
        return False
    if (root / SAFETENSORS_SINGLE_FILE).is_file():
        return True
    index = root / SAFETENSORS_INDEX_FILE
    if not index.is_file():
        return False
    try:
        with index.open() as f:
            weight_map = json.load(f).get("weight_map") or {}
        shard_names = set(weight_map.values())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(shard_names) and all((root / name).is_file() for name in shard_names)


def _is_dexbotic_native_dir(root: Path) -> bool:
    """Detects a raw dexbotic ``DM0ForCausalLM`` dump (e.g. the published ``DM0-base`` checkpoint).

    These dirs ship ``config.json`` with ``model_type == 'dexbotic_dm0'`` plus a ``model.safetensors``
    (or sharded safetensors + index) whose state-dict keys live directly on the architecture (no
    ``dm0_model.`` prefix). They must not be re-fed to the LeRobot-policy safetensors loader, which would
    report every key as missing/unexpected.
    """
    cfg = root / "config.json"
    if not (root.is_dir() and cfg.is_file() and _local_dm0_safetensors_present(root)):
        return False
    if (root / DM0_ARCH_CONFIG_FILENAME).is_file():
        return False
    try:
        with cfg.open() as f:
            return json.load(f).get("model_type") == _DEXBOTIC_DM0_MODEL_TYPE
    except (OSError, json.JSONDecodeError):
        return False


def _load_dm0_for_causal_lm(model_name_or_path: str, device: str | torch.device) -> DM0ForCausalLM:
    """Load dexbotic ``DM0ForCausalLM`` from a local folder.

    Three modes:

    1. ``dm0_arch_config.json`` present: build the dexbotic architecture from this file with random
       init and skip weight loading. Used when the caller is a LeRobot policy ``from_pretrained``
       (the trained weights live in the policy-level ``model.safetensors`` and will be loaded on
       top by the parent ``_load_as_safetensor``).
    2. dexbotic-native dump (``config.json`` + ``model.safetensors`` *or* sharded
       ``model.safetensors.index.json`` + ``model-*-of-*.safetensors``, without ``dm0_arch_config.json``):
       build the architecture from ``config.json`` and load weights via safetensors.
    3. Fallback: defer to ``DM0ForCausalLM.from_pretrained`` (e.g. HF Hub IDs).

    HuggingFace ``from_pretrained`` may call ``DM0Config()`` with no args while diffing defaults,
    which raises ``Missing required field - 'llm_config'`` for dexbotic configs. The first two
    branches avoid that path entirely.
    """
    root = Path(model_name_or_path)

    if _is_arch_config_only_dir(root):
        with (root / DM0_ARCH_CONFIG_FILENAME).open() as f:
            arch_cfg = DexboticDM0ArchConfig(**json.load(f))
        model = DM0ForCausalLM(arch_cfg)
        model.to(device)
        return model

    weights = root / SAFETENSORS_SINGLE_FILE
    index_weights = root / SAFETENSORS_INDEX_FILE
    cfg_path = root / "config.json"
    if root.is_dir() and cfg_path.is_file() and (weights.is_file() or index_weights.is_file()):
        with cfg_path.open() as f:
            arch_cfg = DexboticDM0ArchConfig(**json.load(f))
        model = DM0ForCausalLM(arch_cfg)
        # Load tensors on CPU first to avoid CUDA OOM when the GPU is already occupied, then move the model.
        load_device = "cpu"
        _load_dm0_safetensors_into_model(model, root, load_device)
        if packaging.version.parse(safetensors.__version__) < packaging.version.parse("0.4.3"):
            model.to(load_device)
        model.to(device)
        return model
    return DM0ForCausalLM.from_pretrained(model_name_or_path)


def _load_dm0_safetensors_into_model(model: DM0ForCausalLM, root: Path, load_device: str) -> None:
    """Load dexbotic checkpoint tensors from a single ``model.safetensors`` or HF-style sharded files."""
    weights = root / SAFETENSORS_SINGLE_FILE
    load_kwargs: dict[str, Any] = {"strict": False}
    if packaging.version.parse(safetensors.__version__) >= packaging.version.parse("0.4.3"):
        load_kwargs["device"] = load_device

    if weights.is_file():
        missing, unexpected = load_model_as_safetensor(model, str(weights), **load_kwargs)
    else:
        index_path = root / SAFETENSORS_INDEX_FILE
        if not index_path.is_file():
            raise FileNotFoundError(
                f"Expected {SAFETENSORS_SINGLE_FILE} or {SAFETENSORS_INDEX_FILE} under {root}"
            )
        with index_path.open() as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty weight_map in {index_path}")
        shard_names = sorted(set(weight_map.values()))
        file_kw: dict[str, Any] = {}
        if packaging.version.parse(safetensors.__version__) >= packaging.version.parse("0.4.3"):
            file_kw["device"] = load_device
        unexpected_acc: list[str] = []
        for sname in shard_names:
            shard_path = root / sname
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"DM0 safetensors shard listed in index but not found: {shard_path}"
                )
            shard_sd = load_safetensors_file(str(shard_path), **file_kw)
            inc = model.load_state_dict(shard_sd, strict=False)
            unexpected_acc.extend(inc.unexpected_keys)
        unexpected = list(dict.fromkeys(unexpected_acc))
        index_keys = set(weight_map.keys())
        missing = sorted(set(model.state_dict().keys()) - index_keys)

    if unexpected:
        logger.warning(
            "DM0 dexbotic weight load: %d unexpected key(s), e.g. %s",
            len(unexpected),
            unexpected[:5],
        )
    if missing:
        suspicious = [k for k in missing if not k.endswith((".inv_freq", ".rotary_emb.inv_freq"))]
        if suspicious:
            logger.warning(
                "DM0 dexbotic weight load: %d missing key(s), e.g. %s",
                len(suspicious),
                suspicious[:5],
            )


class DM0Policy(PreTrainedPolicy):
    config_class = DM0Config
    name = "dm0"

    def __init__(
        self,
        config: DM0Config,
        dataset_stats: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.dm0_model = _load_dm0_for_causal_lm(config.model_name_or_path, config.device)

        tok_path = config.tokenizer_name_or_path or config.model_name_or_path
        # Avoid ``AutoTokenizer`` here: it may load the folder's ``config.json`` as ``AutoConfig`` and
        # trigger dexbotic ``DM0Config`` default-init bugs. DM0 checkpoints ship ``Qwen2Tokenizer``.
        #
        # ``fix_mistral_regex``: transformers>=5.0 inspects local ``config.json`` and, when it lacks a
        # ``transformers_version`` field (true for both DM0-base ``model_type=dexbotic_dm0`` and our
        # saved policy ``type=dm0``), heuristically classifies the tokenizer as Mistral and warns
        # unless the flag is passed explicitly. The fix it offers replaces the ``pre_tokenizer`` with
        # a Mistral-shaped Split regex, which would silently change Qwen2 tokenization. Pass ``False``
        # so transformers takes neither the warn branch nor the patch branch -- pre-tokenizer stays
        # the original Qwen2 one. ``False`` is only accepted on transformers>=5.0; older versions
        # don't have this code path at all.
        tok_kw: dict[str, Any] = {}
        if packaging.version.parse(transformers.__version__) >= packaging.version.parse("5.0.0"):
            tok_kw["fix_mistral_regex"] = False
        self.tokenizer = Qwen2Tokenizer.from_pretrained(str(tok_path), **tok_kw)
        # Cap padded length to match dexbotic training (DM0Tokenization pads to
        # ``tokenizer.model_max_length``); without this Qwen2's default 32k+ blows up attention memory.
        if config.tokenizer_max_length is not None:
            self.tokenizer.model_max_length = int(config.tokenizer_max_length)
        self.tokenization_func = DM0Tokenization(self.tokenizer, chat_template="step")

        dev = torch.device(config.device)
        if config.norm_stats_path:
            stats = load_norm_stats(
                config.norm_stats_path,
                dev,
                config.max_state_dim,
                config.max_action_dim,
            )
            self.register_buffer("_action_min", stats["action_min"], persistent=False)
            self.register_buffer("_action_max", stats["action_max"], persistent=False)
            self.register_buffer("_state_min", stats["state_min"], persistent=False)
            self.register_buffer("_state_max", stats["state_max"], persistent=False)
        else:
            logger.warning("DM0Config.norm_stats_path is None: skipping quantile stats (forward may be wrong).")
            z = torch.zeros(config.max_action_dim, device=dev)
            o = torch.ones(config.max_action_dim, device=dev)
            self.register_buffer("_action_min", z - 1.0, persistent=False)
            self.register_buffer("_action_max", o, persistent=False)
            self.register_buffer("_state_min", z - 1.0, persistent=False)
            self.register_buffer("_state_max", o, persistent=False)

        if config.original_action_dim is None:
            ft = config.output_features.get(ACTION)
            if ft is not None and len(ft.shape) > 0:
                self._original_action_dim = int(ft.shape[-1])
            else:
                self._original_action_dim = config.max_action_dim
        else:
            self._original_action_dim = int(config.original_action_dim)

        self._action_queue: deque[Tensor] = deque(maxlen=config.n_action_steps)

        if config.gradient_checkpointing and hasattr(self.dm0_model, "gradient_checkpointing_enable"):
            # use_reentrant=False is required for the flag to actually propagate to LLM layers
            # under modern transformers (probe showed 0 modules flagged without it, 58 with).
            # `enable_input_require_grads` makes the input embedding output require grad so that
            # checkpointed recompute can backprop through it (HF Trainer does this automatically;
            # accelerate does not).
            self.dm0_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            if hasattr(self.dm0_model, "enable_input_require_grads"):
                self.dm0_model.enable_input_require_grads()

        if config.freeze_vision_encoder:
            for p in self.dm0_model.model.mm_vision_tower.parameters():
                p.requires_grad = False
        if config.train_expert_only:
            for p in self.dm0_model.model.llm.parameters():
                p.requires_grad = False
            for p in self.dm0_model.model.mm_vision_tower.parameters():
                p.requires_grad = False
            for p in self.dm0_model.model.mm_projector.parameters():
                p.requires_grad = False

        # EMA 状态（延迟初始化，在第一次 update() 时填充）
        self._ema_params: dict[str, torch.Tensor] | None = None
        self._ema_active: bool = False

        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> T:
        model_id = str(pretrained_name_or_path)
        cfg_keys = (
            "force_download",
            "resume_download",
            "proxies",
            "token",
            "cache_dir",
            "local_files_only",
            "revision",
        )
        cfg_kwargs = {k: v for k, v in kwargs.items() if k in cfg_keys}
        if "cli_overrides" in kwargs:
            cfg_kwargs["cli_overrides"] = kwargs["cli_overrides"]

        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                **cfg_kwargs,
            )
        ckpt = Path(model_id)
        skip = set(cfg_keys) | {"cli_overrides", "strict"}
        instance_kwargs = {k: v for k, v in kwargs.items() if k not in skip}

        if ckpt.is_dir():
            if (ckpt / "norm_stats.json").is_file():
                config.norm_stats_path = str((ckpt / "norm_stats.json").resolve())
            if (ckpt / "tokenizer_config.json").is_file() or (ckpt / "tokenizer.json").is_file():
                config.tokenizer_name_or_path = str(ckpt.resolve())
            # When loading a LeRobot-saved DM0 checkpoint, prefer the bundled architecture config so
            # we don't need the original dexbotic base model on disk just to instantiate the module
            # (the trained weights live in the policy-level ``model.safetensors`` and will overwrite
            # the random init done in ``__init__`` via the parent's ``_load_as_safetensor``).
            if _is_arch_config_only_dir(ckpt):
                config.model_name_or_path = str(ckpt.resolve())
            elif _is_dexbotic_native_dir(ckpt):
                # Raw dexbotic dump: ``__init__`` already loads the weights via
                # ``_load_dm0_for_causal_lm``. Skip the parent ``_load_as_safetensor`` entirely so we
                # don't try to re-load a non-prefixed dexbotic state-dict as a LeRobot policy.
                config.model_name_or_path = str(ckpt.resolve())
                instance = cls(config, **instance_kwargs)
                instance.to(config.device)
                instance.eval()
                return instance

        return super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            strict=strict,
            **instance_kwargs,
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        super()._save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        ns = self.config.norm_stats_path
        if ns and Path(ns).is_file():
            shutil.copy2(ns, save_directory / "norm_stats.json")
        # Persist the dexbotic architecture config so the checkpoint is self-contained: ``from_pretrained``
        # can rebuild ``DM0ForCausalLM`` without requiring the original base checkpoint on disk.
        try:
            arch_cfg = self.dm0_model.config.to_dict()
            with open(save_directory / DM0_ARCH_CONFIG_FILENAME, "w") as f:
                json.dump(arch_cfg, f, indent=2)
        except Exception as e:
            logger.warning("Failed to persist DM0 architecture config (%s); checkpoint will still "
                           "require config.model_name_or_path to point at a dexbotic base.", e)
        cfg_path = save_directory / CONFIG_NAME
        if cfg_path.is_file():
            with open(cfg_path) as f:
                d = json.load(f)
            if (save_directory / "norm_stats.json").is_file():
                d["norm_stats_path"] = "norm_stats.json"
            d["tokenizer_name_or_path"] = None
            with open(cfg_path, "w") as f:
                json.dump(d, f, indent=4)

    def get_optim_params(self):
        """Return either a flat param list or a list of param groups.

        当 ``config.vit_lr_mult`` 为 ``None`` / 1.0 时返回单组（兼容旧行为）；
        否则按参数名是否包含 ``mm_vision_tower`` 划分两组：
          * ``vit``  : ViT 子树下的可训练参数（含 phase2 时挂在 ViT 上的 LoRA）
          * ``other``: 其它可训练参数（LLM / projector / action expert / ...）
        每组写入显式 ``lr``，与 ``LambdaLR`` 协作时两组 lr 会同步缩放且比例恒定。
        """
        cfg = self.config
        vit_mult = getattr(cfg, "vit_lr_mult", None)

        trainable = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        if vit_mult is None or vit_mult == 1.0:
            return [p for _, p in trainable]

        base_lr = float(cfg.optimizer_lr)
        vit_lr = base_lr * float(vit_mult)
        vit_substr = "mm_vision_tower"

        vit_params = [p for n, p in trainable if vit_substr in n]
        other_params = [p for n, p in trainable if vit_substr not in n]

        groups: list[dict[str, Any]] = []
        if other_params:
            groups.append({"params": other_params, "lr": base_lr, "name": "other"})
        if vit_params:
            groups.append({"params": vit_params, "lr": vit_lr, "name": "vit"})

        logger.info(
            "DM0 optim param groups: other=%d @ lr=%.3e, vit=%d @ lr=%.3e (mult=%.3f)",
            len(other_params), base_lr, len(vit_params), vit_lr, float(vit_mult),
        )
        return groups

    def reset(self) -> None:
        self._action_queue.clear()

    def _image_keys(self) -> list[str]:
        assert self.config.input_features is not None
        return sorted(
            k for k, ft in self.config.input_features.items() if ft.type == FeatureType.VISUAL
        )

    def _prepare_images(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        from torchvision.transforms.functional import to_pil_image

        keys = self._image_keys()
        if not keys:
            raise ValueError("DM0Policy requires at least one visual feature in config.input_features.")

        bsz = batch[keys[0]].shape[0]
        dev = batch[keys[0]].device
        model_dtype = next(self.dm0_model.parameters()).dtype

        per_cam: list[Tensor] = []
        masks: list[Tensor] = []
        for i in range(self.config.num_images):
            if i < len(keys):
                img = batch[keys[i]]
                if img.dim() == 5:
                    img = img[:, -1]
                pil_list = []
                for b in range(bsz):
                    chw = img[b].detach().float().cpu()
                    if chw.max() > 1.5:
                        chw = chw / 255.0
                    pil_list.append(to_pil_image(chw.clamp(0, 1)))
                cam = self.dm0_model.process_images(pil_list).to(device=dev, dtype=model_dtype)
                per_cam.append(cam)
                masks.append(torch.ones(bsz, dtype=torch.bool, device=dev))
            else:
                ref = per_cam[0] if per_cam else torch.zeros(bsz, 3, 224, 224, device=dev, dtype=model_dtype)
                per_cam.append(torch.zeros_like(ref))
                masks.append(torch.zeros(bsz, dtype=torch.bool, device=dev))

        images = torch.stack(per_cam, dim=1)
        image_masks = torch.stack(masks, dim=1)
        return images, image_masks

    def _tasks_as_strings(self, batch: dict[str, Tensor]) -> list[str]:
        raw = batch.get("task")
        if OBS_STATE in batch:
            bsz = int(batch[OBS_STATE].shape[0])
        else:
            bsz = int(batch[ACTION].shape[0])
        if raw is None:
            logger.warning("Batch missing 'task'; using empty string for DM0 tokenization.")
            return [""] * bsz
        if isinstance(raw, str):
            return [raw] * bsz
        if isinstance(raw, (list, tuple)):
            return [str(t) for t in raw]
        if isinstance(raw, Tensor):
            if raw.ndim == 0:
                return [str(raw.item())] * bsz
            return [str(raw[i].item()) for i in range(raw.shape[0])]
        return [str(raw)] * bsz

    def _tokenize_task(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        texts = self._tasks_as_strings(batch)

        ids_list = []
        for t in texts:
            enc = self.tokenization_func([{"from": "human", "value": t}])
            ids_list.append(torch.as_tensor(enc["input_ids"], dtype=torch.long))
        dev = batch[OBS_STATE].device if OBS_STATE in batch else batch[ACTION].device
        input_ids = torch.stack(ids_list, dim=0).to(dev)
        # Dexbotic DM0 attention expects bool padding masks (not int64).
        attention_mask = input_ids != self.tokenizer.pad_token_id
        return input_ids, attention_mask

    def _norm_action(self, x: Tensor) -> Tensor:
        return quantile_normalize(x, self._action_min, self._action_max)

    def _denorm_action(self, x: Tensor) -> Tensor:
        return quantile_denormalize(x, self._action_min, self._action_max)

    def _norm_state(self, x: Tensor) -> Tensor:
        return quantile_normalize(x, self._state_min, self._state_max)

    def _denorm_state(self, x: Tensor) -> Tensor:
        return quantile_denormalize(x, self._state_min, self._state_max)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        images, image_masks = self._prepare_images(batch)
        state = batch[OBS_STATE]
        if state.dim() == 3:
            state = state[:, -1]
        state = pad_to_dim(state, self.config.max_state_dim)
        action = pad_to_dim(batch[ACTION], self.config.max_action_dim)

        if self.config.use_delta_action:
            action = compute_delta(
                action,
                state,
                tuple(self.config.non_delta_mask),
                periodic_mask=tuple(self.config.periodic_mask),
                periodic_range=tuple(self.config.periodic_range),
            )
        action = self._norm_action(action)
        state_n = self._norm_state(state)

        input_ids, attention_mask = self._tokenize_task(batch)
        input_ids = input_ids.to(images.device)
        attention_mask = attention_mask.to(images.device)

        out = self.dm0_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_masks=image_masks,
            states=state_n,
            actions=action,
        )
        loss = out.loss
        return loss, {"action_loss": float(loss.detach())}

    def _init_ema(self):
        """初始化 EMA 影子参数为当前可训练参数的拷贝。

        延迟到第一次 update() 调用时执行，确保 PEFT/DDP 包装已完成，参数名稳定。
        对应 OpenPI: train.py:113 — ema_params = params（初始同源）
        """
        self._ema_params = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                self._ema_params[name] = param.data.clone()

    def update(self):
        """每个优化器步后调用，更新 EMA 影子参数。

        对应 OpenPI: train.py:169-175
        公式: ema = decay * ema + (1 - decay) * param
        """
        if self.config.ema_decay is None:
            return

        # 延迟初始化：首次调用时创建 EMA 影子参数
        if self._ema_params is None:
            self._init_ema()

        decay = self.config.ema_decay
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad and name in self._ema_params:
                    self._ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def _swap_to_ema(self) -> dict[str, torch.Tensor] | None:
        """将模型参数替换为 EMA 值，返回原始参数备份。

        使用 _ema_active 标志防止嵌套调用（select_action 调用 predict_action_chunk）。

        对应 OpenPI: checkpoints.py:145-152 中保存 EMA 参数的逻辑。
        """
        if self.config.ema_decay is None:
            return None
        if self._ema_params is None or self._ema_active:
            return None
        backup = {}
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in self._ema_params:
                    backup[name] = param.data.clone()
                    param.data.copy_(self._ema_params[name])
        self._ema_active = True
        return backup

    def _restore_from_backup(self, backup: dict[str, torch.Tensor] | None):
        """从备份恢复模型参数（EMA 推理后还原为训练参数）。"""
        if backup is None:
            return
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])
        self._ema_active = False

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        images, image_masks = self._prepare_images(batch)
        state = batch[OBS_STATE]
        if state.dim() == 3:
            state = state[:, -1]
        state = pad_to_dim(state, self.config.max_state_dim)
        state_n = self._norm_state(state)
        input_ids, attention_mask = self._tokenize_task(batch)
        input_ids = input_ids.to(images.device)
        attention_mask = attention_mask.to(images.device)

        raw = self.dm0_model.inference_action(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_masks=image_masks,
            states=state_n,
            diffusion_steps=self.config.diffusion_steps,
        )
        actions = self._denorm_action(raw)
        if self.config.use_delta_action:
            actions = compute_absolute(
                actions,
                state,
                tuple(self.config.non_delta_mask),
                periodic_mask=tuple(self.config.periodic_mask),
                periodic_range=tuple(self.config.periodic_range),
            )
        return actions[..., : self._original_action_dim]

    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        self.eval()
        backup = self._swap_to_ema()
        try:
            if len(self._action_queue) == 0:
                chunk = self.predict_action_chunk(batch)
                n = min(self.config.n_action_steps, chunk.shape[1])
                for i in range(n):
                    self._action_queue.append(chunk[:, i])
            return self._action_queue.popleft()
        finally:
            self._restore_from_backup(backup)

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """Default PEFT targets for DM0: LoRA on ViT attention ``out_proj`` and MLP ``c_fc`` / ``c_proj``.

        Mirrors the dexbotic ``r1_pro_dm0_freeze_lora.py`` recipe verbatim
        (``r=16, alpha=32, dropout=0.05``, targets ``["out_proj", "c_fc", "c_proj"]``).
        These names are specific to the dexbotic ``PEVisionTower`` (``transformer.resblocks.*``);
        HF ``CLIPVisionModel`` / ``SiglipVisionModel`` use a different naming scheme but the
        DM0-base checkpoint we ship with always uses PE (``mm_vision_tower="pe_lang_l14_728"``),
        so this is safe.

        Note: ``out_proj`` only contributes to the LoRA forward thanks to
        :func:`_patch_pe_self_attention_for_lora` (applied at module import). Without that patch
        dexbotic's ``SelfAttention.forward`` calls ``F.linear(attn, self.out_proj.weight, ...)``
        directly, which silently bypasses the LoRA branch (and was the root cause of the
        ``grad_norm == 0`` flat loss curve).

        The regex is anchored at the ``mm_vision_tower`` sub-tree so we never accidentally
        LoRA-wrap the LLM's attention.

        ``lora_alpha`` / ``lora_dropout`` are not exposed via :class:`PeftConfig` so we bake them
        in here; override by editing this method or by passing a fully-formed ``peft_config`` to
        :meth:`wrap_with_peft`.
        """
        target_modules = (
            r"dm0_model\.model\.mm_vision_tower\.vision_tower\..*\.(out_proj|c_fc|c_proj)"
        )
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        }

    def _validate_peft_config(self, peft_config) -> None:
        """Allow PEFT fine-tuning when ``model_name_or_path`` points at a pretrained DM0 dir,
        even without ``pretrained_path`` (which dexbotic-native checkpoints typically lack).
        """
        if not self.config.pretrained_path and not self.config.model_name_or_path:
            raise ValueError(
                "DM0 PEFT fine-tuning requires either `policy.path` (LeRobot-saved checkpoint) "
                "or `policy.model_name_or_path` (dexbotic DM0 base) to be set."
            )
