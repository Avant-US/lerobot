"""DM0 image augmentation: 1:1 port of dexbotic ``policy_dm0`` / ``policy_color_dm0``.

dexbotic config (see ``dexbotic/data/dataset/augmentations.py`` and
``playground/benchmarks/real/r1_pro_dm0_freeze_lora.py``)::

    aug_policy = ["dm0", "color_dm0", "color_dm0"]

LeRobot port:
    * The first camera (in dataset's ``camera_keys`` order, by default ``head``)
      gets the geometric ``policy_dm0``.
    * The remaining cameras (wrists / hands) get the color-only
      ``policy_color_dm0``.

The output of either policy is a 728x728 RGB image, identical to dexbotic.
The downstream ViT processor inside DM0 will resize 728 -> the model's
expected input size (e.g. 384) on its own.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
import torch.utils.data as torch_data
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.type_definitions import Targets

__all__ = [
    "PadToSquare",
    "policy_dm0",
    "policy_color_dm0",
    "AlbumentationsTorchTransform",
    "DM0AugmentedDataset",
    "build_dm0_per_camera_augs",
]


class PadToSquare(A.DualTransform):
    """Pad the shorter side so the image becomes square.

    Direct port of ``dexbotic.data.dataset.augmentations.PadToSquare`` so the
    pixel layout is identical to the original DM0 recipe.
    """

    _targets = (Targets.IMAGE, Targets.MASK)

    def __init__(
        self,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        border_mode: int = cv2.BORDER_CONSTANT,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.fill = fill
        self.fill_mask = fill_mask
        self.border_mode = border_mode

    @staticmethod
    def _pad(arr: np.ndarray, value: float | tuple[float, ...], border_mode: int) -> np.ndarray:
        h, w = arr.shape[:2]
        s = max(h, w)
        pad_top = (s - h) // 2
        pad_bottom = s - h - pad_top
        pad_left = (s - w) // 2
        pad_right = s - w - pad_left
        return fgeometric.pad_with_params(
            arr,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=border_mode,
            value=value,
        )

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return self._pad(img, self.fill, self.border_mode)

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        return self._pad(mask, self.fill_mask, self.border_mode)


def policy_dm0(p: float = 0.5, size: int = 728) -> A.Compose:
    """Geometric + color augmentation for the head camera (dexbotic ``dm0``)."""
    return A.Compose(
        [
            PadToSquare(border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=1.0),
            A.RandomResizedCrop(size=(size, size), scale=(0.95, 0.95), ratio=(1.0, 1.0), p=1.0),
            A.Rotate(limit=(-5, 5), p=1.0),
            A.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.1, p=p),
        ]
    )


def policy_color_dm0(p: float = 0.5, size: int = 728) -> A.Compose:
    """Color-only augmentation for the wrist/hand cameras (dexbotic ``color_dm0``)."""
    return A.Compose(
        [
            PadToSquare(border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=1.0),
            A.Resize(size, size, p=1.0),
            A.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.1, p=p),
        ]
    )


class AlbumentationsTorchTransform:
    """Adapter: wrap an ``A.Compose`` so it accepts/returns a torch image tensor.

    Input  : ``torch.Tensor`` of shape ``(C, H, W)`` float32 in ``[0, 1]``
             (the layout produced by LeRobot's ``hf_transform_to_torch``).
    Output : ``torch.Tensor`` of shape ``(C, H, W)`` float32 in ``[0, 1]``.

    The bridge round-trips through ``np.uint8 (H, W, C)`` since albumentations
    operates on numpy images.
    """

    def __init__(self, compose: A.Compose) -> None:
        self._compose = compose

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"AlbumentationsTorchTransform expects torch.Tensor, got {type(image)!r}")
        if image.ndim != 3:
            raise ValueError(f"Expected (C,H,W) image, got shape {tuple(image.shape)}")

        np_img = image.detach().cpu().clamp(0.0, 1.0).numpy()
        np_img = (np_img * 255.0 + 0.5).astype(np.uint8)
        np_img = np.transpose(np_img, (1, 2, 0))  # (C,H,W) -> (H,W,C)
        if np_img.shape[2] == 1:
            np_img = np.repeat(np_img, 3, axis=2)
        np_img = np.ascontiguousarray(np_img)

        out_np = self._compose(image=np_img)["image"]
        if out_np.ndim != 3:
            raise RuntimeError(f"Augmented image is not 3-D: shape={out_np.shape}")

        out = torch.from_numpy(out_np).to(torch.float32) / 255.0
        out = out.permute(2, 0, 1).contiguous()  # (H,W,C) -> (C,H,W)
        return out


def build_dm0_per_camera_augs(
    cam_keys: Iterable[str],
    head_cam_substr: str | None = "head",
    wrist_cam_substrs: Iterable[str] | None = ("wrist", "hand"),
    aug_prob: float = 0.5,
    size: int = 728,
) -> dict[str, AlbumentationsTorchTransform]:
    """Map every camera key to either ``policy_dm0`` (head) or ``policy_color_dm0`` (wrist).

    Dispatch rules, in order:

    1. If ``head_cam_substr`` is non-empty and matches the key (case-insensitive),
       the camera uses ``policy_dm0``.
    2. Otherwise, if any of ``wrist_cam_substrs`` matches the key,
       the camera uses ``policy_color_dm0``.
    3. Fallback (no substring matched any camera): the FIRST camera in ``cam_keys``
       gets ``policy_dm0`` and the rest get ``policy_color_dm0``. This mirrors
       dexbotic's positional ``aug_policy=["dm0", "color_dm0", "color_dm0"]``.

    If substring matching identifies wrists but no head, the first unmatched
    camera is promoted to ``policy_dm0``.
    """
    cam_keys = list(cam_keys)
    head_aug = AlbumentationsTorchTransform(policy_dm0(p=aug_prob, size=size))
    wrist_aug = AlbumentationsTorchTransform(policy_color_dm0(p=aug_prob, size=size))

    head_substr = head_cam_substr.lower() if head_cam_substr else None
    wrist_substrs_lower = [s.lower() for s in wrist_cam_substrs] if wrist_cam_substrs else []

    cam_to_aug: dict[str, AlbumentationsTorchTransform] = {}
    matched_head = False
    for cam in cam_keys:
        cam_lower = cam.lower()
        if head_substr and head_substr in cam_lower:
            cam_to_aug[cam] = head_aug
            matched_head = True
        elif any(s in cam_lower for s in wrist_substrs_lower):
            cam_to_aug[cam] = wrist_aug

    if not cam_to_aug:
        for i, cam in enumerate(cam_keys):
            cam_to_aug[cam] = head_aug if i == 0 else wrist_aug
        return cam_to_aug

    unset = [c for c in cam_keys if c not in cam_to_aug]
    if not matched_head and unset:
        cam_to_aug[unset[0]] = head_aug
        unset = unset[1:]
    for c in unset:
        cam_to_aug[c] = wrist_aug
    return cam_to_aug


class DM0AugmentedDataset(torch_data.Dataset):
    """Thin wrapper that applies per-camera augmentation on top of a ``LeRobotDataset``.

    All other attributes (``meta``, ``num_frames``, ``num_episodes``, ``episodes``,
    ``fps``, ``features``, ...) are forwarded to the wrapped instance via
    ``__getattr__``.
    """

    def __init__(
        self,
        base: torch_data.Dataset,
        cam_to_aug: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    ) -> None:
        super().__init__()
        self._base = base
        self._cam_to_aug = dict(cam_to_aug)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx):
        item = self._base[idx]
        for cam, aug in self._cam_to_aug.items():
            if cam in item:
                item[cam] = aug(item[cam])
        return item

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._base, name)

    def __repr__(self) -> str:
        cam_summary = ", ".join(self._cam_to_aug.keys())
        return f"DM0AugmentedDataset(cams=[{cam_summary}], base={self._base!r})"
