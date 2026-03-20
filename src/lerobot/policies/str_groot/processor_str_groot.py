"""Pre / post processor factories for the StrGroot policy."""

from __future__ import annotations

from typing import Any

import torch

from lerobot.policies.str_groot.configuration_str_groot import StrGrootConfig
from lerobot.processor import (
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def make_str_groot_pre_post_processors(
    config: StrGrootConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build minimal pre- and post-processor pipelines for StrGroot.

    The preprocessor normalises actions (MIN_MAX → [-1, 1] for StarVLA's
    flow-matching head) and moves tensors to the target device.
    The postprocessor un-normalises predicted actions back to the original
    value range and moves them to CPU.
    """
    all_features = {**config.input_features, **config.output_features}

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        NormalizerProcessorStep(
            features=all_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=input_steps,
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=output_steps,
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    return preprocessor, postprocessor
