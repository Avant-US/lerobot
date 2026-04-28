# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""DM0 policy (dexbotic) for LeRobot — phase 1 imports dexbotic and registers HF Auto classes."""

from transformers import AutoConfig, AutoModelForCausalLM

from dexbotic.model.dm0.dm0_arch import DM0Config as DM0ArchConfig
from dexbotic.model.dm0.dm0_arch import DM0ForCausalLM

try:
    AutoConfig.register("dexbotic_dm0", DM0ArchConfig)
    AutoModelForCausalLM.register(DM0ArchConfig, DM0ForCausalLM)
except ValueError:
    pass

from lerobot.policies.dm0.configuration_dm0 import DM0Config
from lerobot.policies.dm0.modeling_dm0 import DM0Policy

__all__ = [
    "DM0ArchConfig",
    "DM0Config",
    "DM0ForCausalLM",
    "DM0Policy",
]
