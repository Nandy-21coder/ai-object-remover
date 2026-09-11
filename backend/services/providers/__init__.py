"""
providers package
Modular real AI inpainting provider adapters and registry.
"""

from typing import Dict, Type

from services.providers.base import (
    BaseInpaintingProvider,
    ProviderNotConfiguredError,
    InvalidAPIKeyError,
    RateLimitError,
    ProviderTimeoutError,
    InpaintingAPIError,
)
from services.providers.stability import StabilityInpaintingProvider
from services.providers.replicate import ReplicateInpaintingProvider
from services.providers.clipdrop import ClipdropInpaintingProvider
from services.providers.fal import FalInpaintingProvider
from services.providers.local import LocalInpaintingProvider
from services.providers.lama import LamaInpaintingProvider
from services.providers.huggingface import HuggingFaceInpaintingProvider

PROVIDER_REGISTRY: Dict[str, Type[BaseInpaintingProvider]] = {
    "local": LocalInpaintingProvider,
    "lama": LamaInpaintingProvider,
    "stability": StabilityInpaintingProvider,
    "replicate": ReplicateInpaintingProvider,
    "clipdrop": ClipdropInpaintingProvider,
    "fal": FalInpaintingProvider,
    "huggingface": HuggingFaceInpaintingProvider,
}

__all__ = [
    "BaseInpaintingProvider",
    "ProviderNotConfiguredError",
    "InvalidAPIKeyError",
    "RateLimitError",
    "ProviderTimeoutError",
    "InpaintingAPIError",
    "LocalInpaintingProvider",
    "LamaInpaintingProvider",
    "StabilityInpaintingProvider",
    "ReplicateInpaintingProvider",
    "ClipdropInpaintingProvider",
    "FalInpaintingProvider",
    "HuggingFaceInpaintingProvider",
    "PROVIDER_REGISTRY",
]

