"""vast.ai GPU cloud integration for Captain Claw.

Provides GPU instance management with automatic Ollama setup,
accessible through Flight Deck routes.
"""

from captain_claw.vastai.client import VastAIClient, VastAPIError
from captain_claw.vastai.manager import VastAIManager
from captain_claw.vastai.models import (
    CreateInstanceRequest,
    InstanceConnectionInfo,
    PullModelRequest,
    SetAutoStopRequest,
    VastAccountInfo,
    VastInstance,
    VastInstanceState,
    VastOffer,
    VastOfferFilter,
)

__all__ = [
    "VastAIClient",
    "VastAIManager",
    "VastAPIError",
    "VastOffer",
    "VastOfferFilter",
    "VastInstance",
    "VastInstanceState",
    "VastAccountInfo",
    "CreateInstanceRequest",
    "PullModelRequest",
    "InstanceConnectionInfo",
]
