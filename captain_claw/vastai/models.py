"""Pydantic data models for the vast.ai GPU cloud integration."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Search / Offers
# ---------------------------------------------------------------------------


class VastOfferFilter(BaseModel):
    """Filters for searching available GPU offers on vast.ai."""

    gpu_name: str = ""
    """GPU model name filter, e.g. ``"RTX_4090"``, ``"H100_SXM"``."""

    min_gpu_ram_gb: float = 0
    """Minimum GPU VRAM in GB (e.g. 24 for 24 GB cards)."""

    max_price_per_hour: float = 0
    """Maximum total cost per hour ($/hr). 0 = no limit."""

    min_reliability: float = 0.95
    """Minimum host reliability score (0.0 -- 1.0)."""

    min_disk_gb: float = 50
    """Minimum available disk space in GB."""

    min_inet_down_mbps: float = 100
    """Minimum download bandwidth in Mbps."""

    num_gpus: int = 1
    """Exact number of GPUs required."""

    direct: bool = True
    """Only show offers with direct port mapping (low-latency)."""

    verified: bool = True
    """Only show verified hosts."""

    sort_by: str = "dph_total"
    """Sort field. Common: ``dph_total``, ``gpu_ram``, ``total_flops``."""

    limit: int = 20
    """Max results to return."""


class VastOffer(BaseModel):
    """A single available GPU offer from the vast.ai marketplace."""

    id: int
    gpu_name: str = ""
    gpu_ram_gb: float = 0
    num_gpus: int = 1
    cpu_cores: int = 0
    ram_gb: float = 0
    disk_gb: float = 0
    dph_total: float = 0
    """Total cost per hour ($/hr) on-demand."""
    storage_cost_per_gb_month: float = 0
    reliability: float = 0
    inet_down_mbps: float = 0
    inet_up_mbps: float = 0
    cuda_version: float = 0
    direct_port_count: int = 0
    geolocation: str = ""
    host_id: int = 0
    machine_id: int = 0
    verified: bool = False

    @property
    def has_direct_ports(self) -> bool:
        return self.direct_port_count > 0


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------


class VastInstanceState(StrEnum):
    """Lifecycle states for a vast.ai instance."""

    CREATING = "creating"
    LOADING = "loading"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    EXITED = "exited"
    ERROR = "error"
    DESTROYED = "destroyed"  # local-only sentinel after DELETE


# Map vast.ai API status strings to our enum.
_VAST_STATUS_MAP: dict[str, VastInstanceState] = {
    "creating": VastInstanceState.CREATING,
    "loading": VastInstanceState.LOADING,
    "running": VastInstanceState.RUNNING,
    "stopping": VastInstanceState.STOPPING,
    "stopped": VastInstanceState.STOPPED,
    "exited": VastInstanceState.EXITED,
    "error": VastInstanceState.ERROR,
}


def parse_vast_status(raw: str) -> VastInstanceState:
    """Convert a vast.ai API status string to ``VastInstanceState``."""
    return _VAST_STATUS_MAP.get(raw.lower().strip(), VastInstanceState.ERROR)


class VastInstance(BaseModel):
    """A managed vast.ai GPU instance with Ollama."""

    id: int
    offer_id: int = 0
    gpu_name: str = ""
    num_gpus: int = 1
    gpu_ram_gb: float = 0
    state: VastInstanceState = VastInstanceState.CREATING
    public_ip: str = ""
    ollama_port: int = 0
    """Mapped external port for Ollama (internal 11434)."""
    ssh_port: int = 0
    auth_token: str = ""
    """Bearer token (OPEN_BUTTON_TOKEN) for authenticating with the instance."""
    dph_total: float = 0
    """Cost per hour while running."""
    disk_gb: float = 0
    created_at: str = ""
    label: str = ""

    # Auto-stop after inactivity
    auto_stop_minutes: int = 0
    """Auto-stop after N minutes of inactivity. 0 = disabled."""
    last_activity_at: str = ""
    """ISO timestamp of last Ollama API activity on this instance."""

    # Ollama status
    ollama_ready: bool = False
    """True once the Ollama health check passes."""
    ollama_error: str = ""
    """Error message if Ollama setup failed."""
    models_loaded: list[str] = Field(default_factory=list)
    """Model tags currently available on this instance."""

    @property
    def is_active(self) -> bool:
        """True if the instance is in a state that costs GPU money."""
        return self.state in (
            VastInstanceState.CREATING,
            VastInstanceState.LOADING,
            VastInstanceState.RUNNING,
        )

    @property
    def is_transitioning(self) -> bool:
        """True if the instance is between stable states."""
        return self.state in (
            VastInstanceState.CREATING,
            VastInstanceState.LOADING,
            VastInstanceState.STOPPING,
        )

    @property
    def ollama_base_url(self) -> str:
        """Full Ollama API base URL for this instance, or empty string."""
        if not self.public_ip or not self.ollama_port:
            return ""
        # The ollama/ollama image serves plain HTTP (no TLS).
        return f"http://{self.public_ip}:{self.ollama_port}"


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------


class VastAccountInfo(BaseModel):
    """vast.ai account information."""

    balance: float = 0
    email: str = ""
    username: str = ""
    ssh_key: str = ""


# ---------------------------------------------------------------------------
# Request / response helpers for Flight Deck routes
# ---------------------------------------------------------------------------


class CreateInstanceRequest(BaseModel):
    """Request body for creating a new vast.ai instance."""

    offer_id: int
    label: str = ""
    disk_gb: int = 64
    pre_pull_model: str = ""
    """If set, automatically pull this model after Ollama starts."""


class PullModelRequest(BaseModel):
    """Request body for pulling a model on a running instance."""

    model: str
    """Model tag, e.g. ``llama3.2``, ``deepseek-r1:70b``."""


class SetAutoStopRequest(BaseModel):
    """Request body for setting the auto-stop timer on an instance."""

    auto_stop_minutes: int = 0
    """Minutes of inactivity before auto-stop. 0 = disabled.
    Allowed values: 0, 1, 2, 5, 10."""


class InstanceConnectionInfo(BaseModel):
    """Connection details for using a vast.ai instance as an Ollama provider."""

    provider: str = "ollama"
    base_url: str = ""
    api_key: str = ""
    ollama_ready: bool = False
    models: list[str] = Field(default_factory=list)
