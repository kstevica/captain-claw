"""Async REST client for the vast.ai API.

Uses httpx directly rather than the ``vastai`` CLI/SDK package (which is
synchronous and CLI-focused).  This gives us full async control and avoids
an extra dependency.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from captain_claw.vastai.models import (
    VastAccountInfo,
    VastOffer,
    VastOfferFilter,
)

log = logging.getLogger(__name__)

VAST_API_BASE = "https://console.vast.ai/api/v0"


class VastAPIError(Exception):
    """Raised when the vast.ai API returns an error."""

    def __init__(self, message: str, status_code: int = 0, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class VastAIClient:
    """Low-level async client for the vast.ai REST API.

    All methods raise :class:`VastAPIError` on HTTP errors.
    """

    def __init__(self, api_key: str, *, base_url: str = VAST_API_BASE):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, read=60.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | list | None = None,
        params: dict | None = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        try:
            resp = await self._http.request(method, url, json=json, params=params)
        except httpx.HTTPError as exc:
            raise VastAPIError(f"HTTP error talking to vast.ai: {exc}") from exc

        if not resp.is_success:
            raise VastAPIError(
                f"vast.ai API {method} {path} returned {resp.status_code}",
                status_code=resp.status_code,
                body=resp.text[:2000],
            )
        # Some endpoints return empty body on success (e.g. DELETE).
        if not resp.text.strip():
            return {}
        return resp.json()

    async def _get(self, path: str, **kw: Any) -> Any:
        return await self._request("GET", path, **kw)

    async def _post(self, path: str, **kw: Any) -> Any:
        return await self._request("POST", path, **kw)

    async def _put(self, path: str, **kw: Any) -> Any:
        return await self._request("PUT", path, **kw)

    async def _delete(self, path: str, **kw: Any) -> Any:
        return await self._request("DELETE", path, **kw)

    # ------------------------------------------------------------------
    # Search offers
    # ------------------------------------------------------------------

    async def search_offers(self, filters: VastOfferFilter) -> list[VastOffer]:
        """Search the vast.ai marketplace for available GPU offers.

        Returns a list of :class:`VastOffer` sorted by the filter's
        ``sort_by`` field.

        Uses the GET ``/bundles/?q=<json>`` format which is more reliable
        than the POST body format.
        """
        import json as _json

        # Build the query object in vast.ai's filter format.
        query: dict[str, Any] = {
            "verified": {"eq": filters.verified},
            "external": {"eq": False},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "num_gpus": {"eq": filters.num_gpus},
            "order": [[filters.sort_by, "asc"]],
            "type": "on-demand",
            "allocated_storage": 5.0,
            "limit": filters.limit,
        }

        # GPU name: vast.ai only supports exact match, so we filter client-side
        # for partial/substring matches. We still pass exact match to the API
        # when the name looks like a complete GPU model (helps narrow results).
        gpu_name_filter = ""
        if filters.gpu_name:
            gpu_name_filter = filters.gpu_name.replace("_", " ").strip().upper()
            # Don't send to API — we'll filter in Python for substring support.
        if filters.min_gpu_ram_gb > 0:
            query["gpu_ram"] = {"gte": filters.min_gpu_ram_gb * 1024}  # API uses MB
        if filters.max_price_per_hour > 0:
            query["dph_total"] = {"lte": filters.max_price_per_hour}
        if filters.min_reliability > 0:
            query["reliability2"] = {"gte": filters.min_reliability}
        if filters.min_disk_gb > 0:
            query["disk_space"] = {"gte": filters.min_disk_gb}
        if filters.min_inet_down_mbps > 0:
            query["inet_down"] = {"gte": filters.min_inet_down_mbps}
        if filters.direct:
            query["direct_port_count"] = {"gte": 1}

        # When filtering by GPU name client-side, fetch more to compensate.
        if gpu_name_filter:
            query["limit"] = max(filters.limit * 5, 100)

        log.debug("vast.ai search_offers query=%s", query)
        data = await self._get("/bundles/", params={"q": _json.dumps(query)})

        offers_raw = data if isinstance(data, list) else data.get("offers", [])
        offers: list[VastOffer] = []
        for raw in offers_raw:
            try:
                offer = _parse_offer(raw)
            except Exception:
                log.debug("Skipping unparseable offer: %s", raw.get("id", "?"))
                continue
            # Client-side GPU name substring filter.
            if gpu_name_filter and gpu_name_filter not in offer.gpu_name.upper():
                continue
            offers.append(offer)
        # Re-apply the original limit after filtering.
        return offers[: filters.limit]

    # ------------------------------------------------------------------
    # Instance lifecycle
    # ------------------------------------------------------------------

    async def create_instance(
        self,
        offer_id: int,
        *,
        image: str = "ollama/ollama",
        disk_gb: int = 64,
        env: dict[str, str] | None = None,
        onstart_cmd: str = "",
        direct: bool = True,
        ssh: bool = True,
        label: str = "",
    ) -> tuple[int, str]:
        """Accept an offer and create a new instance.

        Returns ``(instance_id, instance_api_key)``.
        """
        # vast.ai expects env as a plain dict of KEY: VALUE.
        env_dict = dict(env or {})

        body: dict[str, Any] = {
            "image": image,
            "disk": float(disk_gb),
            "env": env_dict,
            "runtype": "ssh" if ssh else "args",
        }
        if onstart_cmd:
            body["onstart"] = onstart_cmd
        if label:
            body["label"] = label

        log.info("vast.ai create_instance offer=%d image=%s disk=%dGB", offer_id, image, disk_gb)
        data = await self._put(f"/asks/{offer_id}/", json=body)
        instance_id = data.get("new_contract")
        if not instance_id:
            raise VastAPIError(
                f"create_instance did not return an instance ID: {data}",
            )
        instance_api_key = str(data.get("instance_api_key", "") or "")
        return int(instance_id), instance_api_key

    async def get_instance(self, instance_id: int) -> dict[str, Any]:
        """Get full details for a single instance."""
        data = await self._get(f"/instances/{instance_id}/")
        # The API may return {"instances": [obj]} or the object directly.
        if "instances" in data and isinstance(data["instances"], list):
            for inst in data["instances"]:
                if inst.get("id") == instance_id:
                    return inst
        return data

    async def list_instances(self) -> list[dict[str, Any]]:
        """List all instances on the account."""
        data = await self._get("/instances/")
        if isinstance(data, list):
            return data
        return data.get("instances", [])

    async def stop_instance(self, instance_id: int) -> None:
        """Stop a running instance (GPU billing stops, storage continues)."""
        log.info("vast.ai stop_instance id=%d", instance_id)
        await self._put(f"/instances/{instance_id}/", json={"state": "stopped"})

    async def start_instance(self, instance_id: int) -> None:
        """Start a stopped instance."""
        log.info("vast.ai start_instance id=%d", instance_id)
        await self._put(f"/instances/{instance_id}/", json={"state": "running"})

    async def destroy_instance(self, instance_id: int) -> None:
        """Permanently destroy an instance and all its data."""
        log.info("vast.ai destroy_instance id=%d", instance_id)
        await self._delete(f"/instances/{instance_id}/")

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_account(self) -> VastAccountInfo:
        """Get account info including balance.

        Note: vast.ai uses ``credit`` for the actual spendable balance,
        not ``balance`` (which is always 0 for credit-based accounts).
        """
        data = await self._get("/users/current/")
        # Use 'credit' first (actual spendable amount), fall back to 'balance'.
        balance = float(data.get("credit", 0) or 0)
        if balance == 0:
            balance = float(data.get("balance", 0) or 0)
        return VastAccountInfo(
            balance=balance,
            email=str(data.get("email", "") or ""),
            username=str(data.get("username", "") or ""),
            ssh_key=str(data.get("ssh_key", "") or ""),
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_offer(raw: dict[str, Any]) -> VastOffer:
    """Parse a raw offer dict from the vast.ai API into a ``VastOffer``."""
    return VastOffer(
        id=int(raw.get("id", 0)),
        gpu_name=str(raw.get("gpu_name", "") or ""),
        gpu_ram_gb=round(float(raw.get("gpu_ram", 0) or 0) / 1024, 1),  # MB -> GB
        num_gpus=int(raw.get("num_gpus", 1) or 1),
        cpu_cores=int(raw.get("cpu_cores_effective", 0) or raw.get("cpu_cores", 0) or 0),
        ram_gb=round(float(raw.get("cpu_ram", 0) or 0) / 1024, 1),  # MB -> GB
        disk_gb=round(float(raw.get("disk_space", 0) or 0), 1),
        dph_total=round(float(raw.get("dph_total", 0) or 0), 4),
        storage_cost_per_gb_month=round(float(raw.get("storage_cost", 0) or 0), 4),
        reliability=round(float(raw.get("reliability2", 0) or raw.get("reliability", 0) or 0), 4),
        inet_down_mbps=round(float(raw.get("inet_down", 0) or 0), 1),
        inet_up_mbps=round(float(raw.get("inet_up", 0) or 0), 1),
        cuda_version=float(raw.get("cuda_max_good", 0) or 0),
        direct_port_count=int(raw.get("direct_port_count", 0) or 0),
        geolocation=str(raw.get("geolocation", "") or ""),
        host_id=int(raw.get("host_id", 0) or 0),
        machine_id=int(raw.get("machine_id", 0) or 0),
        verified=bool(raw.get("verification") == "verified" or raw.get("verified")),
    )


def extract_port_mapping(
    ports_data: dict[str, Any] | list | None,
    internal_port: int = 11434,
) -> tuple[str, int]:
    """Extract ``(public_ip, external_port)`` for *internal_port* from
    a vast.ai instance's port data.

    Returns ``("", 0)`` if the mapping is not found.
    """
    if not ports_data:
        return ("", 0)

    # vast.ai returns ports as a dict like:
    # {"11434/tcp": [{"HostIp": "1.2.3.4", "HostPort": "34567"}]}
    key = f"{internal_port}/tcp"
    if isinstance(ports_data, dict):
        mapping = ports_data.get(key)
        if isinstance(mapping, list) and mapping:
            entry = mapping[0]
            return (
                str(entry.get("HostIp", "") or ""),
                int(entry.get("HostPort", 0) or 0),
            )

    return ("", 0)
