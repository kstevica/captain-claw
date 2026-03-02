"""Instance capabilities registry for BotPort routing."""

from __future__ import annotations

import logging

from botport.connection_manager import ConnectionManager
from botport.models import InstanceInfo

log = logging.getLogger(__name__)


class Registry:
    """Capability-aware wrapper around ConnectionManager.

    Provides queries to find instances by expertise, tools, or load.
    """

    def __init__(self, connections: ConnectionManager) -> None:
        self._connections = connections

    def find_by_expertise(
        self,
        tags: list[str],
        exclude_instance: str | None = None,
    ) -> list[tuple[InstanceInfo, int]]:
        """Find available instances matching expertise tags.

        Uses fuzzy matching: a query tag matches an instance tag if either
        is a substring of the other (e.g. "m&a" matches "m&a / deal advisory",
        "valuation" matches "valuation support").

        Returns list of (instance, match_count) sorted by match_count descending,
        then by active_concerns ascending (prefer least loaded).
        """
        if not tags:
            return []

        query_tags = [t.lower().strip() for t in tags if t.strip()]
        if not query_tags:
            return []

        candidates: list[tuple[InstanceInfo, int]] = []
        for instance in self._connections.list_available(exclude=exclude_instance):
            instance_tags = list(instance.all_expertise_tags())
            matched = self._count_fuzzy_matches(query_tags, instance_tags)
            if matched > 0:
                candidates.append((instance, matched))

        # Sort by match count (desc), then active concerns (asc).
        candidates.sort(key=lambda x: (-x[1], x[0].active_concerns))
        return candidates

    @staticmethod
    def _count_fuzzy_matches(
        query_tags: list[str],
        instance_tags: list[str],
    ) -> int:
        """Count how many query tags fuzzy-match at least one instance tag.

        A match occurs if:
          - exact equality, OR
          - query tag is a substring of an instance tag, OR
          - an instance tag is a substring of the query tag
        """
        matched = 0
        for qt in query_tags:
            for it in instance_tags:
                if qt == it or qt in it or it in qt:
                    matched += 1
                    break  # one match per query tag is enough
        return matched

    def find_by_tool(
        self,
        tool_name: str,
        exclude_instance: str | None = None,
    ) -> list[InstanceInfo]:
        """Find available instances that have a specific tool."""
        tool_lower = tool_name.lower().strip()
        return [
            i for i in self._connections.list_available(exclude=exclude_instance)
            if tool_lower in {t.lower() for t in i.tools}
        ]

    def find_by_persona_name(
        self,
        name: str,
        exclude_instance: str | None = None,
    ) -> list[InstanceInfo]:
        """Find available instances with a specific persona name."""
        name_lower = name.lower().strip()
        results: list[InstanceInfo] = []
        for instance in self._connections.list_available(exclude=exclude_instance):
            for persona in instance.personas:
                if persona.name.lower().strip() == name_lower:
                    results.append(instance)
                    break
        return results

    def get_least_loaded(
        self,
        exclude_instance: str | None = None,
    ) -> InstanceInfo | None:
        """Get the available instance with the lowest active concern count."""
        available = self._connections.list_available(exclude=exclude_instance)
        if not available:
            return None
        return min(available, key=lambda i: i.active_concerns)

    def get_all_expertise_tags(self) -> dict[str, list[str]]:
        """Return all expertise tags grouped by instance name."""
        result: dict[str, list[str]] = {}
        for instance in self._connections.list_connected():
            tags = sorted(instance.all_expertise_tags())
            if tags:
                result[instance.name] = tags
        return result

    def get_summary(self) -> list[dict]:
        """Return a summary of all instances for the BotPort tool's list_agents action."""
        summaries: list[dict] = []
        for instance in self._connections.list_connected():
            summaries.append({
                "name": instance.name,
                "status": instance.status,
                "personas": [
                    {"name": p.name, "expertise": p.expertise_tags}
                    for p in instance.personas
                ],
                "tools_count": len(instance.tools),
                "models": instance.models,
                "active_concerns": instance.active_concerns,
                "max_concurrent": instance.max_concurrent,
            })
        return summaries
