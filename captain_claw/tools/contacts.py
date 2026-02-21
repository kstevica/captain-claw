"""Contacts tool for cross-session persistent address book management."""

from typing import Any

from captain_claw.logging import get_logger
from captain_claw.session import get_session_manager
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class ContactsTool(Tool):
    """Manage a persistent cross-session address book."""

    name = "contacts"
    description = (
        "Manage a persistent cross-session address book. "
        "Contacts survive across sessions and store names, roles, organizations, "
        "emails, and accumulated context notes. "
        "Use action 'add' to create, 'list' to query, 'search' to find by name/org, "
        "'info' for detailed view, 'update' to modify, and 'remove' to delete."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "search", "info", "update", "remove"],
                "description": "Operation to perform.",
            },
            "name": {
                "type": "string",
                "description": "Contact name (required for 'add').",
            },
            "contact_id": {
                "type": "string",
                "description": "Contact ID, #index, or name (for 'info'/'update'/'remove').",
            },
            "description": {
                "type": "string",
                "description": "Short description of the person.",
            },
            "position": {
                "type": "string",
                "description": "Job title.",
            },
            "organization": {
                "type": "string",
                "description": "Company or organization.",
            },
            "relation": {
                "type": "string",
                "description": "Relationship: colleague, client, manager, friend, vendor, etc.",
            },
            "email": {
                "type": "string",
                "description": "Email address(es), comma-separated.",
            },
            "phone": {
                "type": "string",
                "description": "Phone number.",
            },
            "importance": {
                "type": "integer",
                "description": "Importance score 1-10 (manual override pins the value).",
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated tags.",
            },
            "notes": {
                "type": "string",
                "description": "Context notes to append.",
            },
            "query": {
                "type": "string",
                "description": "Search query (for 'search').",
            },
            "privacy_tier": {
                "type": "string",
                "enum": ["normal", "private"],
                "description": "Privacy tier (default: normal). Private contacts are not auto-injected.",
            },
        },
        "required": ["action"],
    }

    async def execute(
        self,
        action: str,
        name: str | None = None,
        contact_id: str | None = None,
        description: str | None = None,
        position: str | None = None,
        organization: str | None = None,
        relation: str | None = None,
        email: str | None = None,
        phone: str | None = None,
        importance: int | None = None,
        tags: str | None = None,
        notes: str | None = None,
        query: str | None = None,
        privacy_tier: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        session_id = str(kwargs.get("_session_id", "") or "").strip() or None
        sm = get_session_manager()

        try:
            if action == "add":
                return await self._add(
                    sm, name, description, position, organization,
                    relation, email, phone, importance, tags, notes,
                    privacy_tier, session_id,
                )
            if action == "list":
                return await self._list(sm)
            if action == "search":
                return await self._search(sm, query)
            if action == "info":
                return await self._info(sm, contact_id)
            if action == "update":
                return await self._update(
                    sm, contact_id, name, description, position,
                    organization, relation, email, phone, importance,
                    tags, notes, privacy_tier,
                )
            if action == "remove":
                return await self._remove(sm, contact_id)
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            log.error("Contacts tool error", action=action, error=str(e))
            return ToolResult(success=False, error=str(e))

    # ------------------------------------------------------------------

    @staticmethod
    async def _add(
        sm: Any,
        name: str | None,
        description: str | None,
        position: str | None,
        organization: str | None,
        relation: str | None,
        email: str | None,
        phone: str | None,
        importance: int | None,
        tags: str | None,
        notes: str | None,
        privacy_tier: str | None,
        session_id: str | None,
    ) -> ToolResult:
        if not name or not name.strip():
            return ToolResult(success=False, error="'name' is required for add.")
        item = await sm.create_contact(
            name=name.strip(),
            description=description,
            position=position,
            organization=organization,
            relation=relation,
            email=email,
            phone=phone,
            importance=importance or 1,
            source_session=session_id,
            tags=tags,
            notes=notes,
            privacy_tier=privacy_tier or "normal",
        )
        return ToolResult(
            success=True,
            content=f"Created contact #{item.id[:8]}: {item.name}",
        )

    @staticmethod
    async def _list(sm: Any) -> ToolResult:
        items = await sm.list_contacts(limit=50)
        if not items:
            return ToolResult(success=True, content="No contacts found.")
        lines: list[str] = []
        for idx, c in enumerate(items, 1):
            org_part = f" @ {c.organization}" if c.organization else ""
            pos_part = f" ({c.position})" if c.position else ""
            lines.append(
                f"#{idx} [{c.importance}] {c.name}{pos_part}{org_part}"
                f" [{c.relation or '-'}]  id={c.id[:8]}"
            )
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _search(sm: Any, query: str | None) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(success=False, error="'query' is required for search.")
        items = await sm.search_contacts(query.strip(), limit=20)
        if not items:
            return ToolResult(success=True, content=f"No contacts matching: {query}")
        lines: list[str] = []
        for idx, c in enumerate(items, 1):
            org_part = f" @ {c.organization}" if c.organization else ""
            lines.append(
                f"#{idx} [{c.importance}] {c.name}{org_part}"
                f"  id={c.id[:8]}"
            )
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _info(sm: Any, contact_id: str | None) -> ToolResult:
        if not contact_id:
            return ToolResult(success=False, error="'contact_id' is required for info.")
        item = await sm.select_contact(contact_id)
        if not item:
            return ToolResult(success=False, error=f"Contact not found: {contact_id}")
        parts = [f"Name: {item.name}", f"ID: {item.id}"]
        if item.position:
            parts.append(f"Position: {item.position}")
        if item.organization:
            parts.append(f"Organization: {item.organization}")
        if item.relation:
            parts.append(f"Relation: {item.relation}")
        if item.email:
            parts.append(f"Email: {item.email}")
        if item.phone:
            parts.append(f"Phone: {item.phone}")
        parts.append(f"Importance: {item.importance} (pinned={item.importance_pinned})")
        parts.append(f"Mentions: {item.mention_count}")
        if item.last_seen_at:
            parts.append(f"Last seen: {item.last_seen_at}")
        if item.tags:
            parts.append(f"Tags: {item.tags}")
        if item.description:
            parts.append(f"Description: {item.description}")
        if item.notes:
            parts.append(f"Notes: {item.notes}")
        parts.append(f"Privacy: {item.privacy_tier}")
        return ToolResult(success=True, content="\n".join(parts))

    @staticmethod
    async def _update(
        sm: Any,
        contact_id: str | None,
        name: str | None,
        description: str | None,
        position: str | None,
        organization: str | None,
        relation: str | None,
        email: str | None,
        phone: str | None,
        importance: int | None,
        tags: str | None,
        notes: str | None,
        privacy_tier: str | None,
    ) -> ToolResult:
        if not contact_id:
            return ToolResult(success=False, error="'contact_id' is required for update.")
        item = await sm.select_contact(contact_id)
        if not item:
            return ToolResult(success=False, error=f"Contact not found: {contact_id}")
        kwargs: dict[str, Any] = {}
        if name is not None:
            kwargs["name"] = name
        if description is not None:
            kwargs["description"] = description
        if position is not None:
            kwargs["position"] = position
        if organization is not None:
            kwargs["organization"] = organization
        if relation is not None:
            kwargs["relation"] = relation
        if email is not None:
            kwargs["email"] = email
        if phone is not None:
            kwargs["phone"] = phone
        if importance is not None:
            kwargs["importance"] = max(1, min(10, importance))
            kwargs["importance_pinned"] = True
        if tags is not None:
            kwargs["tags"] = tags
        if privacy_tier is not None:
            kwargs["privacy_tier"] = privacy_tier
        # Append notes rather than replacing
        if notes is not None:
            existing = item.notes or ""
            if existing:
                kwargs["notes"] = existing.rstrip() + "\n" + notes
            else:
                kwargs["notes"] = notes
        if not kwargs:
            return ToolResult(success=False, error="No fields to update.")
        ok = await sm.update_contact(item.id, **kwargs)
        if not ok:
            return ToolResult(success=False, error="Update failed.")
        return ToolResult(success=True, content=f"Updated contact #{item.id[:8]}.")

    @staticmethod
    async def _remove(sm: Any, contact_id: str | None) -> ToolResult:
        if not contact_id:
            return ToolResult(success=False, error="'contact_id' is required for remove.")
        item = await sm.select_contact(contact_id)
        if not item:
            return ToolResult(success=False, error=f"Contact not found: {contact_id}")
        ok = await sm.delete_contact(item.id)
        if not ok:
            return ToolResult(success=False, error="Delete failed.")
        return ToolResult(success=True, content=f"Removed contact #{item.id[:8]}: {item.name}")
