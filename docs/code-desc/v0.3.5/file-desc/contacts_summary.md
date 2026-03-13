# Summary: contacts.py

# contacts.py Summary

Manages a persistent cross-session address book that survives between user sessions, storing comprehensive contact information including names, roles, organizations, emails, and contextual notes. Implements a Tool interface providing six core operations (add, list, search, info, update, remove) with support for contact importance scoring, privacy tiers, and relationship categorization.

## Purpose

Solves the problem of maintaining a durable, queryable contact database across multiple conversation sessions without losing relationship context or metadata. Enables users to build and reference a personal network database with rich attributes (position, organization, relation type, tags, accumulated notes) while supporting privacy controls and importance-based prioritization.

## Most Important Functions/Classes

1. **ContactsTool (class)** – Main Tool subclass implementing the contact management interface. Defines the schema for all parameters (action, name, contact_id, email, phone, importance, tags, notes, privacy_tier, etc.) and routes incoming requests to appropriate handler methods. Inherits from Tool registry system.

2. **execute() (async method)** – Entry point dispatcher that validates the action parameter and routes to specific handlers (_add, _list, _search, _info, _update, _remove). Extracts session_id from kwargs and retrieves the session manager. Wraps all operations in exception handling with logging.

3. **_add() (async static method)** – Creates a new contact with validation (name required), calls sm.create_contact() with all provided fields, defaults importance to 1 and privacy_tier to "normal", returns success with truncated contact ID.

4. **_search() (async static method)** – Queries contacts by name/organization using sm.search_contacts(query), returns up to 20 results formatted with importance score, organization, and ID. Validates query parameter is non-empty.

5. **_info() (async static method)** – Retrieves full contact details via sm.select_contact(), formats comprehensive output including position, organization, relation, email, phone, importance (with pinned flag), mention count, last_seen_at, tags, description, notes, and privacy tier.

6. **_update() (async static method)** – Selectively updates contact fields (only non-None parameters), clamps importance to 1-10 range and sets importance_pinned flag when importance is provided, appends notes to existing content rather than replacing, validates contact exists before updating.

## Architecture & Dependencies

- **Dependencies**: captain_claw.logging (get_logger), captain_claw.session (get_session_manager), captain_claw.tools.registry (Tool, ToolResult)
- **Session Manager Integration**: All operations delegate to session manager (sm) methods: create_contact, list_contacts, search_contacts, select_contact, update_contact, delete_contact
- **Data Model**: Contact objects expose attributes: id, name, position, organization, relation, email, phone, importance, importance_pinned, mention_count, last_seen_at, tags, description, notes, privacy_tier
- **Design Pattern**: Static methods for each action handler promote testability and reduce state coupling; async/await throughout for non-blocking I/O
- **Error Handling**: Graceful validation with ToolResult(success=False, error=...) for missing parameters, not-found contacts, and operation failures