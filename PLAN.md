# To-Do Memory — Implementation Plan

Cross-session, persistent to-do list that the agent can read/write as a pseudo-tool, with auto-capture heuristics and context injection.

---

## 1. Config Section — `config.py` + `config.yaml`

**File:** `captain_claw/config.py`

Add `TodoConfig` pydantic model and register it on the `Config` class:

```python
class TodoConfig(BaseModel):
    enabled: bool = True
    auto_capture: bool = True          # keyword-pattern auto-capture
    inject_on_session_load: bool = True # inject summary into context
    max_items_in_prompt: int = 10       # cap injected items
    archive_after_days: int = 30        # auto-archive completed items
```

Add `todo: TodoConfig = Field(default_factory=TodoConfig)` to the `Config` class.

**File:** `config.yaml` — add default `todo:` section after `orchestrator:`.

---

## 2. Database Layer — `session/__init__.py`

**New table** in `_ensure_db()`:

```sql
CREATE TABLE IF NOT EXISTS todo_items (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    responsible     TEXT NOT NULL DEFAULT 'bot',
    priority        TEXT NOT NULL DEFAULT 'normal',
    source_session  TEXT,
    target_session  TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    completed_at    TEXT,
    context         TEXT,
    tags            TEXT
)
```

Indices: `idx_todo_status` on `(status)`, `idx_todo_responsible_status` on `(responsible, status)`.

**New dataclass:** `TodoItem` with `from_row()` classmethod (same pattern as `CronJob`).

**New SessionManager methods:**
- `create_todo(content, responsible, priority, source_session, target_session, context, tags) → TodoItem`
- `load_todo(todo_id) → TodoItem | None`
- `select_todo(selector) → TodoItem | None` — by id, name search, or #index
- `list_todos(limit, status_filter, responsible_filter, session_filter) → list[TodoItem]`
- `update_todo(todo_id, *, status, responsible, priority, target_session, content, tags) → bool`
- `delete_todo(todo_id) → bool`
- `archive_old_todos(days) → int` — mark done items older than N days as archived
- `get_todo_summary(session_id, max_items) → list[TodoItem]` — items relevant to a session: own items + global (target_session=NULL) pending/in_progress items, ordered by priority then created_at

---

## 3. Todo Tool — `tools/todo.py`

New file: `captain_claw/tools/todo.py`

Extends `Tool` base class. Registered as `"todo"` tool.

**Parameters (JSON schema):**
```json
{
  "type": "object",
  "properties": {
    "action": {
      "type": "string",
      "enum": ["add", "list", "update", "remove"],
      "description": "Operation to perform"
    },
    "content": { "type": "string", "description": "Task description (for add)" },
    "todo_id": { "type": "string", "description": "Todo ID or #index (for update/remove)" },
    "status": { "type": "string", "enum": ["pending", "in_progress", "done", "cancelled"] },
    "responsible": { "type": "string", "enum": ["bot", "human"] },
    "priority": { "type": "string", "enum": ["low", "normal", "high", "urgent"] },
    "tags": { "type": "string", "description": "Comma-separated tags" },
    "filter_status": { "type": "string", "description": "Filter by status (for list)" },
    "filter_responsible": { "type": "string", "description": "Filter by responsible (for list)" }
  },
  "required": ["action"]
}
```

**execute() logic:**
- `add`: calls `session_manager.create_todo()`, uses `_session_id` from kwargs for source_session
- `list`: calls `session_manager.list_todos()` with filters, formats as text table
- `update`: calls `session_manager.update_todo()` with provided fields
- `remove`: calls `session_manager.delete_todo()`

Returns `ToolResult` in all cases.

---

## 4. Tool Registration — `tools/__init__.py` + `agent_context_mixin.py`

**File:** `captain_claw/tools/__init__.py` — add import and export of `TodoTool`.

**File:** `captain_claw/agent_context_mixin.py` — in `_register_tools()`, add `todo` to the registration chain:
```python
elif tool_name == "todo":
    self.tools.register(TodoTool())
```

**File:** `config.yaml` — add `"todo"` to `tools.enabled` list.

---

## 5. Context Injection — `agent_context_mixin.py`

**New method** `_build_todo_context_note(self) → str`:
- If `config.todo.inject_on_session_load` is False, return empty
- Call `session_manager.get_todo_summary(session_id, max_items)`
- Format as compact text block:
  ```
  Active to-do items:
  #1 [high/bot] Implement caching layer (pending)
  #2 [normal/human] Review PR #42 (pending)
  You have a "todo" tool available to manage these items.
  ```
- Return empty string if no items

**Injection point** in `_build_messages()` — after the `list_task_note` block (line 765), add:

```python
todo_note = self._build_todo_context_note()
if todo_note:
    candidate_messages.append({
        "role": "assistant",
        "content": todo_note,
        "tool_name": "todo_context",
        "token_count": self._count_tokens(todo_note),
    })
```

Also track in `last_context_window`: `"todo_note_used": 1 if todo_note else 0`.

---

## 6. Auto-Capture — `agent_context_mixin.py` or new dedicated method

**New method** `_auto_capture_todos(self, user_message: str, assistant_response: str) → None`:

Conservative keyword-pattern matching on the **user message**:
- `remind me to ...`
- `don't forget to ...`
- `todo: ...` / `TODO: ...`
- `save this to todo ...`
- `add to todo ...`
- `we need to ...` (only with explicit "todo" or "save" qualifier)
- `I should ...` (only with explicit "todo" or "save" qualifier)

For matched patterns:
- Extract the task text after the trigger phrase
- Create todo with `responsible="human"` for user-stated tasks
- Create todo with `responsible="bot"` when agent says "I'll do X later" / deferred work

Conservative keyword-pattern matching on the **assistant response** (only when the agent explicitly defers):
- `I'll handle that later`
- `I can't do X because Y` → create blocker todo for human
- `after you provide X` → create todo for human

**Call site:** after each completed turn in the agent pipeline, gated by `config.todo.auto_capture`.

---

## 7. `/todo` Slash Command — CLI + Command Dispatch

**File:** `captain_claw/cli.py`

Add `/todo` to the autocomplete list and help text. Parser produces:
- `/todo` or `/todo list` → `"TODO_LIST"`
- `/todo add <text>` → `"TODO_ADD:<json>"`
- `/todo done <id|#index>` → `"TODO_DONE:<selector>"`
- `/todo remove <id|#index>` → `"TODO_REMOVE:<selector>"`
- `/todo assign bot|human <id|#index>` → `"TODO_ASSIGN:<json>"`

**File:** `captain_claw/local_command_dispatch.py`

Handle `TODO_LIST`, `TODO_ADD:`, `TODO_DONE:`, `TODO_REMOVE:`, `TODO_ASSIGN:` — following the exact `/cron` dispatch pattern. Use `ui.print_message()` for output formatting.

---

## 8. Web API — `web_server.py`

REST endpoints (following `/api/cron/jobs` pattern):

- `GET /api/todos` — list todos (query params: status, responsible, session_id)
- `POST /api/todos` — create todo
- `PATCH /api/todos/{id}` — update todo (status, priority, responsible, content, tags)
- `DELETE /api/todos/{id}` — delete todo

Add to command palette definitions at the top of the file.

Route registration in `_setup_routes()` alongside cron routes.

---

## Implementation Order

1. **Config** (`config.py`, `config.yaml`)
2. **DB schema + dataclass + SessionManager methods** (`session/__init__.py`)
3. **Todo tool** (`tools/todo.py`, `tools/__init__.py`)
4. **Tool registration** (`agent_context_mixin.py` registration block)
5. **Context injection** (`agent_context_mixin.py` `_build_messages` + new `_build_todo_context_note`)
6. **Auto-capture** (new method in `agent_context_mixin.py`, hook in pipeline)
7. **CLI slash command** (`cli.py` parser + `local_command_dispatch.py` handlers)
8. **Web API** (`web_server.py` routes + handlers)

Each step is independently testable. Steps 1-5 give a fully working system (tool + context injection). Steps 6-8 add polish.
