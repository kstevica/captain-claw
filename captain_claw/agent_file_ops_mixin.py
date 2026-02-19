"""File/script output helpers for Agent."""

from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shlex
import sys
from typing import Any

from captain_claw.llm import Message
from captain_claw.logging import get_logger

log = get_logger(__name__)

_STRUCTURED_RESULT_MARKER = "# captain-claw: structured-result-protocol-v1"

class AgentFileOpsMixin:
    @staticmethod
    def _extract_code_block(text: str) -> tuple[str | None, str | None]:
        """Extract first fenced code block as (language, code)."""
        if not text:
            return None, None
        match = re.search(r"```([A-Za-z0-9_+\-]*)\n(.*?)```", text, flags=re.DOTALL)
        if not match:
            return None, None
        language = (match.group(1) or "").strip().lower() or None
        code = (match.group(2) or "").strip()
        if not code:
            return language, None
        return language, code

    @staticmethod
    def _infer_script_extension(language: str | None, code: str) -> str:
        """Infer script extension from language tag or content."""
        lang = (language or "").strip().lower()
        mapping = {
            "python": ".py",
            "py": ".py",
            "bash": ".sh",
            "sh": ".sh",
            "shell": ".sh",
            "zsh": ".sh",
            "javascript": ".js",
            "js": ".js",
            "node": ".js",
            "ruby": ".rb",
            "rb": ".rb",
        }
        if lang in mapping:
            return mapping[lang]
        stripped = (code or "").lstrip()
        if stripped.startswith("#!/usr/bin/env python") or stripped.startswith("#!/usr/bin/python"):
            return ".py"
        if stripped.startswith("#!/usr/bin/env bash") or stripped.startswith("#!/bin/bash"):
            return ".sh"
        if stripped.startswith("#!/usr/bin/env sh") or stripped.startswith("#!/bin/sh"):
            return ".sh"
        return ".py"

    @staticmethod
    def _supported_script_extension(ext: str) -> bool:
        """Return whether extension can be executed with built-in runner mapping."""
        return (ext or "").lower() in {".py", ".sh", ".js", ".rb"}

    @staticmethod
    def _build_script_runner_command(script_path: Path) -> str | None:
        """Build shell command that runs script from its own directory."""
        ext = script_path.suffix.lower()
        filename = shlex.quote(script_path.name)
        if ext == ".py":
            runner = f"python3 {filename}"
        elif ext == ".sh":
            runner = f"bash {filename}"
        elif ext == ".js":
            runner = f"node {filename}"
        elif ext == ".rb":
            runner = f"ruby {filename}"
        else:
            return None
        script_dir = shlex.quote(str(script_path.parent))
        return f"cd {script_dir} && {runner}"

    @staticmethod
    def _build_python_runner_command(script_path: Path, result_path: Path | None = None) -> str:
        """Build shell command using the active Python interpreter."""
        interpreter = shlex.quote(str(Path(sys.executable).resolve()))
        script_dir = shlex.quote(str(script_path.parent))
        filename = shlex.quote(script_path.name)
        result_arg = f" {shlex.quote(str(result_path))}" if result_path is not None else ""
        return f"cd {script_dir} && {interpreter} {filename}{result_arg}"

    @staticmethod
    def _normalize_session_slug(raw_id: str | None) -> str:
        """Normalize arbitrary id into safe folder slug."""
        normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(raw_id or "").strip()).strip("-")
        return normalized or "default"

    @staticmethod
    def _build_structured_result_wrapper(code: str) -> str:
        """Wrap Python code so execution writes structured JSON result payload."""
        raw = (code or "").strip()
        if not raw:
            raw = "def main():\n    return {}\n"
        if _STRUCTURED_RESULT_MARKER in raw:
            return raw if raw.endswith("\n") else f"{raw}\n"
        encoded_code = json.dumps(raw, ensure_ascii=True)
        return (
            "#!/usr/bin/env python3\n"
            "\"\"\"Captain Claw structured script wrapper.\"\"\"\n\n"
            "import contextlib\n"
            "import io\n"
            "import json\n"
            "import pathlib\n"
            "import sys\n\n"
            f"{_STRUCTURED_RESULT_MARKER}\n"
            f"_CAPTAIN_CLAW_USER_CODE = {encoded_code}\n\n"
            "def _captain_claw_user_main() -> object:\n"
            "    user_globals = {\n"
            "        '__name__': '__captain_claw_user_script__',\n"
            "        '__file__': __file__,\n"
            "    }\n"
            "    exec(compile(_CAPTAIN_CLAW_USER_CODE, __file__, 'exec'), user_globals, user_globals)\n"
            "    main_fn = user_globals.get('main')\n"
            "    if callable(main_fn):\n"
            "        return main_fn()\n"
            "    return user_globals.get('RESULT')\n\n"
            "def main() -> int:\n"
            "    result_path = pathlib.Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None\n"
            "    payload: dict[str, object] = {'success': True, 'data': None}\n"
            "    stdout_buffer = io.StringIO()\n"
            "    try:\n"
            "        with contextlib.redirect_stdout(stdout_buffer):\n"
            "            data = _captain_claw_user_main()\n"
            "        if data is None:\n"
            "            captured = stdout_buffer.getvalue().strip()\n"
            "            payload['data'] = {'stdout': captured} if captured else {}\n"
            "        else:\n"
            "            payload['data'] = data\n"
            "    except Exception as exc:\n"
            "        payload = {'success': False, 'error': str(exc)}\n"
            "    if result_path is not None:\n"
            "        result_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "        result_path.write_text(json.dumps(payload), encoding='utf-8')\n"
            "    return 0 if bool(payload.get('success')) else 1\n\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n"
        )

    def _build_structured_result_path(
        self,
        script_path: Path,
        *,
        session_slug: str | None = None,
    ) -> Path:
        """Build deterministic JSON result path for script execution payloads."""
        slug = self._normalize_session_slug(session_slug or self._current_session_slug())
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        result_name = f"{script_path.stem}_{stamp}.result.json"
        return (
            self.tools.get_saved_base_path(create=True)
            / "tmp"
            / slug
            / "script_results"
            / result_name
        )

    @staticmethod
    def _read_structured_result_payload(path: Path) -> dict[str, Any] | None:
        """Load structured script result payload from JSON file."""
        try:
            if not path.is_file():
                return None
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                return None
            return parsed
        except Exception:
            return None

    def _current_session_slug(self) -> str:
        """Return normalized session id used for folder scoping."""
        session_key = "default"
        if self.session and self.session.id:
            session_key = self._normalize_session_slug(str(self.session.id))
        return session_key

    def _build_script_relative_path(
        self,
        user_input: str,
        extension: str,
        *,
        session_slug: str | None = None,
    ) -> str:
        """Build script path under scripts/<session-id>/."""
        ext = extension if extension.startswith(".") else f".{extension}"
        requested = self._extract_requested_write_path(user_input)
        filename: str
        if requested:
            candidate = Path(requested).name
            if "." in candidate:
                base = Path(candidate).stem or "generated_script"
                req_ext = Path(candidate).suffix.lower()
                filename = base + (req_ext if self._supported_script_extension(req_ext) else ext)
            else:
                filename = candidate + ext
        else:
            stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            filename = f"generated_script_{stamp}{ext}"
        return f"scripts/{self._normalize_session_slug(session_slug or self._current_session_slug())}/{filename}"

    @staticmethod
    def _parse_written_path_from_tool_output(tool_output: str) -> Path | None:
        """Parse final path from write tool output."""
        text = (tool_output or "").strip()
        match = re.search(r"\bto\s+(.+?)(?:\s+\(requested:|\s*$)", text)
        if not match:
            return None
        raw_path = match.group(1).strip()
        if not raw_path:
            return None
        try:
            return Path(raw_path)
        except Exception:
            return None

    async def _synthesize_script_content(
        self,
        user_input: str,
        turn_usage: dict[str, int],
    ) -> tuple[str, str]:
        """Generate script content when model answer omitted code block."""
        synth_messages = [
            Message(
                role="system",
                content=self.instructions.load("script_synthesis_system_prompt.md"),
            ),
            Message(role="user", content=user_input),
        ]
        try:
            self._set_runtime_status("thinking")
            response = await self._complete_with_guards(
                messages=synth_messages,
                tools=None,
                interaction_label="script_synthesis",
                turn_usage=turn_usage,
            )
            language, code = self._extract_code_block(response.content or "")
            if code:
                return code, self._infer_script_extension(language, code)
            raw = (response.content or "").strip()
            if raw:
                return raw, self._infer_script_extension(language, raw)
        except Exception as e:
            log.warning("Script synthesis fallback failed", error=str(e))

        # Deterministic fallback scaffold.
        safe_request = re.sub(r"\s+", " ", (user_input or "").strip())[:240]
        scaffold = (
            "#!/usr/bin/env python3\n"
            "\"\"\"Auto-generated script scaffold.\"\"\"\n\n"
            "def main() -> None:\n"
            f"    print({safe_request!r})\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        return scaffold, ".py"

    def _build_python_worker_prompt(
        self,
        user_input: str,
        list_task_plan: dict[str, Any] | None = None,
        *,
        session_slug: str | None = None,
    ) -> str:
        """Build synthesis prompt for batch/list worker script generation."""
        session_id = self._normalize_session_slug(session_slug or self._current_session_slug())
        members_block = ""
        per_member_action = ""
        if isinstance(list_task_plan, dict):
            members = list_task_plan.get("members")
            if isinstance(members, list) and members:
                rendered = "\n".join(f"- {str(item)}" for item in members[:80])
                members_block = f"\nList members to process:\n{rendered}\n"
            per_member_action = str(list_task_plan.get("per_member_action", "")).strip()
        return (
            "Create one runnable Python 3 script for this task.\n"
            "Requirements:\n"
            "- Complete the full task end-to-end.\n"
            "- If a list of entities/items is discovered, iterate ALL items; never stop after the first item.\n"
            "- Use text extraction by default when parsing fetched pages.\n"
            f"- Save per-item outputs under saved/showcase/{session_id}/.\n"
            "- Use deterministic filenames based on item names where requested.\n"
            "- Print concise progress logs so monitor output shows progress.\n"
            "- Define `main()` and return a JSON-serializable summary object for downstream steps.\n"
            f"- Per-member action focus: {per_member_action or 'Follow user request for each member.'}\n"
            "- Return code only.\n\n"
            f"{members_block}"
            f"User request:\n{user_input}"
        )

    async def _run_python_worker_for_list_task(
        self,
        user_input: str,
        turn_usage: dict[str, int],
        list_task_plan: dict[str, Any] | None = None,
        planning_pipeline: dict[str, Any] | None = None,
        session_policy: dict[str, Any] | None = None,
        task_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate/write/run a temporary Python worker for list-style tasks."""
        execution_context: dict[str, Any] | None = None
        active_task_node: dict[str, Any] | None = None
        active_task_id = ""
        active_context = self._resolve_active_execution_context(planning_pipeline)
        if active_context is not None:
            active_task_id, active_task_node, execution_context = active_context
        child_session_id = str((execution_context or {}).get("session_id", "")).strip()
        child_session_slug = self._normalize_session_slug(child_session_id) if child_session_id else ""
        target_session_slug = child_session_slug or self._current_session_slug()

        effective_task_policy: dict[str, Any] | None = dict(task_policy or {})
        allowlist = (execution_context or {}).get("tool_allowlist", [])
        if isinstance(allowlist, list) and allowlist:
            effective_task_policy["allow"] = [str(item).strip() for item in allowlist if str(item).strip()]
        if not effective_task_policy:
            effective_task_policy = task_policy

        worker_prompt = self._build_python_worker_prompt(
            user_input,
            list_task_plan=list_task_plan,
            session_slug=target_session_slug,
        )
        code, extension = await self._synthesize_script_content(worker_prompt, turn_usage)
        if extension.lower() != ".py":
            extension = ".py"
        code = self._build_structured_result_wrapper(code)
        script_rel_path = self._build_script_relative_path(
            f"generate script {datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}{extension}",
            extension,
            session_slug=target_session_slug,
        ).replace("scripts/", "tools/", 1)
        write_state = await self._write_file_with_verification(
            path=script_rel_path,
            content=code,
            turn_usage=turn_usage,
            interaction_label="list_worker_write",
            max_attempts=2,
            session_policy=session_policy,
            task_policy=effective_task_policy,
            session_id_override=child_session_id or None,
        )
        if not bool(write_state.get("success", False)):
            return {"success": False, "step": "write_failed", "error": str(write_state.get("output", ""))}

        written_script_path = Path(str(write_state.get("path", script_rel_path)))
        result_path = self._build_structured_result_path(
            written_script_path,
            session_slug=target_session_slug,
        )
        run_command = self._build_python_runner_command(written_script_path, result_path=result_path)
        shell_arguments: dict[str, Any] = {"command": run_command}
        timeout_seconds = (execution_context or {}).get("timeout_seconds")
        if timeout_seconds is not None:
            try:
                shell_arguments["timeout"] = int(float(timeout_seconds))
            except Exception:
                pass
        try:
            shell_result = await self._execute_tool_with_guard(
                name="shell",
                arguments=shell_arguments,
                interaction_label="list_worker_run",
                turn_usage=turn_usage,
                session_policy=session_policy,
                task_policy=effective_task_policy,
                session_id_override=child_session_id or None,
            )
            shell_output = shell_result.content if shell_result.success else f"Error: {shell_result.error}"
        except Exception as e:
            shell_result = None
            shell_output = f"Error: {str(e)}"

        structured_result = self._read_structured_result_payload(result_path)
        if structured_result is None:
            structured_result = {
                "success": bool(shell_result and shell_result.success),
                "data": {"stdout": shell_output},
            }
        structured_data = structured_result.get("data")
        if isinstance(structured_data, dict):
            captured_stdout = str(structured_data.get("stdout", "")).strip()
            if captured_stdout:
                shell_output = f"{shell_output}\n[structured_stdout]\n{captured_stdout}".strip()
            elif structured_data:
                try:
                    compact = json.dumps(structured_data, ensure_ascii=True)
                except Exception:
                    compact = str(structured_data)
                shell_output = f"{shell_output}\n[structured_data] {compact}".strip()
        elif structured_data is not None and structured_data != "":
            shell_output = f"{shell_output}\n[structured_data] {structured_data}".strip()

        self._add_session_message(
            role="tool",
            content=shell_output,
            tool_name="shell",
            tool_arguments={"command": run_command},
        )
        self._emit_tool_output("shell", {"command": run_command}, shell_output)

        if execution_context is not None:
            artifacts = execution_context.setdefault("artifacts", [])
            if isinstance(artifacts, list):
                script_artifact = str(written_script_path)
                result_artifact = str(result_path)
                if script_artifact not in artifacts:
                    artifacts.append(script_artifact)
                if result_artifact not in artifacts:
                    artifacts.append(result_artifact)
            variables = execution_context.setdefault("variables", {})
            if isinstance(variables, dict):
                if bool(structured_result.get("success", False)):
                    variables["output"] = structured_result.get("data")
                else:
                    variables["output_error"] = structured_result.get("error")
            self._record_execution_context_event(
                execution_context,
                "python_worker_run",
                {
                    "task_id": active_task_id,
                    "script_path": str(written_script_path),
                    "result_path": str(result_path),
                    "success": bool(structured_result.get("success", False)),
                },
            )
            if isinstance(active_task_node, dict):
                active_task_node["result"] = {
                    "success": bool(structured_result.get("success", False)),
                    "data": structured_result.get("data"),
                    "error": str(structured_result.get("error", "") or ""),
                    "artifacts": list(execution_context.get("artifacts", [])),
                    "completed_at": datetime.now(UTC).isoformat(),
                }

        if not bool(structured_result.get("success", False)):
            return {
                "success": False,
                "step": "run_failed",
                "path": str(written_script_path),
                "error": str(structured_result.get("error", shell_output)),
                "result_path": str(result_path),
                "result": structured_result,
            }
        return {
            "success": True,
            "step": "completed",
            "path": str(written_script_path),
            "result_path": str(result_path),
            "result": structured_result,
        }

    async def _maybe_auto_script_requested_output(
        self,
        user_input: str,
        output_text: str,
        turn_start_idx: int,
        turn_usage: dict[str, int],
        session_policy: dict[str, Any] | None = None,
        task_policy: dict[str, Any] | None = None,
    ) -> str:
        """Guarantee explicit script requests produce write+run tool actions."""
        text = (output_text or "").strip()
        if not self._is_explicit_script_request(user_input):
            return text

        has_write = self._turn_has_successful_tool(turn_start_idx, "write")
        has_shell = self._turn_has_successful_tool(turn_start_idx, "shell")
        if has_write and has_shell:
            return text

        written_script_path: Path | None = None

        if not has_write:
            language, code = self._extract_code_block(text)
            if not code:
                code, inferred_ext = await self._synthesize_script_content(user_input, turn_usage)
            else:
                inferred_ext = self._infer_script_extension(language, code)
            if inferred_ext.lower() == ".py":
                code = self._build_structured_result_wrapper(code)

            script_rel_path = self._build_script_relative_path(user_input, inferred_ext)
            write_state = await self._write_file_with_verification(
                path=script_rel_path,
                content=code,
                turn_usage=turn_usage,
                interaction_label="auto_script_write",
                max_attempts=2,
                session_policy=session_policy,
                task_policy=task_policy,
            )
            if not bool(write_state.get("success", False)):
                return (
                    f"{text}\n\n"
                    "Note: explicit script request could not be completed because write failed.\n"
                    f"{str(write_state.get('output', ''))}"
                ).strip()

            written_script_path = Path(str(write_state.get("path", script_rel_path)))
        else:
            # Reuse existing write result path from this turn when available.
            if self.session:
                for msg in reversed(self.session.messages[turn_start_idx:]):
                    if msg.get("role") != "tool" or str(msg.get("tool_name")) != "write":
                        continue
                    content = str(msg.get("content", "")).strip()
                    if content.lower().startswith("error:"):
                        continue
                    parsed = self._parse_written_path_from_tool_output(content)
                    if parsed:
                        written_script_path = parsed
                        break

        if has_shell:
            return text

        if not written_script_path:
            return (
                f"{text}\n\n"
                "Note: script was requested but executable path could not be resolved for run."
            ).strip()

        result_path: Path | None = None
        if written_script_path.suffix.lower() == ".py":
            result_path = self._build_structured_result_path(written_script_path)
            run_command = self._build_python_runner_command(written_script_path, result_path=result_path)
        else:
            run_command = self._build_script_runner_command(written_script_path)
        if not run_command:
            return (
                f"{text}\n\n"
                f"Note: script saved to {written_script_path}, but auto-run is unsupported for extension "
                f"'{written_script_path.suffix or '(none)'}'."
            ).strip()

        try:
            shell_result = await self._execute_tool_with_guard(
                name="shell",
                arguments={"command": run_command},
                interaction_label="auto_script_run",
                turn_usage=turn_usage,
                session_policy=session_policy,
                task_policy=task_policy,
            )
            shell_output = (
                shell_result.content if shell_result.success else f"Error: {shell_result.error}"
            )
        except Exception as e:
            shell_result = None
            shell_output = f"Error: {str(e)}"

        structured_result = (
            self._read_structured_result_payload(result_path) if result_path is not None else None
        )
        if isinstance(structured_result, dict):
            structured_data = structured_result.get("data")
            if isinstance(structured_data, dict):
                captured_stdout = str(structured_data.get("stdout", "")).strip()
                if captured_stdout:
                    shell_output = f"{shell_output}\n[structured_stdout]\n{captured_stdout}".strip()
                elif structured_data:
                    try:
                        compact = json.dumps(structured_data, ensure_ascii=True)
                    except Exception:
                        compact = str(structured_data)
                    shell_output = f"{shell_output}\n[structured_data] {compact}".strip()
            elif structured_data is not None and structured_data != "":
                shell_output = f"{shell_output}\n[structured_data] {structured_data}".strip()

        self._add_session_message(
            role="tool",
            content=shell_output,
            tool_name="shell",
            tool_arguments={"command": run_command},
        )
        self._emit_tool_output("shell", {"command": run_command}, shell_output)

        success = bool(shell_result and shell_result.success)
        if isinstance(structured_result, dict):
            success = bool(structured_result.get("success", False))

        if success:
            structured_suffix = ""
            if isinstance(structured_result, dict):
                structured_suffix = (
                    f"\nStructured result saved to {result_path}."
                    if result_path is not None
                    else ""
                )
            return (
                f"{text}\n\n"
                f"Script saved and executed from {written_script_path.parent}.{structured_suffix}"
            ).strip()

        structured_error = ""
        if isinstance(structured_result, dict):
            structured_error = str(structured_result.get("error", "")).strip()
        return (
            f"{text}\n\n"
            f"Script saved to {written_script_path}, but execution failed.\n"
            f"{structured_error or shell_output}"
        ).strip()

    @staticmethod
    def _extract_requested_write_paths(user_input: str) -> list[str]:
        """Extract requested target file paths from user input."""
        text = (user_input or "").strip()
        if not text:
            return []
        if not re.search(r"\b(write|save|store|export|dump)\b", text, flags=re.IGNORECASE):
            return []

        patterns = [
            r"(?:to|into|as)\s+(?:a\s+)?(?:file\s+)?[`\"']?([A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,16})",
            r"(?:file\s+|named\s+|name\s+it\s+)[`\"']?([A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,16})",
        ]
        seen: set[str] = set()
        ordered: list[str] = []
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if not matches:
                continue
            for match in matches:
                candidate = match.strip().rstrip(".,;:!?)]}>")
                lowered = candidate.lower()
                if lowered.startswith(("http://", "https://")):
                    continue
                if not candidate:
                    continue
                key = lowered
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(candidate)
        return ordered

    @staticmethod
    def _extract_requested_write_path(user_input: str) -> str | None:
        """Extract last requested target file path from user input."""
        paths = AgentFileOpsMixin._extract_requested_write_paths(user_input)
        if not paths:
            return None
        return paths[-1]

    @staticmethod
    def _is_explicit_file_save_request(user_input: str) -> bool:
        """Detect whether user explicitly asked for file creation/saving."""
        text = (user_input or "").strip().lower()
        if not text:
            return False
        has_write_action = bool(re.search(r"\b(write|save|store|export|dump|create)\b", text))
        has_file_target = bool(re.search(r"\bfile\b|\bfiles\b", text))
        return has_write_action and has_file_target

    @staticmethod
    def _extract_named_file_blocks(output_text: str) -> list[tuple[str, str]]:
        """Extract `(filename, content)` pairs from assistant text blocks."""
        text = (output_text or "").strip()
        if not text:
            return []
        header_re = re.compile(r"^\s*filename\s*:\s*(.+?)\s*$", flags=re.IGNORECASE | re.MULTILINE)
        matches = list(header_re.finditer(text))
        if not matches:
            return []

        blocks: list[tuple[str, str]] = []
        seen_paths: set[str] = set()
        for idx, match in enumerate(matches):
            raw_path = (match.group(1) or "").strip().strip("`\"'")
            raw_path = raw_path.rstrip(".,;:!?)]}>")
            if not raw_path:
                continue
            if not re.search(r"\.[A-Za-z0-9]{1,16}$", raw_path):
                continue

            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            chunk = text[start:end].strip()
            if not chunk:
                continue
            chunk_lines = chunk.splitlines()
            if chunk_lines and chunk_lines[0].strip() == "---":
                chunk = "\n".join(chunk_lines[1:]).strip()
            if not chunk:
                continue
            key = raw_path.lower()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            blocks.append((raw_path, chunk))
        return blocks

    def _collect_recent_file_blocks_from_session(
        self,
        max_assistant_messages: int = 12,
    ) -> list[tuple[str, str]]:
        """Collect most recent assistant `Filename:` blocks from session history."""
        if not self.session:
            return []

        checked = 0
        for msg in reversed(self.session.messages):
            if str(msg.get("role", "")).strip().lower() != "assistant":
                continue
            checked += 1
            blocks = self._extract_named_file_blocks(str(msg.get("content", "")))
            if blocks:
                return blocks
            if checked >= max_assistant_messages:
                break
        return []

    def _candidate_write_paths_for_verification(
        self,
        requested_path: str,
        parsed_written: Path | None = None,
    ) -> list[Path]:
        """Build candidate on-disk paths to verify file was actually saved."""
        candidates: list[Path] = []
        seen: set[str] = set()

        def _add(path: Path) -> None:
            try:
                resolved = path.expanduser().resolve()
            except Exception:
                return
            key = str(resolved)
            if key in seen:
                return
            seen.add(key)
            candidates.append(resolved)

        if parsed_written is not None:
            _add(parsed_written)

        raw = str(requested_path or "").strip()
        if not raw:
            return candidates

        requested = Path(raw).expanduser()
        if requested.is_absolute():
            _add(requested)
        else:
            _add(self.runtime_base_path / requested)
            _add(self.tools.get_saved_base_path(create=False) / requested)
            parts = requested.parts
            if parts and parts[0].lower() == "saved":
                tail = Path(*parts[1:]) if len(parts) > 1 else Path("output.txt")
                _add(self.tools.get_saved_base_path(create=False) / tail)
        return candidates

    async def _write_file_with_verification(
        self,
        path: str,
        content: str,
        turn_usage: dict[str, int],
        interaction_label: str,
        max_attempts: int = 2,
        session_policy: dict[str, Any] | None = None,
        task_policy: dict[str, Any] | None = None,
        session_id_override: str | None = None,
    ) -> dict[str, Any]:
        """Write file, verify it exists, and retry once when not persisted."""
        attempts = max(1, int(max_attempts))
        last_output = ""
        last_written_path: Path | None = None

        for attempt in range(1, attempts + 1):
            write_args = {"path": path, "content": content}
            write_args_for_log = {
                "path": path,
                "content_chars": len(content),
                "append": False,
                "attempt": attempt,
            }
            try:
                result = await self._execute_tool_with_guard(
                    name="write",
                    arguments=write_args,
                    interaction_label=interaction_label,
                    turn_usage=turn_usage,
                    session_policy=session_policy,
                    task_policy=task_policy,
                    session_id_override=session_id_override,
                )
                tool_output = result.content if result.success else f"Error: {result.error}"
            except Exception as e:
                result = None
                tool_output = f"Error: {str(e)}"

            self._add_session_message(
                role="tool",
                content=tool_output,
                tool_name="write",
                tool_arguments=write_args_for_log,
            )
            self._emit_tool_output("write", write_args_for_log, tool_output)

            parsed_written = self._parse_written_path_from_tool_output(tool_output)
            if parsed_written is not None:
                last_written_path = parsed_written

            verified_path: Path | None = None
            for candidate in self._candidate_write_paths_for_verification(path, parsed_written):
                try:
                    if candidate.is_file():
                        verified_path = candidate
                        break
                except Exception:
                    continue

            if verified_path is not None:
                return {
                    "success": True,
                    "attempts": attempt,
                    "output": tool_output,
                    "path": str(verified_path),
                }

            last_output = tool_output
            if attempt < attempts:
                self._emit_tool_output(
                    "completion_gate",
                    {
                        "step": "write_verify_retry",
                        "path": path,
                        "attempt": attempt + 1,
                    },
                    (
                        f"step=write_verify_retry\n"
                        f"path={path}\n"
                        f"attempt={attempt + 1}\n"
                        "reason=file_missing_after_write"
                    ),
                )

        return {
            "success": False,
            "attempts": attempts,
            "output": last_output or "Error: write verification failed",
            "path": str(last_written_path) if last_written_path is not None else path,
        }

    async def _maybe_auto_write_requested_output(
        self,
        user_input: str,
        output_text: str,
        turn_start_idx: int,
        turn_usage: dict[str, int],
        session_policy: dict[str, Any] | None = None,
        task_policy: dict[str, Any] | None = None,
    ) -> str:
        """Auto-run write tool when user explicitly requested file output."""
        text = (output_text or "").strip()
        if self._is_explicit_script_request(user_input):
            return await self._maybe_auto_script_requested_output(
                user_input=user_input,
                output_text=text,
                turn_start_idx=turn_start_idx,
                turn_usage=turn_usage,
                session_policy=session_policy,
                task_policy=task_policy,
            )
        requested_paths = self._extract_requested_write_paths(user_input)
        requested_path = requested_paths[-1] if requested_paths else None
        explicit_file_request = self._is_explicit_file_save_request(user_input)

        file_blocks = self._extract_named_file_blocks(text)
        if (
            not file_blocks
            and explicit_file_request
            and re.search(r"\b(all|those)\s+files\b", user_input, flags=re.IGNORECASE)
        ):
            file_blocks = self._collect_recent_file_blocks_from_session()

        if file_blocks:
            saved_entries: list[str] = []
            failed_entries: list[str] = []
            for raw_path, raw_content in file_blocks:
                target_path = raw_path
                if "/" not in raw_path and "\\" not in raw_path:
                    target_path = f"showcase/{self._current_session_slug()}/{raw_path}"
                write_state = await self._write_file_with_verification(
                    path=target_path,
                    content=raw_content,
                    turn_usage=turn_usage,
                    interaction_label="auto_write_output_multi",
                    max_attempts=2,
                    session_policy=session_policy,
                    task_policy=task_policy,
                )
                tool_output = str(write_state.get("output", "")).strip()
                attempts = int(write_state.get("attempts", 1) or 1)
                if bool(write_state.get("success", False)):
                    retry_note = " (retried)" if attempts > 1 else ""
                    saved_entries.append(f"{raw_path}: {tool_output}{retry_note}")
                else:
                    failed_entries.append(f"{raw_path}: {tool_output}")

            if saved_entries and not failed_entries:
                summary = "\n".join(f"- {entry}" for entry in saved_entries)
                return f"{text}\n\nSaved {len(saved_entries)} files:\n{summary}".strip()
            if saved_entries or failed_entries:
                ok_summary = "\n".join(f"- {entry}" for entry in saved_entries) or "- (none)"
                fail_summary = "\n".join(f"- {entry}" for entry in failed_entries) or "- (none)"
                return (
                    f"{text}\n\n"
                    f"Auto file-save summary:\n"
                    f"Saved ({len(saved_entries)}):\n{ok_summary}\n"
                    f"Failed ({len(failed_entries)}):\n{fail_summary}"
                ).strip()
            return text

        if not requested_path:
            return text
        if self._turn_has_successful_tool(turn_start_idx, "write"):
            return text

        write_state = await self._write_file_with_verification(
            path=requested_path,
            content=text,
            turn_usage=turn_usage,
            interaction_label="auto_write_output",
            max_attempts=2,
            session_policy=session_policy,
            task_policy=task_policy,
        )
        tool_output = str(write_state.get("output", "")).strip()
        if bool(write_state.get("success", False)):
            return f"{text}\n\n{tool_output}".strip()
        return (
            f"{text}\n\n"
            f"Note: requested file save to '{requested_path}' failed.\n"
            f"{tool_output}"
        ).strip()

    async def _persist_assistant_response(self, content: str) -> None:
        """Persist assistant response for the current turn."""
        if not self.session:
            return
        self._add_session_message("assistant", content)
        await self.session_manager.save_session(self.session)
        memory = getattr(self, "memory", None)
        if memory is not None:
            memory.schedule_background_sync("assistant_saved")
