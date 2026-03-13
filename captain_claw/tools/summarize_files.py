"""Summarize files tool — batch file summarization with context-efficient LLM calls.

Reads all files in a folder, summarizes each one individually via internal LLM
calls, then combines all summaries into a final output.  All intermediate work
stays out of the main conversation context — only the output file path and
compact metadata are returned.
"""

from __future__ import annotations

import asyncio
import glob as glob_module
import time
from pathlib import Path
from typing import Any, Callable

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

# File extensions treated as readable text.
_TEXT_EXTENSIONS: set[str] = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".html",
    ".htm", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".log",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h",
    ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".bash", ".zsh",
    ".sql", ".r", ".m", ".swift", ".kt", ".scala", ".lua", ".pl",
    ".tex", ".rst", ".org", ".adoc", ".textile", ".wiki",
    ".css", ".scss", ".less", ".sass",
    ".env", ".gitignore", ".dockerfile",
}

# Supported document types with specialised extractors.
_DOCUMENT_EXTENSIONS: set[str] = {".pdf", ".docx", ".xlsx", ".pptx"}

# Maximum file size to attempt reading (10 MB).
_MAX_FILE_BYTES: int = 10_000_000

# Conservative per-call input char limit (~100k tokens ≈ 400k chars).
_MAX_INPUT_CHARS: int = 400_000


class SummarizeFilesTool(Tool):
    """Summarize all files in a folder and produce a combined summary."""

    name = "summarize_files"
    description = (
        "ALWAYS use this tool when the user asks to go through, review, analyse, "
        "or summarise documents/files in a folder. Handles the entire pipeline: "
        "reads all files (text, PDF, DOCX, XLSX, PPTX), summarises each via LLM, "
        "and combines into a final output saved to disk. Returns only the output "
        "file path and metadata — use 'read' to access the summary afterwards. "
        "Do NOT use pdf_extract/docx_extract/xlsx_extract individually when "
        "multiple files need processing — use this tool instead."
    )
    timeout_seconds = 3600.0  # 1 hour — large folders may have 100+ files × LLM calls

    parameters = {
        "type": "object",
        "properties": {
            "folder": {
                "type": "string",
                "description": (
                    "Path to the folder containing files to summarize."
                ),
            },
            "pattern": {
                "type": "string",
                "description": (
                    "Glob pattern to filter files (default: '**/*' — recursive). "
                    "Examples: '**/*.pdf', '*.txt' (top-level only), '**/*.md'"
                ),
            },
            "target_words": {
                "type": "number",
                "description": (
                    "Target word count for the final combined summary "
                    "(default: 1000)."
                ),
            },
            "summary_style": {
                "type": "string",
                "description": (
                    "Style for individual summaries: 'concise', 'detailed', "
                    "or 'bullet_points' (default: 'concise')."
                ),
            },
            "instructions": {
                "type": "string",
                "description": (
                    "Additional instructions for summarisation, e.g. "
                    "'focus on financial data', 'extract key findings'."
                ),
            },
        },
        "required": ["folder"],
    }

    def __init__(self) -> None:
        super().__init__()
        # Set by agent during registration for personality-aware summarisation.
        self._user_id: str | None = None

    def set_user_mode(self, user_id: str) -> None:
        """Switch to user-profile mode for personality-aware summarisation."""
        self._user_id = user_id

    # ── public entry point ────────────────────────────────────────

    async def execute(  # noqa: C901 — inherently sequential pipeline
        self,
        folder: str,
        pattern: str = "**/*",
        target_words: int = 1000,
        summary_style: str = "concise",
        instructions: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        runtime_base = kwargs.get("_runtime_base_path")
        saved_base = kwargs.get("_saved_base_path")
        session_id = kwargs.get("_session_id", "default")
        abort_event: asyncio.Event | None = kwargs.get("_abort_event")
        stream_cb: Callable[[str], None] | None = kwargs.get("_stream_callback")

        # 1. Resolve folder ────────────────────────────────────────
        folder_path = self._resolve_folder(folder, runtime_base)
        if folder_path is None or not folder_path.is_dir():
            return ToolResult(success=False, error=f"Folder not found: {folder}")

        # 2. Discover files ────────────────────────────────────────
        files = self._discover_files(folder_path, pattern)
        if not files:
            return ToolResult(
                success=False,
                error=f"No readable files matching '{pattern}' in {folder}",
            )

        self._log(stream_cb, f"📂 Found {len(files)} file(s) to summarise in {folder_path}")
        for f in files:
            self._log(stream_cb, f"   • {f.name} ({f.stat().st_size:,} bytes)")

        # 3. Prepare output directory ──────────────────────────────
        output_dir = self._build_output_dir(saved_base, runtime_base, session_id)

        # 4. Get LLM provider ─────────────────────────────────────
        try:
            from captain_claw.llm import Message, get_provider

            provider = get_provider()
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"LLM provider not available: {exc}",
            )

        # 5. Load personality context ──────────────────────────────
        personality_block = self._load_personality_context()
        if personality_block:
            self._log(stream_cb, "🎭 Using agent personality and user persona in summarisation")

        # 6. Token tracking setup ──────────────────────────────────
        total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        llm_calls = 0
        t_start = time.monotonic()

        # 7. Summarise each file ───────────────────────────────────
        summaries: list[tuple[str, str]] = []  # (filename, summary_text)
        total_chars_read = 0
        errors: list[str] = []

        for idx, file_path in enumerate(files):
            if abort_event and abort_event.is_set():
                return ToolResult(success=False, error="Aborted by user")

            self._log(
                stream_cb,
                f"📄 [{idx + 1}/{len(files)}] Reading: {file_path.name}",
            )

            # Read content
            content, read_error = self._read_file_content(file_path)
            if read_error:
                self._log(stream_cb, f"   ⚠️  Skip ({read_error})")
                errors.append(f"{file_path.name}: {read_error}")
                continue

            content_len = len(content)
            total_chars_read += content_len
            self._log(
                stream_cb,
                f"   Read {content_len:,} chars"
                + (f" (chunking required)" if content_len > _MAX_INPUT_CHARS else ""),
            )

            # Summarise via LLM
            self._log(stream_cb, f"   🤖 Summarising…")
            file_t0 = time.monotonic()
            try:
                summary, calls = await self._summarise_single_file(
                    provider, Message, file_path, content,
                    summary_style, instructions, personality_block,
                    total_usage, session_id,
                )
                llm_calls += calls
            except Exception as exc:
                log.warning(
                    "Summarise failed",
                    file=file_path.name,
                    error=str(exc),
                )
                self._log(stream_cb, f"   ❌ Failed: {exc}")
                errors.append(f"{file_path.name}: {exc}")
                continue

            summaries.append((file_path.name, summary))
            summary_words = len(summary.split())
            file_elapsed = time.monotonic() - file_t0
            elapsed_total = time.monotonic() - t_start
            remaining = len(files) - (idx + 1)
            avg_per_file = elapsed_total / (idx + 1) if idx > 0 else file_elapsed
            eta = avg_per_file * remaining
            self._log(
                stream_cb,
                f"   ✅ Done — {summary_words} words in {file_elapsed:.1f}s"
                f" (tokens: {total_usage['total_tokens']:,}"
                f" | ETA: ~{int(eta)}s for {remaining} remaining)",
            )

            # Persist individual summary to disk
            ind_path = output_dir / f"{file_path.stem}_summary.md"
            ind_path.write_text(
                f"# Summary: {file_path.name}\n\n{summary}",
                encoding="utf-8",
            )

        if not summaries:
            return ToolResult(
                success=False,
                error=f"All files failed: {'; '.join(errors)}",
            )

        # 8. Combine summaries ─────────────────────────────────────
        self._log(
            stream_cb,
            f"🔗 Combining {len(summaries)} summaries into ~{target_words} word output…",
        )

        try:
            combined, calls = await self._combine_summaries(
                provider, Message, summaries, target_words,
                instructions, personality_block, total_usage, session_id,
            )
            llm_calls += calls
        except Exception as exc:
            # Fallback: concatenate raw summaries
            log.warning("Combine step failed, using concatenation", error=str(exc))
            self._log(stream_cb, f"   ⚠️  Combine LLM failed, falling back to concatenation")
            parts = [
                f"## {name}\n\n{text}" for name, text in summaries
            ]
            combined = "\n\n---\n\n".join(parts)

        # 9. Save combined summary ─────────────────────────────────
        combined_path = output_dir / "combined_summary.md"
        combined_path.write_text(combined, encoding="utf-8")

        # 10. Build compact result ─────────────────────────────────
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        word_count = len(combined.split())
        error_note = ""
        if errors:
            shown = "; ".join(errors[:3])
            extra = f" (and {len(errors) - 3} more)" if len(errors) > 3 else ""
            error_note = f"\nSkipped {len(errors)} file(s) with errors: {shown}{extra}"

        self._log(
            stream_cb,
            f"✅ Complete — {len(summaries)} files → ~{word_count} words\n"
            f"   LLM calls: {llm_calls} | "
            f"Tokens: {total_usage['total_tokens']:,} "
            f"(prompt: {total_usage['prompt_tokens']:,}, "
            f"completion: {total_usage['completion_tokens']:,}) | "
            f"Time: {elapsed_ms / 1000:.1f}s",
        )

        return ToolResult(
            success=True,
            content=(
                f"Combined summary saved to: {combined_path}\n"
                f"Files summarised: {len(summaries)}/{len(files)}\n"
                f"Total source content: {total_chars_read:,} chars\n"
                f"Combined summary: ~{word_count} words\n"
                f"Individual summaries in: {output_dir}\n"
                f"Token usage: {total_usage['total_tokens']:,} total "
                f"({total_usage['prompt_tokens']:,} prompt, "
                f"{total_usage['completion_tokens']:,} completion) "
                f"across {llm_calls} LLM calls"
                f"{error_note}"
            ),
        )

    # ── logging helper ───────────────────────────────────────────

    @staticmethod
    def _log(stream_cb: Callable[[str], None] | None, msg: str) -> None:
        """Write to both the stream callback (thinking console) and file log."""
        log.info(msg)
        if stream_cb:
            stream_cb(f"{msg}\n")

    # ── personality loading ──────────────────────────────────────

    def _load_personality_context(self) -> str:
        """Load agent personality and user persona, return as prompt block."""
        try:
            from captain_claw.personality import (
                load_personality,
                load_user_personality,
            )
        except ImportError:
            return ""

        parts: list[str] = []

        # Agent personality (always available)
        agent_p = load_personality()
        if agent_p and agent_p.name:
            agent_parts = [f"You are {agent_p.name}."]
            if agent_p.description:
                agent_parts.append(agent_p.description)
            if agent_p.expertise:
                agent_parts.append(
                    f"Your areas of expertise: {', '.join(agent_p.expertise)}."
                )
            if agent_p.instructions:
                agent_parts.append(
                    f"Additional instructions: {agent_p.instructions}"
                )
            parts.append(" ".join(agent_parts))

        # User persona (if set)
        if self._user_id:
            user_p = load_user_personality(self._user_id)
            if user_p and user_p.name:
                user_parts = [f"You are talking to {user_p.name}."]
                if user_p.description:
                    user_parts.append(user_p.description)
                if user_p.expertise:
                    user_parts.append(
                        f"Their areas of expertise: {', '.join(user_p.expertise)}."
                    )
                if user_p.instructions:
                    user_parts.append(
                        f"Instructions from {user_p.name}: {user_p.instructions}"
                    )
                parts.append(" ".join(user_parts))

        return "\n\n".join(parts)

    # ── token tracking ───────────────────────────────────────────

    @staticmethod
    def _accumulate_usage(
        target: dict[str, int], usage: dict[str, int] | None,
    ) -> None:
        """Add usage values into target totals (mirrors Agent._accumulate_usage)."""
        if not usage:
            return
        prompt = int(usage.get("prompt_tokens", 0))
        completion = int(usage.get("completion_tokens", 0))
        total = int(usage.get("total_tokens", prompt + completion))
        target["prompt_tokens"] += prompt
        target["completion_tokens"] += completion
        target["total_tokens"] += total
        target.setdefault("cache_creation_input_tokens", 0)
        target.setdefault("cache_read_input_tokens", 0)
        target["cache_creation_input_tokens"] += int(
            usage.get("cache_creation_input_tokens", 0)
        )
        target["cache_read_input_tokens"] += int(
            usage.get("cache_read_input_tokens", 0)
        )

    @staticmethod
    def _record_single_usage(
        session_id: str,
        provider: Any,
        response: Any,
        messages: list[Any],
        interaction_label: str,
        latency_ms: int,
        max_tokens: int = 4096,
    ) -> None:
        """Persist a single LLM call's usage to the DB (fire-and-forget)."""
        try:
            from captain_claw.session import get_session_manager

            mgr = get_session_manager()
            usage = getattr(response, "usage", {}) or {}
            model_name = str(getattr(response, "model", "") or "")
            provider_name = str(getattr(provider, "provider", "") or "")
            finish_reason = str(getattr(response, "finish_reason", "") or "")
            content = str(getattr(response, "content", "") or "")

            input_bytes = 0
            for m in messages:
                c = getattr(m, "content", None) or ""
                if isinstance(c, str):
                    input_bytes += len(c.encode("utf-8", errors="replace"))
            output_bytes = len(content.encode("utf-8", errors="replace"))

            loop = asyncio.get_event_loop()
            loop.create_task(mgr.record_llm_usage(
                session_id=session_id if session_id != "default" else None,
                interaction=interaction_label,
                provider=provider_name,
                model=model_name,
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                total_tokens=int(usage.get("total_tokens", 0) or 0),
                cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
                cache_read_input_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                streaming=False,
                tools_enabled=False,
                max_tokens=max_tokens,
                finish_reason=finish_reason,
                error=False,
                latency_ms=latency_ms,
                task_name="summarize_files",
            ))
        except Exception:
            pass  # Never fail the main flow

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _resolve_folder(
        folder: str, runtime_base: Any | None,
    ) -> Path | None:
        """Resolve *folder* to an absolute path (mirrors ReadTool logic)."""
        raw = Path(folder).expanduser()
        if raw.is_absolute():
            candidate = raw.resolve()
            if candidate.is_dir():
                return candidate
        if runtime_base is not None:
            candidate = (Path(runtime_base) / raw).resolve()
            if candidate.is_dir():
                return candidate
        candidate = raw.resolve()
        return candidate if candidate.is_dir() else None

    @staticmethod
    def _discover_files(folder: Path, pattern: str) -> list[Path]:
        """Return sorted list of readable files matching *pattern*."""
        matches = sorted(folder.glob(pattern))
        result: list[Path] = []
        for p in matches:
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext not in _TEXT_EXTENSIONS and ext not in _DOCUMENT_EXTENSIONS:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size > _MAX_FILE_BYTES or size == 0:
                continue
            result.append(p)
        return result

    @staticmethod
    def _build_output_dir(
        saved_base: Any | None,
        runtime_base: Any | None,
        session_id: str,
    ) -> Path:
        """Create and return the output directory for this run."""
        if saved_base is not None:
            root = Path(saved_base).expanduser().resolve()
        elif runtime_base is not None:
            root = (Path(runtime_base).expanduser().resolve() / "saved")
        else:
            root = Path.cwd().resolve() / "saved"

        ts = time.strftime("%Y%m%d_%H%M%S")
        out = root / "summaries" / str(session_id) / ts
        out.mkdir(parents=True, exist_ok=True)
        return out

    # ── file reading ──────────────────────────────────────────────

    @staticmethod
    def _read_file_content(file_path: Path) -> tuple[str | None, str | None]:
        """Read *file_path* and return ``(content, error)``."""
        ext = file_path.suffix.lower()

        # Document types with specialised extractors
        if ext == ".pdf":
            try:
                from captain_claw.tools.document_extract import (
                    _extract_pdf_markdown,
                )

                content, err = _extract_pdf_markdown(file_path, max_pages=200)
                if err:
                    return None, err
                return content, None
            except Exception as exc:
                return None, f"PDF extraction error: {exc}"

        if ext == ".docx":
            try:
                from captain_claw.tools.document_extract import (
                    _extract_docx_markdown,
                )

                return _extract_docx_markdown(file_path), None
            except Exception as exc:
                return None, f"DOCX extraction error: {exc}"

        if ext == ".xlsx":
            try:
                from captain_claw.tools.document_extract import (
                    _extract_xlsx_markdown,
                )

                return _extract_xlsx_markdown(file_path, max_rows=5000), None
            except Exception as exc:
                return None, f"XLSX extraction error: {exc}"

        if ext == ".pptx":
            try:
                from captain_claw.tools.document_extract import (
                    _extract_pptx_markdown,
                )

                return _extract_pptx_markdown(file_path, max_slides=200), None
            except Exception as exc:
                return None, f"PPTX extraction error: {exc}"

        # Plain-text fallback
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            return content, None
        except Exception as exc:
            return None, f"Read error: {exc}"

    # ── LLM calls ─────────────────────────────────────────────────

    @staticmethod
    def _build_summary_system_prompt(
        style: str,
        instructions: str,
        personality_block: str = "",
    ) -> str:
        """Build system prompt for individual file summarisation."""
        style_guide = {
            "concise": (
                "Produce a concise summary that captures the key points, "
                "main arguments, and important details in a few paragraphs."
            ),
            "detailed": (
                "Produce a thorough summary covering all significant points, "
                "supporting details, data, and conclusions."
            ),
            "bullet_points": (
                "Produce a summary as a structured bullet-point list "
                "covering all key points and important details."
            ),
        }
        guidance = style_guide.get(style, style_guide["concise"])

        parts: list[str] = []

        # Inject personality context first (sets tone and perspective)
        if personality_block:
            parts.append(personality_block)
            parts.append("")  # blank line separator

        parts.append("You are a professional summariser.")
        parts.append(guidance)
        parts.append(
            "Do NOT include preamble like 'Here is a summary…'. "
            "Jump straight into the summary content."
        )
        if instructions:
            parts.append(f"Additional instructions: {instructions}")

        return "\n".join(parts)

    async def _summarise_single_file(
        self,
        provider: Any,
        MessageCls: type,
        file_path: Path,
        content: str,
        style: str,
        instructions: str,
        personality_block: str,
        total_usage: dict[str, int],
        session_id: str = "default",
    ) -> tuple[str, int]:
        """Summarise a single file's content, chunking if necessary.

        Returns ``(summary_text, llm_call_count)``.
        """
        system_prompt = self._build_summary_system_prompt(
            style, instructions, personality_block,
        )
        fname = file_path.stem
        calls = 0

        if len(content) <= _MAX_INPUT_CHARS:
            # Single-pass summarisation
            user_prompt = f"FILE: {file_path.name}\n\nCONTENT:\n{content}"
            text, usage = await self._llm_complete(
                provider, MessageCls, system_prompt, user_prompt,
                session_id=session_id,
                interaction_label=f"summarize_files: {fname}",
            )
            self._accumulate_usage(total_usage, usage)
            return text, 1

        # Chunked map-reduce for large files
        chunks = self._split_into_chunks(content, _MAX_INPUT_CHARS)
        chunk_summaries: list[str] = []

        for i, chunk in enumerate(chunks):
            user_prompt = (
                f"FILE: {file_path.name} (part {i + 1}/{len(chunks)})\n\n"
                f"CONTENT:\n{chunk}\n\n"
                f"Summarise this portion of the file."
            )
            chunk_summary, usage = await self._llm_complete(
                provider, MessageCls, system_prompt, user_prompt,
                session_id=session_id,
                interaction_label=f"summarize_files: {fname} chunk {i+1}/{len(chunks)}",
            )
            self._accumulate_usage(total_usage, usage)
            calls += 1
            chunk_summaries.append(chunk_summary)

        # Combine chunk summaries into one file summary
        joined = "\n\n---\n\n".join(
            f"[Part {i + 1}]\n{s}" for i, s in enumerate(chunk_summaries)
        )
        combine_prompt = (
            f"FILE: {file_path.name}\n\n"
            f"The following are partial summaries from different sections "
            f"of a single file. Combine them into one coherent summary.\n\n"
            f"{joined}"
        )
        text, usage = await self._llm_complete(
            provider, MessageCls, system_prompt, combine_prompt,
            session_id=session_id,
            interaction_label=f"summarize_files: {fname} chunk-combine",
        )
        self._accumulate_usage(total_usage, usage)
        calls += 1
        return text, calls

    async def _combine_summaries(
        self,
        provider: Any,
        MessageCls: type,
        summaries: list[tuple[str, str]],
        target_words: int,
        instructions: str,
        personality_block: str,
        total_usage: dict[str, int],
        session_id: str = "default",
    ) -> tuple[str, int]:
        """Combine individual file summaries into one final summary.

        Returns ``(combined_text, llm_call_count)``.
        """
        parts = [
            f"### {name}\n{text}" for name, text in summaries
        ]
        joined = "\n\n".join(parts)

        system_parts: list[str] = []
        if personality_block:
            system_parts.append(personality_block)
            system_parts.append("")

        system_parts.append(
            "You are a professional summariser. You will receive individual "
            "summaries of multiple files. Combine them into a single coherent "
            f"summary of approximately {target_words} words. "
            "Organise the content logically — group related themes, "
            "highlight the most important findings, and ensure the text "
            "flows naturally. Do NOT include preamble like 'Here is…'. "
            "Jump straight into the summary."
        )
        if instructions:
            system_parts.append(f"Additional instructions: {instructions}")

        system_prompt = "\n".join(system_parts)

        user_prompt = (
            f"Individual file summaries ({len(summaries)} files):\n\n{joined}"
        )
        text, usage = await self._llm_complete(
            provider, MessageCls, system_prompt, user_prompt,
            session_id=session_id,
            interaction_label=f"summarize_files: final-combine ({len(summaries)} files)",
        )
        self._accumulate_usage(total_usage, usage)
        return text, 1

    @staticmethod
    async def _llm_complete(
        provider: Any,
        MessageCls: type,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        *,
        session_id: str = "default",
        interaction_label: str = "summarize_files",
    ) -> tuple[str, dict[str, int]]:
        """Make a single LLM call using the configured provider.

        Returns ``(content_text, usage_dict)``.
        Each call is individually recorded in the LLM usage table.
        """
        import time as _time

        _t0 = _time.monotonic()
        messages = [
            MessageCls(role="system", content=system_prompt),
            MessageCls(role="user", content=user_prompt),
        ]
        response = await asyncio.wait_for(
            provider.complete(
                messages=messages,
                tools=None,
                temperature=0.3,
                max_tokens=max_tokens,
            ),
            timeout=120.0,
        )
        _latency_ms = int((_time.monotonic() - _t0) * 1000)
        content = (getattr(response, "content", "") or "").strip()
        usage = getattr(response, "usage", {}) or {}

        # Record this individual call to the DB
        SummarizeFilesTool._record_single_usage(
            session_id=session_id,
            provider=provider,
            response=response,
            messages=messages,
            interaction_label=interaction_label,
            latency_ms=_latency_ms,
            max_tokens=max_tokens,
        )

        return content, usage

    @staticmethod
    def _split_into_chunks(text: str, max_chars: int) -> list[str]:
        """Split *text* into chunks of roughly *max_chars* on paragraph boundaries."""
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para) + 2  # account for the \n\n separator
            if current_len + para_len > max_chars and current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += para_len

        if current:
            chunks.append("\n\n".join(current))

        # Safety: if a single paragraph exceeds max_chars, hard-split it
        final: list[str] = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                final.append(chunk)
            else:
                for i in range(0, len(chunk), max_chars):
                    final.append(chunk[i : i + max_chars])

        return final
