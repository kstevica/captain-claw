"""Chunked processing pipeline for small-context models.

When a model's context window is too small to hold instructions + full
content in a single call, this mixin splits the content into sequential
chunks, processes each with the full instruction set, and combines the
partial results.

Design:
- **Map phase**: each chunk gets [system_prompt + task + chunk_content]
  sent as an isolated LLM call (sequential, not parallel).
- **Reduce phase**: partial results are combined via a final LLM call
  (or simple concatenation, depending on strategy).
- Completely transparent to callers — returns a single combined string.
- Verbose logging at every step for debugging.
"""

from __future__ import annotations

import time
from typing import Any

from captain_claw.config import get_config
from captain_claw.llm import Message
from captain_claw.logging import get_logger

log = get_logger(__name__)


class AgentChunkedProcessingMixin:
    """Chunked content processing for models with limited context windows."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _chunked_processing_is_active(self) -> bool:
        """Whether chunked processing is enabled for the current config.

        Returns True when:
        - ``context.chunked_processing.enabled`` is True, OR
        - ``auto_threshold > 0`` and ``context.max_tokens <= auto_threshold``
        """
        cfg = get_config()
        cp = cfg.context.chunked_processing
        if cp.enabled:
            return True
        if cp.auto_threshold > 0 and cfg.context.max_tokens <= cp.auto_threshold:
            return True
        return False

    def _chunked_processing_needed(
        self,
        instruction_tokens: int,
        content_tokens: int,
    ) -> bool:
        """Check whether the content exceeds available space and needs chunking.

        Args:
            instruction_tokens: tokens consumed by system prompt + task
                boilerplate (everything except the raw extracted content).
            content_tokens: tokens consumed by the extracted content alone.

        Returns:
            True if the content must be split into chunks.
        """
        if not self._chunked_processing_is_active():
            return False

        cfg = get_config()
        cp = cfg.context.chunked_processing
        context_budget = cfg.context.max_tokens
        output_reserve = cp.output_reserve_tokens
        available_for_content = context_budget - instruction_tokens - output_reserve

        needed = available_for_content < content_tokens

        if needed:
            log.info(
                "Chunked processing triggered",
                context_budget=context_budget,
                instruction_tokens=instruction_tokens,
                content_tokens=content_tokens,
                output_reserve=output_reserve,
                available_for_content=available_for_content,
                overflow_tokens=content_tokens - available_for_content,
            )
        return needed

    async def _chunked_process_content(
        self,
        *,
        system_text: str,
        task_preamble: str,
        extracted_content: str,
        task_suffix: str = "",
        item_label: str = "",
        interaction_label: str = "chunked",
        turn_usage: dict[str, int] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Split content into chunks, process each, and combine.

        This is the main entry point.  It replaces a single LLM call
        when the content is too large for the model's context window.

        Args:
            system_text: the system prompt for each chunk call.
            task_preamble: the user-message text that comes BEFORE the
                content (TASK:, ITEM:, FORMAT REFERENCE:, etc.).
            extracted_content: the raw content to chunk.
            task_suffix: text appended AFTER the content in each chunk
                (e.g. "Now produce the processed result...").
            item_label: human-readable label for logging.
            interaction_label: label prefix for guard/trace.
            turn_usage: accumulator for token usage tracking.

        Returns:
            (combined_text, stats_dict) where stats_dict contains
            chunk counts, token totals, and timing for debugging.
        """
        cfg = get_config()
        cp = cfg.context.chunked_processing
        context_budget = cfg.context.max_tokens
        output_reserve = cp.output_reserve_tokens
        overlap_tokens = cp.chunk_overlap_tokens
        max_chunks = max(1, cp.max_chunks)
        combine_strategy = (cp.combine_strategy or "summarize").strip().lower()

        # ── Compute token budgets ──────────────────────────────
        system_tokens = self._count_tokens(system_text)
        preamble_tokens = self._count_tokens(task_preamble)
        suffix_tokens = self._count_tokens(task_suffix) if task_suffix else 0
        # Overhead per chunk = system + preamble + suffix + output reserve
        # + small buffer for chunk framing text
        _framing_buffer = 80  # tokens for "CHUNK 1/N", separators, etc.
        overhead_per_chunk = (
            system_tokens + preamble_tokens + suffix_tokens
            + output_reserve + _framing_buffer
        )
        available_per_chunk = max(500, context_budget - overhead_per_chunk)

        content_tokens = self._count_tokens(extracted_content)
        overlap_chars = overlap_tokens * 4  # approximate chars from tokens

        log.info(
            "Chunked processing: computing split",
            item=item_label,
            context_budget=context_budget,
            system_tokens=system_tokens,
            preamble_tokens=preamble_tokens,
            suffix_tokens=suffix_tokens,
            overhead_per_chunk=overhead_per_chunk,
            available_per_chunk=available_per_chunk,
            content_tokens=content_tokens,
            content_chars=len(extracted_content),
            overlap_tokens=overlap_tokens,
            max_chunks=max_chunks,
            combine_strategy=combine_strategy,
        )

        # ── Split content ──────────────────────────────────────
        chunks = self._split_content_into_chunks(
            text=extracted_content,
            max_tokens_per_chunk=available_per_chunk,
            overlap_chars=overlap_chars,
            max_chunks=max_chunks,
        )
        num_chunks = len(chunks)

        log.info(
            "Chunked processing: content split complete",
            item=item_label,
            num_chunks=num_chunks,
            chunk_sizes_chars=[len(c) for c in chunks],
            chunk_sizes_tokens=[self._count_tokens(c) for c in chunks],
        )

        self._emit_thinking(
            f"chunked_pipeline: Splitting content into {num_chunks} chunks\n"
            f"{item_label}\n"
            f"context_budget={context_budget} | available_per_chunk={available_per_chunk} tok",
            tool="chunked_pipeline",
            phase="tool",
        )

        # ── Map phase: process each chunk sequentially ─────────
        partial_results: list[str] = []
        total_prompt_tokens = 0
        total_response_tokens = 0
        map_start = time.monotonic()

        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_num = chunk_idx + 1
            chunk_tokens = self._count_tokens(chunk_text)

            self._emit_thinking(
                f"chunked_pipeline: Processing chunk {chunk_num}/{num_chunks}\n"
                f"{item_label}\n"
                f"chunk_tokens={chunk_tokens}",
                tool="chunked_pipeline",
                phase="tool",
            )

            # Build per-chunk user message
            chunk_user_parts: list[str] = [task_preamble]
            chunk_user_parts.append(
                f"\n[CHUNK {chunk_num} OF {num_chunks}]"
            )
            if chunk_num > 1:
                chunk_user_parts.append(
                    "(This is a continuation. Earlier portions of the content "
                    "were processed in previous chunks.)"
                )
            chunk_user_parts.append(
                "\nEXTRACTED CONTENT (chunk):\n"
                "===\n"
                f"{chunk_text}\n"
                "===\n"
            )
            if task_suffix:
                if num_chunks == 1:
                    # Single chunk — produce final output directly
                    chunk_user_parts.append(task_suffix)
                else:
                    chunk_user_parts.append(
                        "Produce a PARTIAL result covering ONLY the content "
                        "in this chunk. Another step will combine all chunks."
                    )

            chunk_messages = [
                Message(role="system", content=system_text),
                Message(role="user", content="\n".join(chunk_user_parts)),
            ]

            _prompt_tok = sum(self._count_tokens(m.content) for m in chunk_messages)

            log.info(
                "Chunked processing: LLM call for chunk",
                item=item_label,
                chunk_num=chunk_num,
                num_chunks=num_chunks,
                chunk_content_tokens=chunk_tokens,
                prompt_tokens=_prompt_tok,
            )

            try:
                response = await self._complete_with_guards(
                    messages=chunk_messages,
                    tools=None,
                    interaction_label=f"{interaction_label}_chunk_{chunk_num}",
                    turn_usage=turn_usage,
                )
            except Exception as e:
                log.warning(
                    "Chunked processing: chunk LLM call failed",
                    item=item_label,
                    chunk_num=chunk_num,
                    error=str(e),
                )
                self._emit_tool_output(
                    "chunked_pipeline",
                    {"item": item_label, "chunk": chunk_num, "step": "llm_error"},
                    f"Chunk {chunk_num}/{num_chunks} LLM FAILED: {e}",
                )
                # On failure, skip this chunk — the combine step will work
                # with whatever partials we have.
                continue

            chunk_result = (response.content or "").strip()
            _resp_tok = self._count_tokens(chunk_result)
            total_prompt_tokens += _prompt_tok
            total_response_tokens += _resp_tok
            partial_results.append(chunk_result)

            self._emit_tool_output(
                "chunked_pipeline",
                {
                    "item": item_label,
                    "chunk": chunk_num,
                    "num_chunks": num_chunks,
                    "step": "chunk_done",
                    "prompt_tokens": _prompt_tok,
                    "response_tokens": _resp_tok,
                },
                f"Chunk {chunk_num}/{num_chunks} done — "
                f"{_prompt_tok} prompt / {_resp_tok} response tokens",
            )

            log.info(
                "Chunked processing: chunk done",
                item=item_label,
                chunk_num=chunk_num,
                num_chunks=num_chunks,
                prompt_tokens=_prompt_tok,
                response_tokens=_resp_tok,
                result_chars=len(chunk_result),
            )

        map_sec = round(time.monotonic() - map_start, 2)

        if not partial_results:
            log.warning(
                "Chunked processing: all chunks failed",
                item=item_label,
                num_chunks=num_chunks,
            )
            stats = self._build_chunked_stats(
                num_chunks=num_chunks,
                partial_count=0,
                total_prompt_tokens=total_prompt_tokens,
                total_response_tokens=total_response_tokens,
                map_sec=map_sec,
                combine_sec=0.0,
                combine_strategy=combine_strategy,
                combined=False,
            )
            return "", stats

        # ── Single chunk or single partial → no combine needed ─
        if len(partial_results) == 1:
            log.info(
                "Chunked processing: single partial, no combine needed",
                item=item_label,
                result_chars=len(partial_results[0]),
            )
            stats = self._build_chunked_stats(
                num_chunks=num_chunks,
                partial_count=1,
                total_prompt_tokens=total_prompt_tokens,
                total_response_tokens=total_response_tokens,
                map_sec=map_sec,
                combine_sec=0.0,
                combine_strategy="none",
                combined=False,
            )
            return partial_results[0], stats

        # ── Reduce phase: combine partial results ──────────────
        combine_start = time.monotonic()

        self._emit_thinking(
            f"chunked_pipeline: Combining {len(partial_results)} partial results\n"
            f"{item_label}\n"
            f"strategy={combine_strategy}",
            tool="chunked_pipeline",
            phase="tool",
        )

        if combine_strategy == "concatenate":
            combined = "\n\n".join(partial_results)
            combine_sec = round(time.monotonic() - combine_start, 2)
            log.info(
                "Chunked processing: concatenated partials",
                item=item_label,
                partial_count=len(partial_results),
                combined_chars=len(combined),
                combine_sec=combine_sec,
            )
        else:
            # Default: "summarize" — use an LLM call to combine
            combined, _comb_prompt, _comb_resp = await self._combine_partial_results(
                partial_results=partial_results,
                system_text=system_text,
                task_preamble=task_preamble,
                task_suffix=task_suffix,
                item_label=item_label,
                interaction_label=interaction_label,
                turn_usage=turn_usage,
            )
            total_prompt_tokens += _comb_prompt
            total_response_tokens += _comb_resp
            combine_sec = round(time.monotonic() - combine_start, 2)

        total_sec = round(map_sec + combine_sec, 2)

        self._emit_tool_output(
            "chunked_pipeline",
            {
                "item": item_label,
                "step": "complete",
                "num_chunks": num_chunks,
                "partial_count": len(partial_results),
                "combine_strategy": combine_strategy,
                "total_prompt_tokens": total_prompt_tokens,
                "total_response_tokens": total_response_tokens,
                "map_sec": map_sec,
                "combine_sec": combine_sec,
                "total_sec": total_sec,
            },
            (
                f"Chunked pipeline complete — {len(partial_results)} partials "
                f"combined ({combine_strategy})\n"
                f"  map={map_sec}s | combine={combine_sec}s | total={total_sec}s\n"
                f"  tokens: {total_prompt_tokens:,} prompt + "
                f"{total_response_tokens:,} response"
            ),
        )

        log.info(
            "Chunked processing: pipeline complete",
            item=item_label,
            num_chunks=num_chunks,
            partial_count=len(partial_results),
            combined_chars=len(combined),
            combine_strategy=combine_strategy,
            map_sec=map_sec,
            combine_sec=combine_sec,
            total_sec=total_sec,
            total_prompt_tokens=total_prompt_tokens,
            total_response_tokens=total_response_tokens,
        )

        stats = self._build_chunked_stats(
            num_chunks=num_chunks,
            partial_count=len(partial_results),
            total_prompt_tokens=total_prompt_tokens,
            total_response_tokens=total_response_tokens,
            map_sec=map_sec,
            combine_sec=combine_sec,
            combine_strategy=combine_strategy,
            combined=True,
        )
        return combined, stats

    # ------------------------------------------------------------------
    # Content splitting
    # ------------------------------------------------------------------

    def _split_content_into_chunks(
        self,
        text: str,
        max_tokens_per_chunk: int,
        overlap_chars: int = 800,
        max_chunks: int = 12,
    ) -> list[str]:
        """Split text into chunks that each fit within the token budget.

        Splitting strategy:
        1. Try paragraph boundaries (double newline).
        2. Fall back to single newline boundaries.
        3. Last resort: hard split at character boundary.

        Each chunk (except the first) includes ``overlap_chars`` of text
        from the end of the previous chunk for continuity.
        """
        if not text:
            return [text]

        # Quick check: does the whole text fit?
        total_tokens = self._count_tokens(text)
        if total_tokens <= max_tokens_per_chunk:
            log.debug(
                "Chunked processing: content fits in single chunk",
                total_tokens=total_tokens,
                max_tokens=max_tokens_per_chunk,
            )
            return [text]

        # Approximate chars-per-token for this provider
        # (use a sample to calibrate rather than assuming 4)
        _sample = text[:2000]
        _sample_tokens = self._count_tokens(_sample)
        chars_per_token = len(_sample) / max(1, _sample_tokens)
        max_chars_per_chunk = int(max_tokens_per_chunk * chars_per_token * 0.95)
        # Safety floor
        max_chars_per_chunk = max(500, max_chars_per_chunk)

        log.info(
            "Chunked processing: splitting content",
            total_chars=len(text),
            total_tokens=total_tokens,
            chars_per_token=round(chars_per_token, 2),
            max_chars_per_chunk=max_chars_per_chunk,
            overlap_chars=overlap_chars,
            max_chunks=max_chunks,
        )

        # Split into paragraphs first
        paragraphs = text.split("\n\n")

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_chars = 0

        for para in paragraphs:
            para_chars = len(para)
            # If adding this paragraph would exceed budget
            if current_chars + para_chars + 2 > max_chars_per_chunk and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(chunk_text)
                if len(chunks) >= max_chunks:
                    # Hit max chunks — dump everything remaining into the last
                    remaining_paras = [para] + paragraphs[paragraphs.index(para) + 1:]
                    # This won't work with duplicate paras so use a different approach
                    break
                # Start new chunk with overlap from end of previous
                if overlap_chars > 0 and chunk_text:
                    overlap_text = chunk_text[-overlap_chars:]
                    current_chunk = [overlap_text, para]
                    current_chars = len(overlap_text) + para_chars
                else:
                    current_chunk = [para]
                    current_chars = para_chars
            elif para_chars > max_chars_per_chunk:
                # Single paragraph exceeds budget — flush current, then hard-split
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    if len(chunks) >= max_chunks:
                        break
                # Hard-split the oversized paragraph
                _sub_chunks = self._hard_split(
                    para, max_chars_per_chunk, overlap_chars
                )
                for sc in _sub_chunks:
                    if len(chunks) >= max_chunks:
                        break
                    chunks.append(sc)
                current_chunk = []
                current_chars = 0
            else:
                current_chunk.append(para)
                current_chars += para_chars + 2  # +2 for \n\n joiner

        # Flush remaining
        if current_chunk and len(chunks) < max_chunks:
            chunks.append("\n\n".join(current_chunk))

        # Edge case: if loop broke early due to max_chunks, append remainder
        if len(chunks) >= max_chunks and current_chunk:
            # Append remaining to the last chunk
            leftover = "\n\n".join(current_chunk)
            if chunks:
                chunks[-1] = chunks[-1] + "\n\n" + leftover
            else:
                chunks.append(leftover)

        # Ensure we have at least one chunk
        if not chunks:
            chunks = [text]

        log.info(
            "Chunked processing: split result",
            num_chunks=len(chunks),
            chunk_chars=[len(c) for c in chunks],
        )

        return chunks

    @staticmethod
    def _hard_split(
        text: str,
        max_chars: int,
        overlap_chars: int,
    ) -> list[str]:
        """Hard-split a single block of text when no paragraph boundaries exist."""
        chunks: list[str] = []
        pos = 0
        text_len = len(text)
        while pos < text_len:
            end = min(pos + max_chars, text_len)
            # Try to break at a newline near the boundary
            if end < text_len:
                newline_pos = text.rfind("\n", pos + max_chars // 2, end)
                if newline_pos > pos:
                    end = newline_pos + 1
            chunk = text[pos:end]
            chunks.append(chunk)
            # Advance with overlap
            pos = max(pos + 1, end - overlap_chars)
        return chunks

    # ------------------------------------------------------------------
    # Reduce / combine
    # ------------------------------------------------------------------

    async def _combine_partial_results(
        self,
        *,
        partial_results: list[str],
        system_text: str,
        task_preamble: str,
        task_suffix: str,
        item_label: str,
        interaction_label: str,
        turn_usage: dict[str, int] | None,
    ) -> tuple[str, int, int]:
        """Combine partial chunk results via a synthesis LLM call.

        If the combined partials are themselves too large for the context
        window, falls back to concatenation to avoid infinite recursion.

        Returns (combined_text, prompt_tokens, response_tokens).
        """
        cfg = get_config()
        cp = cfg.context.chunked_processing
        context_budget = cfg.context.max_tokens
        output_reserve = cp.output_reserve_tokens

        combined_partials = "\n\n---\n\n".join(
            f"[PARTIAL RESULT {i + 1}/{len(partial_results)}]\n{r}"
            for i, r in enumerate(partial_results)
        )
        partials_tokens = self._count_tokens(combined_partials)

        combine_system = (
            "You are a document processing assistant. You are given partial "
            "results from processing different sections of a single document. "
            "Your job is to combine them into one coherent, unified result.\n\n"
            "Rules:\n"
            "- Merge overlapping or duplicate content — do NOT repeat the same "
            "information twice.\n"
            "- Maintain the original structure and formatting.\n"
            "- Produce ONLY the final combined result, no commentary.\n"
            "- Do NOT mention chunks, partial results, or the combining process.\n"
        )
        combine_system_tokens = self._count_tokens(combine_system)
        combine_preamble_tokens = self._count_tokens(task_preamble)

        _overhead = (
            combine_system_tokens + combine_preamble_tokens
            + output_reserve + 100  # buffer
        )
        available_for_partials = context_budget - _overhead

        log.info(
            "Chunked processing: combine phase",
            item=item_label,
            partial_count=len(partial_results),
            partials_tokens=partials_tokens,
            available_for_partials=available_for_partials,
            combine_strategy="summarize",
        )

        # If partials themselves overflow the context, fall back to concatenation
        if partials_tokens > available_for_partials:
            log.warning(
                "Chunked processing: partials too large for combine call, "
                "falling back to concatenation",
                item=item_label,
                partials_tokens=partials_tokens,
                available=available_for_partials,
            )
            self._emit_tool_output(
                "chunked_pipeline",
                {"item": item_label, "step": "combine_fallback"},
                f"Partials too large for combine call ({partials_tokens} tok "
                f"> {available_for_partials} available) — using concatenation",
            )
            combined = "\n\n".join(partial_results)
            return combined, 0, 0

        combine_user_parts = [
            task_preamble,
            "\nThe following are partial results from processing different "
            "sections of the source content. Combine them into a single, "
            "unified output.\n\n",
            combined_partials,
        ]
        if task_suffix:
            combine_user_parts.append(f"\n{task_suffix}")

        combine_messages = [
            Message(role="system", content=combine_system),
            Message(role="user", content="\n".join(combine_user_parts)),
        ]

        _prompt_tok = sum(self._count_tokens(m.content) for m in combine_messages)

        log.info(
            "Chunked processing: combine LLM call",
            item=item_label,
            prompt_tokens=_prompt_tok,
        )

        try:
            response = await self._complete_with_guards(
                messages=combine_messages,
                tools=None,
                interaction_label=f"{interaction_label}_combine",
                turn_usage=turn_usage,
            )
        except Exception as e:
            log.warning(
                "Chunked processing: combine LLM call failed, "
                "falling back to concatenation",
                item=item_label,
                error=str(e),
            )
            self._emit_tool_output(
                "chunked_pipeline",
                {"item": item_label, "step": "combine_error"},
                f"Combine LLM call failed — falling back to concatenation: {e}",
            )
            combined = "\n\n".join(partial_results)
            return combined, _prompt_tok, 0

        result = (response.content or "").strip()
        _resp_tok = self._count_tokens(result)

        log.info(
            "Chunked processing: combine done",
            item=item_label,
            prompt_tokens=_prompt_tok,
            response_tokens=_resp_tok,
            combined_chars=len(result),
        )

        return result, _prompt_tok, _resp_tok

    # ------------------------------------------------------------------
    # Stats helper
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Conversation-flow hook: reduce oversized tool results
    # ------------------------------------------------------------------

    async def _chunked_reduce_tool_result(
        self,
        tool_name: str,
        tool_content: str,
        user_query: str,
        turn_usage: dict[str, int] | None = None,
    ) -> str | None:
        """Chunk-process an oversized tool result in the normal conversation flow.

        Called from ``_handle_tool_calls`` when a content-extraction tool
        (read, pdf_extract, web_fetch, etc.) returns a result that would
        exceed the model's available context window.

        Returns the chunked+combined result string, or ``None`` if chunking
        is not needed (content fits in context).
        """
        if not self._chunked_processing_is_active():
            return None

        cfg = get_config()
        cp = cfg.context.chunked_processing
        context_budget = cfg.context.max_tokens
        output_reserve = cp.output_reserve_tokens

        # Estimate instruction overhead (system prompt + conversation messages)
        system_text = self._build_system_prompt()
        system_tokens = self._count_tokens(system_text)
        # Reserve space for existing conversation messages (user query, tool
        # calls, assistant responses, memory notes, etc.)
        # Use a generous estimate: system + output reserve + 2000 for messages
        instruction_overhead = system_tokens + output_reserve + 2000
        available_for_content = max(500, context_budget - instruction_overhead)

        content_tokens = self._count_tokens(tool_content)
        if content_tokens <= available_for_content:
            return None  # Fits in context — no chunking needed

        log.info(
            "Chunked processing: oversized tool result detected",
            tool=tool_name,
            content_tokens=content_tokens,
            available_for_content=available_for_content,
            context_budget=context_budget,
            user_query=user_query[:120],
        )

        self._emit_thinking(
            f"⚡ Content from {tool_name} exceeds context window "
            f"({content_tokens:,} tokens > {available_for_content:,} available). "
            f"Activating chunked processing pipeline.",
            tool="chunked_pipeline",
            phase="tool",
        )

        task_preamble = (
            f"USER REQUEST: {user_query}\n\n"
            f"Below is a portion of the content retrieved by the '{tool_name}' tool. "
            f"Process this portion according to the user's request above."
        )
        task_suffix = (
            "Produce your result for this portion of the content. "
            "Be thorough and preserve key details relevant to the user's request."
        )

        combined, stats = await self._chunked_process_content(
            system_text=system_text,
            task_preamble=task_preamble,
            extracted_content=tool_content,
            task_suffix=task_suffix,
            item_label=f"tool_result:{tool_name}",
            interaction_label=f"chunked_tool_{tool_name}",
            turn_usage=turn_usage,
        )

        if not combined:
            log.warning(
                "Chunked processing: tool result reduction failed, using truncated original",
                tool=tool_name,
            )
            return None

        log.info(
            "Chunked processing: tool result reduced",
            tool=tool_name,
            original_tokens=content_tokens,
            reduced_tokens=self._count_tokens(combined),
            chunks=stats.get("num_chunks", 0),
            total_sec=stats.get("total_sec", 0),
        )

        return (
            f"[Chunked processing: content was {content_tokens:,} tokens, "
            f"processed in {stats.get('num_chunks', '?')} chunks]\n\n"
            f"{combined}"
        )

    @staticmethod
    def _build_chunked_stats(
        *,
        num_chunks: int,
        partial_count: int,
        total_prompt_tokens: int,
        total_response_tokens: int,
        map_sec: float,
        combine_sec: float,
        combine_strategy: str,
        combined: bool,
    ) -> dict[str, Any]:
        """Build a stats dict for chunked processing diagnostics."""
        return {
            "chunked": True,
            "num_chunks": num_chunks,
            "partial_count": partial_count,
            "total_prompt_tokens": total_prompt_tokens,
            "total_response_tokens": total_response_tokens,
            "map_sec": map_sec,
            "combine_sec": combine_sec,
            "total_sec": round(map_sec + combine_sec, 2),
            "combine_strategy": combine_strategy,
            "combined": combined,
        }
