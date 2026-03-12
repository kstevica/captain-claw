"""Glob tool for finding files by pattern."""

import asyncio
import fnmatch
import os
from pathlib import Path
from typing import Any

import glob

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


def _case_insensitive_walk(directory: str, pattern: str) -> list[str]:
    """Walk *directory* recursively and return files matching *pattern* case-insensitively.

    The *pattern* should be a filename-level glob (e.g. ``*kartica*``, ``*.pdf``).
    Directory components like ``subdir/*.pdf`` are NOT supported — only the
    basename of each file is tested.
    """
    # Extract just the filename portion of the pattern (after last /).
    # E.g. "**/*kartica*stranke*" → "*kartica*stranke*"
    parts = pattern.replace("\\", "/").split("/")
    fname_pattern = parts[-1] if parts else pattern
    lower_pattern = fname_pattern.lower()

    results: list[str] = []
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if fnmatch.fnmatch(name.lower(), lower_pattern):
                results.append(os.path.join(root, name))
    return results


class GlobTool(Tool):
    """Find files by pattern."""

    name = "glob"
    description = "Find files matching a glob pattern."
    timeout_seconds = 10.0  # glob is a local FS scan — 10 s is ample
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g., '**/*.py', 'src/**/*.ts')",
            },
            "root": {
                "type": "string",
                "description": "Root directory to search from (default: workspace root)",
            },
            "scope": {
                "type": "string",
                "enum": ["workspace", "workflow"],
                "description": (
                    "Where to search. 'workspace' (default) searches pre-existing "
                    "user files in the workspace. 'workflow' searches ONLY the "
                    "workflow output directory where earlier tasks wrote files."
                ),
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of results (default: 100)",
            },
        },
        "required": ["pattern"],
    }

    async def execute(
        self,
        pattern: str,
        root: str | None = None,
        scope: str | None = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """Find files matching pattern.

        Args:
            pattern: Glob pattern
            root: Optional root directory
            scope: Where to search — ``"workspace"`` (default) for
                pre-existing user files, ``"workflow"`` for files created
                by earlier tasks in the current orchestration run.
            limit: Max results

        Returns:
            ToolResult with matching files
        """
        try:
            workflow_run_dir: str | None = kwargs.get("_workflow_run_dir")
            if isinstance(workflow_run_dir, Path):
                workflow_run_dir = str(workflow_run_dir)

            # Preserve the raw pattern before root is joined — used for
            # the cross-scope hint when the primary search finds nothing.
            raw_pattern = pattern
            user_provided_root = bool(root)

            # Resolve root based on scope.
            if scope == "workflow":
                if workflow_run_dir is not None:
                    root = workflow_run_dir
                else:
                    return ToolResult(
                        success=True,
                        content=(
                            "No workflow output directory available. "
                            "scope='workflow' is only valid during orchestrated "
                            "workflows. Use the default scope to search the workspace."
                        ),
                    )
            elif not root:
                # Default / "workspace" scope: search from workspace root.
                runtime_base = kwargs.get("_runtime_base_path")
                if runtime_base is not None:
                    root = str(runtime_base)

            if root:
                pattern = str(Path(root) / pattern)

            # Also search configured extra read folders when using the
            # default workspace scope (no custom root, not workflow).
            # But NOT when the pattern contains path-specific directory
            # components — e.g. "saved/showcase/SESSION_ID/**/*" is
            # looking for files in a specific directory, not globally.
            # Also skip when the filename portion is just "*" (matches
            # everything), which happens with "somepath/**/*" patterns.
            _pat_parts = raw_pattern.replace("\\", "/").split("/")
            _fname_part = _pat_parts[-1] if _pat_parts else raw_pattern
            _has_specific_dirs = any(
                p and p not in ("*", "**") for p in _pat_parts[:-1]
            )
            search_extra_dirs = (
                scope != "workflow"
                and not user_provided_root
                and not _has_specific_dirs
                and _fname_part not in ("*",)
            )

            # Find files (sync, but run in executor to not block)
            loop = asyncio.get_event_loop()
            matches = await loop.run_in_executor(
                None,
                lambda: glob.glob(pattern, recursive=True)
            )

            # When running inside an orchestrator workflow, apply the
            # timestamp filter ONLY to the workflow-run/ output directory
            # to avoid picking up stale outputs from earlier runs.
            # Pre-existing workspace files (user documents, config, etc.)
            # are INPUTS and must never be filtered out.
            workflow_started_at: float | None = kwargs.get("_workflow_started_at")
            if workflow_started_at is not None and workflow_run_dir is not None:
                wrd_prefix = str(Path(workflow_run_dir).resolve()) + "/"
                cutoff = workflow_started_at - 2.0
                before_count = len(matches)

                def _in_workflow_run(filepath: str) -> bool:
                    """Check if *filepath* is inside the workflow-run dir.

                    Uses resolve() to handle macOS /var → /private/var
                    and other symlink discrepancies.
                    """
                    try:
                        return str(Path(filepath).resolve()).startswith(wrd_prefix)
                    except (OSError, ValueError):
                        return False

                matches = [
                    m for m in matches
                    if (
                        # Only timestamp-filter files inside workflow-run/
                        not _in_workflow_run(m)
                        or _file_modified_after(m, cutoff)
                    )
                ]
                filtered_count = before_count - len(matches)
                if filtered_count > 0:
                    log.info(
                        "Glob workflow filter applied",
                        pattern=pattern,
                        before=before_count,
                        after=len(matches),
                        filtered=filtered_count,
                    )

            # Also search configured extra read folders (case-insensitive).
            # Python's glob.glob is case-sensitive even on macOS, so we use
            # os.walk + fnmatch with lowered names to ensure user files like
            # "Kartica_stranke.pdf" are found when searching "*kartica*".
            extra_matches: dict[str, list[str]] = {}  # dir → absolute matches
            if search_extra_dirs:
                try:
                    from captain_claw.config import get_config
                    extra_dirs = get_config().tools.read.extra_dirs
                except Exception:
                    extra_dirs = []

                if extra_dirs:
                    log.info("Glob searching extra read folders", count=len(extra_dirs), pattern=raw_pattern)

                for extra_dir in extra_dirs:
                    edir = Path(extra_dir).expanduser().resolve()
                    if not edir.is_dir():
                        log.warning("Extra read folder not a directory", path=str(edir))
                        continue
                    try:
                        edir_matches = await loop.run_in_executor(
                            None,
                            lambda d=str(edir), p=raw_pattern: _case_insensitive_walk(d, p),
                        )
                    except Exception as exc:
                        log.warning("Glob extra dir failed", path=str(edir), error=str(exc))
                        continue
                    log.info("Glob extra dir result", path=str(edir), found=len(edir_matches))
                    if edir_matches:
                        extra_matches[str(edir)] = edir_matches

            # Apply limit
            matches = matches[:limit]

            if not matches and not extra_matches:
                # Cross-scope hint: if we searched workspace and found
                # nothing, check whether the workflow output directory has
                # matches for the same pattern.  This helps the LLM
                # self-correct by retrying with scope='workflow'.
                hint = ""
                if scope != "workflow" and workflow_run_dir is not None:
                    wrd_pattern = str(Path(workflow_run_dir) / raw_pattern)
                    wrd_matches = await loop.run_in_executor(
                        None,
                        lambda p=wrd_pattern: glob.glob(p, recursive=True),
                    )
                    if wrd_matches:
                        hint = (
                            f"\n\nHint: {len(wrd_matches)} file(s) matching "
                            f"'{raw_pattern}' exist in the workflow output "
                            f"directory (files created by earlier tasks). "
                            f"Retry with scope='workflow' to find them."
                        )
                return ToolResult(
                    success=True,
                    content=f"No files found matching: {raw_pattern}{hint}",
                )

            # Return workspace-relative paths when the root was auto-resolved
            # to the workspace base.  This ensures paths like
            # "pdf-test/subfolder/file.pdf" are returned instead of long
            # absolute paths, keeping output concise and consistent with how
            # other tools accept relative paths.
            if root:
                root_prefix = str(Path(root).resolve()) + "/"
                matches = [
                    m[len(root_prefix):] if m.startswith(root_prefix) else m
                    for m in matches
                ]

            # Format output with scope-aware label.
            if scope == "workflow":
                output = f"Found {len(matches)} file(s) in workflow output:\n"
            else:
                total = len(matches) + sum(len(v) for v in extra_matches.values())
                output = f"Found {total} file(s):\n"
            output += "\n".join(f"  {m}" for m in matches)

            # Append extra read folder results with clear labeling.
            for edir, edir_files in extra_matches.items():
                edir_files = edir_files[:limit]
                output += f"\n\n  [read folder: {edir}]\n"
                output += "\n".join(f"  {f}" for f in edir_files)

            return ToolResult(
                success=True,
                content=output,
            )

        except Exception as e:
            log.error("Glob failed", pattern=pattern, error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
            )


def _file_modified_after(path: str, cutoff: float) -> bool:
    """Check if a file's mtime is at or after the cutoff timestamp."""
    try:
        return os.path.getmtime(path) >= cutoff
    except OSError:
        # File vanished or inaccessible — include it to be safe.
        return True
