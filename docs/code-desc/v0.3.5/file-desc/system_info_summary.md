# Summary: system_info.py

# system_info.py Summary

**Summary:** This module gathers real-time system environment metrics (memory, disk, network, uptime, load) and formats them for injection into AI agent system prompts. It provides cross-platform support (Linux/macOS/Windows) with graceful fallbacks and three detail levels (nano/micro/normal) to accommodate different prompt contexts.

**Purpose:** Solves the problem of contextualizing AI agent behavior with accurate host environment information. Agents can make better decisions when aware of system constraints (available memory, disk space, network connectivity, system load). The module abstracts OS-specific system calls behind a unified interface and formats output for prompt injection.

**Most Important Functions/Classes:**

1. **`build_system_info_block(detail_level: str)`** — Main entry point that orchestrates all metric collection and returns formatted output. Supports three modes: "nano" (empty), "micro" (single compact line), "normal" (full labeled block). Routes to appropriate formatting based on detail_level.

2. **`_get_memory_usage() → tuple[str, str]`** — Retrieves total and available memory with dual output formats (human-readable and compact). Implements platform-specific logic: reads `/proc/meminfo` on Linux, calls `sysctl`/`vm_stat` on macOS. Returns tuple of (normal format, compact format) for flexible consumption.

3. **`_get_uptime() / _get_uptime_compact()`** — Calculates system uptime from boot time with human-readable output ("5 days, 3 hours") and compact format ("5d3h"). Uses `/proc/uptime` on Linux and `sysctl kern.boottime` on macOS.

4. **`_get_local_ip() / _get_public_ip()`** — Network utilities that retrieve LAN IP (via UDP socket trick to OS interface selection) and public IP (via ipify.org API). Both implement graceful degradation with fallback methods.

5. **`_get_disk_free() / _get_system_load()`** — Utility functions for disk space (via `shutil.disk_usage()`) and system load averages (via `os.getloadavg()`). Format output as human-readable strings with automatic unit selection (GB/MB).

**Architecture Notes:**
- All helper functions are prefixed with `_` and implement try-except patterns for robustness across platforms
- No external dependencies except optional `httpx` for public IP lookup (gracefully skipped if unavailable)
- Designed for prompt injection: output is pre-formatted strings, not raw data structures
- Detail levels allow optimization for token budgets in different agent contexts (nano for ultra-constrained prompts, micro for balanced, normal for verbose)