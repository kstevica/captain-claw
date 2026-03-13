# Summary: execution_queue.py

# execution_queue.py Summary

This module implements a sophisticated lane-based async task execution queue system with per-lane concurrency control and a separate follow-up prompt queue manager for deferred processing. It provides primitives for managing concurrent async operations across multiple logical "lanes" while handling task lifecycle, deduplication, and intelligent queue draining strategies.

## Purpose

Solves the problem of managing concurrent async task execution with:
- **Per-lane concurrency limits**: Different logical lanes (main, cron, subagent, nested, agent_runtime) can have independent concurrency constraints
- **Task lifecycle management**: Tracks task generations to handle stale completions after lane resets
- **Follow-up queue deduplication**: Prevents duplicate prompts from queuing while a session is busy
- **Intelligent queue draining**: Multiple strategies (steer, interrupt, collect, summarize) for handling queued follow-ups
- **Wait time monitoring**: Warns when tasks exceed expected queue wait times

## Most Important Functions/Classes

### 1. **CommandQueueManager**
Core async queue orchestrator managing per-lane task execution with concurrency limits. Maintains lane states, schedules task draining, handles task completion/failure, and supports lane-wide resets. Key methods:
- `enqueue_in_lane()`: Queue a task in a specific lane with optional wait warnings
- `_drain_lane()`: Process queued tasks up to concurrency limit
- `_run_entry()`: Execute individual task with generation tracking and exception handling
- `set_lane_concurrency()`: Dynamically adjust per-lane concurrency limits
- `reset_all_lanes()`: Increment generation counter to invalidate stale task completions

### 2. **FollowupQueueManager**
Manages deferred follow-up prompts with deduplication, capacity capping, and multiple draining strategies. Handles queue consolidation and summarization when capacity is exceeded. Key methods:
- `enqueue_followup()`: Add follow-up run with deduplication and drop policy enforcement
- `schedule_drain()`: Asynchronously drain queue with debouncing and mode-specific consolidation
- `_apply_drop_policy()`: Enforce capacity limits with old/new/summarize strategies
- `_build_collect_prompt()`: Consolidate multiple queued items into single prompt with summary

### 3. **LaneState (dataclass)**
Represents execution state for a single lane: queue of pending tasks, set of active task IDs, concurrency limit, generation counter for stale task detection, and draining flag. Enables independent concurrency management across logical lanes.

### 4. **QueueEntry & FollowupRun (dataclasses)**
- **QueueEntry**: Wraps callable task, future for result/exception, enqueue timestamp, wait threshold, and optional wait callback
- **FollowupRun**: Payload for follow-up prompts with message ID, summary line, metadata, and enqueue timestamp for deduplication and tracking

### 5. **Utility Functions**
- `resolve_session_lane()`: Normalizes session keys to "session:*" lane format
- `normalize_queue_mode()`: Parses string variants into canonical QueueMode enum
- `normalize_queue_drop_policy()`: Parses string variants into canonical QueueDropPolicy enum

## Architecture & Dependencies

**Dependencies**: 
- `asyncio`: Core async runtime
- `collections.deque`: Efficient queue operations
- `dataclasses`: Type-safe configuration and state objects
- `captain_claw.logging`: Structured logging for debugging and monitoring

**Key Design Patterns**:
- **Generation-based invalidation**: Task generation counter prevents stale completions after lane resets
- **Deferred draining**: Drain operations scheduled asynchronously to avoid blocking
- **Multi-mode queue strategies**: Steer (keep latest), interrupt (drop backlog), collect (consolidate), summarize (aggregate dropped items)
- **Debounce-aware draining**: Follow-up queue respects debounce intervals before processing
- **Callback-based wait notifications**: Tasks can register callbacks when wait thresholds exceeded

**Role in System**: Provides foundational async task orchestration for agent runtime, enabling concurrent execution of main tasks, cron jobs, subagent calls, and nested operations while managing follow-up prompt queuing during busy sessions. Critical for preventing queue explosion and ensuring responsive user interactions.