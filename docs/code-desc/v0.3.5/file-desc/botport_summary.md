# Summary: botport.py

# BotPort Tool Summary

**Summary:** BotPortTool is a specialized agent interface that enables Captain Claw instances to delegate tasks to specialist agents across a distributed BotPort network. It provides a unified API for consulting domain experts, managing ongoing conversations, and discovering available capabilities through five core actions (consult, follow_up, close, status, list_agents).

**Purpose:** Solves the problem of task delegation and expertise routing in multi-agent systems. Allows a primary agent to offload specialized work (legal analysis, coding, research, creative writing) to appropriately skilled agents without reimplementing domain expertise, while maintaining conversation context and concern tracking across the network.

**Most Important Functions/Classes/Procedures:**

1. **`BotPortTool` (class)** - Main tool implementation inheriting from Tool base class. Manages BotPort client lifecycle, parameter validation, and action routing. Exposes the tool to Captain Claw's agent framework with 300-second timeout for complex consultations.

2. **`execute()` (async method)** - Primary entry point that validates client connectivity, normalizes action input, and dispatches to appropriate handler methods. Returns ToolResult with success/error status and content payload.

3. **`_consult()` (async method)** - Initiates new specialist consultation by sending task, expertise tags, and context to BotPort network. Extracts concern_id for future reference and formats response with agent attribution and persona information.

4. **`_follow_up()` (async method)** - Continues existing consultation thread using concern_id, enabling multi-turn conversations with specialist agents while preserving context across exchanges.

5. **`_list_agents()` (async method)** - Discovers available agents and their capabilities on the BotPort network, returning JSON-formatted agent metadata for capability matching and selection.

**Architecture & Dependencies:**
- Depends on `captain_claw.tools.registry.Tool` base class and `BotPortClient` for network communication
- Uses async/await pattern throughout for non-blocking I/O
- Implements parameter schema for LLM/agent framework integration
- Manages concern lifecycle (create → follow-up → close) with unique concern_id tracking
- Supports session_id propagation for multi-turn agent conversations
- Returns standardized ToolResult objects for framework compatibility