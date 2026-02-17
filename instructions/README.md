# Captain Claw Instruction Templates

This folder contains externalized LLM instruction templates consumed by
`captain_claw.instructions.InstructionLoader`.

- `system_prompt.md`: Main runtime system prompt.
- `planning_mode_instructions.md`: Planning-mode extension injected into system prompt.
- `compaction_summary_system_prompt.md`: System prompt for compaction summarization.
- `compaction_summary_user_prompt.md`: User prompt for compaction summarization.
- `memory_continuity_header.md`: Header for memory continuity context notes.
- `planning_pipeline_header.md`: Header for planning pipeline context note.
- `planning_pipeline_footer.md`: Footer for planning pipeline context note.
- `script_synthesis_system_prompt.md`: Prompt for script-only synthesis fallback.
- `tool_output_rewrite_system_prompt.md`: Prompt for rewriting raw tool output.
- `tool_output_rewrite_user_prompt.md`: User payload template for tool-output rewrite.
- `session_description_system_prompt.md`: System prompt for automatic session description generation.
- `session_description_user_prompt.md`: User prompt template for session description generation from context.
- `guard_input_system_prompt.md`: Input guard system prompt for outbound-to-LLM safety checks.
- `guard_input_user_prompt.md`: Input guard user payload template.
- `guard_output_system_prompt.md`: Output guard system prompt for LLM response safety checks.
- `guard_output_user_prompt.md`: Output guard user payload template.
- `guard_script_tool_system_prompt.md`: Script/tool guard system prompt for execution safety checks.
- `guard_script_tool_user_prompt.md`: Script/tool guard user payload template.
