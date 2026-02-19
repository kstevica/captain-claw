"""Tools package for Captain Claw."""

from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult, get_tool_registry
from captain_claw.tools.shell import ShellTool
from captain_claw.tools.read import ReadTool
from captain_claw.tools.write import WriteTool
from captain_claw.tools.glob import GlobTool
from captain_claw.tools.web_fetch import WebFetchTool
from captain_claw.tools.web_search import WebSearchTool
from captain_claw.tools.document_extract import (
    DocxExtractTool,
    PdfExtractTool,
    PptxExtractTool,
    XlsxExtractTool,
)
from captain_claw.tools.pocket_tts import PocketTTSTool

__all__ = [
    "Tool",
    "ToolRegistry", 
    "ToolResult",
    "get_tool_registry",
    "ShellTool",
    "ReadTool",
    "WriteTool",
    "GlobTool",
    "WebFetchTool",
    "WebSearchTool",
    "PdfExtractTool",
    "DocxExtractTool",
    "XlsxExtractTool",
    "PptxExtractTool",
    "PocketTTSTool",
]
