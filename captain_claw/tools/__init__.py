"""Tools package for Captain Claw."""

from captain_claw.tools.registry import (
    Tool,
    ToolPolicy,
    ToolPolicyChain,
    ToolRegistry,
    ToolResult,
    get_tool_registry,
)
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
from captain_claw.tools.send_mail import SendMailTool
from captain_claw.tools.google_drive import GoogleDriveTool
from captain_claw.tools.todo import TodoTool
from captain_claw.tools.contacts import ContactsTool

__all__ = [
    "Tool",
    "ToolPolicy",
    "ToolPolicyChain",
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
    "SendMailTool",
    "GoogleDriveTool",
    "TodoTool",
    "ContactsTool",
]
