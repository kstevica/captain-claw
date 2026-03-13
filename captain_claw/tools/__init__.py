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
from captain_claw.tools.edit import EditTool
from captain_claw.tools.glob import GlobTool
from captain_claw.tools.web_fetch import WebFetchTool, WebGetTool
from captain_claw.tools.web_search import WebSearchTool
from captain_claw.tools.document_extract import (
    DocxExtractTool,
    PdfExtractTool,
    PptxExtractTool,
    XlsxExtractTool,
)
from captain_claw.tools.pocket_tts import PocketTTSTool
from captain_claw.tools.image_gen import ImageGenTool
from captain_claw.tools.image_ocr import ImageOcrTool, ImageVisionTool
from captain_claw.tools.send_mail import SendMailTool
from captain_claw.tools.google_drive import GoogleDriveTool
from captain_claw.tools.google_calendar import GoogleCalendarTool
from captain_claw.tools.google_mail import GoogleMailTool
from captain_claw.tools.gws import GwsTool
from captain_claw.tools.personality import PersonalityTool
from captain_claw.tools.todo import TodoTool
from captain_claw.tools.contacts import ContactsTool
from captain_claw.tools.scripts import ScriptsTool
from captain_claw.tools.apis import ApisTool
from captain_claw.tools.direct_api import DirectApiTool
from captain_claw.tools.typesense import TypesenseTool
from captain_claw.tools.datastore import DatastoreTool
from captain_claw.tools.termux import TermuxTool
from captain_claw.tools.playbooks import PlaybooksTool
from captain_claw.tools.botport import BotPortTool
from captain_claw.tools.browser import BrowserTool
from captain_claw.tools.pinchtab import PinchTabTool
from captain_claw.tools.clipboard import ClipboardTool
from captain_claw.tools.screen_capture import ScreenCaptureTool
from captain_claw.tools.summarize_files import SummarizeFilesTool

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
    "EditTool",
    "GlobTool",
    "WebFetchTool",
    "WebGetTool",
    "WebSearchTool",
    "PdfExtractTool",
    "DocxExtractTool",
    "XlsxExtractTool",
    "PptxExtractTool",
    "PocketTTSTool",
    "ImageGenTool",
    "ImageOcrTool",
    "ImageVisionTool",
    "PersonalityTool",
    "SendMailTool",
    "GoogleDriveTool",
    "GoogleCalendarTool",
    "GoogleMailTool",
    "GwsTool",
    "TodoTool",
    "ContactsTool",
    "ScriptsTool",
    "ApisTool",
    "DirectApiTool",
    "TypesenseTool",
    "DatastoreTool",
    "TermuxTool",
    "PlaybooksTool",
    "BotPortTool",
    "BrowserTool",
    "PinchTabTool",
    "ClipboardTool",
    "ScreenCaptureTool",
    "SummarizeFilesTool",
]
