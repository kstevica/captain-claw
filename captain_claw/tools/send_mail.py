"""Send mail tool for dispatching email via Mailgun, SendGrid, or SMTP."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import httpx

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class SendMailTool(Tool):
    """Send email with optional file attachments."""

    name = "send_mail"
    timeout_seconds = 60.0
    description = (
        "Send an email message. Supports to, cc, bcc, subject, "
        "text and/or HTML body, and file attachments from the local filesystem."
    )
    parameters = {
        "type": "object",
        "properties": {
            "to": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of recipient email addresses.",
            },
            "cc": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of CC email addresses.",
            },
            "bcc": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of BCC email addresses.",
            },
            "subject": {
                "type": "string",
                "description": "Email subject line.",
            },
            "body": {
                "type": "string",
                "description": "Plain text body of the email.",
            },
            "html": {
                "type": "string",
                "description": "HTML body of the email (optional, sent alongside or instead of plain text).",
            },
            "attachments": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of local file paths to attach.",
            },
        },
        "required": ["to", "subject"],
    }

    def __init__(self) -> None:
        self.client = httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": "Captain Claw/0.1.0 (Send Mail Tool)"},
        )

    # ------------------------------------------------------------------ #
    #  Main execute
    # ------------------------------------------------------------------ #

    async def execute(
        self,
        to: list[str],
        subject: str,
        body: str = "",
        html: str = "",
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        attachments: list[str] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Send an email using the configured provider."""

        # -- validate inputs ------------------------------------------------
        recipients = [addr.strip() for addr in (to or []) if addr.strip()]
        if not recipients:
            return ToolResult(success=False, error="At least one 'to' address is required.")

        subject_text = (subject or "").strip()
        body_text = (body or "").strip()
        html_text = (html or "").strip()
        if not body_text and not html_text:
            return ToolResult(success=False, error="Either 'body' or 'html' must be provided.")

        cc_list = [addr.strip() for addr in (cc or []) if addr.strip()]
        bcc_list = [addr.strip() for addr in (bcc or []) if addr.strip()]

        # -- load config ----------------------------------------------------
        cfg = get_config()
        mail_cfg = cfg.tools.send_mail
        provider = (mail_cfg.provider or "smtp").strip().lower()
        from_address = (mail_cfg.from_address or "").strip()
        from_name = (mail_cfg.from_name or "").strip()

        if not from_address:
            return ToolResult(
                success=False,
                error="Missing from_address in tools.send_mail config.",
            )

        # -- read attachment files ------------------------------------------
        attachment_data: list[dict[str, Any]] = []
        max_bytes = int(mail_cfg.max_attachment_bytes or 26214400)

        for raw_path in attachments or []:
            file_path = Path(raw_path).expanduser().resolve()
            if not file_path.exists():
                return ToolResult(success=False, error=f"Attachment not found: {file_path}")
            if not file_path.is_file():
                return ToolResult(success=False, error=f"Attachment is not a file: {file_path}")
            size = file_path.stat().st_size
            if size > max_bytes:
                return ToolResult(
                    success=False,
                    error=f"Attachment too large ({size} bytes): {file_path}. Max: {max_bytes}.",
                )
            content_bytes = await asyncio.to_thread(file_path.read_bytes)
            mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            attachment_data.append({
                "filename": file_path.name,
                "content": content_bytes,
                "mime_type": mime_type,
            })

        # -- dispatch to provider -------------------------------------------
        timeout = float(mail_cfg.timeout or 60)
        from_header = f"{from_name} <{from_address}>" if from_name else from_address

        try:
            if provider == "mailgun":
                detail = await self._send_mailgun(
                    mail_cfg, from_header, recipients, cc_list, bcc_list,
                    subject_text, body_text, html_text, attachment_data, timeout,
                )
            elif provider == "sendgrid":
                detail = await self._send_sendgrid(
                    mail_cfg, from_address, from_name, recipients, cc_list, bcc_list,
                    subject_text, body_text, html_text, attachment_data, timeout,
                )
            elif provider == "smtp":
                detail = await self._send_smtp(
                    mail_cfg, from_header, from_address, recipients, cc_list, bcc_list,
                    subject_text, body_text, html_text, attachment_data,
                )
            else:
                return ToolResult(success=False, error=f"Unsupported send_mail provider: {provider}")
        except Exception as exc:
            log.error("send_mail failed", provider=provider, error=str(exc))
            return ToolResult(success=False, error=str(exc))

        # -- build result ---------------------------------------------------
        all_recipients = recipients + cc_list + bcc_list
        attachment_note = f", {len(attachment_data)} attachment(s)" if attachment_data else ""

        lines = [
            f"Email sent via {provider}.",
            f"From: {from_header}",
            f"To: {', '.join(recipients)}",
        ]
        if cc_list:
            lines.append(f"CC: {', '.join(cc_list)}")
        if bcc_list:
            lines.append(f"BCC: {', '.join(bcc_list)}")
        lines.append(f"Subject: {subject_text}")
        lines.append(f"Recipients: {len(all_recipients)}{attachment_note}")
        lines.append(f"Detail: {detail}")

        return ToolResult(success=True, content="\n".join(lines))

    # ------------------------------------------------------------------ #
    #  Provider: Mailgun (HTTP API)
    # ------------------------------------------------------------------ #

    async def _send_mailgun(
        self,
        mail_cfg: Any,
        from_header: str,
        to: list[str],
        cc: list[str],
        bcc: list[str],
        subject: str,
        body: str,
        html: str,
        attachments: list[dict[str, Any]],
        timeout: float,
    ) -> str:
        api_key = (
            (mail_cfg.mailgun_api_key or "").strip()
            or os.environ.get("MAILGUN_API_KEY", "").strip()
        )
        if not api_key:
            raise ValueError(
                "Missing Mailgun API key. Set tools.send_mail.mailgun_api_key "
                "in config or MAILGUN_API_KEY env var."
            )
        domain = (mail_cfg.mailgun_domain or "").strip()
        if not domain:
            raise ValueError("Missing tools.send_mail.mailgun_domain config.")

        base_url = (mail_cfg.mailgun_base_url or "https://api.mailgun.net/v3").rstrip("/")
        url = f"{base_url}/{domain}/messages"

        data: dict[str, Any] = {
            "from": from_header,
            "to": to,
            "subject": subject,
        }
        if cc:
            data["cc"] = cc
        if bcc:
            data["bcc"] = bcc
        if body:
            data["text"] = body
        if html:
            data["html"] = html

        files = [
            ("attachment", (att["filename"], att["content"], att["mime_type"]))
            for att in attachments
        ]

        response = await self.client.post(
            url,
            auth=("api", api_key),
            data=data,
            files=files or None,
            timeout=timeout,
        )
        response.raise_for_status()
        resp_data = response.json()
        return str(resp_data.get("message", f"HTTP {response.status_code}"))

    # ------------------------------------------------------------------ #
    #  Provider: SendGrid (HTTP API)
    # ------------------------------------------------------------------ #

    async def _send_sendgrid(
        self,
        mail_cfg: Any,
        from_address: str,
        from_name: str,
        to: list[str],
        cc: list[str],
        bcc: list[str],
        subject: str,
        body: str,
        html: str,
        attachments: list[dict[str, Any]],
        timeout: float,
    ) -> str:
        api_key = (
            (mail_cfg.sendgrid_api_key or "").strip()
            or os.environ.get("SENDGRID_API_KEY", "").strip()
        )
        if not api_key:
            raise ValueError(
                "Missing SendGrid API key. Set tools.send_mail.sendgrid_api_key "
                "in config or SENDGRID_API_KEY env var."
            )
        url = (mail_cfg.sendgrid_base_url or "https://api.sendgrid.com/v3/mail/send").strip()

        personalization: dict[str, Any] = {
            "to": [{"email": addr} for addr in to],
        }
        if cc:
            personalization["cc"] = [{"email": addr} for addr in cc]
        if bcc:
            personalization["bcc"] = [{"email": addr} for addr in bcc]

        from_obj: dict[str, str] = {"email": from_address}
        if from_name:
            from_obj["name"] = from_name

        content: list[dict[str, str]] = []
        if body:
            content.append({"type": "text/plain", "value": body})
        if html:
            content.append({"type": "text/html", "value": html})

        payload: dict[str, Any] = {
            "personalizations": [personalization],
            "from": from_obj,
            "subject": subject,
            "content": content,
        }

        if attachments:
            payload["attachments"] = [
                {
                    "content": base64.b64encode(att["content"]).decode("ascii"),
                    "filename": att["filename"],
                    "type": att["mime_type"],
                    "disposition": "attachment",
                }
                for att in attachments
            ]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response = await self.client.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        # SendGrid returns 202 with empty body on success.
        return f"HTTP {response.status_code} Accepted"

    # ------------------------------------------------------------------ #
    #  Provider: SMTP
    # ------------------------------------------------------------------ #

    async def _send_smtp(
        self,
        mail_cfg: Any,
        from_header: str,
        from_address: str,
        to: list[str],
        cc: list[str],
        bcc: list[str],
        subject: str,
        body: str,
        html: str,
        attachments: list[dict[str, Any]],
    ) -> str:
        host = (mail_cfg.smtp_host or "localhost").strip()
        port = int(mail_cfg.smtp_port or 587)
        username = (mail_cfg.smtp_username or "").strip()
        password = (mail_cfg.smtp_password or "").strip()
        use_tls = bool(mail_cfg.smtp_use_tls)

        msg = EmailMessage()
        msg["From"] = from_header
        msg["To"] = ", ".join(to)
        if cc:
            msg["Cc"] = ", ".join(cc)
        msg["Subject"] = subject

        # Body content.
        if body and html:
            msg.set_content(body)
            msg.add_alternative(html, subtype="html")
        elif html:
            msg.set_content(html, subtype="html")
        else:
            msg.set_content(body)

        # Attach files.
        for att in attachments:
            maintype, _, subtype = att["mime_type"].partition("/")
            msg.add_attachment(
                att["content"],
                maintype=maintype,
                subtype=subtype or "octet-stream",
                filename=att["filename"],
            )

        # BCC goes in envelope only, not in headers.
        all_recipients = to + cc + bcc

        # Prefer aiosmtplib (native async); fall back to smtplib in a thread.
        try:
            import aiosmtplib  # type: ignore[import-untyped]

            await aiosmtplib.send(
                msg,
                hostname=host,
                port=port,
                username=username or None,
                password=password or None,
                start_tls=use_tls,
                recipients=all_recipients,
            )
        except ImportError:
            import smtplib

            def _send_sync() -> None:
                server = smtplib.SMTP(host, port, timeout=60)
                try:
                    if use_tls:
                        server.starttls()
                    if username and password:
                        server.login(username, password)
                    server.sendmail(from_address, all_recipients, msg.as_string())
                finally:
                    server.quit()

            await asyncio.to_thread(_send_sync)

        return f"SMTP delivery to {host}:{port}"

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
