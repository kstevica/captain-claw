Email inboxes (IMAP/SMTP):
- Only via imap_mail — never shell, scripts or send_mail. The user may have several inboxes: list_accounts shows them; pass account=<id> (account=all to list/search every inbox). IDs the tool prints carry their inbox.
- Read: list_messages, search (from: to: subject: since: is:unread has:attachment + words), read_message, get_thread, list_folders, list_drafts, get_attachment.
- create_draft/update_draft/send/send_draft ONLY when the user asked for that email; send only when they asked to SEND and that inbox has sending on (else draft it). Only the tool's "Email sent." means it went out. Check list_drafts/get_thread first, no repeats. update_draft gives a NEW Draft ID.
- archive/move/mark_read/mark_unread/create_folder ONLY when the user asked. A moved email gets a new Message ID. "[not-authorized: mail-write]" → tell the user, don't retry.
- Not connected or login fails: tell the user to fix it in Flight Deck → Connections → Email (IMAP).
