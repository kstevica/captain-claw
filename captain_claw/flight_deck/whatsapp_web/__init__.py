"""WhatsApp Web — the user's own WhatsApp account, linked as a device, held by
Flight Deck for their agents (plan: ``docs/whatsapp-web-plan.md``).

This package is the **service**: one process, started and watched by Flight
Deck (:mod:`captain_claw.flight_deck.whatsapp_web_supervisor`), that holds a
neonize client (whatsmeow, the WhatsApp multi-device protocol, as a Python
library) per linked user and answers Flight Deck on a loopback port:

* :mod:`.link_state` — one link's state and what each WhatsApp event does to
  it. Pure, so it is tested without WhatsApp.
* :mod:`.parse` — WhatsApp protos → rows (message, edit, revoke, react, skip).
* :mod:`.store` — one link's messages, sealed with fd_crypto: chats, contacts,
  reactions, LID folding, search, the per-agent news cursor, retention.
* :mod:`.engine` — the links: link, pairing code, cancel, unlink, resume after
  a restart, the pairing timeout, the events Flight Deck turns into bells,
  ingest into the store and the reads agents make.
* :mod:`.kept_history` — what's kept from links that ended (a banned number, a
  switch to another one), until the user deletes it.
* :mod:`.neonize_adapter` — the only module that touches neonize: the version
  pin, the libmagic stand-in, and every event handler wrapped so an exception
  is logged (neonize drops them silently).
* :mod:`.api` — the loopback HTTP API (bearer secret from ``service.secret``).
* ``__main__`` — the process: lock, umask, port, run.

The history the phone sends after linking and every message since are kept for
the user's agents (the ``whatsapp_web`` tool). They send only through
:meth:`.engine.Engine.send`: when the user turned sending on, to whitelisted
chats that already exist, within the limits, signed. Nothing is ever marked
read, reacted to or announced as online.

Data, under ``<FD data dir>/whatsapp_web/`` (0700, files 0600):

* ``service.lock`` / ``service.json`` / ``service.secret`` / ``service.log``
* ``<user_id>/session.sqlite3`` — whatsmeow's session: the device keys, so
  whoever has the file has the account
* ``<user_id>/state.json`` — the link's state (no keys, no messages)
* ``<user_id>/store.sqlite3`` — the messages kept (sealed; JIDs plain)
* ``.kept/<user_id>/<entry>/`` — the same store from an earlier link, and
  ``kept.json`` saying which number it was and how much it holds
"""

from __future__ import annotations

DIR_NAME = "whatsapp_web"
NEONIZE_VERSION = "0.5.2"
