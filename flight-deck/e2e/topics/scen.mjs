// Topic-divider scenarios for the chat store (src/stores/chatStore.ts), one
// per process: node --import ./register.mjs scen.mjs <name>  (run.sh runs all)
import { setup, check, done, makeLog, divider, turn, settle } from './lib.mjs'

const name = process.argv[2]
const S = {}
const count = (view, item) => view.filter((v) => v === item).length
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b)

// Signed in, first visit, the agent's replay lands before the log does.
S.firstVisitReplayFirst = async () => {
  const h = await setup({ auth: true })
  h.ev.connected(); h.ev.welcome('s1'); h.ev.replay([['user', 'u1'], ['assistant', 'a1']]); h.ev.replayDone()
  await h.release(); await h.flush()
  check("divider first, the session's turns under it", same(h.view(), ['[new:s1]', 'user:u1', 'assistant:a1']), h.view())
  check('the log keeps that order', same(h.log.view(), ['[new:s1]', 'user:u1', 'assistant:a1']), h.log.view())
  done()
}

// Signed in, first visit, the log lands first.
S.firstVisitHistoryFirst = async () => {
  const h = await setup({ auth: true, holdHistory: false })
  await h.release()
  h.ev.connected(); h.ev.welcome('s1'); h.ev.replay([['user', 'u1'], ['assistant', 'a1']]); h.ev.replayDone()
  await h.flush()
  check("divider first, the session's turns under it", same(h.view(), ['[new:s1]', 'user:u1', 'assistant:a1']), h.view())
  check('the log keeps that order', same(h.log.view(), ['[new:s1]', 'user:u1', 'assistant:a1']), h.log.view())
  done()
}

// Signed in, coming back (another device, a new sign-in): nothing new.
S.revisitReplayFirst = async () => {
  const log = makeLog([divider('new', 's1'), turn('user', 'u1'), turn('assistant', 'a1')])
  const h = await setup({ auth: true, log })
  h.ev.connected(); h.ev.welcome('s1'); h.ev.replay([['user', 'u1'], ['assistant', 'a1']]); h.ev.replayDone()
  await h.release(); await h.flush()
  check('no new divider, no repeated turns', same(h.view(), ['[new:s1]', 'user:u1', 'assistant:a1']), h.view())
  check('nothing written to the log', log.rows.length === 3, log.view())
  h.ev.sessionInfo('s1', 2)
  check('the turn-end session_info marks nothing', count(h.view(), '[resume:s1]') === 0 && count(h.view(), '[new:s1]') === 1, h.view())
  done()
}

// Signed in, coming back after a reply finished with no tab open.
S.revisitMissedReply = async () => {
  const log = makeLog([divider('new', 's1'), turn('user', 'u1'), turn('assistant', 'a1')])
  const h = await setup({ auth: true, log })
  h.ev.connected(); h.ev.welcome('s1')
  h.ev.replay([['user', 'u1'], ['assistant', 'a1'], ['user', 'u2'], ['assistant', 'a2']]); h.ev.replayDone()
  await h.release(); await h.flush()
  const want = ['[new:s1]', 'user:u1', 'assistant:a1', 'user:u2', 'assistant:a2']
  check('the missing turns appended, no divider', same(h.view(), want), h.view())
  check('only the missing turns written', same(log.view(), want), log.view())
  done()
}

// Signed in, a log from before dividers existed: no turn shown twice.
S.preDividerLog = async () => {
  const log = makeLog([turn('user', 'u1'), turn('assistant', 'a1')])
  const h = await setup({ auth: true, log })
  h.ev.connected(); h.ev.welcome('s1'); h.ev.replay([['user', 'u1'], ['assistant', 'a1']]); h.ev.replayDone()
  await h.release(); await h.flush()
  const v = h.view()
  check('no turn repeated', count(v, 'user:u1') === 1 && count(v, 'assistant:a1') === 1, v)
  check('one divider for the session', count(v, '[new:s1]') === 1, v)
  done()
}

// Signed in: the lane moved to another session (elsewhere, or /new here)
// after the replay but before the log loaded: the replay's missing turns stay
// with their own session, above the new divider.
for (const [label, move] of [
  ['sessionMovedDuringLoad', (h) => h.ev.sessionInfo('s2', 0)],
  ['commandNewDuringLoad', (h) => h.ev.commandResult('/new', 'New session created: s2', 's2')],
]) {
  S[label] = async () => {
    const log = makeLog([divider('new', 's1'), turn('user', 'u1'), turn('assistant', 'a1')])
    const h = await setup({ auth: true, log })
    h.ev.connected(); h.ev.welcome('s1')
    h.ev.replay([['user', 'u1'], ['assistant', 'a1'], ['user', 'u2'], ['assistant', 'a2']]); h.ev.replayDone()
    move(h)
    await h.release(); await h.flush()
    check("s1's missing turns stay above s2's divider",
      same(log.view(), ['[new:s1]', 'user:u1', 'assistant:a1', 'user:u2', 'assistant:a2', '[new:s2]']), log.view())
    done()
  }
}

// Signed in, first visit: a reconnect onto a fresh (empty) session before the
// log loaded leaves the earlier session's replay — not filed under the new one.
S.reconnectEmptySessionDuringLoad = async () => {
  const h = await setup({ auth: true })
  h.ev.connected(); h.ev.welcome('s1'); h.ev.replay([['user', 'u1'], ['assistant', 'a1']]); h.ev.replayDone()
  h.ev.disconnected(); h.ev.connected(); h.ev.welcome('s2'); h.ev.replayDone()
  await h.release(); await h.flush()
  const v = h.log.view()
  check("s1's turns not filed under s2", v.indexOf('[new:s2]') > v.indexOf('assistant:a1'), v)
  done()
}

// Signed in, reading the past (hasNewer) when the agent comes back on another
// session: back at the newest page, the next turn's end marks it.
S.hasNewerWelcomeOtherSession = async () => {
  const rows = [divider('new', 's1')]
  for (let i = 0; i < 300; i++) { rows.push(turn('user', 'u' + i)); rows.push(turn('assistant', 'a' + i)) }
  const log = makeLog(rows)
  const h = await setup({ auth: true, log, holdHistory: false })
  h.ev.connected(); h.ev.welcome('s1'); h.ev.replayDone(); await h.release(); await h.flush()
  await h.st().jumpToTopic(h.key, 1); await settle()
  h.ev.disconnected(); h.ev.connected(); h.ev.welcome('s2'); h.ev.replayDone()
  await h.st().loadNewer(h.key); await settle()
  h.ev.user('q'); h.ev.reply('r'); h.ev.sessionInfo('s2', 2); h.ev.ready()
  await h.flush()
  check('s2 gets a divider', h.log.view().some((v) => v.endsWith(':s2]')), h.log.view().slice(-4))
  done()
}

// Signed in: the agent moved to other sessions on another channel, and /new here.
S.switchElsewhereAuthOn = async () => {
  const log = makeLog([divider('new', 's1'), turn('user', 'u1'), turn('assistant', 'a1')])
  const h = await setup({ auth: true, log, holdHistory: false })
  h.ev.connected(); h.ev.welcome('s1'); await h.release(); await h.flush()
  h.ev.sessionInfo('s2', 3)
  check('a resumed session gets its divider', h.view().at(-1) === '[resume:s2]', h.view())
  h.ev.sessionInfo('s3', 0)
  check('a new session gets its divider', h.view().at(-1) === '[new:s3]', h.view())
  h.ev.commandResult('/new', 'New session created: s4', 's4')
  check('/new typed here gets its divider', h.view().at(-1) === '[new:s4]', h.view())
  done()
}

// No sign-in (the desktop app): after a reload the transcript is the agent's
// replay; the turn-end session_info for that session marks nothing.
S.reloadAuthOff = async () => {
  const h = await setup({ auth: false, holdHistory: false })
  h.ev.connected(); h.ev.welcome('s1'); h.ev.replay([['user', 'u1'], ['assistant', 'a1']]); h.ev.replayDone()
  h.ev.reply('a2'); h.ev.sessionInfo('s1', 4); h.ev.ready()
  check('no "Back on an earlier topic"', !h.view().some((v) => v.startsWith('[')), h.view())
  done()
}

// No sign-in: moved to another session elsewhere, or /new here.
S.switchAuthOff = async () => {
  const h = await setup({ auth: false, holdHistory: false })
  h.ev.connected(); h.ev.welcome('s1'); h.ev.replay([['user', 'u1'], ['assistant', 'a1']]); h.ev.replayDone()
  h.ev.sessionInfo('s2', 3)
  check('a resumed session gets its divider', h.view().at(-1) === '[resume:s2]', h.view())
  h.ev.sessionInfo('s2', 4)
  check('…once', count(h.view(), '[resume:s2]') === 1, h.view())
  h.ev.commandResult('/new', 'New session created: s3', 's3')
  check('/new typed here gets its divider', h.view().at(-1) === '[new:s3]', h.view())
  done()
}

// No sign-in: /new here moved the lane to s2; the socket came back on s1 (an
// agent restart, a switch elsewhere). s1's next turn end marks it.
S.reconnectOtherSessionAuthOff = async () => {
  const h = await setup({ auth: false, holdHistory: false })
  h.ev.connected(); h.ev.welcome('s1'); h.ev.replay([['user', 'u1'], ['assistant', 'a1']]); h.ev.replayDone()
  h.ev.sessionInfo('s2', 0); h.ev.commandResult('/new', 'New session created: s2', 's2')
  h.ev.user('u2'); h.ev.reply('a2'); h.ev.sessionInfo('s2', 2); h.ev.ready()
  h.ev.disconnected(); h.ev.connected(); h.ev.welcome('s1')
  h.ev.replay([['user', 'u1'], ['assistant', 'a1']]); h.ev.replayDone()
  h.ev.user('u3'); h.ev.reply('a3'); h.ev.sessionInfo('s1', 4); h.ev.ready()
  const last = [...h.view()].reverse().find((v) => v.startsWith('['))
  check('the last divider is for s1, after [new:s2]', !!last && last.endsWith(':s1]'), h.view())
  done()
}

// ── A long chat: the session's divider is older than the loaded window ──
//
// cohort001 (2026-11): a log with many tool rows puts the session's only
// divider beyond the newest 500 rows. A reload then marked "New topic" for the
// session the chat was already on, and the replay merge, anchored on that new
// divider, saved the agent's whole session a second time (58 rows at once).

/** A log of one session: its divider, then `pairs` turns — more rows than a page. */
function longLog(pairs = 300) {
  const rows = [divider('new', 's1')]
  for (let i = 0; i < pairs; i++) { rows.push(turn('user', 'u' + i)); rows.push(turn('assistant', 'a' + i)) }
  return makeLog(rows)
}
/** The agent's replay of the session's last `n` turns. */
const tail = (pairs, n) => {
  const out = []
  for (let i = pairs - n; i < pairs; i++) { out.push(['user', 'u' + i]); out.push(['assistant', 'a' + i]) }
  return out
}

for (const [label, replayFirst] of [['longLogHistoryFirst', false], ['longLogReplayFirst', true]]) {
  S[label] = async () => {
    const log = longLog()
    const rows = log.rows.length
    const h = await setup({ auth: true, log, holdHistory: replayFirst })
    if (!replayFirst) await h.release()
    h.ev.connected(); h.ev.welcome('s1'); h.ev.replay(tail(300, 30)); h.ev.replayDone()
    if (replayFirst) await h.release()
    await h.flush()
    check('nothing written to the log', log.rows.length === rows, log.view().slice(rows - 2))
    check('no divider for the session it was already on', !h.view().some((v) => v.startsWith('[')), h.view().filter((v) => v.startsWith('[')))
    check('no turn shown twice', count(h.view(), 'user:u299') === 1 && count(h.view(), 'user:u280') === 1, h.view().slice(-4))
    done()
  }
}

// The same long chat with a reply it missed: only that reply is added.
S.longLogMissedReply = async () => {
  const log = longLog()
  const rows = log.rows.length
  const h = await setup({ auth: true, log, holdHistory: false })
  await h.release()
  h.ev.connected(); h.ev.welcome('s1'); h.ev.replay([...tail(300, 10), ['user', 'u300'], ['assistant', 'a300']]); h.ev.replayDone()
  await h.flush()
  check('only the missing turns written', same(log.view().slice(rows), ['user:u300', 'assistant:a300']), log.view().slice(rows))
  done()
}

// Back on the main chat after a cron run's own session marked a divider here
// (a scheduled job runs in its own session): the turns above it are not added again.
S.otherSessionDividerLast = async () => {
  const log = makeLog([divider('new', 's1'), turn('user', 'u1'), turn('assistant', 'a1'), turn('user', 'u2'),
    turn('assistant', 'a2'), divider('new', 'cron1'), turn('assistant', 'cron result')])
  const h = await setup({ auth: true, log, holdHistory: false })
  await h.release()
  h.ev.connected(); h.ev.welcome('s1'); h.ev.replay([['user', 'u1'], ['assistant', 'a1'], ['user', 'u2'], ['assistant', 'a2']]); h.ev.replayDone()
  await h.flush()
  const v = log.view()
  check('no turn written twice', count(v, 'user:u1') === 1 && count(v, 'assistant:a2') === 1, v)
  check('marked as back on s1', v.at(-1) === '[resume:s1]', v)
  done()
}

// Back on a session whose turns are all older than the window (another session
// filled the newest page meanwhile): its replay can't be checked here and isn't
// written again; only a turn newer than the window is.
S.resumedSessionOlderThanWindow = async () => {
  const rows = [divider('new', 's1'), turn('user', 'old question'), turn('assistant', 'old answer'), divider('new', 's2')]
  for (let i = 0; i < 300; i++) { rows.push(turn('user', 'q' + i)); rows.push(turn('assistant', 'r' + i)) }
  const log = makeLog(rows)
  log.rows.slice(0, 3).forEach((r) => { r.created_at = '2026-10-20T08:00:00Z' })   // s1, two days earlier
  const before = log.rows.length
  const h = await setup({ auth: true, log, holdHistory: false })
  await h.release()
  h.ev.connected(); h.ev.welcome('s1')
  h.emit('replay_batch', { messages: [
    { type: 'chat_message', role: 'user', content: 'old question', replay: true, timestamp: '2026-10-20T08:00:00Z' },
    { type: 'chat_message', role: 'assistant', content: 'old answer', replay: true, timestamp: '2026-10-20T08:00:01Z' },
    { type: 'chat_message', role: 'user', content: 'asked on the phone', replay: true, timestamp: '2026-10-23T08:00:00Z' },
  ] })
  h.ev.replayDone()
  await h.flush()
  check('only the newer turn written, under "back on s1"', same(log.view().slice(before), ['[resume:s1]', 'user:asked on the phone']), log.view().slice(before))
  done()
}

// The server's backstop: the chat took a turn for missing that the log has
// after all (beyond anything the chat can see) — not written, and gone from view.
S.serverKnowsHealedTurn = async () => {
  const long = 'a reply long enough that the log would never hold it twice by chance'
  const rows = [divider('new', 's1'), turn('assistant', long)]
  for (let i = 0; i < 300; i++) { rows.push(turn('user', 'u' + i)); rows.push(turn('assistant', 'a' + i)) }
  const log = makeLog(rows)
  const before = log.rows.length
  const h = await setup({ auth: true, log, holdHistory: false })
  await h.release()
  h.ev.connected(); h.ev.welcome('s1'); h.ev.replay([...tail(300, 5), ['assistant', long]]); h.ev.replayDone()
  await h.flush(); await h.flush()
  check('nothing written', log.rows.length === before, log.view().slice(before))
  check('not shown again at the bottom', h.view().at(-1) === 'assistant:a299', h.view().slice(-2))
  done()
}

// A turn the log really is missing goes in flagged, so the server can drop
// one it already has (db.add_chat_messages).
S.healedRowsFlagged = async () => {
  const log = makeLog([divider('new', 's1'), turn('user', 'u1'), turn('assistant', 'a1')])
  const h = await setup({ auth: true, log, holdHistory: false })
  await h.release()
  h.ev.connected(); h.ev.welcome('s1'); h.ev.replay([['user', 'u1'], ['assistant', 'a1'], ['user', 'u2'], ['assistant', 'a2']]); h.ev.replayDone()
  await h.flush()
  const added = log.rows.slice(3).map((r) => JSON.parse(r.metadata || '{}').healed === true)
  check('the healed rows say so', added.length === 2 && added.every(Boolean), log.rows.slice(3))
  h.ev.user('u3'); await h.flush()
  check('a live row does not', JSON.parse(log.rows.at(-1).metadata || '{}').healed === undefined, log.rows.at(-1))
  done()
}

// ── A turn with a picture (the agent's copy holds more than was typed) ──
//
// Glasses, 2026-11-25: the agent keeps a turn with a picture as the picture's
// line, its guidance for the picture, then the words; the log has the words,
// then the picture's line. The chat took the agent's copy for missing and
// saved it again, guidance and all.

const PIC = '[Attached image: /w/saved/downloads/s1/image-1.png]'
const GUIDE = '(The image(s) above are attached and you can SEE them directly — describe/analyze them from what you see.)'
const ASKED = 'I accepted the invitation, how do I get back to that channel?'
const agentCopy = `${PIC}\n${GUIDE}\n${ASKED}`
const logCopy = `${ASKED}\n\n${PIC}`

// A turn kept before the replay carried the words: still matched.
S.pictureTurnKeptBeforeWords = async () => {
  const log = makeLog([divider('new', 's1'), turn('user', logCopy), turn('assistant', 'a1')])
  const h = await setup({ auth: true, log, holdHistory: false })
  await h.release()
  h.ev.connected(); h.ev.welcome('s1'); h.ev.replay([['user', agentCopy], ['assistant', 'a1']]); h.ev.replayDone()
  await h.flush()
  check('nothing written', log.rows.length === 3, log.view())
  check('shown once', count(h.view(), `user:${logCopy}`) === 1 && h.view().length === 3, h.view())
  done()
}

// The replay says what was typed: matched on that.
S.pictureTurnWithWords = async () => {
  const log = makeLog([divider('new', 's1'), turn('user', logCopy), turn('assistant', 'a1')])
  const h = await setup({ auth: true, log, holdHistory: false })
  await h.release()
  h.ev.connected(); h.ev.welcome('s1'); h.ev.replay([['user', agentCopy, { words: ASKED }], ['assistant', 'a1']]); h.ev.replayDone()
  await h.flush()
  check('nothing written', log.rows.length === 3, log.view())
  check('shown once', h.view().length === 3, h.view())
  done()
}

// A picture turn the log really is missing (sent from the phone): saved as
// it was typed, with the picture — never the agent's guidance.
S.missingPictureTurnSavedAsTyped = async () => {
  const log = makeLog([divider('new', 's1'), turn('user', 'u1'), turn('assistant', 'a1')])
  const h = await setup({ auth: true, log, holdHistory: false })
  await h.release()
  h.ev.connected(); h.ev.welcome('s1')
  h.ev.replay([['user', 'u1'], ['assistant', 'a1'], ['user', agentCopy, { words: ASKED }], ['assistant', 'a2']]); h.ev.replayDone()
  await h.flush()
  const saved = log.rows.slice(3)
  check('the turn and its reply written', saved.length === 2, log.view())
  check('as typed, with the picture', saved[0]?.content === logCopy, saved[0]?.content)
  check('no guidance shown', !h.view().some((v) => v.includes('SEE them')), h.view())
  check('healed from s1', JSON.parse(saved[0]?.metadata || '{}').agent_session === 's1', saved[0]?.metadata)
  done()
}

// A tab that was away while another tab wrote the session's short lines
// (glasses, 2026-11-23): its replay takes them for missing; the log keeps one.
S.staleTabShortLines = async () => {
  const log = makeLog([divider('new', 's0'), turn('user', 'u0'), turn('assistant', 'a0')])
  const h = await setup({ auth: true, log, holdHistory: false })
  await h.release()
  h.ev.connected(); h.ev.welcome('s0'); h.ev.replay([['user', 'u0'], ['assistant', 'a0']]); h.ev.replayDone()
  await h.flush()
  // Another tab: the agent moved to s1, and the turns went in there.
  log.add([divider('new', 's1'), turn('user', 'E-ink?'), turn('assistant', 'a1'), turn('user', 'no')])
  const before = log.rows.length
  h.ev.disconnected(); h.ev.connected(); h.ev.welcome('s1')
  h.ev.replay([['user', 'E-ink?'], ['assistant', 'a1'], ['user', 'no']]); h.ev.replayDone()
  await h.flush(); await h.flush()
  const again = log.rows.slice(before).filter((r) => r.role === 'user')
  check('the short lines not written again', again.length === 0, log.view().slice(before))
  done()
}

// ── A message sent as the socket dies (held again, sent once) ──
//
// Not dividers, but the same store and fake socket: a frame the agent never
// took goes back on hold when the socket goes, and is sent again once the
// chat is connected — unless the reconnect's replay shows the agent took it.

/** A connected lane whose socket takes frames (they are recorded). */
async function liveLane() {
  const h = await setup({ auth: true, holdHistory: false })
  await h.release()
  h.ev.connected(); h.ev.welcome('s1'); h.ev.replay([['user', 'u1'], ['assistant', 'a1']]); h.ev.replayDone()
  await h.flush()
  const sent = []
  h.ws.send = (content) => { sent.push(content); return `web_${sent.length}` }
  return { h, sent }
}

// The frame died with the socket (an agent woke from sleep, a dropped link).
S.heldFrameDiedWithSocket = async () => {
  const { h, sent } = await liveLane()
  h.st().sendMessage(h.key, 'thank you')
  await settle()
  check('sent once', same(sent, ['thank you']), sent)
  h.ev.disconnected(); await settle()
  check('back on hold, bubble gone', h.ses().held.length === 1 && count(h.view(), 'user:thank you') === 0, { held: h.ses().held, view: h.view() })
  h.ev.connected(); h.ev.welcome('s1'); h.ev.replay([['user', 'u1'], ['assistant', 'a1']]); h.ev.replayDone()
  await h.flush()
  check('sent again on the new socket', same(sent, ['thank you', 'thank you']), sent)
  check('nothing left on hold, one bubble', h.ses().held.length === 0 && count(h.view(), 'user:thank you') === 1, { held: h.ses().held, view: h.view() })
  done()
}

// The agent took it, but the socket went before its "thinking" came back.
S.heldFrameTakenBeforeDrop = async () => {
  const { h, sent } = await liveLane()
  h.st().sendMessage(h.key, 'hello?')
  await settle()
  h.ev.disconnected(); await settle()
  h.ev.connected(); h.ev.welcome('s1')
  h.ev.replay([['user', 'u1'], ['assistant', 'a1'], ['user', 'hello?'], ['assistant', 'hi']]); h.ev.replayDone()
  await h.flush()
  check('not sent twice', same(sent, ['hello?']), sent)
  check('nothing left on hold', h.ses().held.length === 0, h.ses().held)
  check('the turn shows once', count(h.view(), 'user:hello?') === 1 && count(h.view(), 'assistant:hi') === 1, h.view())
  done()
}

// The socket wasn't open after all when the message went (closing).
S.heldWhenSocketNotOpen = async () => {
  const { h, sent } = await liveLane()
  h.ws.send = () => undefined
  h.st().sendMessage(h.key, 'are you there')
  await settle()
  check('held, not busy, no bubble', h.ses().held.length === 1 && !h.ses().busy && count(h.view(), 'user:are you there') === 0,
    { held: h.ses().held, busy: h.ses().busy, view: h.view() })
  h.ws.send = (content) => { sent.push(content); return `web_${sent.length}` }
  h.ev.disconnected(); h.ev.connected(); h.ev.welcome('s1'); h.ev.replay([['user', 'u1'], ['assistant', 'a1']]); h.ev.replayDone()
  await h.flush()
  check('sent once connected', same(sent, ['are you there']), sent)
  done()
}

// ── Several written while the agent is busy (they go as one) ──

/** A live lane on a turn of its own ("first" sent, the agent on it). */
async function busyLane() {
  const { h, sent } = await liveLane()
  h.st().sendMessage(h.key, 'first')
  await settle()
  h.emit('status', { status: 'thinking' })
  await settle()
  return { h, sent }
}

// Two written during a turn: once it ends they go together, one turn.
S.heldGoAsOne = async () => {
  const { h, sent } = await busyLane()
  h.st().sendMessage(h.key, 'also this')
  h.st().sendMessage(h.key, 'and this')
  await settle()
  check('both held while busy', h.ses().held.length === 2 && same(sent, ['first']), { held: h.ses().held, sent })
  h.ev.reply('done'); h.ev.ready()
  await h.flush()
  check('sent once, as one message', same(sent, ['first', 'also this\n\nand this']), sent)
  check('nothing left on hold', h.ses().held.length === 0, h.ses().held)
  check('one bubble for both', count(h.view(), 'user:also this\n\nand this') === 1, h.view())
  done()
}

// A new-topic one waits for the turn after (the agent acts on a cue only at
// the start of a message), and takes what was written after it along.
S.heldNewTopicOwnTurn = async () => {
  const { h, sent } = await busyLane()
  h.st().sendMessage(h.key, 'a')
  h.st().sendMessage(h.key, 'new topic: b')
  h.st().sendMessage(h.key, 'c')
  await settle()
  h.ev.reply('done'); h.ev.ready()
  await h.flush()
  check('the first batch goes alone', same(sent, ['first', 'a']), sent)
  check('the new-topic batch still waits', h.ses().held.length === 2, h.ses().held)
  h.emit('status', { status: 'thinking' }); h.ev.reply('ok'); h.ev.ready()
  await h.flush()
  check('then the new topic, with what came after it', same(sent, ['first', 'a', 'new topic: b\n\nc']), sent)
  check('nothing left on hold', h.ses().held.length === 0, h.ses().held)
  done()
}

// "Send all at next step": one note to the running turn; refused (the turn
// had just ended), it waits again as that one message and goes by itself.
S.heldSendAllInTurn = async () => {
  const { h, sent } = await busyLane()
  const btw = []
  h.ws.sendBtw = (content) => { btw.push(content) }
  h.st().sendMessage(h.key, 'x')
  h.st().sendMessage(h.key, 'y')
  await settle()
  check('handed over', h.st().sendHeldInTurn(h.key) === true, null)
  check('one note for both', same(btw, ['x\n\ny']), btw)
  check('nothing left on hold', h.ses().held.length === 0, h.ses().held)
  h.ev.reply('done'); h.ev.ready()
  h.emit('command_result', { command: '/btw', busy: false, content: '' })
  await h.flush()
  check('refused: sent as one message once free', same(sent, ['first', 'x\n\ny']), sent)
  check('nothing left on hold after', h.ses().held.length === 0, h.ses().held)
  done()
}

if (!S[name]) { console.log('scenarios:', Object.keys(S).join(' ')); process.exit(name ? 1 : 0) }
await S[name]()
