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

if (!S[name]) { console.log('scenarios:', Object.keys(S).join(' ')); process.exit(name ? 1 : 0) }
await S[name]()
