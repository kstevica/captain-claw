// The Today home's list rules. Run: node --test src/components/home/todayLists.test.mjs
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { allFailed, allSettled, priorityTasks } from './todayLists.ts'

const task = (id, lane, urgency, extra = {}) => ({
  id, lane, urgency, status: 'open', owes_next: lane === 'working' ? 'agent' : 'them',
  title: id, last_event_at: 100, updated_at: 100, ...extra,
})

test('an in-progress (working lane) task is a priority, with no waiting work at all', () => {
  assert.deepEqual(priorityTasks([task('draft-price-list', 'working', 10)], []).map((t) => t.id), ['draft-price-list'])
})

test('in progress and waiting on others together, once each, most urgent first', () => {
  const working = [task('w1', 'working', 20), task('w2', 'working', 5)]
  const waiting = [task('them', 'waiting', 30), task('w1', 'working', 20)]
  assert.deepEqual(priorityTasks(working, waiting).map((t) => t.id), ['them', 'w1', 'w2'])
})

test('never the person\'s move or a snoozed task, whatever a lane read returns', () => {
  const waiting = [task('mine', 'needs_me', 80), task('later', 'snoozed', 99), task('them', 'waiting', 1)]
  assert.deepEqual(priorityTasks([], waiting).map((t) => t.id), ['them'])
})

test('equal urgency: the most recently moved first; closed tasks never', () => {
  const working = [task('old', 'working', 5, { last_event_at: 10 }), task('new', 'working', 5, { last_event_at: 50 }),
    task('closed', 'working', 9, { status: 'done' })]
  assert.deepEqual(priorityTasks(working, []).map((t) => t.id), ['new', 'old'])
})

test('a section is settled only when every source answered or failed', () => {
  const loading = { data: null, error: '' }
  const answered = { data: [], error: '' }
  const failed = { data: null, error: 'Failed' }
  assert.equal(allSettled(answered, loading), false)
  assert.equal(allSettled(answered, failed), true)
  assert.equal(allFailed(failed, failed), true)
  assert.equal(allFailed(failed, answered), false)
  // A source that failed on a reload still shows its earlier answer.
  assert.equal(allFailed({ data: [], error: 'Failed' }, failed), false)
})
