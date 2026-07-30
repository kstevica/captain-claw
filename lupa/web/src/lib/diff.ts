/** Minimal line diff (LCS) for the "what changed between rounds" view. */

export interface DiffLine { type: 'same' | 'add' | 'del'; text: string }

const MAX_LINES = 1200  // LCS is O(n·m); cap keeps worst case ~1.4M cells

export function lineDiff(a: string, b: string): DiffLine[] {
  const A = a.split('\n').slice(0, MAX_LINES)
  const B = b.split('\n').slice(0, MAX_LINES)
  const n = A.length, m = B.length
  // dp[i][j] = LCS length of A[i:], B[j:]
  const w = m + 1
  const dp = new Uint16Array((n + 1) * w)
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i * w + j] = A[i] === B[j]
        ? dp[(i + 1) * w + j + 1] + 1
        : Math.max(dp[(i + 1) * w + j], dp[i * w + j + 1])
    }
  }
  const out: DiffLine[] = []
  let i = 0, j = 0
  while (i < n && j < m) {
    if (A[i] === B[j]) { out.push({ type: 'same', text: A[i] }); i++; j++ }
    else if (dp[(i + 1) * w + j] >= dp[i * w + j + 1]) {
      out.push({ type: 'del', text: A[i] }); i++
    } else { out.push({ type: 'add', text: B[j] }); j++ }
  }
  while (i < n) { out.push({ type: 'del', text: A[i] }); i++ }
  while (j < m) { out.push({ type: 'add', text: B[j] }); j++ }
  return out
}

/** Collapse long unchanged runs so the changes stand out. */
export function collapseSame(lines: DiffLine[], context = 2): (DiffLine | { type: 'skip'; count: number })[] {
  const out: (DiffLine | { type: 'skip'; count: number })[] = []
  let run: DiffLine[] = []
  const flush = (isEnd: boolean, isStart: boolean) => {
    const keepHead = isStart ? 0 : context
    const keepTail = isEnd ? 0 : context
    if (run.length <= keepHead + keepTail + 1) { out.push(...run) }
    else {
      out.push(...run.slice(0, keepHead))
      out.push({ type: 'skip', count: run.length - keepHead - keepTail })
      out.push(...run.slice(run.length - keepTail))
    }
    run = []
  }
  let seenChange = false
  for (const l of lines) {
    if (l.type === 'same') { run.push(l) }
    else {
      if (run.length) flush(false, !seenChange)
      seenChange = true
      out.push(l)
    }
  }
  if (run.length) flush(true, !seenChange)
  return out
}
