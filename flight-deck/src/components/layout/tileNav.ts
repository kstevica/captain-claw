/** Up or down: the nearest tile in the next row that way (the rows aren't all the same length). */
export function tileInNextRow(tiles: HTMLElement[], from: HTMLElement, way: number) {
  const box = from.getBoundingClientRect()
  const middle = box.left + box.width / 2
  let best: HTMLElement | undefined
  let bestScore = Infinity
  for (const tile of tiles) {
    const b = tile.getBoundingClientRect()
    const rows = (b.top - box.top) * way
    if (rows < 1) continue
    const score = rows * 1000 + Math.abs(b.left + b.width / 2 - middle)
    if (score < bestScore) {
      best = tile
      bestScore = score
    }
  }
  return best
}
