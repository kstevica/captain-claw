/** A token count, short: 950 · 1.2k · 48k. */
export function fmtTokens(n?: number): string {
  const v = n || 0
  if (v >= 1000) return `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k`
  return String(v)
}

/** A file's size the short way: "340 KB", "2.4 MB" (never "0 KB"). */
export function sizeText(size: number): string {
  return size >= 1_000_000 ? `${(size / 1_000_000).toFixed(1)} MB` : `${Math.max(1, Math.round(size / 1000))} KB`
}
