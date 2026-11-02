import { useEffect, useState } from 'react'
import { schedulerApi } from '../../services/schedulerApi'

export interface SchedulePreview { next: number[]; error: string }

/** When a schedule would next run, or why it can't, as Flight Deck reads it. */
export function useSchedulePreview(schedule: string, tz: string): SchedulePreview | null {
  const [preview, setPreview] = useState<SchedulePreview | null>(null)
  useEffect(() => {
    let cancelled = false
    const t = setTimeout(() => {
      schedulerApi.preview(schedule, tz)
        .then((p) => { if (!cancelled) setPreview({ next: p.next, error: p.error }) })
        .catch(() => { if (!cancelled) setPreview(null) })
    }, 200)
    return () => { cancelled = true; clearTimeout(t) }
  }, [schedule, tz])
  return preview
}
