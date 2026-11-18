import { useEffect, useState } from 'react'
import { getRunData, type RunData } from '../../services/firstMate'
import { errText } from './ui'

/** One run's extra data, loaded once for its row: the Extra data tab's count and its
 *  list share it. */
export function useRunData(runId: string) {
  const [data, setData] = useState<RunData | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let stop = false
    getRunData(runId).then((d) => { if (!stop) setData(d) }).catch((e) => { if (!stop) setError(errText(e)) })
    return () => { stop = true }
  }, [runId])
  return { data, setData, error, setError }
}

export type RunDataState = ReturnType<typeof useRunData>
