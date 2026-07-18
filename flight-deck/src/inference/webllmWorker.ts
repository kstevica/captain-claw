// WebLLM engine host — runs inside a WebWorker so inference never blocks
// the Flight Deck UI thread. The main-thread side talks to it through
// CreateWebWorkerMLCEngine (see localInference.ts).
import { WebWorkerMLCEngineHandler } from '@mlc-ai/web-llm'

const handler = new WebWorkerMLCEngineHandler()
self.onmessage = (msg: MessageEvent) => handler.onmessage(msg)
