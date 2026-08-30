import { Model, Runtime, generate, topCandidates, type Counters } from './engine.js'
import { buildPrompt } from './prompt.js'

export async function inferFixed (options: {
  device: any
  modelBuffer: ArrayBuffer
  prompt: string
  tools?: unknown
  maxTokens?: number
  onLayer?: (layer: number) => void
}) {
  const m = new Model(options.modelBuffer)
  const r = new Runtime(options.device, m)
  const prompt = buildPrompt(options.tools || [], options.prompt)
  const ids = [2, ...m.tok.encode(prompt)]
  const gen: number[] = []
  const counter: Counters = { dispatches: 0, flops: 0, forwardMs: 0 }
  const max = Math.max(1, options.maxTokens || 96)
  for (let i = 0; i < max; i++) {
    const started = performance.now()
    const logits = await generate([...ids, ...gen], m, r, options.onLayer || (() => {}), counter)
    counter.forwardMs += performance.now() - started
    const best = topCandidates(logits, m.tok, 1)[0]
    gen.push(best.id)
    if (best.id === 1) break
  }
  return m.tok.decode(gen)
}
