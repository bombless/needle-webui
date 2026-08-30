export type Counters = { dispatches: number; flops: number; forwardMs: number }
export class Model {
  g: any
  t: any[]
  tok: any
  constructor(buf: ArrayBuffer)
}
export class Runtime {
  counter: Counters | null
  weightBytes: number
  peakBytes: number
  constructor(device: any, model: Model)
}
export function generate(tokens: number[], model: Model, runtime: Runtime, step: (layer: number) => void, counters: Counters, isCancelled?: () => boolean): Promise<Float32Array>
export function topCandidates(logits: Float32Array, tok: any, k: number): any[]
