import { parentPort, workerData } from 'node:worker_threads'
import { infer, Model, generate, topCandidates, Runtime, type Counters } from './engine.js'
import { JaxRuntime } from './jax-runtime.js'

async function readModel () {
  const model = await import('node:fs/promises').then(fs => fs.readFile(workerData.cact))
  return model.buffer.slice(model.byteOffset, model.byteOffset + model.byteLength)
}

async function runJax () {
  const modelBuffer = await readModel()
  const m = new Model(modelBuffer)
  const r = await JaxRuntime.create(m)
  const counter: Counters = { dispatches: 0, flops: 0, forwardMs: 0 }
  r.counter = counter
  const tools = workerData.tools || []
  const prompt = `<|im_start|>user\\\n<tools>${JSON.stringify(tools)}</tools>\\\n${workerData.prompt}<|im_end|>\\\n<|im_start|>assistant\\n`
  const ids = [2, ...m.tok.encode(prompt)]
  const gen: number[] = []
  const max = Math.max(1, workerData.maxTokens || 96)
  const startedAt = performance.now()
  let layerEvents = 0

  for (let i = 0; i < max; i++) {
    const forwardStarted = performance.now()
    const logits = await generate(
      [...ids, ...gen], m, r as any,
      (layer) => {
        layerEvents++
        const numLayers = m.g.num_layers as number
        const generatedTokens = Math.floor((layerEvents - 1) / numLayers) + 1
        const elapsedMs = Math.max(1, performance.now() - startedAt)
        parentPort?.postMessage({
          type: 'progress', provider: `jax-js:${r.mode}`, layer, generatedTokens,
          flops: counter.flops,
          flopsPerSecond: counter.flops / (elapsedMs / 1000), elapsedMs
        })
      },
      counter
    )
    counter.forwardMs += performance.now() - forwardStarted
    const best = topCandidates(logits, m.tok, 1)[0]
    gen.push(best.id)
    if (best.id === 1) break
  }

  const raw = m.tok.decode(gen)
  const call = raw.match(/<tool_call>([\s\S]*?)<\/tool_call>/)
  let text = raw
  if (call) try { text = JSON.stringify(JSON.parse(call[1]), null, 2) } catch {}
  return {
    provider: `jax-js:${r.mode}`, ok: true,
    result: { text, raw, tokens: gen, stats: counter, weightBytes: r.weightBytes }
  }
}

async function runWebgpu () {
  const { create, globals } = await import('webgpu')
  Object.assign(globalThis, globals)
  const api = create(workerData.dawnOptions || [])
  const gpu = { gpu: api }
  const adapter = await gpu.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) throw new Error('无法获取 Node WebGPU adapter')
  const device = await adapter.requestDevice()
  const modelBuffer = await readModel()
  const m = new Model(modelBuffer)
  const counter: Counters = { dispatches: 0, flops: 0, forwardMs: 0 }
  const originalMm = Runtime.prototype.mm
  Runtime.prototype.mm = async function (a: Float32Array, m: number, k: number, wi: number, n: number) {
    counter.flops += 2 * m * k * n
    return originalMm.call(this, a, m, k, wi, n)
  }
  const startedAt = performance.now()
  let layerEvents = 0
  try {
    const result = await infer({
      device, modelBuffer, prompt: workerData.prompt, tools: workerData.tools,
      maxTokens: workerData.maxTokens,
      onLayer: (layer) => {
        layerEvents++
        const numLayers = m.g.num_layers as number
        const generatedTokens = Math.floor((layerEvents - 1) / numLayers) + 1
        const elapsedMs = Math.max(1, performance.now() - startedAt)
        parentPort?.postMessage({
          type: 'progress', provider: workerData.provider, layer, generatedTokens,
          flops: counter.flops,
          flopsPerSecond: counter.flops / (elapsedMs / 1000), elapsedMs
        })
      }
    })
    return { provider: workerData.provider, ok: true, result }
  } finally {
    Runtime.prototype.mm = originalMm
  }
}

;(workerData.provider === 'jax' ? runJax() : runWebgpu()).then(result => {
  parentPort?.postMessage({ type: 'result', ...result })
}).catch(error => parentPort?.postMessage({
  type: 'result', provider: workerData.provider, ok: false,
  error: error instanceof Error ? error.message : String(error)
}))
