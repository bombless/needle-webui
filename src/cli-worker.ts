import { parentPort, workerData } from 'node:worker_threads'
import { create, globals } from 'webgpu'
import { inferFixed } from './infer.js'
import { Model, Runtime } from './engine.js'

Object.assign(globalThis, globals)

async function main () {
  const api = create(workerData.dawnOptions || [])
  const gpu = { gpu: api }
  const adapter = await gpu.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) throw new Error('无法获取 Node WebGPU adapter')
  const device = await adapter.requestDevice()
  const model = await import('node:fs/promises').then(fs => fs.readFile(workerData.cact))
  const modelBuffer = model.buffer.slice(model.byteOffset, model.byteOffset + model.byteLength)
  const modelInfo = new Model(modelBuffer)
  let liveFlops = 0
  const originalMm = Runtime.prototype.mm
  Runtime.prototype.mm = async function (a: Float32Array, m: number, k: number, wi: number, n: number) {
    liveFlops += 2 * m * k * n
    return originalMm.call(this, a, m, k, wi, n)
  }
  const startedAt = performance.now()
  let layerEvents = 0
  try {
    const result = await inferFixed({
      device,
      modelBuffer,
      prompt: workerData.prompt,
      tools: workerData.tools,
      maxTokens: workerData.maxTokens,
      onLayer: (layer) => {
        layerEvents++
        const numLayers = modelInfo.g.num_layers as number
        const generatedTokens = Math.floor((layerEvents - 1) / numLayers) + 1
        const elapsedMs = Math.max(1, performance.now() - startedAt)
        parentPort?.postMessage({
          type: 'progress',
          provider: workerData.provider,
          layer,
          generatedTokens,
          flops: liveFlops,
          flopsPerSecond: liveFlops / (elapsedMs / 1000),
          elapsedMs
        })
      }
    })
    parentPort?.postMessage({
      type: 'result',
      provider: workerData.provider,
      ok: true,
      result
    })
  } finally {
    Runtime.prototype.mm = originalMm
  }
}

main().catch(error => parentPort?.postMessage({
  type: 'result',
  provider: workerData.provider,
  ok: false,
  error: error instanceof Error ? error.message : String(error)
}))
