import { parentPort, workerData } from 'node:worker_threads'
import { create, globals } from 'webgpu'
import { infer, Model } from './webgpu-main.js'

Object.assign(globalThis, globals)

async function main () {
  const api = create(workerData.dawnOptions || [])
  const gpu = { gpu: api }
  const adapter = await gpu.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) throw new Error('无法获取 Node WebGPU adapter')
  const device = await adapter.requestDevice()
  const model = await import('node:fs/promises').then(fs => fs.readFile(workerData.cact))
  const modelInfo = new Model(model.buffer.slice(model.byteOffset, model.byteOffset + model.byteLength))
  const startedAt = performance.now()
  let layerEvents = 0
  const result = await infer({
    device,
    modelBuffer: model.buffer.slice(model.byteOffset, model.byteOffset + model.byteLength),
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
        flops: 0,
        flopsPerSecond: 0,
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
}

main().catch(error => parentPort?.postMessage({
  type: 'result',
  provider: workerData.provider,
  ok: false,
  error: error instanceof Error ? error.message : String(error)
}))
