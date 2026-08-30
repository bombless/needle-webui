import { parentPort, workerData } from 'node:worker_threads'
import { create, globals } from 'webgpu'
import { infer } from './webgpu-main.js'

Object.assign(globalThis, globals)

async function main () {
  const api = create(workerData.dawnOptions || [])
  const gpu = { gpu: api }
  const adapter = await gpu.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) throw new Error('无法获取 Node WebGPU adapter')
  const device = await adapter.requestDevice()
  const model = await import('node:fs/promises').then(fs => fs.readFile(workerData.cact))
  const result = await infer({
    device,
    modelBuffer: model.buffer.slice(model.byteOffset, model.byteOffset + model.byteLength),
    prompt: workerData.prompt,
    tools: workerData.tools,
    maxTokens: workerData.maxTokens
  })
  parentPort?.postMessage({ provider: workerData.provider, ok: true, result })
}

main().catch(error => parentPort?.postMessage({
  provider: workerData.provider,
  ok: false,
  error: error instanceof Error ? error.message : String(error)
}))
