import { EXAMPLE_TOOLS, EXAMPLE_PROMPT } from './example.js'
import { Model, Runtime, generate, topCandidates, type Counters } from './engine.js'

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
let gpu: any = null
let model: Model | null = null
let runtime: Runtime | null = null
let cancelled = false

function formatBytes (n: number) {
  if (n >= 1073741824) return `${(n / 1073741824).toFixed(2)} GiB`
  if (n >= 1048576) return `${(n / 1048576).toFixed(1)} MiB`
  return `${Math.round(n / 1024)} KiB`
}
function formatFlops (n: number) {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}G`
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`
  return `${Math.round(n)}`
}
function add (role: string, text: string) {
  const messages = $<HTMLDivElement>('messages')
  if (messages.querySelector('.empty')) messages.innerHTML = ''
  const element = document.createElement('div')
  element.className = `message ${role}`
  element.textContent = text
  messages.appendChild(element)
  messages.scrollTop = messages.scrollHeight
  return element
}
function showDebug (promptIds: number[], generated: number[], steps: unknown[]) {
  $<HTMLPreElement>('debugPrompt').textContent = JSON.stringify(promptIds)
  $<HTMLPreElement>('debugGenerated').textContent = JSON.stringify(generated)
  $<HTMLPreElement>('debugSteps').textContent = JSON.stringify(steps, null, 2)
}
async function run () {
  const query = $<HTMLTextAreaElement>('query').value.trim()
  if (!query || !model || !runtime) return
  const currentRuntime = runtime
  const send = $<HTMLButtonElement>('send')
  send.disabled = true
  add('user', query)
  const assistant = add('assistant', '生成中…')
  const tools = JSON.parse($<HTMLTextAreaElement>('tools').value || '[]')
  const prompt = `<|im_start|>user\\
<tools>${JSON.stringify(tools)}</tools>\\
${query}<|im_end|>\\
<|im_start|>assistant\\
`
  const ids = [2, ...model.tok.encode(prompt)]
  const generated: number[] = []
  const steps: unknown[] = []
  const max = Number($<HTMLInputElement>('maxTokens').value) || 96
  const k = Math.max(1, Math.min(20, Number($<HTMLInputElement>('topK').value) || 5))
  const started = performance.now()
  const counter: Counters = { dispatches: 0, flops: 0, forwardMs: 0 }
  cancelled = false
  currentRuntime.counter = counter
  $<HTMLButtonElement>('cancel').disabled = false
  showDebug(ids, [], [])
  try {
    let previous = performance.now()
    for (let i = 0; i < max; i++) {
      const forwardStart = performance.now()
      const logits = await generate([...ids, ...generated], model, currentRuntime, layer => {
        $<HTMLDivElement>('progress').textContent = `WebGPU Needle · layer ${layer}/${model!.g.num_layers}`
      }, counter, () => cancelled)
      const now = performance.now()
      const candidates = topCandidates(logits, model.tok, k)
      const best = candidates[0]
      counter.forwardMs += now - forwardStart
      steps.push({ step: i + 1, selected: best, topK: candidates })
      generated.push(best.id)
      showDebug(ids, generated, steps)
      assistant.textContent = model.tok.decode(generated) || '生成中…'
      $<HTMLDivElement>('metrics').textContent = `${i === 0 ? '首 token prefill' : 'decode'} · 已生成 ${i + 1}/${max} · 当前 ${(1000 / Math.max(1, now - previous)).toFixed(2)} tok/s · 总 ${(generated.length / (Math.max(1, now - started) / 1000)).toFixed(2)} tok/s · ${counter.dispatches} shader dispatch · ${formatFlops(counter.flops)} FLOPs · 显存 ${formatBytes(currentRuntime.peakBytes)}（权重 ${formatBytes(currentRuntime.weightBytes)}）`
      previous = now
      if (best.id === 1) break
      let repeated = 1
      for (let j = generated.length - 2; j >= 0 && generated[j] === best.id; j--) repeated++
      if (repeated >= 8) break
    }
    const text = model.tok.decode(generated)
    const call = text.match(/<tool_call>([\s\S]*?)<\/tool_call>/)
    let output = text
    if (call) try { output = JSON.stringify(JSON.parse(call[1]), null, 2) } catch {}
    assistant.textContent = output || '(empty; see token diagnostics)'
    $<HTMLSpanElement>('timing').textContent = `· ${Math.round(performance.now() - started)} ms · WebGPU`
  } catch (error: any) {
    assistant.textContent = '推理失败：' + (error?.message || error)
    showDebug(ids, generated, steps)
  } finally {
    currentRuntime.counter = null
    $<HTMLButtonElement>('cancel').disabled = true
    $<HTMLDivElement>('progress').textContent = ''
    send.disabled = false
  }
}

$('send').addEventListener('click', run)
$('cancel').addEventListener('click', () => { cancelled = true; $<HTMLDivElement>('metrics').textContent = '正在停止…' })
$('query').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); run() }
})
$('loadDemo').addEventListener('click', () => {
  $<HTMLTextAreaElement>('tools').value = JSON.stringify(EXAMPLE_TOOLS, null, 2)
  $<HTMLTextAreaElement>('query').value = EXAMPLE_PROMPT
  $<HTMLTextAreaElement>('query').focus()
})
$('modelFile').addEventListener('change', async event => {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (!file) return
  $<HTMLDivElement>('modelInfo').textContent = '正在解析 CQ 权重并上传 GPU…'
  try {
    model = new Model(await file.arrayBuffer())
    runtime = new Runtime(gpu, model)
    $<HTMLDivElement>('modelInfo').textContent = `${file.name} · ${model.g.d_model}d · ${model.g.num_layers} 层 · ${model.t.length} tensors · 文件权重 ${formatBytes(model.t.reduce((sum, tensor) => sum + tensor.data.byteLength, 0))} · GPU ${formatBytes(runtime.weightBytes)}`
  } catch (error: any) {
    model = null; runtime = null
    $<HTMLDivElement>('modelInfo').textContent = '加载失败：' + (error?.message || error)
  }
})

void (async () => {
  try {
    if (!('gpu' in navigator)) throw Error('当前浏览器没有 WebGPU')
    const adapter = await (navigator as any).gpu.requestAdapter()
    if (!adapter) throw Error('无法获取 WebGPU adapter')
    gpu = await adapter.requestDevice()
    const info = adapter.info || {}
    $<HTMLDivElement>('gpuStatus').textContent = `WebGPU 就绪 · ${info.vendor || 'unknown'} ${info.architecture || ''}`
  } catch (error: any) {
    $<HTMLDivElement>('gpuStatus').textContent = 'WebGPU 不可用：' + (error?.message || error)
  }
})()
