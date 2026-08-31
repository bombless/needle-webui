import { Worker } from 'node:worker_threads'
import { readFile, access } from 'node:fs/promises'
import { resolve, dirname, basename, extname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { EXAMPLE_TOOLS, EXAMPLE_PROMPT } from './example.js'

type Args = {
  cact?: string
  pkl?: string
  tools: unknown
  prompt: string
  timeout: number
  maxTokens: number
  providers: string[]
  dumpPrefill: boolean
}

function value (argv: string[], name: string, fallback?: string) {
  const i = argv.indexOf(name)
  return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback
}

async function findTokenizer (pkl: string) {
  const stem = basename(pkl, extname(pkl))
  const candidates = [
    join(dirname(pkl), `${stem}.cact`),
    resolve(`${stem}.cact`),
    resolve('needle2.cact')
  ]
  for (const candidate of candidates) {
    try { await access(candidate); return candidate } catch {}
  }
  throw new Error(`--pkl 需要 tokenizer；未找到 ${stem}.cact，请同时指定 --cact tokenizer.cact`)
}

async function args (argv: string[]): Promise<Args> {
  const pkl = value(argv, '--pkl')
  const cactArg = value(argv, '--cact', value(argv, '--model'))
  if (!pkl && !cactArg)
    throw new Error('用法: npm run cli -- --cact model.cact [--jax] 或 npm run cli -- --jax --pkl model.pkl')

  const cact = pkl ? await findTokenizer(resolve(pkl)) : resolve(cactArg!)
  const toolsPath = value(argv, '--tools')
  const tools = toolsPath
    ? JSON.parse(await readFile(resolve(toolsPath), 'utf8'))
    : EXAMPLE_TOOLS
  const prompt = value(argv, '--prompt', EXAMPLE_PROMPT)!
  const providers = argv.includes('--jax')
    ? ['jax']
    : (value(argv, '--providers', 'webgpu') || 'webgpu').split(',').map(x => x.trim()).filter(Boolean)

  if (pkl && !argv.includes('--jax'))
    throw new Error('--pkl 目前只支持 --jax')

  return {
    cact,
    pkl: pkl ? resolve(pkl) : undefined,
    tools,
    prompt,
    timeout: Number(value(argv, '--timeout', '120000')),
    maxTokens: Number(value(argv, '--max-tokens', '96')),
    providers,
    dumpPrefill: argv.includes('--dump-prefill')
  }
}

function formatFlopsPerSecond (n: number) {
  if (n >= 1e15) return `${(n / 1e15).toFixed(2)} PFLOP/s`
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)} TFLOP/s`
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GFLOP/s`
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MFLOP/s`
  if (n >= 1e3) return `${(n / 1e3).toFixed(2)} KFLOP/s`
  return `${Math.round(n)} FLOP/s`
}

function formatFlops (n: number) {
  if (n >= 1e15) return `${(n / 1e15).toFixed(2)} PFLOPs`
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)} TFLOPs`
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GFLOPs`
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MFLOPs`
  return `${Math.round(n)} FLOPs`
}

function runWorker (provider: string, a: Args) {
  return new Promise((resolveResult) => {
    const worker = new Worker(resolve(fileURLToPath(new URL('./cli-worker.ts', import.meta.url))), {
      workerData: { ...a, provider, dawnOptions: provider === 'webgpu' ? [] : [`backend=${provider}`] }
    })
    let settled = false
    let lastProgressAt = 0
    let lastPrintedToken = 0
    const finish = (result: unknown) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolveResult(result)
    }
    const timer = setTimeout(() => {
      worker.terminate()
      finish({ provider, ok: false, timeout: true, error: `timeout ${a.timeout} ms` })
    }, a.timeout)
    worker.on('message', (message) => {
      if (message?.type === 'progress') {
        const now = Date.now()
        if (now - lastProgressAt < 250 && message.generatedTokens === lastPrintedToken) return
        lastProgressAt = now
        lastPrintedToken = message.generatedTokens
        process.stdout.write(
          `[${provider}] 运行中 · 已生成 ${message.generatedTokens}/${a.maxTokens} token · ` +
          `${formatFlopsPerSecond(message.flopsPerSecond)} · ` +
          `${formatFlops(message.flops)} · layer ${message.layer}\n`
        )
        return
      }
      if (message?.type === 'result') finish(message)
      else finish(message)
    })
    worker.once('error', error => finish({ provider, ok: false, error: error.message }))
    worker.once('exit', code => {
      if (code !== 0) finish({ provider, ok: false, error: `worker exited with code ${code}` })
    })
  })
}

try {
  const a = await args(process.argv.slice(2))
  const results = await Promise.all(a.providers.map(provider => runWorker(provider, a)))
  process.stdout.write(JSON.stringify({ prompt: a.prompt, results }, null, 2) + '\n')
  process.exitCode = results.every((x: any) => x.ok) ? 0 : 1
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error))
  process.exitCode = 2
}

