import { Worker } from 'node:worker_threads'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

type Args = {
  cact: string
  tools: string
  prompt: string
  timeout: number
  maxTokens: number
  providers: string[]
}

function args (argv: string[]): Args {
  const value = (name: string, fallback?: string) => {
    const i = argv.indexOf(name)
    return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback
  }
  const cact = value('--cact', value('--model'))
  const tools = value('--tools')
  const prompt = value('--prompt')
  if (!cact || !tools || prompt === undefined)
    throw new Error('用法: npm run cli -- --cact model.cact --tools tools.json --prompt "text"')
  return {
    cact: resolve(cact),
    tools: resolve(tools),
    prompt,
    timeout: Number(value('--timeout', '120000')),
    maxTokens: Number(value('--max-tokens', '96')),
    providers: (value('--providers', 'webgpu') || 'webgpu').split(',').map(x => x.trim()).filter(Boolean)
  }
}

function runWorker (provider: string, a: Args) {
  return new Promise((resolveResult) => {
    const worker = new Worker(resolve(fileURLToPath(new URL('./cli-worker.ts', import.meta.url))), {
      workerData: { ...a, provider, dawnOptions: provider === 'webgpu' ? [] : [`backend=${provider}`] }
    })
    let settled = false
    const finish = (result: unknown) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolveResult(result)
    }
    const timer = setTimeout(() => {
      worker.terminate()
      finish({ provider, ok: false, timeout: true, error: `超过 ${a.timeout} ms` })
    }, a.timeout)
    worker.once('message', (message) => {
      finish(message)
      void worker.terminate()
    })
    worker.once('error', error => finish({ provider, ok: false, error: error.message }))
    worker.once('exit', code => {
      if (code !== 0) finish({ provider, ok: false, error: `worker exited with code ${code}` })
    })
  })
}

try {
  const a = args(process.argv.slice(2))
  const results = await Promise.all(a.providers.map(provider => runWorker(provider, a)))
  process.stdout.write(JSON.stringify({ prompt: a.prompt, results }, null, 2) + '\n')
  process.exitCode = results.every((x: any) => x.ok) ? 0 : 1
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error))
  process.exitCode = 2
}
