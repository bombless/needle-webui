import { Worker } from 'node:worker_threads'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

type Tool = {
  name: string
  description: string
  parameters: Record<string, unknown>
}

type Args = {
  cact: string
  tools: unknown
  prompt: string
  timeout: number
  maxTokens: number
  providers: string[]
}

const EXAMPLE_TOOLS: Tool[] = [
  {
    name: 'set_lights',
    description: '调节灯光',
    parameters: {
      type: 'object',
      properties: {
        room: { type: 'string' },
        brightness: { type: 'integer', minimum: 0, maximum: 100 }
      },
      required: ['room', 'brightness']
    }
  }
]

const EXAMPLE_PROMPT = '调用 set_lights, 房间 1, 亮度 0'

function value (argv: string[], name: string, fallback?: string) {
  const i = argv.indexOf(name)
  return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback
}

async function args (argv: string[]): Promise<Args> {
  const cact = value(argv, '--cact', value(argv, '--model'))
  if (!cact)
    throw new Error('用法: npm run cli -- --cact model.cact [--tools tools.json] [--prompt "text"]')

  const toolsPath = value(argv, '--tools')
  const tools = toolsPath
    ? JSON.parse(await readFile(resolve(toolsPath), 'utf8'))
    : EXAMPLE_TOOLS
  const prompt = value(argv, '--prompt', EXAMPLE_PROMPT)!

  return {
    cact: resolve(cact),
    tools,
    prompt,
    timeout: Number(value(argv, '--timeout', '120000')),
    maxTokens: Number(value(argv, '--max-tokens', '96')),
    providers: (value(argv, '--providers', 'webgpu') || 'webgpu').split(',').map(x => x.trim()).filter(Boolean)
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
  const a = await args(process.argv.slice(2))
  const results = await Promise.all(a.providers.map(provider => runWorker(provider, a)))
  process.stdout.write(JSON.stringify({ prompt: a.prompt, results }, null, 2) + '\n')
  process.exitCode = results.every((x: any) => x.ok) ? 0 : 1
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error))
  process.exitCode = 2
}
