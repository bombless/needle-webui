import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'

const execFileAsync = promisify(execFile)

export async function loadPklWeights (pklPath: string) {
  const dir = await mkdtemp(join(tmpdir(), 'needle-pkl-'))
  const binPath = join(dir, 'weights.f32')
  const metaPath = join(dir, 'meta.json')
  const script = resolve(fileURLToPath(new URL('../scripts/export-pkl.py', import.meta.url)))
  try {
    await execFileAsync('python', [script, pklPath, binPath, metaPath], {
      windowsHide: true,
      maxBuffer: 1024 * 1024
    })
    const [raw, metaRaw] = await Promise.all([readFile(binPath), readFile(metaPath, 'utf8')])
    const meta = JSON.parse(metaRaw)
    if (!Array.isArray(meta.shapes) || meta.shapes.length !== meta.num_weights)
      throw Error('PKL 导出元数据无效')
    if (raw.byteLength % 4 !== 0)
      throw Error(`PKL 导出权重不是 float32 对齐数据 (${raw.byteLength} bytes)`)
    const all = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength))
    const weights: Float32Array[] = []
    let off = 0
    for (const shape of meta.shapes) {
      const size = shape.reduce((a: number, b: number) => a * b, 1)
      if (!Number.isInteger(size) || off + size > all.length)
        throw Error(`PKL 权重 shape 无效: ${JSON.stringify(shape)}`)
      weights.push(all.slice(off, off + size))
      off += size
    }
    if (off !== all.length)
      throw Error(`PKL 权重数据多出 ${all.length - off} 个 float32`)
    return { weights, meta }
  } finally {
    await rm(dir, { recursive: true, force: true }).catch(() => {})
  }
}
