import { init, defaultDevice, numpy as np, type Device } from '@jax-js/jax'
import type { Counters } from './engine.js'

type BackendMode = 'webgpu' | 'wasm'

/**
 * Runtime adapter for jax-js. The model parser and the rest of the Needle
 * implementation stay unchanged; only GEMM is delegated to jax-js.
 *
 * jax-js owns its device buffers, so this deliberately converts the existing
 * Float32Array weights/activations at the GEMM boundary. This keeps the
 * existing custom WebGPU runtime available for A/B comparison.
 */
export class JaxRuntime {
  m: any
  w: Float32Array[] = []
  counter: Counters | null = null
  weightBytes = 0
  peakBytes = 0
  mode: BackendMode

  private constructor (m: any, mode: BackendMode) {
    this.m = m
    this.mode = mode
    for (const t of m.t.filter((x: any) => x.dtype !== 4)) {
      let a: Float32Array
      if (t.dtype === 1) {
        const d = new DataView(t.data.buffer, t.data.byteOffset, t.data.byteLength)
        a = new Float32Array(t.data.byteLength / 2)
        for (let i = 0; i < a.length; i++) {
          const bits = d.getUint16(i * 2, true)
          const s = bits >>> 15
          const e = (bits >>> 10) & 31
          const f = bits & 1023
          a[i] = !e
            ? ((s ? -1 : 1) * Math.pow(2, -14) * f) / 1024
            : (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024)
        }
      } else if (t.dtype === 2) {
        a = new Float32Array(t.data.buffer, t.data.byteOffset, t.data.byteLength / 4).slice()
      } else {
        // CQ tensors are already decoded by the same helper used by Runtime.
        // The model constructor keeps a private decoder, so reuse Runtime once
        // in the factory below instead of duplicating CQ decoding here.
        throw Error('JaxRuntime requires decoded model weights')
      }
      this.w.push(a)
      this.weightBytes += a.byteLength
    }
    this.peakBytes = this.weightBytes
  }

  static async create (m: any, decodedWeights: Float32Array[], preferred: BackendMode = 'webgpu') {
    let mode = preferred
    try {
      const wanted: Device = preferred === 'webgpu' ? 'webgpu' : 'wasm'
      const available = await init(wanted)
      if (!available.includes(wanted)) {
        if (preferred === 'webgpu') {
          const fallback = await init('wasm')
          if (!fallback.includes('wasm')) throw Error('jax-js 没有可用 backend')
          mode = 'wasm'
        } else {
          throw Error('jax-js WASM backend 不可用')
        }
      }
      defaultDevice(mode)
    } catch (error) {
      throw Error(`jax-js 初始化失败：${(error as Error)?.message || error}`)
    }

    const r = Object.create(JaxRuntime.prototype) as JaxRuntime
    r.m = m
    r.mode = mode
    r.w = decodedWeights
    r.weightBytes = decodedWeights.reduce((n, x) => n + x.byteLength, 0)
    r.peakBytes = r.weightBytes
    r.counter = null
    return r
  }

  async mm (a: Float32Array, m: number, k: number, wi: number, n: number) {
    this.counter && (this.counter.dispatches++, (this.counter.flops += 2 * m * k * n))
    const aa = np.array(a).reshape([m, k])
    const bb = np.array(this.w[wi]).reshape([n, k])
    // Needle's weights are [out, in], so this is A @ B^T.
    const yy = np.einsum('mk,nk->mn', aa, bb)
    const data = await yy.data()
    const out = new Float32Array(data)
    yy.dispose()
    this.peakBytes = Math.max(this.peakBytes, this.weightBytes + a.byteLength + out.byteLength)
    const bad = out.findIndex(x => !Number.isFinite(x))
    if (bad >= 0) throw Error(`jax-js GEMM 输出在索引 ${bad} 出现 NaN/Infinity（${m}x${k} · ${k}x${n}）`)
    return out
  }
}
