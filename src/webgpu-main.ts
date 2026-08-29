type T = {
  dtype: number
  shape: number[]
  group: number
  bits: number
  data: Uint8Array
}
const $ = <TEl extends HTMLElement>(id: string) =>
  document.getElementById(id) as TEl
const TAG = 0x05e12a83,
  H = 120,
  R = 44,
  FP16 = 1,
  FP32 = 2,
  CQ = 3,
  RAW = 4,
  STORAGE = 128,
  COPY_SRC = 4,
  COPY_DST = 8,
  MAP_READ = 1,
  UNIFORM = 64,
  SHADER = 4
const gpuStatus = $<HTMLDivElement>('gpuStatus'),
  msgs = $<HTMLDivElement>('messages'),
  progress = $<HTMLDivElement>('progress'),
  metrics = $<HTMLDivElement>('metrics'),
  cancel = $<HTMLButtonElement>('cancel')
let gpu: any = null,
  model: Model | null = null,
  rt: Runtime | null = null,
  cancelled = false,
  cqNonFiniteScales = 0
const DEBUG_LATENTS = new URLSearchParams(location.search).has('DEBUG_LATENTS')

type Counters = { dispatches: number; flops: number; forwardMs: number }
type Candidate = {
  id: number
  token: string
  logit: number
  probability: number
}
function firstNonFinite (a: ArrayLike<number>) {
  for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) return i
  return -1
}
function dtypeName (dtype: number) {
  return dtype === FP16
    ? 'FP16'
    : dtype === FP32
    ? 'FP32'
    : dtype === CQ
    ? 'CQ'
    : dtype === RAW
    ? 'RAW'
    : `dtype ${dtype}`
}
function dumpLatent (
  layer: number,
  x: Float32Array,
  N: number,
  lanes: number,
  D: number
) {
  if (!DEBUG_LATENTS) return
  const snapshot = x.slice()
  let min = Infinity,
    max = -Infinity,
    sum = 0,
    sumsq = 0
  for (const v of snapshot) {
    min = Math.min(min, v)
    max = Math.max(max, v)
    sum += v
    sumsq += v * v
  }
  const record = {
    layer,
    shape: [N, lanes, D],
    min,
    max,
    mean: sum / snapshot.length,
    rms: Math.sqrt(sumsq / snapshot.length),
    values: snapshot
  }
  const w = window as any
  ;(w.__needleLatents || (w.__needleLatents = [])).push(record)
  console.log(`[latent] layer_${String(layer).padStart(2, '0')}`, {
    layer,
    shape: record.shape,
    min,
    max,
    mean: record.mean,
    rms: record.rms
  })
}
function hf (x: number) {
  const s = x >>> 15,
    e = (x >>> 10) & 31,
    f = x & 1023
  if (!e) return ((s ? -1 : 1) * Math.pow(2, -14) * f) / 1024
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024)
}
function hfIeee (x: number) {
  const s = x >>> 15,
    e = (x >>> 10) & 31,
    f = x & 1023
  if (!e) return ((s ? -1 : 1) * Math.pow(2, -14) * f) / 1024
  if (e === 31) return f ? NaN : (s ? -1 : 1) * Infinity
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024)
}
function fwht (a: Float32Array) {
  for (let n = 1; n < a.length; n <<= 1)
    for (let i = 0; i < a.length; i += n << 1)
      for (let j = 0; j < n; j++) {
        const x = a[i + j],
          y = a[i + j + n]
        a[i + j] = x + y
        a[i + j + n] = x - y
      }
  const q = 1 / Math.sqrt(a.length)
  for (let i = 0; i < a.length; i++) a[i] *= q
}
function unpack (t: T, row: number, inPad: number, rowBytes?: number) {
  const b = t.bits,
    out = new Uint8Array(inPad),
    packedRowBytes = b === 5 ? inPad >> 2 : (inPad * b) / 8,
    stride = rowBytes || packedRowBytes
  if (b === 5) {
    const off = row * stride
    for (let i = 0; i < inPad; i++) {
      const c = (t.data[off + (i >> 2)] >> ((i & 3) * 2)) & 3
      out[i] = c === 3 ? 0 : c + 1
    }
    return out
  }
  const off = row * stride,
    mask = (1 << b) - 1
  for (let i = 0; i < inPad; i++) {
    const bit = i * b,
      bi = off + (bit >> 3),
      sh = bit & 7
    let v = t.data[bi] >> sh
    if (sh + b > 8) v |= t.data[bi + 1] << (8 - sh)
    out[i] = v & mask
  }
  return out
}
function cq (t: T, cb: Float32Array) {
  const out = t.shape[0],
    dim = t.shape[1],
    g = t.group || 128,
    b = t.bits,
    inPad = Math.ceil(dim / g) * g,
    rb = b === 5 ? inPad / 4 : (inPad * b) / 8,
    groups = inPad / g,
    normBytes = groups * 2,
    rowBytes = rb + normBytes,
    dst = new Float32Array(out * dim),
    code =
      b === 2
        ? cb.subarray(0, 4)
        : b === 3
        ? cb.subarray(4, 12)
        : b === 4
        ? cb.subarray(12, 28)
        : new Float32Array(
            [-1.2240064, 0, 1.2240064].map(v => v / Math.sqrt(g))
          )
  if (firstNonFinite(cb) >= 0)
    throw Error(`CQ codebook 在索引 ${firstNonFinite(cb)} 出现非有限值`)
  const dv = new DataView(t.data.buffer, t.data.byteOffset, t.data.byteLength),
    tmp = new Float32Array(g)
  for (let r = 0; r < out; r++) {
    const ids = unpack(t, r, inPad, rowBytes),
      rowOffset = r * rowBytes
    for (let q = 0; q < groups; q++) {
      const scaleBits = dv.getUint16(rowOffset + rb + q * 2, true),
        scale = hf(scaleBits)
      if (!Number.isFinite(scale)) {
        cqNonFiniteScales++
        for (let j = 0; j < g; j++) tmp[j] = 0
        continue
      }
      for (let j = 0; j < g; j++) {
        const v = (code[ids[q * g + j]] || 0) * scale
        if (!Number.isFinite(v))
          throw Error(
            `CQ 解码输入非有限：row=${r} group=${q} value=${j} scale=0x${scaleBits
              .toString(16)
              .padStart(4, '0')}`
          )
        tmp[j] = v
      }
      fwht(tmp)
      for (let j = 0; j < g && q * g + j < dim; j++) {
        if (!Number.isFinite(tmp[j]))
          throw Error(
            `CQ FWHT 输出非有限：row=${r} group=${q} value=${j} scale=0x${scaleBits
              .toString(16)
              .padStart(4, '0')}`
          )
        dst[r * dim + q * g + j] = tmp[j]
      }
    }
  }
  return dst
}

class Tok {
  p: string[] = []
  s: number[] = []
  ty: number[] = []
  first = new Map<string, number[]>()
  constructor (b: Uint8Array) {
    if (b.byteLength < 24)
      throw Error(`tokenizer header truncated (${b.byteLength} bytes)`)
    const d = new DataView(b.buffer, b.byteOffset, b.byteLength),
      n = d.getUint32(0, true)
    let o = 24
    const td = new TextDecoder()
    for (let i = 0; i < n; i++) {
      if (o + 7 > b.byteLength)
        throw Error(
          `tokenizer record ${i} header truncated at ${o}/${b.byteLength}`
        )
      this.s.push(d.getFloat32(o, true))
      this.ty.push(d.getUint8(o + 4))
      const l = d.getUint16(o + 5, true)
      o += 7
      if (o + l > b.byteLength)
        throw Error(
          `tokenizer record ${i} surface truncated at ${o}+${l}/${b.byteLength}`
        )
      const p = td.decode(b.subarray(o, o + l))
      o += l
      this.p.push(p)
      this.first.set(p[0], [...(this.first.get(p[0]) || []), i])
    }
  }
  encode (text: string) {
    const c = Array.from('▁' + text.replaceAll(' ', '▁')),
      n = c.length,
      inf = 1e30,
      dp = new Float64Array(n + 1),
      pr = new Int32Array(n + 1),
      pi = new Int32Array(n + 1)
    dp.fill(inf)
    dp[0] = 0
    for (let i = 0; i < n; i++) {
      for (const id of this.first.get(c[i]) || []) {
        const q = Array.from(this.p[id])
        if (i + q.length > n || q.some((x, j) => x !== c[i + j])) continue
        const v = dp[i] - this.s[id]
        if (v < dp[i + q.length]) {
          dp[i + q.length] = v
          pr[i + q.length] = i
          pi[i + q.length] = id
        }
      }
      if (dp[i + 1] >= inf) {
        const bytes = new TextEncoder().encode(c[i])
        for (const x of bytes) {
          const id = this.p.indexOf(
            `<0x${x.toString(16).padStart(2, '0').toUpperCase()}>`
          )
          if (id >= 0) {
            dp[i + 1] = dp[i] + 20
            pr[i + 1] = i
            pi[i + 1] = id
            break
          }
        }
      }
    }
    const r: number[] = []
    for (let i = n; i > 0 && pr[i] < i; i = pr[i]) r.push(pi[i])
    return r.reverse()
  }
  tokenLabel (id: number) {
    const p = this.p[id] || `<id:${id}>`
    if (this.ty[id] === 4) {
      const m = p.match(/<0x([0-9A-Fa-f]{2})>/)
      return m ? `[byte 0x${m[1].toUpperCase()}]` : p
    }
    return p.replaceAll('▁', '[space]')
  }
  decode (ids: number[]) {
    let s = ''
    for (const id of ids) {
      const p = this.p[id] || ''
      if (this.ty[id] === 4) {
        const m = p.match(/<0x([0-9A-Fa-f]{2})>/)
        if (m) s += String.fromCharCode(parseInt(m[1], 16))
      } else s += p
    }
    return s.replaceAll('▁', ' ').trim()
  }
}

class Model {
  g: Record<string, any>
  t: T[]
  cb: Float32Array
  tok: Tok
  constructor (buf: ArrayBuffer) {
    const d = new DataView(buf),
      raw = new Uint8Array(buf)
    if (d.getUint32(0, true) !== TAG) throw Error('无效 .cact')
    const nt = d.getUint32(4, true),
      cn = d.getUint32(8, true)
    this.cb = new Float32Array(buf.slice(H, H + cn * 4))
    this.g = {}
    ;[
      'vocab_size',
      'd_model',
      'num_heads',
      'num_kv_heads',
      'num_layers',
      'head_dim',
      'max_seq_len',
      'hada_n',
      'mhc_lanes',
      'engram_slots',
      'engram_sub_dim',
      'num_engram_tables',
      'engram_conv_taps',
      'engram_conv_dilation'
    ].forEach((k, i) => (this.g[k] = d.getUint32(20 + i * 4, true)))
    this.g.kv_window = d.getUint32(12, true)
    this.g.kv_bits = d.getUint32(16, true)
    this.g.engram_orders = []
    const no = d.getUint32(76, true)
    for (let i = 0; i < no; i++)
      this.g.engram_orders.push(d.getUint32(80 + i * 4, true))
    this.g.engram_layers = []
    const ns = d.getUint32(96, true)
    for (let i = 0; i < ns; i++)
      this.g.engram_layers.push(d.getUint32(100 + i * 4, true))
    this.g.rope_theta = d.getFloat32(116, true)
    let o = H + cn * 4
    this.t = []
    for (let i = 0; i < nt; i++) {
      const ty = d.getUint8(o),
        nd = d.getUint8(o + 1),
        shape = [] as number[]
      for (let j = 0; j < nd; j++) shape.push(d.getUint32(o + 4 + j * 4, true))
      const off = Number(d.getBigUint64(o + 20, true)),
        nb = Number(d.getBigUint64(o + 28, true))
      if (
        !Number.isSafeInteger(off) ||
        !Number.isSafeInteger(nb) ||
        off < 0 ||
        nb < 0 ||
        off + nb > raw.byteLength
      )
        throw Error(
          `tensor ${i} data out of bounds: offset ${off}, size ${nb}, file ${raw.byteLength}`
        )
      this.t.push({
        dtype: ty,
        shape,
        group: d.getUint32(o + 36, true),
        bits: d.getUint32(o + 40, true),
        data: raw.subarray(off, off + nb)
      })
      o += R
    }
    const rawTok = this.t.find(x => x.dtype === RAW)
    if (!rawTok) throw Error('模型没有 tokenizer')
    this.tok = new Tok(rawTok.data)
  }
}

class Runtime {
  m: Model
  dev: any
  w: Float32Array[] = []
  wb: any[] = []
  pipe: any
  layout: any
  uni: any
  counter: Counters | null = null
  weightBytes = 0
  peakBytes = 0
  constructor (dev: any, m: Model) {
    this.dev = dev
    this.m = m
    for (const t of m.t.filter(x => x.dtype !== RAW)) {
      let a: Float32Array
      if (t.dtype === FP16) {
        const d = new DataView(
          t.data.buffer,
          t.data.byteOffset,
          t.data.byteLength
        )
        a = new Float32Array(t.data.byteLength / 2)
        for (let i = 0; i < a.length; i++)
          a[i] = hfIeee(d.getUint16(i * 2, true))
      } else if (t.dtype === FP32)
        a = new Float32Array(
          t.data.buffer,
          t.data.byteOffset,
          t.data.byteLength / 4
        ).slice()
      else a = cq(t, m.cb)
      const bad = firstNonFinite(a)
      if (bad >= 0) {
        const rawOffset = t.dtype === FP16 ? bad * 2 : bad * 4
        const rawBytes = Math.min(4, t.data.byteLength - rawOffset)
        let bits = ''
        if (rawBytes >= 2)
          bits = new DataView(
            t.data.buffer,
            t.data.byteOffset + rawOffset,
            rawBytes
          )
            .getUint16(0, true)
            .toString(16)
            .padStart(4, '0')
        throw Error(
          `权重 tensor ${this.w.length}（${dtypeName(t.dtype)} ${t.shape.join(
            'x'
          )}，${
            t.data.byteLength
          } bytes）在索引 ${bad} 出现非有限值，raw=0x${bits}`
        )
      }
      this.w.push(a)
      const b = dev.createBuffer({
        size: Math.max(4, a.byteLength),
        usage: STORAGE | COPY_DST
      })
      this.weightBytes += b.size
      dev.queue.writeBuffer(b, 0, a)
      this.wb.push(b)
    }
    this.peakBytes = this.weightBytes + 16
    this.pipe = dev.createComputePipeline({
      layout: 'auto',
      compute: {
        module: dev.createShaderModule({
          code: `struct U{m:u32,n:u32,k:u32,};@group(0)@binding(0)var<storage,read>a:array<f32>;@group(0)@binding(1)var<storage,read>b:array<f32>;@group(0)@binding(2)var<storage,read_write>c:array<f32>;@group(0)@binding(3)var<uniform>u:U;@compute@workgroup_size(8,8)fn main(@builtin(global_invocation_id)i:vec3<u32>){if(i.x>=u.n||i.y>=u.m){return;}var s:f32=0.;for(var j:u32=0u;j<u.k;j++){s+=a[i.y*u.k+j]*b[i.x*u.k+j];}c[i.y*u.n+i.x]=s;}`
        }),
        entryPoint: 'main'
      }
    })
    this.layout = this.pipe.getBindGroupLayout(0)
    this.uni = dev.createBuffer({ size: 16, usage: UNIFORM | COPY_DST })
  }
  async mm (a: Float32Array, m: number, k: number, wi: number, n: number) {
    this.counter &&
      (this.counter.dispatches++, (this.counter.flops += 2 * m * k * n))
    const aBytes = Math.max(4, a.byteLength),
      outBytes = Math.max(4, m * n * 4)
    this.peakBytes = Math.max(
      this.peakBytes,
      this.weightBytes + 16 + aBytes + outBytes * 2
    )
    const ab = this.dev.createBuffer({
        size: aBytes,
        usage: STORAGE | COPY_DST
      }),
      cb = this.dev.createBuffer({ size: outBytes, usage: STORAGE | COPY_SRC }),
      rb = this.dev.createBuffer({ size: outBytes, usage: MAP_READ | COPY_DST })
    this.dev.queue.writeBuffer(ab, 0, a)
    this.dev.queue.writeBuffer(this.uni, 0, new Uint32Array([m, n, k, 0]))
    const bg = this.dev.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: ab } },
        { binding: 1, resource: { buffer: this.wb[wi] } },
        { binding: 2, resource: { buffer: cb } },
        { binding: 3, resource: { buffer: this.uni } }
      ]
    })
    const e = this.dev.createCommandEncoder(),
      p = e.beginComputePass()
    p.setPipeline(this.pipe)
    p.setBindGroup(0, bg)
    p.dispatchWorkgroups(Math.ceil(n / 8), Math.ceil(m / 8))
    p.end()
    e.copyBufferToBuffer(cb, 0, rb, 0, m * n * 4)
    this.dev.queue.submit([e.finish()])
    await this.dev.queue.onSubmittedWorkDone()
    await rb.mapAsync(1)
    const out = new Float32Array(rb.getMappedRange().slice(0))
    rb.unmap()
    ab.destroy()
    cb.destroy()
    rb.destroy()
    const bad = firstNonFinite(out)
    if (bad >= 0)
      throw Error(
        `GEMM tensor ${wi} 输出在索引 ${bad} 出现 NaN/Infinity（${m}x${k} · ${k}x${n}）`
      )
    return out
  }
}

function norm (x: Float32Array, sc: Float32Array) {
  const d = sc.length,
    o = new Float32Array(x.length)
  for (let i = 0; i < x.length / d; i++) {
    let r = 0
    for (let j = 0; j < d; j++) {
      const v = x[i * d + j]
      r += v * v
    }
    r = Math.sqrt(r / d + 1e-6)
    for (let j = 0; j < d; j++) o[i * d + j] = (x[i * d + j] * (1 + sc[j])) / r
  }
  return o
}
function silu (x: number) {
  return x / (1 + Math.exp(-Math.max(-40, Math.min(40, x))))
}
function sink (a: Float32Array, n: number) {
  for (let z = 0; z < 20; z++) {
    for (let i = 0; i < n; i++) {
      let mx = -Infinity
      for (let j = 0; j < n; j++) mx = Math.max(mx, a[i * n + j])
      let s = 0
      for (let j = 0; j < n; j++) {
        const e = Math.exp(a[i * n + j] - mx)
        a[i * n + j] = e
        s += e
      }
      for (let j = 0; j < n; j++) a[i * n + j] /= Math.max(s, 1e-12)
    }
    for (let j = 0; j < n; j++) {
      let s = 0
      for (let i = 0; i < n; i++) s += a[i * n + j]
      for (let i = 0; i < n; i++) a[i * n + j] /= Math.max(s, 1e-12)
    }
  }
  return a
}
function att (
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  T: number,
  h: number,
  kh: number,
  d: number
) {
  const o = new Float32Array(T * h * d),
    rep = h / kh,
    sc = 1 / Math.sqrt(d)
  for (let t = 0; t < T; t++)
    for (let z = 0; z < h; z++) {
      const kz = Math.floor(z / rep)
      let mx = -1e30
      for (let j = 0; j <= t; j++) {
        let s = 0
        for (let x = 0; x < d; x++)
          s += q[(t * h + z) * d + x] * k[(j * kh + kz) * d + x] * sc
        mx = Math.max(mx, s)
      }
      let den = 0
      for (let j = 0; j <= t; j++) {
        let s = 0
        for (let x = 0; x < d; x++)
          s += q[(t * h + z) * d + x] * k[(j * kh + kz) * d + x] * sc
        den += Math.exp(s - mx)
      }
      for (let x = 0; x < d; x++) {
        let s = 0
        for (let j = 0; j <= t; j++) {
          let z2 = 0
          for (let y = 0; y < d; y++)
            z2 += q[(t * h + z) * d + y] * k[(j * kh + kz) * d + y] * sc
          s += Math.exp(z2 - mx) * v[(j * kh + kz) * d + x]
        }
        o[(t * h + z) * d + x] = s / den
      }
    }
  return o
}
function rotate (
  x: Float32Array,
  T: number,
  h: number,
  d: number,
  theta: number
) {
  const o = x.slice(),
    half = d / 2
  for (let t = 0; t < T; t++)
    for (let z = 0; z < h; z++)
      for (let j = 0; j < half; j++) {
        const a = t * Math.pow(theta, (-2 * j) / d),
          c = Math.cos(a),
          s = Math.sin(a),
          i = (t * h + z) * d + j,
          q = i + half,
          X = x[i],
          Y = x[q]
        o[i] = X * c - Y * s
        o[q] = Y * c + X * s
      }
  return o
}

async function generate (
  tokens: number[],
  m: Model,
  r: Runtime,
  step: (n: number) => void,
  c: Counters
) {
  if (cancelled) throw Error('推理已停止')
  const g = m.g,
    D = g.d_model,
    Hd = g.head_dim,
    L = g.num_layers,
    n = g.mhc_lanes,
    N = tokens.length
  let wi = 1
  const ls: any[] = []
  for (let l = 0; l < L; l++) {
    ls.push({
      norm: wi++,
      q: wi++,
      k: wi++,
      v: wi++,
      qn: wi++,
      kn: wi++,
      gate: wi++,
      out: wi++,
      post: wi++,
      ag: wi++,
      pre: wi++,
      d1: wi++,
      d2: wi++,
      d3: wi++
    })
  }
  const mh = {
    ap: wi++,
    apost: wi++,
    ar: wi++,
    bp: wi++,
    bpost: wi++,
    br: wi++,
    pp: wi++,
    ppo: wi++,
    pr: wi++
  }
  const es: number[] = []
  for (let s = 0; s < (g.engram_layers as number[]).length; s++)
    es.push(wi), (wi += 4)
  const final = wi++
  let x = new Float32Array(N * D * n)
  const E = r.w[0]
  for (let t = 0; t < N; t++)
    for (let l = 0; l < n; l++)
      x.set(E.subarray(tokens[t] * D, tokens[t] * D + D), (t * n + l) * D)
  if (DEBUG_LATENTS) {
    ;(window as any).__needleLatents = []
    console.log('[latent] enabled', {
      layers: L,
      d_model: D,
      mhc_lanes: n,
      tokens: N
    })
  }
  for (let l = 0; l < L; l++) {
    if (cancelled) throw Error('推理已停止')
    const z = x.slice(),
      nx = new Float32Array(z.length)
    for (let t = 0; t < N; t++) {
      let rr = 0
      for (let j = 0; j < n * D; j++) {
        const v = z[t * n * D + j]
        rr += v * v
      }
      rr = Math.sqrt(rr / (n * D) + 1e-6)
      for (let j = 0; j < n * D; j++) nx[t * n * D + j] = z[t * n * D + j] / rr
    }
    const lc = ls[l],
      hp = await r.mm(nx, N, n * D, mh.pp, n),
      u = new Float32Array(N * D)
    for (let t = 0; t < N; t++)
      for (let lane = 0; lane < n; lane++) {
        const gate =
          1 /
          (1 +
            Math.exp(
              -(
                hp[t * n + lane] * r.w[mh.ap][l] +
                (r.w[mh.bp][l * n + lane] || 0) +
                8 * (lane === l % n ? 1 : 0) -
                4
              )
            ))
        for (let d = 0; d < D; d++)
          u[t * D + d] += gate * z[(t * n + lane) * D + d]
      }
    for (let si = 0; si < (g.engram_layers as number[]).length; si++)
      if ((g.engram_layers as number[])[si] === l) {
        const base = es[si],
          orders = g.engram_orders as number[],
          heads = g.num_engram_tables / orders.length,
          sub = g.engram_sub_dim,
          flat = new Float32Array(N * heads * orders.length * sub)
        for (let t = 0; t < N; t++)
          for (let oi = 0; oi < orders.length; oi++)
            for (let h = 0; h < heads; h++) {
              let a = Math.imul(0x9e3779b9, oi * heads + h + 1) >>> 0
              for (let j = 0; j < orders[oi]; j++) {
                const tok = t - j >= 0 ? tokens[t - j] : 0
                a = Math.imul((a ^ tok) >>> 0, 0x01000193) >>> 0
              }
              const id = ((a ^ (a >>> 15)) >>> 0) % g.engram_slots,
                table = r.w[base],
                row = (oi * heads + h) * g.engram_slots + id
              flat.set(
                table.subarray(row * sub, row * sub + sub),
                (t * g.num_engram_tables + oi * heads + h) * sub
              )
            }
        const ek = await r.mm(
            flat,
            N,
            orders.length * heads * sub,
            base + 1,
            D
          ),
          ev = await r.mm(flat, N, orders.length * heads * sub, base + 2, D)
        for (let t = 0; t < N; t++) {
          let dot = 0,
            nxv = 0,
            ekv = 0
          for (let d = 0; d < D; d++) {
            const uv = u[t * D + d]
            dot += uv * ek[t * D + d]
            nxv += uv * uv
            ekv += ek[t * D + d] * ek[t * D + d]
          }
          const a = 1 / (1 + Math.exp(-dot / Math.sqrt(Math.max(1, nxv * ekv))))
          for (let d = 0; d < D; d++) u[t * D + d] += a * ev[t * D + d]
        }
      }
    let un = norm(u, r.w[lc.norm]),
      q0 = await r.mm(un, N, D, lc.q, g.num_heads * Hd),
      k0 = await r.mm(un, N, D, lc.k, g.num_kv_heads * Hd),
      v0 = await r.mm(un, N, D, lc.v, g.num_kv_heads * Hd)
    let q = new Float32Array(q0.length),
      k = new Float32Array(k0.length)
    for (let t = 0; t < N; t++) {
      for (let h = 0; h < g.num_heads; h++) {
        let rr = 0
        for (let d = 0; d < Hd; d++) {
          const v = q0[(t * g.num_heads + h) * Hd + d]
          rr += v * v
        }
        rr = Math.sqrt(rr / Hd + 1e-6)
        for (let d = 0; d < Hd; d++)
          q[(t * g.num_heads + h) * Hd + d] =
            (q0[(t * g.num_heads + h) * Hd + d] * (1 + r.w[lc.qn][d])) / rr
      }
      for (let h = 0; h < g.num_kv_heads; h++) {
        let rr = 0
        for (let d = 0; d < Hd; d++) {
          const v = k0[(t * g.num_kv_heads + h) * Hd + d]
          rr += v * v
        }
        rr = Math.sqrt(rr / Hd + 1e-6)
        for (let d = 0; d < Hd; d++)
          k[(t * g.num_kv_heads + h) * Hd + d] =
            (k0[(t * g.num_kv_heads + h) * Hd + d] * (1 + r.w[lc.kn][d])) / rr
      }
    }
    q = rotate(q, N, g.num_heads, Hd, g.rope_theta)
    k = rotate(k, N, g.num_kv_heads, Hd, g.rope_theta)
    let a = att(q, k, v0, N, g.num_heads, g.num_kv_heads, Hd)
    const gate = await r.mm(un, N, D, lc.gate, g.num_heads * Hd)
    for (let i = 0; i < a.length; i++) a[i] *= 1 / (1 + Math.exp(-gate[i]))
    let ao = await r.mm(a, N, g.num_heads * Hd, lc.out, D)
    ao = norm(ao, r.w[lc.post])
    const block = new Float32Array(N * D)
    for (let i = 0; i < block.length; i++)
      block[i] =
        un[i] + (r.w[lc.ag][l] ? 1 / (1 + Math.exp(-r.w[lc.ag][l])) : 0) * ao[i]
    let h = norm(block, r.w[lc.pre])
    const hada = g.hada_n,
      tmp = new Float32Array(hada)
    for (let t = 0; t < N; t++) {
      tmp.fill(0)
      tmp.set(h.subarray(t * D, t * D + D))
      for (let j = 0; j < hada; j++) tmp[j] *= r.w[lc.d1][j] || 1
      fwht(tmp)
      for (let j = 0; j < hada; j++)
        tmp[j] = silu(tmp[j] * (r.w[lc.d2][j] || 1))
      fwht(tmp)
      for (let d = 0; d < D; d++)
        block[t * D + d] += tmp[d] * (r.w[lc.d3][d] || 0.02)
    }
    const hp2 = await r.mm(nx, N, n * D, mh.ppo, n),
      res = await r.mm(nx, N, n * D, mh.pr, n * n)
    for (let t = 0; t < N; t++) {
      const sm = sink(res.subarray(t * n * n, (t + 1) * n * n), n)
      for (let i = 0; i < n; i++)
        for (let d = 0; d < D; d++) {
          let v = 0
          for (let j = 0; j < n; j++)
            v += sm[i * n + j] * z[(t * n + j) * D + d]
          const post =
            2 /
            (1 +
              Math.exp(
                -(
                  hp2[t * n + i] * r.w[mh.apost][l] +
                  (r.w[mh.bpost][l * n + i] || 0) -
                  4 * (1 - (i === l % n ? 1 : 0))
                )
              ))
          x[(t * n + i) * D + d] = v + post * (block[t * D + d] - un[t * D + d])
        }
    }
    dumpLatent(l, x, N, n, D)
    step(l + 1)
  }
  const last = new Float32Array(D)
  for (let d = 0; d < D; d++) {
    let s = 0
    for (let l = 0; l < n; l++) s += x[((N - 1) * n + l) * D + d]
    last[d] = s / n
  }
  const fn = norm(last, r.w[final])
  return await r.mm(fn, 1, D, 0, g.vocab_size)
}

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
function topCandidates (logits: Float32Array, tok: Tok, k: number): Candidate[] {
  const bad = firstNonFinite(logits)
  if (bad >= 0)
    throw Error(`logits 在索引 ${bad} 出现 NaN/Infinity，无法选择 token`)
  let max = -Infinity
  for (const x of logits) if (x > max) max = x
  let sum = 0
  for (const x of logits) sum += Math.exp(x - max)
  const ids = Array.from({ length: logits.length }, (_, i) => i)
    .sort((a, b) => logits[b] - logits[a])
    .slice(0, k)
  return ids.map(id => ({
    id,
    token: tok.tokenLabel(id),
    logit: logits[id],
    probability: Math.exp(logits[id] - max) / sum
  }))
}
function showDebug (promptIds: number[], generated: number[], steps: unknown[]) {
  $<HTMLPreElement>('debugPrompt').textContent = JSON.stringify(promptIds)
  $<HTMLPreElement>('debugGenerated').textContent = JSON.stringify(generated)
  $<HTMLPreElement>('debugSteps').textContent = JSON.stringify(steps, null, 2)
}
async function run () {
  const q = $<HTMLTextAreaElement>('query').value.trim()
  if (!q || !model || !rt) return
  const runtime = rt
  const b = $<HTMLButtonElement>('send')
  b.disabled = true
  add('user', q)
  const assistant = add('assistant', '生成中…')
  const tools = JSON.parse($<HTMLTextAreaElement>('tools').value || '[]'),
    prompt = `<|im_start|>user\
<tools>${JSON.stringify(tools)}</tools>\
${q}<|im_end|>\
<|im_start|>assistant\
`,
    ids = [2, ...model.tok.encode(prompt)],
    gen: number[] = [],
    steps: unknown[] = []
  const max = Number($<HTMLInputElement>('maxTokens').value) || 96,
    k = Math.max(
      1,
      Math.min(20, Number($<HTMLInputElement>('topK').value) || 5)
    ),
    start = performance.now(),
    counter: Counters = { dispatches: 0, flops: 0, forwardMs: 0 }
  cancelled = false
  runtime.counter = counter
  cancel.disabled = false
  showDebug(ids, [], [])
  try {
    let previous = performance.now()
    for (let i = 0; i < max; i++) {
      const forwardStart = performance.now(),
        phaseStart = performance.now()
      const logits = await generate(
        [...ids, ...gen],
        model,
        runtime,
        n => {
          progress.textContent = `WebGPU Needle · layer ${n}/${
            model!.g.num_layers
          }`
          const elapsed = Math.max(1, performance.now() - phaseStart)
          metrics.textContent = `${
            i === 0 ? 'prefill' : 'forward'
          } · layer ${n}/${model!.g.num_layers} · ${
            ids.length + gen.length
          } tokens · ${(ids.length / (elapsed / 1000)).toFixed(2)} tok/s · ${
            counter.dispatches
          } shader dispatch · ${formatFlops(
            counter.flops
          )} FLOPs · 显存 ${formatBytes(runtime.peakBytes)}（权重 ${formatBytes(
            runtime.weightBytes
          )}）`
        },
        counter
      )
      const now = performance.now(),
        tokenMs = now - previous
      previous = now
      counter.forwardMs += now - forwardStart
      const candidates = topCandidates(logits, model.tok, k),
        best = candidates[0]
      steps.push({
        step: i + 1,
        selected: {
          id: best.id,
          token: best.token,
          logit: best.logit,
          probability: best.probability
        },
        topK: candidates
      })
      gen.push(best.id)
      showDebug(ids, gen, steps)
      assistant.textContent = model.tok.decode(gen) || '生成中…'
      metrics.textContent = `${
        i === 0 ? '首 token prefill' : 'decode'
      } · 已生成 ${i + 1}/${max} · 当前 ${(1000 / Math.max(1, tokenMs)).toFixed(
        2
      )} tok/s · 总 ${(gen.length / (Math.max(1, now - start) / 1000)).toFixed(
        2
      )} tok/s · ${counter.dispatches} shader dispatch · ${formatFlops(
        counter.flops
      )} FLOPs · 显存 ${formatBytes(runtime.peakBytes)}（权重 ${formatBytes(
        runtime.weightBytes
      )}）`
      if (best.id === 1) break
      let repeated = 1
      for (let j = gen.length - 2; j >= 0 && gen[j] === best.id; j--) repeated++
      if (repeated >= 8) {
        metrics.textContent += ` · 重复 token ${model.tok.tokenLabel(
          best.id
        )}，已自动停止`
        break
      }
    }
    showDebug(ids, gen, steps)
    const text = model.tok.decode(gen),
      m = text.match(/<tool_call>([\s\S]*?)<\/tool_call>/)
    let out = text
    if (m)
      try {
        out = JSON.stringify(JSON.parse(m[1]), null, 2)
      } catch {}
    assistant.textContent = out || '(empty; see token diagnostics)'
    $('timing').textContent = `· ${Math.round(
      performance.now() - start
    )} ms · WebGPU`
  } catch (e: any) {
    showDebug(ids, gen, steps)
    assistant.textContent = '推理失败：' + (e?.message || e)
  } finally {
    runtime.counter = null
    cancel.disabled = true
    progress.textContent = ''
    b.disabled = false
  }
}
function add (role: string, text: string) {
  if (msgs.querySelector('.empty')) msgs.innerHTML = ''
  const e = document.createElement('div')
  e.className = `message ${role}`
  e.textContent = text
  msgs.appendChild(e)
  msgs.scrollTop = msgs.scrollHeight
  return e
}
$('send').addEventListener('click', run)
cancel.addEventListener('click', () => {
  cancelled = true
  metrics.textContent = '正在停止…'
})
$('query').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    run()
  }
})
$('loadDemo').addEventListener('click', () => {
  $<HTMLTextAreaElement>('tools').value = JSON.stringify(
    [
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
    ],
    null,
    2
  )
})
$('modelFile').addEventListener('change', async e => {
  const f = (e.target as HTMLInputElement).files?.[0]
  if (!f) return
  $<HTMLDivElement>('modelInfo').textContent = '正在解析 CQ 权重并上传 GPU…'
  try {
    cqNonFiniteScales = 0
    model = new Model(await f.arrayBuffer())
    rt = new Runtime(gpu, model)
    $<HTMLDivElement>('modelInfo').textContent = `${f.name} · ${
      model.g.d_model
    }d · ${model.g.num_layers} 层 · ${
      model.t.length
    } tensors · 文件权重 ${formatBytes(
      model.t
        .filter(x => x.dtype !== RAW)
        .reduce((s, x) => s + x.data.byteLength, 0)
    )} · CQ 异常 norm ${cqNonFiniteScales} 组 · GPU 权重 ${formatBytes(
      rt.weightBytes
    )} · 峰值 ${formatBytes(rt.peakBytes)}`
    $<HTMLDivElement>('modelInfo').className = 'ok'
  } catch (e: any) {
    console.error(e)
    $<HTMLDivElement>('modelInfo').textContent = e.message
    $<HTMLDivElement>('modelInfo').className = 'bad'
  }
})
;(async () => {
  try {
    if (!('gpu' in navigator)) throw Error('当前浏览器不支持 WebGPU')
    const a = await (navigator as any).gpu.requestAdapter()
    if (!a) throw Error('未找到 WebGPU 适配器')
    gpu = await a.requestDevice()
    gpuStatus.textContent = `WebGPU 已启用 · ${a.name || '默认适配器'}`
    gpuStatus.className = 'status ok'
  } catch (e: any) {
    gpuStatus.textContent = e.message
    gpuStatus.className = 'status bad'
  }
})()
