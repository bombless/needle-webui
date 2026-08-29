type Tensor = {
  dtype: number
  shape: number[]
  offset: number
  nbytes: number
  group: number
  bits: number
  data: Uint8Array
}
type Cact = {
  geometry: Record<string, number>
  tensors: Tensor[]
  codebook: Float32Array
  tokenizer?: Tokenizer
}
type GPUBufferLike = any
const $ = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T
const TAG = 0x05e12a83,
  HEADER = 120,
  REC = 48,
  FP16 = 1,
  FP32 = 2,
  CQ = 3,
  RAW = 4,
  BUFS = 1,
  BUFCOPY = 4,
  BUFDST = 8,
  STAGE = 4
const gpuStatus = $<HTMLDivElement>('gpuStatus'),
  messages = $<HTMLDivElement>('messages'),
  progress = $<HTMLDivElement>('progress')
let device: any = null,
  model: Cact | null = null,
  runner: NeedleGPU | null = null

function half (h: number) {
  const s = (h >>> 15) & 1,
    e = (h >>> 10) & 31,
    f = h & 1023
  if (!e) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024)
  if (e === 31) return f ? NaN : (s ? -1 : 1) * Infinity
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024)
}
function parseCact (buf: ArrayBuffer): Cact {
  const d = new DataView(buf),
    raw = new Uint8Array(buf)
  if (d.getUint32(0, true) !== TAG) throw Error('不是有效的 Needle .cact 文件')
  const n = d.getUint32(4, true),
    cbn = d.getUint32(8, true)
  const names = [
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
  ]
  const g: Record<string, number> = {}
  names.forEach((k, i) => (g[k] = d.getUint32(20 + i * 4, true)))
  g.kv_window = d.getUint32(12, true)
  g.kv_bits = d.getUint32(16, true)
  g.rope_theta = d.getFloat32(116, true)
  const cb = new Float32Array(buf.slice(HEADER, HEADER + cbn * 4))
  let o = HEADER + cbn * 4
  const ts: Tensor[] = []
  for (let i = 0; i < n; i++) {
    const dtype = d.getUint8(o),
      nd = d.getUint8(o + 1),
      shape: number[] = []
    for (let j = 0; j < nd; j++) shape.push(d.getUint32(o + 4 + j * 4, true))
    const offset = Number(d.getBigUint64(o + 20, true)),
      nbytes = Number(d.getBigUint64(o + 28, true))
    ts.push({
      dtype,
      shape,
      offset,
      nbytes,
      group: d.getUint32(o + 36, true),
      bits: d.getUint32(o + 40, true),
      data: raw.subarray(offset, offset + nbytes)
    })
    o += REC
  }
  const tk = ts.find(t => t.dtype === RAW)?.data
  return {
    geometry: g,
    tensors: ts,
    codebook: cb,
    tokenizer: tk ? new Tokenizer(tk) : undefined
  }
}
function cbFor (bits: number, cb: Float32Array) {
  if (bits === 2) return cb.subarray(0, 4)
  if (bits === 3) return cb.subarray(4, 12)
  if (bits === 4) return cb.subarray(12, 28)
  return new Float32Array(
    [-1.2240064, 0, 1.2240064].map(x => x / Math.sqrt(128))
  )
}
function unpack (
  src: Uint8Array,
  row: number,
  rows: number,
  inPad: number,
  bits: number
) {
  const out = new Uint8Array(inPad)
  if (bits === 5) {
    const bpr = inPad >> 2,
      off = row * bpr
    for (let i = 0; i < inPad; i++) {
      const c = (src[off + (i >> 2)] >> ((i & 3) * 2)) & 3
      out[i] = c === 3 ? 0 : c + 1
    }
    return out
  }
  const bpr = (inPad * bits) / 8,
    off = row * bpr,
    mask = (1 << bits) - 1
  for (let i = 0; i < inPad; i++) {
    const bit = i * bits,
      bi = off + (bit >> 3),
      sh = bit & 7
    let v = src[bi] >> sh
    if (sh + bits > 8) v |= src[bi + 1] << (8 - sh)
    out[i] = v & mask
  }
  return out
}
function fwht (a: Float32Array, base: number, n: number) {
  for (let len = 1; len < n; len <<= 1)
    for (let i = 0; i < n; i += len << 1)
      for (let j = 0; j < len; j++) {
        const x = a[base + i + j],
          y = a[base + i + j + len]
        a[base + i + j] = x + y
        a[base + i + j + len] = x - y
      }
  const q = 1 / Math.sqrt(n)
  for (let i = 0; i < n; i++) a[base + i] *= q
}
function dequant (t: Tensor, cb: Float32Array) {
  const [out, inDim] = t.shape,
    g = t.group || 128,
    bits = t.bits,
    inPad = Math.ceil(inDim / g) * g,
    rowBytes = bits === 5 ? inPad / 4 : (inPad * bits) / 8,
    normStride = (inPad / g) * 2,
    dst = new Float32Array(out * inDim),
    code = cbFor(bits, cb),
    dv = new DataView(t.data.buffer, t.data.byteOffset, t.data.byteLength),
    tmp = new Float32Array(g)
  for (let r = 0; r < out; r++) {
    const ids = unpack(t.data, r, out, inPad, bits),
      rowOff = r * (rowBytes + normStride)
    for (let q = 0; q < inPad / g; q++) {
      const scale = half(dv.getUint16(rowOff + rowBytes + q * 2, true))
      for (let j = 0; j < g; j++) tmp[j] = (code[ids[q * g + j]] || 0) * scale
      fwht(tmp, 0, g)
      for (let j = 0; j < g && q * g + j < inDim; j++)
        dst[r * inDim + q * g + j] = tmp[j]
    }
  }
  return dst
}

class Tokenizer {
  pieces: string[] = []
  scores: number[] = []
  types: number[] = []
  byFirst = new Map<string, number[]>()
  special = new Map<string, number>()
  constructor (blob: Uint8Array) {
    const d = new DataView(blob.buffer, blob.byteOffset, blob.byteLength),
      n = d.getUint32(0, true)
    let o = 24
    const td = new TextDecoder()
    for (let i = 0; i < n; i++) {
      const score = d.getFloat32(o, true),
        type = d.getUint8(o + 4),
        len = d.getUint16(o + 5, true)
      o += 8
      const p = td.decode(blob.subarray(o, o + len))
      o += len
      this.pieces.push(p)
      this.scores.push(score)
      this.types.push(type)
      this.special.set(p, i)
      const k = p[0]
      if (k) this.byFirst.set(k, [...(this.byFirst.get(k) || []), i])
    }
  }
  encode (text: string) {
    let s = '▁' + text.replaceAll(' ', '▁'),
      c = Array.from(s),
      N = c.length,
      inf = 1e30,
      dp = new Float64Array(N + 1),
      prev = new Int32Array(N + 1),
      pick = new Int32Array(N + 1)
    dp.fill(inf)
    dp[0] = 0
    for (let i = 0; i < N; i++) {
      if (dp[i] >= inf) continue
      for (const id of this.byFirst.get(c[i]) || []) {
        const pc = Array.from(this.pieces[id])
        if (i + pc.length > N) continue
        let ok = true
        for (let j = 0; j < pc.length; j++)
          if (c[i + j] !== pc[j]) {
            ok = false
            break
          }
        if (ok && dp[i] - this.scores[id] < dp[i + pc.length]) {
          dp[i + pc.length] = dp[i] - this.scores[id]
          prev[i + pc.length] = i
          pick[i + pc.length] = id
        }
      }
      if (dp[i + 1] >= inf) {
        const b = new TextEncoder().encode(c[i])
        for (const x of b) {
          const id = this.special.get(
            `<0x${x.toString(16).padStart(2, '0').toUpperCase()}>`
          )
          if (id !== undefined && dp[i] + 20 < dp[i + 1]) {
            dp[i + 1] = dp[i] + 20
            prev[i + 1] = i
            pick[i + 1] = id
          }
        }
      }
    }
    const out: number[] = []
    for (let p = N; p > 0 && prev[p] < p; p = prev[p]) out.push(pick[p])
    return out.reverse()
  }
  decode (ids: number[]) {
    let s = ''
    for (const id of ids) {
      const p = this.pieces[id] || ''
      if (this.types[id] === 4) {
        const m = p.match(/<0x([0-9A-Fa-f]{2})>/)
        if (m) s += String.fromCharCode(parseInt(m[1], 16))
      } else s += p
    }
    return s.replaceAll('▁', ' ').trim()
  }
}

class NeedleGPU {
  dev: any
  cfg: Record<string, number>
  weights: Float32Array[] = []
  gpuW: GPUBufferLike[] = []
  shader: any
  pipe: any
  read: any
  uniform: any
  bindLayout: any
  constructor (dev: any, c: Cact) {
    this.dev = dev
    this.cfg = c.geometry
    for (const t of c.tensors.filter(x => x.dtype !== RAW)) {
      let a: Float32Array
      if (t.dtype === FP16) {
        const d = new DataView(
          t.data.buffer,
          t.data.byteOffset,
          t.data.byteLength
        )
        a = new Float32Array(t.data.byteLength / 2)
        for (let i = 0; i < a.length; i++) a[i] = half(d.getUint16(i * 2, true))
      } else if (t.dtype === FP32)
        a = new Float32Array(
          t.data.buffer,
          t.data.byteOffset,
          t.data.byteLength / 4
        ).slice()
      else a = dequant(t, c.codebook)
      this.weights.push(a)
      const b = dev.createBuffer({
        size: Math.max(4, a.byteLength),
        usage: BUFS | BUFDST
      })
      dev.queue.writeBuffer(b, 0, a)
      this.gpuW.push(b)
    }
    this.make()
  }
  make () {
    const code = `struct U{m:u32;n:u32;k:u32;};@group(0)@binding(0)var<storage,read>a:array<f32>;@group(0)@binding(1)var<storage,read>b:array<f32>;@group(0)@binding(2)var<storage,read_write>c:array<f32>;@group(0)@binding(3)var<uniform>u:U;@compute@workgroup_size(8,8)fn main(@builtin(global_invocation_id)id:vec3<u32>){let r=id.y;let col=id.x;if(r>=u.m||col>=u.n){return;}var s:f32=0.0;for(var j:u32=0u;j<u.k;j++){s+=a[r*u.k+j]*b[col*u.k+j];}c[r*u.n+col]=s;}`
    this.shader = this.dev.createShaderModule({ code })
    this.pipe = this.dev.createComputePipeline({
      layout: 'auto',
      compute: { module: this.shader, entryPoint: 'main' }
    })
    this.bindLayout = this.pipe.getBindGroupLayout(0)
    this.uniform = this.dev.createBuffer({ size: 16, usage: BUFDST | BUFS })
  }
  async mm (
    a: Float32Array,
    m: number,
    k: number,
    w: Float32Array,
    n: number,
    wi: number
  ) {
    const ab = this.dev.createBuffer({
      size: Math.max(4, a.byteLength),
      usage: BUFS | BUFDST
    })
    this.dev.queue.writeBuffer(ab, 0, a)
    const cb = this.dev.createBuffer({
      size: Math.max(4, m * n * 4),
      usage: BUFS | BUFCOPY
    })
    this.dev.queue.writeBuffer(this.uniform, 0, new Uint32Array([m, n, k, 0]))
    const bg = this.dev.createBindGroup({
      layout: this.bindLayout,
      entries: [
        { binding: 0, resource: { buffer: ab } },
        { binding: 1, resource: { buffer: this.gpuW[wi] } },
        { binding: 2, resource: { buffer: cb } },
        { binding: 3, resource: { buffer: this.uniform } }
      ]
    })
    const enc = this.dev.createCommandEncoder(),
      p = enc.beginComputePass()
    p.setPipeline(this.pipe)
    p.setBindGroup(0, bg)
    p.dispatchWorkgroups(Math.ceil(n / 8), Math.ceil(m / 8))
    p.end()
    enc.copyBufferToBuffer(cb, 0, this.readback(m * n), 0, m * n * 4)
    this.dev.queue.submit([enc.finish()])
    const r = await this.readResult(m * n)
    ab.destroy()
    cb.destroy()
    return r
  }
  readback (n: number) {
    if (this.read && this.read.size >= n * 4) return this.read
    this.read?.destroy()
    this.read = this.dev.createBuffer({ size: n * 4, usage: 2 | BUFCOPY })
    return this.read
  }
  async readResult (n: number) {
    await this.dev.queue.onSubmittedWorkDone()
    const s = this.read.mapAsync(1)
    await s
    const x = new Float32Array(this.read.getMappedRange().slice(0))
    this.read.unmap()
    return x
  }
  async mat (a: Float32Array, m: number, k: number, wi: number, n: number) {
    return this.mm(a, m, k, this.weights[wi], n, wi)
  }
}

function norm (x: Float32Array, scale: Float32Array) {
  const d = scale.length,
    out = new Float32Array(x.length)
  for (let t = 0; t < x.length / d; t++) {
    let r = 0
    for (let j = 0; j < d; j++) {
      const v = x[t * d + j]
      r += v * v
    }
    r = Math.sqrt(r / d + 1e-6)
    for (let j = 0; j < d; j++)
      out[t * d + j] = (x[t * d + j] * (1 + scale[j])) / r
  }
  return out
}
function sigmoid (x: number) {
  return 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, x))))
}
function rope (x: Float32Array, T: number, H: number, D: number, theta: number) {
  const out = x.slice(),
    halfD = D / 2
  for (let t = 0; t < T; t++)
    for (let h = 0; h < H; h++)
      for (let j = 0; j < halfD; j += 1) {
        const inv = Math.pow(theta, (-2 * j) / D),
          a = t * inv,
          c = Math.cos(a),
          s = Math.sin(a),
          i = (t * H + h) * D + j,
          q = (t * H + h) * D + halfD + j,
          x1 = x[i],
          x2 = x[q]
        out[i] = x1 * c - x2 * s
        out[q] = x2 * c + x1 * s
      }
  return out
}
function attention (
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  T: number,
  H: number,
  KV: number,
  D: number
) {
  const out = new Float32Array(T * H * D),
    rep = H / KV,
    sc = 1 / Math.sqrt(D)
  for (let t = 0; t < T; t++)
    for (let h = 0; h < H; h++) {
      const kh = Math.floor(h / rep)
      let mx = -1e30
      for (let j = 0; j <= t; j++) {
        let z = 0
        for (let d = 0; d < D; d++)
          z += q[(t * H + h) * D + d] * k[(j * KV + kh) * D + d] * sc
        if (z > mx) mx = z
      }
      let den = 0
      for (let j = 0; j <= t; j++) {
        let z = 0
        for (let d = 0; d < D; d++)
          z += q[(t * H + h) * D + d] * k[(j * KV + kh) * D + d] * sc
        den += Math.exp(z - mx)
      }
      for (let d = 0; d < D; d++) {
        let z = 0
        for (let j = 0; j <= t; j++) {
          let e = 0
          for (let dd = 0; dd < D; dd++)
            e += q[(t * H + h) * D + dd] * k[(j * KV + kh) * D + dd] * sc
          z += Math.exp(e - mx) * v[(j * KV + kh) * D + d]
        }
        out[(t * H + h) * D + d] = z / den
      }
    }
  return out
}
function hashIdx (
  tokens: number[],
  t: number,
  order: number,
  head: number,
  slots: number
) {
  let a = (0x9e3779b9 * Math.imul(1, head + 1)) >>> 0
  for (let j = 0; j < order; j++) {
    const u = t - j >= 0 ? tokens[t - j] : 0
    a = Math.imul((a ^ (u >>> 0)) >>> 0, 0x01000193) >>> 0
  }
  return ((a ^ (a >>> 15)) >>> 0) % slots
}

async function infer (
  g: Cact,
  rg: NeedleGPU,
  tokens: number[],
  limit: number,
  onStep: (n: number) => void
) {
  const cfg = g.geometry,
    D = cfg.d_model,
    H = cfg.num_heads,
    KV = cfg.num_kv_heads,
    HD = cfg.head_dim,
    L = cfg.num_layers,
    N = tokens.length
  const ts = rg.weights
  let idx = 1,
    emb = idx++
  let x = new Float32Array(N * D)
  for (let t = 0; t < N; t++) {
    const e = ts[0],
      v = tokens[t]
    x.set(e.subarray(v * D, v * D + D), t * D)
  }
  const layers: any[] = []
  for (let l = 0; l < L; l++) {
    const a: any = {
      norm: idx++,
      q: idx++,
      k: idx++,
      v: idx++,
      qn: idx++,
      kn: idx++,
      gate: idx++,
      out: idx++,
      post: idx++,
      ag: idx++,
      pre: idx++,
      d1: idx++,
      d2: idx++,
      d3: idx++
    }
    layers.push(a)
  }
  const mhc = {
    ap: idx++,
    ao: idx++,
    ar: idx++,
    bp: idx++,
    bo: idx++,
    br: idx++,
    pp: idx++,
    po: idx++,
    pr: idx++
  }
  const sites: number[] = []
  for (let s = 0; s < cfg.engram_layers?.length || 2; s++)
    sites.push(idx + 4 * s)
  idx += sites.length * 4
  idx++
  const lane = cfg.mhc_lanes
  let cells = new Float32Array(N * lane * D)
  for (let t = 0; t < N; t++)
    for (let l = 0; l < lane; l++)
      cells.set(x.subarray(t * D, t * D + D), (t * lane + l) * D)
  const pscale = rg.weights
  for (let li = 0; li < L; li++) {
    const z = cells.slice()
    const nx = new Float32Array(N * lane * D)
    for (let t = 0; t < N; t++) {
      let r = 0
      for (let j = 0; j < lane * D; j++) {
        const v = z[t * lane * D + j]
        r += v * v
      }
      r = Math.sqrt(r / (lane * D) + 1e-6)
      for (let j = 0; j < lane * D; j++)
        nx[t * lane * D + j] = z[t * lane * D + j] / r
    }
    const lc = layers[li],
      pre = await rg.mat(nx, N, lane * D, lc.pp, lane),
      u = new Float32Array(N * D)
    for (let t = 0; t < N; t++)
      for (let l = 0; l < lane; l++) {
        const w = sigmoid(
          (pre[t * lane + l] * pscale[lc.ap][li] || 0) +
            (pscale[lc.bp][li * lane + l] || 0) +
            8 * (li % lane === l ? 1 : 0) -
            4
        )
        for (let d = 0; d < D; d++)
          u[t * D + d] += w * z[(t * lane + l) * D + d]
      }
    let un = norm(u, pscale[lc.norm])
    const q0 = await rg.mat(un, N, D, lc.q, H * HD),
      k0 = await rg.mat(un, N, D, lc.k, KV * HD),
      v0 = await rg.mat(un, N, D, lc.v, KV * HD)
    let q = new Float32Array(q0.length),
      k = new Float32Array(k0.length)
    for (let t = 0; t < N; t++) {
      for (let h = 0; h < H; h++) {
        const rr = Math.sqrt(
          q0
            .subarray((t * H + h) * HD, (t * H + h + 1) * HD)
            .reduce((a, b) => a + b * b, 0) /
            HD +
            1e-6
        )
        for (let d = 0; d < HD; d++)
          q[(t * H + h) * HD + d] =
            (q0[(t * H + h) * HD + d] * (1 + pscale[lc.qn][d])) / rr
      }
      for (let h = 0; h < KV; h++) {
        let rr = 0
        for (let d = 0; d < HD; d++) {
          const vv = k0[(t * KV + h) * HD + d]
          rr += vv * vv
        }
        rr = Math.sqrt(rr / HD + 1e-6)
        for (let d = 0; d < HD; d++)
          k[(t * KV + h) * HD + d] =
            (k0[(t * KV + h) * HD + d] * (1 + pscale[lc.kn][d])) / rr
      }
    }
    q = rope(q, N, H, HD, cfg.rope_theta)
    k = rope(k, N, KV, HD, cfg.rope_theta)
    const att = attention(q, k, v0, N, H, KV, HD)
    const gate = await rg.mat(un, N, D, lc.gate, H * HD)
    for (let i = 0; i < gate.length; i++) gate[i] = sigmoid(gate[i]) * att[i]
    const ao = await rg.mat(gate, N, H * HD, lc.out, D)
    for (let i = 0; i < ao.length; i++) ao[i] *= sigmoid(pscale[lc.ag][li])
    const post = norm(new Float32Array(ao), pscale[lc.post])
    const pre2 = await rg.mat(nx, N, lane * D, lc.po, lane)
    const res = await rg.mat(nx, N, lane * D, lc.pr, lane * lane)
    for (let t = 0; t < N; t++)
      for (let l = 0; l < lane; l++) {
        const hp =
          2 *
          sigmoid(
            (pre2[t * lane + l] * pscale[lc.ao][li] || 0) +
              (pscale[lc.bo][li * lane + l] || 0) -
              4 * (l === li % lane ? 0 : 1)
          )
        for (let j = 0; j < lane; j++) {
          let rr =
            (res[t * lane * lane + l * lane + j] || 0) *
              (pscale[lc.ar][li] || 0) +
            (pscale[lc.br][li * lane + j] || 0)
          rr = Math.exp(rr)
          cells[(t * lane + l) * D] += 0
          void rr
        }
        for (let d = 0; d < D; d++)
          cells[(t * lane + l) * D + d] =
            z[(t * lane + l) * D + d] + hp * post[t * D + d]
      }
    // Hadamard MLP residual.
    const hpre = norm(cells.slice(0, N * lane * D), pscale[lc.pre])
    for (let t = 0; t < N; t++) {
      const a = hpre.subarray(t * lane * D, (t + 1) * lane * D)
      const tmp = new Float32Array(cfg.hada_n)
      tmp.set(a)
      fwht(tmp, 0, cfg.hada_n)
      for (let j = 0; j < cfg.hada_n; j++)
        tmp[j] =
          sigmoid(tmp[j] * (pscale[lc.d2][j] || 1)) *
          tmp[j] *
          (pscale[lc.d1][j] || 1)
      fwht(tmp, 0, cfg.hada_n)
      for (let d = 0; d < D; d++)
        for (let l = 0; l < lane; l++)
          cells[(t * lane + l) * D + d] +=
            (tmp[d] * (pscale[lc.d3][d] || 0.02)) / lane
    }
    onStep(li + 1)
  }
  let finalIdx = idx
  const fn = norm(
    new Float32Array(
      Array.from({ length: N * D }, (_, i) => {
        let s = 0
        for (let l = 0; l < lane; l++)
          s += cells[(Math.floor(i / D) * lane + l) * D + (i % D)]
        return s / lane
      })
    ),
    pscale[finalIdx]
  )
  const last = new Float32Array(D)
  last.set(fn.subarray((N - 1) * D, N * D))
  const logits = await rg.mat(last, 1, D, 0, cfg.vocab_size)
  let best = 0,
    bv = -1e30
  for (let i = 0; i < logits.length; i++)
    if (logits[i] > bv) {
      bv = logits[i]
      best = i
    }
  return best
}

function add (role: string, text: string) {
  if (messages.querySelector('.empty')) messages.innerHTML = ''
  const e = document.createElement('div')
  e.className = `message ${role}`
  e.textContent = text
  messages.appendChild(e)
  messages.scrollTop = messages.scrollHeight
}
function promptFor (q: string, tools: any[]) {
  return `<|im_start|>user\n<tools>${JSON.stringify(
    tools || []
  )}</tools>\n${q}<|im_end|>\n<|im_start|>assistant\n`
}
async function send () {
  const q = $<HTMLTextAreaElement>('query').value.trim()
  if (!q) return
  if (!model?.tokenizer || !runner) {
    add('assistant', '请先加载包含 tokenizer 的 .cact 模型。')
    return
  }
  const b = $<HTMLButtonElement>('send')
  b.disabled = true
  add('user', q)
  const start = performance.now()
  try {
    const tools = JSON.parse($<HTMLTextAreaElement>('tools').value || '[]')
    const ids = [2, ...model.tokenizer.encode(promptFor(q, tools))]
    const max = Number($<HTMLInputElement>('maxTokens').value) || 96
    let next = 0,
      generated: number[] = []
    for (let i = 0; i < max; i++) {
      next = await infer(
        model,
        runner,
        [...ids, ...generated],
        max,
        n =>
          (progress.textContent = `WebGPU Needle 层 ${n}/${model.geometry.num_layers}`)
      )
      generated.push(next)
      if (next === 1) break
    }
    const text = model.tokenizer.decode(generated),
      m = text.match(/<tool_call>([\s\S]*?)<\/tool_call>/)
    let shown = text
    if (m)
      try {
        shown = JSON.stringify(JSON.parse(m[1]), null, 2)
      } catch {}
    add('assistant', shown || '(模型返回空结果)')
    $('timing').textContent = `· ${Math.round(
      performance.now() - start
    )} ms · WebGPU`
  } catch (e: any) {
    add('assistant', '推理失败：' + (e?.message || e))
  } finally {
    progress.textContent = ''
    b.disabled = false
  }
}

$('modelFile').addEventListener('change', async e => {
  const f = (e.target as HTMLInputElement).files?.[0]
  if (!f) return
  $<HTMLDivElement>('modelInfo').textContent = '正在加载并解码 CQ 权重…'
  try {
    if (!device) throw Error('WebGPU 尚未初始化')
    model = parseCact(await f.arrayBuffer())
    if (!model.tokenizer) throw Error('模型缺少 tokenizer RAW tensor')
    runner = new NeedleGPU(device, model)
    $<HTMLDivElement>('modelInfo').textContent = `${f.name} · ${
      model.geometry.d_model
    }d · ${model.geometry.num_layers} 层 · ${
      model.tensors.length
    } tensors · GPU ${Math.round(
      runner.weights.reduce((n, a) => n + a.byteLength, 0) / 1048576
    )} MB`
    $<HTMLDivElement>('modelInfo').className = 'ok'
  } catch (e: any) {
    $<HTMLDivElement>('modelInfo').textContent = e.message
    $<HTMLDivElement>('modelInfo').className = 'bad'
  }
})
$('send').addEventListener('click', send)
$('query').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    send()
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
;(async () => {
  try {
    if (!('gpu' in navigator)) {
      gpuStatus.textContent = '当前浏览器不支持 WebGPU'
      gpuStatus.className = 'status bad'
      return
    }
    const a = await (navigator as any).gpu.requestAdapter()
    if (!a) throw Error('未找到 WebGPU 适配器')
    device = await a.requestDevice()
    gpuStatus.textContent = `WebGPU 已启用 · ${a.name || '默认适配器'}`
    gpuStatus.className = 'status ok'
  } catch (e: any) {
    gpuStatus.textContent = e.message
    gpuStatus.className = 'status bad'
  }
})()
