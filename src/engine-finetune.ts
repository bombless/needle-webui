import { Model, Runtime, topCandidates } from './engine-core.js'

type Counters = { dispatches: number; flops: number; forwardMs: number }

function fwht (a: Float32Array) {
  for (let n = 1; n < a.length; n <<= 1)
    for (let i = 0; i < a.length; i += n << 1)
      for (let j = 0; j < n; j++) {
        const x = a[i + j], y = a[i + j + n]
        a[i + j] = x + y
        a[i + j + n] = x - y
      }
  const q = 1 / Math.sqrt(a.length)
  for (let i = 0; i < a.length; i++) a[i] *= q
}

function norm (x: Float32Array, sc: Float32Array) {
  const d = sc.length, o = new Float32Array(x.length)
  for (let i = 0; i < x.length / d; i++) {
    let r = 0
    for (let j = 0; j < d; j++) r += x[i * d + j] * x[i * d + j]
    r = Math.sqrt(r / d + 1e-6)
    for (let j = 0; j < d; j++) o[i * d + j] = x[i * d + j] * (1 + sc[j]) / r
  }
  return o
}

function silu (x: number) { return x / (1 + Math.exp(-Math.max(-40, Math.min(40, x)))) }

function sink (a: Float32Array, n: number) {
  for (let z = 0; z < 20; z++) {
    for (let i = 0; i < n; i++) {
      let mx = -Infinity
      for (let j = 0; j < n; j++) mx = Math.max(mx, a[i * n + j])
      let s = 0
      for (let j = 0; j < n; j++) { const e = Math.exp(a[i * n + j] - mx); a[i * n + j] = e; s += e }
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

function att (q: Float32Array, k: Float32Array, v: Float32Array, T: number, h: number, kh: number, d: number) {
  const o = new Float32Array(T * h * d), rep = h / kh, sc = 1 / Math.sqrt(d)
  for (let t = 0; t < T; t++) for (let z = 0; z < h; z++) {
    const kz = Math.floor(z / rep)
    let mx = -1e30
    for (let j = 0; j <= t; j++) {
      let s = 0
      for (let x = 0; x < d; x++) s += q[(t * h + z) * d + x] * k[(j * kh + kz) * d + x] * sc
      mx = Math.max(mx, s)
    }
    let den = 0
    for (let j = 0; j <= t; j++) {
      let s = 0
      for (let x = 0; x < d; x++) s += q[(t * h + z) * d + x] * k[(j * kh + kz) * d + x] * sc
      den += Math.exp(s - mx)
    }
    for (let x = 0; x < d; x++) {
      let s = 0
      for (let j = 0; j <= t; j++) {
        let z2 = 0
        for (let y = 0; y < d; y++) z2 += q[(t * h + z) * d + y] * k[(j * kh + kz) * d + y] * sc
        s += Math.exp(z2 - mx) * v[(j * kh + kz) * d + x]
      }
      o[(t * h + z) * d + x] = s / den
    }
  }
  return o
}

function rotate (x: Float32Array, T: number, h: number, d: number, theta: number) {
  const o = x.slice(), half = d / 2
  for (let t = 0; t < T; t++) for (let z = 0; z < h; z++) for (let j = 0; j < half; j++) {
    const a = t * Math.pow(theta, (-2 * j) / d), c = Math.cos(a), s = Math.sin(a)
    const i = (t * h + z) * d + j, q = i + half, X = x[i], Y = x[q]
    o[i] = X * c - Y * s; o[q] = Y * c + X * s
  }
  return o
}

function hashIndex (tokens: number[], t: number, order: number, oi: number, head: number, heads: number, slots: number) {
  let a = Math.imul(0x9e3779b9, oi * heads + head + 1) >>> 0
  for (let j = 0; j < order; j++) {
    const tok = t - j >= 0 ? tokens[t - j] : 0
    a = Math.imul((a ^ tok) >>> 0, 0x01000193) >>> 0
  }
  return ((a ^ (a >>> 15)) >>> 0) % slots
}

async function generate (tokens: number[], m: any, r: any, step: (n: number) => void, c: Counters, isCancelled: () => boolean = () => false) {
  if (isCancelled()) throw Error('推理已停止')
  const g = m.g, D = g.d_model, Hd = g.head_dim, L = g.num_layers, n = g.mhc_lanes, N = tokens.length
  let wi = 1
  const ls: any[] = []
  for (let l = 0; l < L; l++) ls.push({ norm: wi++, q: wi++, k: wi++, v: wi++, qn: wi++, kn: wi++, gate: wi++, out: wi++, post: wi++, ag: wi++, pre: wi++, d1: wi++, d2: wi++, d3: wi++ })
  const mh = { ap: wi++, apost: wi++, ar: wi++, bp: wi++, bpost: wi++, br: wi++, pp: wi++, ppo: wi++, pr: wi++ }
  const es: number[] = []
  for (let s = 0; s < (g.engram_layers as number[]).length; s++) es.push(wi), wi += 4
  const final = wi++

  // Needle: embedding(tokens) * sqrt(d_model), before the first mHC block.
  let x = new Float32Array(N * D * n)
  const E = r.w[0], embedScale = Math.sqrt(D)
  for (let t = 0; t < N; t++) for (let lane = 0; lane < n; lane++) {
    const dst = (t * n + lane) * D, src = tokens[t] * D
    for (let d = 0; d < D; d++) x[dst + d] = E[src + d] * embedScale
  }

  for (let l = 0; l < L; l++) {
    if (isCancelled()) throw Error('推理已停止')
    const z = x.slice(), nx = new Float32Array(z.length)
    for (let t = 0; t < N; t++) {
      let rr = 0
      for (let j = 0; j < n * D; j++) { const v = z[t * n * D + j]; rr += v * v }
      rr = Math.sqrt(rr / (n * D) + 1e-6)
      for (let j = 0; j < n * D; j++) nx[t * n * D + j] = z[t * n * D + j] / rr
    }
    const lc = ls[l], hp = await r.mm(nx, N, n * D, mh.pp, n), u = new Float32Array(N * D)
    for (let t = 0; t < N; t++) for (let lane = 0; lane < n; lane++) {
      const gate = 1 / (1 + Math.exp(-(hp[t * n + lane] * r.w[mh.ap][l] + (r.w[mh.bp][l * n + lane] || 0) + 8 * (lane === l % n ? 1 : 0) - 4)))
      for (let d = 0; d < D; d++) u[t * D + d] += gate * z[(t * n + lane) * D + d]
    }

    // Needle Engram: ngram_ok + learned causal 4-tap convolution + cosine gate.
    for (let si = 0; si < (g.engram_layers as number[]).length; si++) if ((g.engram_layers as number[])[si] === l) {
      const base = es[si], orders = g.engram_orders as number[], heads = g.num_engram_tables / orders.length, sub = g.engram_sub_dim
      const flat = new Float32Array(N * heads * orders.length * sub)
      for (let t = 0; t < N; t++) for (let oi = 0; oi < orders.length; oi++) for (let h = 0; h < heads; h++) {
        if (t < orders[oi] - 1) continue
        const id = hashIndex(tokens, t, orders[oi], oi, h, heads, g.engram_slots)
        const table = r.w[base], row = (oi * heads + h) * g.engram_slots + id
        flat.set(table.subarray(row * sub, row * sub + sub), (t * g.num_engram_tables + oi * heads + h) * sub)
      }
      const ek = await r.mm(flat, N, orders.length * heads * sub, base + 1, D)
      const rawEv = await r.mm(flat, N, orders.length * heads * sub, base + 2, D)
      const taps = r.w[base + 3], tapCount = g.engram_conv_taps || 4, maxOrder = Math.max(...orders), dilation = g.engram_conv_dilation || 1
      const ev = new Float32Array(N * D)
      for (let t = 0; t < N; t++) for (let d = 0; d < D; d++) {
        let sum = 0
        for (let j = 0; j < tapCount; j++) {
          const srcT = t - j * dilation
          // Python: tap_ok[j] is mask diagonal j*max(orders).
          if (srcT < 0 || t < j * maxOrder) continue
          sum += taps[j * D + d] * rawEv[srcT * D + d]
        }
        ev[t * D + d] = sum
      }
      for (let t = 0; t < N; t++) {
        let dot = 0, nxv = 0, ekv = 0
        for (let d = 0; d < D; d++) { const uv = u[t * D + d], kv = ek[t * D + d]; dot += uv * kv; nxv += uv * uv; ekv += kv * kv }
        const alpha = 1 / (1 + Math.exp(-dot / Math.sqrt(Math.max(1, nxv * ekv))))
        for (let d = 0; d < D; d++) u[t * D + d] += alpha * ev[t * D + d]
      }
    }

    const un = norm(u, r.w[lc.norm])
    const q0 = await r.mm(un, N, D, lc.q, g.num_heads * Hd), k0 = await r.mm(un, N, D, lc.k, g.num_kv_heads * Hd), v0 = await r.mm(un, N, D, lc.v, g.num_kv_heads * Hd)
    let q = new Float32Array(q0.length), k = new Float32Array(k0.length)
    for (let t = 0; t < N; t++) {
      for (let h = 0; h < g.num_heads; h++) {
        let rr = 0
        for (let d = 0; d < Hd; d++) { const v = q0[(t * g.num_heads + h) * Hd + d]; rr += v * v }
        rr = Math.sqrt(rr / Hd + 1e-6)
        for (let d = 0; d < Hd; d++) q[(t * g.num_heads + h) * Hd + d] = q0[(t * g.num_heads + h) * Hd + d] * (1 + r.w[lc.qn][d]) / rr
      }
      for (let h = 0; h < g.num_kv_heads; h++) {
        let rr = 0
        for (let d = 0; d < Hd; d++) { const v = k0[(t * g.num_kv_heads + h) * Hd + d]; rr += v * v }
        rr = Math.sqrt(rr / Hd + 1e-6)
        for (let d = 0; d < Hd; d++) k[(t * g.num_kv_heads + h) * Hd + d] = k0[(t * g.num_kv_heads + h) * Hd + d] * (1 + r.w[lc.kn][d]) / rr
      }
    }
    q = rotate(q, N, g.num_heads, Hd, g.rope_theta); k = rotate(k, N, g.num_kv_heads, Hd, g.rope_theta)
    let a = att(q, k, v0, N, g.num_heads, g.num_kv_heads, Hd)
    const gate = await r.mm(un, N, D, lc.gate, g.num_heads * Hd)
    for (let i = 0; i < a.length; i++) a[i] *= 1 / (1 + Math.exp(-gate[i]))
    let ao = await r.mm(a, N, g.num_heads * Hd, lc.out, D); ao = norm(ao, r.w[lc.post])
    const block = new Float32Array(N * D)
    for (let i = 0; i < block.length; i++) block[i] = un[i] + (r.w[lc.ag][l] ? 1 / (1 + Math.exp(-r.w[lc.ag][l])) : 0) * ao[i]
    const h = norm(block, r.w[lc.pre]), hada = g.hada_n, tmp = new Float32Array(hada)
    for (let t = 0; t < N; t++) {
      tmp.fill(0); tmp.set(h.subarray(t * D, t * D + D))
      for (let j = 0; j < hada; j++) tmp[j] *= r.w[lc.d1][j] || 1
      fwht(tmp)
      for (let j = 0; j < hada; j++) tmp[j] = silu(tmp[j] * (r.w[lc.d2][j] || 1))
      fwht(tmp)
      for (let d = 0; d < D; d++) block[t * D + d] += tmp[d] * (r.w[lc.d3][d] || 0.02)
    }
    const hp2 = await r.mm(nx, N, n * D, mh.ppo, n), res = await r.mm(nx, N, n * D, mh.pr, n * n)
    for (let t = 0; t < N; t++) {
      const sm = sink(res.subarray(t * n * n, (t + 1) * n * n), n)
      for (let i = 0; i < n; i++) for (let d = 0; d < D; d++) {
        let v = 0
        for (let j = 0; j < n; j++) v += sm[i * n + j] * z[(t * n + j) * D + d]
        const post = 2 / (1 + Math.exp(-(hp2[t * n + i] * r.w[mh.apost][l] + (r.w[mh.bpost][l * n + i] || 0) - 4 * (1 - (i === l % n ? 1 : 0)))))
        x[(t * n + i) * D + d] = v + post * (block[t * D + d] - un[t * D + d])
      }
    }
    step(l + 1)
  }
  const last = new Float32Array(D)
  for (let d = 0; d < D; d++) { let s = 0; for (let l = 0; l < n; l++) s += x[((N - 1) * n + l) * D + d]; last[d] = s / n }
  const fn = norm(last, r.w[final])
  return await r.mm(fn, 1, D, 0, g.vocab_size)
}

export { Model, Runtime, generate, topCandidates, type Counters }
