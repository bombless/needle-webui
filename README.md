# Needle WebUI

浏览器端的 Needle 2 `.cact` 推理 UI。模型文件直接由浏览器读取，CQ 权重在加载阶段解码；线性层 GEMM 支持在现有 WebGPU compute shader 与 `jax-js` 之间切换。

## 开发

```sh
npm install
npm run dev
```

打开 Vite 页面后选择由 `needle build` 生成的 `.cact` 文件，再输入请求。

## 计算后端

页面的“计算后端”可以选择：

- **WebGPU**：保留原来的 Needle WebGPU compute shader 实现。
- **jax-js**：使用 `@jax-js/jax` 的 NumPy/JAX API 执行 GEMM；优先初始化 jax-js 的 WebGPU backend，不可用时自动回退 WASM。

两种模式共享同一套 `.cact` 解析、CQ 解码、SAN/mHC、Hadamard MLP、GQA/RoPE、engram lookup、tokenizer 与 greedy decoding，因此可以直接做数值和性能 A/B 对比。jax-js 的 WebGPU/WASM backend 由 `@jax-js/jax` 自己管理，不复用页面已有的 `GPUDevice`。

## Node.js CLI

Node.js 使用 `webgpu`（Dawn）获取同一物理 GPU 的 WebGPU adapter；浏览器中的 `GPUDevice` 无法跨进程直接转移，因此 CLI worker 会各自创建兼容的 device。多个 provider 会并行运行，全部完成或超时后一次性打印 JSON 结果：

```sh
npm run cli -- --cact needle2.cact --tools tools.json --prompt "my prompt"
```

`--tools` 和 `--prompt` 都是可选的；省略时直接复用页面“填入示例工具”里的工具定义和请求：`set_lights`，提示词为 `调用 set_lights, 房间 1, 亮度 0`。因此最简调用为：

```sh
npm run cli -- --cact needle2.cact
```

可选参数：`--providers webgpu`（也可传逗号分隔的 Dawn backend，例如 `vulkan,d3d12`）、`--timeout 120000`、`--max-tokens 96`。注意 npm 参数需要放在第二个 `--` 之后。

## 实现说明

- `.cact` 头、tensor directory、CQ2/CQ3/CQ4（以及 ternary record）解析参考 `cactus-compute/needle` 的 `needle/model/export.py`。
- SentencePiece tokenizer 使用 `.cact` 中的 RAW tokenizer dump，不需要额外下载 tokenizer。
- SAN 的 mHC、Hadamard MLP、GQA/RoPE、engram lookup 和 greedy decoding 均在浏览器侧实现。
- 线性层的 GEMM 可以使用原生 WebGPU shader，或者通过 `@jax-js/jax` 的 `np.einsum` 交给 jax-js backend。
- 工具输入沿用 Needle 的 `<tools>...</tools>` / `<tool_call>...</tool_call>` JSON 契约。

Needle 2 的官方 Python API 仍以 `cactus-compute/needle` 为参考；浏览器实现不依赖 Python 或后端服务。jax-js 用于浏览器端数值计算，不负责模型格式解析。

> 注意：浏览器 WebGPU 的单个 storage buffer 有实现相关上限；当前原生 WebGPU 运行时将权重拆成独立 tensor buffer，以避免把整个模型塞进单个 storage binding。解码后的 FP32 权重会明显大于原始 `.cact` 文件，因此需要足够的系统/GPU 内存。
