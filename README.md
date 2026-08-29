# Needle WebUI

浏览器端的 Needle 2 `.cact` 推理 UI。模型文件直接由浏览器读取，CQ 权重在加载阶段解码，矩阵计算通过 WebGPU compute shader 执行。

## 开发

```sh
npm install
npm run dev
```

打开 Vite 页面后选择由 `needle build` 生成的 `.cact` 文件，再输入请求。

## 实现说明

- `.cact` 头、tensor directory、CQ2/CQ3/CQ4（以及 ternary record）解析参考 `cactus-compute/needle` 的 `needle/model/export.py`。
- SentencePiece tokenizer 使用 `.cact` 中的 RAW tokenizer dump，不需要额外下载 tokenizer。
- SAN 的 mHC、Hadamard MLP、GQA/RoPE、engram lookup 和 greedy decoding 均在浏览器侧实现；线性层的 GEMM 使用 WebGPU compute shader。
- 工具输入沿用 Needle 的 `<tools>...</tools>` / `<tool_call>...</tool_call>` JSON 契约。

Needle 2 的官方 Python API 仍以 `cactus-compute/needle` 为参考；浏览器实现不依赖 Python、JAX 或后端服务。

> 注意：浏览器 WebGPU 的单个 storage buffer 有实现相关上限；当前运行时将权重拆成独立 tensor buffer，以避免把整个模型塞进单个 storage binding。解码后的 FP32 权重会明显大于原始 `.cact` 文件，因此需要足够的系统/GPU 内存。
