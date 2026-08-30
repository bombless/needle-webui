export function buildPrompt (tools: unknown, prompt: string) {
  return `<|im_start|>user\n<tools>${JSON.stringify(tools)}</tools>\n${prompt}<|im_end|>\n<|im_start|>assistant\n`
}
