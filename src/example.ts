export type ExampleTool = {
  name: string
  description: string
  parameters: Record<string, unknown>
}

export const EXAMPLE_TOOLS: ExampleTool[] = [
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

export const EXAMPLE_PROMPT = '调用 set_lights, 房间 1, 亮度 0'
