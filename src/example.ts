export type ExampleTool = {
  name: string
  description: string
  parameters: Record<string, unknown>
}

export const EXAMPLE_TOOLS: ExampleTool[] = [
  {
    name: 'set_lights',
    description: 'set brightness',
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

export const EXAMPLE_PROMPT = 'turn brightness in room 1 to 0'
