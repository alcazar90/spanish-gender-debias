from enum import Enum

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    class MessageRole(Enum):
        USER = "user"
        SYSTEM = "system"

    role: MessageRole
    content: str


class LLMToolDefinition(BaseModel):
    """Tool definition for the LLM model. Provides the necessary information to dump the tool to the respective API"""

    name: str
    description: str
    input_schema: dict = Field(
        None,
        alias="inputSchema",
    )

    def openai_dump(self):
        """See OpenAI API reference chat completion for function calling: https://platform.openai.com/docs/api-reference/chat/create"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": self.input_schema,
            },
        }

    def anthropic_dump(self):
        """See Anthropic API reference chat completion for tool use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#json-mode"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "json": self.input_schema,
            },
        }
