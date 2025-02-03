from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from debiasing.configs import settings


class TextPart(BaseModel):
    type: str = "text"
    text: str


class ToolPart(BaseModel):
    type: str = "tool"
    name: str
    arguments: dict[str, Any]


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
    structured_output: bool = False

    def openai_dump(self):
        """See OpenAI API reference chat completion for function calling: https://platform.openai.com/docs/api-reference/chat/create"""
        # Ref: https://github.com/openai/openai-python/blob/19ecaafeda91480d0dfd7ce44e7317220b9d48b6/src/openai/types/shared/response_format_json_schema.py#L13
        # Ref: https://openai.com/index/introducing-structured-outputs-in-the-api/
        if len(self.description) > settings.OPENAI_TOOL_DESCRIPTION_LENGTH:
            raise ValueError(
                f"The description for tool {self.name} is too long given the OpenAI ChatCompletion endpoint. Max length is 1024 characters."
            )
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": self.structured_output,
                "parameters": self.input_schema,
            },
        }

    def anthropic_dump(self):
        """See Anthropic API reference chat completion for tool use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#json-mode"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
