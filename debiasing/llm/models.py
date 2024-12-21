import json
from abc import ABC, abstractmethod

import requests
from pydantic import BaseModel

from debiasing.configs import logger, settings
from debiasing.llm.utils import LLMMessage, LLMToolDefinition


# TODO: Create a decorator for LLMModel.get_answer to modify the behavior and implement ReACT...
class ModelConfigs(BaseModel):
    max_tokens: int = settings.MAX_TOKENS
    temperature: float = settings.TEMPERATURE


class LLMModel(ABC):
    def __init__(
        self,
        configs: ModelConfigs | None = None,
        system: str | None = None,
        tools: list[LLMToolDefinition] | None = None,
    ):
        self.configs = configs if configs is not None else ModelConfigs()
        self.system = system
        self.tools = tools or []

    @abstractmethod
    def get_answer(
        self,
        messages: list[str],
        system: str | None = None,
        force_tool: bool = False,
    ) -> tuple[str, dict]:
        raise NotImplementedError


class AntrophicCompletion(LLMModel):
    def __init__(
        self,
        configs: ModelConfigs | None = None,
        system: str | None = None,
        model_id: str = settings.ANTROPHIC_COMPLETION_MODEL,
        tools: list[LLMToolDefinition] | None = None,
    ):
        super().__init__(
            configs=configs,
            system=system,
            tools=tools,
        )
        self.model_id = model_id

    def get_answer(
        self,
        messages: list[LLMMessage],
        force_tool: bool = False,
    ) -> tuple[str, dict]:
        parsed_messages = [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]

        headers = {
            "x-api-key": settings.ANTROPHIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        tool_config = (
            [tool.anthropic_dump() for tool in self.tools] if self.tools else None
        )

        # Ref: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#forcing-tool-use
        body = {
            "model": self.model_id,
            "messages": parsed_messages,
            "max_tokens": self.configs.max_tokens,
            "temperature": self.configs.temperature,
            **(
                {
                    "system": self.system,
                }
                if self.system
                else {}
            ),
            **(
                {
                    "tools": tool_config,
                    "tool_choice": {"type": "any" if force_tool else "auto"},
                }
                if tool_config
                else {}
            ),
        }

        try:
            response = requests.post(
                settings.ANTROPHIC_CHAT_ENDPOINT,
                headers=headers,
                data=json.dumps(body),
                timeout=settings.LLM_TIMEOUT,
            )
            logger.info(f"LLM Anthropic response: {response.text}")

            # TODO: determine if the LLM response called a tool
            # In Anthropic, 'stop_reason' == 'tool_use'if a tool was used
            response.raise_for_status()
            response = response.json()
            text = response["content"][0]["text"]
            return text, response
        except requests.exceptions.RequestException as err:
            print(f"Request failed: {err}")
            return str(err), {}


class OpenAICompletion(LLMModel):
    def __init__(
        self,
        configs: ModelConfigs | None = None,
        system: str | None = None,
        model_id: str = settings.OPENAI_COMPLETION_MODEL,
        tools: list[LLMToolDefinition] | None = None,
    ):
        super().__init__(
            configs=configs,
            system=system,
            tools=tools,
        )
        self.model_id = model_id

    def get_answer(
        self,
        messages: list[LLMMessage],
        force_tool: bool = False,
    ) -> tuple[str, dict]:
        parsed_messages = [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]
        if self.system:
            parsed_messages.insert(0, {"role": "system", "content": self.system})

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        }

        # Note: tool_choice is required if tools are provided
        tool_config = (
            [tool.openai_dump() for tool in self.tools] if self.tools else None
        )

        body = {
            "model": self.model_id,
            "messages": parsed_messages,
            "max_completion_tokens": self.configs.max_tokens,
            "temperature": self.configs.temperature,
            **(
                {
                    "tools": tool_config,
                    "tool_choice": "required" if force_tool else "auto",
                }
                if tool_config
                else {}
            ),
        }

        try:
            response = requests.post(
                settings.OPENAI_CHAT_ENDPOINT,
                headers=headers,
                json=body,
                timeout=settings.LLM_TIMEOUT,
            )

            logger.info(f"LLM OpenAI response: {response.text}")

            # TODO: determine if the LLM response called a tool
            # In OpenAI, in response['choices'] if a response_part['finish_reason'] == 'tool_calls' then a tool was used
            response.raise_for_status()
            response = response.json()
            text = response["choices"][0]["message"]["content"]
            return text, response
        except requests.exceptions.RequestException as err:
            print(f"Request failed: {err}")
            print(f"response content: {response.content}")
            return str(err), {}
