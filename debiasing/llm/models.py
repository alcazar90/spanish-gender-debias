import json
from abc import ABC, abstractmethod
from enum import Enum

import requests
from pydantic import BaseModel

from debiasing.configs import settings


class LLMMessage(BaseModel):
    class MessageRole(Enum):
        USER = "user"
        SYSTEM = "system"

    role: MessageRole
    content: str


class ModelConfigs(BaseModel):
    max_tokens: int = settings.MAX_TOKENS
    temperature: float = settings.TEMPERATURE


class LLMModel(ABC):
    def __init__(
        self,
        configs: ModelConfigs | None = None,
        system: str | None = None,
    ):
        self.configs = configs if configs is not None else ModelConfigs()
        self.system = system

    @abstractmethod
    def get_answer(
        self,
        messages: list[str],
        system: str | None = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError


class AntrophicCompletion(LLMModel):
    def __init__(
        self,
        configs: ModelConfigs | None = None,
        system: str | None = None,
        model_id: str = settings.ANTROPHIC_COMPLETION_MODEL,
    ):
        super().__init__(
            configs=configs,
            system=system,
        )
        self.model_id = model_id

    def get_answer(
        self,
        messages: list[LLMMessage],
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

        body = {
            "model": self.model_id,
            "messages": parsed_messages,
            "max_tokens": self.configs.max_tokens,
            "temperature": self.configs.temperature,
        }

        try:
            response = requests.post(
                settings.ANTROPHIC_CHAT_ENDPOINT,
                headers=headers,
                data=json.dumps(body),
                timeout=settings.LLM_TIMEOUT,
            )

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
    ):
        super().__init__(
            configs=configs,
            system=system,
        )
        self.model_id = model_id

    def get_answer(
        self,
        messages: list[LLMMessage],
    ) -> tuple[str, dict]:
        parsed_messages = [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        }

        body = {
            "model": self.model_id,
            "messages": parsed_messages,
            "max_tokens": self.configs.max_tokens,
            "temperature": self.configs.temperature,
        }

        try:
            response = requests.post(
                settings.OPENAI_CHAT_ENDPOINT,
                headers=headers,
                json=body,
                timeout=settings.LLM_TIMEOUT,
            )

            response.raise_for_status()
            response = response.json()
            text = response["choices"][0]["message"]["content"]
            return text, response
        except requests.exceptions.RequestException as err:
            print(f"Request failed: {err}")
            return str(err), {}
