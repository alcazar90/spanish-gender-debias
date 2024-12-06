from abc import ABC, abstractmethod
from enum import Enum

import json
import requests
from pydantic import BaseModel

from debiasing.configs import settings

class LLMMessage(BaseModel):
    class MessageRole(Enum):
        USER = "user"
        SYSTEM = "system"

    role: MessageRole
    content: str


class LLMModel(ABC):
    def __init__(self):
        self.settings = settings

    @abstractmethod
    def get_answer(
        self,
        messages: list[str],
    ) -> tuple[str, dict]:
        pass

class Antrophic(LLMModel):
    def __init__(self):
        super().__init__()

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
            "content-type": "application/json"
        }

        body = {
            "model": settings.ANTROPHIC_COMPLETION_MODEL,
            "messages": parsed_messages,
            "max_tokens": settings.MAX_TOKENS,
            "temperature": settings.TEMPERATURE,
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
            return response
            # text = response["choices"][0]["message"]["content"]
            # return text, response
        except requests.exceptions.RequestException as err:
            print(f"Request failed: {err}")
            return str(err), {}


class OpenAICompletion(LLMModel):
    def __init__(self):
        super().__init__()

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
            "Authorization": f"Bearer {self.settings.OPENAI_API_KEY}",
        }

        body = {
            "model": self.settings.OPENAI_COMPLETION_MODEL,
            "messages": parsed_messages,
            "max_tokens": self.settings.MAX_TOKENS,
            "temperature": self.settings.TEMPERATURE,
        }

        try:
            response = requests.post(
                settings.OPENAI_CHAT_ENDPOINT,
                headers=headers,
                json=body,
                timeout=self.settings.LLM_TIMEOUT,
            )

            response.raise_for_status()
            response = response.json()
            text = response["choices"][0]["message"]["content"]
            return text, response
        except requests.exceptions.RequestException as err:
            print(f"Request failed: {err}")
            return str(err), {}
