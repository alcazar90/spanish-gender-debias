from abc import ABC, abstractmethod
from debiasing.llm.models import AntrophicCompletion, ModelConfigs, OpenAICompletion
from debiasing.llm.tools import DEBIASER, GENDER_BIAS_MULTI_LABEL_CLASSIFIER
from debiasing.configs import logger
from debiasing.llm.utils import LLMMessage


class Agent(ABC):
    def __init__(
        self,
        provider="anthropic",
        tools=None,
        model_config=ModelConfigs(max_tokens=400, temperature=0.8),
    ):
        if provider == "anthropic":
            self.llm = AntrophicCompletion(
                configs=model_config,
                tools=tools,
            )
        elif provider == "openai":
            self.llm = OpenAICompletion(
                configs=model_config,
                tools=tools,
            )
        else:
            raise ValueError("Invalid provider")

    @abstractmethod
    def execute_task(self, msgs):
        raise NotImplementedError


class Debiaser(Agent):
    def __init__(
        self,
        provider="anthropic",
        tools=[
            GENDER_BIAS_MULTI_LABEL_CLASSIFIER,
            DEBIASER,
        ],
    ):
        super().__init__(provider, tools)
        self._UNBIASED_OUTPUT = "UNBIASED"

    def execute_task(self, msgs):
        text, tool, response = self.llm.get_answer(msgs)
        logger.debug(f"LLM Response: {response}")

        if tool and tool.name == GENDER_BIAS_MULTI_LABEL_CLASSIFIER.name:
            logger.info("Debiaser tool detected")

            bias_label_detected = tool.arguments.get("bias_label", None)
            bias_text_detected = tool.arguments.get("bias_text", None)
            score_levels = tool.arguments.get("score_label", None)

            if len(bias_label_detected) == 1 and bias_label_detected[0] == "UNBIASED":
                logger.info("Gender bias multilabel classifier detected no bias")
                return self._UNBIASED_OUTPUT
            tool_result_into_text = "Analyzing the previous text I have the following information about gender biases:\n"
            for text, label, score in zip(
                bias_text_detected, bias_label_detected, score_levels
            ):
                tool_result_into_text += (
                    f"The text '{text}' contains {label} with score {score}\n"
                )

            logger.info(tool_result_into_text)

            msgs.append(
                LLMMessage(
                    role=LLMMessage.MessageRole.SYSTEM,
                    content=tool_result_into_text,
                )
            )
            msgs.append(
                LLMMessage(
                    role=LLMMessage.MessageRole.USER,
                    content="Now youl debias the text",
                )
            )

            text, tool, response = self.llm.get_answer(msgs)
            for r in tool.arguments["reasoning"]:
                logger.info(r)

            return tool.arguments["debiasing_text"]
        return self._UNBIASED_OUTPUT
