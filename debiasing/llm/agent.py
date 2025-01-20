from abc import ABC, abstractmethod
from debiasing.llm.models import AntrophicCompletion, ModelConfigs, OpenAICompletion
from debiasing.llm.tools import DEBIASER, GENDER_BIAS_MULTI_LABEL_CLASSIFIER
from debiasing.configs import logger
from debiasing.llm.utils import LLMMessage
from typing import Any
from pydantic import BaseModel, Field


class ToolActivation(BaseModel):
    """Log of the tools activated and their results during the execution"""

    tool_name: str
    tool_results: dict[str, Any]
    step_id: int | None = Field(
        ...,
        description="Step of the agent execution where the tool was activated, i.e. related to the agent's control flow",
    )


class AgentResponse(BaseModel):
    """Response of the agent after the execution of the task. Includes the messages exchanged with the user, the tools activated, the LLM responses, and the final response"""

    tool_activations: list[ToolActivation] = []
    messages: list[LLMMessage] = []
    llm_responses: list[dict[str, Any]] = []
    response: str | None = Field(
        None,
        description="The response of the agent after the execution of the task, it represents a terminal state in the agent's control flow",
    )


class AgentPrediction(BaseModel):
    """Prediction of the agent given an input message or a list of messages. This is the dataclass for the output of the Debiaser.prediction method"""
    input: str
    biases: str | None = None
    scores: str | None = None
    debias_reasoning: str | None = None
    output: str | None = None


class AgentError(Exception):
    def __init__(self, message: str, agent_response: AgentResponse):
        super().__init__(message)
        self.agent_response = agent_response


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
    def execute_task(
        self,
        msgs: list[LLMMessage],
    ) -> AgentResponse:
        raise NotImplementedError

    @abstractmethod
    def prediction(
        self,
        input: str | LLMMessage | list[LLMMessage],
    ) -> dict[str, Any]:
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
        self._DEBIASING_FAILURE_MSG = "Debiasing tool not activated"

    def execute_task(
        self,
        msgs,
    ) -> AgentResponse:
        msgs = msgs.copy()
        agent_response = AgentResponse(messages=msgs)
        text, tool, response = self.llm.get_answer(msgs)
        agent_response.llm_responses.append(response)

        # Determine whether there is gender bias in the text
        if tool and tool.name == GENDER_BIAS_MULTI_LABEL_CLASSIFIER.name:
            logger.info("Debiaser tool detected")

            bias_label_detected = tool.arguments.get("bias_label", None)
            bias_text_detected = tool.arguments.get("bias_text", None)
            score_levels = tool.arguments.get("score_label", None)

            agent_response.tool_activations.append(
                ToolActivation(
                    tool_name=tool.name,
                    tool_results={
                        "bias_labels": bias_label_detected,
                        "bias_texts": bias_text_detected,
                        "scores": score_levels,
                    },
                    step_id=1,
                )
            )

            # Check if the classifier detected no bias. In this case, the text is considered unbiased.
            if len(bias_label_detected) == 1 and bias_label_detected[0] == "UNBIASED":
                logger.info("Gender bias multilabel classifier detected no bias")
                agent_response.response = self._UNBIASED_OUTPUT
                return agent_response

            # Otherwise, proceed with the debiasing process preparing the messages given the bias detected, i.e. format the prompt
            tool_result_into_text = "Analyzing the previous text I have the following information about gender biases:\n"
            for text, label, score in zip(
                bias_text_detected, bias_label_detected, score_levels
            ):
                tool_result_into_text += (
                    f"The text '{text}' contains {label} with score {score}\n"
                )

            logger.info(tool_result_into_text)

            msgs += [
                LLMMessage(
                    role=LLMMessage.MessageRole.SYSTEM,
                    content=tool_result_into_text,
                ),
                LLMMessage(
                    role=LLMMessage.MessageRole.USER,
                    content="Now you will debias the text",
                ),
            ]

            text, tool, response = self.llm.get_answer(msgs)
            agent_response.llm_responses.append(response)

            # Determine whether the debiaser tool was activated, i.e. step 2 of the agent execution, or control flow
            if tool and tool.name == DEBIASER.name:
                for r in tool.arguments["reasoning"]:
                    logger.info(r)

                # Save the results of the debiasing process, i.e. the debiased text and the reasoning
                agent_response.tool_activations.append(
                    ToolActivation(
                        tool_name=tool.name,
                        tool_results={
                            "debiasing_text": tool.arguments["debiasing_text"],
                            "reasoning": tool.arguments["reasoning"],
                        },
                        step_id=2,
                    )
                )
                # TODO: insert a "reasoning loop" that works with another LLM about the reasons give by the tool
                # trying to iterate and ask for more information about the bias detected
                agent_response.response = tool.arguments["debiasing_text"]
                return agent_response
            else:
                # If the debiaser tool was not activated, return an AgentError which means the debiasing process failed
                # and the text is considered biased.
                logger.error(self._DEBIASING_FAILURE_MSG)
                agent_response.response = self._DEBIASING_FAILURE_MSG
                raise AgentError(self._DEBIASING_FAILURE_MSG, agent_response)

        agent_response.response = self._UNBIASED_OUTPUT
        return agent_response

    def prediction(
        self,
        input: str | LLMMessage | list[LLMMessage],
    ) -> AgentPrediction:
        """
        Predict the output of the agent given an input message or a list of messages
        """
        # Capture the input
        if isinstance(input, str):
            input_str = input
            input = [LLMMessage(role=LLMMessage.MessageRole.USER, content=input)]
        elif isinstance(input, LLMMessage):
            input_str = input.content
            input = [input]
        elif isinstance(input, list) and all(
            isinstance(msg, LLMMessage) for msg in input
        ):
            input_str = input[0].content
        else:
            raise ValueError("Invalid input type")

        debiaser_response = self.execute_task(input)

        # Create an AgentPrediction object
        prediction = AgentPrediction(input=input_str, output=self._UNBIASED_OUTPUT)

        # Parse the debiaser response into the output dictionary
        for tool in debiaser_response.tool_activations:
            if tool.tool_name == GENDER_BIAS_MULTI_LABEL_CLASSIFIER.name:
                prediction.biases = ", ".join(
                    tool.tool_results.get("bias_labels", [""])
                )
                prediction.scores = ", ".join(
                    [str(score) for score in tool.tool_results.get("scores", [""])]
                )
            elif tool.tool_name == DEBIASER.name:
                prediction.debias_reasoning = ", ".join(
                    tool.tool_results.get("reasoning", [""])
                )
                prediction.output = tool.tool_results.get("debiasing_text", None)
        return prediction
