import weave
from abc import ABC, abstractmethod
from datetime import datetime
from debiasing.llm.models import AntrophicCompletion, ModelConfigs, OpenAICompletion
from debiasing.llm.tools import DEBIASER, GENDER_BIAS_MULTI_LABEL_CLASSIFIER
from debiasing.configs import logger
from debiasing.llm.utils import LLMMessage
from typing import Any
from pydantic import BaseModel, Field
from debiasing.configs import settings

client = weave.init(settings.WANDB_PROJECT)


# System prompt to instantiate the LLM used by the Debiaser agent
DEBIASER_SYSTEM_PROMPT = weave.StringPrompt("""You are a linguistic expert in gender bias analysis specialized in Spanish language communication. 

Tools at your disposal:
- gender_bias_classifier: Identifies specific types of gender biases in text
- debiaser: Neutralizes identified gender biases while preserving semantic meaning

Core principles:
- Minimal linguistic intervention
- Maintain original text's communicative intent
- Prioritize cultural sensitivity and linguistic precision
""")


# Prompt for system message with the GENDER_BIAS_MULTI_LABEL_CLASSIFIER tool result
# and user message to ask the LLM to debias the text, both prompt correspond to the
# second step of the agent execution
DEBIASER_SYSTEM_MSG = weave.StringPrompt("""Gender bias analysis detected in the text:
{bias_details}
""")


DEBIASER_USER_MSG = weave.StringPrompt("""Proceed to neutralize the original text from the gender biases detected using the debiaser tool.
Keep in mind the following debiasing instructions:
- Neutralize while keeping the text edition as minimal as possible
- Use Spanish-specific inclusive language strategies
- Avoid complex or verbose reformulations
- Ensure natural, fluid language
""")


# Log the prompts in the Weave system
weave.publish(DEBIASER_SYSTEM_PROMPT, name="DEBIASER_SYSTEM_PROMPT")
weave.publish(DEBIASER_SYSTEM_MSG, name="DEBIASER_SYSTEM_MSG")
weave.publish(DEBIASER_USER_MSG, name="DEBIASER_USER_MSG")


# Output message when the text is considered unbiased
UNBIASED_OUTPUT = "UNBIASED"


# Output message when the debiasing process fails
DEBIASING_FAILURE_MSG = "Debiasing tool not activated"


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
        system=None,
        model_config=ModelConfigs(max_tokens=400, temperature=0.8),
    ):
        if provider == "anthropic":
            self.llm = AntrophicCompletion(
                configs=model_config,
                tools=tools,
                system=system,
            )
        elif provider == "openai":
            self.llm = OpenAICompletion(
                configs=model_config,
                tools=tools,
                system=system,
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
    def predict(
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
        system=DEBIASER_SYSTEM_PROMPT.format(),
    ):
        super().__init__(provider, tools, system)
        self._UNBIASED_OUTPUT = UNBIASED_OUTPUT
        self._DEBIASING_FAILURE_MSG = DEBIASING_FAILURE_MSG
        self._DEBIASING_SYSTEM_MSG = DEBIASER_SYSTEM_MSG
        self._DEBIASING_USER_MSG = DEBIASER_USER_MSG.format()

    @weave.op()
    def generate_debias_prompt(
        self,
        bias_texts,
        bias_labels,
        scores,
    ) -> str:
        """
        Generate the prompt text based on detected biases, labels, and scores.

        Parameters:
        bias_texts (list[str]): The texts where biases were detected.
        bias_labels (list[str]): The labels of the detected biases.
        scores (list[float]): The scores of the detected biases.

        Returns:
        str: The generated prompt text.
        """
        bias_details = ""
        for text, label, score in zip(bias_texts, bias_labels, scores):
            bias_details += f"This segment of the text '{text}' triggers a {label} bias detection with a score of {score}\n"

        prompt = self._DEBIASING_SYSTEM_MSG.format(bias_details=bias_details)
        return prompt

    @weave.op()
    def execute_task(
        self,
        msgs,
    ) -> AgentResponse:
        msgs = msgs.copy()

        agent_response = AgentResponse(messages=msgs)
        text, tool, response = self.llm.get_answer(msgs)
        agent_response.llm_responses.append(response)

        logger.info(f"LLM text: {text}")
        logger.info(f"LLM tool: {tool}")

        # Determine whether there is gender bias in the text
        if tool and tool.name == GENDER_BIAS_MULTI_LABEL_CLASSIFIER.name:
            logger.info("'gender_bias_classifier' tool activated")

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

            logger.info(f"Current agent response: {agent_response}")

            # Check if the classifier detected no bias. In this case, the text is considered unbiased.
            if len(bias_label_detected) == 1 and bias_label_detected[0] == "UNBIASED":
                logger.info("Gender bias multilabel classifier detected no bias")
                agent_response.response = self._UNBIASED_OUTPUT
                return agent_response

            # Otherwise, proceed with the debiasing process preparing the messages given the bias detected, i.e. format the prompt
            # for the user to debias the text. Take step 1 results and create a prompt for step 2
            tool_result_into_text = self.generate_debias_prompt(
                bias_text_detected, bias_label_detected, score_levels
            )

            logger.info(tool_result_into_text)

            msgs += [
                LLMMessage(
                    role=LLMMessage.MessageRole.SYSTEM,
                    content=tool_result_into_text,
                ),
                LLMMessage(
                    role=LLMMessage.MessageRole.USER,
                    content=self._DEBIASING_USER_MSG,
                ),
            ]

            logger.info(f"Current messages: {msgs}")

            # Now that we are in step 2, we need to force the tool to be activated, because we are expecting the debiaser tool to be activated
            # and make corrections to the text
            logger.info("Executing step 2 of the agent")
            # logger.info("Keep only the debiaser tool in the LLM tool inventory")
            # self.llm.update_tools(tools=[DEBIASER])
            logger.info(f"Current tools: {self.llm.tools} ")
            logger.info("Forcing tool activation by setting force_tool=True")
            text, tool, response = self.llm.get_answer(msgs, force_tool=True)
            agent_response.llm_responses.append(response)

            # Determine whether the debiaser tool was activated, i.e. step 2 of the agent execution, or control flow
            if tool and tool.name == DEBIASER.name:
                logger.info("'debiaser' tool activated")

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
                raise AgentError(agent_response.llm_responses, agent_response)

        agent_response.response = self._UNBIASED_OUTPUT
        return agent_response

    @weave.op(
        tracing_sample_rate=1.0,
        call_display_name=lambda call: f"{call.func_name}__{datetime.now()}",
    )
    def predict(
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

