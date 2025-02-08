import weave

from abc import ABC
from abc import abstractmethod
from debiasing.configs import logger
from debiasing.configs import settings
from debiasing.llm.models import AntrophicCompletion
from debiasing.llm.models import ModelConfigs
from debiasing.llm.models import OpenAICompletion
from debiasing.llm.tools import DEBIASER_TOOL
from debiasing.llm.tools import GENDER_BIAS_MULTI_LABEL_CLASSIFIER_TOOL
from debiasing.llm.prompts import CRITIC_SYSTEM_PROMPT
from debiasing.llm.prompts import CRITIC_SYSTEM_MSG
from debiasing.llm.prompts import CRITIC_USER_MSG
from debiasing.llm.prompts import CRITIC_USER_MSG_ITERATION
from debiasing.llm.prompts import DETECTOR_SYSTEM_PROMPT
from debiasing.llm.prompts import NEUTRALIZER_SYSTEM_PROMPT
from debiasing.llm.prompts import NEUTRALIZER_SYSTEM_MSG
from debiasing.llm.prompts import NEUTRALIZER_USER_MSG
from debiasing.llm.utils import LLMMessage
from enum import StrEnum
from typing import Any
from pydantic import BaseModel
from pydantic import Field


# Constants for the debiasing process, such as the output message when the text
#  is considered unbiased and the debiasing process fails
UNBIASED_OUTPUT = "UNBIASED"
NEUTRALIZER_FAILURE_MSG = "Debiasing tool not activated"
DEBIASER_OUTPUT = "The debiasing process has been completed. The final debiased text is: '{debiased_text}'"


class AgentType(StrEnum):
    """Enum for different types of agents in the debiasing system"""

    DEBIASER = "debiaser"
    BIAS_DETECTOR = "bias_detector"
    BIAS_NEUTRALIZER = "bias_neutralizer"
    DEBIASER_CRITIC = "debiaser_critic"


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
    agent_type: AgentType = Field(
        ...,
        description="Type of the agent that generated the response",
    )

    class Config:
        use_enum_values = True  # This will serialize the enum to its value


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
        model_config=ModelConfigs(
            max_tokens=settings.MAX_TOKENS,
            temperature=settings.TEMPERATURE,
        ),
    ):
        self.provider = provider
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

    def predict(
        self,
        input: str | LLMMessage | list[LLMMessage],
    ) -> dict[str, Any]:
        raise NotImplementedError


# Bias Detection Agent for step (i)
class BiasDetector(Agent):
    def __init__(
        self,
        provider="anthropic",
        tools=[GENDER_BIAS_MULTI_LABEL_CLASSIFIER_TOOL],
        system=DETECTOR_SYSTEM_PROMPT.format(),  # You'll need to define DETECTOR_SYSTEM_PROMPT
        model_config=ModelConfigs(
            max_tokens=settings.MAX_TOKENS,
            temperature=0.0,
        ),
    ):
        logger.info(f"Initializing BiasDetector with provider={provider}")
        super().__init__(
            provider,
            tools,
            system,
            model_config,
        )
        self._UNBIASED_OUTPUT = UNBIASED_OUTPUT

    @weave.op(call_display_name="Detecting bias")
    def execute_task(
        self,
        msgs: list[LLMMessage],
    ) -> AgentResponse:
        logger.info("Starting bias detection task")
        msgs = msgs.copy()
        agent_response = AgentResponse(
            messages=msgs,
            agent_type=AgentType.BIAS_DETECTOR,
        )

        logger.debug(f"Input messages for bias detection: {msgs}")
        text, tool, response = self.llm.get_answer(msgs)
        agent_response.llm_responses.append(response)

        if tool and tool.name == GENDER_BIAS_MULTI_LABEL_CLASSIFIER_TOOL.name:
            logger.info("Gender bias classifier tool activated")
            bias_label_detected = tool.arguments.get("bias_label", None)
            bias_text_detected = tool.arguments.get("bias_text", None)
            score_levels = tool.arguments.get("score_label", None)

            logger.info(f"Detected bias labels: {bias_label_detected}")
            logger.debug(f"Detected bias texts: {bias_text_detected}")
            logger.debug(f"Bias detection scores: {score_levels}")

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
            if len(bias_label_detected) == 1 and bias_label_detected[0] == "UNBIASED":
                logger.info("No bias detected in text")
                agent_response.response = self._UNBIASED_OUTPUT
                return agent_response

            logger.info("Bias detected, preparing response")
            agent_response.response = {
                "bias_detected": True,
                "bias_labels": bias_label_detected,
                "bias_texts": bias_text_detected,
                "scores": score_levels,
            }
            return agent_response

        logger.info("No bias classifier tool activation, marking as unbiased")
        agent_response.response = self._UNBIASED_OUTPUT
        return agent_response


# Debiasing Agent for step (ii)
class BiasNeutralizer(Agent):
    def __init__(
        self,
        provider="anthropic",
        tools=[DEBIASER_TOOL],
        system=NEUTRALIZER_SYSTEM_PROMPT.format(),  # You'll need to define NEUTRALIZER_SYSTEM_PROMPT
        model_config=ModelConfigs(
            max_tokens=settings.MAX_TOKENS,
            temperature=0.2,
        ),
    ):
        logger.info(f"Initializing BiasNeutralizer with provider={provider}")
        super().__init__(
            provider,
            tools,
            system,
            model_config,
        )
        self._NEUTRALIZER_FAILURE_MSG = NEUTRALIZER_FAILURE_MSG
        self._NEUTRALIZER_SYSTEM_MSG = NEUTRALIZER_SYSTEM_MSG
        self._NEUTRALIZER_USER_MSG = NEUTRALIZER_USER_MSG

    def generate_debias_prompt(
        self,
        bias_texts: list[str],
        bias_labels: list[str],
        scores: list[float],
    ) -> str:
        logger.debug("Generating debiasing prompt")
        bias_details = ""
        for text, label, score in zip(bias_texts, bias_labels, scores):
            bias_details += f"This segment of the text '{text}' triggers a {label} bias detection with a score of {score}\n"
        logger.debug(f"Generated bias details: {bias_details}")
        return self._NEUTRALIZER_SYSTEM_MSG.format(bias_details=bias_details)

    @weave.op(call_display_name="Neutralizing bias")
    def execute_task(
        self,
        msgs: list[LLMMessage],
        bias_info: dict,
    ) -> AgentResponse:
        logger.info("Starting bias neutralization task")
        logger.debug(f"Bias info received: {bias_info}")

        msgs = msgs.copy()
        agent_response = AgentResponse(
            messages=msgs,
            agent_type=AgentType.BIAS_NEUTRALIZER,
        )
        # Generate debiasing prompt based on detected biases
        debias_prompt = self.generate_debias_prompt(
            bias_info["bias_texts"], bias_info["bias_labels"], bias_info["scores"]
        )
        # Add the debiasing prompt to messages
        msgs += [
            LLMMessage(
                role=LLMMessage.MessageRole.SYSTEM,
                content=debias_prompt,
            ),
            LLMMessage(
                role=LLMMessage.MessageRole.USER,
                content=self._NEUTRALIZER_USER_MSG.format(),
            ),
        ]
        agent_response.messages.extend(msgs[-2:])

        logger.debug("Requesting debiased version using debiaser tool")
        text, tool, response = self.llm.get_answer(msgs, force_tool=True)
        agent_response.llm_responses.append(response)

        if tool and tool.name == DEBIASER_TOOL.name:
            logger.info("Debiaser tool successfully activated")
            debiasing_text = tool.arguments["debiasing_text"]
            reasoning = tool.arguments["reasoning"]

            logger.info("Debiasing completed successfully")
            logger.debug(f"Debiased text: {debiasing_text}")
            logger.debug(f"Debiasing reasoning: {reasoning}")

            agent_response.tool_activations.append(
                ToolActivation(
                    tool_name=tool.name,
                    tool_results={
                        "debiasing_text": debiasing_text,
                        "reasoning": reasoning,
                    },
                    step_id=2,
                )
            )
            agent_response.response = {
                "debiased_text": debiasing_text,
                "reasoning": reasoning,
            }
            return agent_response

        logger.error("Debiaser tool failed to activate")
        agent_response.response = self._NEUTRALIZER_FAILURE_MSG
        raise AgentError(agent_response.llm_responses, agent_response)


class DebiaserCritic(Agent):
    def __init__(
        self,
        provider="anthropic",
        tools=None,
        system=CRITIC_SYSTEM_PROMPT.format(),
    ):
        logger.info(f"Initializing DebiaserCritic with provider={provider}")
        super().__init__(
            provider,
            tools,
            system,
        )

    @weave.op(call_display_name="Critiquing debiasing")
    def execute_task(
        self,
        msgs,
    ) -> AgentResponse:
        logger.info("Starting critic task, aka self-reflection process")
        msgs = msgs.copy()
        agent_response = AgentResponse(
            messages=msgs,
            agent_type=AgentType.DEBIASER_CRITIC,
        )

        logger.debug("Requesting critique from LLM")
        text, _, response = self.llm.get_answer(msgs)
        agent_response.llm_responses.append(response)

        critique = text.text.strip()
        logger.info(f"Critique received: {critique}")

        agent_response.response = text.text.strip()  # TextPart, no tool activated
        return agent_response


# Refactored main Debiaser class to orchestrate the agents
class Debiaser(Agent):
    def __init__(
        self,
        provider="anthropic",
        detector_provider="anthropic",  # Allow different provider for detector
        neutralizer_provider="anthropic",
        critic_provider="anthropic",
        detector_system_prompt=DETECTOR_SYSTEM_PROMPT.format(),
        neutralizer_system_prompt=NEUTRALIZER_SYSTEM_PROMPT.format(),
        critic_system_prompt=CRITIC_SYSTEM_PROMPT.format(),
        max_reasoning_steps=1,
    ):
        logger.info(
            f"Initializing Debiaser with providers: main={provider}, "
            f"detector={detector_provider}, neutralizer={neutralizer_provider}, "
            f"critic={critic_provider}"
        )

        # Initialize the main Debiaser agent, it doesn't matter system prompt
        # because we don't consume the Debiaser llm model directly
        super().__init__(
            provider,
            None,
            system=None,
        )
        self.max_reasoning_steps = max_reasoning_steps

        logger.info("Initializing sub-agents")
        self.detector = BiasDetector(
            provider=detector_provider,
            system=detector_system_prompt,
        )
        self.neutralizer = BiasNeutralizer(
            provider=neutralizer_provider,
            system=neutralizer_system_prompt,
        )
        self.critic = DebiaserCritic(
            provider=critic_provider,
            system=critic_system_prompt,
        )
        logger.info("All sub-agents initialized successfully")

    @weave.op(call_display_name="Debiaser agentic process")
    def execute_task(
        self,
        msgs: list[LLMMessage],
    ) -> AgentResponse:
        logger.info("Starting main debiasing process")
        msgs = msgs.copy()
        agent_response = AgentResponse(
            messages=msgs,
            agent_type=AgentType.DEBIASER,
        )

        # Step 1: Detect bias
        logger.info("Step 1: Detecting bias")
        detector_response = self.detector.execute_task(msgs)
        agent_response.llm_responses.extend(detector_response.llm_responses)
        agent_response.tool_activations.extend(detector_response.tool_activations)

        # If no bias detected, return early
        if detector_response.response == self.detector._UNBIASED_OUTPUT:
            logger.info("No bias detected, ending process")
            agent_response.response = self.detector._UNBIASED_OUTPUT
            return agent_response

        # Step 2: Neutralize bias
        logger.info("Step 2: Neutralizing detected bias")
        neutralizer_response = self.neutralizer.execute_task(
            msgs, detector_response.response
        )
        agent_response.llm_responses.extend(neutralizer_response.llm_responses)
        agent_response.tool_activations.extend(neutralizer_response.tool_activations)

        debiased_text = neutralizer_response.response["debiased_text"]
        reasoning = neutralizer_response.response["reasoning"]

        # Step 3: Criticism and refinement loop
        logger.info("Step 3: Starting criticism and refinement loop")
        msgs += [
            LLMMessage(
                role=LLMMessage.MessageRole.SYSTEM,
                content=CRITIC_SYSTEM_MSG.format(
                    debiased_text=debiased_text,
                    reasoning=",".join(reasoning),
                ),
            ),
            LLMMessage(
                role=LLMMessage.MessageRole.USER,
                content=CRITIC_USER_MSG.format(),
            ),
        ]
        agent_response.messages.extend(msgs[-2:])

        # Iterative refinement with critic
        for i in range(self.max_reasoning_steps):
            logger.info(f"Starting reasoning iteration {i + 1}/{self.max_reasoning_steps}")
            critic_response = self.critic.execute_task(agent_response.messages)
            agent_response.llm_responses.extend(critic_response.llm_responses)

            if not critic_response.response:
                logger.info("No critique provided, ending refinement loop")
                break
            elif critic_response.response == "SUCCESSFULLY_DEBIASED":
                logger.info("Text successfully debiased, ending refinement loop")
                break

            if i < self.max_reasoning_steps - 1:
                logger.info("Applying critique and requesting new neutralization")
                msgs += [
                    LLMMessage(
                        role=LLMMessage.MessageRole.SYSTEM,
                        content=critic_response.response,
                    ),
                    LLMMessage(
                        role=LLMMessage.MessageRole.USER,
                        content=CRITIC_USER_MSG_ITERATION.format(),
                    ),
                ]
                agent_response.messages.extend(msgs[-2:])

                neutralizer_response = self.neutralizer.execute_task(
                    msgs, detector_response.response
                )
                agent_response.llm_responses.extend(neutralizer_response.llm_responses)
                agent_response.tool_activations.extend(
                    neutralizer_response.tool_activations
                )

                debiased_text = neutralizer_response.response["debiased_text"]

        logger.info("Debiasing process completed")
        agent_response.response = debiased_text
        agent_response.messages.append(
            LLMMessage(
                role=LLMMessage.MessageRole.SYSTEM,
                content=DEBIASER_OUTPUT.format(debiased_text=debiased_text),
            )
        )
        return agent_response


class DebiaserModel(weave.Model):
    """
    Debiaser Model class to predict the output of the Debiaser Agent given an input message or a list

    The class is a W&B Weave wrapper around the Debiaser Agent that allows to track and log the predictions,
    plus the prompts and the reasoning steps during the debiasing process.
    """

    llm_provider: str
    max_reasoning_steps: int
    detector_system_prompt: str | weave.StringPrompt = DETECTOR_SYSTEM_PROMPT
    neutralizer_system_prompt: str | weave.StringPrompt = NEUTRALIZER_SYSTEM_PROMPT
    critic_system_prompt: str | weave.StringPrompt = CRITIC_SYSTEM_PROMPT

    @weave.op()
    def predict(
        self,
        input: str | LLMMessage | list[LLMMessage],
    ) -> AgentPrediction:
        """
        Predict the output of the Debiaser Agent given an input message or a list of messages
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

        # Initialize the Debiaser Agent, providing the necessary configuration
        # to each of the sub-agents, i.e. detector, neutralizer, critic
        client = Debiaser(
            provider=self.llm_provider,
            max_reasoning_steps=self.max_reasoning_steps,
            detector_system_prompt=self.detector_system_prompt.format(),
            neutralizer_system_prompt=self.neutralizer_system_prompt.format(),
            critic_system_prompt=self.critic_system_prompt.format(),
        )

        response = client.execute_task(input)

        # Create an AgentPrediction object
        prediction = AgentPrediction(
            input=input_str, output=client.detector._UNBIASED_OUTPUT
        )

        # Parse the debiaser response into the output dictionary
        for tool in response.tool_activations:
            if tool.tool_name == GENDER_BIAS_MULTI_LABEL_CLASSIFIER_TOOL.name:
                prediction.biases = ", ".join(
                    tool.tool_results.get("bias_labels", [""])
                )
                prediction.scores = ", ".join(
                    [str(score) for score in tool.tool_results.get("scores", [""])]
                )
            elif tool.tool_name == DEBIASER_TOOL.name:
                prediction.debias_reasoning = ", ".join(
                    tool.tool_results.get("reasoning", [""])
                )
                prediction.output = tool.tool_results.get("debiasing_text", None)
        return prediction
