import weave
from abc import ABC, abstractmethod
from datetime import datetime
from debiasing.llm.models import AntrophicCompletion
from debiasing.llm.models import ModelConfigs
from debiasing.llm.models import OpenAICompletion
from debiasing.llm.tools import DEBIASER
from debiasing.llm.tools import GENDER_BIAS_MULTI_LABEL_CLASSIFIER
from debiasing.llm.prompts import CRITIC_SYSTEM_PROMPT
from debiasing.llm.prompts import DEBIASER_SYSTEM_PROMPT
from debiasing.llm.prompts import DEBIASER_SYSTEM_MSG
from debiasing.llm.prompts import DEBIASER_USER_MSG
from debiasing.configs import logger
from debiasing.llm.utils import LLMMessage
from typing import Any
from pydantic import BaseModel, Field
from debiasing.configs import settings


# Output message when the text is considered unbiased and the debiasing process fails
UNBIASED_OUTPUT = "UNBIASED"
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
        model_config=ModelConfigs(max_tokens=settings.MAX_TOKENS, 
                                  temperature=settings.TEMPERATURE,),
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


class DebiaserCritic(Agent):
    def __init__(
        self,
        provider="anthropic",
        tools=None,
        system=CRITIC_SYSTEM_PROMPT.format(),
    ):
        super().__init__(provider, tools, system)

    def generate_critic_prompt(
        self,
        debiasing_text: str,
        reasoning: list[str],
    ) -> str:
        pass

    def execute_task(
        self,
        msgs,
    ) -> AgentResponse:
        msgs = msgs.copy()
        agent_response = AgentResponse(messages=msgs)

        # Get the response from the LLM
        text, _, response = self.llm.get_answer(msgs)
        agent_response.llm_responses.append(response)
        agent_response.response = text.text.strip()  # TextPart, no tool activated
        return agent_response


class Debiaser(Agent):
    def __init__(
        self,
        provider="anthropic",
        tools=[
            GENDER_BIAS_MULTI_LABEL_CLASSIFIER,
            DEBIASER,
        ],
        system=DEBIASER_SYSTEM_PROMPT.format(),
        max_reasoning_steps=3,  # Set max iterations for reasoning loop
    ):
        super().__init__(provider, tools, system)
        self._UNBIASED_OUTPUT = UNBIASED_OUTPUT
        self._DEBIASING_FAILURE_MSG = DEBIASING_FAILURE_MSG
        self._DEBIASING_SYSTEM_MSG = DEBIASER_SYSTEM_MSG
        self._DEBIASING_USER_MSG = DEBIASER_USER_MSG.format()
        self.max_reasoning_steps = max_reasoning_steps

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

            agent_response.messages.extend(msgs[-2:])
            logger.info(f"Current messages: {msgs}")

            # Now that we are in step 2, we need to force the tool to be activated, because we are expecting the debiaser tool to be activated
            # and make corrections to the text
            logger.info("Executing step 2 of the agent")
            logger.info(f"Current tools: {self.llm.tools} ")
            logger.info("Forcing tool activation by setting force_tool=True")
            text, tool, response = self.llm.get_answer(msgs, force_tool=True)
            agent_response.llm_responses.append(response)

            logger.info(f"LLM response before checking 1st debiaser tool activation: {response}")
            # Determine whether the debiaser tool was activated, i.e. step 2 of the agent execution, or control flow
            if tool and tool.name == DEBIASER.name:
                debiasing_text = tool.arguments["debiasing_text"]
                reasoning = tool.arguments["reasoning"]
                logger.info("'debiaser' tool activated")
                logger.info("Debiased text: " + debiasing_text)

                for r in reasoning:
                    logger.info(r)

                # Save the results of the debiasing process, i.e. the debiased text and the reasonin
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

                # Append a system message with the debiased text and the reasoning behind the debiasing
                # and a user message to ask for feedback on the debiasing process
                msgs += [
                    LLMMessage(
                        role=LLMMessage.MessageRole.SYSTEM,
                        content="Here is the debiased version: '"  + debiasing_text +  "' and the reasoning behind the debiasing: '" + ','.join(reasoning) + "'",
                    ),
                    LLMMessage(
                        role=LLMMessage.MessageRole.USER,
                        content="Do you have any feedback on the debiasing process?",
                    ),
                    # None if self.max_reasoning_steps == 0 else LLMMessage(
                    #     role=LLMMessage.MessageRole.USER,
                    #     content="Do you have any feedback on the debiasing process?",
                    # )
                ]

                agent_response.messages.extend(msgs[-2:])

                # A "reasoning loop" that works with another LLM about the reasons give by the tool
                # trying to iterate and ask for more information about the bias detected
                critic = DebiaserCritic(provider="anthropic")

                for i in range(self.max_reasoning_steps):
                    logger.info(f"Starting reasoning iteration {i + 1}")

                    critic_response = critic.execute_task(agent_response.messages)
                    agent_response.llm_responses.append(critic_response.llm_responses)

                    # Stop the loop if no new feedback is given
                    if not critic_response.response:
                        logger.info("No further feedback provided, ending iteration")
                        break
                    elif critic_response.response == "SUCCESSFULLY_DEBIASED":
                        logger.info("The debiasing process has been successfully completed")
                        break

                    logger.info(f"Critic feedback: {critic_response.response}")

                    # Modify the input for the next iteration
                    # feedback_msg = f"Here is my feedback to incorporate in the debiasing process: {critic_response.response}"

                    if i < self.max_reasoning_steps - 1:
                        msgs += [
                            LLMMessage(
                                role=LLMMessage.MessageRole.SYSTEM,
                                content=critic_response.response,
                            ),
                            LLMMessage(
                                role=LLMMessage.MessageRole.USER,
                                # NOTE: Se menciona que se use el feedback Y la herramienta debiaser tool para continuar el refinamiento
                                content="Use the debiaser tool and the feedback provide to refine the debiasing process and proposed a new debiased text",
                            ),
                        ]
                    elif i == self.max_reasoning_steps - 1:
                        # If the maximum number of reasoning steps is reached, end the debiasing process
                        msgs += [
                            LLMMessage(
                                role=LLMMessage.MessageRole.SYSTEM,
                                content=critic_response.response,
                            ),
                            LLMMessage(
                                role=LLMMessage.MessageRole.USER,
                                content="The maximum number of reasoning steps has been reached, ending the debiasing process",
                            ),
                        ]

                    agent_response.messages.extend(msgs[-2:])

                    text, tool, response = self.llm.get_answer(msgs, force_tool=True)
                    agent_response.llm_responses.append(response)

                    if tool and tool.name == DEBIASER.name:
                        debiasing_text = tool.arguments["debiasing_text"]
                        reasoning = tool.arguments["reasoning"]

                        agent_response.tool_activations.append(
                            ToolActivation(
                                tool_name=tool.name,
                                tool_results={
                                    "debiasing_text": debiasing_text,
                                    "reasoning": reasoning,
                                },
                                step_id=i + 3,
                            )
                        )
                    else:
                        logger.info(
                            "Debiaser tool failed to activate, stopping refinement"
                        )
                        break
                agent_response.response = debiasing_text
                agent_response.messages.append(
                    LLMMessage(
                        role=LLMMessage.MessageRole.SYSTEM,
                        content="The debiasing process has been completed. THe final debiased text is: '" + debiasing_text + "'",
                    )
                )
                return agent_response

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
