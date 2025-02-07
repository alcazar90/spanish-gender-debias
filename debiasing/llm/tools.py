import os
import weave
from enum import StrEnum
from pydantic import BaseModel, Field
from debiasing.llm.utils import LLMToolDefinition

from debiasing.configs import settings
from debiasing.configs import logger

client = weave.init(settings.WANDB_PROJECT)

# Get the directory of the current script
DEBIASING_PROMPT_EXAMPLES = "debiasing_example.txt"
CLASSIFIER_PROMPT_EXAMPLES = "biases_example.txt"

script_dir = os.path.dirname(__file__)
classifier_file_path = os.path.join(script_dir, CLASSIFIER_PROMPT_EXAMPLES)
debiasing_file_path = os.path.join(script_dir, DEBIASING_PROMPT_EXAMPLES)

# Read the classifier example file
try:
    with open(classifier_file_path, "r", encoding="utf-8") as f:
        _BIASES_PROMPT_EXAMPLES = f.read()
        logger.info(f"File {CLASSIFIER_PROMPT_EXAMPLES} read successfully.")
except FileNotFoundError:
    print(f"The file {CLASSIFIER_PROMPT_EXAMPLES} was not found.")
except IOError as e:
    print(f"An I/O error occurred: {e}")

# Read the debiasing example file
try:
    with open(debiasing_file_path, "r", encoding="utf-8") as f:
        _DEBIASER_PROMPT_EXAMPLES = f.read()
        logger.info(f"File {DEBIASING_PROMPT_EXAMPLES} read successfully.")
except FileNotFoundError:
    print(f"The file {DEBIASING_PROMPT_EXAMPLES} was not found.")
except IOError as e:
    print(f"An I/O error occurred: {e}")

# Define a simple calculator tool using BaseModel
# This is a dummy calculator tool for showing how to define a tool and for testing purposes
class Calculator(BaseModel):
    """A simple calculator tool"""

    operation: str
    number1: int
    number2: int

    model_config = {
        "json_schema_extra": {
            "required": [
                "operation",
                "number1",
                "number2",
            ],
            "additionalProperties": False,
        }
    }


CALCULATOR_DESCRIPTION = "A simple calculator tool that performs basic arithmetic operations and can be used to identify the result of the operation in a user request."
CALCULATOR = LLMToolDefinition(
    name="calculator",
    description=CALCULATOR_DESCRIPTION,
    inputSchema=Calculator.model_json_schema(),
    structured_output=False,
)


# Define a GenderBiasesEnum class with the different kind of biases to detect in the text, providing
# a description and examples for each one...
# Best tool practices: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#best-practices-for-tool-definitions
# TODO: Improve gender descriptions and examples provides (in spanish (?)) ask for support to Darinka
class GenderBiasesEnum(StrEnum):
    """Kind of gender biases to detect in the text
    Ref: Based on Table 2 from work https://arxiv.org/pdf/2201.08675
    """

    GENERIC_PRONOUNS = "GENERIC_PRONOUNS"
    STEREOTYPING_BIAS = "STEREOTYPING_BIAS"
    SEXISM = "SEXISM"
    EXCLUSIONARY_TERMS = "EXCLUSIONARY_TERMS"
    SEMANTIC_BIAS = "SEMANTIC_BIAS"
    UNBIASED = "UNBIASED"

    @property
    def description(self) -> str:
        """Return the description of a given gender bias.
        Sexism description is taken from the work: https://arxiv.org/pdf/2112.14168 (page 5)
        """
        descriptions = {
            GenderBiasesEnum.GENERIC_PRONOUNS: "Generic pronouns bias is the use of gender-specific pronouns when a gender-neutral pronoun would be more appropriate.",
            GenderBiasesEnum.STEREOTYPING_BIAS: "Stereotyping bias is the use of stereotypes to make assumptions about a person's abilities, interests, or characteristics.",
            GenderBiasesEnum.SEXISM: "Sexism can be defined as discrimination, stereotyping, or prejudice based on one's sex.",
            GenderBiasesEnum.EXCLUSIONARY_TERMS: "Exclusionary terms bias is the use of terms that exclude or marginalize a particular gender, often by using male-oriented terms as the default.",
            GenderBiasesEnum.SEMANTIC_BIAS: "Semantic bias is the use of words or phrases that have a gendered connotation, which can reinforce stereotypes or biases.",
            GenderBiasesEnum.UNBIASED: "No gender bias detected",
        }
        return descriptions[self]

    @property
    def examples(self) -> list[str]:
        """Return examples of a given gender bias"""
        # TODO: Continue adding examples in the list, incorporate in spanish
        examples = {
            GenderBiasesEnum.GENERIC_PRONOUNS: [
                "A programmer must carry his laptop with him to work.",
                "A nurse should ensure that she gests adqueate rest.",
            ],
            GenderBiasesEnum.STEREOTYPING_BIAS: [
                "Senators need theirs wives to support them throughout their campagin.",
                # "The event was kid-friendly for all the mothers working in the company.",
                "All boys are aggresive.",
                # "Mary must love dolls because all girls like playing with them.",
            ],
            GenderBiasesEnum.SEXISM: [
                "Women are incompetent at work.",
                "They're probably surprised at how smart you are, for a girl.",
            ],
            GenderBiasesEnum.EXCLUSIONARY_TERMS: [
                "Chariman",
                "Businessman",
                # "Policeman", "Cameraman",
                # "Man-bread", "Man-sip",
            ],
            GenderBiasesEnum.SEMANTIC_BIAS: [
                "'Cookie': lovely woman.",
                "A woman's tongue three inches long can kill a man six feet high.",
            ],
            GenderBiasesEnum.UNBIASED: [
                "The programmer must carry the laptop to work."
            ],
        }
        return examples[self]


# # Define the tool description leveraging the GenderBiasesEnum class to provide the description and examples
GENDER_BIAS_CLASSIFIER_DESCRIPTION = (
    "Identify (if any) one ore more of the following gender biases in the text:\n"
)
# GENDER_BIAS_CLASSIFIER_DESCRIPTION += "\n\n".join(
#     [
#         f"{gender.name}: {gender.description}\n -> Examples: {';'.join(gender.examples)}"
#         for gender in GenderBiasesEnum
#     ]
# )

GENDER_BIAS_CLASSIFIER_DESCRIPTION += "\n\n".join(
    [f"{gender.name}: {gender.description}\n" for gender in GenderBiasesEnum]
)

GENDER_BIAS_CLASSIFIER_DESCRIPTION += "Don't return None if no bias is detected."

# Provide examples of biases to detect in the text
GENDER_BIAS_CLASSIFIER_DESCRIPTION += _BIASES_PROMPT_EXAMPLES
GENDER_BIAS_CLASSIFIER_DESCRIPTION += "Remember the response should include the bias_label, the bias_text and the score_label for each bias detected. Otherwise, no bias detected, return UNBIASED or don't activate the tool"

GENDER_BIAS_CLASSIFIER_DESCRIPTION = weave.StringPrompt(
    GENDER_BIAS_CLASSIFIER_DESCRIPTION
)
weave.publish(
    GENDER_BIAS_CLASSIFIER_DESCRIPTION, name="GENDER_BIAS_CLASSIFIER_DESCRIPTION"
)


# This kind of tool is a structured output, i.e. complete a json schema for the output, in this case, the output is the bias_label and the bias_text
# Therefore, our multi label classification is to detect specific labels (define in GenderBiasesEnum) in the text and specify the specific part of the text that contains
# Ref: Using structured outputs with function calling, see the following references:
# https://openai.com/index/introducing-structured-outputs-in-the-api/
# https://platform.openai.com/docs/guides/function-calling#turn-on-structured-outputs-by-setting-strict-true
class MultiLabelGenderBiasClassifier(
    BaseModel,
    extra="forbid",
):
    """
    A multi-label gender bias classifier tool which perform some kind of named entity recognition identifying possible gender biases within a text.

    Ref: https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/extracting_structured_json.ipynb
    """

    bias_label: list[GenderBiasesEnum] = Field(
        description="List of biases labels detected within the text.",
        title="bias_label",
    )

    bias_text: list[str] = Field(
        description="A list with the specific parts of the text that trigger a gender bias detection specified in bias_label. If no bias is detected, the text is considered unbiased an use None.",
        title="bias_text",
    )

    score_label: list[float] = Field(
        description="A list with the classification score of the bias_label detected in the text, ranging from 0.0 to 1.0.",
        title="score_label",
    )

    model_config = {
        "json_schema_extra": {
            "required": [
                "bias_label",
                "bias_text",
                "score_label",
            ],
            "examples": [
                # Example with multiple bias types
                {
                    "bias_label": ["GENERIC_PRONOUNS", "SEXISM"],
                    "bias_text": [
                        "A programmer must carry his laptop.",
                        "A programmer must carry his laptop with him to work.",
                    ],
                    "score_label": [0.9, 1.0],
                },
                # Example with no bias detected
                {
                    "bias_label": ["UNBIASED"],
                    "bias_text": [
                        "The programmer works diligently.",
                    ],
                    "score_label": [1.0],
                },
            ],
            # NOTE: additionalProperties required for using OpenAI structured outputs,
            # https://platform.openai.com/docs/guides/structured-outputs/additionalproperties-false-must-always-be-set-in-objects
            "additionalProperties": False,
        }
    }


# Now we will define Multi-label gender bias classifier tool using LLMToolDefinition
# TODO: Improve the description of the tool and use GENDER_BIAS_CLASSIFIER_DESCRIPTION, currently
# the description is passing as a system prompt to the model (see debiasing-doc.ipynb). See the documentation
# and best practices in OpenAI and Anthropic for defining the description of the tool, as well as the number of
# characters allowed for the description.
GENDER_BIAS_MULTI_LABEL_CLASSIFIER = LLMToolDefinition(
    name="gender_bias_classifier",
    description=GENDER_BIAS_CLASSIFIER_DESCRIPTION.format(),
    # description="Identify gender biases in a given text (if any) and for each bias specify in which specific part of the text is located. Therefore, there is a correspondence between the bias_label and the bias_text.",
    inputSchema=MultiLabelGenderBiasClassifier.model_json_schema(),
    structured_output=True,
)


# Define a tool for a kind of sequence-to-sequence transfer style for biased-to-debiased gender text...
class DebiasingText(
    BaseModel,
    extra="forbid",
):
    """A tool for debiasing a given text"""

    original_text: str = Field(
        description="The original text containing the bias or biases to be neutralized.",
        title="Original Text",
        example="A programmer must carry his laptop to work.",
    )

    debiasing_text: str = Field(
        description="The neutralized version of the original text, with all identified gender biases removed. This version maintains the original semantic meaning, communication intent, and context, but uses gender-inclusive language. It represents the final, bias-free text output.",
        # description="The neutralized version of the text with gender biases removed while maintaining original context and meaning.",
        title="Debiased Text",
        example="A programmer must carry their laptop to work.",
    )
    reasoning: list[str] = Field(
        description="A detailed linguistic breakdown explaining the specific debiasing decisions made. Each entry describes: 1) The original biased text segment, 2) The detected bias type, 3) The rationale for the modification, 4) The specific inclusive language strategy used. This provides transparency into how and why the text was neutralized.",
        # description="Detailed explanation of each debiasing decision, including linguistic and cultural considerations.",
        # title="reasoning",
        title="Debiasing Reasoning",
        example=[
            "Original phrase 'A programmer must carry his laptop' contained a Generic Pronouns bias. Replaced gender-specific 'his' with gender-neutral 'their' to ensure inclusive language while preserving the original meaning."
        ],
    )

    model_config = {
        "json_schema_extra": {
            "required": [
                "original_text",
                "debiasing_text",
                "reasoning",
            ],
            "additionalProperties": False,
        }
    }


# DEBIASING_TEXT_DESCRIPTION = weave.StringPrompt("""
# A sophisticatrd text analysis and modification tool designed to detect and neutralize gender biases in text. The tool performs a precise linguistic intervention, identifying specific gender bias types and transforming the text to use inclusive language while meticulously preserving the original semantic meaning and communicative intent.
# """)

DEBIASING_TEXT_DESCRIPTION = weave.StringPrompt("""
A sophisticated text analysis and modification tool designed to detect and neutralize gender biases in text. 
The tool performs a precise linguistic intervention, identifying specific gender bias types and transforming the text to use inclusive language while meticulously preserving the original semantic meaning and communicative intent in spanish.
The response should include:
1. The original biased text.
2. The neutralized version of the text, with all biases removed.
3. A detailed explanation of how the debiasing was applied, including the rationale and specific linguistic strategies used.
Ensure that all three pieces of information are included in the response.

{debiasing_example}
                                                
Provide the debiased version and a detailed explanation of the debiasing process, including the rationale and specific linguistic strategies used. Ensure that the original meaning and context of the text are preserved while making minimal changes to the wording.
""".format(debiasing_example=_DEBIASER_PROMPT_EXAMPLES))


weave.publish(DEBIASING_TEXT_DESCRIPTION, name="DEBIASING_TEXT_DESCRIPTION")

# A text analysis and modification tool desgiend for debiasing a text that contain grnder biases. The fragment of the text will be given with the specific gender bias detected, your work is neutralize the biases applying minimal modifications and present a debiased text version while preserving the original meaning and context.

DEBIASER = LLMToolDefinition(
    name="debiaser",
    description=DEBIASING_TEXT_DESCRIPTION.format(),
    inputSchema=DebiasingText.model_json_schema(),
    structured_output=True,
)
