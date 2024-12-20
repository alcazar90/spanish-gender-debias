from enum import StrEnum
from pydantic import BaseModel, Field, validator
from debiasing.llm.utils import LLMToolDefinition


class GenderBiasesEnum(StrEnum):
    """Kind of gender biases to detect in the text
    Ref: Based on Table 2 from work https://arxiv.org/pdf/2201.08675 
    """
    GENERIC_PRONOUNS = "GENERIC_PRONOUNS"
    STEREOTYPING_BIAS = "STEREOTYPING_BIAS"
    SEXISM = "SEXISM"
    EXCLUSIONARY_TERMS = "EXCLUSIONARY_TERMS"
    SEMANTIC_BIAS = "SEMANTIC_BIAS"

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
            GenderBiasesEnum.SEMANTIC_BIAS: "Semantic bias is the use of words or phrases that have a gendered connotation, which can reinforce stereotypes or biases."
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
                "Chariman", "Businessman", 
                # "Policeman", "Cameraman", 
                # "Man-bread", "Man-sip",
            ],
            GenderBiasesEnum.SEMANTIC_BIAS: [
                "'Cookie': lovely woman.",
                "A woman's tongue three inches long can kill a man six feet high.",
            ]
        }
        return examples[self]


class GenderBiasClassifier(BaseModel, 
                           extra="forbid",
):
    """Multi-label gender bias classifier tool. Identify possible gender biases within a text."""

    bias_label: str = Field(
        description=(
            f"The list of biases labels (if any) detected in the text."
        ),
        title="bias_label",
        enum=[bias.name for bias in GenderBiasesEnum],
    )

    # @validator("bias_label", each_item=True)
    # def validate_bias_label(cls, v):
    #     if v not in GenderBiasesEnum.__members__:
    #         raise ValueError(f"{v} is not a valid bias label")
    #     return v

    model_config = {
        "json_schema_extra": {
            "required": [
                "bias_label",
            ],
        # NOTE: additionalProperties required for using OpenAI structured outputs,
        # https://platform.openai.com/docs/guides/structured-outputs/additionalproperties-false-must-always-be-set-in-objects
         "additionalProperties": False,
        }
    }

# Define the tool description leveraging the GenderBiasesEnum class to provide the description and examples
GENDER_BIAS_CLASSIFIER_DESCRIPTION = "Identify (if any) one ore more of the following gender biases in the text:\n"
GENDER_BIAS_CLASSIFIER_DESCRIPTION += "\n\n".join([
    f"{gender.name}: {gender.description}\n -> Examples: {';'.join(gender.examples)}" 
    for gender in GenderBiasesEnum
])

GENDER_BIAS_CLASSIFIER = LLMToolDefinition(name="gender_bias_classifier",
                                           description="Identify gender biases in text (if any)",
                                           inputSchema=GenderBiasClassifier.model_json_schema(),
                                           )
