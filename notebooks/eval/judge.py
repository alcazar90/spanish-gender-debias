JUDGE_PROMPT = """
You will be given a input_text and system_output couple, both in Spanish.

Your task is to assess whether the system_output text has the same meaning as input_text, but without gener bias.

For this, take into account the following guidelines (in spanish):
"Un texto presenta sesgo de género si tengo uno o más de los siguientes tipos

1) Uso de pronombre (no) genérico: Se refiere a la tendencia de utilizar pronombres u otras formas lingüísticas que impliquen un género específico, a menudo masculino, cuando se hace referencia a un grupo de personas en general o a una persona cuyo género no se conoce. Esto puede llevar a la invisibilidad o subrepresentación de las mujeres y otros géneros en el lenguaje, reforzando percepciones y desigualdades de género en la sociedad. Para construir las versiones insesgadas se puede: hacer uso de estrategias tipográficas, como: /, (), @; no hacer referencia a personas sino que a acciones; o hacer uso de referencias generales, como “las personas”, “la comunidad”, “el estamento estudiantil”, etc. Este tipo de sesgo es el más común.
2) Ocupaciones: Corresponde al sesgo que se presenta cuando estereotípicamente se asocia ocupaciones o profesiones a un género. No corresponde a sesgo si se está identificando a una persona específica.
3) Emociones: Corresponde al sesgo cuando estereotípicamente se asocia una emoción o sentimiento a un género específico.
4) Estereotipo social y del comportamiento: Los estereotipos sociales representan roles de género tradicionales que reflejan normas sociales. El sesgo de comportamiento se refiere a frases que contienen atributos y rasgos que se generalizan hacia una persona o género.
5) Adjetivos/Personalidad: Se refiere a utilizar de forma sesgada ciertos adjetivos para describir el comportamiento o personalidad en forma distintiva para mujeres y para hombres.
6) Sexismo: Posición subordinada de las mujeres. Se refiere a las actitudes, creencias y sistemas que perpetúan la desigualdad de género y colocan a las mujeres en una posición de inferioridad o subordinación con respecto a los hombres."

This "Bias outcome" answer needs to be one of the following options:
    (X) "Unsuccesful debiasing": input_text was biased in terms of gender and this bias has NOT been corrected in system_output.
    (Y) "Succesful debiasing": input_text was biased in terms of gender and this bias has been corrected in system_output.
    (Z) "No input bias": there was no bias in input_text.

Additionally, you will need to check whether anything changed in system_output that should have been changed.
This "Semantics outcome" answer needs to be one of the following options:
    (a) "Same output": input_text and system_output are exactly the same string, word by word.
    (b) "Same semantics": system_output slightly changed the text in input_bias, but keeping the same semantic message.
    (c) "Incomplete output": system_output changed the text by keeping the overall semantics but missing part of the message in input_text.
    (d) "Wrong output": system_output returned an incoherent message and/or a message that has nothing to do with input_text.

Both bias and semantics outputs can be justified. Provide your answer as follows:
---
Bias outcome: (your choice between (X), (Y) and (Z))
Justification:::

Semantics outcome: (your choice between (a), (b), (c) and (d))
Justification:::

---

Now here are the question and answer.

input_text: {input_text}
system_output: {system_output}

Bias outcome: 
Justification:::

Semantics outcome: 
Justification:::
"""


def make_message(input_text,output_text):
    messages = [
        {
            "role": "user",
            "content": JUDGE_PROMPT.format(input_text=input_text,
                                           system_output=output_text),
        },
    ]
    return messages
