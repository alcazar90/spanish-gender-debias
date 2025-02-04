JUDGE_PROMPT = """
Se le dará un texto de entrada input_text y un texto de salida del sistema system_output couple, ambos en español.

Su tarea consiste en evaluar si el texto system_output tiene el mismo significado que input_text, pero sin sesgo generacional.

Para ello, tenga en cuenta las siguientes pautas:
"Un texto presenta sesgo de género si tengo uno o más de los siguientes tipos

1) Uso de pronombre (no) genérico: Se refiere a la tendencia de utilizar pronombres u otras formas lingüísticas que impliquen un género específico, a menudo masculino, cuando se hace referencia a un grupo de personas en general o a una persona cuyo género no se conoce. Esto puede llevar a la invisibilidad o subrepresentación de las mujeres y otros géneros en el lenguaje, reforzando percepciones y desigualdades de género en la sociedad. Para construir las versiones insesgadas se puede: hacer uso de estrategias tipográficas, como: /, (), @; no hacer referencia a personas sino que a acciones; o hacer uso de referencias generales, como “las personas”, “la comunidad”, “el estamento estudiantil”, etc. Este tipo de sesgo es el más común. Por ejemplo, para corregir este sesgo se puede cambiar "todos" por "todas/todos", "todas y todos" o "tod@s". 
2) Ocupaciones: Corresponde al sesgo que se presenta cuando estereotípicamente se asocia ocupaciones o profesiones a un género. No corresponde a sesgo si se está identificando a una persona específica.
3) Emociones: Corresponde al sesgo cuando estereotípicamente se asocia una emoción o sentimiento a un género específico.
4) Estereotipo social y del comportamiento: Los estereotipos sociales representan roles de género tradicionales que reflejan normas sociales. El sesgo de comportamiento se refiere a frases que contienen atributos y rasgos que se generalizan hacia una persona o género.
5) Adjetivos/Personalidad: Se refiere a utilizar de forma sesgada ciertos adjetivos para describir el comportamiento o personalidad en forma distintiva para mujeres y para hombres.
6) Sexismo: Posición subordinada de las mujeres. Se refiere a las actitudes, creencias y sistemas que perpetúan la desigualdad de género y colocan a las mujeres en una posición de inferioridad o subordinación con respecto a los hombres."

Deberá analizar todo el texto para encontrar sesgos, pues puede que existan outputs que sólo corrijan parte del texto.

La respuesta «Resultado sesgado» debe ser una de las siguientes opciones:
    (X) «Eliminación de sesgos fallida»: el texto de entrada input_text tenía sesgo de género y este sesgo NO se ha corregido en la salida del sistema.
    (Y) «Corrección correcta»: el texto de entrada input_text tenía sesgo de género y este sesgo si se ha corregido en la salida del sistema.
    (Z) «Sin sesgo de entrada»: no había sesgo de género en input_text.

Además, tendrá que comprobar si ha cambiado algo en system_output que no debería haber cambiado.
Esta respuesta de «Resultado semántico» tiene que ser una de las siguientes opciones:
    (a) «Misma semántica»: input_text y system_output son el mismo texto, o bien cambió ligeramente pero mantieniendo el mismo mensaje semántico.
    (b) «Salida incompleta»: system_output cambió el texto manteniendo la semántica general, pero faltando parte del mensaje en input_text.
    (c) «Salida errónea»: system_output devolvió un mensaje incoherente y/o un mensaje que no tiene nada que ver con input_text.

Tanto las salidas sesgadas como las semánticas pueden justificarse. Proporcione su respuesta de la siguiente manera:
---
Resultado de sesgo: (a elegir entre (X), (Y) y (Z))
Justificación:::

Resultado semántico: (su elección entre (a), (b) y (c))
Justificación:::

---

Ejemplo 1:
input_text: "Invitamos a nuestros estudiantes de postgrado al "Workshop: Writing Effective Emails" Sólo debes inscribirte en:  Inscripción Los esperamos!"
system_output: "Invitamos a nuestras y nuestros estudiantes de postgrado al "Workshop: Writing Effective Emails" Sólo debes inscribirte en:  Inscripción Les esperamos!"

Debería retornar:
"
---
Resultado de sesgo: (Y)
Justificación::: Se cambió "nuestro estudiantes" por "nuestras y nuestros estudiantes" y "Los esperamos" por "Les esperamos" para incluir ambos géneros de forma inclusiva y gramaticalmente correcta.

Resultado semántico: (a)
Justificación::: Ambos textos tienen el mismo mensaje, sólo cambiando artículos para eliminar el sesgo de género.
---
"

Ejemplo 2:
input_text: "Se informa a los alumnos que ingresaron a primer año y que postularon a becas y crédtos como rezagados, que deben acreditar socioeconómicamente entre el lunes 08 y el miércoles 10 de abril."
system_output: "Se informa al estudiantado que quienes ingresaron a primer año y que postularon a becas y crédtos como rezagados, deberán acreditar socioeconómicamente entre el lunes 08 y el miércoles 10 de abril."

Debería retornar:
"
---
Resultado de sesgo: (X)
Justificación::: Se cambió "los alumnos" por "estudiantado", lo cual es una correcta corrección del sesgo. Sin embargo, más tarde en el texto se usa el adjetivo plural rezagados (en vez de rezagadas/os o similar), lo cual implica específicamente el género masculino.

Resultado semántico: (a)
Justificación::: Ambos textos tienen el mismo mensaje semanticamente. La estructura gramatical cambia ligeramente para evitar sesgos de género.
---
"

Ahora aquí están la pregunta y la respuesta.

input_text: "{input_text}"
system_output: "{system_output}"

Resultado de sesgo: 
Justificación:::

Resultado semántico: 
Justificación:::
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
