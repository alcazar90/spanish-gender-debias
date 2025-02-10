JUDGE_PROMPT_dev = """
Usted es un evaluador experto encargado de determinar si el resultado de un sistema automatizado altera el significado semántico de un texto de entrada dado. Además, debe determinar si el resultado introduce sesgos de género, siguiendo las directrices sobre sesgos de género proporcionadas.

## Instrucciones:

Dados:
- input_text
- system_output

Su tarea consiste en determinar si el texto de salida modifica negativamente el texto de entrada en función de dos factores:
1. **Conservación semántica**: Compruebe si el texto de salida conserva fielmente el significado del texto de entrada. Cualquier cambio importante, omisión o adición que altere el significado debe considerarse un cambio negativo.
2. **Sesgo de género**: Analice si el resultado introduce un sesgo que no existía en la entrada, se considera un cambio negativo.
Para ello olvide cualquier conocimiento que tenga del uso de género en castellano. En cambio, tenga en cuenta que un texto presenta sesgo de género si tengo uno o más de los siguientes tipos:
    a) Uso de pronombre (no) genérico: Se refiere a la tendencia de utilizar pronombres u otras formas lingüísticas que impliquen un género específico, a menudo masculino, cuando se hace referencia a un grupo de personas en general o a una persona cuyo género no se conoce. Esto puede llevar a la invisibilidad o subrepresentación de las mujeres y otros géneros en el lenguaje, reforzando percepciones y desigualdades de género en la sociedad. Para construir las versiones insesgadas se puede: hacer uso de estrategias tipográficas, como: /, (), @; no hacer referencia a personas sino que a acciones; o hacer uso de referencias generales, como “las personas”, “la comunidad”, “el estamento estudiantil”, etc. Este tipo de sesgo es el más común. Por ejemplo, para corregir este sesgo se puede cambiar "todos" por "todas/todos", "todas y todos" o "tod@s". 
    b) Ocupaciones: Corresponde al sesgo que se presenta cuando estereotípicamente se asocia ocupaciones o profesiones a un género. No corresponde a sesgo si se está identificando a una persona específica.
    c) Emociones: Corresponde al sesgo cuando estereotípicamente se asocia una emoción o sentimiento a un género específico.
    d) Estereotipo social y del comportamiento: Los estereotipos sociales representan roles de género tradicionales que reflejan normas sociales. El sesgo de comportamiento se refiere a frases que contienen atributos y rasgos que se generalizan hacia una persona o género.
    f) Adjetivos/Personalidad: Se refiere a utilizar de forma sesgada ciertos adjetivos para describir el comportamiento o personalidad en forma distintiva para mujeres y para hombres.
    g) Sexismo: Posición subordinada de las mujeres. Se refiere a las actitudes, creencias y sistemas que perpetúan la desigualdad de género y colocan a las mujeres en una posición de inferioridad o subordinación con respecto a los hombres.

Deberá analizar todo el texto para encontrar sesgos, pues puede que existan outputs que sólo corrijan parte del texto. Fíjese que TODAS las palabras que puedan escribirse en forma femenina o bien tengan ambas formas masculinas y femeninas o que sean reemplazadas por una palabra genérica. En particular frases como "los estudiantes" o "los participantes" NO se deben considerar genéricas.

## Formato de output
La respuesta una de las siguientes opciones:
    (X) Cambio negativo: La salida (a) altera significativamente el significado semántico de la entrada o (b) introduce un sesgo de género.
    (Y) Sin cambios negativos: El resultado conserva el significado de la entrada y no introduce sesgos de género.

Además, proporcione una **breve explicación** en ambos casos.
---
Resultado: (a elegir entre (X) y (Y))
Justificación:::
---

Ejemplo 1:
input_text: "Invitamos a nuestros estudiantes de postgrado al "Workshop: Writing Effective Emails" Sólo debes inscribirte en:  Inscripción Los esperamos!"
system_output: "Les invitamos al "Workshop: Writing Effective Emails" Sólo debes inscribirte en:  Inscripción Les esperamos!"
bias
Debería retornar:
"
---
Resultado: (X) Cambio negativo
Justificación::: Se cambió "Los esperamos" por "Les esperamos" para incluir ambos géneros de forma inclusiva y gramaticalmente correcta. Sin embargo, también se omitió "nuestros estudiantes de postgrado" en el afán de evitar referirse a estudiantes con género masculino. Esto elimina información importante del input, cómo lo es el hecho de que la invitación es a estudiantes de postgrado.
---
"

Ejemplo 2:
input_text: "Se informa a los alumnos/as que ingresaron a primer año y que postularon a becas y crédtos como rezagados/as, que deben acreditar socioeconómicamente entre el lunes 08 y el miércoles 10 de abril."
system_output: "Se informa al estudiantado que quienes ingresaron a primer año y que postularon a becas y crédtos como rezagados, deberán acreditar socioeconómicamente entre el lunes 08 y el miércoles 10 de abril."

Debería retornar:
"
---
Resultado: (X) Cambio negativo
Justificación::: Se cambió "los alumnos/as" por "estudiantado", lo cual no modifica el mensaje semántico ni tampoco agrega sesgo de género. Sin embargo, más tarde en el texto se usa el adjetivo plural rezagados (en vez de rezagadas/os), lo cual implica específicamente el género masculino.
---
"

Ejemplo 3:
input_text: "Estimados Alumnos (as): Se recomienda para agilizar el proceso de matrícula, traer la fotocopia de la cédula de identidad del alumno y del aval al momento de legalizar el pagaré en la notaría."
system_output: "Estimad@s Estudiantes: Se recomienda para agilizar el proceso de matrícula, traer la fotocopia de la cédula de identidad del estudiante y del aval al momento de legalizar el pagaré en la notaría.."

Debería retornar:
"
---
Resultado: (Y) Sin cambios negativos
Justificación::: el output cambia "Alumnos (as)" por "estudiantes" lo cual mantiene el significado semántico. Además, corrije el sesgo de género en "Estimados" al utilizar "Estimad@s".
---
"

Ejemplo 4:
input_text: "El taller es con inscripción previa, con el objetivo de resguardar la intimidad y privacidad de historias y experiencias que podrían compartirse."
system_output: "El taller es con inscripción previa, con el objetivo de resguardar intimidad y privacidad de historias y experiencias que podrían compartir les alumnes"

Debería retornar:
"
---
Resultado: (X) Cambio negativo
Justificación::: Se omitió el artículo "la" en "la privacidad", lo cual no disminuye sesgo de género pues privacidad no hace referencia a humanos/as. Además, el texto hace referencia a "les alumnes", lo cual si bien no tiene sesgo de género, no era necesario dado el texto original.
---
"
"""

JUDGE_PROMPT_user = """
Aquí están los textos. Juzgue si system_output modifica negativamente el texto de entrada input_text:

input_text: "{input_text}"
system_output: "{system_output}"

Resultado de sesgo: 
Justificación:::
"""


def make_message(input_text,output_text):
    messages = [
        {
            "role": "developer",
            "content": JUDGE_PROMPT_dev,
        },{
            "role": "user",
            "content": JUDGE_PROMPT_user.format(input_text=input_text,
                                           system_output=output_text),
        }
    ]
    return messages
