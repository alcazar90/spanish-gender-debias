import sys
sys.path.append('..')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model import init_model
import torch
from huggingface_hub import login
import utils as u
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Configure CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # You can restrict this to specific origins in production
    allow_methods=['*'],  # Allow all HTTP methods (including OPTIONS)
    allow_headers=['*'],
)

MODEL_ID = os.getenv("MODEL_ID")
MODEL_TYPE = os.getenv("MODEL_TYPE")
BIT = os.getenv("BIT")
HF_TOKEN = os.getenv("HF_TOKEN")
REVISION = os.getenv("REVISION")
MNT_FACTOR = float(os.getenv("MNT_FACTOR", "1.2"))

INPUT_COL =  {
    "SEQ_2_SEQ": "seq2seq_document",
    "CAUSAL": "causal_document"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

login(token=HF_TOKEN)

model, tokenizer = init_model(MODEL_ID, MODEL_TYPE, BIT, revision=REVISION)
generation_config = u.get_model_generation_config(model, MODEL_TYPE, tokenizer)

class TextData(BaseModel):
    text: str

@app.post('/debias')
async def receive_text(text_data: TextData):
    text = text_data.text
    model_input = (
        """
        <human>: ¿Puedes reescribir el siguiente texto sin sesgo de género?
        {}
        <assistant>:
        """.strip()
        if MODEL_TYPE == "CAUSAL" else
        "Eliminar sesgo de género del siguiente texto:\n{}"
    )

    raw_sentences = u.prepare_raw_documents(text)
    sentences = [
        model_input.format(s)
        for s in raw_sentences
    ]
    max_new_tokens_factor = MNT_FACTOR

    if len(sentences):
        last_n_sentences = -2
        unbias_sentences = sentences[last_n_sentences:]

        generations = [
            u.generate_output(
                model,
                model_input,
                tokenizer,
                generation_config,
                DEVICE,
                max_new_tokens_factor
            )
            for model_input in unbias_sentences
        ]

        generations = [
            u.get_processed_generation(MODEL_TYPE, generation)
            for generation in generations
        ]

        replace_sentences = raw_sentences[:last_n_sentences] + generations
        print(generations, replace_sentences)
        try:
            positions, substitutions = list(zip(*[
                u.get_positions_and_substitutions(raw_sentences[i], replace_sentences[i])
                for i in range(len(sentences))
            ]))

            positions = list(positions)

            for i in range(len(positions)):
                if i:
                    prev_words = " ".join(raw_sentences[0:i]).split(" ")
                    positions[i] = [
                        j + len(prev_words)
                        for j in positions[i]
                    ]

            return {
                'input': " ".join(raw_sentences[last_n_sentences:]),
                "generated": " ".join(generations),
                "diff": {
                    "input_positions": [i for p in positions for i in p],
                    "substitutions": [i for s in substitutions for i in s],
                }
            }

        except Exception as e:
            print(f"ERROR: {e}.\nGeneration: {generation}.\nInput: {text}")
            raise e

    else:
        raise HTTPException(status_code=400, detail='Text not provided in request')
