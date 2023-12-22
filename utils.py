import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)

import nltk
from nltk.tokenize import sent_tokenize
import difflib

new_tokens = {
    "SEQ_2_SEQ": ['í', 'ñ', 'ú', '¡', 'Í', '¿', 'Á', 'Ó', 'Ú', 'Ñ'],
    "CAUSAL": []
}

def preprocess_function(examples, tokenizer, padding="max_length",
                        max_source_length=512, max_target_length=512,
                       input_label="input", target_label="target"):


    model_inputs = tokenizer(examples[input_label],
                             max_length=max_source_length,
                             padding=padding,
                             truncation=True)

    labels = tokenizer(text_target=examples[target_label],
                       max_length=max_target_length,
                       padding=padding,
                       truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels
    # by -100 when we want to ignore padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [
                (l if l != tokenizer.pad_token_id else -100)
                for l in label
            ]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def prepare_raw_documents(document):
  sentences = None

  try:
    sentences = [
      f.replace("\n", "").replace("\r", "").strip()
      for f in sent_tokenize(document)
    ]

  except Exception as e:
    print(e, document)

  return sentences

def get_seq2seq_4bit_model(config):
    return AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )

def get_causal_4bit_model(config):
    return AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        return_dict=True,
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        trust_remote_code=True,
    )

def get_gender_model(config, lm_type, bit, ):
    model_types = {
        "SEQ_2_SEQ_4bit": get_seq2seq_4bit_model,
        "CAUSAL_4bit": get_causal_4bit_model
    }

    return model_types[f"{lm_type}_{bit}"](config)

def get_model_generation_config(model, lm_type, tokenizer=None):
    generation_config = model.generation_config

    if lm_type == "SEQ_2_SEQ":
        generation_config.max_new_tokens = 1000
        generation_config.num_return_sequences = 1
        generation_config.do_sample = False

    elif lm_type == "CAUSAL":
        generation_config.num_return_sequences = 1
        generation_config.do_sample = False
        generation_config.pad_token_id = tokenizer.eos_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id

    return generation_config
        
def generate_output(model, input, tokenizer, generation_config, device, max_new_tokens_factor=None):

    if max_new_tokens_factor is not None:
        input_parts = input.split("\n")
        prompt_len = len(
            tokenizer(
                " ".join(input_parts[1:-1] if "<assistant>" in input else input_parts[1:]),
                return_tensors="pt"
            ).to(device).input_ids[0]
        )
        generation_config.max_new_tokens = int(prompt_len * max_new_tokens_factor)
    
    with torch.inference_mode():
        encoding = tokenizer(input, return_tensors="pt").to(device)      
        outputs = model.generate(
            input_ids = encoding["input_ids"],
            attention_mask = encoding["attention_mask"],
            generation_config = generation_config,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True, spaces_between_special_tokens=False)

def get_rouge_f_mean(scorer, target, generation):
    scores = scorer.score(target, generation)
    return np.mean([
        score_tuple[2] # fmeasure
        for score_tuple in scores.values()
    ])

def get_positions_and_substitutions(input_text, output_text):
    output_text = output_text[:-1] if output_text[-1] == "." and input_text[-1] != "." else output_text
    input_words = input_text.split()
    output_words = output_text.split()

    matcher = difflib.SequenceMatcher(None, input_words, output_words)
    diff = list(matcher.get_opcodes())

    positions = []
    substitutions = []
    
    for opcode in diff:
        if opcode[0] in ['insert', 'replace']:
            positions.append(
                opcode[1] if opcode[4] - opcode[3] == 1 else [opcode[3], opcode[4]]
            )
            substitutions.append(
                ' '.join(
                    output_words[opcode[3]:opcode[4]]
                ) + (" <<INSERT>>" if opcode[0] == "insert" else "")
            )

        elif opcode[0] == 'delete':
            positions.append(opcode[1])
            substitutions.append('<<REMOVE>>')

    for i, p in enumerate(positions):
        if "<<INSERT>>" in substitutions[i]:
            if p < len(input_words):
                substitutions[i] = f"{substitutions[i].replace(' <<INSERT>>', '')} {input_words[p]}"
            else:
                substitutions[i] = substitutions[i].replace(' <<INSERT>>', '')
    

    if len(substitutions):
        last_pos = positions[-1] if isinstance(positions[-1], int) else positions[-1][-1]
        
        substitutions[-1] = (
            substitutions[-1] + " "
            if substitutions[-1][-1] != " "
            and input_words[last_pos if len(input_words) < last_pos else -1][-1] == " "
            else substitutions[-1]
        )

    assert len(substitutions) == len(positions)

    return positions, substitutions

def get_processed_generation(model_type, generation):

    if model_type == "CAUSAL":
        output_proc = [
            o
            for o in list(set(
                generation.split("<assistant>: ")[1:]
            ))
            if o
        ]

        max_output_proc = max(output_proc, key=len)
        split_lines = max_output_proc.split("\n")
        tmp = [
            l
            for l in split_lines
            if set(l) - set("<assistant>: ")
        ]

        if len(tmp):
            generation = "\n".join(tmp)

        else:
            generation = max_output_proc

    return generation