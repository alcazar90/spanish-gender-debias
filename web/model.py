import sys
sys.path.append('..')

from transformers import (
    AutoTokenizer,
)

from peft import (
    PeftConfig,
    PeftModel
)

import utils as u

def init_model(model_id, model_type, bit, revision="main"):
    config = PeftConfig.from_pretrained(model_id, revision=revision)

    # Model
    model = u.get_gender_model(config, model_type, bit)
    model = PeftModel.from_pretrained(model, model_id, device_map="auto", revision=revision)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path,
        use_fast=model_type == "CAUSAL"
    )
    _ = tokenizer.add_tokens(new_tokens = u.new_tokens[model_type])
    model.resize_token_embeddings(len(tokenizer))

    if model_type == "CAUSAL":
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("Peft model loaded")
    return model, tokenizer