from transformers import AutoTokenizer, AutoModelForCausaiLM
from unsloth import FastLanguageModel

def load_model(model_name: str = "unsloth/gemma-4-E2B-it"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_sequence_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
