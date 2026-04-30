# from transformers import AutoTokenizer, AutoModelForCausalLM
import json

import torch
from unsloth import FastLanguageModel
from prompts.simple_prompt import get_simple_translation_prompt
from jsonL import split_and_save


def load_model(model_name: str = "unsloth/gemma-4-E2B-it"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Attach LoRA adapters for fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0.05,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 42,
   )
    return model, tokenizer

def collate_batch(batch,tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_texts = []
 
    for record in batch:
        prompt = get_simple_translation_prompt(
            source_text=record["source"],
            source_language=record["source_language"],
            target_language=record["target_language"]
        )
        
        completion = json.dumps({"translated_text": record["target"]}, ensure_ascii=False)

        # completion = f'{{"translated_text": "{record["target"]}"}}'
        full_texts.append(prompt + completion + tokenizer.eos_token)
 
    encodings = tokenizer(
        text = full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
 
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    labels = input_ids.clone()
    
    for i, record in enumerate(batch):
        prompt = get_simple_translation_prompt(
            source_text=record["source"],
            source_language=record["source_language"],
            target_language=record["target_language"]
        )
    
    labels[labels == tokenizer.pad_token_id] = -100
 
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

if __name__ == "__main__":
    model, tokenizer = load_model()
    print("Model loaded successfully!")
    print(f"Tokenizer vocab size: {tokenizer.tokenizer.vocab_size}")

    sample_batch = [
        {
            "source": "আমি স্কুলে যাই",
            "source_language": "Bangla",
            "target_language": "English",
            "target": "I go to school"
        }
    ]
    batch = collate_batch(sample_batch, tokenizer)

    print("Batch created successfully!")
    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("labels shape:", batch["labels"].shape)