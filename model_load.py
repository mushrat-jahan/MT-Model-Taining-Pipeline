import json
import torch
import os
from unsloth import FastLanguageModel
from datasets import Dataset
from prompts.simple_prompt import get_simple_translation_prompt
from utils.jsonL import split_and_save

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_model(model_name: str = "unsloth/gemma-4-E2B-it"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = True,
    )
    
    tokenizer.chat_template_kwargs = {"enable_thinking": False}
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
 # Attach LoRA adapters for fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
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

        full_texts.append(prompt + completion + tokenizer.eos_token)
 
    encodings = tokenizer(
        text = full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
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

    #prompt length to mask label
        prompt_tokenized = tokenizer(
            text = prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=4096,
        )
        prompt_length = len(prompt_tokenized["input_ids"])
        labels[i, :prompt_length] = -100
    
    labels[labels == tokenizer.pad_token_id] = -100
 
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

