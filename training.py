import json
import torch
import os
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from prompts.simple_prompt import get_simple_translation_prompt
from model_load import load_model, collate_batch
from jsonL import split_and_save

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Config-----------------------------------------------------------------------------
LORA_DIR = "D:\\office\\EBLICT\\Model training for MT\\outputs\\lora"
GGUF_DIR = "D:\\office\\EBLICT\\Model training for MT\\outputs\\gguf"
MAX_SEQ_LEN = 4096
BATCH_SIZE = 4
GRAD_ACCUM = 4
EPOCHS = 10
LR = 2e-4
GGUF_QUANT = "q4_k_m"   
 
 
# ── Formatting function -
def format_record(record):
    prompt = get_simple_translation_prompt(
        source_text=record["source"],
        source_language=record["source_language"],
        target_language=record["target_language"],
    )
    completion = json.dumps(
        {"translated_text": record["target"]}, ensure_ascii=False
    )
    return {"text": prompt + completion}
 
 
# ── Load JSONL
def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records).map(format_record)
 # ── Main Training Loop -
 
 def main():
     print("Preparing splits data...")
     split_and_save(INPUT_CSV, SPLITS_DIR)
     
     #load model and LoRA from model_load.py
     print("Loading model..........")
     model, tokenizer = load_model()
     
     #load dataset
     train_dataset = load_jsonl(os.path.join(SPLITS_DIR, "train.jsonl"))
     val_dataset = load_jsonl(os.path.join(SPLITS_DIR, "validation.jsonl"))
     
     #configure trainer
     trainer = SFTTrainer(
         model = model,
         tokenizer = tokenizer,
         train_dataset = train_dataset,
         eval_dataset = val_dataset,
         data_collator = lamda batch: collate_batch(batch, tokenizer),
         
         args = SFTConfig(
             dataset_text_field = "text",
             max_seq_length = MAX_SEQ_LEN,
             output_dir = LORA_DIR,
             num_train_epochs = EPOCHS,
             per_device_train_batch_size = BATCH_SIZE,
             gradient_accumulation_steps = GRAD_ACCUM,
            learning_rate = LR,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            warmup_ratio = 0.05,
            lr_scheduler_type = "cosine",
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            load_best_model_at_end = True,
            logging_steps = 20,
            report_to = "none",
            dataset_num_proc = 2,
             ),
     )     
     print("Starting training...")
     trainer.train()
     
 

