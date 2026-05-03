import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
import torch
from unsloth import FastLanguageModel
from prompts.simple_prompt import get_simple_translation_prompt

#data_loader 

loader = DataLoader("D:\\office\\EBLICT\\Model training for MT\\Dataset\\input5k.csv")
print(loader.get_cleaned_data().head())

loader.save_cleaned_data("D:\\office\\EBLICT\\Model training for MT\\Dataset\\processed_data.csv", index=False)

#data preprocessing
class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        self._data_load_and_clean()

    def _data_load_and_clean(self):
        # load raw csv file
        df = pd.read_csv(self.filepath)

        # Fill empty values in final_text_version using text_data
        df["final_text_data"] = df["final_text_data"].replace("", pd.NA)
        df["source_text"] = df["final_text_data"].fillna(df["text_data"])

        # keep desired columns
        desired_cols = ["source_text", "final_text_version", "source_language", "target_language"]
        cols_to_drop = [col for col in df.columns if col not in desired_cols]
        df = df.drop(columns=cols_to_drop)

        # Rename target column
        df = df.rename(columns={"final_text_version": "target_text"})

        self.df = df

    def get_cleaned_data(self):
        return self.df.copy()
    
    def save_cleaned_data(self, output_path: str, index: bool = False):
        self.df.to_csv(output_path, index=index)

#data splitting and saving as jsonl

def save_as_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    # print(f"Saved {len(data)} records to: {output_path}")

def split_and_save(input_csv_path: str, output_dir: str):
    # Load and clean data
    loader = DataLoader(input_csv_path)
    df = loader.get_cleaned_data()

    # Build records list
    records = [
        {
            "source": row["source_text"],
            "target": row["target_text"],
            "source_language": row["source_language"],
            "target_language": row["target_language"]
        }
        for _, row in df.iterrows()
    ]

    # Split: 80% train | 20% temp
    train_data, temp_data = train_test_split(records, test_size=0.2, random_state=42)

    # Split temp: 50% test | 50% validation  →  each becomes 10% of total
    test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    save_as_jsonl(train_data, os.path.join(output_dir, "train.jsonl"))
    save_as_jsonl(test_data,  os.path.join(output_dir, "test.jsonl"))
    save_as_jsonl(val_data,   os.path.join(output_dir, "validation.jsonl"))


if __name__ == "__main__":
    input_csv  = "Model training for MT\\Dataset\\input5k.csv"
    output_dir = "Model training for MT\\Dataset\\splits"
    split_and_save(input_csv, output_dir)

#model loading and collate function
def load_model(model_name: str = "unsloth/gemma-4-E2B-it"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 4096,
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
        prompt_length = len(prompt_tokenized["input_ids"][0])
        labels[i, :prompt_length] = -100
    
    labels[labels == tokenizer.pad_token_id] = -100
 
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

#training 


