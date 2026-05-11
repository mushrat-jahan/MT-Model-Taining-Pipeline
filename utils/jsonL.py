import sys
import json
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_processing import DataLoader
from sklearn.model_selection import train_test_split


INPUT_CSV = "/home/mushrat/MT-model-training-pipeline/MT-Model-Taining-Pipeline/Dataset/AcceptedMachineTranslationData.csv"
SPLITS_DIR = "/home/mushrat/MT-model-training-pipeline/MT-Model-Taining-Pipeline/output"

def save_as_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} records to: {output_path}")

def split_and_save():
    loader = DataLoader(INPUT_CSV)
    df = loader.get_cleaned_data()

    records = [
        {
            "source": row["source_text"],
            "target": row["target_text"],
            "source_language": row["source_language"],
            "target_language": row["target_language"]
        }
        for _, row in df.iterrows()
    ]

    train_data, temp_data = train_test_split(records, test_size=0.2, random_state=42)
    test_data, val_data   = train_test_split(temp_data, test_size=0.5, random_state=42)

    os.makedirs(SPLITS_DIR, exist_ok=True)
    save_as_jsonl(train_data, os.path.join(SPLITS_DIR, "train.jsonl"))
    save_as_jsonl(test_data,  os.path.join(SPLITS_DIR, "test.jsonl"))
    save_as_jsonl(val_data,   os.path.join(SPLITS_DIR, "validation.jsonl"))
