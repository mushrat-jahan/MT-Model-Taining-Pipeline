import json
import os
from data_processing import DataLoader
from sklearn.model_selection import train_test_split

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