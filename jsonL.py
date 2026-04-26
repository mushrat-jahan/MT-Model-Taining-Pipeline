import json
import os
from data_processing import DataLoader

def save_as_jsonl(input_csv_path: str, output_jsonl_path: str):
    loader = DataLoader(input_csv_path)
    df = loader.get_cleaned_data()

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            record = {
                "source_text": row["source_text"],
                "target_text": row["target_text"],
                "source_language": row["source_language"],
                "target_language": row["target_language"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Data successfully saved to {output_jsonl_path}")


# if __name__ == "__main__":
#     input_csv = "Model training for MT\\Dataset\\input5k.csv"
#     output_jsonl = "Model training for MT\\Dataset\\processed_data.jsonl"

#     save_as_jsonl(input_csv, output_jsonl)