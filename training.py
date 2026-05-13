from unsloth import FastLanguageModel
import json
import torch
import os
import mlflow
from transformers import TrainerCallback
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from prompts.simple_prompt import get_simple_translation_prompt
from model_load import load_model, collate_batch
from utils.jsonL import split_and_save, SPLITS_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Config--------
LORA_DIR = "/home/mushrat/MT-model-training-pipeline/MT-Model-Taining-Pipeline/output/lora"
GGUF_DIR = "/home/mushrat/MT-model-training-pipeline/MT-Model-Taining-Pipeline/output/model"
MLFLOW_DIR = "/home/mushrat/MT-model-training-pipeline/MT-Model-Taining-Pipeline/output/mlflow"
MAX_SEQ_LEN = 4096
BATCH_SIZE = 4
GRAD_ACCUM = 4
LOAD_IN_4BIT = True
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
LORA_BIAS = "none"
WARMUP_RATIO = 0.05
LR_SCHEDULER = "cosine"
EPOCHS = 10
LR = 2e-4
GGUF_QUANT = "q4_k_m"
MODEL_NAME = "unsloth/gemma-4-E2B-it"
MLFLOW_EXPERIMENT_NAME = "mt_gemma_finetuning"



# MLflow ------
class MLflowCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(metrics, step=state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        mlflow.log_metric("epoch", state.epoch, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            eval_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(eval_metrics, step=state.global_step)


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
    os.makedirs(MLFLOW_DIR, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{MODEL_NAME.split('/')[-1]}-finetune"):

        mlflow.log_params({
            "model_name": MODEL_NAME,
            "max_seq_len": MAX_SEQ_LEN,
            "load_in_4bit": LOAD_IN_4BIT,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "lora_bias": LORA_BIAS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "warmup_ratio": WARMUP_RATIO,
            "lr_scheduler": LR_SCHEDULER,
            "gguf_quant": GGUF_QUANT,
        })

        # Prepare splits
        print("Preparing data splits")
        split_and_save()

        # Load model + LoRA
        print("Loading model")
        model, tokenizer = load_model()

        # Log GPU info
        if torch.cuda.is_available():
            mlflow.log_params({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB",
            })

        # Load dataset
        train_dataset = load_jsonl(os.path.join(SPLITS_DIR, "train.jsonl"))
        val_dataset = load_jsonl(os.path.join(SPLITS_DIR, "validation.jsonl"))

        mlflow.log_params({
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        })

        # Configure SFTTrainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=lambda batch: collate_batch(batch, tokenizer),
            args=SFTConfig(
                dataset_text_field="text",
                max_seq_length=MAX_SEQ_LEN,
                output_dir=LORA_DIR,
                num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRAD_ACCUM,
                learning_rate=LR,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                warmup_ratio=0.05,
                lr_scheduler_type="cosine",
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                logging_steps=20,
                report_to="none",
                dataset_num_proc=2,
            ),
        )

        print("Starting training")
        trainer.train()

        # Save LoRA
        os.makedirs(LORA_DIR, exist_ok=True)
        model.save_pretrained(LORA_DIR)
        tokenizer.save_pretrained(LORA_DIR)
        mlflow.log_artifacts(LORA_DIR, artifact_path="lora_weights")
        print("LoRA weights saved.")

        # Save GGUF
        os.makedirs(GGUF_DIR, exist_ok=True)
        print(f"Exporting to GGUF ({GGUF_QUANT})")
        model.save_pretrained_gguf(
            GGUF_DIR,
            tokenizer,
            quantization_method=GGUF_QUANT,
        )
        mlflow.log_artifacts(GGUF_DIR, artifact_path="gguf_model")
        print(f"Process complete. GGUF saved to {GGUF_DIR}")

        mlflow.log_param("status", "completed")
        print(
            "Training run completed and logged to MLflow. Launch UI with: mlflow ui --backend-store-uri file://"
            + MLFLOW_DIR
        )


if __name__ == "__main__":
    main()