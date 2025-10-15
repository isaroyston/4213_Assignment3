"""
Main script to run full fine-tuning or LoRA experiments on DistilBERT for sentiment classification.
"""
import argparse
import json
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'f1_weighted': f1_score(labels, predictions, average='weighted'),
        'precision': precision_score(labels, predictions, average='macro'),
        'recall': recall_score(labels, predictions, average='macro')
    }


def load_data():
    train = load_from_disk("processed_data/train")
    val = load_from_disk("processed_data/validation")
    test = load_from_disk("processed_data/test")
    return train, val, test


def run_full_finetuning():
    print("Starting Full Fine-tuning...")
    
    train, val, test = load_data()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3
    )
    
    training_args = TrainingArguments(
        output_dir="full_finetuning_results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        save_total_limit=2
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # Evaluate
    val_results = trainer.evaluate(val)
    test_results = trainer.evaluate(test)
    
    # Save model and results
    model.save_pretrained("full_finetuning_model")
    tokenizer.save_pretrained("full_finetuning_model")
    
    results = {
        "validation": val_results,
        "test": test_results
    }
    
    with open("full_finetuning_results/results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Full Fine-tuning Complete!")
    print(f"Val Accuracy: {val_results['eval_accuracy']:.4f}")
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")


def run_lora():
    print("Starting LoRA Fine-tuning...")
    
    train, val, test = load_data()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir="lora_results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        save_total_limit=2
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # Evaluate
    val_results = trainer.evaluate(val)
    test_results = trainer.evaluate(test)
    
    # Save model and results
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    results = {
        "validation": val_results,
        "test": test_results,
        "trainable_params": trainable_params,
        "total_params": total_params
    }
    
    with open("lora_results/results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"LoRA Fine-tuning Complete!")
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")
    print(f"Val Accuracy: {val_results['eval_accuracy']:.4f}")
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run sentiment classification experiments")
    parser.add_argument(
        "--method",
        type=str,
        choices=["full", "lora", "both"],
        required=True,
        help="Training method: 'full' for full fine-tuning, 'lora' for LoRA, 'both' for both"
    )
    
    args = parser.parse_args()
    
    if args.method == "full":
        run_full_finetuning()
    elif args.method == "lora":
        run_lora()
    else:
        run_full_finetuning()
        run_lora()


if __name__ == "__main__":
    main()
