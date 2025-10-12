from datasets import load_dataset, DatasetDict
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import json
import os

dataset = load_dataset("financial_phrasebank", "sentences_50agree",trust_remote_code=True)

print(dataset)

train_test_split = dataset['train'].train_test_split(test_size = 0.2, seed = 42, stratify_by_column='label')
val_test_split = train_test_split['test'].train_test_split(test_size = 0.5, seed = 42, stratify_by_column='label')

dataset_split = {
    'train': train_test_split['train'],
    'validation': val_test_split['train'],
    'test': val_test_split['test']
}

print(f"\nTrain samples: {len(dataset_split['train'])}")
print(f"Validation samples: {len(dataset_split['validation'])}")
print(f"Test samples: {len(dataset_split['test'])}")


# Label mapping
label_names = ['negative', 'neutral', 'positive']
id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
label2id = {'negative': 0, 'neutral': 1, 'positive': 2}

# Label distribution
print("\n" + "="*60)
print("Label Distribution Across Splits:")
print("="*60)

for split_name in ['train', 'validation', 'test']:
    labels = dataset_split[split_name]['label']
    unique, counts = np.unique(labels, return_counts=True)
    
    print(f"\n{split_name.upper()}:")
    for label_id, count in zip(unique, counts):
        percentage = count / len(labels) * 100
        print(f"  {id2label[label_id]}: {count:4d} ({percentage:5.2f}%)")
    print(f"  Total: {len(labels)}")

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)

# Apply tokenization to all splits
tokenized_datasets = {
    'train': dataset_split['train'].map(tokenize_function, batched=True),
    'validation': dataset_split['validation'].map(tokenize_function, batched=True),
    'test': dataset_split['test'].map(tokenize_function, batched=True)
}

output_dir = 'processed_data'
os.makedirs(output_dir, exist_ok=True)

for split_name, split_data in tokenized_datasets.items():
    split_data.save_to_disk(f'{output_dir}/{split_name}')

with open(f'{output_dir}/label_mappings.json', 'w') as f:
    json.dump({'id2label': id2label, 'label2id': label2id}, f)