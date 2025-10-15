# Fine-tuning DistilBERT for Financial Sentiment Classification

Comparison of Full Fine-tuning vs LoRA on Financial PhraseBank dataset.

Model: DistilBERT  
Dataset: Financial PhraseBank (3-class sentiment)  
Task: Text classification (negative, neutral, positive)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
```bash
python dataset_prep.py
```
Datasets saved to processed_data folder in dir

### 3. Run Experiments

**Option A: Using main.py**
```bash
# Full fine-tuning
python main.py --method full

# LoRA fine-tuning
python main.py --method lora

# Both
python main.py --method both
```

**Option B: Using Notebooks**
- `full_finetuning.ipynb` - Full fine-tuning implementation
- `lora.ipynb` - LoRA implementation
- `visuals.ipynb` - Generate visualizations

---

## Results

Models and results are saved to:
- `full_finetuning_results/` - Full fine-tuning outputs
- `lora_results/` - LoRA outputs
- Results include: metrics (JSON), trained models, and checkpoints

**Key Findings:**
- LoRA achieves comparable performance with **~99% fewer trainable parameters**
- Full fine-tuning: ~67M parameters
- LoRA: ~740K parameters (1.1% of total)

---

## Repository Structure
```
├── main.py                    # Entry point to reproduce results
├── dataset_prep.py            # Data preprocessing
├── full_finetuning.ipynb      # Full fine-tuning notebook
├── lora.ipynb                 # LoRA notebook  
├── visuals.ipynb              # Visualization generation
├── requirements.txt           # Dependencies
└── processed_data/            # Tokenized dataset (generated)
```

---

## Notes

- Dataset downloads automatically via `dataset_prep.py`
- Models and processed data excluded from repo (download on first run)
