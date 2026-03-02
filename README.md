# Tamil Political Multiclass Sentiment Analysis - CHMOD_777

**DravidianLangTech@ACL 2026 Shared Task**
- **Competition:** [Codabench #11327](https://www.codabench.org/competitions/11327/)
- **Team:** **CHMOD_777**
  - Arunaggiri Pandian Karunanidhi (Micron Technology)
  - Prabalakshmi Arumugam (Boise State University)

This repository contains our code for Tamil political multiclass sentiment analysis.

## Results

| Model | Dev Macro F1 | Notes |
|-------|--------------|-------|
| **MuRIL (augmented)** | **35.79%** | Best model |
| IndicBERT-v3-1B (augmented) | 32.09% | Larger model |
| IndicBERT-v3-1B (original) | 30.28% | No augmentation |
| MuRIL (original) | 21.40% | No augmentation |

### Key Findings

1. **LLM augmentation provides 67% improvement:** 21.4% → 35.79% F1
2. **Smaller models benefit more:** MuRIL outperforms larger IndicBERT-1B after augmentation

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.36+
- CUDA 11.8+ (optional, for GPU training)

## Project Structure

```
political_multiclass_sentiment_analysis/
├── src/
│   ├── train_transformer.py   # Transformer model training
│   ├── train_llm.py           # LLM fine-tuning with LoRA
│   ├── inference.py           # Prediction generation
│   ├── analyze.py             # Model analysis & visualization
│   ├── data_utils.py          # Data loading utilities
│   ├── augmentation.py        # Data augmentation utilities
│   ├── augment_with_gemini.py # Gemini-based augmentation
│   └── augment_with_llm.py    # LLM augmentation utilities
├── configs/                   # Model configurations
├── data/                      # Dataset (not included)
├── outputs/
│   ├── models/               # Trained models
│   └── predictions/          # Submission files
├── scripts/
│   └── standalone_gemini_augment_v3.py  # GCP Vertex AI augmentation script
├── requirements.txt
├── run.sh                    # Convenience script
├── paper.md                  # Paper draft
└── README.md
```

## Usage

### 1. Data Augmentation (Optional)

Using Gemini on GCP Vertex AI:
```bash
# Upload to Vertex AI Workbench and run:
python scripts/standalone_gemini_augment_v3.py
```

### 2. Training

```bash
# Train MuRIL (best model)
./run.sh train-transformer muril_base

# Or with custom parameters
python src/train_transformer.py \
    --model_name google/muril-base-cased \
    --data_dir data \
    --train_file PS_train_final.csv \
    --output_dir outputs/models/muril_augmented \
    --max_length 256 \
    --batch_size 16 \
    --epochs 50 \
    --learning_rate 1e-5 \
    --focal_gamma 3.0 \
    --use_weighted_sampler \
    --early_stopping_patience 10
```

### 3. Inference

```bash
# Generate predictions
./run.sh predict outputs/models/muril_augmented/best_model

# Or with Python
python src/inference.py \
    --model_dir outputs/models/muril_augmented \
    --test_file data/PS_test_without_labels.csv \
    --output_file outputs/predictions/run1.csv
```

### 4. Analysis

```bash
# Analyze model performance
./run.sh analyze-model outputs/models/muril_augmented
```

## Data Augmentation

We use Gemini 2.5 Flash for LLM-based paraphrasing:
- **Original data:** 4,352 samples
- **Augmented data:** 15,316 samples (3.52x)
- **Improvement:** +67% Macro F1

## Models Used

- **MuRIL** (google/muril-base-cased): 236M parameters - Best for Tamil
- **IndicBERT-v3-1B** (ai4bharat/IndicBERT-v3-1B): 1B parameters
- **IndicBERT-v3-270M** (ai4bharat/IndicBERT-v3-270M): 270M parameters

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | google/muril-base-cased |
| Max length | 256 |
| Batch size | 16 |
| Learning rate | 1e-5 |
| Loss function | Focal Loss (gamma=3.0) |
| Weighted sampler | Yes |
| Epochs | 50 (early stopping) |
| Patience | 10 epochs |

---
<p align="center">Author: Arunaggiri Pandian Karunanidhi</p>
