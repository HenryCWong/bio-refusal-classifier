# Biosecurity Prompt Classifier

Binary text classifier that predicts whether a natural language prompt should be refused on biosecurity grounds. Five models are trained and compared: TF-IDF + Logistic Regression, BERT, RoBERTa, BioBERT, and DistilBERT.

---

## Setup

### Requirements

Python 3.10+ and a GPU are recommended. Tested on Python 3.13 with an NVIDIA RTX 3080 (10GB VRAM).

### Installation

```bash
# Clone or download the project
cd cbai-interview

# Create and activate virtual environment
python -m venv cbai
source cbai/bin/activate  # Linux/Mac
# or: cbai\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets scikit-learn pandas numpy matplotlib seaborn accelerate sentencepiece protobuf tiktoken
```

### Data

No manual data download required. All datasets are fetched automatically from HuggingFace on first run and cached locally (~2.5GB total):

- `cais/wmdp` (wmdp-bio) — hazardous prompts (~1,273 examples)
- `cais/mmlu` (all) — safe general + biology prompts (~400MB)
- `qiaojin/PubMedQA` (pqa_labeled) — safe biomedical prompts (~1,000 examples)

---

## File Structure

```
cbai-interview/
├── BERT-finetune.py                  # BERT fine-tuning and evaluation
├── RoBERaT-finetune.py               # RoBERTa fine-tuning and evaluation
├── biobert-finetune.py               # BioBERT fine-tuning and evaluation
├── distilbert-finetune.py            # DistilBERT fine-tuning and evaluation
├── logistic_regression.py            # TF-IDF + Logistic Regression baseline
├── error_analysis.py                 # Standalone error analysis (run after training)
├── biosecurity_classifier_report.md  # Full results report
├── results_summary.md                # Results tables across all configurations
└── README.md                         # This file

# Generated after training:
├── bert-biosec-classifier-output/
├── roberta-biosec-classifier-output/
├── biobert-biosec-classifier-output/
├── distilbert-biosec-classifier-output/
└── confusion_matrix.png
```

---

## Running the Scripts

### 1. Logistic Regression Baseline (fastest, ~1 minute)

```bash
python logistic_regression.py
```

Outputs test set metrics and adversarial test results directly to console. No GPU required.

### 2. Transformer Models (~5-7 minutes each)

Run each fine-tuning script independently. Each loads data, trains for 3 epochs, evaluates on the test set, and saves the best checkpoint.

```bash
python BERT-finetune.py
python RoBERaT-finetune.py
python biobert-finetune.py
python distilbert-finetune.py
```

**Memory note:** Each script requires ~4-5GB free VRAM. If running multiple models sequentially, clear GPU memory between runs:

```python
import torch, gc
del model, trainer
gc.collect()
torch.cuda.empty_cache()
```

If you encounter OOM errors, reduce `per_device_train_batch_size` to 8 and add `gradient_accumulation_steps=2`.

### 3. Error Analysis (run after training)

```bash
python error_analysis.py
```

Configure the model to analyse at the top of the file:

```python
# Lines 25-26 — change these to switch between models
MODEL_NAME = "bert-base-uncased"
MODEL_DIR  = "./bert-biosec-classifier-output"
```

Valid configurations:

| MODEL_NAME | MODEL_DIR |
|---|---|
| `"bert-base-uncased"` | `"./bert-biosec-classifier-output"` |
| `"roberta-base"` | `"./roberta-biosec-classifier-output"` |
| `"dmis-lab/biobert-base-cased-v1.2"` | `"./biobert-biosec-classifier-output"` |
| `"distilbert-base-uncased"` | `"./distilbert-biosec-classifier-output"` |

---

## Expected Output

Each fine-tuning script prints epoch-by-epoch training metrics followed by test set results:

```
Test Set Results:
  Accuracy:  0.9963
  Precision: 1.0000
  Recall:    0.9948
  F1 Score:  0.9974
```

The error analysis script additionally prints per-class metrics, false negative/positive analysis, threshold analysis, and adversarial test results, and saves `confidence_distribution.png`.

---

## Notes

**AI assistance:** Claude was used for debugging, grammer and spell check, reformatting code to become readable, and markdown formatting  for this README and other md documents (specifically for tables).

**DeBERTa:** Attempted but incompatible with the current environment (Python 3.13 + CUDA 13.0). Training loss was consistently 0.0 due to NaN overflow in the disentangled attention mechanism under both fp16 and bf16. Excluded from final results.

---

## Results Summary (best configuration per model)

| Model | Best data | F1 | Adversarial (14 prompts) |
|---|---|---|---|
| TF-IDF + Logistic Regression | MMLU + PubMedQA | 0.8951 | 10/14 (71%) |
| BERT | MMLU (no filter) | 0.9947 | 10/14 (71%) |
| RoBERTa | MMLU + PubMedQA | **0.9974** | 10/14 (71%) |
| BioBERT | MMLU + PubMedQA | **0.9974** | 10/14 (71%) |
| DistilBERT | MMLU (no filter) | 0.9947 | 9/14 (64%) |

See `biosecurity_classifier_report.md` for full analysis and `results_summary.md` for all configurations.
