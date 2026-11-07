# English-Malay Sentiment Analysis

Work-in-progress FYP for three-class sentiment classification of English-Malay code-mixed tweets.

## Approaches

- **A1:** fine-tuned XLM-R baseline.
- **A2:** A1 probabilities fused with an English-Malay sentiment ontology.
- **A3:** A2 fusion guarded by an ontology coverage threshold.

## Current results

| Split | A1 Macro-F1 | A2 Macro-F1 | A3 Macro-F1 |
|---|---:|---:|---:|
| Validation (207 rows) | 60.01% | 61.62% | 61.62% |
| Official test (444 rows) | 55.02% | 55.08% | 55.08% |

The current A2 and A3 predictions are identical. These are interim results; data-quality work and a UI are planned next. Detailed aggregate metrics are in `reports/metrics/`.

## Dataset

This repository does not redistribute tweet text or personal information. Download Version 2 of the [Annotated The dUCk Tweets Dataset](https://data.mendeley.com/datasets/876tc4dkts/2) and place these files in `data/raw/`:

- `eng_malay_training_dataset.csv`
- `eng_malay_testing_dataset.csv`

Run `notebooks/01_data_preparation.ipynb` to generate the processed splits.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The first A1 run downloads `FacebookAI/xlm-roberta-base` from Hugging Face. Set `HF_HUB_OFFLINE=1` only when the model is already cached.

## Run order

```powershell
python 02_train_A1.py
python 03_run_A2.py
python 04_run_A3.py
python 05_test.py
python 06_data_audit.py
```

- A1 uses 826 training rows and 207 validation rows.
- A2 selects the fusion weight on validation data.
- A3 selects the fusion weight and coverage threshold on validation data.
- `05_test.py` evaluates the frozen configuration on the 444-row official test split.
- `06_data_audit.py` creates human-review candidates without automatically changing labels.

Generated datasets, tweet-level predictions, trained models, checkpoints, local audit files, and editor settings are intentionally excluded from Git. Aggregate metrics and source code remain reproducible.
