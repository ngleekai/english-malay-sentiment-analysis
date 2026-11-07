"""One-time evaluation of frozen A1, A2, and A3 on the official test set."""
from pathlib import Path
import argparse
from datetime import datetime
import hashlib
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as transformers_logging

from src.evaluation import LABEL_NAMES, classification_metrics
from src.ontology_fusion import ProductSentimentOntology, coverage_aware_fuse_probabilities

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models/A1_xlmr/best_model"
TEST_PATH = ROOT / "data/processed/test.csv"
A2_CONFIG_PATH = ROOT / "results/A2/metrics.json"
A3_CONFIG_PATH = ROOT / "results/A3/metrics.json"
LEXICON_PATH = ROOT / "ontology/sentiment_lexicon.csv"
ORIGINAL_OUT = ROOT / "results/test"
MAX_LENGTH = 96
BATCH_SIZE = 8


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predict_a1(texts):
    if not torch.cuda.is_available():
        raise RuntimeError("Official test inference requires the CUDA GPU used for training.")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True
    ).to(device)
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = tokenizer(
                texts[start:start + BATCH_SIZE],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**batch).logits
            probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probabilities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerun", action="store_true",
        help="Repeat the frozen test without overwriting the original results.",
    )
    args = parser.parse_args()
    if args.rerun:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = ROOT / "results" / f"test_rerun_{timestamp}"
    else:
        out = ORIGINAL_OUT
        if out.exists():
            raise FileExistsError(
                f"Official test results already exist: {out}. "
                "Use --rerun to create a separate reproducibility run."
            )
    required = [MODEL_DIR / "model.safetensors", TEST_PATH, A2_CONFIG_PATH,
                A3_CONFIG_PATH, LEXICON_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")

    a2_config = json.loads(A2_CONFIG_PATH.read_text(encoding="utf-8"))
    a3_config = json.loads(A3_CONFIG_PATH.read_text(encoding="utf-8"))
    test = pd.read_csv(TEST_PATH)
    if len(test) != 444 or set(test.label.unique()) - {0, 1, 2}:
        raise ValueError("Unexpected official test data.")

    print("OFFICIAL TEST STARTED")
    print(f"Test rows: {len(test)}")
    print("Frozen parameters: A2 alpha=0.40 | A3 alpha=0.40, threshold=0.02")
    print("Running predictions...", flush=True)

    # Predictions are created from text before labels are used for scoring.
    a1_probabilities = predict_a1(test.tweet_text.astype(str).tolist())
    features = ProductSentimentOntology(LEXICON_PATH).analyse_frame(test)
    ontology = features[[f"ontology_prob_{name}" for name in LABEL_NAMES]].to_numpy()
    a2_probabilities, a2_gate = coverage_aware_fuse_probabilities(
        a1_probabilities, ontology, features.coverage, features.match_count,
        alpha=float(a2_config["selected_alpha"]), threshold=0.0,
    )
    a3_probabilities, a3_gate = coverage_aware_fuse_probabilities(
        a1_probabilities, ontology, features.coverage, features.match_count,
        alpha=float(a3_config["selected_alpha"]),
        threshold=float(a3_config["selected_coverage_threshold"]),
    )

    predictions = {
        "A1": a1_probabilities.argmax(1),
        "A2": a2_probabilities.argmax(1),
        "A3": a3_probabilities.argmax(1),
    }
    y = test.label.to_numpy(dtype=int)
    metrics = {name: classification_metrics(y, values)
               for name, values in predictions.items()}

    result = pd.concat([test.reset_index(drop=True), features], axis=1)
    for approach, probabilities in {
        "A1": a1_probabilities, "A2": a2_probabilities, "A3": a3_probabilities
    }.items():
        for index, label_name in enumerate(LABEL_NAMES):
            result[f"{approach}_prob_{label_name}"] = probabilities[:, index]
        result[f"{approach}_prediction"] = predictions[approach]
    result["A2_fusion_used"] = a2_gate
    result["A3_fusion_used"] = a3_gate

    out.mkdir(parents=True)
    result.to_csv(out / "predictions.csv", index=False)
    comparison = pd.DataFrame([
        {"approach": name, **metrics[name]} for name in ("A1", "A2", "A3")
    ])
    comparison.to_csv(out / "metrics.csv", index=False)
    for name in ("A1", "A2", "A3"):
        matrix = confusion_matrix(y, predictions[name], labels=[0, 1, 2])
        pd.DataFrame(matrix, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(
            out / f"{name}_confusion_matrix.csv"
        )
    summary = {
        "status": "official test reproducibility rerun" if args.rerun else "final official test completed",
        "test_rows": len(test),
        "frozen_parameters": {
            "A2": {"alpha": a2_config["selected_alpha"]},
            "A3": {"alpha": a3_config["selected_alpha"],
                   "coverage_threshold": a3_config["selected_coverage_threshold"]},
        },
        "fusion_rows": {"A2": int(a2_gate.sum()), "A3": int(a3_gate.sum())},
        "test_metrics": metrics,
        "official_test_used": True,
        "frozen_input_sha256": {
            "model": sha256(MODEL_DIR / "model.safetensors"),
            "test": sha256(TEST_PATH),
            "lexicon": sha256(LEXICON_PATH),
            "A2_config": sha256(A2_CONFIG_PATH),
            "A3_config": sha256(A3_CONFIG_PATH),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFINAL OFFICIAL TEST COMPLETE")
    print("Approach | Macro-F1 | Accuracy")
    for name in ("A1", "A2", "A3"):
        print(f"{name:8} | {metrics[name]['macro_f1'] * 100:7.2f}% | {metrics[name]['accuracy'] * 100:7.2f}%")
    print(f"A2 fusion used: {a2_gate.sum()} / {len(test)} rows")
    print(f"A3 fusion used: {a3_gate.sum()} / {len(test)} rows")
    print(f"Results saved to: {out}")


if __name__ == "__main__":
    main()
