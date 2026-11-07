"""A2: select matched ontology late-fusion alpha on validation only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.evaluation import LABEL_NAMES, classification_metrics
from src.ontology_fusion import ProductSentimentOntology, coverage_aware_fuse_probabilities

ROOT = Path(__file__).resolve().parent
A1_PATH = ROOT / "results/A1/validation_predictions.csv"
OUT = ROOT / "results/A2"
ALPHAS = np.round(np.arange(0.05, 0.51, 0.05), 2)


def main():
    if (OUT / "validation_predictions.csv").exists():
        raise FileExistsError("A2 output already exists; move it before a new run.")
    if not A1_PATH.exists():
        raise FileNotFoundError("Run 02_train_A1.py first.")
    frame = pd.read_csv(A1_PATH)
    y = frame.label.to_numpy(dtype=int)
    x = frame[[f"xlmr_prob_{name}" for name in LABEL_NAMES]].to_numpy()
    features = ProductSentimentOntology(ROOT / "ontology/sentiment_lexicon.csv").analyse_frame(frame)
    ontology = features[[f"ontology_prob_{name}" for name in LABEL_NAMES]].to_numpy()
    trials, outputs = [], {}
    for alpha in ALPHAS:
        fused, gate = coverage_aware_fuse_probabilities(
            x, ontology, features.coverage, features.match_count,
            alpha=float(alpha), threshold=0.0,
        )
        row = {"alpha": float(alpha), "fused_rows": int(gate.sum()),
               **classification_metrics(y, fused.argmax(1))}
        trials.append(row)
        outputs[float(alpha)] = (fused, gate)
    best = sorted(trials, key=lambda r: (-r["macro_f1"], -r["accuracy"], r["alpha"]))[0]
    fused, gate = outputs[best["alpha"]]
    result = pd.concat([frame.reset_index(drop=True), features], axis=1)
    for index, name in enumerate(LABEL_NAMES):
        result[f"A2_prob_{name}"] = fused[:, index]
    result["A2_prediction"] = fused.argmax(1)
    result["A2_fusion_used"] = gate
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trials).to_csv(OUT / "alpha_search.csv", index=False)
    result.to_csv(OUT / "validation_predictions.csv", index=False)
    summary = {
        "approach": "A2",
        "formula": "(1-alpha)*A1_probability + alpha*ontology_probability",
        "selected_alpha": best["alpha"],
        "fused_rows": best["fused_rows"],
        "validation_metrics": {k: v for k, v in best.items() if k not in {"alpha", "fused_rows"}},
        "official_test_used": False,
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics = summary["validation_metrics"]
    print("A2 COMPLETE")
    print(f"Selected alpha: {summary['selected_alpha']:.2f}")
    print(f"Ontology fusion used: {summary['fused_rows']} / {len(frame)} rows")
    print(f"Validation Macro-F1: {metrics['macro_f1'] * 100:.2f}%")
    print(f"Validation accuracy: {metrics['accuracy'] * 100:.2f}%")
    print("Official test used: No")
    print(f"Full results saved to: {OUT}")


if __name__ == "__main__":
    main()

