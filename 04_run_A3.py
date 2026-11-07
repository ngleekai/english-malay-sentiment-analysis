"""A3: select alpha and coverage threshold on validation only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.evaluation import LABEL_NAMES, classification_metrics
from src.ontology_fusion import ProductSentimentOntology, coverage_aware_fuse_probabilities

ROOT = Path(__file__).resolve().parent
A1_PATH = ROOT / "results/A1/validation_predictions.csv"
OUT = ROOT / "results/A3"
ALPHAS = np.round(np.arange(0.05, 0.51, 0.05), 2)
THRESHOLDS = (0.02, 0.05, 0.10, 0.15, 0.20)


def main():
    if (OUT / "validation_predictions.csv").exists():
        raise FileExistsError("A3 output already exists; move it before a new run.")
    if not A1_PATH.exists():
        raise FileNotFoundError("Run 02_train_A1.py first.")
    frame = pd.read_csv(A1_PATH)
    y = frame.label.to_numpy(dtype=int)
    x = frame[[f"xlmr_prob_{name}" for name in LABEL_NAMES]].to_numpy()
    features = ProductSentimentOntology(ROOT / "ontology/sentiment_lexicon.csv").analyse_frame(frame)
    ontology = features[[f"ontology_prob_{name}" for name in LABEL_NAMES]].to_numpy()
    trials, outputs = [], {}
    for alpha in ALPHAS:
        for threshold in THRESHOLDS:
            fused, gate = coverage_aware_fuse_probabilities(
                x, ontology, features.coverage, features.match_count,
                alpha=float(alpha), threshold=float(threshold),
            )
            key = (float(alpha), float(threshold))
            trials.append({"alpha": key[0], "coverage_threshold": key[1],
                           "fused_rows": int(gate.sum()),
                           **classification_metrics(y, fused.argmax(1))})
            outputs[key] = (fused, gate)
    best = sorted(trials, key=lambda r: (
        -r["macro_f1"], -r["accuracy"], r["alpha"], -r["coverage_threshold"]
    ))[0]
    fused, gate = outputs[(best["alpha"], best["coverage_threshold"])]
    result = pd.concat([frame.reset_index(drop=True), features], axis=1)
    for index, name in enumerate(LABEL_NAMES):
        result[f"A3_prob_{name}"] = fused[:, index]
    result["A3_prediction"] = fused.argmax(1)
    result["A3_fusion_used"] = gate
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trials).to_csv(OUT / "parameter_search.csv", index=False)
    result.to_csv(OUT / "validation_predictions.csv", index=False)
    summary = {
        "approach": "A3",
        "gate": "match_count > 0 and coverage >= coverage_threshold",
        "coverage": "matched sentiment content tokens / eligible content tokens",
        "selected_alpha": best["alpha"],
        "selected_coverage_threshold": best["coverage_threshold"],
        "fused_rows": best["fused_rows"],
        "validation_metrics": {k: v for k, v in best.items()
                               if k not in {"alpha", "coverage_threshold", "fused_rows"}},
        "official_test_used": False,
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics = summary["validation_metrics"]
    print("A3 COMPLETE")
    print(f"Selected alpha: {summary['selected_alpha']:.2f}")
    print(f"Selected coverage threshold: {summary['selected_coverage_threshold']:.2f}")
    print(f"Ontology fusion used: {summary['fused_rows']} / {len(frame)} rows")
    print(f"Validation Macro-F1: {metrics['macro_f1'] * 100:.2f}%")
    print(f"Validation accuracy: {metrics['accuracy'] * 100:.2f}%")
    print("Official test used: No")
    print(f"Full results saved to: {OUT}")


if __name__ == "__main__":
    main()
