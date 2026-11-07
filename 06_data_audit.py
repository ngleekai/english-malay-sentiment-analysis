"""Create human-review candidates without changing any dataset labels."""
from pathlib import Path
from collections import Counter
import json
import re

import pandas as pd

from src.ontology_fusion import NEGATORS, ProductSentimentOntology, tokenize

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data/processed/train.csv"
VALIDATION_PATH = ROOT / "data/processed/validation.csv"
LEXICON_PATH = ROOT / "ontology/sentiment_lexicon.csv"
OUT = ROOT / "results/data_audit"
CANDIDATES_PATH = OUT / "label_audit_candidates.csv"

LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Transparent heuristics: these only flag text for human review.
ENGLISH_HINTS = {
    "the", "this", "that", "is", "are", "and", "but", "not", "very",
    "really", "good", "bad", "best", "worst", "service", "product",
    "delivery", "quality", "price", "love", "hate", "fast", "slow",
}
MALAY_HINTS = {
    "yang", "dan", "ini", "itu", "tak", "tidak", "tapi", "sangat",
    "memang", "bagus", "teruk", "barang", "harga", "penghantaran",
    "suka", "benci", "cepat", "lambat", "lah", "ni", "tu",
}
INFORMAL_MARKERS = {
    "x", "xde", "takleh", "gila", "giler", "sumpah", "lah", "leh",
    "lol", "omg", "wtf", "mantap", "padu",
}


def normalized_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\b(?:url|user)\b", " ", text)
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def ontology_conflict(label, score):
    """Return True only for strong, obvious polarity disagreement."""
    if label == 0 and score >= 0.40:
        return True
    if label == 2 and score <= -0.40:
        return True
    if label == 1 and abs(score) >= 0.60:
        return True
    return False


def main():
    if CANDIDATES_PATH.exists():
        raise FileExistsError(
            f"Audit file already exists: {CANDIDATES_PATH}. "
            "It may contain manual decisions, so it will not be overwritten."
        )

    frames = []
    for split, path in (("train", TRAIN_PATH), ("validation", VALIDATION_PATH)):
        frame = pd.read_csv(path).copy()
        frame.insert(0, "split", split)
        frame.insert(1, "row_id", range(len(frame)))
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["normalized_text"] = data.tweet_text.map(normalized_text)

    label_sets = data.groupby("normalized_text").label.agg(lambda values: set(values))
    conflicting_duplicates = {
        text for text, labels in label_sets.items() if text and len(labels) > 1
    }
    duplicate_counts = data.normalized_text.value_counts()

    ontology = ProductSentimentOntology(LEXICON_PATH)
    records = []
    reason_counts = Counter()
    for row in data.itertuples(index=False):
        tokens = tokenize(row.tweet_text)
        token_set = set(tokens)
        analysis = ontology.analyse(row.tweet_text)
        reasons = []

        if row.normalized_text in conflicting_duplicates:
            reasons.append("conflicting_duplicate_labels")
        elif row.normalized_text and duplicate_counts[row.normalized_text] > 1:
            reasons.append("duplicate_same_label")
        if ontology_conflict(int(row.label), float(analysis["ontology_score"])):
            reasons.append("strong_ontology_label_conflict")
        negators = sorted(token_set & NEGATORS)
        if negators:
            reasons.append("contains_negation")
        english = sorted(token_set & ENGLISH_HINTS)
        malay = sorted(token_set & MALAY_HINTS)
        if english and malay:
            reasons.append("possible_english_malay_mix")
        informal = sorted(token_set & INFORMAL_MARKERS)
        if informal:
            reasons.append("informal_or_slang_marker")
        if len(tokens) <= 1:
            reasons.append("very_short_text")

        if not reasons:
            continue
        for reason in reasons:
            reason_counts[reason] += 1
        matched_terms = [match["term"] for match in analysis["matches"]]
        priority = (
            1 if "conflicting_duplicate_labels" in reasons else
            2 if "strong_ontology_label_conflict" in reasons else
            3 if "very_short_text" in reasons else
            4
        )
        records.append({
            "priority": priority,
            "split": row.split,
            "row_id": row.row_id,
            "tweet_text": row.tweet_text,
            "current_label": int(row.label),
            "current_label_name": LABEL_NAMES[int(row.label)],
            "suspicious_reasons": "; ".join(reasons),
            "matched_sentiment_terms": "; ".join(matched_terms),
            "detected_negators": "; ".join(negators),
            "detected_informal_markers": "; ".join(informal),
            "ontology_score": round(float(analysis["ontology_score"]), 4),
            "your_decision": "",
            "corrected_label": "",
            "notes": "",
        })

    candidates = pd.DataFrame(records).sort_values(
        ["priority", "split", "row_id"], kind="stable"
    )
    OUT.mkdir(parents=True)
    candidates.to_csv(CANDIDATES_PATH, index=False)
    candidates.loc[candidates.priority.le(2)].to_csv(
        OUT / "review_first_high_priority.csv", index=False
    )
    summary = {
        "rows_checked": {"train": len(frames[0]), "validation": len(frames[1])},
        "candidate_rows": len(candidates),
        "high_priority_rows": int(candidates.priority.le(2).sum()),
        "reason_counts": dict(reason_counts.most_common()),
        "labels_changed": 0,
        "official_test_used": False,
        "warning": "Every candidate requires human review; a flag is not proof of a wrong label.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(
        [{"reason": reason, "candidate_rows": count}
         for reason, count in reason_counts.most_common()]
    ).to_csv(OUT / "reason_counts.csv", index=False)

    print("DATA QUALITY AUDIT COMPLETE")
    print(f"Rows checked: {len(data)} (train {len(frames[0])} + validation {len(frames[1])})")
    print(f"Candidate rows for human review: {len(candidates)}")
    print(f"High-priority rows: {summary['high_priority_rows']}")
    print("Labels changed automatically: 0")
    print("Official test used: No")
    print(f"Review file: {CANDIDATES_PATH}")


if __name__ == "__main__":
    main()

