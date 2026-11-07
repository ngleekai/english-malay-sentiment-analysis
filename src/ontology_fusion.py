"""Ontology matching and late-fusion utilities for Approaches 2 and 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", flags=re.IGNORECASE)
NEGATORS = {
    "tak", "tidak", "x", "bukan", "jangan", "belum", "no", "not",
    "never", "dont", "don't", "cant", "can't", "cannot", "without",
    "takleh", "xde",
}
INTENSIFIERS = {
    "sangat", "memang", "really", "very", "terlalu", "paling", "so",
    "gila", "betul", "amat", "super", "sumpah",
}
DOWNTONERS = {"agak", "quite", "sedikit", "slightly", "maybe"}

# Fixed English-Malay function-word list used only in the coverage denominator.
# The XLM-R input is never altered by this list.
COVERAGE_FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "for", "from", "had", "has", "have", "he", "her", "hers", "him",
    "his", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on",
    "or", "our", "ours", "she", "so", "that", "the", "their", "them",
    "they", "this", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "you", "your", "yours",
    "ada", "aku", "anda", "apa", "awak", "bagi", "bahawa", "bila", "boleh",
    "dah", "dalam", "dan", "dari", "daripada", "dekat", "dengan", "dia", "di",
    "ini", "itu", "jadi", "juga", "kalau", "kami", "kat", "ke", "kepada",
    "kita", "lah", "lagi", "macam", "mahu", "mana", "mau", "mereka", "mesti",
    "nak", "ni", "oleh", "pada", "pun", "sahaja", "saja", "saya", "sebab",
    "sebagai", "semua", "seorang", "seperti", "sudah", "tak", "tapi", "telah",
    "tidak", "tu", "untuk", "yang", "yg",
}
COVERAGE_PLACEHOLDERS = {"url", "user"}
SCORING_VERSION = "mean_weighted_tanh_v2"
COVERAGE_VERSION = "sentiment_only_v3"
MATCHING_VERSION = "clause_bounded_v4"
CLAUSE_BREAKS = re.compile(r"[.!?;,:\n\r。！？；，：]")
CONTRAST_WORDS = {"but", "however", "yet", "tapi", "tetapi", "namun", "tp"}


@dataclass(frozen=True)
class LexiconEntry:
    term: str
    tokens: tuple[str, ...]
    language: str
    polarity: int
    weight: float
    aspect: str


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


class ProductSentimentOntology:
    """Frozen lexicon matcher with negation and intensity handling."""

    def __init__(self, lexicon_path: str | Path):
        self.lexicon_path = Path(lexicon_path)
        frame = pd.read_csv(self.lexicon_path)
        required = {"term", "entry_type", "language", "polarity", "weight", "aspect"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Lexicon is missing columns: {sorted(missing)}")
        self.frame = frame.copy()

        sentiment_frame = frame.loc[frame["entry_type"].eq("sentiment")].copy()
        sentiment_frame["polarity"] = sentiment_frame["polarity"].astype(int)
        sentiment_frame["weight"] = sentiment_frame["weight"].astype(float)
        self.entries: dict[tuple[str, ...], LexiconEntry] = {}
        for row in sentiment_frame.itertuples(index=False):
            term_tokens = tuple(tokenize(row.term))
            if term_tokens:
                self.entries[term_tokens] = LexiconEntry(
                    term=str(row.term),
                    tokens=term_tokens,
                    language=str(row.language),
                    polarity=int(row.polarity),
                    weight=float(row.weight),
                    aspect=str(row.aspect),
                )
        self.max_sentiment_phrase = max(map(len, self.entries), default=1)

        aspect_frame = frame.loc[frame["entry_type"].eq("aspect")]
        self.aspect_entries = {
            tuple(tokenize(row.term)): str(row.aspect)
            for row in aspect_frame.itertuples(index=False)
            if tokenize(row.term)
        }
        self.max_aspect_phrase = max(map(len, self.aspect_entries), default=1)

    @staticmethod
    def _longest_match(tokens, start, lookup, max_phrase):
        for length in range(min(max_phrase, len(tokens) - start), 0, -1):
            key = tuple(tokens[start : start + length])
            if key in lookup:
                return key, lookup[key]
        return None, None

    def analyse(self, text: str) -> dict:
        text = str(text)
        token_matches = list(TOKEN_PATTERN.finditer(text))
        tokens = [m.group().lower() for m in token_matches]
        clause_ids = []
        clause = 0
        for position, match in enumerate(token_matches):
            gap = text[token_matches[position - 1].end():match.start()] if position else ""
            if CLAUSE_BREAKS.search(gap) or tokens[position] in CONTRAST_WORDS:
                clause += 1
            clause_ids.append(clause)
        matches = []
        sentiment_positions = set()
        aspect_positions = set()
        signed_scores = []

        index = 0
        while index < len(tokens):
            phrase_limit = min(self.max_sentiment_phrase, len(tokens) - index)
            while clause_ids[index + phrase_limit - 1] != clause_ids[index]:
                phrase_limit -= 1
            key, entry = self._longest_match(
                tokens, index, self.entries, phrase_limit
            )
            if entry is None:
                index += 1
                continue

            context = [tokens[j] for j in range(max(0, index - 3), index)
                       if clause_ids[j] == clause_ids[index]]
            multiplier = 1.0
            phrase_contains_negator = any(token in NEGATORS for token in key)
            negated = (not phrase_contains_negator) and any(
                token in NEGATORS for token in context
            )
            if any(token in INTENSIFIERS for token in context[-2:]):
                multiplier *= 1.5
            if any(token in DOWNTONERS for token in context[-2:]):
                multiplier *= 0.75

            polarity = -entry.polarity if negated else entry.polarity
            signed_scores.append(polarity * entry.weight * multiplier)
            sentiment_positions.update(range(index, index + len(key)))
            matches.append(
                {
                    "term": entry.term,
                    "polarity": polarity,
                    "weight": round(entry.weight * multiplier, 4),
                    "negated": negated,
                    "aspect": entry.aspect,
                }
            )
            index += len(key)

        aspects = []
        index = 0
        while index < len(tokens):
            key, aspect = self._longest_match(
                tokens, index, self.aspect_entries, self.max_aspect_phrase
            )
            if aspect is None:
                index += 1
                continue
            aspects.append(aspect)
            aspect_positions.update(range(index, index + len(key)))
            index += len(key)

        eligible_positions = {
            position
            for position, token in enumerate(tokens)
            if token not in COVERAGE_FUNCTION_WORDS
            and token not in COVERAGE_PLACEHOLDERS
        }
        # Only sentiment evidence opens the fusion gate. Aspect words remain
        # in the denominator and are still extracted as evaluation targets.
        matched_content_positions = sentiment_positions & eligible_positions
        content_token_count = len(eligible_positions)
        matched_content_token_count = len(matched_content_positions)

        score = 0.0
        if signed_scores:
            # A fixed monotonic bound retains intensity and lexicon weight.
            # Dividing by adjusted weights would cancel them for one match.
            # The scale (1.0) is fixed, not selected on validation labels.
            score = float(np.tanh(np.mean(signed_scores)))
        probabilities = ontology_probabilities(score, bool(matches))
        return {
            "ontology_score": score,
            "coverage": matched_content_token_count / max(content_token_count, 1),
            "aspect_coverage": len(aspect_positions & eligible_positions) / max(content_token_count, 1),
            "combined_coverage": len((sentiment_positions | aspect_positions) & eligible_positions) / max(content_token_count, 1),
            "content_token_count": content_token_count,
            "matched_content_token_count": matched_content_token_count,
            "match_count": len(matches),
            "matches": matches,
            "aspects": sorted(set(aspects)),
            "ontology_prob_negative": probabilities[0],
            "ontology_prob_neutral": probabilities[1],
            "ontology_prob_positive": probabilities[2],
        }

    def analyse_frame(self, frame: pd.DataFrame, text_column="tweet_text") -> pd.DataFrame:
        result = pd.DataFrame(
            [self.analyse(text) for text in frame[text_column].astype(str)]
        )
        result["matches"] = result["matches"].map(json.dumps)
        result["aspects"] = result["aspects"].map(json.dumps)
        return result


def ontology_probabilities(score: float, has_match: bool, smoothing: float = 0.10):
    """Map score to a smoothed [negative, neutral, positive] prior."""
    if not has_match:
        raw = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        clipped = float(np.clip(score, -1, 1))
        raw = np.array(
            [max(-clipped, 0.0), 1.0 - abs(clipped), max(clipped, 0.0)],
            dtype=float,
        )
    return (1.0 - smoothing) * raw + smoothing / 3.0


def fuse_probabilities(xlmr_probabilities, ontology_probabilities_array, alpha: float):
    """Apply ungated late fusion to every sample."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    xlmr = np.asarray(xlmr_probabilities, dtype=float)
    ontology = np.asarray(ontology_probabilities_array, dtype=float)
    if xlmr.shape != ontology.shape or xlmr.ndim != 2 or xlmr.shape[1] != 3:
        raise ValueError("Both probability arrays must have shape (n_samples, 3)")
    fused = (1.0 - alpha) * xlmr + alpha * ontology
    return fused / fused.sum(axis=1, keepdims=True)


def coverage_aware_fuse_probabilities(
    xlmr_probabilities,
    ontology_probabilities_array,
    coverage,
    match_count,
    alpha: float,
    threshold: float,
):
    """Fuse only rows with a sentiment match and sufficient ontology coverage."""
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    xlmr = np.asarray(xlmr_probabilities, dtype=float)
    ontology = np.asarray(ontology_probabilities_array, dtype=float)
    coverage = np.asarray(coverage, dtype=float)
    match_count = np.asarray(match_count)
    if xlmr.shape != ontology.shape or xlmr.ndim != 2 or xlmr.shape[1] != 3:
        raise ValueError("Both probability arrays must have shape (n_samples, 3)")
    if coverage.shape != (len(xlmr),) or match_count.shape != (len(xlmr),):
        raise ValueError("coverage and match_count must have one value per sample")

    gate = (match_count > 0) & (coverage >= float(threshold))
    fused = xlmr.copy()
    if gate.any():
        fused[gate] = fuse_probabilities(xlmr[gate], ontology[gate], alpha)
    return fused, gate


def write_owl(lexicon_path: str | Path, output_path: str | Path):
    """Export RDF/XML individuals mirroring the frozen CSV vocabulary."""
    frame = pd.read_csv(lexicon_path)
    rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    rdfs = "http://www.w3.org/2000/01/rdf-schema#"
    owl = "http://www.w3.org/2002/07/owl#"
    base = "http://example.org/fyp/product-sentiment#"
    ET.register_namespace("rdf", rdf)
    ET.register_namespace("rdfs", rdfs)
    ET.register_namespace("owl", owl)
    ET.register_namespace("ps", base)
    root = ET.Element(f"{{{rdf}}}RDF")
    ontology = ET.SubElement(root, f"{{{owl}}}Ontology", {f"{{{rdf}}}about": base})
    ET.SubElement(ontology, f"{{{rdfs}}}label").text = "English-Malay Product Sentiment Ontology"

    for class_name in ["SentimentExpression", "ProductAspect", "Polarity", "Language"]:
        node = ET.SubElement(root, f"{{{owl}}}Class", {f"{{{rdf}}}about": base + class_name})
        ET.SubElement(node, f"{{{rdfs}}}label").text = class_name

    used_ids = set()
    for number, row in enumerate(frame.itertuples(index=False), start=1):
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", str(row.term)).strip("_").lower()
        identifier = f"{row.entry_type}_{safe}"
        if identifier in used_ids:
            identifier = f"{identifier}_{number}"
        used_ids.add(identifier)
        node = ET.SubElement(
            root,
            f"{{{owl}}}NamedIndividual",
            {f"{{{rdf}}}about": base + identifier},
        )
        class_name = "SentimentExpression" if row.entry_type == "sentiment" else "ProductAspect"
        ET.SubElement(node, f"{{{rdf}}}type", {f"{{{rdf}}}resource": base + class_name})
        ET.SubElement(node, f"{{{rdfs}}}label").text = str(row.term)
        ET.SubElement(node, f"{{{base}}}language").text = str(row.language)
        ET.SubElement(node, f"{{{base}}}aspect").text = str(row.aspect)
        if row.entry_type == "sentiment":
            ET.SubElement(node, f"{{{base}}}polarity").text = str(int(row.polarity))
            ET.SubElement(node, f"{{{base}}}weight").text = str(float(row.weight))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)

