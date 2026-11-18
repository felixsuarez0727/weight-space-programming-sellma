"""
Evaluation harness for Swedish Language Models and Regression Controls.

Computes:
1. Swedish Compliance Rate (% of outputs purely in Swedish without leakage).
2. English Leakage Rate (% of outputs switching to English).
3. Keyword / Concept Recall (% of domain keywords correctly retrieved).
4. Repetition Loop Detection (pathological degeneracies).
5. Cross-Lingual English Regression Rate (checking preservation of general abilities).
"""

import re
from typing import Any
import numpy as np


SWEDISH_STOPWORDS = {
    "och", "att", "det", "som", "i", "på", "är", "av", "för", "med", "till",
    "den", "har", "de", "inte", "om", "ett", "en", "men", "eller", "när",
    "var", "hur", "så", "kan", "vi", "man", "ska", "skulle", "från", "sig",
    "under", "där", "efter", "alla", "dessa", "bland", "mellan", "finns",
    "vara", "blir", "blev", "hade", "också", "mer", "mycket", "mot", "utan",
    "då", "sedan", "genom", "vid", "bara", "vilket", "vilken", "vilka", "vad"
}

ENGLISH_STOPWORDS = {
    "the", "is", "are", "in", "and", "to", "of", "that", "it", "with",
    "as", "for", "was", "on", "at", "by", "this", "have", "from", "be",
    "or", "an", "which", "you", "we", "they", "sorry", "translate",
    "language", "english", "swedish", "cannot", "don't", "please", "into",
    "would", "should", "could", "their", "there", "about", "what", "how"
}

LEAKAGE_PHRASES = [
    "i'm sorry", "i cannot", "translate it into", "in english",
    "i don't know how to say", "as an ai", "i speak english",
    "can you please translate"
]


def tokenize_words(text: str) -> list[str]:
    """Extract lowercase alphanumeric word tokens."""
    return re.findall(r"\b[a-zA-ZåäöÅÄÖéÉüÜ]+\b", text.lower())


def detect_language_stats(text: str) -> dict[str, float]:
    """Calculate token ratios for Swedish and English stop words."""
    words = tokenize_words(text)
    if not words:
        return {"sv_ratio": 0.0, "en_ratio": 0.0, "has_sv_chars": 0.0, "word_count": 0}

    sv_count = sum(1 for w in words if w in SWEDISH_STOPWORDS)
    en_count = sum(1 for w in words if w in ENGLISH_STOPWORDS)
    
    # Check Swedish diacritics å, ä, ö
    has_sv_chars = 1.0 if bool(re.search(r"[åäöÅÄÖ]", text)) else 0.0

    return {
        "sv_ratio": sv_count / len(words),
        "en_ratio": en_count / len(words),
        "has_sv_chars": has_sv_chars,
        "word_count": len(words),
    }


def is_repetition_loop(text: str, n: int = 4, threshold: int = 3) -> bool:
    """Detect if any n-gram repeats more than threshold times."""
    words = tokenize_words(text)
    if len(words) < n:
        return False
    ngrams: dict[tuple, int] = {}
    for i in range(len(words) - n + 1):
        ng = tuple(words[i : i + n])
        ngrams[ng] = ngrams.get(ng, 0) + 1
        if ngrams[ng] >= threshold:
            return True
    return False


def evaluate_single_response(
    response: str,
    expected_language: str,
    keywords: list[str],
) -> dict[str, Any]:
    """Evaluate a single generated response."""
    stats = detect_language_stats(response)
    resp_lower = response.lower()

    # Check explicit leakage phrases
    has_leakage_phrase = any(phrase in resp_lower for phrase in LEAKAGE_PHRASES)

    if expected_language == "sv":
        # Compliant if predominantly Swedish and not heavily English
        is_compliant = (
            (stats["sv_ratio"] > 0.15 or stats["has_sv_chars"] == 1.0)
            and stats["en_ratio"] < 0.25
            and not has_leakage_phrase
            and stats["word_count"] >= 3
        )
        is_leakage = (
            stats["en_ratio"] > stats["sv_ratio"]
            or has_leakage_phrase
            or ("sorry" in resp_lower and "english" in resp_lower)
        )
    else:  # expected English
        is_compliant = (
            stats["en_ratio"] > 0.15
            and stats["word_count"] >= 3
            and stats["sv_ratio"] < 0.20
        )
        is_leakage = stats["sv_ratio"] > stats["en_ratio"]

    # Check keyword coverage (recall)
    keyword_hits = 0
    if keywords:
        for kw in keywords:
            if kw.lower() in resp_lower:
                keyword_hits += 1
        keyword_recall = keyword_hits / len(keywords)
    else:
        keyword_recall = 1.0

    loop_flag = is_repetition_loop(response)

    return {
        "is_compliant": bool(is_compliant),
        "is_leakage": bool(is_leakage),
        "keyword_recall": float(keyword_recall),
        "is_loop": bool(loop_flag),
        "sv_ratio": float(stats["sv_ratio"]),
        "en_ratio": float(stats["en_ratio"]),
        "word_count": int(stats["word_count"]),
    }


def evaluate_model_on_dataset(
    model,
    tokenizer,
    dataset: list[dict[str, Any]],
    system_prompt: str | None = None,
    max_new_tokens: int = 80,
    from_generate_fn = None,
) -> dict[str, Any]:
    """
    Run evaluation on an entire dataset, returning aggregate metrics and item-level details.
    """
    from src.model_loader import generate_response

    gen_fn = from_generate_fn or generate_response

    results = []
    category_metrics: dict[str, list[dict[str, Any]]] = {}

    for item in dataset:
        prompt = item["prompt"]
        expected_lang = item.get("expected_language", "sv")
        keywords = item.get("keywords", [])
        category = item.get("category", "general")

        response = gen_fn(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )

        eval_res = evaluate_single_response(response, expected_lang, keywords)
        record = {
            "id": item.get("id"),
            "category": category,
            "prompt": prompt,
            "response": response,
            "eval": eval_res,
        }
        results.append(record)

        if category not in category_metrics:
            category_metrics[category] = []
        category_metrics[category].append(eval_res)

    # Compute aggregates
    total = len(results)
    compliance_rate = sum(r["eval"]["is_compliant"] for r in results) / total if total > 0 else 0.0
    leakage_rate = sum(r["eval"]["is_leakage"] for r in results) / total if total > 0 else 0.0
    avg_keyword_recall = float(np.mean([r["eval"]["keyword_recall"] for r in results])) if total > 0 else 0.0
    loop_rate = sum(r["eval"]["is_loop"] for r in results) / total if total > 0 else 0.0
    avg_word_count = float(np.mean([r["eval"]["word_count"] for r in results])) if total > 0 else 0.0

    cat_breakdown = {}
    for cat, items in category_metrics.items():
        cat_breakdown[cat] = {
            "count": len(items),
            "compliance_rate": float(np.mean([i["is_compliant"] for i in items])),
            "leakage_rate": float(np.mean([i["is_leakage"] for i in items])),
            "avg_keyword_recall": float(np.mean([i["keyword_recall"] for i in items])),
        }

    return {
        "total_samples": total,
        "compliance_rate": float(compliance_rate),
        "leakage_rate": float(leakage_rate),
        "avg_keyword_recall": float(avg_keyword_recall),
        "loop_rate": float(loop_rate),
        "avg_word_count": float(avg_word_count),
        "category_breakdown": cat_breakdown,
        "records": results,
    }
