from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"\S+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"\u201C\u2018])|(?<=[.!?])\n\n")
_AVOID_ADVERB = {"only", "early", "family", "friendly", "holy", "wholly", "worldly"}
_BUCKETS = [(1, 4), (5, 9), (10, 14), (15, 19), (20, 29), (30, 39), (40, 10**9)]


def compute_stats(text: str) -> dict:
    words = _WORD.findall(text)
    n_words = len(words)

    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    n_sent = len(sentences)
    sent_lens = [max(1, len(_WORD.findall(s))) for s in sentences]

    mean = sum(sent_lens) / n_sent if n_sent else 0.0
    variance = sum((x - mean) ** 2 for x in sent_lens) / n_sent if n_sent else 0.0
    std = math.sqrt(variance)
    cv = round(std / mean, 3) if mean else 0.0

    starts = Counter(s.split()[0].lower().strip("\"'\u201C\u201D") for s in sentences if s.split())
    top_starts = starts.most_common(8)
    start_repeat = round(100 * (top_starts[0][1] / n_sent), 1) if top_starts and n_sent else 0.0

    ly = [w for w in words if w.lower().endswith("ly") and w.lower() not in _AVOID_ADVERB]
    adverb_per_1k = round(1000 * len(ly) / n_words, 1) if n_words else 0.0

    passive = re.findall(r"\b(was|were|been)\b\s+\w+ed\b", text, re.IGNORECASE)
    passive_per_1k = round(1000 * len(passive) / n_words, 1) if n_words else 0.0

    lowered = [w.lower() for w in words]
    unique_ratio = round(100 * len(set(lowered)) / n_words, 1) if n_words else 0.0

    paragraphs = [p for p in text.split("\n") if p.strip()]

    len_buckets = [
        {"label": f"{lo}-{hi if hi < 10**9 else '+'}", "count": sum(1 for x in sent_lens if lo <= x <= hi)}
        for lo, hi in _BUCKETS
    ]

    burst_label = "machine-flat" if cv and cv < 0.85 else ("human-range" if cv else "n/a")

    return {
        "words": n_words,
        "sentences": n_sent,
        "paragraphs": len(paragraphs),
        "avg_sentence_len": round(mean, 1),
        "sentence_std": round(std, 1),
        "burstiness": cv,
        "burst_label": burst_label,
        "long_sentences": sum(1 for x in sent_lens if x >= 35),
        "short_sentences": sum(1 for x in sent_lens if x <= 6),
        "len_buckets": len_buckets,
        "top_starts": [{"word": w, "count": c} for w, c in top_starts],
        "start_repeat_pct": start_repeat,
        "adverb_per_1k": adverb_per_1k,
        "passive_per_1k": passive_per_1k,
        "unique_ratio": unique_ratio,
    }