from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

_WORD = re.compile(r"\S+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"\u201C\u2018])|(?<=[.!?])\n\n")
_BUCKETS = [(1, 4), (5, 9), (10, 14), (15, 19), (20, 29), (30, 39), (40, 10**9)]
_STOP = frozenset(
    """the a an and or but if then so for of to in on at by with from up down over
    under out off again further once here there when where why how all any both each
    few more most other some such no nor not only own same than too very just can will
    would could should may might must is are was were be been being do does did have has
    had having he she it they we you i me him her them us his hers their our your my its
    this that these those as into than while as like of at by for with about against
    between through during before after above below to from up down in out on off
    over under again further then once here there when where why how all any both each
    few more most other some such no nor not only own same so than too very can will
    just don't aren't isn't wasn't weren't haven't hasn't hadn't won't wouldn't can't
    couldn't didn't doesn't should couldn't'""".split()
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _len_bucket_hist(sent_lens: list[int]) -> dict[str, float]:
    n = max(1, len(sent_lens))
    hist: dict[str, float] = {}
    for lo, hi in _BUCKETS:
        label = f"{lo}-{hi if hi < 10**9 else '+'}"
        hist[label] = sum(1 for x in sent_lens if lo <= x <= hi) / n
    return hist


def build_profile(texts: list[str]) -> dict:
    """Fingerprint a writer: sentence rhythm, lexicon, and pacing stats."""
    joined = "\n\n".join(texts)
    words = _WORD.findall(joined)
    sent_lens = [max(1, len(_WORD.findall(s))) for s in _sentences(joined)]
    n_sent = max(1, len(sent_lens))
    mean = sum(sent_lens) / n_sent
    variance = sum((x - mean) ** 2 for x in sent_lens) / n_sent
    std = math.sqrt(variance)
    cv = std / mean if mean else 0.0

    content = [
        w.lower()
        for w in words
        if len(w) >= 4 and w.lower() not in _STOP and w[0].isalpha()
    ]
    freq = Counter(content)
    top_words = freq.most_common(400)
    top_pct = {w: 1e4 * c / max(1, len(content)) for w, c in top_words}

    starts = Counter(
        s.split()[0].lower().strip("\"'“”") for s in _sentences(joined) if s.split()
    )
    start_pct = {w: 100 * c / n_sent for w, c in starts.most_common(12)}

    paras = [p for p in joined.split("\n\n") if p.strip()]
    para_lens = [len(_WORD.findall(p)) for p in paras] if paras else [0]
    para_mean = sum(para_lens) / max(1, len(para_lens))

    ly = [
        w
        for w in words
        if w.lower().endswith("ly")
        and w.lower() not in {"only", "early", "family", "friendly", "holy", "wholly", "worldly"}
    ]
    return {
        "n_words": len(words),
        "n_sentences": len(sent_lens),
        "avg_sentence_len": round(mean, 2),
        "sentence_std": round(std, 2),
        "burstiness": round(cv, 3),
        "len_buckets": _len_bucket_hist(sent_lens),
        "top_content_words": top_pct,
        "top_starts": start_pct,
        "ly_adverb_per_1k": round(1000 * len(ly) / max(1, len(words)), 1),
        "para_len_mean": round(para_mean, 1),
    }


def save_profile(profile: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(profile, indent=1), encoding="utf-8")


def load_profile(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _smoothed_hist(hist: dict[str, float]) -> dict[str, float]:
    total = sum(hist.values()) or 1.0
    return {k: (v + 0.02) / (total + 0.02 * len(hist)) for k, v in hist.items()}


def _kl(hist_a: dict[str, float], hist_b: dict[str, float]) -> float:
    a, b = _smoothed_hist(hist_a), _smoothed_hist(hist_b)
    return sum(a[k] * math.log(a[k] / b.get(k, 0.02)) for k in a)


def profile_distance(text: str, profile: dict) -> dict:
    """How far *text* sits from the writer's profile. 0 = indistinguishable."""
    words = _WORD.findall(text)
    sent_lens = [max(1, len(_WORD.findall(s))) for s in _sentences(text)]
    n_sent = max(1, len(sent_lens))
    mean = sum(sent_lens) / n_sent
    variance = sum((x - mean) ** 2 for x in sent_lens) / n_sent
    std = math.sqrt(variance)
    cv = std / mean if mean else 0.0

    content = [
        w.lower()
        for w in words
        if len(w) >= 4 and w.lower() not in _STOP and w[0].isalpha()
    ]
    text_words = {w for w, _ in Counter(content).most_common(400)}
    prof_words = set(profile["top_content_words"])
    overlap = len(text_words & prof_words) / max(1, len(prof_words))

    hist_d = _kl(profile["len_buckets"], _len_bucket_hist(sent_lens))
    cv_d = abs(cv - profile["burstiness"]) / max(1e-6, profile["burstiness"])
    mean_d = abs(mean - profile["avg_sentence_len"]) / max(1e-6, profile["avg_sentence_len"])
    dist = min(1.0, 0.5 * hist_d + 0.3 * cv_d + 0.2 * mean_d)
    return {
        "distance": round(dist, 3),
        "hist_kl": round(hist_d, 3),
        "cv_delta": round(cv_d, 3),
        "mean_delta": round(mean_d, 3),
        "vocab_overlap": round(overlap, 3),
        "text_burstiness": round(cv, 3),
        "text_avg_len": round(mean, 1),
    }