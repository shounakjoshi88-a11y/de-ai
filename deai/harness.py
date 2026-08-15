from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from .detector import scan
from .stats import compute_stats
from .watermark_probe import probe as _watermark_probe

_WORD = re.compile(r"\S+")
_LOG = Path(__file__).parent.parent / "calibration.jsonl"

# External detectors we can't call locally (proprietary APIs, no keys).
# Their real scores are recorded by hand via `record` / CLI --log and replayed
# in reports so the trend is tracked even though local signals are proxies.
_EXTERNAL_DETECTORS = ("zerogpt", "gptzero", "quillbot", "originality", "turnitin")


def _zipf_or_none():
    try:
        from wordfreq import zipf_frequency  # type: ignore

        return zipf_frequency
    except Exception:
        return None


_zipf = _zipf_or_none()


def _word_surprise(text: str) -> float:
    """Mean per-token 'unpredictability' using wordfreq zipf as a stand-in
    for LM perplexity: AI text clings to common words (low surprise), human
    text reaches for rarer ones (high surprise). Requires wordfreq; without
    it returns 0.0 (unknown)."""
    if _zipf is None:
        return 0.0
    words = [w.lower() for w in _WORD.findall(text) if w.isalpha()]
    if not words:
        return 0.0
    scores = []
    for w in words:
        z = _zipf(w, "en")
        if z > 0:
            scores.append(max(0.0, 7.0 - z))
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def _normalize(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def score(text: str, profile: dict | None = None) -> dict:
    """Local proxy detector set. Each family maps to 0..1 (1 = 'reads AI').

    These are *proxies* for the proprietary services — the real calibration
    lives in `record`/`report` where hand-entered ZeroGPT/GPTZero/Quillbot
    scores are logged against the same text.
    """
    stats = compute_stats(text)
    n_words = max(1, stats["words"])

    tells = scan(text)
    tell_density = 1000 * len(tells) / n_words

    surprise = _word_surprise(text)
    burst = stats["burstiness"]
    wm = _watermark_probe(text) or {}

    # family scores, each 0..1 where 1 is the AI-leaning end
    families = {
        # high surprise (rare, varied words) is human-leaning -> invert
        "perplexity": 1.0 - _normalize(surprise, 0.0, 3.0),
        # human prose is bursty (cv ~ 0.85-1.3+); flat text is AI-leaning
        "burstiness": 1.0 - _normalize(burst, 0.0, 1.2),
        "tell_density": _normalize(tell_density, 0.0, 25.0),
    }
    if wm.get("after"):
        families["watermark"] = _normalize(wm["after"].get("z", 0.0), 0.0, 8.0)

    profile_score = None
    if profile:
        from .profile import profile_distance

        d = profile_distance(text, profile)
        profile_score = {"distance": d["distance"], **d}

    # composite: equal weight over the available families
    weights = [v for v in families.values()]
    composite = round(sum(weights) / max(1, len(weights)), 3) if weights else 0.0

    return {
        "ai_score": composite,
        "families": families,
        "raw": {
            "word_surprise": surprise,
            "burstiness": burst,
            "tell_density": round(tell_density, 2),
            "watermark": wm.get("after"),
        },
        "profile": profile_score,
        "note": (
            "local proxy set — real services need calibration.jsonl entries"
        ),
    }


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def record(text: str, scores: dict[str, float], profile: dict | None = None) -> dict:
    """Log a hand-entered external detector score set against this text."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": _fingerprint(text),
        "words": len(_WORD.findall(text)),
        "scores": {k: float(v) for k, v in scores.items() if k in _EXTERNAL_DETECTORS},
    }
    if profile:
        from .profile import profile_distance

        entry["profile_distance"] = profile_distance(text, profile)["distance"]
    _LOG.parent.mkdir(parents=True, exist_ok=True)
    with _LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def report(text: str, profile: dict | None = None) -> dict:
    """Full report: local proxy scores + the latest external calibration
    logged for this exact text (or the global trend if none yet)."""
    local = score(text, profile)
    fp = _fingerprint(text)
    history: list[dict] = []
    if _LOG.exists():
        for line in _LOG.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("fingerprint") == fp:
                history.append(e)
    external = history[-1]["scores"] if history else None
    external_seen = len(history)
    return {
        "fingerprint": fp,
        "local": local,
        "external": external,
        "external_entries": external_seen,
        "external_trend": [
            {"ts": e["ts"], "scores": e["scores"]} for e in history[-6:]
        ],
    }


def build_profile_cli(corpus: list[str] | None = None) -> dict:
    from .profile import build_profile

    if corpus is None:
        corpus = []
    return build_profile(corpus)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .profile import load_profile

    ap = argparse.ArgumentParser(prog="deai.harness", description="de-ai detector harness")
    ap.add_argument("text", help="path to a text file to score")
    ap.add_argument("--profile", help="path to a writer profile JSON")
    ap.add_argument("--log", nargs="*", default=[], metavar="DETECTOR=SCORE",
                    help="record external scores, e.g. --log zerogpt=0.7 gptzero=0.2")
    args = ap.parse_args(argv)

    text = Path(args.text).read_text(encoding="utf-8")
    profile = load_profile(args.profile) if args.profile else None

    if args.log:
        scores = {}
        for item in args.log:
            if "=" in item:
                k, v = item.split("=", 1)
                scores[k.strip().lower()] = float(v)
        entry = record(text, scores, profile)
        print("recorded:", json.dumps(entry, indent=1))

    rep = report(text, profile)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())