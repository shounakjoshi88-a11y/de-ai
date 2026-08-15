from __future__ import annotations

import re

from .detector import Match, scan

_ALPHA = re.compile(r"[A-Za-z\u00C0-\u024F]")


def _capitalize_next(text: str, from_idx: int) -> str:
    m = _ALPHA.search(text, from_idx)
    if not m:
        return text
    return text[: m.start()] + m.group(0).upper() + text[m.end() :]


def _apply_one(text: str, match: Match) -> str:
    if match.fix == "dash_period":
        after = text[match.end : match.end + 8]
        word = re.match(r"\s*([A-Za-z]+)", after)
        bridge = word and word.group(1).lower() in (
            "and", "but", "so", "still", "yet", "then", "however",
            "moreover", "furthermore", "although", "though", "while",
        )
        if not bridge and after.lstrip()[:1].islower():
            return text[: match.start] + ", " + text[match.end :]
        text = text[: match.start] + ". " + text[match.end :]
        return _capitalize_next(text, match.start + 2)
    if match.fix == "strip_opener":
        text = text[: match.start] + text[match.end :]
        return _capitalize_next(text, match.start)
    if match.fix == "remove_char":
        return text[: match.start] + text[match.end :]
    if match.fix == "collapse_spaces":
        return text[: match.start] + " " + text[match.end :]
    if match.fix == "semicolon_comma":
        return text[: match.start] + "," + text[match.end :]
    if match.fix == "strip_emphasis":
        head = text[match.start : match.start + 2]
        if head in ("**", "__", "~~"):
            return text[: match.start] + text[match.start + 2 : match.end - 2] + text[match.end :]
        return text[: match.start] + text[match.start + 1 : match.end - 1] + text[match.end :]
    return text


def apply_fixes(text: str, matches: list[Match] | None = None, fix_types: list[str] | None = None) -> dict:
    if matches is None:
        matches = scan(text)
    fixable = [m for m in matches if m.fix is not None]
    if fix_types is not None:
        fixable = [m for m in fixable if m.fix in fix_types]

    spans = [(m.start, m.end) for m in fixable]
    tokens = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    changed_token_idx: set[int] = set()

    fixed = text
    for m in sorted(fixable, key=lambda x: x.start, reverse=True):
        fixed = _apply_one(fixed, m)

    fixed_tokens = re.findall(r"\S+", fixed)
    total_tokens = len(tokens)
    if total_tokens:
        for idx, (ts, te) in enumerate(tokens):
            for s, e in spans:
                if ts < e and te > s:
                    changed_token_idx.add(idx)
                    break
        tokens_changed = len(changed_token_idx)
    else:
        tokens_changed = 0

    chars_total = max(len(text), 1)
    changed_chars = sum(max(0, min(e, len(text)) - s) for s, e in spans)
    char_pct = round(100 * changed_chars / chars_total, 1)
    token_pct = round(100 * tokens_changed / total_tokens, 1) if total_tokens else 0.0

    remaining = len([m for m in matches if m.fix is None])

    return {
        "text": fixed,
        "applied": len(fixable),
        "remaining_flags": remaining,
        "disruption": {
            "tokens_total": total_tokens,
            "tokens_changed": tokens_changed,
            "token_pct": token_pct,
            "char_pct": char_pct,
            "remaining_flags": remaining,
            "note": (
                "Anthropic's Claude watermark (SynthID-Text, live since Aug 2026) is a "
                "statistical bias in token choices. The token_pct above is the share of "
                "tokens your edits actually replaced - the honest proxy for how much the "
                "mark degrades. Heavy rewrite of flagged zones (constructions, tic words) "
                "is the strongest lever; light touch-ups barely move it."
            ),
        },
    }