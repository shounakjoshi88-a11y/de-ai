from __future__ import annotations

from dataclasses import dataclass

from .rules import RULES, Rule


@dataclass
class Match:
    rule_id: str
    category: str
    name: str
    severity: str
    line: int
    col: int
    start: int
    end: int
    match_text: str
    line_text: str
    fix: str | None = None

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "category": self.category,
            "name": self.name,
            "severity": self.severity,
            "line": self.line,
            "col": self.col,
            "match": self.match_text[:200],
            "line_text": self.line_text[:500],
            "fix": self.fix,
        }


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _line_of(starts: list[int], offset: int) -> int:
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo


def scan(text: str, rules: tuple[Rule, ...] = RULES) -> list[Match]:
    starts = _line_starts(text)
    matches: list[Match] = []
    for rule in rules:
        exceptions = {e.lower() for e in rule.exceptions}
        rule_matches: list[Match] = []
        for pattern in rule.compiled():
            for m in pattern.finditer(text):
                raw = m.group(0)
                if raw.strip().lower() in exceptions:
                    continue
                line_idx = _line_of(starts, m.start())
                line_start = starts[line_idx]
                line_end = text.find("\n", m.start())
                if line_end == -1:
                    line_end = len(text)
                line_text = text[line_start:line_end].strip()
                if not line_text:
                    continue
                col = m.start() - line_start
                rule_matches.append(
                    Match(
                        rule_id=rule.id,
                        category=rule.category,
                        name=rule.name,
                        severity=rule.severity,
                        line=line_idx + 1,
                        col=col,
                        start=m.start(),
                        end=m.end(),
                        match_text=raw,
                        line_text=line_text,
                        fix=rule.fix,
                    )
                )
        rule_matches.sort(key=lambda x: (x.start, -(x.end - x.start)))
        kept: list[Match] = []
        for m in rule_matches:
            if any(m.start >= k.start and m.end <= k.end for k in kept):
                continue
            kept.append(m)
        matches.extend(kept)
    matches.sort(key=lambda x: (x.line, x.col))
    return matches


def group_by_category(matches: list[Match]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for m in matches:
        grouped.setdefault(m.category, []).append(m.to_dict())
    return grouped