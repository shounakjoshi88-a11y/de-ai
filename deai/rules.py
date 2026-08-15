from __future__ import annotations

import re
from dataclasses import dataclass

RE_FLAGS = re.IGNORECASE | re.MULTILINE


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    name: str
    severity: str
    patterns: tuple[str, ...]
    fix: str | None = None
    exceptions: tuple[str, ...] = ()
    flags: int = RE_FLAGS

    def compiled(self) -> list[re.Pattern[str]]:
        return [re.compile(p, self.flags) for p in self.patterns]


RULES: tuple[Rule, ...] = (
    # ---- punctuation ----
    Rule(
        "em_dash",
        "punctuation",
        "Em dash",
        "warn",
        (r"\s*\u2014\s*",),
        fix="dash_period",
    ),
    Rule(
        "en_dash",
        "punctuation",
        "En dash",
        "info",
        (r"\u2013",),
    ),
    Rule(
        "semicolon",
        "punctuation",
        "Semicolon",
        "info",
        (r";",),
        fix="semicolon_comma",
    ),
    Rule(
        "double_space",
        "punctuation",
        "Double space",
        "warn",
        (r" {2,}",),
        fix="collapse_spaces",
    ),
    Rule(
        "zero_width",
        "hidden_chars",
        "Hidden / zero-width character",
        "error",
        (r"[\u200B-\u200F\u2060-\u2063\uFEFF]",),
        fix="remove_char",
    ),
    # ---- filler openers (auto-fixable) ----
    Rule(
        "filler_opener",
        "filler_openers",
        "AI filler opener",
        "warn",
        (
            r"^(Moreover|Furthermore|Additionally|Notably|Importantly|Interestingly|"
            r"In conclusion|To sum up|Needless to say|It goes without saying|"
            r"As we all know|At the end of the day)[,]?\s+",
            r"^(In addition|In recent years|In today'?s world)[,]?\s+",
            r"^(It is important to note|It'?s worth noting)[,]?\s+",
        ),
        fix="strip_opener",
    ),
    Rule(
        "filler_mid",
        "filler_openers",
        "Filler word mid-sentence",
        "info",
        (r"\b(moreover|furthermore|additionally)\b",),
    ),
    # ---- constructions (flag only) ----
    Rule(
        "not_but",
        "constructions",
        "Not X but Y",
        "warn",
        (r"\bnot\b[^.!?\n]{0,60}?\bbut\b",),
    ),
    Rule(
        "not_only",
        "constructions",
        "Not only X but also Y",
        "warn",
        (r"\bnot only\b[^.!?\n]{0,80}?\bbut( also)?\b",),
    ),
    Rule(
        "this_isnt",
        "constructions",
        "\"This isn't X, it's Y\"",
        "warn",
        (r"\bthis isn'?t\b[^.!?\n]{0,80}?\bit'?s\b",),
    ),
    Rule(
        "rule_of_three",
        "constructions",
        "Possible rule-of-three list",
        "info",
        (r"\b([A-Za-z][\w '’-]{0,30}?), ([A-Za-z][\w '’-]{0,30}?), and ([A-Za-z][\w '’-]{0,30}?)[.,;]",),
    ),
    Rule(
        "and_or_three",
        "constructions",
        "Repeated 'and' chain",
        "info",
        (r"\b([A-Za-z][\w '’-]{0,20}?) and ([A-Za-z][\w '’-]{0,20}?) and ([A-Za-z][\w '’-]{0,20}?)[.,]",),
    ),
    # ---- tic words (flag only) ----
    Rule(
        "tic_words",
        "tic_words",
        "AI vocabulary tic",
        "warn",
        (
            r"\b(seemed|seemingly|delve|delved|delving|tapestry|testament|pivotal|"
            r"meticulous|seamless|seamlessly|robust|leverage|foster|fosters|"
            r"showcase|showcases|underscore|underscores|underscored|realm|myriad|"
            r"plethora)\b",
            r"\b(in the realm of|a testament to)\b",
        ),
    ),
    # ---- hedge adverbs (flag only) ----
    Rule(
        "hedges",
        "hedges",
        "Hedge adverb",
        "warn",
        (r"\b(arguably|virtually|essentially|somewhat|literally|basically|simply|clearly|obviously|undoubtedly)\b",),
    ),
    Rule(
        "weak_adverbs",
        "hedges",
        "Weak intensifier",
        "info",
        (r"\b(quite|rather)\b",),
    ),
    # ---- exclamatory machinery (flag only) ----
    Rule(
        "exclamations",
        "exclamations",
        "AI exclamatory machinery",
        "warn",
        (r"\b(the answer is clear|it'?s that simple|the rest is history|as simple as that|there you have it)\b",),
    ),
    # ---- Anthropic-faq tells (flag only) ----
    Rule(
        "anthropic_faq",
        "anthropic_faq",
        "Anthropic-flagged tell",
        "warn",
        (r"\bquietly\b",),
    ),
    # ---- emoji (fixable: safe removal) ----
    Rule(
        "emoji",
        "emoji",
        "Emoji",
        "warn",
        (
            r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D\u2B00-\u2BFF"
            r"\u2194-\u21AA\u231A-\u23FA\u25AA-\u25FE]+",
        ),
        fix="remove_char",
        exceptions=("✓", "✕", "✎", "✗", "✔", "☑", "☐", "✖", "➤", "➜", "➔", "▸", "▹"),
    ),
    # ---- markdown emphasis (fixable: strip markers) ----
    Rule(
        "markdown_emphasis",
        "markdown",
        "Markdown emphasis",
        "warn",
        (
            r"\*{2}([^*\n]{1,80})\*{2}",
            r"(?<![\w*])\*([^*\n]*[A-Za-z][^*\n]*)\*(?![\w*])",
            r"(?<!\w)__([^_\n]{1,80})__(?!\w)",
            r"~~([^~\n]{1,80})~~",
        ),
        fix="strip_emphasis",
    ),
    # ---- conversational openers (fixable) ----
    Rule(
        "chat_opener",
        "filler_openers",
        "Conversational AI opener",
        "warn",
        (r"^(Honestly|Honestly speaking|To be honest|To be fair|Real talk|Frankly|Straight up|Straight-up|So basically)[,]\s+",),
        fix="strip_opener",
    ),
    # ---- intensifier stacks (flag only) ----
    Rule(
        "superlatives",
        "intensifiers",
        "AI intensifier stack",
        "warn",
        (r"\b(absolutely|totally|genuinely|insanely|wildly|ridiculously|completely|seriously|flat[- ]out)\b",),
    ),
    # ---- rhetorical pivots (flag only) ----
    Rule(
        "rhetorical_question",
        "rhetorical",
        "Rhetorical 'or did X' pivot",
        "info",
        (r"\bor did (they|you|he|she|we|it|anyone|anybody|everyone)\b[^!?\n]{0,80}\?",),
    ),
    # ---- story-beat setups (flag only) ----
    Rule(
        "from_to_now",
        "story_beats",
        "From X to Y, and now Z arc",
        "info",
        (r"\bfrom\b[^.!?\n]{0,60}\bto\b[^.!?\n]{0,80}\b(and now|and later|then finally|and then finally)\b",),
    ),
    Rule(
        "image_of",
        "story_beats",
        "'The image of…' setup",
        "info",
        (r"\bthe (image|thought|idea|vision) of\b",),
    ),
    # ---- chat slang (flag only) ----
    Rule(
        "chat_tic",
        "chat_tic",
        "Chat-slang tic",
        "warn",
        (r"\b(bro|bruh|lol|lmfao|nah|yep|nope|tbh|ngl|gonna|wanna|kinda|sorta|imma)\b",),
    ),
    # ---- sign-off bridges (flag only) ----
    Rule(
        "signoff_bridge",
        "signoffs",
        "AI sign-off bridge",
        "warn",
        (r"(?<!\w)(hope this helps|happy to help|glad to help|let me know if you (need|want|have|run into)|feel free to (reach out|ask)|with that in mind|that said,)(?=[\s.!?]|$)",),
    ),
    # ---- piled punctuation (flag only) ----
    Rule(
        "exclamation_pile",
        "exclamations",
        "Piled punctuation",
        "warn",
        (r"!{2,}|\?{2,}",),
    ),
    Rule(
        "ellipsis_tic",
        "exclamations",
        "Ellipsis tic",
        "info",
        (r"\.{3,}",),
    ),
    # ---- bullet lists (flag only) ----
    Rule(
        "list_ai",
        "list_ai",
        "Markdown bullet list",
        "info",
        (r"^\s*[-*]\s+\S+",),
    ),
)

CATEGORY_LABELS = {
    "punctuation": "Punctuation",
    "hidden_chars": "Hidden characters",
    "filler_openers": "Filler openers",
    "constructions": "AI constructions",
    "tic_words": "AI vocabulary tics",
    "hedges": "Hedges & weak adverbs",
    "exclamations": "Exclamatory machinery",
    "anthropic_faq": "Anthropic-flagged tells",
    "emoji": "Emoji",
    "markdown": "Markdown emphasis",
    "intensifiers": "Intensifier stacks",
    "rhetorical": "Rhetorical pivots",
    "story_beats": "Story beats",
    "chat_tic": "Chat-slang tics",
    "signoffs": "AI sign-offs",
    "list_ai": "Bullet lists",
}