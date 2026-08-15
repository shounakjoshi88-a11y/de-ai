from __future__ import annotations

import difflib
import re
import warnings
from dataclasses import dataclass
from typing import Callable

from .detector import scan
from .fixers import _apply_one, _capitalize_next

RE_FLAGS = re.IGNORECASE | re.MULTILINE

Replacer = Callable[[re.Match[str]], str]

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import lemminflect as _lf
except Exception:  # pragma: no cover - engine degrades to built-in conjugation
    _lf = None

_VERB_SUF = {"e": "", "es": "s", "s": "s", "ed": "ed", "d": "d", "ing": "ing", "": ""}

_TAG_FROM_SUF = {
    "": "VB", "e": "VB", "y": "VB",
    "s": "VBZ", "es": "VBZ", "ies": "VBZ",
    "ed": "VBD", "d": "VBD", "ied": "VBD",
    "ing": "VBG", "ying": "VBG",
}


def _conjugate(bare: str, suf: str) -> str:
    if suf in ("ed", "d"):
        return bare + ("d" if bare.endswith("e") else "ed")
    if suf == "ing":
        return bare[:-1] + "ing" if bare.endswith("e") else bare + "ing"
    return bare + _VERB_SUF.get(suf, suf)


def _inflect(bare: str, tag: str) -> str:
    if _lf is not None:
        lemmas = _lf.getLemma(bare, "VERB")
        if lemmas and lemmas[0]:
            got = _lf.getInflection(lemmas[0], tag=tag)
            if got and got[0]:
                return got[0]
        got = _lf.getInflection(bare, tag=tag)
        if got and got[0]:
            return got[0]
    return _conjugate(bare, {"VB": "", "VBZ": "s", "VBD": "ed", "VBG": "ing"}[tag])


def _verb(bare: str, sufmap: dict[str, str] | None = None, tagmap: dict[str, str] | None = None) -> Replacer:
    table = dict(sufmap or {})
    tags = dict(_TAG_FROM_SUF, **(tagmap or {}))

    def repl(m: re.Match[str]) -> str:
        suf = m.group(1) or ""
        prev = re.search(r"(\w+)\s*$", m.string[: m.start()])
        if prev and prev.group(1).lower() in ("has", "have", "had"):
            return _inflect(bare, "VBN")
        if suf in table:
            return table[suf]
        return _inflect(bare, tags.get(suf, "VB"))

    return repl


@dataclass(frozen=True)
class Rewrite:
    id: str
    name: str
    pattern: str
    repl: str | Replacer
    flags: int = RE_FLAGS

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern, self.flags)


OPENERS = (
    "Moreover|Furthermore|Additionally|Notably|Importantly|Interestingly|"
    "In conclusion|To sum up|Needless to say|It goes without saying|"
    "As we all know|At the end of the day|In addition|In recent years|"
    "In today'?s world|Overall|All in all|What'?s more|On top of that|"
    "When all is said and done|In the final analysis|Ultimately|"
    "Ultimately speaking|In essence"
)

REWRITES: tuple[Rewrite, ...] = (
    # ---- filler openers: strip (line start and after a period) ----
    Rewrite("filler_opener", "Strip filler opener",
            rf"^({OPENERS})[,:]?\s+", ""),
    Rewrite("filler_opener_mid", "Strip filler opener (mid)",
            rf"(?<=[.!?] )({OPENERS})[,:]?\s+", ""),
    Rewrite(
        "world_family",
        "Strip 'in today's fast-paced world' family",
        r"(?:^|(?<=[.!?] ))in (?:today'?s|the|an|this|our) (?:ever-)?"
        r"(?:changing|evolving|fast[- ]paced|digital|modern) "
        r"(?:world|landscape|era|age)(?!\s+of\b)[,]?\s+",
        "",
    ),
    Rewrite(
        "moving_forward",
        "Strip 'moving forward,' bridge",
        r"(?:^|(?<=[.!?] ))(?:moving|going) forward[,]?\s+",
        "",
    ),
    Rewrite(
        "note_that",
        "Strip 'note that' bridge",
        r"\b(it is important to note|it'?s worth noting|it is worth noting|"
        r"it'?s worth mentioning|it is worth mentioning|it should be noted|"
        r"it is evident|it is clear|it is apparent|it goes without saying|"
        r"studies have shown|studies show|research shows|research has shown)"
        r"[,:]?\s+(?:that\s+)?",
        "",
    ),
    Rewrite(
        "goes_without_saying",
        "Strip 'it goes without saying'",
        r"(?<=[,.] )it goes without saying\s+that\s+",
        "",
    ),
    Rewrite(
        "chat_opener",
        "Strip conversational opener",
        r"^(Honestly|Honestly speaking|To be honest|To be fair|Real talk|"
        r"Frankly|Straight up|Straight-up|So basically|Quite frankly|"
        r"In all honesty)[,]\s+",
        "",
    ),
    Rewrite(
        "answer_clear",
        "Strip exclamatory machinery",
        r"\b(the answer is clear|it'?s that simple|the rest is history|"
        r"as simple as that|there you have it)[,:]?\s+",
        "",
    ),
    Rewrite(
        "no_surprise",
        "Strip 'no surprise' bridge",
        r"\b(it'?s no surprise that|it is no surprise that|"
        r"it comes as no surprise that|unsurprisingly)[,:]?\s+",
        "",
    ),
    Rewrite(
        "being_said",
        "Strip 'that being said' bridge",
        r"\b(with that being said|with that said|that being said|that said,)\s*",
        "",
    ),
    Rewrite(
        "simple_put",
        "Strip 'simply put,'",
        r"\b(?:simply put|put simply)[,]\s+",
        "",
    ),
    Rewrite(
        "whether_beginner",
        "Strip 'whether you're a beginner or an expert,'",
        r"\bWhether you'?re (?:a )?(?:beginner|novice|newbie|student) "
        r"or (?:an )?(?:expert|professional|veteran|pro), ",
        "",
    ),
    # ---- constructions: structural rewrite ----
    Rewrite(
        "not_only_but_also",
        "Not only X but also Y → X and Y",
        r"\bnot only\s+([^;.!?\n]{1,60}?)\s*,?\s+but also\s+([^;.!?\n]{1,60}?)([.,;!?]|$)",
        r"\1 and \2\3",
    ),
    Rewrite(
        "not_only_inv",
        "Not only did X, (but) Y also → X and Y",
        r"\bnot only did\s+(he|she|they|we|you|i)\s+([a-z]{2,30}?)\s*"
        r"(?:,\s+)?(?:but\s+)?(?:he|she|they|we|you|i)\s+also\s+"
        r"([a-z]{2,30}?)([.,;!?]|$)",
        lambda m: m.group(1).capitalize() + " " + _inflect(m.group(2), "VBD")
        + " and " + _inflect(m.group(3), "VBD") + (m.group(4) or ""),
    ),
    Rewrite(
        "not_only_be",
        "Not only is X adj, it is Y → X is adj and Y",
        r"\bnot only (is|are|was|were)\s+(he|she|it|they|we|you|i)\s+"
        r"([a-z]{2,30}?)\s*(?:,?\s+(?:but\s+)?)(?:he|she|it|they|we|you|i)?\s*"
        r"(?:also\s+)?(is|are|was|were)\s+([a-z]{2,30}?)([.,;!?]|$)",
        lambda m: m.group(2).capitalize() + " " + m.group(1) + " " + m.group(3)
        + (" and " + m.group(5) if m.group(1) == m.group(4) else
           " and " + m.group(4) + " " + m.group(5))
        + (m.group(6) or ""),
    ),
    Rewrite(
        "not_but",
        "Not X but Y → Y",
        r"\bnot\s+(just\s+)?([^,;.!?\n]{1,24}?)\s+but\s+([^,;.!?\n]{1,40}?)([.,;!?]|$)",
        lambda m: m.group(3) + (m.group(4) or "")
        if not re.search(r"\b(is|was|are|were|am|be|been|being|have|has|had|"
                          r"will|would|shall|should|could|can|may|might|must|"
                          r"do|does|did|get|gets|got|'s|'re|'ve|'ll|'d)\b", m.group(3) or "")
        else m.group(0),
    ),
    Rewrite(
        "this_isnt",
        "This isn't X, it's Y → It's Y",
        r"\bthis (?:isn'?t|is not)\s+(just\s+)?([^,;.!?\n]{1,40}?),?\s+"
        r"(?:it'?s|it is)\s+([^,;.!?\n]{1,40}?)([.,;!?]|$)",
        r"it's \3\4",
    ),
    # ---- AI vocabulary tics: verb swaps (lemminflect-conjugated) ----
    Rewrite("delve_into", "delve into → look into", r"\bdelve(s|d|ing)?\b(?=\s+into\b)", _verb("look")),
    Rewrite("delve_plain", "delve → dig", r"\bdelve(s|d|ing)?\b(?!\s+into\b)", _verb("dig")),
    Rewrite("utilize", "utilize → use", r"\butiliz(e|es|ed|ing)\b", _verb("use")),
    Rewrite("leverage", "leverage → use", r"\bleverage(s|d|ing)?\b", _verb("use")),
    Rewrite("showcase", "showcase → show", r"\bshowcas(e|es|ed|ing)\b", _verb("show")),
    Rewrite("underscore", "underscore → highlight", r"\bunderscor(e|es|ed|ing)\b", _verb("highlight")),
    Rewrite("foster", "foster → encourage", r"\bfoster(s|ed|ing)?\b", _verb("encourage")),
    Rewrite("facilitate", "facilitate → support", r"\bfacilitate(s|d|ing)?\b", _verb("support")),
    Rewrite("elucidate", "elucidate → explain", r"\belucidat(e|es|ed|ing)\b", _verb("explain")),
    Rewrite("embark", "embark on → start on", r"\bembark(s|ed|ing)?\b(?=\s+on\b)", _verb("start")),
    Rewrite("endeavor_to", "endeavor to → try to", r"\bendeavor(s|ed|ing)?\b(?=\s+to\b)", _verb("try")),
    Rewrite("endeavor_noun", "endeavor(s) → effort(s)", r"\bendeavor(s)?\b(?!\s+to\b)", r"effort\1"),
    Rewrite("encompass", "encompass → cover", r"\bencompass(e|es|ed|ing)\b", _verb("cover")),
    Rewrite("commence", "commence → start", r"\bcommenc(e|es|ed|ing)\b", _verb("start")),
    Rewrite("assist", "assist → help", r"\bassist(s|ed|ing)?\b", _verb("help")),
    Rewrite("bolster", "bolster → boost", r"\bbolster(s|ed|ing)?\b", _verb("boost")),
    Rewrite("streamline", "streamline → simplify", r"\bstreamlin(e|es|ed|ing)\b", _verb("simplify")),
    Rewrite("optimize", "optimize → improve", r"\boptimiz(e|es|ed|ing)\b", _verb("improve")),
    Rewrite("harness", "harness → use", r"\bharness(s|ed|ing)?\b", _verb("use")),
    Rewrite("elevate", "elevate → improve", r"\belevate(s|d|ing)?\b", _verb("improve")),
    Rewrite("expedite", "expedite → hasten", r"\bexpedit(e|es|ed|ing)\b", _verb("hasten")),
    Rewrite("ascertain", "ascertain → determine", r"\bascertain(s|ed|ing)?\b", _verb("determine")),
    Rewrite("propel", "propel → drive", r"\bpropel(s|led|ling)?\b", _verb("drive", tagmap={"s": "VBZ", "led": "VBD", "ling": "VBG"})),
    Rewrite("spearhead", "spearhead → lead", r"\bspearhead(s|ed|ing)?\b", _verb("lead")),
    Rewrite("orchestrate", "orchestrate → organize", r"\borchestrat(e|es|ed|ing)\b", _verb("organize")),
    Rewrite("unpack", "unpack → explain", r"\bunpack(s|ed|ing)?\b", _verb("explain")),
    Rewrite("usher_in", "usher in → bring in", r"\busher(s|ed|ing)?\b(?=\s+in\b)", _verb("bring")),
    Rewrite("revolutionize", "revolutionize → change", r"\brevolutioniz(e|es|ed|ing)\b", _verb("change")),
    Rewrite("boast", "boast → have", r"\bboast(s|ed|ing)?\b", _verb("have")),
    # guarded verb swaps: lookahead narrows the object to AI-typical targets
    Rewrite("address_guard", "address (the issue…) → handle",
            r"\baddress(e|es|ed|ing)?\b(?=\s+(?:the|this|these|those|a)\s+"
            r"(?:issue|problem|challenge|concern|question|need|gap|barrier|root\s+cause)s?\b)",
            _verb("handle")),
    Rewrite("navigate_guard", "navigate (the complexities…) → handle",
            r"\bnavigat(e|es|ed|ing)\b(?=\s+(?:the|these|those)\s+"
            r"(?:complexit(?:y|ies)|challenges?|hurdles?|obstacles?|waters?|landscape)\b)",
            _verb("handle")),
    Rewrite("unlock_guard", "unlock (the potential…) → release",
            r"\bunlock(s|ed|ing)?\b(?=\s+(?:the|its|their|our|your)\s+"
            r"(?:potential|possibilit(?:y|ies)|value|benefit(?:s)?|power|creativit(?:y|ies))\b)",
            _verb("release")),
    Rewrite("amplify_guard", "amplify (the impact…) → increase",
            r"\bamplif(y|ies|ied|ying)\b(?=\s+(?:the|its|their|our)\s+"
            r"(?:impact|reach|message|voice|visibility|awareness|effort|effect)s?\b)",
            _verb("increase", tagmap={"y": "VB", "ies": "VBZ", "ied": "VBD", "ying": "VBG"})),
    Rewrite("champion_guard", "champion (the cause…) → support",
            r"\bchampion(s|ed|ing)?\b(?=\s+(?:the|this|these|that|a|an)\b)",
            _verb("support")),
    Rewrite(
        "plays_role",
        "plays a key role in → is key to",
        r"\b(play|plays|played|playing) a (crucial|vital|key|important|significant|pivotal) role in\s+",
        lambda m: {"play": "is", "plays": "is", "played": "was", "playing": "being"}[m.group(1)]
        + " " + m.group(2) + " to ",
    ),
    # ---- AI vocabulary tics: adjective/noun swaps ----
    Rewrite("robust", "robust → strong", r"\brobust\b", "strong"),
    Rewrite("robustly", "robustly → strongly", r"\brobustly\b", "strongly"),
    Rewrite("robustness", "robustness → strength", r"\brobustness\b", "strength"),
    Rewrite("seamless", "seamless → smooth", r"\bseamless\b", "smooth"),
    Rewrite("seamlessly", "seamlessly → smoothly", r"\bseamlessly\b", "smoothly"),
    Rewrite("frictionless", "frictionless → smooth", r"\bfrictionless\b", "smooth"),
    Rewrite("meticulous", "meticulous → careful", r"\bmeticulous\b", "careful"),
    Rewrite("meticulously", "meticulously → carefully", r"\bmeticulously\b", "carefully"),
    Rewrite("pivotal", "pivotal → key", r"\bpivotal\b", "key"),
    Rewrite("crucial", "crucial → key", r"\bcrucial\b", "key"),
    Rewrite("intricate", "intricate → complex", r"\bintricate\b", "complex"),
    Rewrite("multifaceted", "multifaceted → complex", r"\bmultifaceted\b", "complex"),
    Rewrite("invaluable", "invaluable → valuable", r"\binvaluable\b", "valuable"),
    Rewrite("comprehensive", "comprehensive → complete", r"\bcomprehensive\b", "complete"),
    Rewrite("extensive", "extensive → broad", r"\bextensive\b", "broad"),
    Rewrite("innovative", "innovative → fresh", r"\binnovative\b", "fresh"),
    Rewrite("ever_evolving", "ever-evolving → changing", r"\bever[- ]evolving\b", "changing"),
    Rewrite("cutting_edge", "cutting-edge → the latest", r"\bcutting[- ]edge\b(?=\s+[A-Za-z])", "the latest"),
    Rewrite("state_of_art", "state-of-the-art → modern", r"\bstate[- ]of[- ]the[- ]art\b", "modern"),
    Rewrite("game_changing", "game-changing → major", r"\bgame[- ]changing\b", "major"),
    Rewrite("game_changer", "game-changer → big change", r"\bgame[- ]changers?\b", "big change"),
    Rewrite("paradigm_shift", "paradigm shift → big change", r"\bparadigm shift\b", "big change"),
    Rewrite("testament_to", "testament to → proof of", r"\b(?:a\s+)?testament to\b", "proof of"),
    Rewrite("tapestry_of", "tapestry of → mix of", r"\ba tapestry of\b", "a mix of"),
    Rewrite("realm_of", "a realm of → a world of", r"\ba realm of\s+", "a world of "),
    Rewrite("in_realm", "in the realm of → in", r"\bin the realm of\s+", "in "),
    Rewrite("realm_general", "realm → world", r"\brealm\b", "world"),
    Rewrite("landscape", "landscape → world", r"\blandscapes?\b", "world"),
    Rewrite("journey", "journey → path", r"\bjourneys?\b", "path"),
    Rewrite("cornerstone", "cornerstone → foundation", r"\bcornerstone\b", "foundation"),
    Rewrite("deep_dive", "a deep dive into → a close look at", r"\ba deep dive into\b", "a close look at"),
    Rewrite("takeaway", "takeaway → point", r"\b(?:key\s+)?takeaways?\b", "point"),
    Rewrite("seemingly", "seemingly → drop", r"\bseemingly[,]?\s+", ""),
    Rewrite("quietly", "quietly → drop", r"\bquietly[,]?\s+", ""),
    # ---- hedges: drop or soften ----
    Rewrite("hedge_drop", "Hedge adverb → drop",
            r"\b(arguably|literally|basically|simply|clearly|obviously|"
            r"undoubtedly|essentially|genuinely|truly)[,]?\s+(?!put\b)", ""),
    Rewrite("somewhat", "somewhat → a bit", r"\bsomewhat\s+", "a bit "),
    Rewrite("quite", "quite → drop", r"\bquite\s+", ""),
    Rewrite("rather", "rather → drop", r"(?<!\bwould )(?<!'d )\brather\s+(?!than\b)", ""),
    Rewrite("virtually", "virtually → nearly", r"\bvirtually\s+", "nearly "),
    Rewrite("honestly_trailing", "honestly → drop (after comma)",
            r"(?<=,)\s*honestly\b", ""),
    Rewrite("honestly_mid", "honestly → drop (filler)",
            r"\bhonestly\s+(?!about\b|with\b|to\b|said\b|say\b|says\b|speak\b|"
            r"speaks\b|spoke\b|spoken\b|tell\b|tells\b|told\b|answer\b|answers\b|"
            r"answered\b|reply\b|replied\b|ask\b|asked\b|explain\b|explained\b|"
            r"describe\b|described\b|report\b|reported\b|admit\b|admitted\b|"
            r"talk\b|talked\b)", ""),
    Rewrite("honestly_sentence", "honestly → drop (after period)",
            r"(?<=[.!?] )honestly[,]?\s+", ""),
    Rewrite("frankly_mid", "frankly → drop",
            r"(?<=[.!?] )frankly[,]?\s+|(?<=,)\s*frankly\b", ""),
    # ---- filler mid-sentence ----
    Rewrite("filler_mid", "moreover/furthermore → also",
            r"(?<![,;]\s)\b(moreover|furthermore|additionally)[,]?\s+", "also, "),
    Rewrite("clause_filler", "Strip comma-clause filler",
            r"([,;])\s*(moreover|furthermore|however|nonetheless|nevertheless|"
            r"additionally|consequently|notably|importantly|unsurprisingly|"
            r"surprisingly)\s*,",
            lambda m: "; " if m.group(1) == ";" else ""),
    # ---- wordy machinery ----
    Rewrite("in_order_to", "in order to → to", r"\bin order to\b", "to"),
    Rewrite("due_to_fact", "due to the fact that → because", r"\bdue to the fact that\b", "because"),
    Rewrite("in_effort_to", "in an effort to → to", r"\bin an effort to\b", "to"),
    Rewrite("in_bid_to", "in a bid to → to", r"\bin a bid to\b", "to"),
    Rewrite("in_attempt_to", "in an attempt to → to", r"\bin an attempt to\b", "to"),
    Rewrite("in_light_of", "in light of → because of", r"\bin light of\b", "because of"),
    Rewrite("result_of", "as a result of → because of", r"\bas a result of\b", "because of"),
    Rewrite("owing_to", "owing to → because of", r"\bow(?:ing|es) to\b", "because of"),
    Rewrite("with_respect_to", "with respect to → about", r"\bwith respect to\b", "about"),
    Rewrite("with_regard_to", "with regard to → about", r"\bwith regard to\b", "about"),
    Rewrite("pertaining_to", "pertaining to → about", r"\bpertaining to\b", "about"),
    Rewrite("end_of_day", "at the end of the day → in the end",
            r"\bat the end of the day[,]?\s+", "in the end, "),
    Rewrite("fast_paced", "in the fast-paced world of → in",
            r"\bin the fast[- ]paced world of\s+|\bin the world of\s+", "in "),
    Rewrite("when_it_comes_to", "when it comes to → regarding", r"\bwhen it comes to\s+", "regarding "),
    Rewrite("in_terms_of", "in terms of → regarding", r"\bin terms of\s+", "regarding "),
    Rewrite("at_its_core", "at its core → at heart", r"\bat its core[,]?\s+", "at heart, "),
    Rewrite("across_the_globe", "across the globe → worldwide", r"\bacross the globe\b", "worldwide"),
    Rewrite("whether_it_be", "whether it be → whether it's", r"\bwhether it be\b", "whether it's"),
    Rewrite("first_and_foremost", "First and foremost → First", r"\bFirst and foremost[,]?\s+", "First, "),
    Rewrite("last_but_not_least", "Last but not least → Finally", r"\bLast but not least[,]?\s+", "Finally, "),
    # ---- quantifiers ----
    Rewrite("myriad", "myriad of → many", r"\ba myriad of\s+|\bmyriad\s+", "many "),
    Rewrite("plethora", "plethora of → many", r"\ba plethora of\s+", "many "),
    Rewrite("numerous", "numerous → many", r"\bnumerous\b", "many"),
    Rewrite("host_of", "a host of → many", r"\ba host of\s+", "many "),
    Rewrite("wealth_of", "a wealth of → lots of", r"\ba wealth of\s+", "lots of "),
    Rewrite("abundance_of", "an abundance of → plenty of", r"\ban abundance of\s+", "plenty of "),
    Rewrite("multitude_of", "a multitude of → many", r"\ba multitude of\s+", "many "),
    Rewrite("slew_of", "a slew of → a lot of", r"\ba slew of\s+", "a lot of "),
    Rewrite("a_number_of", "a number of → several", r"\ba number of\s+", "several "),
    Rewrite("array_of", "array of → many", r"\ba (wide )?array of\s+|\ba wide range of\s+", "many "),
    # ---- conversational closers ----
    Rewrite(
        "lets_dive",
        "Let's dive in → Let's start / look at",
        r"\blet'?s (?:delve into|dive into)\s+",
        "let's look at ",
    ),
    Rewrite(
        "lets_dive_and",
        "Let's dive in and → Let's",
        r"\blet'?s (?:dive in|jump in|jump right in)\s+and\s+",
        "let's ",
    ),
    Rewrite(
        "lets_dive_standalone",
        "Let's dive in → Let's start",
        r"\blet'?s (?:dive in|jump in)\b(?!\s+and\b)",
        "let's start",
    ),
    # ---- story-beat setups: guarded rewrites (skip when they read human) ----
    Rewrite(
        "chat_slang_strip",
        "Strip sentence-start chat slang (Bro, Nah…)",
        r"^(?:Bro|Bruh|Lol|Nah|Ngl|Tbh|Yep|Nope)[,]\s+",
        "",
    ),
    Rewrite(
        "rule_of_three_comma",
        "Rule-of-three: drop Oxford comma",
        r"\b([A-Za-z][\w '’-]{0,30}?), ([A-Za-z][\w '’-]{0,30}?), and "
        r"([A-Za-z][\w '’-]{0,30}?)([.,;])",
        lambda m: m.group(1) + ", " + m.group(2) + " and " + m.group(3) + m.group(4)
        if not re.search(r"\band\b|\bor\b", m.group(2) + " " + m.group(3))
        and not re.search(r"\b(?:a|an|the) [a-z]+ of\b", m.group(2) + " " + m.group(3))
        and not re.match(r"(she|he|it|they|we|you|i|there|that)\b", m.group(3), re.I)
        and not re.search(r"\b(stood|went|came|said|sat|took|was|were|is|are|"
                          r"has|have|had|does|did)\b", m.group(3))
        else m.group(0),
    ),
    Rewrite(
        "image_of_rewrite",
        "'The image of X is Y' → 'It's Y to picture X'",
        r"\bThe image of\b([^.!?\n]{5,200}?) is "
        r"(hilarious|funny|wild|absurd|amusing|ridiculous|poignant|beautiful|sad|sweet)\.",
        r"It's \2 to picture\1.",
    ),
    Rewrite(
        "going_from_strip",
        "Strip 'Going from' opener",
        r"(?:^|(?<=[.!?]\s))Going from\b",
        "From ",
    ),
    Rewrite(
        "from_to_now",
        "From X to Y, and now Z → From X to Y, now Z",
        r"(\bfrom\b[^.!?\n]{0,60}\bto\b[^.!?\n]{0,80}),\s*and now\b",
        r"\1, now",
    ),
    Rewrite(
        "ellipsis_split",
        "Ellipsis → sentence split",
        r"(?<=\w)\.{3,}\s+([a-z])",
        lambda m: ". " + m.group(1).upper(),
    ),
    Rewrite(
        "superlative_drop",
        "Drop intensifier before absolutes",
        r"\b(completely|totally|absolutely|genuinely|truly)\s+"
        r"(different|new|unique|unnecessary|clear|obvious|worthless|useless)\b",
        r"\2",
    ),
    # ---- humanizer bank: over-formal tics that classifiers weight heavily ----
    Rewrite("ensure", "ensure → make sure",
            r"\bensure(s|d|ing)?\b",
            _verb("make sure", sufmap={
                "": "make sure", "s": "makes sure", "d": "made sure",
                "ed": "made sure", "ing": "making sure"})),
    Rewrite("significant", "significant → big", r"\bsignificant\b", "big"),
    Rewrite("vital", "vital/essential/paramount → key",
            r"\b(vital|essential|paramount|imperative)\b", "key"),
    Rewrite("various", "various → different", r"\bvarious\b", "different"),
    Rewrite("multiple", "multiple → several", r"\bmultiple\b", "several"),
    Rewrite("diverse", "diverse → different", r"\bdiverse\b", "different"),
    Rewrite("unprecedented", "unprecedented → unseen", r"\bunprecedented\b", "unseen"),
    Rewrite("demonstrate", "demonstrate → show",
            r"\bdemonstrat(e|es|ed|ing)\b", _verb("show")),
    Rewrite("highlight", "highlight → show", r"\bhighlight(s|ed|ing)?\b", _verb("show")),
    Rewrite("strive", "strive → try", r"\bstriv(e|es|ed|ing)\b", _verb("try")),
    Rewrite("intriguing", "intriguing/fascinating → interesting",
            r"\b(intriguing|fascinating)\b", "interesting"),
    Rewrite("profound", "profound → deep", r"\bprofound\b", "deep"),
    Rewrite("indeed_strip", "Strip 'indeed' opener",
            r"(?:^|(?<=[.!?] ))indeed[,]?\s+", ""),
    Rewrite("as_well_as", "as well as → and",
            r"(?<!, )\bas well as\b", "and"),
    Rewrite("in_addition_to", "in addition to → besides",
            r"\bin addition to\b", "besides"),
    Rewrite("not_to_mention", "not to mention → and",
            r"\bnot to mention\b", "and"),
    Rewrite("despite_fact", "despite/in spite of the fact that → although",
            r"\b(?:despite|in spite of) the fact that\b", "although"),
    Rewrite("in_spite_of", "in spite of → despite",
            r"\bin spite of\b", "despite"),
    Rewrite("safe_to_say", "Strip 'it is safe to say that'",
            r"\bit'?s safe to say that\s+|\bit is safe to say that\s+", ""),
    Rewrite("my_opinion", "Strip 'in my opinion / I believe that' openers",
            r"(?:^|(?<=[.!?] ))(?:in my opinion|from my perspective|"
            r"from my point of view|I believe that|I think that|"
            r"It is my belief that|it'?s my belief that)[,]?\s+", ""),
    Rewrite("modern_era", "Strip 'in recent times / the modern era'",
            r"(?:^|(?<=[.!?] ))in (?:recent times|this day and age|"
            r"the modern era|today'?s fast[- ]paced world)[,]?\s+", ""),
)

# rule ids whose tells are fully eliminated by the rewrite bank or mechanical fixes
COVERED: frozenset[str] = frozenset({
    "filler_opener", "filler_mid", "chat_opener", "hedges", "weak_adverbs",
    "exclamations", "anthropic_faq", "not_but", "not_only", "this_isnt",
    "emoji", "markdown_emphasis", "zero_width", "double_space", "em_dash",
})

MECHANICAL_FIXES: frozenset[str] = frozenset({
    "dash_period", "remove_char", "collapse_spaces", "strip_emphasis",
})


@dataclass
class Change:
    name: str
    before: str
    after: str
    line: int


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _at_sentence_start(text: str, idx: int) -> bool:
    if idx == 0:
        return True
    if text[idx - 1] == "\n":
        return True
    return idx >= 2 and text[idx - 2] in ".!?" and text[idx - 1] in " \n"


def _matched_tokens(a: list[str], b: list[str]) -> int:
    sm = difflib.SequenceMatcher(None, a, b, autojunk=True)
    return sum(block.size for block in sm.get_matching_blocks())


def paraphrase(text: str) -> dict:
    # 1. mechanical safe fixes (openers handled by the rewrite bank)
    matches = scan(text)
    mechanical = [m for m in matches if m.fix in MECHANICAL_FIXES]
    work = text
    mech_chars = 0
    for m in sorted(mechanical, key=lambda x: x.start, reverse=True):
        work = _apply_one(work, m)
        mech_chars += m.end - m.start

    # 2. rewrite bank over the fixed text
    candidates: list[tuple[int, int, Rewrite, re.Match[str], str]] = []
    for rw in REWRITES:
        for m in rw.compiled().finditer(work):
            after = rw.repl(m) if callable(rw.repl) else m.expand(rw.repl)
            if after == m.group(0):
                continue
            candidates.append((m.start(), m.end(), rw, m, after))

    candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))
    kept: list[tuple[int, int, Rewrite, re.Match[str], str]] = []
    for c in candidates:
        if any(c[0] >= k[0] and c[1] <= k[1] for k in kept):
            continue
        kept.append(c)

    changes: list[Change] = []
    for s, e, rw, m, after in kept:
        changes.append(Change(rw.name, m.group(0)[:80], after[:80], _line_of(work, s)))

    applied = work
    final_positions: list[int] = []
    delta = 0
    for s, e, rw, m, after in sorted(kept, key=lambda c: c[0]):
        final_positions.append(s + delta)
        delta += len(after) - (e - s)

    for s, e, rw, m, after in sorted(kept, key=lambda c: c[0], reverse=True):
        applied = applied[:s] + after + applied[e:]

    for pos in final_positions:
        if pos <= len(applied) and _at_sentence_start(applied, pos):
            applied = _capitalize_next(applied, pos)

    # 3. post-clean
    applied = re.sub(r";", ",", applied)
    applied = re.sub(r" {2,}", " ", applied)
    applied = re.sub(r"\s+([,.;:!?])", r"\1", applied)
    applied = re.sub(r"[,;:]\s*([.!?])", r"\1", applied)
    applied = re.sub(r"[ \t]+\n", "\n", applied).rstrip()

    # 4. stats
    tokens_orig = re.findall(r"\S+", text)
    tokens_new = re.findall(r"\S+", applied)
    total = len(tokens_orig)
    lcs = _matched_tokens(tokens_orig, tokens_new)
    changed = total - lcs
    chars_changed = mech_chars + sum(len(c.before) for c in changes)
    char_pct = round(100 * chars_changed / max(len(text), 1), 1)
    token_pct = round(100 * changed / total, 1) if total else 0.0

    remaining = len([m for m in scan(applied) if m.rule_id not in COVERED])

    return {
        "text": applied,
        "mechanical": len(mechanical),
        "rewrites": len(changes),
        "applied": len(mechanical) + len(changes),
        "changes": [
            {"name": c.name, "before": c.before, "after": c.after, "line": c.line}
            for c in sorted(changes, key=lambda c: c.line)
        ],
        "remaining_flags": remaining,
        "disruption": {
            "tokens_total": total,
            "tokens_changed": changed,
            "token_pct": token_pct,
            "char_pct": char_pct,
            "remaining_flags": remaining,
            "note": (
                "Deterministic paraphrase: the rewrite bank replaces flagged "
                "constructions, tic words, openers and hedges with plain human "
                "equivalents, then strips mechanical tells (emoji, markdown "
                "emphasis, dashes). token_pct is the share of tokens actually "
                "rewritten - the honest proxy for how far the Claude watermark "
                "degrades. Flag-only tells (slang, story beats) still need a "
                "hand rewrite for the deepest effect."
            ),
        },
    }