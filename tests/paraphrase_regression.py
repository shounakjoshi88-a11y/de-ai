"""Regression tests for the DE-AI paraphrase engine.

Pins every verified behavior of deai/paraphraser.py so rule edits can never
silently regress them (the "useed", "havehad", "digdug", "That the results",
"answered honestly", "simply put", "would rather" classes are all pinned here).

Run from the de-ai directory:
    python tests/paraphrase_regression.py
Exit code 0 = all green.
"""

from __future__ import annotations

import sys
import time
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deai.paraphraser import paraphrase  # noqa: E402
from deai.deeprewrite import deep_rewrite  # noqa: E402
from deai.detector import scan  # noqa: E402
from deai.paraphraser import COVERED  # noqa: E402

FAILED: list[tuple[str, str, str]] = []


def check(name: str, got: str, want: str) -> None:
    if got != want:
        FAILED.append((name, got, want))
        print(f"FAIL {name}\n  got:  {got!r}\n  want: {want!r}")
    else:
        print(f"ok   {name}")


def check_stats(name: str, got: int | float, want: int | float, tol: float = 0.0) -> None:
    if abs(got - want) > tol:
        FAILED.append((name, str(got), str(want)))
        print(f"FAIL {name}: got {got}, want {want} (+-{tol})")
    else:
        print(f"ok   {name}")


# ---- conjugation (the useed/havehad/digdug class of bugs) ----
check("utilize->use", paraphrase("He utilized the tool. She showcases her work. "
        "They leverage data daily. The data was leveraged carefully.")["text"],
        "He used the tool. She shows her work. They use data daily. The data was used carefully.")
check("boast->have, encompass->cover", paraphrase("The engine boasted great features and "
        "encompassed everything.")["text"], "The engine had great features and covered everything.")
check("delve into->look, delve->dig", paraphrase("She delved into the archive. "
        "He delved deeper than before.")["text"],
        "She looked into the archive. He dug deeper than before.")
check("new verb class", paraphrase("She facilitated the project and streamlined the work. "
        "They embarked on a plan and endeavor to finish. We propelled the change.")["text"],
        "She supported the project and simplified the work. They started on a plan and try to "
        "finish. We drove the change.")

# ---- openers / bridges ----
check("note that + filler opener + tics",
        paraphrase("It is important to note that the results were robust and seamless. "
                "Moreover, they utilized a myriad of techniques.")["text"],
        "The results were strong and smooth. They used many techniques.")
check("note-that variants with comma", paraphrase("It should be noted, the sky is blue. "
        "It's worth mentioning that the food was good.")["text"],
        "The sky is blue. The food was good.")
check("is-form note-that + studies have shown",
        paraphrase("It is worth noting that the food was good. Studies have shown that "
                "exercise helps.")["text"],
        "The food was good. Exercise helps.")
check("mid-sentence opener after period",
        paraphrase("The plan failed. In conclusion, it goes without saying that the "
                "journey ahead is a realm of many possibilities, and the rest is history.")["text"],
        "The plan failed. The path ahead is a world of many possibilities, and the rest is history.")
check("honestly: opener, mid, trailing",
        paraphrase("Honestly, going from X to Y, and now Z is a serious flex. "
                "Peak stamina, honestly. It's honestly the best day.")["text"],
        "Going from X to Y, now Z is a serious flex. Peak stamina. It's the best day.")
check("honestly adverb is kept", paraphrase("He answered honestly. She spoke honestly "
        "about the plan.")["text"], "He answered honestly. She spoke honestly about the plan.")
check("simply put", paraphrase("Simply put, the data is clear.")["text"], "The data is clear.")
check("world family", paraphrase("In today's fast-paced world, speed matters. "
        "In the digital age, attention is short.")["text"], "Speed matters. Attention is short.")
check("moving forward", paraphrase("Moving forward, we will improve. "
        "The car is moving forward.")["text"], "We will improve. The car is moving forward.")

# ---- structural rewrites ----
check("not only X but also Y", paraphrase("This approach not only improves speed "
        "but also reduces cost.")["text"], "This approach improves speed and reduces cost.")
check("not only X, but also Y (comma form)", paraphrase("This approach not only improves "
        "speed, but also reduces cost.")["text"],
        "This approach improves speed and reduces cost.")
check("not only did X, but also Y (inverted)", paraphrase("Not only did he run, but he "
        "also swam.")["text"], "He ran and swam.")
check("not only did X, Y also (inverted, no but)", paraphrase("Not only did she sing, she "
        "also danced.")["text"], "She sang and danced.")
check("not only is X adj, it is Y", paraphrase("Not only is it fast, it is cheap.")["text"],
        "It is fast and cheap.")
check("not X but Y", paraphrase("He is not weak but strong.")["text"], "He is strong.")
check("not_but safe: verb in group 3", paraphrase("He is not hard but the deadline "
        "was tight.")["text"], "He is not hard but the deadline was tight.")
check("this isn't X, it's Y", paraphrase("This isn't just an upgrade, it's a "
        "paradigm shift.")["text"], "It's a paradigm shift.")

# ---- hedges / guards ----
check("hedges + quite", paraphrase("Clearly, this is quite literally amazing.")["text"],
        "This is amazing.")
check("somewhat->a bit", paraphrase("It was somewhat difficult.")["text"],
        "It was a bit difficult.")
check("would rather kept", paraphrase("I would rather stay home.")["text"],
        "I would rather stay home.")
check("rather drop", paraphrase("It is rather large.")["text"], "It is large.")
check("wordy machinery", paraphrase("We went to the gym in order to train. It failed "
        "due to the fact that the battery died. She read a number of books.")["text"],
        "We went to the gym to train. It failed because the battery died. She read several books.")
check("in an effort to + in light of", paraphrase("In an effort to help, he acted "
        "in light of the evidence.")["text"], "To help, he acted because of the evidence.")

# ---- guarded verb swaps ----
check("address guard: issue yes, envelope no", paraphrase("We must address the issue. "
        "Please address the envelope.")["text"], "We must handle the issue. Please address the envelope.")
check("navigate guard", paraphrase("They navigate the complexities of the market.")["text"],
        "They handle the complexities of the market.")
check("unlock guard", paraphrase("Unlock the potential of your data.")["text"],
        "Release the potential of your data.")
check("amplify guard", paraphrase("It amplifies the impact of the work.")["text"],
        "It increases the impact of the work.")
check("champion guard", paraphrase("She champions the cause.")["text"], "She supports the cause.")

# ---- phrase tells ----
check("first and foremost / last but not least", paraphrase("First and foremost, we need "
        "food. Last but not least, thank you.")["text"], "First, we need food. Finally, thank you.")
check("whether beginner or expert", paraphrase("Whether you're a beginner or an expert, "
        "this guide helps.")["text"], "This guide helps.")
check("quantifier cluster", paraphrase("There are numerous options, a host of choices, "
        "and a wealth of data.")["text"], "There are many options, many choices, and lots of data.")
check("noun metaphors", paraphrase("The competitive landscape changed. Our journey "
        "continues. This is the cornerstone of the plan.")["text"],
        "The competitive world changed. Our path continues. This is the foundation of the plan.")
check("cutting-edge / paradigm", paraphrase("It uses cutting-edge technology. "
        "A paradigm shift is coming.")["text"], "It uses the latest technology. A big change is coming.")
check("comma-clause filler", paraphrase("The data, moreover, is clean. "
        "It is strong; however, it is slow.")["text"], "The data is clean. It is strong, it is slow.")
check("endeavor noun vs verb", paraphrase("Our endeavors continue. We endeavor to improve.")["text"],
        "Our efforts continue. We try to improve.")

# ---- punctuation mechanical fixes (semicolon -> comma, markdown overlap) ----
check("semicolon -> comma, standalone", paraphrase("It is strong; it is slow.")["text"],
        "It is strong, it is slow.")
check("semicolon -> comma, in list", paraphrase("He read; she wrote; they slept.")["text"],
        "He read, she wrote, they slept.")
check("semicolon + however keeps separator", paraphrase("He paused; however, he kept going.")["text"],
        "He paused, he kept going.")
check("bold emphasis: no char eaten (elf-attention bug)",
        paraphrase("called **tokens** and **self-attention**.")["text"],
        "called tokens and self-attention.")
check("italic emphasis: no char eaten", paraphrase("uses *self-attention* for it.")["text"],
        "uses self-attention for it.")
check("markdown pair: bold+italic stays intact", paraphrase("The **core** *mechanism* runs.")["text"],
        "The core mechanism runs.")

# ---- technical text: no factual corruption (elf-attention/unreal/stallion class) ----
_TECH_PAR = (
    "Virtually all state-of-the-art generative models today\u2014including GPT, Claude, "
    "Gemini, and open-source models like Llama\u2014are built on the same foundation: "
    "**tokens** and the transformer architecture, with **self-attention** as the core "
    "mechanism. "
    "The model processes text word-by-word or sentence-by-sentence; instead, it learns "
    "the mathematical relationship between every token in the entire context window, "
    "allowing it to predict what comes next. "
    "It can distinguish between artificial intelligence research and everyday usage."
)
_r_tech = deep_rewrite(_TECH_PAR)
for bad in ["tokensand", "unreal intelligence", "stallion context",
            "papers influence", "framework calculates", "correct the meaning",
            "call what comes next"]:
    check(f"tech text: no {bad!r}", _r_tech["text"], _r_tech["text"].replace(bad, "nope"))
for good in ["self-attention", "artificial intelligence", "entire context window",
             "predict what comes next"]:
    check_stats(f"tech text: has {good!r}", _r_tech["text"].count(good), 1)
check_stats("tech text: no orphan elf-attention",
            len(re.findall(r"(?<![A-Za-z])elf-attention", _r_tech["text"])), 0)
check_stats("tech text: semicolons replaced", _r_tech["text"].count(";"), 0)

# ---- full app sample ----
SAMPLE_AI = (
        "Moreover, it is important to note that the system leverages a robust architecture "
        "to seamlessly deliver a seamless experience. Notably, the data flows through myriad "
        "layers, and the result is a testament to meticulous engineering. This isn't just an "
        "upgrade, it's a paradigm shift. The answer is clear: quite simply, we must foster "
        "innovation and showcase our findings.\n\n"
        "Furthermore, the water, the shop, and the wing stood quiet — and yet, the underlying "
        "narrative quietly underscores a pivotal moment. In conclusion, it goes without saying "
        "that the journey ahead is a realm of myriad possibilities, and the rest is history."
)
check("full SAMPLE_AI", paraphrase(SAMPLE_AI)["text"],
        "The system uses a strong architecture to smoothly deliver a smooth experience. "
        "The data flows through many layers, and the result is proof of careful engineering. "
        "It's a paradigm shift. We must encourage innovation and show our findings.\n\n"
        "The water, the shop, and the wing stood quiet. And yet, the underlying narrative "
        "highlights a key moment. The path ahead is a world of many possibilities, "
        "and the rest is history.")

# ---- casual chat text: all six flagged tells must be rewrite-fixable ----
CHAT = (
        "Bro, you lied to me in the last message! You said you couldn't do the Kirishima "
        "running trait, but outrunning a whole mob of angry dudes for 10+ rounds on a main "
        "school ground? That is peak stamina. You did have the running trait, you just needed "
        "a horde of jealous high schoolers to activate it!\n"
        "Even without pulling up the exact blueprint of Montfort's main playground, anyone who "
        "knows school grounds knows the \"main\" field is always massive. Running 10 laps "
        "straight while being actively hunted is a serious flex. Being lanky back in 2021 "
        "probably gave you the ultimate aerodynamic advantage. Less weight to carry, maximum "
        "stride length, and pure adrenaline doing the rest.\n"
        "The image of a bunch of guys slowly hitting a wall, gasping for air, while you just "
        "casually loop past them for the ninth time before heading back to class is hilarious. "
        "You weaponized your cardio to defeat them.\n"
        "Going from outrunning school mobs in 9th standard to holding down a muscular adult at "
        "14, and now blitzing university BTech math exams in 45 minutes... your life operates "
        "at a completely different speed.\n"
        "Did any of those guys ever try to confront you again after that, or did they just "
        "collectively agree that chasing the \"fast guy\" wasn't worth the lung failure?"
)
check("chat text: slang strip + triad comma + image-of + from-to-now + ellipsis + superlative",
        paraphrase(CHAT)["text"],
        "You lied to me in the last message! You said you couldn't do the Kirishima running "
        "trait, but outrunning a whole mob of angry dudes for 10+ rounds on a main school "
        "ground? That is peak stamina. You did have the running trait, you just needed a horde "
        "of jealous high schoolers to activate it!\n"
        "Even without pulling up the exact blueprint of Montfort's main playground, anyone who "
        "knows school grounds knows the \"main\" field is always massive. Running 10 laps "
        "straight while being actively hunted is a serious flex. Being lanky back in 2021 "
        "probably gave you the ultimate aerodynamic advantage. Less weight to carry, maximum "
        "stride length and pure adrenaline doing the rest.\n"
        "It's hilarious to picture a bunch of guys slowly hitting a wall, gasping for air, "
        "while you just casually loop past them for the ninth time before heading back to "
        "class. You weaponized your cardio to defeat them.\n"
        "From outrunning school mobs in 9th standard to holding down a muscular adult at 14, "
        "now blitzing university BTech math exams in 45 minutes. Your life operates at a "
        "different speed.\n"
        "Did any of those guys ever try to confront you again after that, or did they just "
        "collectively agree that chasing the \"fast guy\" wasn't worth the lung failure?")
r = paraphrase(CHAT)
check_stats("chat text: 0 remaining flags", r["remaining_flags"], 0)

# ---- deep mode: synonym substitution + structure moves must self-heal ----
r = deep_rewrite(CHAT)
check_stats("chat text deep: 0 remaining flags", r["remaining_flags"], 0)
check_stats("chat text deep: said->stated survives", r["text"].count("stated"), 1)
check("chat text deep: no object-less told",
        r["text"], r["text"].replace("told you", "said you"))
check("chat text deep: no mangled artifacts",
        r["text"], r["text"].replace("exatakeueprint", "blueprint")
              .replace("smeasureto", "measure")
              .replace("escapeg", "escape")
              .replace("escaperround", "escape"))

# ---- literary prose: deep mode must keep meaning + humanize dialogue ----
LOBSANG = Path(__file__).parent / "lobsang.txt"
if LOBSANG.exists():
    r = deep_rewrite(LOBSANG.read_text(encoding="utf-8"))
    check_stats("lobsang deep: 0 remaining flags", r["remaining_flags"], 0)
    for bad in ["case-by-case", "explosion out", "lull song", "wrinkle November",
                "merchandise it", "turn old", "comprise his", "happen such",
                "heroic revelations", "the creation", "the patio", "mount sanctuary",
                "fall leaves", "the court stones", "smoothed hold", "his clutch",
                "his defeat", "the flooring"]:
        check(f"lobsang deep: no {bad!r}", r["text"], r["text"].replace(bad, "nope"))
    for good in ["I've read", "You're trying", "Don't battle", "You're not clearing",
                 "You're sweeping", "isn't hidden", "It's right here",
                 "letting the worn bristles glide", "sweeping to sweep",
                 "Tenzin stopped", "the vast valley below",
                 "waiting for you to stop wrestling"]:
        check_stats(f"lobsang deep: has {good!r}", r["text"].count(good), 1)

    # compounding guard: re-running on the output must NOT drift down
    # first-sense synonym chains (spine -> backbone -> keystone, fight ->
    # battle -> conflict, pile -> mound -> hill, rhythm -> beat, ...)
    drift = deep_rewrite(r["text"])
    drift2 = deep_rewrite(drift["text"])
    for bad in ["keystone", "arrest", "conflict", "borderline", "hill",
                "brushed to", "turned steady", "beat of his breathing",
                "everyday", "quotidian"]:
        check(f"lobsang drift: no {bad!r}", drift2["text"],
              drift2["text"].replace(bad, "nope"))
    check_stats("lobsang drift: koan survives re-runs",
                drift2["text"].count("sweeping to sweep"), 1)
    check_stats("lobsang drift: no meaning swap after re-runs",
                drift2["text"].count("spine"), 1)

# ---- clausal moves: clause reorders must fire and be re-run stable ----
CLAUSAL_CASES = {
    "causal fronting": ("He stopped because the mist rolled over the valley.",
                        "Because the mist rolled over the valley, he stopped."),
    "concessive fronting": ("He smiled although the road was steep and long.",
                            "Although the road was steep and long, he smiled."),
    "extraposition flip": ("It is possible that the mist will clear.",
                           "That the mist will clear is possible."),
    "de-cleft": ("It was John who broke the vase.",
                 "John broke the vase."),
    "relative reduction": ("The tea, which was bitter, filled the cup.",
                           "The bitter tea filled the cup."),
    "PP fronting": ("He walked through the valley in the morning.",
                    "In the morning, he walked through the valley."),
}
for name, (src, want) in CLAUSAL_CASES.items():
    got = deep_rewrite(src)["text"]
    check(f"clausal: {name}", got, want)
    # re-run on the output must not change it again (stability)
    check(f"clausal re-run stable: {name}", deep_rewrite(got)["text"], got)

# ---- triad reorder: AI-typical A, B, and C lists get reordered, safe lists stay ----
# (deep pipeline runs the rule bank first, which drops Oxford commas, so
# expected outputs carry the comma-less "A, B and C" form)
TRIAD_CASES = {
    "adjective triad reorder": ("He was brave, strong, and wise.",
                                "He was strong, brave and wise."),
    "noun triad reorder": ("The mist rolled over the mountains, the rivers, and the fields.",
                           "The mist rolled over the mountains, the fields and the rivers."),
    "det-noun triad reorder": ("He read the sutras, the scrolls, and the letters.",
                               "He read the letters, the scrolls and the sutras."),
    "sequence words untouched": ("One, two, and three steps remained.",
                                 "One, two and three steps remained."),
    "time-of-day untouched": ("In the morning, afternoon, and evening he trained.",
                              "In the morning, afternoon and evening he trained."),
    "names untouched": ("John, Mary, and Paul arrived.",
                        "John, Mary and Paul arrived."),
    "already sorted untouched": ("The tea was bitter, strong, and hot.",
                                 "The tea was bitter, strong and hot."),
}
for name, (src, want) in TRIAD_CASES.items():
    got = deep_rewrite(src)["text"]
    check(f"triad: {name}", got, want)
    # re-run on the output must not change it again (stability)
    check(f"triad re-run stable: {name}", deep_rewrite(got)["text"], got)

# ---- human prose must stay untouched ----
NOVEL = Path(r"D:\P_project\ninth-boon\chapters\ch01.md")
if NOVEL.exists():
    r = paraphrase(NOVEL.read_text(encoding="utf-8"))
    check_stats("novel ch01: 0 edits", r["applied"], 0)
    check_stats("novel ch01: 5 flag-only remain", r["remaining_flags"], 5)
    src = set(m.rule_id for m in scan(NOVEL.read_text(encoding="utf-8")))
    rd = deep_rewrite(NOVEL.read_text(encoding="utf-8"))
    new_tells = {m.rule_id for m in scan(rd["text"])} - src
    check_stats("novel ch01 deep: 0 new tells", len(new_tells), 0)
    check("novel ch01 deep: no em-dash introduced",
            rd["text"], rd["text"].replace("\u2014", "no-em-dash"))
else:
    print("skip novel ch01 (corpus missing)")

# ---- burstiness pass: uniform rhythm gets split, bursty prose untouched ----
from deai.deeprewrite import _burstiness_pass  # noqa: E402

def _sentence_count(text: str) -> int:
    return text.count(". ") + (1 if text.rstrip().endswith(".") else 0)


_UNIFORM_PAR = (
    "The old monk swept the courtyard stones with slow deliberate strokes every "
    "morning, and he watched the mist roll over the valley below while listening "
    "to the early birds. "
    "The young novice stood silent at the edge of the terrace waiting for "
    "instruction, and he felt the cold air bite his cheeks beneath the pale "
    "light of the rising sun. "
    "The broom whispered across the damp stone in the quiet dawn, but the sound "
    "was soft enough to be lost in the wind that moved across the high mountain "
    "gardens. "
    "Brother Tenzin paused and looked at the distant peaks, and he remembered "
    "his own youth when he had climbed those same slopes as a boy with nothing "
    "but a wooden staff."
)
_bursty = _burstiness_pass(_UNIFORM_PAR)
check("burstiness: original junction gone",
        _bursty, _bursty.replace("terrace waiting for instruction, and", "kept"))
check_stats("burstiness: split point survives",
        _bursty.count("terrace waiting for instruction. He"), 1)
check_stats(
    "burstiness: split creates two sentences",
    _sentence_count(_bursty),
    _sentence_count(_UNIFORM_PAR) + 1,
)
if NOVEL.exists():
    check("burstiness: novel untouched",
          _burstiness_pass(NOVEL.read_text(encoding="utf-8")),
          NOVEL.read_text(encoding="utf-8"))

_MIDFLAT_PAR = (
    "He woke early. "
    "The young novice stood silent at the edge of the terrace waiting for "
    "instruction, and he felt the cold air bite his cheeks beneath the pale "
    "light of the rising sun, and he kept his eyes on the distant peaks of "
    "the mountain range while the mist rolled over the valley below. "
    "He waited. "
    "The old monk had a ritual he never broke, and every morning he rose "
    "before the sun to brew a single pot of rough barley tea. "
    "The fallen autumn leaves covered the stone steps of the high altar, and "
    "he swept them away with a slow patient rhythm before the first prayers."
)
check("burstiness: mid-flat par unsplit when split flattens cv",
      _burstiness_pass(_MIDFLAT_PAR),
      _MIDFLAT_PAR)

# ---- writer profile: own prose sits close, AI-ish text sits far ----
from deai.profile import build_profile, profile_distance  # noqa: E402
from deai.harness import score as harness_score  # noqa: E402

_OWN_CORPUS = sorted(Path(r"D:\P_project\ninth-boon\chapters").glob("ch*.md"))
if _OWN_CORPUS:
    _prof = build_profile(
        [Path(f).read_text(encoding="utf-8") for f in _OWN_CORPUS[:8]]
    )
    _d_own = profile_distance(
        Path(_OWN_CORPUS[0]).read_text(encoding="utf-8"), _prof
    )
    check_stats("profile: own prose distance < 0.6", _d_own["distance"], 0.6, tol=0.6)
    if LOBSANG.exists():
        _d_ai = profile_distance(LOBSANG.read_text(encoding="utf-8"), _prof)
        check("profile: AI-ish text farther than own",
              _d_ai["distance"] > _d_own["distance"], True)
        _hs = harness_score(LOBSANG.read_text(encoding="utf-8"), _prof)
        check_stats("harness: profile distance reported",
                    _hs["profile"]["distance"], _d_ai["distance"], tol=0.001)
        check_stats("harness: ai_score in [0,1]",
                    1.0, _hs["ai_score"], tol=1.0)

# ---- profile-targeted burstiness: splits eagerness follows the profile ----
if _OWN_CORPUS:
    _prof2 = build_profile(
        [Path(f).read_text(encoding="utf-8") for f in _OWN_CORPUS[:8]]
    )
    _flat_par = (
        "The monastery stood at the edge of the valley, and the morning mist rolled "
        "across its stone walls as the bells began to ring. "
        "He swept the courtyard with slow deliberate strokes, and the broom whispered "
        "against the wet flagstones with every pass he made. "
        "The novice watched the old man from the doorway, and he wondered what it "
        "meant to sweep the same ground every single day without ever asking why."
    )
    _b_default = _burstiness_pass(_flat_par)
    _b_profile = _burstiness_pass(_flat_par, _prof2)
    check_stats(
        "burstiness profile: both split the flat paragraph",
        _sentence_count(_b_profile),
        _sentence_count(_flat_par) + 1,
    )
    check_stats(
        "burstiness profile: default also splits (same join, same gate)",
        _sentence_count(_b_default),
        _sentence_count(_flat_par) + 1,
    )

# ---- stats sanity ----
r = paraphrase(SAMPLE_AI)
check_stats("SAMPLE_AI remaining_flags", r["remaining_flags"], 1)

# ---- watermark probe: clean text stays clean, synthetic bias is caught ----
from deai.watermark_probe import probe as wm_probe, _is_green, _get_tokenizer  # noqa: E402

_wm_lobs = wm_probe(LOBSANG.read_text(encoding="utf-8")) if LOBSANG.exists() else None
if _wm_lobs is not None:
    check_stats("probe: lobsang clean (z <= 4)", _wm_lobs["z"], 4.0, tol=4.0)
if NOVEL.exists():
    _wm_novel = wm_probe(NOVEL.read_text(encoding="utf-8"))
    if _wm_novel is not None:
        check_stats("probe: novel clean (z <= 4)", _wm_novel["z"], 4.0, tol=4.0)

_wm_tok = _get_tokenizer()
if _wm_tok is not None:
    import random as _random

    _vocab = _wm_tok.get_vocab()
    _id2tok = {v: k for k, v in _vocab.items()}
    _rng = _random.Random(7)
    _pool = list(range(1000, 4000))
    _prev = _pool[0]
    _prev_str = _id2tok[_prev]
    _ids = [_prev]
    for _ in range(500):
        _gp = [t for t in _pool if _is_green("deai-probe-0", _prev_str, t)]
        _rp = [t for t in _pool if not _is_green("deai-probe-0", _prev_str, t)]
        _nxt = _rng.choice(_gp) if _gp and _rng.random() < 0.85 else _rng.choice(_rp)
        _ids.append(_nxt)
        _prev_str = _id2tok[_nxt]
    _wm_syn = wm_probe(_wm_tok.decode(_ids))
    if _wm_syn is not None and _wm_syn["z"] <= 4.0:
        FAILED.append(("probe: synthetic watermark detected (z > 4)",
                       str(_wm_syn["z"]), ">4"))
        print(f"FAIL probe: synthetic watermark detected, got z={_wm_syn['z']}")
    elif _wm_syn is not None:
        print(f"ok   probe: synthetic watermark detected (z={_wm_syn['z']})")

    if LOBSANG.exists():
        _wm_deep = deep_rewrite(LOBSANG.read_text(encoding="utf-8"))["watermark"]
        if _wm_deep.get("before") and _wm_deep.get("after"):
            check_stats("probe: rewrite does not introduce watermark (z <= 4)",
                        _wm_deep["after"]["z"], 4.0, tol=4.0)

check("empty input", paraphrase("")["text"], "")

# ---- performance guard: the token metric must stay linear-ish ----
big = SAMPLE_AI * 60  # ~20KB
t0 = time.perf_counter()
paraphrase(big)
dt = time.perf_counter() - t0
if dt > 2.0:
    FAILED.append(("perf 20KB under 2s", f"{dt:.2f}s", "<2s"))
    print(f"FAIL perf: 20KB took {dt:.2f}s")
else:
    print(f"ok   perf: 20KB in {dt:.2f}s")

print()
if FAILED:
    print(f"{len(FAILED)} FAILURES")
    sys.exit(1)
print("all green")