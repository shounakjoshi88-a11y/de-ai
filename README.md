# de-ai

A **deterministic, no-LLM** text rewriter that reduces the stylistic "tells" AI detectors look for in prose — without calling a language model, without changing the meaning, and without injecting new errors.

It scans your text for machine-writing patterns (repetitive sentence shapes, mechanical connectives, first-sense synonym habits, flat rhythm, stilted vocabulary) and rewrites it with a hand-tuned **rule bank + deep rewrite passes**. Every behavior is pinned by a regression suite, so edits can never silently regress a fix.

> Not a magic bullet, and not for evading plagiarism rules on someone else's work — see [Limitations](#limitations). It is a deterministic style tool for making *your own* text read more human.

---

## Why another rewriter?

LLM-based paraphrasing tools are:
- **Non-deterministic** — run the same text twice, get different output.
- **Costly** — every call hits a model.
- **Meaning-drift prone** — a model freely rewrites and quietly changes facts.

de-ai takes the opposite approach: **zero model calls**, fully deterministic output, and hard guardrails that refuse to mangle meaning — especially in technical text (the class of bug where `self-attention` became `elf-attention` or `server` became `waiter`).

---

## Features

- **Rule-bank detector** — flags known AI tells with severity and a machine-applicable fix, grouped by category.
- **Mechanical fixes** — `;` → `,`, em-dash / dash handling, clause-filler rewrites (`; however,` survives correctly), and markdown-safe emphasis stripping.
- **Deep rewrite passes** — lexical, clausal, triad, syntactic, burstiness (CV-gated sentence splitting), and humanization passes that run in a fixed order with a final **verification step** that reverts any swap that introduced a new tell.
- **Technical-vocabulary guard** — a curated blocklist (`_TECH_TERMS`, 600+ ML/AI/security/CS terms) that the lexical pass may never touch, plus ambiguity traps for words like `entire`, `model`, `document`, `adjust`, `predict` whose noun-sense synonyms would wreck meaning.
- **Profile targeting** — build a writing profile from a corpus; burstiness splitting and rhythm follow that profile instead of a fixed default.
- **Watermark probe** — estimates the A/B watermark z-score of text (optional; needs a GPT-2 tokenizer), and verifies a rewrite doesn't introduce a detectable watermark.
- **Web UI + REST API** — FastAPI server with a browser UI and JSON endpoints.
- **Regression-pinned** — 156+ assertions covering every verified behavior.

---

## How it works

### Pipeline

```
 input
   │
   ▼
 ┌────────────────────────────┐
 │ Rule bank (rules.py)       │  match AI tells → Rule(severity, fix)
 ├────────────────────────────┤
 │ Detector (detector.py)     │  scan() → matches; group_by_category()
 ├────────────────────────────┤
 │ Fixers (fixers.py)         │  apply mechanical fixes (clause filler,
 │                            │  dash/period, semicolon→comma, ...)
 ├────────────────────────────┤
 │ Paraphraser (paraphraser.py)│ paraphrase() → mechanical pass + post-clean
 ├────────────────────────────┤
 │ Deep passes (deeprewrite.py)│ deep_rewrite():
 │   lexical → clausal → triad │    meaning + style passes
 │   → syntactic → burstiness  │
 │   → humanize → verify       │    verification reverts bad swaps
 └────────────────────────────┘
   │
   ▼
 output + remaining_flags + applied counts
```

### Modules

| Module | Purpose |
| --- | --- |
| `deai/rules.py` | The rule bank: each tell is a `Rule` with a category, severity, regex pattern(s), and an optional machine fix. |
| `deai/detector.py` | `scan(text)` → matches; `group_by_category(matches)`. |
| `deai/fixers.py` | `apply_fixes(text, matches, types)`; per-fix handlers (e.g. `clause_filler`, `dash_period`, `semicolon_comma`). |
| `deai/paraphraser.py` | `paraphrase(text)` — mechanical rewrite + post-clean. Exports `COVERED`. |
| `deai/deeprewrite.py` | `deep_rewrite(text, max_scrub=0, profile=None)` — the deep passes, plus all meaning-guard vocab (`_TECH_TERMS`, `_AMBIGUOUS`, `_TARGET_TRAPS`, `_STILTED_TARGETS`, `_NEEDS_OBJECT`). |
| `deai/profile.py` | `build_profile(corpus)`, `load_profile(path)`, `profile_distance(text, profile)`. |
| `deai/harness.py` | `report(text, profile)` / `score(...)` — profile distance + detector-style scoring; uses `calibration.jsonl` for real-service calibration. |
| `deai/stats.py` | `compute_stats(text)` — word/sentence metrics, tell density. |
| `deai/watermark_probe.py` | `probe(text)` — optional A/B watermark z-score estimate. |
| `app.py` | FastAPI server + static browser UI. |
| `tests/paraphrase_regression.py` | The regression suite (run: `python tests/paraphrase_regression.py`). |

---

## Installation

```bash
pip install -r requirements.txt
# fastapi, uvicorn, lemminflect, wn, wordfreq
```

`wn` (WordNet) and `lemminflect` are used for synonym/lemma lookups in the lexical pass; `wordfreq` for common-word weighting.

---

## Usage

### Web UI / server

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/ in a browser.

### REST API

| Endpoint | Body | Returns |
| --- | --- | --- |
| `POST /api/analyze` | `{"text": "..."}` | stats, tells grouped by category, severity counts, tell density, fixable count |
| `POST /api/fix` | `{"text": "...", "types": [...]}` | text with requested fixes applied |
| `POST /api/paraphrase` | `{"text": "...", "mode": "standard"\|"deep", "scrub": bool, "profile": "name"\|null}` | rewritten text, `remaining_flags`, applied counts |
| `POST /api/harness` | `{"text": "...", "profile": "name"\|null}` | full report: profile distance + detector-style score |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/paraphrase \
  -H "Content-Type: application/json" \
  -d '{"text":"Virtually all state-of-the-art models are built on tokens and self-attention.","mode":"deep"}'
```

### Library

```python
from deai.paraphraser import paraphrase
from deai.deeprewrite import deep_rewrite
from deai.detector import scan

# standard pass
out = paraphrase("It is strong; it is slow.")
# {"text": "It is strong, it is slow.", "remaining_flags": 0, ...}

# deep pass
r = deep_rewrite("Virtually all state-of-the-art generative models today ...", profile=None)
print(r["text"], r["remaining_flags"])
```

---

## Profiles

A writing profile (average sentence length, burstiness/variance, common-word habits) can be built from your own corpus and used to target the rewrite:

```python
from deai.profile import build_profile, profile_distance

prof = build_profile([open("my_writing.md", encoding="utf-8").read()])
dist  = profile_distance(some_text, prof)
```

Profiles are user-local — `profiles/` is git-ignored and the server loads them by name (`/api/paraphrase` with `"profile": "name"`).

---

## Guardrails: what the engine will not do

- **Never auto-swaps technical vocabulary.** `_TECH_TERMS` (600+ terms: tokens, transformer, self-attention, server, endpoint, convolution, quantization, ...) is off-limits to the lexical pass — the source word *or* any candidate synonym in the set blocks the swap. This kills the `server → waiter`, `endpoint → terminus`, `convolution → swirl` class of bug.
- **Never swaps known ambiguity traps.** `_AMBIGUOUS` (e.g. `entire`, `model`, `document`, `adjust`, `predict`) — words whose first-sense synonyms drift the meaning (the `entire context window → stallion context window` class). `_TARGET_TRAPS` blocks specific wrong-sense targets (e.g. `artificial → unreal`).
- **Refuses unsafe syntactic moves.** `_NEEDS_OBJECT` verbs (`tell`, `give`, ...) can't become targets when the source verb is used intransitively (`said → told you` guarded).
- **Verifies after rewriting.** The deep pipeline re-scans its own output and reverts changes that introduced new tells (the compounding-drift guard: `spine → backbone → keystone` chains are blocked).

---

## Tests

```bash
python tests/paraphrase_regression.py
```

Exit code `0` = all green. The suite pins 156+ behaviors: semicolon→comma, markdown emphasis (no eaten characters), comma-clause fillers, the technical-vocabulary block, burstiness CV gating, clausal moves, drift-on-re-run stability, watermark probe sanity, and performance (linear-ish token metric on ~20KB input).

Some blocks are optional fixtures (your own corpus / novel chapters) and are skipped when absent — the suite stays green on a fresh clone.

---

## Project layout

```
de-ai/
├── app.py                     # FastAPI server + static UI
├── requirements.txt
├── README.md
├── deai/
│   ├── __init__.py
│   ├── rules.py               # rule bank (AI tells)
│   ├── detector.py            # scan + categorization
│   ├── fixers.py              # mechanical fix handlers
│   ├── paraphraser.py         # standard paraphrase pass
│   ├── deeprewrite.py         # deep passes + meaning guards
│   ├── profile.py             # writing profiles
│   ├── harness.py             # report / scoring
│   ├── stats.py               # text metrics
│   └── watermark_probe.py     # optional A/B watermark z-score
├── static/
│   └── index.html             # browser UI
└── tests/
    └── paraphrase_regression.py
```

---

## Limitations

Honest limits of this approach — read before relying on it:

1. **Not guaranteed against any specific commercial detector.** AI-detection is an arms race. Detector models, feature sets, and thresholds change; no rule-based rewrite can promise to stay under the radar of every detector forever. Treat "score went down on tool X today" as evidence, not a guarantee.
2. **Meaning-preserving ≠ meaning-identical.** Synonym swaps preserve the *sense* of a sentence but can shift nuance, emphasis, or register. Always proofread the output. If a rewrite ever reads wrong, it's a bug worth reporting — but subtle drift is inherent to any synonym-level rewriter.
3. **The technical blocklist is curated, not exhaustive.** `_TECH_TERMS` covers 600+ common ML/AI/security/CS terms, but rare, niche, or newly-coined jargon can still fall through and get a wrong-sense swap. If you hit one, add it to the set and re-run the regression.
4. **English-only.** The lexical pass relies on WordNet and English lemmatization (`lemminflect`). No support for other languages, code, or mixed-language text.
5. **Deterministic by design → low creative variation.** Same input, same output. If you want radically different phrasings you need more seed text or external iteration — this tool won't surprise you with variety.
6. **Heuristic NLP has edge cases.** WordNet first-sense lookup and POS heuristics are imperfect; the guardrails catch the known failure classes (documented above) but cannot be proven exhaustive for all English.
7. **Watermark probe is optional and approximate.** `watermark_probe` needs a GPT-2 tokenizer and estimates an A/B watermark z-score; it is a *proxy*, not a measurement of any real detector. Detector scores in `harness` are local proxies unless you add real `calibration.jsonl` entries yourself.
8. **Personal data discipline is on you.** The engine reads whatever text you give it. Nothing is sent anywhere (no network calls), but don't paste sensitive content into a shared server.
9. **Ethical note.** This tool changes *style*, not facts, and is meant for your own writing. Don't use it to misattribute others' work or to cheat academic-integrity systems — it won't make dishonest use honest.

---

*Built and tested on Windows/Python 3.x. The engine makes no network calls; everything runs locally.*
