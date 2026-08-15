from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from deai.detector import group_by_category, scan
from deai.fixers import apply_fixes
from deai.stats import compute_stats

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="de-ai")


class AnalyzeRequest(BaseModel):
    text: str


class FixRequest(BaseModel):
    text: str
    types: list[str] | None = None


class ParaphraseRequest(BaseModel):
    text: str
    mode: str = "standard"
    scrub: bool = False
    profile: str | None = None


class HarnessRequest(BaseModel):
    text: str
    profile: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    matches = scan(req.text)
    stats = compute_stats(req.text)
    tells = group_by_category(matches)
    counts = {cat: len(items) for cat, items in tells.items()}
    by_severity = {"info": 0, "warn": 0, "error": 0}
    for m in matches:
        by_severity[m.severity] += 1
    fixable = sum(1 for m in matches if m.fix is not None)
    tell_density = round(1000 * len(matches) / stats["words"], 1) if stats["words"] else 0.0
    return {
        "stats": stats,
        "tells": tells,
        "counts": counts,
        "by_severity": by_severity,
        "total": len(matches),
        "fixable": fixable,
        "tell_density": tell_density,
    }


@app.post("/api/fix")
def fix(req: FixRequest) -> dict:
    matches = scan(req.text)
    return apply_fixes(req.text, matches, req.types)


@app.post("/api/paraphrase")
def paraphrase(req: ParaphraseRequest) -> dict:
    if req.mode == "deep":
        from deai.deeprewrite import deep_rewrite

        profile = _load_profile(req.profile)
        return deep_rewrite(req.text, max_scrub=3 if req.scrub else 0, profile=profile)
    from deai.paraphraser import paraphrase as run_paraphrase

    return run_paraphrase(req.text)


@app.post("/api/harness")
def harness(req: HarnessRequest) -> dict:
    from deai.harness import report

    profile = _load_profile(req.profile)
    return report(req.text, profile)


def _load_profile(name: str | None) -> dict | None:
    if not name:
        return None
    from deai.profile import load_profile

    path = Path(__file__).parent / "profiles" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"profile not found: {name}")
    return load_profile(path)