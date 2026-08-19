from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from deai.detector import group_by_category, scan
from deai.fixers import apply_fixes
from deai.stats import compute_stats

STATIC_DIR = Path(__file__).parent / "static"

#: Reject container uploads over this many bytes (base64-decoded) up front.
MAX_CONTAINER_BYTES = 256 * 1024 * 1024

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


class LayerARequest(BaseModel):
    text: str
    aggressive: bool = False
    strip_emoji_glue: bool = False


class CleanLayerARequest(BaseModel):
    text: str
    nfkc: bool = False
    aggressive_homoglyphs: bool = False
    normalize_spaces: bool = True
    strip_emoji_glue: bool = False
    strip_bidi: bool = False


class StylometryRequest(BaseModel):
    text: str


class DetectWatermarkRequest(BaseModel):
    text: str
    markllm_scheme: str | None = None
    markllm_dir: str | None = None
    markllm_model: str | None = None
    markllm_timeout: float | None = None


class ContainerRequest(BaseModel):
    file: str = Field(description="Base64-encoded file bytes")
    filename: str = Field(description="Original filename (suffix drives format detection)")


class LayerBRequest(BaseModel):
    text: str
    backend: str = "deterministic"
    strength: str = "paraphrase"
    model: str | None = None
    base_url: str | None = None
    allow_remote: bool = False
    temperature: float = 0.9
    candidates: int = 1
    max_loops: int = 1
    layer_a_after: bool = True
    markllm_scheme: str | None = None
    markllm_dir: str | None = None
    markllm_model: str | None = None
    markllm_timeout: float = 180.0
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


@app.post("/api/layer-a")
def layer_a(req: LayerARequest) -> dict:
    from deai.text_unicode import human_report, inspect_text

    report = inspect_text(
        req.text, aggressive=req.aggressive, strip_emoji_glue=req.strip_emoji_glue
    )
    return {"report": report.to_dict(), "human_report": human_report(report)}


@app.post("/api/clean-layer-a")
def clean_layer_a(req: CleanLayerARequest) -> dict:
    from deai.text_unicode import clean_text

    cleaned, stats = clean_text(
        req.text,
        nfkc=req.nfkc,
        aggressive_homoglyphs=req.aggressive_homoglyphs,
        normalize_spaces=req.normalize_spaces,
        strip_emoji_glue=req.strip_emoji_glue,
        strip_bidi=req.strip_bidi,
    )
    return {"cleaned": cleaned, "stats": stats}


@app.post("/api/stylometry")
def stylometry(req: StylometryRequest) -> dict:
    from deai.stylometry import score_text_stylometry

    report = score_text_stylometry(req.text)
    return report.to_dict()


@app.post("/api/detect-watermark")
def detect_watermark(req: DetectWatermarkRequest) -> dict:
    from deai.text_detectors import MarkLLMTextDetector, run_all_text_detectors

    markllm = None
    if req.markllm_scheme:
        markllm = MarkLLMTextDetector(
            scheme=req.markllm_scheme,
            upstream_dir=req.markllm_dir,
            model=req.markllm_model,
            timeout=req.markllm_timeout or 180.0,
        )
    return {"detectors": run_all_text_detectors(req.text, markllm=markllm)}


@app.post("/api/container-inspect")
def container_inspect(req: ContainerRequest) -> dict:
    from deai.container_meta import inspect_container
    from deai.format_dispatch import classify_bytes

    data = _decode_upload(req)
    kind = classify_bytes(data, Path(req.filename).suffix)
    if kind != "container":
        raise HTTPException(422, f"not a text container: {kind}")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / Path(req.filename).name
        path.write_bytes(data)
        report = inspect_container(path)
    return report.to_dict()


@app.post("/api/container-clean")
def container_clean(req: ContainerRequest) -> dict:
    from deai.container_meta import clean_container
    from deai.format_dispatch import classify_bytes

    data = _decode_upload(req)
    kind = classify_bytes(data, Path(req.filename).suffix)
    if kind != "container":
        raise HTTPException(422, f"not a text container: {kind}")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / Path(req.filename).name
        dst = Path(td) / f"cleaned{Path(req.filename).suffix}"
        src.write_bytes(data)
        result = clean_container(src, dst)
        result["file_base64"] = base64.b64encode(dst.read_bytes()).decode("ascii")
    return result


@app.post("/api/layer-b")
def layer_b(req: LayerBRequest) -> dict:
    from deai.layerb import rewrite

    if req.backend not in ("deterministic", "print-prompt", "ollama", "openai-compatible"):
        raise HTTPException(422, f"unknown backend: {req.backend}")
    profile = _load_profile(req.profile)
    out, info = rewrite(
        req.text,
        backend=req.backend,
        model=req.model,
        base_url=req.base_url,
        strength=req.strength,
        layer_a_after=req.layer_a_after,
        temperature=req.temperature,
        candidates=req.candidates,
        max_loops=req.max_loops,
        allow_remote=req.allow_remote,
        markllm_scheme=req.markllm_scheme,
        markllm_dir=req.markllm_dir,
        markllm_model=req.markllm_model,
        markllm_timeout=req.markllm_timeout,
        profile=profile,
    )
    return {"text": out, "info": info}


def _decode_upload(req: ContainerRequest) -> bytes:
    try:
        data = base64.b64decode(req.file, validate=True)
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"invalid base64: {e}") from e
    if not data:
        raise HTTPException(422, "empty file")
    if len(data) > MAX_CONTAINER_BYTES:
        raise HTTPException(413, "file too large")
    return data


def _load_profile(name: str | None) -> dict | None:
    if not name:
        return None
    from deai.profile import load_profile

    path = Path(__file__).parent / "profiles" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"profile not found: {name}")
    return load_profile(path)