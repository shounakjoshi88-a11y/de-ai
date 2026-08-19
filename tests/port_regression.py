"""Regression tests for the watermarks-remover port (text scope).

Pins the ported modules so upstream porting edits can never silently regress
them: Layer A Unicode inspection/cleaning, stylometry, text-watermark
detector plumbing, Layer B rewrite loop, format dispatch, and container
inspection/cleaning including C2PA stripping from embedded images.

Run from the de-ai directory:
    python tests/port_regression.py
Exit code 0 = all green.
"""

from __future__ import annotations

import base64
import io
import os
import random
import struct
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deai.container_meta import (  # noqa: E402
    clean_container,
    inspect_container,
)
from deai.format_dispatch import classify_bytes, classify  # noqa: E402
from deai.layerb import rewrite  # noqa: E402
from deai.stylometry import score_text_stylometry  # noqa: E402
from deai.text_detectors import (  # noqa: E402
    detector_status,
    run_all_text_detectors,
    run_text_detectors,
)
from deai.text_unicode import clean_text, inspect_text  # noqa: E402

FAILED: list[tuple[str, str, str]] = []


def check(name: str, got: object, want: object) -> None:
    if got != want:
        FAILED.append((name, str(got), str(want)))
        print(f"FAIL {name}\n  got:  {got!r}\n  want: {want!r}")
    else:
        print(f"ok   {name}")


def check_true(name: str, cond: bool) -> None:
    check(name, cond, True)


# ---- Layer A: text_unicode ----
_insp = inspect_text("hello\u200b world")
check("layer-a: inspect finds zwsp", _insp.suspicious_total, 1)
_out, _stats = clean_text("hello\u200b world")
check("layer-a: clean strips zwsp", _out, "hello world")
check("layer-a: removed_count", _stats["removed_count"], 1)
check("layer-a: nfkc option", clean_text("ｆｕｌｌ", nfkc=True)[0], "full")

# ---- stylometry ----
_STYLE_AI = (
    "The rapid advancement of artificial intelligence has transformed numerous "
    "industries across the globe. Companies are increasingly adopting machine "
    "learning solutions to optimize their operations and improve efficiency. "
    "Policymakers are struggling to keep pace with the speed of innovation. "
    "This paradigm shift presents both unprecedented opportunities and "
    "significant challenges for stakeholders worldwide."
)
_srep = score_text_stylometry(_STYLE_AI)
check_true("stylometry: status ok", _srep.status == "ok")
check_true("stylometry: word_count >= 30", _srep.word_count >= 30)
check("stylometry: confidence is one of", _srep.confidence_level in ("CLEAN", "LOW", "MEDIUM", "HIGH"), True)

# ---- text detectors ----
check_true("detectors: status is dict", isinstance(detector_status(), dict))
_alld = run_all_text_detectors("hello world")
check("detectors: names", [d["detector"] for d in _alld], ["probe", "markllm", "claude-text"])
check_true("detectors: probe is available", _alld[0]["available"])
check("detectors: probe short text clean", _alld[0]["verdict"], "clean")
check_true("detectors: unconfigured remain fail-soft", all(d.get("available") is False and "error" in d for d in _alld[1:]))
check("detectors: usable only probe", [d["detector"] for d in run_text_detectors("hello world")], ["probe"])
check_true("detectors: status shows probe", detector_status().get("probe") is True)

# ---- Layer B: rewrite loop ----
_LB_IN = (
    "The rapid advancement of artificial intelligence has transformed numerous "
    "industries across the globe. Companies are increasingly adopting machine "
    "learning solutions to optimize their operations and improve efficiency."
)
_lb_out, _lb_info = rewrite(_LB_IN, backend="deterministic", candidates=3)
check("layer-b: deterministic backend", _lb_info["backend"], "deterministic")
check_true("layer-b: attempts >= 1", _lb_info["attempts_made"] >= 1)
check_true("layer-b: output changed", _lb_out != _LB_IN)
check("layer-b: evaluator", _lb_info["evaluator"], "probe")
check("layer-b: clean text passes", _lb_info["passed"], True)
_prompt, _pin = rewrite(_LB_IN, backend="print-prompt")
check("layer-b: print-prompt mode", _pin["mode"], "print-prompt")
check_true("layer-b: print-prompt returns prompt", _prompt.startswith("Rewrite the following text"))
check_true("layer-b: divergence in [0,1]", 0.0 <= _lb_info["candidate_scores"][0]["lexical_divergence"] <= 1.0)

# ---- Layer B: probe evaluator on watermarked input ----
from deai.watermark_probe import _get_tokenizer as _probe_tok  # noqa: E402
from deai.watermark_probe import _is_green as _probe_green  # noqa: E402
from deai.watermark_probe import probe as _probe  # noqa: E402
from deai.layerb import _pick_evaluator as _pick_ev  # noqa: E402

_probe_tok = _probe_tok()
if _probe_tok is not None:
    check("layer-b: evaluator priority (no markllm)", _pick_ev(None)[0], "probe")
    _pv = _probe_tok.get_vocab()
    _id2tok = {_v: _k for _k, _v in _pv.items()}
    _rng = random.Random(7)
    _pool = [i for i in range(1000, 4000)]
    _prev = _pool[0]
    _prev_str = _id2tok[_prev]
    _ids = [_prev]
    for _ in range(500):
        _gp = [t for t in _pool if _probe_green("deai-probe-0", _prev_str, t)]
        _rp = [t for t in _pool if not _probe_green("deai-probe-0", _prev_str, t)]
        _nxt = _rng.choice(_gp) if _gp and _rng.random() < 0.85 else _rng.choice(_rp)
        _ids.append(_nxt)
        _prev_str = _id2tok[_nxt]
    _wm_in = _probe_tok.decode(_ids)
    _wm_in_report = _probe(_wm_in)
    assert _wm_in_report is not None
    check_true("layer-b: synthetic input flagged watermarked", _wm_in_report["verdict"] == "watermarked")
    _wm_out, _wm_info = rewrite(_wm_in, backend="deterministic", candidates=2, max_loops=2)
    check("layer-b: watermarked input fails evaluator", _wm_info["passed"], False)
    check_true("layer-b: attempts exhausted", _wm_info["attempts_made"] >= 1)

# ---- format dispatch ----
_buf = io.BytesIO()
with zipfile.ZipFile(_buf, "w") as _z:
    _z.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
    _z.writestr("word/document.xml", "<w:document/>")
_zipdata = _buf.getvalue()
check("dispatch: docx ext", classify_bytes(_zipdata, ".docx"), "container")
check("dispatch: docx sniff (no ext)", classify_bytes(_zipdata), "container")
check("dispatch: txt", classify_bytes(b"hello world", ".txt"), "text")
check("dispatch: md", classify_bytes(b"# hi", ".md"), "container")
check("dispatch: pdf", classify_bytes(b"%PDF-1.7 fake", ".pdf"), "container")
check("dispatch: png out of scope", classify_bytes(b"\x89PNG\r\n\x1a\nfake", ".png"), "unknown")

import tempfile  # noqa: E402

with tempfile.TemporaryDirectory() as _td:
    _p = Path(_td) / "noext"
    _p.write_bytes(_zipdata)
    check("dispatch: classify file by sniff", classify(_p), "container")
    _p2 = Path(_td) / "note.md"
    _p2.write_text("# hi", encoding="utf-8")
    check("dispatch: classify file md", classify(_p2), "container")


# ---- container inspect/clean with C2PA embedded image ----
def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", __import__("zlib").crc32(tag + payload) & 0xFFFFFFFF)
    )


_c2pa_png = (
    b"\x89PNG\r\n\x1a\n"
    + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    + _chunk(b"c2pa", b'{"signature":"C2PA fake manifest"}')
    + _chunk(b"IDAT", __import__("zlib").compress(b"\x00\xff\x00\x00"))
    + _chunk(b"IEND", b"")
)
with tempfile.TemporaryDirectory() as _td:
    _p = Path(_td) / "test.docx"
    with zipfile.ZipFile(_p, "w") as _z:
        _z.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="png" ContentType="image/png"/>'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        _z.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/></Relationships>',
        )
        _z.writestr(
            "word/document.xml",
            '<?xml version="1.0"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>Hello docx with embedded c2pa image</w:t></w:r></w:p>"
            "</w:body></w:document>",
        )
        _z.writestr(
            "word/_rels/document.xml.rels",
            '<?xml version="1.0"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId2" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            'Target="media/image1.png"/></Relationships>',
        )
        _z.writestr("word/media/image1.png", _c2pa_png)
        _z.writestr(
            "docProps/core.xml",
            '<?xml version="1.0"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties">'
            "<dc:creator>AI Generator v9</dc:creator></cp:coreProperties>",
        )

    _rep = inspect_container(_p)
    check("container: format docx", _rep.format, "docx")
    check_true("container: has_c2pa", _rep.has_c2pa)
    check_true("container: has_ai_metadata", _rep.has_ai_metadata)
    check_true(
        "container: finding names embedded image",
        any("image1.png" in f for f in _rep.findings),
    )

    _dst = Path(_td) / "cleaned.docx"
    _res = clean_container(_p, _dst)
    check_true("container: c2pa stripped", not _res["still_has_c2pa"])
    check_true("container: ai metadata stripped", not _res["still_has_ai_metadata"])
    with zipfile.ZipFile(_dst) as _z:
        _png_out = _z.read("word/media/image1.png")
        _core_out = _z.read("docProps/core.xml").decode("utf-8")
    check_true("container: png chunk gone", b"c2pa" not in _png_out)
    check_true("container: dc:creator scrubbed", "AI Generator" not in _core_out)

print()
if FAILED:
    print(f"{len(FAILED)} FAILURES")
    sys.exit(1)
print("all green")