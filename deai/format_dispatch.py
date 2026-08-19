"""Route a file or byte stream to the text or container pipeline.

Ported from watermarks-remover (MIT).

The routers (inspect_file, clean_file) and the audits (audit_lib) all need the
same answer: given a path or bytes, which pipeline owns it? That decision used
to live in three copies with subtly different extension tables and sniffing.
This module is the single interface for it.

Scope: this port handles text and text-bearing containers (which may embed
images with C2PA metadata). Standalone raster image / AV files are out of
scope and classify as "unknown" so callers refuse to mangle them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .container_meta import detect_container_format
from .image_meta import detect_format as detect_image_format

Kind = Literal["text", "container", "unknown"]

#: Bytes read for header-only sniffing. Zip-based containers (docx/odt/...)
#: need the full central directory, which sits at the end of the archive, so
#: only a PK header triggers a whole-file read.
CLASSIFY_HEADER_BYTES = 4096

CONTAINER_EXTS = {
    ".svg",
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".odt",
    ".epub",
    ".html",
    ".htm",
    ".md",
    ".markdown",
    ".mdx",
}
TEXT_EXTS = {
    ".txt",
    ".text",
    ".css",
    ".js",
    ".py",
    ".rs",
    ".go",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
}


def classify_bytes(data: bytes, suffix: str | None = None) -> Kind:
    """Classify *data* by extension first, then by magic bytes.

    The extension wins when it names a known format; otherwise the bytes are
    sniffed for container signatures. Unrecognized bytes classify as "unknown"
    — callers that must not mangle unknown binaries refuse unless the user
    explicitly forces a kind (--as / --force-text).

    *data* must cover the whole file: zip-based containers (docx/odt) are
    detected from their central directory, which sits at the end of the bytes.
    """
    ext = (suffix or "").lower()
    if ext in CONTAINER_EXTS:
        return "container"
    if ext in TEXT_EXTS:
        return "text"
    if data:
        sniff_path = Path("input") if not ext else Path(f"input{ext}")
        if detect_container_format(sniff_path, data) != "unknown":
            return "container"
    return "unknown"


def classify(path: Path) -> Kind:
    """Classify a file on disk by extension, then by its bytes.

    Known extensions are routed without reading the file. For unknown
    extensions a 4096-byte header is sniffed once; only when the header is a
    zip local header (PK) is the whole file read, because the container
    signature (docx/xlsx/pptx/odt/epub) lives in the central directory at
    the end of the archive.
    """
    ext = path.suffix.lower()
    if ext in CONTAINER_EXTS:
        return "container"
    if ext in TEXT_EXTS:
        return "text"
    with path.open("rb") as fh:
        head = fh.read(CLASSIFY_HEADER_BYTES)
    if head:
        data = path.read_bytes() if head[:4] == b"PK\x03\x04" else head
        sniff_path = Path("input") if not ext else Path(f"input{ext}")
        if detect_container_format(sniff_path, data) != "unknown":
            return "container"
    return "unknown"