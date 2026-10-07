"""PDF parser: turns a PyMuPDF page dict into ordered text blocks.

Purpose: column-aware reading order, header/footer/watermark removal, bold/size/superscript flags.
Inputs: a dict as returned by PyMuPDF page.get_text("dict"), plus parser settings from config.toml.
Outputs: list[Block] in reading order. Data flow: pdf_parser -> chunker.
The core function blocks_from_dict() is pure (no PDF needed), so it is unit-testable.
parse_pdf() is the only function that touches PyMuPDF; it imports it lazily.
How to change: thresholds come from config.toml [parser]; regexes are marked '# EDIT HERE:'.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator

# EDIT HERE: dot-leader pattern that identifies table-of-contents lines.
TOC_RE = re.compile(r"\.{4,}\s*\d+\s*$|(?:\s\.){4,}\s*\d+\s*$")
# A line whose direction vector is not horizontal is treated as a watermark.
MAX_ANGLE_COS_DEVIATION = 0.05


@dataclass
class Block:
    """One text block in reading order."""
    text: str
    bbox: tuple[float, float, float, float]
    page: int                      # 1-based PDF page index
    size: float = 0.0              # largest span font size
    bold: bool = False             # True if any span is bold
    superscripts: list[str] = field(default_factory=list)   # superscript span texts (footnote markers)
    is_toc: bool = False


def _span_is_bold(span: dict[str, Any]) -> bool:
    """Bold if flag bit 16 is set or the font name says Bold."""
    return bool(span.get("flags", 0) & 16) or "bold" in str(span.get("font", "")).lower()


def _span_is_super(span: dict[str, Any]) -> bool:
    """Superscript if flag bit 1 is set (PyMuPDF convention)."""
    return bool(span.get("flags", 0) & 1)


def _is_horizontal(line: dict[str, Any]) -> bool:
    """False for rotated/diagonal lines (watermarks)."""
    dx, dy = line.get("dir", (1.0, 0.0))
    return abs(dx) >= 1 - MAX_ANGLE_COS_DEVIATION and abs(dy) <= MAX_ANGLE_COS_DEVIATION


def _line_text(line: dict[str, Any]) -> str:
    return "".join(s.get("text", "") for s in line.get("spans", []))


def _block_from_dict_block(raw: dict[str, Any], page_no: int, strip: list[re.Pattern]) -> Block | None:
    """Convert one text block; drop rotated and pattern-matched lines. Returns None if nothing is left."""
    kept: list[str] = []
    size, bold, supers = 0.0, False, []
    x0 = y0 = 1e9
    x1 = y1 = -1e9
    for line in raw.get("lines", []):
        if not _is_horizontal(line):
            continue
        text = _line_text(line).strip()
        if not text or any(p.search(text) for p in strip):
            continue
        kept.append(text)
        lx0, ly0, lx1, ly1 = line["bbox"]
        x0, y0, x1, y1 = min(x0, lx0), min(y0, ly0), max(x1, lx1), max(y1, ly1)
        for sp in line.get("spans", []):
            size = max(size, float(sp.get("size", 0)))
            bold = bold or _span_is_bold(sp)
            if _span_is_super(sp) and sp.get("text", "").strip():
                supers.append(sp["text"].strip())
    if not kept:
        return None
    text = "\n".join(kept)
    return Block(text=text, bbox=(x0, y0, x1, y1), page=page_no, size=size, bold=bold,
                 superscripts=supers, is_toc=bool(TOC_RE.search(text)))


def order_blocks(blocks: list[Block], page_width: float) -> list[Block]:
    """Reading order: auto-detect 1 vs 2 columns; full-width blocks split bands.

    A block is full-width if it crosses the page midline by a clear margin on both sides.
    Within each band: left column top-to-bottom, then right column top-to-bottom.
    """
    mid = page_width / 2
    margin = page_width * 0.05

    def full_width(b: Block) -> bool:
        return b.bbox[0] < mid - margin and b.bbox[2] > mid + margin

    ordered: list[Block] = []
    band: list[Block] = []

    def flush() -> None:
        left = sorted((b for b in band if b.bbox[2] <= mid + margin), key=lambda b: (b.bbox[1], b.bbox[0]))
        right = sorted((b for b in band if b.bbox[2] > mid + margin), key=lambda b: (b.bbox[1], b.bbox[0]))
        # One-column band: everything is "left" or "right" only by chance, so sort by y instead.
        if not left or not right:
            ordered.extend(sorted(band, key=lambda b: (b.bbox[1], b.bbox[0])))
        else:
            ordered.extend(left + right)
        band.clear()

    for b in sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0])):
        if full_width(b):
            flush()
            ordered.append(b)
        else:
            band.append(b)
    flush()
    return ordered


def blocks_from_dict(page_dict: dict[str, Any], page_no: int, parser_cfg: dict[str, Any] | None = None,
                     keep_toc: bool = False) -> list[Block]:
    """Pure function: PyMuPDF page dict -> ordered Blocks.

    parser_cfg keys (config.toml [parser]): header_fraction, footer_fraction, strip_patterns.
    TOC lines (dot leaders) are dropped unless keep_toc is True.
    """
    cfg = parser_cfg or {}
    header_f = float(cfg.get("header_fraction", 0.06))
    footer_f = float(cfg.get("footer_fraction", 0.06))
    strip = [re.compile(p) for p in cfg.get("strip_patterns", [])]
    width, height = float(page_dict["width"]), float(page_dict["height"])

    blocks: list[Block] = []
    for raw in page_dict.get("blocks", []):
        if raw.get("type", 0) != 0:        # 0 = text; images are handled by the vision step
            continue
        b = _block_from_dict_block(raw, page_no, strip)
        if b is None:
            continue
        y_top, y_bot = b.bbox[1], b.bbox[3]
        if y_bot <= height * header_f or y_top >= height * (1 - footer_f):
            continue                        # running header / footer by page fraction
        if b.is_toc and not keep_toc:
            continue
        blocks.append(b)
    return order_blocks(blocks, width)


def parse_pdf(path: str, parser_cfg: dict[str, Any] | None = None) -> Iterator[list[Block]]:
    """Yield the ordered blocks of each page of a PDF. Requires PyMuPDF (imported lazily)."""
    import fitz  # noqa: PLC0415  (lazy: unit tests do not need PyMuPDF)
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            yield blocks_from_dict(page.get_text("dict"), i, parser_cfg)


def parse_page(path: str, page_no: int, parser_cfg: dict[str, Any] | None = None) -> list[Block]:
    """Return the ordered blocks of one page (1-based). Used by the `page` debug command."""
    import fitz  # noqa: PLC0415
    with fitz.open(path) as doc:
        return blocks_from_dict(doc[page_no - 1].get_text("dict"), page_no, parser_cfg)
