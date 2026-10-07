"""PDF page links for Markdown outputs.

Purpose: turn every citation [code, edition, clause, PDF p.N] in a Markdown file into a link to that
PDF page, e.g. [ASME BPVC VIII-1, 2025, UG-37, PDF p.123](file:///C:/.../file.pdf#page=123).
Inputs: Markdown text, config ([links], [[documents]]). Output: Markdown text with links.
Data flow: runs in rag.write_markdown AFTER citecheck. The link text is the unchanged citation, so
citecheck still matches it. Text inside ``` fences (verbatim excerpts) is never touched.
N is the 1-based PDF page index (as the viewer counts), not the printed page number. The printed page
number is not stored by the parser, so only the PDF index is shown.
How to change: anchor style is config [links] anchor_style (viewers differ: '#page={n}' or
'#page={n}&zoom=100' etc.); EDIT HERE markers below for the path/URL building.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from asme_rag.citecheck import CITATION_RE
from asme_rag.config import resolve_path

FENCE_RE = re.compile(r"(```.*?```)", re.S)
ALREADY_LINK_RE = re.compile(r"\]\(file:")


def doc_for_citation(cfg: dict[str, Any], code: str, edition: int) -> dict[str, Any] | None:
    """Find the registered document a citation refers to (by division or code text and edition)."""
    c = code.lower().strip()
    for d in cfg["documents"]:
        if int(d["edition"]) != edition:
            continue
        full = f"{d['code']} {d.get('division', '')}".lower().strip()
        div = d.get("division", "").lower()
        if c in full or full in c or (div and div in c) or (not div and d["code"].lower() in c):
            return d
    return None


def pdf_url(cfg: dict[str, Any], doc: dict[str, Any], page: int) -> str:
    """file:/// URL (percent-encoded, so spaces in file names are safe) plus the configured anchor."""
    links = cfg.get("links", {})
    base = resolve_path(cfg, links.get("pdf_base_dir", "data/pdfs"))
    # EDIT HERE: the file name comes from the registered pdf_path; the folder from [links] pdf_base_dir.
    uri = (base / Path(doc["pdf_path"]).name).resolve().as_uri()
    return uri + links.get("anchor_style", "#page={n}").format(n=page)


def add_pdf_links(markdown: str, cfg: dict[str, Any]) -> str:
    """Add page links to every citation outside code fences. No-op if links.pdf_page_links is false."""
    if not cfg.get("links", {}).get("pdf_page_links", True):
        return markdown

    def link(m: re.Match) -> str:
        doc = doc_for_citation(cfg, m.group(1).strip(), int(m.group(2)))
        if doc is None:
            return m.group(0)                      # unknown book: leave the citation as plain text
        return f"{m.group(0)}({pdf_url(cfg, doc, int(m.group(4)))})"

    parts = FENCE_RE.split(markdown)
    for i in range(0, len(parts), 2):             # even parts are outside fences
        parts[i] = _link_outside(parts[i], link)
    return "".join(parts)


def _link_outside(text: str, fn) -> str:
    """Apply fn to citations not already followed by a link."""
    out, last = [], 0
    for m in CITATION_RE.finditer(text):
        out.append(text[last:m.start()])
        out.append(m.group(0) if text.startswith("(file:", m.end()) else fn(m))
        last = m.end()
    out.append(text[last:])
    return "".join(out)
