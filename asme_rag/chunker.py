"""Chunker: turns ordered page Blocks into Chunks with mandatory metadata.

Purpose: detect parts, appendices, clause headings, tables, figure captions, notes, footnotes and
equations; split long clauses at sub-paragraph boundaries while keeping the parent clause ID.
Inputs: list of (page_no, list[Block]) from pdf_parser, one document config dict, max_chars.
Outputs: list[Chunk]. Data flow: pdf_parser -> chunker -> storage (phase 3) / xref (this phase).
Profiles: code_book, materials_data, material_specs, standard, change_summary.
How to change: edit PROFILES below (regexes marked '# EDIT HERE:') or override per profile in
config.toml under [chunker.<profile>] with clause_id_regexes = [...].
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from asme_rag.pdf_parser import Block

# EDIT HERE: clause-heading regexes per profile. Group 1 must capture the clause ID.
PROFILES: dict[str, dict[str, Any]] = {
    "code_book": {
        "clause_id_regexes": [
            r"^(U[A-Z]{1,2}-\d+(?:\.\d+)?)\b",           # VIII-1 style: UG-37, UW-16.1
            r"^(\d{1,2}-\d+(?:\.\d+)?)\b",                # appendix style: 1-1, 13-9
            r"^(\d{1,2}\.\d+(?:\.\d+)*)\s+[A-Z]",         # VIII-2 style: 4.12.1 Title
        ],
        "part_regex": r"^(Part\s+U[A-Z]{1,3}|PART\s+\d+)\b",
        "appendix_regex": r"^((?:Mandatory|Nonmandatory)\s+Appendix\s+[A-Z0-9]+)",
    },
    "materials_data": {"clause_id_regexes": [], "part_regex": r"^(Subpart\s+\d+)\b",
                       "appendix_regex": r"^(Mandatory Appendix\s+[A-Z0-9]+)"},
    "material_specs": {"clause_id_regexes": [r"^(\d{1,2})\.\s+[A-Z]"],      # section number inside a spec
                       "part_regex": r"^(S[AB]-\d+[A-Z]?(?:/S[AB]-\d+[A-Z]?)?)\b",   # EDIT HERE: spec number
                       "appendix_regex": r"^(Appendix\s+[A-Z0-9]+)"},
    "standard": {"clause_id_regexes": [r"^(\d{1,2}(?:\.\d+)+)\s+[A-Z]"],
                 "part_regex": r"^$^", "appendix_regex": r"^((?:Mandatory|Nonmandatory)\s+Appendix\s+[A-Z0-9]+)"},
    "change_summary": {"clause_id_regexes": [], "part_regex": r"^$^", "appendix_regex": r"^$^"},
}
# EDIT HERE: table / figure / note / footnote / equation detection.
TABLE_RE = re.compile(r"^(Table\s+[A-Z0-9][A-Z0-9\-\.]*)\b", re.I)
FIGURE_RE = re.compile(r"^(FIG\.|Fig\.|Figure)\s+([A-Z0-9][A-Z0-9\-\.]*)", re.I)
NOTE_RE = re.compile(r"^(GENERAL NOTES?|NOTES?)\b[:\s]", re.I)
FOOTNOTE_RE = re.compile(r"^(\d{1,2})\s+\S")                   # footnote body starts with its marker
SUBPARA_RE = re.compile(r"^\(([a-z0-9]{1,3})\)\s", re.M)
EQUATION_HINT_RE = re.compile(r"(?:^|\s)(?:=|\u2264|\u2265|\u00d7)\s*\S|[A-Za-z]\s*=\s*\S")
# EDIT HERE: characters suggesting garbled extraction (private-use area, replacement char, controls).
GARBLE_RE = re.compile(r"[\ue000-\uf8ff\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass
class Chunk:
    """A retrievable unit. Fields listed in section 3 of the build prompt are all present."""
    id: str
    doc_id: str
    code: str
    division: str
    edition: int
    part_or_subsection: str
    clause_id: str
    pdf_page: int
    type: str                         # text/table/figure_caption/figure_description/equation/footnote/note
    text: str
    model_generated: bool = False
    verified: bool = False
    status: str = "mandatory"         # mandatory / non-mandatory / supplementary
    heading_path: list[str] = field(default_factory=list)
    bold_flags: bool = False
    rows: list[list[str]] = field(default_factory=list)   # structured table rows
    unreliable: bool = False          # garbled text or inconsistent table columns
    superscripts: list[str] = field(default_factory=list)
    linked_to: str = ""               # clause ID of the table/clause a note belongs to


def parse_table_rows(text: str) -> tuple[list[list[str]], bool]:
    """Split table text into rows of cells (2+ spaces or tabs separate cells).

    Returns (rows, unreliable). Unreliable if column counts differ by more than one across data rows.
    """
    rows = [re.split(r"\s{2,}|\t", ln.strip()) for ln in text.splitlines()[1:] if ln.strip()]
    counts = {len(r) for r in rows}
    return rows, bool(rows) and (max(counts) - min(counts) > 1)


class _State:
    """Mutable position while walking a document (kept local to one chunk_document call)."""
    def __init__(self) -> None:
        self.part = ""
        self.appendix = ""
        self.clause = ""
        self.title = ""
        self.status = "mandatory"
        self.buf: list[Block] = []
        self.last_table = ""


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split at sub-paragraph starts '(a)' when text exceeds max_chars; otherwise at line breaks."""
    if len(text) <= max_chars:
        return [text]
    cuts = [m.start() for m in SUBPARA_RE.finditer(text)] or \
           [m.end() for m in re.finditer(r"\n", text)]
    pieces, start, last_ok = [], 0, 0
    for c in cuts + [len(text)]:
        if c - start > max_chars and last_ok > start:
            pieces.append(text[start:last_ok].strip())
            start = last_ok
        last_ok = c
    pieces.append(text[start:].strip())
    out: list[str] = []
    for p in pieces:                   # hard split anything still too long (no boundary available)
        out.extend(p[i:i + max_chars] for i in range(0, len(p), max_chars)) if len(p) > max_chars else out.append(p)
    return [p for p in out if p]


def chunk_document(pages: list[tuple[int, list[Block]]], doc: dict[str, Any],
                   max_chars: int = 3000, profile_override: dict[str, Any] | None = None) -> list[Chunk]:
    """Walk blocks in order and emit Chunks. `doc` is one [[documents]] entry from config.toml."""
    prof = dict(PROFILES[doc["parser_profile"]])
    prof.update(profile_override or {})
    id_res = [re.compile(r, re.M) for r in prof["clause_id_regexes"]]
    part_re, app_re = re.compile(prof["part_regex"]), re.compile(prof["appendix_regex"])
    st, out = _State(), []
    counter = {"n": 0}

    def path() -> list[str]:
        return [x for x in (st.part, st.appendix, st.clause) if x]

    def make(page: int, typ: str, text: str, clause: str, blocks: list[Block], **kw: Any) -> Chunk:
        counter["n"] += 1
        return Chunk(id=f"{doc['id']}:{counter['n']:05d}", doc_id=doc["id"], code=doc["code"],
                     division=doc.get("division", ""), edition=doc["edition"],
                     part_or_subsection=st.appendix or st.part, clause_id=clause, pdf_page=page,
                     type=typ, text=text, status=st.status, heading_path=path(),
                     bold_flags=any(b.bold for b in blocks),
                     superscripts=[s for b in blocks for s in b.superscripts],
                     unreliable=bool(GARBLE_RE.search(text)) or kw.pop("unreliable", False), **kw)

    def flush() -> None:
        """Emit the buffered clause body as one or more chunks (split at sub-paragraphs)."""
        if not st.buf:
            return
        text = "\n".join(b.text for b in st.buf)
        page = st.buf[0].page
        typ = "equation" if EQUATION_HINT_RE.search(text) and len(text) < 300 and not SUBPARA_RE.search(text) else "text"
        for piece in _split_long(text, max_chars):
            out.append(make(page, typ, piece, st.clause, st.buf))
        st.buf = []

    for _page_no, blocks in pages:
        for b in blocks:
            first = b.text.splitlines()[0].strip()
            m_app, m_part = app_re.match(first), part_re.match(first)
            m_tab, m_fig = TABLE_RE.match(first), FIGURE_RE.match(first)
            m_clause = next((m for r in id_res if (m := r.match(first))), None)
            short_or_bold = b.bold or len(first) < 120
            if m_app and short_or_bold:
                flush(); st.appendix = m_app.group(1); st.clause = ""
                st.status = "non-mandatory" if m_app.group(1).lower().startswith("non") else "mandatory"
            elif m_part and short_or_bold:
                flush(); st.part = m_part.group(1); st.appendix = ""; st.clause = ""; st.status = "mandatory"
            elif m_tab and short_or_bold:
                flush()
                cid = "Table " + m_tab.group(1).split(None, 1)[1].rstrip(".").upper()
                rows, bad = parse_table_rows(b.text)
                out.append(make(b.page, "table", b.text, cid, [b], rows=rows, unreliable=bad))
                st.last_table = cid
            elif m_fig and short_or_bold and len(b.text) < 400:
                flush()
                out.append(make(b.page, "figure_caption", b.text, f"Fig. {m_fig.group(2)}", [b]))
            elif NOTE_RE.match(first):
                flush()
                out.append(make(b.page, "note", b.text, st.clause or st.last_table, [b], linked_to=st.last_table or st.clause))
            elif m_clause and b.bold:
                flush(); st.clause = m_clause.group(1); st.buf = [b]
            elif FOOTNOTE_RE.match(first) and b.size and b.size < 8:
                flush()
                out.append(make(b.page, "footnote", b.text, st.clause, [b], linked_to=st.clause))
            else:
                st.buf.append(b)
    flush()
    return out
