"""Cross-reference index: parse references in chunks and link them to targets.

Purpose: find "see UG-37", ranges, tables, figures, appendices, other books, and footnote markers.
Inputs: list[Chunk] from the chunker (all ingested documents together), config documents list.
Outputs: list[Link] with from_chunk, kind, target, certainty (certain/fuzzy), status
(resolved / unresolved because the book is not ingested). Data flow: chunker -> xref -> storage/rag.
Depth: expansion limited to depth 1 (code lives in phase 4c; this module only builds the links).
How to change: edit REF_PATTERNS (marked '# EDIT HERE:') or BOOK_MAP for other books.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from asme_rag.chunker import Chunk

# EDIT HERE: map phrases in the text to doc division / doc id keys. Values are config document divisions.
BOOK_MAP = [
    (re.compile(r"Division\s*2|VIII-2", re.I), "VIII-2"),
    (re.compile(r"Division\s*1|VIII-1", re.I), "VIII-1"),
    (re.compile(r"Section\s*II,?\s*Part\s*D|II-D", re.I), "II-D"),
    (re.compile(r"Section\s*II,?\s*Part\s*A|II-A", re.I), "II-A"),
    (re.compile(r"B16\.5", re.I), "B16.5"),
    (re.compile(r"B36\.10", re.I), "B36.10"),
    (re.compile(r"B16\.47", re.I), "B16.47"),
    (re.compile(r"B31\.3", re.I), "B31.3"),
    (re.compile(r"Section\s*IX", re.I), "IX"),
]
# EDIT HERE: reference patterns. Order matters: ranges before single clauses.
CLAUSE = r"U[A-Z]{1,2}-\d+(?:\.\d+)?(?:\([a-z0-9]+\))*"
RANGE_RE = re.compile(rf"({CLAUSE})\s+(?:through|to)\s+({CLAUSE})")
CLAUSE_RE = re.compile(rf"\b({CLAUSE})")
TABLE_REF_RE = re.compile(r"\bTables?\s+([A-Z0-9][A-Z0-9\-\.]*)")
FIG_REF_RE = re.compile(r"\bFig(?:ure|s?\.)\s+([A-Z0-9][A-Z0-9\-\.]*)")
APPENDIX_RE = re.compile(r"\b((?:Mandatory|Nonmandatory)\s+Appendix\s+[A-Z0-9]+|Appendix\s+\d+)")
PART_RE = re.compile(r"\bPart\s+(U[A-Z]{1,3}|\d+(?:\.\d+)*[A-Z]?)\b")
OTHER_BOOK_PART_RE = re.compile(r"((?:Division\s*[12]|Section\s*II,?\s*Part\s*[AD]|B16\.5|B36\.10|B16\.47|B31\.3|Section\s*IX)"
                                r"[^.;]{0,40}?(?:Part|Table|Fig\.|para\.|paragraph)\s*[A-Z0-9][A-Z0-9\-\.]*)", re.I)


@dataclass
class Link:
    """One reference found in a chunk."""
    from_chunk: str
    kind: str          # clause / range / table / figure / appendix / part / book / footnote / table_note
    target: str
    certainty: str     # certain / fuzzy
    status: str        # resolved / unresolved
    target_doc: str = ""        # doc id when resolved or the book name when unresolved
    detail: str = ""


def _doc_for_division(docs: list[dict[str, Any]], division: str) -> str | None:
    """Return the doc id whose division or code matches (e.g. 'VIII-2', 'B16.5')."""
    for d in docs:
        if d.get("division") == division or d["code"].endswith(division):
            return d["id"]
    return None


def build_index(chunks: list[Chunk], docs: list[dict[str, Any]]) -> list[Link]:
    """Parse all chunks and return Links. Deterministic order: by chunk id, then position in text."""
    by_doc_clause: dict[tuple[str, str], str] = {}
    for c in chunks:
        if c.clause_id:
            by_doc_clause.setdefault((c.doc_id, c.clause_id.lower()), c.id)
    links: list[Link] = []
    for c in chunks:
        text, doc_id = c.text, c.doc_id
        # 1. Other-book references first; their targets must not be resolved inside this book.
        masked = text
        for m in OTHER_BOOK_PART_RE.finditer(text):
            book = next((name for rx, name in BOOK_MAP if rx.search(m.group(1))), "unknown")
            target_doc = _doc_for_division(docs, book)
            links.append(Link(c.id, "book", m.group(1).strip().rstrip("."), "certain",
                              "resolved" if target_doc else "unresolved",
                              target_doc or book, "other book"))
            masked = masked.replace(m.group(1), " " * len(m.group(1)))
        # 2. Ranges, then single clauses (range endpoints are not repeated as singles).
        for m in RANGE_RE.finditer(masked):
            links.append(Link(c.id, "range", f"{m.group(1)} through {m.group(2)}", "certain",
                              "resolved" if (doc_id, m.group(1).lower()) in by_doc_clause else "unresolved",
                              doc_id, "every clause between the endpoints"))
            masked = masked.replace(m.group(0), " " * len(m.group(0)))
        for m in list(TABLE_REF_RE.finditer(masked)):
            tgt = f"Table {m.group(1).rstrip('.').upper()}"
            if tgt.lower() == c.clause_id.lower():
                continue
            ok = (doc_id, tgt.lower()) in by_doc_clause
            links.append(Link(c.id, "table", tgt, "certain" if ok else "fuzzy",
                              "resolved" if ok else "unresolved", doc_id))
            masked = masked.replace(m.group(0), " " * len(m.group(0)))   # keep 'UG-37.1' out of the clause scan
        for m in list(FIG_REF_RE.finditer(masked)):
            tgt = f"Fig. {m.group(1).rstrip('.').upper()}"
            ok = (doc_id, tgt.lower()) in by_doc_clause
            links.append(Link(c.id, "figure", tgt, "certain" if ok else "fuzzy",
                              "resolved" if ok else "unresolved", doc_id))
            masked = masked.replace(m.group(0), " " * len(m.group(0)))
        for m in CLAUSE_RE.finditer(masked):
            tgt = m.group(1)
            base = re.sub(r"(\([a-z0-9]+\))+$", "", tgt)
            if base.lower() == c.clause_id.lower():
                continue                                    # self-reference
            ok = (doc_id, base.lower()) in by_doc_clause
            links.append(Link(c.id, "clause", tgt, "certain" if ok else "fuzzy",
                              "resolved" if ok else "unresolved", doc_id))
        for m in APPENDIX_RE.finditer(masked):
            links.append(Link(c.id, "appendix", m.group(1), "fuzzy", "unresolved", doc_id,
                              "appendix target is a heading, not a clause ID"))
        for m in PART_RE.finditer(masked):
            links.append(Link(c.id, "part", f"Part {m.group(1)}", "fuzzy", "unresolved", doc_id))
    links.extend(link_footnotes(chunks))
    return links


def link_footnotes(chunks: list[Chunk]) -> list[Link]:
    """Link superscript markers to footnote chunks on the same page, and table notes to their table."""
    links: list[Link] = []
    foot = {(c.doc_id, c.pdf_page, c.text.split()[0]): c for c in chunks if c.type == "footnote"}
    for c in chunks:
        for mark in c.superscripts:
            f = foot.get((c.doc_id, c.pdf_page, mark))
            if f:
                links.append(Link(c.id, "footnote", f"footnote {mark}", "certain", "resolved", c.doc_id, f.id))
        if c.type == "note" and c.linked_to.lower().startswith("table"):
            tbl = next((t for t in chunks if t.type == "table" and t.doc_id == c.doc_id
                        and t.clause_id == c.linked_to), None)
            if tbl:
                links.append(Link(tbl.id, "table_note", c.linked_to, "certain", "resolved", c.doc_id, c.id))
    return links
