"""Citation check: confirm that every citation in an answer matches a retrieved excerpt.

Purpose: catch invented or wrong citations. The tool never silently corrects them: unmatched
citations are listed as UNVERIFIED CITATION.
Inputs: answer text, list of storage.Hit (the excerpts the model saw).
Outputs: list[CiteResult] and a Markdown block. Data flow: rag.answer -> citecheck -> rag.render_markdown.
A citation is [code, edition, clause, PDF p.N]; it must match ONE excerpt on all four parts.
A sub-paragraph suffix in the cited clause (UG-37(a)) matches the excerpt clause UG-37.
How to change: CITATION_RE and BRACKET_PAGE_RE below (marked '# EDIT HERE:').
Link rewriting for PDF pages (phase 4d) happens AFTER this check and must keep the link text intact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# EDIT HERE: the citation pattern the model is told to use: [code, edition, clause, PDF p.N]
CITATION_RE = re.compile(r"\[([^\[\],]+?),\s*(\d{4}),\s*([^\[\]]+?),\s*PDF\s*p\.?\s*(\d+)\s*\]")
# Anything that looks like a citation (mentions 'PDF p.') but did not parse is reported as malformed.
BRACKET_PAGE_RE = re.compile(r"\[[^\[\]]*PDF\s*p\.?[^\[\]]*\]")
SUBPARA_SUFFIX_RE = re.compile(r"(\([A-Za-z0-9]+\))+$")


@dataclass
class CiteResult:
    """Outcome for one citation found in the answer."""
    text: str
    status: str                 # matched / unverified
    reason: str = ""
    excerpt: int = 0            # 1-based excerpt number when matched
    warning: str = ""           # e.g. cites a model-generated chunk


def _code_matches(cited: str, chunk: dict[str, Any]) -> bool:
    """Cited code text must contain the chunk's division (or code) or vice versa, case-insensitive."""
    c = cited.lower().strip()
    full = f"{chunk['code']} {chunk['division']}".lower().strip()
    return c in full or full in c or (bool(chunk["division"]) and chunk["division"].lower() in c) \
        or (not chunk["division"] and chunk["code"].lower() in c)


def check_citations(answer: str, hits: list[Any]) -> list[CiteResult]:
    """Parse every citation in `answer` and match it against `hits` (storage.Hit objects)."""
    results: list[CiteResult] = []
    parsed_spans = []
    for m in CITATION_RE.finditer(answer):
        parsed_spans.append(m.span())
        code, edition, clause, page = m.group(1).strip(), int(m.group(2)), m.group(3).strip(), int(m.group(4))
        base = SUBPARA_SUFFIX_RE.sub("", clause).lower()
        reasons, found = [], None
        for i, h in enumerate(hits, 1):
            c = h.chunk
            ok_code, ok_ed = _code_matches(code, c), int(c["edition"]) == edition
            ok_cl, ok_pg = (c["clause_id"] or "").lower() == base, int(c["pdf_page"]) == page
            if ok_code and ok_ed and ok_cl and ok_pg:
                found = (i, c)
                break
        if found:
            warn = "cites a model-generated chunk (unverified; not a source of values)" \
                if found[1].get("model_generated") else ""
            results.append(CiteResult(m.group(0), "matched", excerpt=found[0], warning=warn))
        else:
            same_clause = [h for h in hits if (h.chunk["clause_id"] or "").lower() == base]
            if not same_clause:
                reasons.append("clause not in retrieved excerpts")
            elif not any(int(h.chunk["pdf_page"]) == page for h in same_clause):
                reasons.append("clause found but on a different PDF page: "
                               + ", ".join(sorted({str(h.chunk['pdf_page']) for h in same_clause})))
            elif not any(int(h.chunk["edition"]) == edition for h in same_clause):
                reasons.append("edition does not match")
            else:
                reasons.append("code/book does not match")
            results.append(CiteResult(m.group(0), "unverified", "; ".join(reasons)))
    for m in BRACKET_PAGE_RE.finditer(answer):
        if not any(s <= m.start() and m.end() <= e for s, e in parsed_spans):
            results.append(CiteResult(m.group(0), "unverified", "malformed citation (expected [code, edition, clause, PDF p.N])"))
    return results


def render_check(results: list[CiteResult], answer: str) -> str:
    """Markdown block listing matched counts and every UNVERIFIED CITATION."""
    if not results:
        if answer.lower().lstrip().startswith("not found"):
            return "No citations (answer is 'not found')."
        return "WARNING: the answer contains no parsable citations. Treat every statement as unverified."
    bad = [r for r in results if r.status == "unverified"]
    lines = [f"{len(results) - len(bad)} of {len(results)} citations matched a retrieved excerpt."]
    for r in results:
        if r.status == "matched" and r.warning:
            lines.append(f"- WARNING {r.text} (excerpt E{r.excerpt}): {r.warning}")
    for r in bad:
        lines.append(f"- UNVERIFIED CITATION: {r.text} - {r.reason}")
    return "\n".join(lines)


def passed(results: list[CiteResult], answer: str) -> bool:
    """True if there is no unverified citation and (citations exist or the answer is 'not found')."""
    if any(r.status == "unverified" for r in results):
        return False
    return bool(results) or answer.lower().lstrip().startswith("not found")
