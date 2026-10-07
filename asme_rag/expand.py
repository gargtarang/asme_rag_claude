"""Cross-reference expansion: reference packs, `refs`, `rulemap`. MAXIMUM DEPTH IS 1.

Purpose: when a question is asked, collect everything the anchor clauses DIRECTLY refer to, show it
inline with citations and as a reference pack, so nobody has to jump around the PDF.
Inputs: database (chunks + links from phase 2/3), config [xref], anchor chunks (retrieval hits or a clause).
Outputs: Pack (level 0, level 1, selected expansions, further references not expanded, unresolved)
and Markdown renderings. Data flow: storage.hybrid_search -> expand.build_pack -> rag.answer / cli.
Rules: level 0 = anchors; level 1 = what the anchors refer to; level-1 items are NOT searched for
onward references (those are listed, numbered, as 'Further references (not expanded)'). Footnotes and
table notes belong to their item (not a depth level). The code (never the chat model) decides what is
referenced. The limit is the constant MAX_DEPTH_HARD below.
How to change: MAX_DEPTH_HARD, cap constants, and the [xref] block in config.toml.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable

from asme_rag import storage
from asme_rag.rag import cite

# EDIT HERE: hard maximum expansion depth. config.toml [xref] max_depth above this is rejected.
MAX_DEPTH_HARD = 1
PART_PREFIX_CAP = 8            # chunks collected for a 'Part 4.12' style reference
NOTE_KINDS = ("footnote", "table_note")
BODY_TYPES = ("text", "table", "figure_caption", "equation", "figure_description")
TARGET_ID_RE = re.compile(r"(Part|Table|Fig\.|para\.|paragraph)\s*([A-Z0-9][A-Z0-9\-\.]*?)\.?$", re.I)


def xref_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read [xref] with defaults and reject a max_depth above the hard maximum."""
    x = {"max_depth": 1, "max_clauses": 40, "ask_when_larger_than": 40, "include_footnotes": True,
         "show_unexpanded": True, "write_pack_file": True, "prompt_to_expand": True, "prompt_max_items": 10}
    x.update(cfg.get("xref", {}))
    if int(x["max_depth"]) > MAX_DEPTH_HARD:
        raise ValueError(f"config xref.max_depth = {x['max_depth']} is above the hard maximum "
                         f"{MAX_DEPTH_HARD} (constant MAX_DEPTH_HARD in asme_rag/expand.py)")
    return x


@dataclass
class Item:
    """One collected chunk with the reason it is in the pack."""
    chunk: dict[str, Any]
    group: str                      # level0 / level1 / selected
    why: str
    notes: list[dict[str, Any]] = field(default_factory=list)   # footnotes and table notes (own text)
    refs_inline: list[str] = field(default_factory=list)        # ids of level-1 items referenced from here


@dataclass
class Further:
    """A reference found inside a level-1 item (or selected item). Listed, never followed automatically."""
    n: int
    parent_id: str
    parent_cite: str
    kind: str
    target: str
    certainty: str
    status: str
    target_doc: str
    detail: str = ""


@dataclass
class Unresolved:
    """A reference that could not be resolved (book not ingested, or target not found)."""
    parent_cite: str
    kind: str
    target: str
    certainty: str
    reason: str
    inline: str = ""                # text shown in place, e.g. 'NOT INGESTED: VIII-2, Part 4.12'


@dataclass
class Pack:
    items: list[Item] = field(default_factory=list)
    further: list[Further] = field(default_factory=list)
    unresolved: list[Unresolved] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)     # size-cap messages etc.

    def ids(self) -> set[str]:
        return {i.chunk["id"] for i in self.items}

    def group(self, g: str) -> list[Item]:
        return [i for i in self.items if i.group == g]


def _chunk(con: sqlite3.Connection, cid: str) -> dict[str, Any] | None:
    r = con.execute("select * from chunks where id=?", (cid,)).fetchone()
    return dict(r) if r else None


def _links_from(con: sqlite3.Connection, cid: str) -> list[sqlite3.Row]:
    return con.execute("select rowid as rid, * from links where from_chunk=? order by rid", (cid,)).fetchall()


def _base_clause(target: str) -> str:
    return re.sub(r"(\([A-Za-z0-9]+\))+$", "", target).strip()


def _clause_chunks(con: sqlite3.Connection, doc: str, clause: str) -> list[dict[str, Any]]:
    rows = con.execute("select * from chunks where doc_id=? and lower(clause_id)=lower(?) and type in "
                       "('text','table','figure_caption','equation','figure_description') order by rowid",
                       (doc, clause)).fetchall()
    return [dict(r) for r in rows]


def resolve_link(con: sqlite3.Connection, link: sqlite3.Row, cap: int) -> tuple[list[dict[str, Any]], Unresolved | None, str]:
    """Resolve one link to target chunks. Returns (chunks, unresolved-or-None, parent-less reason text)."""
    kind, target, doc = link["kind"], link["target"], link["target_doc"]
    mk = lambda reason, inline="": Unresolved("", kind, target, link["certainty"], reason, inline)
    if kind == "clause":
        got = _clause_chunks(con, doc, _base_clause(target))
        return (got, None, "") if got else ([], mk("target clause not found"), "")
    if kind in ("table", "figure"):
        got = _clause_chunks(con, doc, target)
        return (got, None, "") if got else ([], mk("target not found"), "")
    if kind == "range":
        a, b = [s.strip() for s in target.split(" through ")]
        rows = [dict(r) for r in con.execute("select * from chunks where doc_id=? order by rowid", (doc,))]
        ids = [r["clause_id"].lower() for r in rows]
        if a.lower() not in ids or b.lower() not in ids:
            return [], mk("range endpoint not found"), ""
        lo, hi = ids.index(a.lower()), len(ids) - 1 - ids[::-1].index(b.lower())
        got = [r for r in rows[lo:hi + 1] if r["type"] in BODY_TYPES]
        return got[:cap], None, f"range {target}"
    if kind == "book":
        if not con.execute("select 1 from chunks where doc_id=? limit 1", (doc,)).fetchone():
            return [], mk("book not ingested", f"NOT INGESTED: {doc}, {target}"), ""
        m = TARGET_ID_RE.search(target)
        if not m:
            return [], mk("no clause id in the reference"), ""
        ident = m.group(2).rstrip(".")
        wanted = f"Table {ident.upper()}" if m.group(1).lower() == "table" else ident
        got = _clause_chunks(con, doc, wanted)
        if not got and m.group(1).lower() == "part":
            rows = con.execute("select * from chunks where doc_id=? and (clause_id=? or clause_id like ?) "
                               "and type in ('text','table','figure_caption','equation') order by rowid limit ?",
                               (doc, ident, ident + ".%", PART_PREFIX_CAP)).fetchall()
            got = [dict(r) for r in rows]
        if not got:
            return [], mk("target not found in the ingested book (whole-part reference?)",
                          f"NOT FOUND in ingested {doc}: {target}"), ""
        return got, None, ""
    # appendix / part references point at headings, not clauses: possible (fuzzy), listed as unresolved
    return [], mk("heading reference (no clause id)"), ""


def notes_of(con: sqlite3.Connection, chunk: dict[str, Any], settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Footnotes (linked by superscript marker) and table notes belong to the item: return their chunks."""
    if not settings["include_footnotes"]:
        return []
    out = []
    for l in _links_from(con, chunk["id"]):
        if l["kind"] in NOTE_KINDS:
            n = _chunk(con, l["detail"])
            if n:
                out.append(n)
    return out


def _add(pack: Pack, con, chunk: dict[str, Any], group: str, why: str, settings: dict[str, Any]) -> Item | None:
    if chunk["id"] in pack.ids():
        return None                 # loops and repeats never add an item twice
    item = Item(chunk, group, why, notes_of(con, chunk, settings))
    pack.items.append(item)
    return item


def _step(con, pack: Pack, parent: dict[str, Any], settings: dict[str, Any], group: str, why_prefix: str,
          cap_left: Callable[[], int]) -> list[Item]:
    """One reference step from `parent`: add what it directly refers to as `group` items."""
    added: list[Item] = []
    for l in _links_from(con, parent["id"]):
        if l["kind"] in NOTE_KINDS:
            continue
        chunks, unres, _ = resolve_link(con, l, max(cap_left(), 0))
        if unres:
            unres.parent_cite = cite(parent)
            pack.unresolved.append(unres)
            continue
        for c in chunks:
            if cap_left() <= 0:
                pack.notes.append(f"Collection stopped at max_clauses = {settings['max_clauses']}.")
                return added
            it = _add(pack, con, c, group, f"{why_prefix} {cite(parent)} refers to {l['target']} ({l['certainty']})", settings)
            if it:
                added.append(it)
    return added


def _add_further(con, pack: Pack, parent: dict[str, Any]) -> None:
    """List (not follow) the onward references of a level-1 item. Targets already in the pack are skipped."""
    seen = {(f.target.lower(), f.target_doc) for f in pack.further}
    for l in _links_from(con, parent["id"]):
        if l["kind"] in NOTE_KINDS:
            continue
        chunks, unres, _ = resolve_link(con, l, 5)
        if chunks and all(c["id"] in pack.ids() for c in chunks):
            continue
        key = (l["target"].lower(), l["target_doc"])
        if key in seen:
            continue
        seen.add(key)
        pack.further.append(Further(len(pack.further) + 1, parent["id"], cite(parent), l["kind"], l["target"],
                                    l["certainty"], "unresolved" if unres else "resolved", l["target_doc"],
                                    unres.reason if unres else ""))


def build_pack(con: sqlite3.Connection, cfg: dict[str, Any], anchors: list[dict[str, Any]],
               ask: Callable[[str], str] | None = None) -> Pack:
    """Level 0 = anchors; level 1 = their direct references; onward references listed, not followed."""
    st = xref_settings(cfg)
    pack = Pack()
    for a in anchors:
        _add(pack, con, a, "level0", "anchor found by retrieval", st)
    cap_left = lambda: int(st["max_clauses"]) - len(pack.items)
    larger = int(st["ask_when_larger_than"])
    for a in [i.chunk for i in pack.group("level0")]:
        _step(con, pack, a, st, "level1", "directly referred:", cap_left)
    level1 = pack.group("level1")
    if len(pack.items) > larger and ask is not None:
        reply = ask(f"{len(pack.items)} items collected (more than {larger}). Include: all / first N / clauses only? ").strip().lower()
        if reply.startswith("clauses"):
            pack.items = [i for i in pack.items if i.group == "level0" or i.chunk["type"] == "text"]
        elif reply.isdigit():
            keep = int(reply)
            pack.items = pack.group("level0") + pack.group("level1")[:keep]
        pack.notes.append(f"Size control: answered '{reply}'.")
    elif len(pack.items) > larger:
        pack.notes.append(f"{len(pack.items)} items collected (above {larger}); no one to ask, kept all up to max_clauses.")
    if st["show_unexpanded"]:
        for i in pack.group("level1"):
            _add_further(con, pack, i.chunk)
    return pack


def expand_selection(con: sqlite3.Connection, cfg: dict[str, Any], pack: Pack, picks: list[int]) -> Pack:
    """Expand each chosen 'further' reference by EXACTLY one step from its parent.

    The new items go under 'Selected expansions'. Their own onward references are listed (numbered
    after the existing ones) but never followed.
    """
    st = xref_settings(cfg)
    cap_left = lambda: int(st["max_clauses"]) - len(pack.items)
    for n in picks:
        f = next((x for x in pack.further if x.n == n), None)
        if f is None:
            pack.notes.append(f"Further reference #{n} does not exist.")
            continue
        row = next((l for l in _links_from(con, f.parent_id)
                    if l["target"] == f.target and l["target_doc"] == f.target_doc), None)
        if row is None:
            continue
        chunks, unres, _ = resolve_link(con, row, max(cap_left(), 0))
        if unres:
            unres.parent_cite = f.parent_cite
            pack.unresolved.append(unres)
            continue
        for c in chunks:
            it = _add(pack, con, c, "selected", f"selected expansion #{n}: {f.parent_cite} refers to {f.target}", st)
            if it:
                _add_further(con, pack, c)
    return pack


def parse_selection(reply: str, maximum: int) -> list[int]:
    """'2,5' -> [2, 5]; 'all' -> every number; 'none' or '' -> []. Out-of-range numbers are dropped."""
    r = reply.strip().lower()
    if r in ("", "none", "n"):
        return []
    if r == "all":
        return list(range(1, maximum + 1))
    return sorted({int(x) for x in re.split(r"[,\s]+", r) if x.isdigit() and 1 <= int(x) <= maximum})


def prompt_expand(con, cfg: dict[str, Any], pack: Pack, ask: Callable[[str], str] | None) -> Pack:
    """Print-free helper: ask 'Expand which?' once and run one step per chosen item (if prompting is on)."""
    st = xref_settings(cfg)
    if not (st["prompt_to_expand"] and pack.further and ask):
        return pack
    picks = parse_selection(ask("Expand which? (e.g. 2,5 / none / all) "), len(pack.further))
    return expand_selection(con, cfg, pack, picks) if picks else pack


# ---------------------------------------------------------------- rendering

def _snip(text: str, n: int = 300) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[:n].rstrip() + " ..."


def render_inline(pack: Pack, con: sqlite3.Connection) -> str:
    """Each anchor statement with its citation, followed directly by what it refers to (with citations)."""
    lines = []
    for a in pack.group("level0"):
        parts = []
        for l in _links_from(con, a.chunk["id"]):
            if l["kind"] in NOTE_KINDS:
                continue
            chunks, unres, _ = resolve_link(con, l, 5)
            for c in chunks[:3]:
                parts.append(f"see {cite(c)}: {_snip(c['text'], 160)}")
            if unres:
                parts.append(unres.inline or f"UNRESOLVED ({unres.certainty}): {l['target']} - {unres.reason}")
        for n in a.notes:
            label = "table note" if n["type"] == "note" else f"footnote [p.{n['pdf_page']}]"
            parts.append(f"{label}: {_snip(n['text'], 160)}")
        lines.append(f"- {_snip(a.chunk['text'], 200)} {cite(a.chunk)}" + (" -> " + " | ".join(parts) if parts else ""))
    return "\n".join(lines) or "No anchors."


def render_pack(pack: Pack) -> str:
    """Markdown reference pack: Level 0, Level 1, Selected expansions, Further (not expanded), Unresolved."""
    out = []
    titles = [("level0", "Level 0 (anchors)"), ("level1", "Level 1 (directly referred)"),
              ("selected", "Selected expansions")]
    for g, title in titles:
        items = pack.group(g)
        if not items and g == "selected":
            continue
        out += [f"## {title}", ""]
        if not items:
            out += ["None.", ""]
        for i in items:
            c = i.chunk
            out += [f"### {cite(c)} {c['type']}", f"Why included: {i.why}", "", "```text", c["text"], "```"]
            for n in i.notes:
                label = "Table note" if n["type"] == "note" else f"Footnote (PDF p.{n['pdf_page']})"
                out += [f"{label}:", "```text", n["text"], "```"]
            out.append("")
    out += ["## Further references (not expanded)", ""]
    if not pack.further:
        out += ["None.", ""]
    for certainty in ("certain", "fuzzy"):
        grp = [f for f in pack.further if f.certainty == certainty]
        if grp:
            out.append("Certain:" if certainty == "certain" else "Possible (fuzzy):")
            out += [f"{f.n}. {f.target} ({f.kind}) - from {f.parent_cite}" + (f" - {f.detail}" if f.detail else "")
                    for f in grp] + [""]
    out += ["## Unresolved", ""]
    if not pack.unresolved:
        out += ["None.", ""]
    for certainty in ("certain", "fuzzy"):
        grp = [u for u in pack.unresolved if u.certainty == certainty]
        if grp:
            out.append("Certain:" if certainty == "certain" else "Possible (fuzzy):")
            out += [f"- {u.inline or u.target} ({u.kind}) - from {u.parent_cite} - {u.reason}" for u in grp] + [""]
    if pack.notes:
        out += ["## Notes", ""] + [f"- {n}" for n in pack.notes] + [""]
    return "\n".join(out)


def needs_input(pack: Pack) -> list[str]:
    """NEEDS INPUT lines for every certain unresolved reference (e.g. a book that is not ingested)."""
    return [f"NEEDS INPUT: {u.inline or u.target} - {u.reason} (from {u.parent_cite})"
            for u in pack.unresolved if u.certainty == "certain"]


# ---------------------------------------------------------------- rulemap

def rulemap(con, client, cfg: dict[str, Any], topic: str, filters: dict[str, Any] | None = None,
            ask: Callable[[str], str] | None = None) -> tuple[str, Pack]:
    """Rule map for a design topic: governing clauses, direct references, tables/charts, cross-book hops.

    Deterministic parts come from the pack. 'Inputs required' is asked from the model on the pack text
    only (grounded prompt) and is labelled as model output; without a client it is skipped.
    """
    hits = storage.hybrid_search(con, client, cfg, topic, filters)
    pack = build_pack(con, cfg, [h.chunk for h in hits], ask)
    lvl0, lvl1 = pack.group("level0"), pack.group("level1")
    out = [f"# Rule map: {topic}", "", "## Governing clauses (level 0)", ""]
    out += [f"- {cite(i.chunk)} {i.chunk['type']}" for i in lvl0] or ["None found."]
    out += ["", "## Direct references (level 1)", ""]
    out += [f"- {cite(i.chunk)} {i.chunk['type']} - {i.why}" for i in lvl1] or ["None."]
    tables = [i for i in pack.items if i.chunk["type"] in ("table", "figure_caption", "figure_description")]
    out += ["", "## Tables and charts needed", ""]
    out += [f"- {cite(i.chunk)} {i.chunk['clause_id']}" for i in tables] or ["None found."]
    hops = con.execute("select kind, target, status, target_doc from links where kind='book' and from_chunk in (%s)"
                       % ",".join("?" * len(lvl0)), [i.chunk["id"] for i in lvl0]).fetchall() if lvl0 else []
    out += ["", "## Cross-book first hops", ""]
    out += [f"- {h['target']} -> {h['target_doc']} ({h['status']})" for h in hops] or ["None."]
    out += ["", "## Inputs required (model-generated from the pack text, UNVERIFIED)", ""]
    if client is not None and pack.items:
        text = "\n\n".join(f"[E{n}] {cite(i.chunk)}\n{i.chunk['text']}" for n, i in enumerate(pack.items, 1))
        from asme_rag import rag
        msgs = [{"role": "system", "content": rag.SYSTEM_PROMPT},
                {"role": "user", "content": f"Excerpts:\n\n{text}\n\nList every input (geometry, loads, material "
                                            f"data, tables or charts) needed to apply these rules to: {topic}. "
                                            f"Do not compute anything."}]
        out.append(rag.ensure_verify_line(client.chat(msgs)))
    else:
        out.append("Not generated.")
    out += ["", "## Further references (not expanded)", ""]
    out += [f"{f.n}. {f.target} - from {f.parent_cite}" for f in pack.further] or ["None."]
    ni = needs_input(pack)
    out += ["", "## NEEDS INPUT", ""] + (ni or ["None."])
    return "\n".join(out) + "\n", pack
