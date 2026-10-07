"""SQLite storage, FTS5 keyword search, embeddings and hybrid retrieval.

Purpose: persist chunks, links and embeddings; search them.
Inputs: Chunk / Link objects (chunker, xref), query strings, filters (doc_id, code, edition).
Outputs: Hit objects (chunk row + score). Data flow: chunker/xref -> storage -> rag (phase 4).
Hybrid = keyword (FTS5 bm25) and embedding (cosine) ranks merged with reciprocal-rank fusion.
How to change: STOPWORDS and RRF_K below; top_k and prefixes come from config.toml.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from asme_rag.chunker import Chunk
from asme_rag.xref import Link

# EDIT HERE: words ignored when building the FTS query.
STOPWORDS = {"the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "is", "are", "be", "what",
             "which", "how", "does", "do", "with", "by", "at", "as", "that", "this", "it", "from"}
RRF_K = 60   # EDIT HERE: reciprocal-rank-fusion constant (larger = flatter blending)

SCHEMA = """
create table if not exists chunks(
  id text primary key, doc_id text, code text, division text, edition integer,
  part_or_subsection text, clause_id text, pdf_page integer, type text, text text,
  model_generated integer, verified integer, status text, heading_path text, bold_flags integer,
  rows_json text, unreliable integer, superscripts text, linked_to text);
create index if not exists ix_chunks_doc on chunks(doc_id, clause_id);
create virtual table if not exists chunks_fts using fts5(text, clause_id, heading, tokenize='unicode61');
create table if not exists links(
  from_chunk text, kind text, target text, certainty text, status text, target_doc text, detail text);
create table if not exists embeddings(chunk_id text primary key, dim integer, vec blob);
"""


@dataclass
class Hit:
    """A retrieval result: the chunk row as a dict plus ranking info."""
    chunk: dict[str, Any]
    score: float
    sources: list[str] = field(default_factory=list)    # 'keyword' and/or 'embedding'


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the database and make sure the schema exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def delete_doc(con: sqlite3.Connection, doc_id: str) -> None:
    """Remove all rows of one document (used before re-ingesting it)."""
    ids = [r[0] for r in con.execute("select id from chunks where doc_id=?", (doc_id,))]
    for i in ids:
        con.execute("delete from chunks_fts where rowid=(select rowid from chunks where id=?)", (i,))
        con.execute("delete from embeddings where chunk_id=?", (i,))
        con.execute("delete from links where from_chunk=?", (i,))
    con.execute("delete from chunks where doc_id=?", (doc_id,))
    con.commit()


def insert_chunks(con: sqlite3.Connection, chunks: Iterable[Chunk]) -> int:
    """Insert chunks and their FTS rows. Returns the count."""
    n = 0
    for c in chunks:
        cur = con.execute(
            "insert or replace into chunks values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (c.id, c.doc_id, c.code, c.division, c.edition, c.part_or_subsection, c.clause_id,
             c.pdf_page, c.type, c.text, int(c.model_generated), int(c.verified), c.status,
             json.dumps(c.heading_path), int(c.bold_flags), json.dumps(c.rows), int(c.unreliable),
             json.dumps(c.superscripts), c.linked_to))
        con.execute("insert into chunks_fts(rowid,text,clause_id,heading) values(?,?,?,?)",
                    (cur.lastrowid, c.text, c.clause_id, " ".join(c.heading_path)))
        n += 1
    con.commit()
    return n


def insert_links(con: sqlite3.Connection, links: Iterable[Link]) -> int:
    """Insert cross-reference links. Returns the count."""
    rows = [(l.from_chunk, l.kind, l.target, l.certainty, l.status, l.target_doc, l.detail) for l in links]
    con.executemany("insert into links values(?,?,?,?,?,?,?)", rows)
    con.commit()
    return len(rows)


def build_fts_query(q: str) -> str:
    """Turn free text into a safe FTS5 query.

    Clause IDs such as UG-37 or Table UG-23 are kept as quoted phrases (the hyphen would otherwise
    be parsed as an operator). Other words are quoted individually and joined with OR; stop-words dropped.
    """
    parts: list[str] = []
    for m in re.finditer(r"[A-Za-z]{1,3}-\d+(?:\.\d+)?|\d+-\d+(?:\.\d+)?|[A-Za-z0-9\.]+", q):
        tok = m.group(0).strip(".")
        if not tok or tok.lower() in STOPWORDS:
            continue
        parts.append('"' + tok.replace('"', "") + '"')
    return " OR ".join(dict.fromkeys(parts))      # dict.fromkeys removes duplicates, keeps order


def _filter_sql(filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
    """Build 'and c.col=?' clauses for doc_id, code, edition, division."""
    sql, args = "", []
    for k in ("doc_id", "code", "edition", "division"):
        if filters and filters.get(k) not in (None, ""):
            sql += f" and c.{k}=?"
            args.append(filters[k])
    return sql, args


def keyword_search(con: sqlite3.Connection, query: str, filters: dict[str, Any] | None = None,
                   k: int = 8) -> list[Hit]:
    """FTS5 bm25 search. clause_id matches are weighted higher than body text."""
    fq = build_fts_query(query)
    if not fq:
        return []
    fsql, fargs = _filter_sql(filters)
    rows = con.execute(
        "select c.*, bm25(chunks_fts, 1.0, 5.0, 1.0) as s from chunks_fts "
        "join chunks c on c.rowid = chunks_fts.rowid "
        f"where chunks_fts match ?{fsql} order by s limit ?", [fq, *fargs, k]).fetchall()
    return [Hit(dict(r), -r["s"], ["keyword"]) for r in rows]


def embed_chunks(con: sqlite3.Connection, client: Any, cfg: dict[str, Any], redo: bool = False,
                 batch: int = 16) -> int:
    """Create embeddings for chunks that lack one (all of them if redo). Uses the nomic doc prefix."""
    prefix = cfg["lmstudio"].get("doc_prefix", "search_document: ")
    if redo:
        con.execute("delete from embeddings")
    rows = con.execute("select id, text from chunks where id not in (select chunk_id from embeddings)").fetchall()
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        vecs = client.embed([prefix + r["text"] for r in part])
        for r, v in zip(part, vecs):
            arr = np.asarray(v, dtype=np.float32)
            con.execute("insert or replace into embeddings values(?,?,?)", (r["id"], len(arr), arr.tobytes()))
        con.commit()
    return len(rows)


def vector_search(con: sqlite3.Connection, client: Any, cfg: dict[str, Any], query: str,
                  filters: dict[str, Any] | None = None, k: int = 8) -> list[Hit]:
    """Cosine similarity over stored embeddings, restricted by filters. Uses the nomic query prefix."""
    fsql, fargs = _filter_sql(filters)
    rows = con.execute(f"select c.*, e.vec, e.dim from chunks c join embeddings e on e.chunk_id=c.id "
                       f"where 1=1{fsql}", fargs).fetchall()
    if not rows:
        return []
    qv = np.asarray(client.embed([cfg["lmstudio"].get("query_prefix", "search_query: ") + query])[0], dtype=np.float32)
    mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
    sims = (mat @ qv) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(qv) + 1e-9)
    order = np.argsort(-sims)[:k]
    out = []
    for i in order:
        d = {key: rows[i][key] for key in rows[i].keys() if key not in ("vec", "dim")}
        out.append(Hit(d, float(sims[i]), ["embedding"]))
    return out


def hybrid_search(con: sqlite3.Connection, client: Any, cfg: dict[str, Any], query: str,
                  filters: dict[str, Any] | None = None, k: int | None = None) -> list[Hit]:
    """Merge keyword and embedding ranks with reciprocal-rank fusion. Falls back to keyword only."""
    k = k or int(cfg.get("retrieval", {}).get("top_k", 8))
    lists = [keyword_search(con, query, filters, k * 2)]
    try:
        lists.append(vector_search(con, client, cfg, query, filters, k * 2))
    except Exception:       # server down or no embeddings: keyword search still works
        pass
    merged: dict[str, Hit] = {}
    for lst in lists:
        for rank, h in enumerate(lst):
            m = merged.setdefault(h.chunk["id"], Hit(h.chunk, 0.0, []))
            m.score += 1.0 / (RRF_K + rank + 1)
            m.sources = sorted(set(m.sources) | set(h.sources))
    return sorted(merged.values(), key=lambda h: -h.score)[:k]


def get_clause(con: sqlite3.Connection, clause_id: str, doc_id: str | None = None) -> list[dict[str, Any]]:
    """All chunks of a clause (case-insensitive), in document order."""
    sql = "select * from chunks where lower(clause_id)=lower(?)"
    args: list[Any] = [clause_id]
    if doc_id:
        sql += " and doc_id=?"
        args.append(doc_id)
    return [dict(r) for r in con.execute(sql + " order by doc_id, id", args)]


def structure_counts(con: sqlite3.Connection, part: str | None = None) -> list[tuple[str, str, int]]:
    """(doc_id, part_or_subsection, chunk count) rows for the `inspect` command."""
    sql, args = "select doc_id, part_or_subsection, count(*) from chunks", []
    if part:
        sql += " where part_or_subsection like ?"
        args.append(f"%{part}%")
    return [tuple(r) for r in con.execute(sql + " group by doc_id, part_or_subsection order by 1,2", args)]
