"""Ingest pipeline: PDF pages -> chunks -> links -> database.

Purpose: glue between pdf_parser, chunker, xref and storage.
Inputs: config dict, a database connection, per-document pages (list of (page_no, blocks)).
Outputs: counts. Data flow: pdf_parser -> chunker -> xref -> storage.
How to change: per-profile chunk overrides come from config.toml [chunker.<profile>].
"""
from __future__ import annotations

import sqlite3
from typing import Any

from asme_rag import storage
from asme_rag.chunker import chunk_document
from asme_rag.config import resolve_path
from asme_rag.pdf_parser import Block, parse_pdf
from asme_rag.xref import build_index


def ingest_pages(con: sqlite3.Connection, cfg: dict[str, Any], doc: dict[str, Any],
                 pages: list[tuple[int, list[Block]]]) -> tuple[int, int]:
    """Replace one document's chunks in the database. Returns (chunks, links within that document).

    Links are rebuilt over ALL stored chunks afterwards by rebuild_links(), so that references to
    other ingested books resolve correctly regardless of ingest order.
    """
    storage.delete_doc(con, doc["id"])
    override = cfg.get("chunker", {}).get(doc["parser_profile"])
    chunks = chunk_document(pages, doc, int(cfg.get("retrieval", {}).get("chunk_max_chars", 3000)), override)
    return storage.insert_chunks(con, chunks), 0


def rebuild_links(con: sqlite3.Connection, cfg: dict[str, Any]) -> int:
    """Recompute the whole cross-reference index from the stored chunks."""
    from asme_rag.chunker import Chunk
    con.execute("delete from links")
    chunks = []
    for r in con.execute("select * from chunks order by id"):
        import json
        chunks.append(Chunk(id=r["id"], doc_id=r["doc_id"], code=r["code"], division=r["division"],
                            edition=r["edition"], part_or_subsection=r["part_or_subsection"],
                            clause_id=r["clause_id"], pdf_page=r["pdf_page"], type=r["type"], text=r["text"],
                            superscripts=json.loads(r["superscripts"]), linked_to=r["linked_to"]))
    return storage.insert_links(con, build_index(chunks, cfg["documents"]))


def ingest_all(con: sqlite3.Connection, cfg: dict[str, Any], only: str | None = None,
               log=print) -> dict[str, int]:
    """Parse and store every registered PDF that exists. Missing PDFs are reported, not fatal."""
    totals: dict[str, int] = {}
    for doc in cfg["documents"]:
        if only and doc["id"] != only:
            continue
        pdf = resolve_path(cfg, doc["pdf_path"])
        if not pdf.exists():
            log(f"{doc['id']}: PDF missing ({pdf}) - skipped. NOT INGESTED: ask the user for the file.")
            continue
        pages = [(i, blocks) for i, blocks in enumerate(parse_pdf(str(pdf), cfg.get("parser")), start=1)]
        n, _ = ingest_pages(con, cfg, doc, pages)
        totals[doc["id"]] = n
        log(f"{doc['id']}: {len(pages)} pages, {n} chunks")
    log(f"links: {rebuild_links(con, cfg)}")
    return totals
