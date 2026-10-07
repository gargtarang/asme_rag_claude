"""Command line interface for asme_rag.

Purpose: dispatch `python -m asme_rag <cmd>`. Phase 3 commands: check, ingest, embed, inspect, page, clause, search.
Data flow: cli -> config -> lmclient / storage. Later phases register more commands in COMMANDS.
How to change: add a function cmd_xxx(cfg, args) and an entry in COMMANDS.
"""
from __future__ import annotations

import argparse
import sqlite3
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

from asme_rag.config import load_config, resolve_path
from asme_rag.lmclient import LMClient


def tiny_png() -> bytes:
    """Build a valid 8x8 white PNG in memory (used by `check` to test vision)."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * 8 for _ in range(8))
    ihdr = struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def cmd_check(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Verify config, LM Studio connectivity, loaded models, PDFs and database state."""
    problems = 0
    print(f"config: OK ({len(cfg.get('documents', []))} documents registered)")
    client = LMClient(cfg)
    try:
        models = client.list_models()
        print(f"LM Studio: reachable at {client.base} ({len(models)} models)")
        for name in (client.chat_model, client.embed_model):
            ok = name in models
            print(f"  model {name}: {'loaded' if ok else 'NOT LOADED'}")
            problems += 0 if ok else 1
    except Exception as exc:  # network errors vary; report, do not crash
        print(f"LM Studio: NOT reachable ({exc})")
        return 1
    for doc in cfg.get("documents", []):
        p = resolve_path(cfg, doc["pdf_path"])
        ok = p.exists()
        print(f"  PDF {doc['id']}: {'found' if ok else 'MISSING'} ({p})")
        problems += 0 if ok else 1
    db = resolve_path(cfg, cfg["paths"]["db"])
    if db.exists():
        with sqlite3.connect(db) as con:
            tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
        print(f"database: {db} ({len(tables)} tables)")
    else:
        print(f"database: not created yet ({db}); run `ingest`")
    if cfg.get("vision", {}).get("enabled", False):
        try:
            reply = client.chat_with_image("Reply with the word OK.", tiny_png(),
                                           model=cfg["vision"].get("model"))
            print(f"vision: working (reply: {reply[:40]!r})")
        except Exception as exc:
            print(f"vision: UNAVAILABLE, falling back to pointers only ({exc})")
    else:
        print("vision: disabled (pointers only)")
    return 1 if problems else 0


def _filters(args: argparse.Namespace) -> dict[str, Any]:
    """Collect --doc/--code/--edition options into a filter dict."""
    return {"doc_id": getattr(args, "doc", None), "code": getattr(args, "code", None),
            "edition": getattr(args, "edition", None)}


def _db(cfg: dict[str, Any]):
    from asme_rag import storage
    return storage.connect(resolve_path(cfg, cfg["paths"]["db"]))


def cmd_ingest(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Parse and chunk all registered PDFs, then embed unless --no-embed."""
    from asme_rag import storage
    from asme_rag.ingest import ingest_all
    con = _db(cfg)
    totals = ingest_all(con, cfg, only=args.doc)
    if totals and not args.no_embed:
        print(f"embedded {storage.embed_chunks(con, LMClient(cfg), cfg)} chunks")
    return 0


def cmd_embed(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Create embeddings for chunks that have none (--redo: all)."""
    from asme_rag import storage
    print(f"embedded {storage.embed_chunks(_db(cfg), LMClient(cfg), cfg, redo=args.redo)} chunks")
    return 0


def cmd_inspect(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """List structure (document, part/appendix) with chunk counts."""
    from asme_rag import storage
    for doc_id, part, n in storage.structure_counts(_db(cfg), args.part):
        print(f"{doc_id:16} {part or '(none)':40} {n:6}")
    return 0


def cmd_page(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Show reading order and detected headings of one PDF page (text only, for debugging)."""
    from asme_rag.pdf_parser import parse_page
    docs = [d for d in cfg["documents"] if not args.doc or d["id"] == args.doc]
    if len(docs) != 1:
        print("NEEDS INPUT: give --doc ID (one of: " + ", ".join(d["id"] for d in cfg["documents"]) + ")")
        return 2
    for i, b in enumerate(parse_page(str(resolve_path(cfg, docs[0]["pdf_path"])), args.n, cfg.get("parser")), 1):
        flag = " [bold]" if b.bold else ""
        print(f"--- block {i} size={b.size:.1f}{flag} sup={b.superscripts}\n{b.text}")
    return 0


def cmd_clause(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Print a clause with its metadata."""
    from asme_rag import storage
    rows = storage.get_clause(_db(cfg), args.id, args.doc)
    if not rows:
        print(f"not found: {args.id}")
        return 1
    for r in rows:
        print(f"[{r['code']} {r['division']}, {r['edition']}, {r['clause_id']}, PDF p.{r['pdf_page']}] "
              f"type={r['type']} status={r['status']} unreliable={bool(r['unreliable'])}\n{r['text']}\n")
    return 0


def cmd_search(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Retrieval only: print ranked hits with citations."""
    from asme_rag import storage
    hits = storage.hybrid_search(_db(cfg), LMClient(cfg), cfg, args.query, _filters(args), args.k)
    for i, h in enumerate(hits, 1):
        c = h.chunk
        print(f"{i}. [{c['code']} {c['division']}, {c['edition']}, {c['clause_id']}, PDF p.{c['pdf_page']}] "
              f"({'+'.join(h.sources)}, {h.score:.4f})\n   {c['text'][:200].replace(chr(10), ' ')}")
    if not hits:
        print("no results")
    return 0


def cmd_ask(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Single grounded answer. Terminal shows only a status line and the output path."""
    from asme_rag import rag
    gov, status = rag.confirm_governing(args.question, args.governing,
                                        input if sys.stdin.isatty() else None)
    if status == "needs_input":
        print("NEEDS INPUT: governing code and edition (rerun with --governing \"VIII-1 2025\")")
        return 2
    from asme_rag import expand
    con = _db(cfg)
    ask_fn = input if (sys.stdin.isatty() and not args.no_prompt) else None
    counter = {"n": 0}

    def progress(_tok: str) -> None:
        """Streaming keeps the connection alive; show a counter on stderr (not part of the output file)."""
        counter["n"] += 1
        if counter["n"] % 20 == 0:
            print(f"\rgenerating... {counter['n']} chunks", end="", file=sys.stderr, flush=True)

    print("retrieving and waiting for the model (first request may take minutes while LM Studio loads it)...",
          file=sys.stderr, flush=True)
    res = rag.answer(con, LMClient(cfg), cfg, args.question, None, _filters(args), gov or "",
                     on_token=progress, ask=ask_fn)
    print(file=sys.stderr)
    if res.extras.get("pack") is not None:
        pack = res.extras["pack"]
        if ask_fn is None and pack.further:     # --no-prompt: only print the numbered list
            for f in pack.further:
                print(f"  {f.n}. {f.target} - from {f.parent_cite}")
        else:
            expand.prompt_expand(con, cfg, pack, ask_fn)
    path = rag.write_markdown(cfg, res)
    err = res.extras.get("model_error")
    print(f"{'MODEL ERROR (excerpts saved, no answer)' if err else 'answered'} with {len(res.hits)} excerpts; saved: {path}")
    return 1 if err else 0


def cmd_chat(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Interactive chat with streaming output; keeps the last 3 Q/A."""
    from asme_rag import rag
    rag.chat_loop(_db(cfg), LMClient(cfg), cfg, _filters(args), args.governing)
    return 0


def cmd_eval(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Compare the chat models listed in config [eval] on tests/eval_questions.toml."""
    from asme_rag import evaluation
    path = resolve_path(cfg, args.file)
    if not path.exists():
        print(f"NEEDS INPUT: write your questions in {path} ([[question]] with question, expected_clause, "
              "optional expected_page, expect_not_found)")
        return 2
    md = evaluation.run_eval(_db(cfg), cfg, evaluation.load_questions(path), cfg["eval"]["models"])
    print(f"eval finished; saved: {evaluation.write_eval(cfg, md)}")
    return 0


def _write_text(cfg: dict[str, Any], stem: str, text: str) -> Path:
    """Write a Markdown file under paths.outputs (same naming as answers) and optionally open it."""
    import datetime as dt
    from asme_rag import rag
    folder = resolve_path(cfg, cfg["paths"]["outputs"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{dt.datetime.now():%Y-%m-%d_%H%M}_{rag.slugify(stem)}.md"
    from asme_rag import pagelinks
    path.write_text(pagelinks.add_pdf_links(text, cfg), encoding="utf-8")
    if cfg["paths"].get("open_in_vscode", False):
        import os, subprocess
        try:
            subprocess.Popen(["code", "-r", str(path)], shell=(os.name == "nt"))
        except OSError:
            pass
    return path


def cmd_refs(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Treat a clause as a new anchor and show its level-1 references (one step, never further)."""
    from asme_rag import expand, storage
    con = _db(cfg)
    anchors = storage.get_clause(con, args.id, args.doc)
    if not anchors:
        print(f"not found: {args.id}")
        return 1
    ask_fn = input if (sys.stdin.isatty() and not args.no_prompt) else None
    pack = expand.build_pack(con, cfg, anchors, ask_fn)
    if ask_fn is None:
        for f in pack.further:
            print(f"  {f.n}. {f.target} - from {f.parent_cite}")
    else:
        expand.prompt_expand(con, cfg, pack, ask_fn)
    text = (f"# Cross-references of {args.id}\n\n## References inline\n\n{expand.render_inline(pack, con)}\n\n"
            f"# Reference pack\n\n{expand.render_pack(pack)}")
    print(f"{len(pack.items)} items; saved: {_write_text(cfg, 'refs-' + args.id, text)}")
    return 0


def cmd_rulemap(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Rule map for a design topic, before any calculation."""
    from asme_rag import expand
    ask_fn = input if (sys.stdin.isatty() and not args.no_prompt) else None
    text, pack = expand.rulemap(_db(cfg), LMClient(cfg), cfg, args.topic, _filters(args), ask_fn)
    if ask_fn is not None:
        expand.prompt_expand(_db(cfg), cfg, pack, ask_fn)
    text += "\n# Reference pack\n\n" + expand.render_pack(pack)
    print(f"rule map with {len(pack.items)} items; saved: {_write_text(cfg, 'rulemap-' + args.topic, text)}")
    return 0


COMMANDS = {"check": cmd_check, "refs": cmd_refs, "rulemap": cmd_rulemap, "eval": cmd_eval, "ask": cmd_ask, "chat": cmd_chat, "ingest": cmd_ingest, "embed": cmd_embed, "inspect": cmd_inspect,
            "page": cmd_page, "clause": cmd_clause, "search": cmd_search}


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run a command. Forces UTF-8 on the console (Windows)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="asme_rag")
    parser.add_argument("--config", default="config.toml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    def filt(p):
        p.add_argument("--doc"); p.add_argument("--code"); p.add_argument("--edition", type=int)
    sub.add_parser("check")
    p = sub.add_parser("ingest"); p.add_argument("--no-embed", action="store_true"); p.add_argument("--doc")
    p = sub.add_parser("embed"); p.add_argument("--redo", action="store_true")
    p = sub.add_parser("inspect"); p.add_argument("--part")
    p = sub.add_parser("page"); p.add_argument("n", type=int); p.add_argument("--doc")
    p = sub.add_parser("clause"); p.add_argument("id"); p.add_argument("--doc")
    p = sub.add_parser("refs"); p.add_argument("id"); p.add_argument("--doc"); p.add_argument("--no-prompt", action="store_true")
    p = sub.add_parser("rulemap"); p.add_argument("topic"); p.add_argument("--no-prompt", action="store_true"); filt(p)
    p = sub.add_parser("eval"); p.add_argument("--file", default="tests/eval_questions.toml")
    p = sub.add_parser("ask"); p.add_argument("question"); p.add_argument("--governing"); p.add_argument("--no-prompt", action="store_true"); filt(p)
    p = sub.add_parser("chat"); p.add_argument("--governing"); filt(p)
    p = sub.add_parser("search"); p.add_argument("query"); p.add_argument("-k", type=int, default=8); filt(p)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    return COMMANDS[args.cmd](cfg, args)
