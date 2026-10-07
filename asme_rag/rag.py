"""Grounded question answering: retrieval -> prompt -> local model -> Markdown file.

Purpose: answer ONLY from retrieved excerpts, with citations, and save a Markdown record in which the
verbatim excerpts come BEFORE the model's answer.
Inputs: database connection, LM client, config, question, optional history (last 3 Q/A), filters.
Outputs: Result object and a file outputs/chat/YYYY-MM-DD_HHMM_<topic>.md.
Data flow: storage.hybrid_search -> rag.answer -> rag.write_markdown. Phase 4b adds citecheck,
phase 4c adds cross-reference packs, phase 4d adds PDF page links (hooks are marked below).
How to change: edit SYSTEM_PROMPT (marked '# EDIT HERE:'), DESIGN_WORDS, or [chat] keys in config.toml.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from asme_rag import citecheck, pagelinks, storage
from asme_rag.config import resolve_path

# EDIT HERE: runtime system prompt (text fixed by the project specification).
SYSTEM_PROMPT = """You are an offline assistant for the ASME Boiler and Pressure Vessel Code and related ASME standards.
Answer ONLY from the numbered excerpts provided.
Cite every statement as [code, edition, clause, PDF p.N].
If the excerpts do not contain the answer, reply "not found" and name the book, part or section where to look.
Never invent equations, constants, table values or clause numbers.
Flag any table or equation that looks garbled in the excerpt.
Do not perform final design calculations and never output design numbers; show the equations, the methodology and every input needed (for example thicknesses, membrane, primary and secondary stresses) so the calculator can compute them.
If a paragraph refers to another book or part, quote the reference and say whether that book is in the excerpts.
If information is missing or ambiguous, list it under NEEDS INPUT instead of assuming.
End with a line starting "Verify:" telling the user what to check in the real code."""

# EDIT HERE: words that mark a design task (the governing code and edition are then confirmed first).
DESIGN_WORDS = re.compile(r"\b(design|calculate|calculation|size|sizing|check my|my chamber|my nozzle|"
                          r"required thickness|will it pass)\b", re.I)
DEFAULT_GOVERNING = "VIII-1 2025"
DEFAULT_VERIFY = "Verify: check every cited clause, table and value against the real code PDF before use."


@dataclass
class Result:
    """Everything needed to write the Markdown record."""
    question: str
    retrieval_query: str
    hits: list[storage.Hit]
    answer: str
    model: str
    filters: dict[str, Any]
    governing: str = ""
    temperature: float = 0.1
    extras: dict[str, Any] = field(default_factory=dict)   # hooks for citecheck / xref packs (later phases)


def is_design_question(q: str) -> bool:
    """True if the question looks like a design task (section 5.3a: confirm governing code first)."""
    return bool(DESIGN_WORDS.search(q))


def confirm_governing(question: str, given: str | None, ask: Callable[[str], str] | None) -> tuple[str | None, str]:
    """Return (governing, status). status is 'not_design', 'given', 'confirmed' or 'needs_input'.

    Design questions need a confirmed code and edition. `given` comes from --governing; otherwise
    `ask` (interactive prompt) is used; with neither, the caller must print NEEDS INPUT.
    """
    if not is_design_question(question):
        return given, "not_design"
    if given:
        return given, "given"
    if ask is None:
        return None, "needs_input"
    reply = ask(f"Which code and edition governs? [default {DEFAULT_GOVERNING}] ").strip()
    return (reply or DEFAULT_GOVERNING), "confirmed"


def retrieval_query(question: str, history: list[tuple[str, str]], max_words: int = 6) -> str:
    """Short follow-ups reuse the previous question so retrieval has context."""
    if history and len(question.split()) <= max_words:
        return f"{history[-1][0]} {question}"
    return question


def cite(c: dict[str, Any]) -> str:
    """Plain citation text for a chunk row: [code division, edition, clause, PDF p.N]."""
    code = f"{c['code']} {c['division']}".strip()
    return f"[{code}, {c['edition']}, {c['clause_id'] or 'no clause id'}, PDF p.{c['pdf_page']}]"


def fit_to_budget(hits: list[storage.Hit], each_chars: int, total_chars: int) -> list[storage.Hit]:
    """Longest prefix of `hits` whose (truncated) texts fit the total prompt budget. Order is kept, so the
    excerpt numbers E1..En in the prompt equal those in the Markdown file. At least one hit is kept."""
    kept, used = [], 0
    for h in hits:
        n = min(len(h.chunk["text"]), each_chars) + 120          # +120 for the excerpt header
        if kept and used + n > total_chars:
            break
        kept.append(h)
        used += n
    return kept


def format_excerpts(hits: list[storage.Hit], each_chars: int = 1500) -> str:
    """Numbered excerpts for the model prompt. Model-generated or unreliable chunks are labelled.
    Texts longer than each_chars are cut in the PROMPT only (the Markdown file keeps the full text)."""
    parts = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        flags = []
        if c.get("model_generated"):
            flags.append("model-generated, unverified")
        if c.get("unreliable"):
            flags.append("text may be garbled")
        tag = f" ({'; '.join(flags)})" if flags else ""
        body = c["text"] if len(c["text"]) <= each_chars else c["text"][:each_chars] + " [... truncated in the prompt]"
        parts.append(f"[E{i}] {cite(c)} type={c['type']}{tag}\n{body}")
    return "\n\n".join(parts)


def build_messages(question: str, hits: list[storage.Hit], history: list[tuple[str, str]],
                   governing: str = "", each_chars: int = 1500) -> list[dict[str, str]]:
    """System prompt, last 3 Q/A, then the question with its excerpts."""
    system = SYSTEM_PROMPT + (f"\nGoverning code and edition (confirmed by the user): {governing}" if governing else "")
    msgs = [{"role": "system", "content": system}]
    for q, a in history[-3:]:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": f"Excerpts:\n\n{format_excerpts(hits, each_chars)}\n\nQuestion: {question}"})
    return msgs


def ensure_verify_line(text: str) -> str:
    """The answer must end with a 'Verify:' line; add a default one if the model forgot."""
    return text if re.search(r"^\s*Verify:", text, re.M) else text.rstrip() + "\n\n" + DEFAULT_VERIFY


def answer(con, client, cfg: dict[str, Any], question: str, history: list[tuple[str, str]] | None = None,
           filters: dict[str, Any] | None = None, governing: str = "",
           on_token: Callable[[str], None] | None = None, use_xref: bool = True,
           ask: Callable[[str], str] | None = None) -> Result:
    """Retrieve, ask the model (streaming if on_token is given), and return a Result.

    With no excerpts the model is NOT called: the answer is 'not found' (no chance to invent content).
    """
    history = history or []
    q = retrieval_query(question, history, int(cfg.get("chat", {}).get("followup_max_words", 6)))
    hits = storage.hybrid_search(con, client, cfg, q, filters)
    pack = inline_md = None
    hits_for_check, omitted, sent, model_error = hits, 0, hits, ""
    if use_xref and hits:
        # Phase 4c: the CODE collects what the anchors refer to (depth 1); the model only reads it.
        from asme_rag import expand
        pack = expand.build_pack(con, cfg, [h.chunk for h in hits], ask)
        limit = int(expand.xref_settings(cfg)["prompt_max_items"])
        hits = hits + [storage.Hit(i.chunk, 0.0, ["xref level 1"]) for i in pack.group("level1")[:limit]]
        inline_md = expand.render_inline(pack, con)
    if not hits:
        text = ('not found - no excerpts were retrieved. Look in the book, part or section that governs this '
                'topic and check that the document is ingested (run `check`).\n\n' + DEFAULT_VERIFY)
    else:
        rc = cfg.get("retrieval", {})
        each = int(rc.get("prompt_excerpt_chars", 1500))       # EDIT HERE (config [retrieval]): prompt size limits
        sent = fit_to_budget(hits, each, int(rc.get("prompt_total_chars", 14000)))
        msgs = build_messages(question, sent, history, governing, each)
        model_error = ""
        try:
            if on_token:
                text = "".join(_tee(client.chat_stream(msgs), on_token))
            else:
                text = client.chat(msgs)
        except OSError as exc:             # timeouts and connection errors (TimeoutError, URLError are OSErrors)
            model_error = f"{type(exc).__name__}: {exc}"
            text = (f"MODEL ERROR: {model_error}\n\nNo answer was generated. The retrieved excerpts above are saved. "
                    f"Check that LM Studio has the chat model loaded (the first request can take minutes while it loads), "
                    f"raise [lmstudio] chat_timeout_s in config.toml (now {client.chat_timeout:.0f} s), or lower "
                    f"[retrieval] prompt_total_chars / top_k to shorten the prompt.")
        text = ensure_verify_line(text)
        hits_for_check = sent
        omitted = len(hits) - len(sent)
    res = Result(question, q, hits, text, client.chat_model, dict(filters or {}), governing, client.temperature)
    results = citecheck.check_citations(text, hits_for_check if hits else hits)
    if hits and omitted:
        res.extras["prompt_note"] = f"Only E1-E{len(sent)} of {len(hits)} excerpts were sent to the model (prompt size limit)."
    if hits and model_error:
        res.extras["model_error"] = model_error
    res.extras["citation_results"] = results
    res.extras["citation_check_md"] = citecheck.render_check(results, text)
    if pack is not None:
        res.extras["pack"], res.extras["inline_md"] = pack, inline_md
    return res


def _tee(it: Iterator[str], cb: Callable[[str], None]) -> Iterator[str]:
    """Pass tokens through while calling cb on each one."""
    for tok in it:
        cb(tok)
        yield tok


def slugify(text: str, max_len: int = 40) -> str:
    """File-name-safe topic from the question."""
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return (s[:max_len].rstrip("-")) or "question"


def render_markdown(r: Result, now: dt.datetime | None = None) -> str:
    """Markdown record. Order: question, retrieved excerpts (verbatim), answer, checks, Verify line."""
    now = now or dt.datetime.now()
    out = [f"# {r.question}", "",
           f"- Date: {now:%Y-%m-%d %H:%M}", f"- Model: {r.model} (temperature {r.temperature})",
           f"- Filters: {({k: v for k, v in r.filters.items() if v} or 'none')}",
           f"- Governing code and edition: {r.governing or 'not set'}",
           f"- Retrieval query: {r.retrieval_query}"]
    if r.extras.get("prompt_note"):
        out.append(f"- Note: {r.extras['prompt_note']}")
    out += ["", "## Retrieved excerpts (verbatim)", ""]
    if not r.hits:
        out += ["No excerpts retrieved.", ""]
    for i, h in enumerate(r.hits, 1):
        c = h.chunk
        flags = " (model-generated figure description, unverified)" if c.get("model_generated") else ""
        flags += " (text may be garbled)" if c.get("unreliable") else ""
        out += [f"### E{i}: {c['clause_id'] or 'no clause id'}, PDF p.{c['pdf_page']}{flags}",
                f"{cite(c)} chunk `{c['id']}` via {'+'.join(h.sources)}", "", "```text", c["text"], "```", ""]
    body = r.answer.rstrip().splitlines()
    verify = [ln for ln in body if ln.strip().startswith("Verify:")]
    main = [ln for ln in body if not ln.strip().startswith("Verify:")]
    out += ["## Answer", "", "\n".join(main).strip(), "", "## Citation check", "",
            r.extras.get("citation_check_md", "Not run in this build (added in phase 4b)."), ""]
    pack = r.extras.get("pack")
    if pack is not None:
        from asme_rag import expand
        out += ["## References inline (collected by code, depth 1)", "", r.extras.get("inline_md", ""), "",
                "# Reference pack", "", expand.render_pack(pack), ""]
        ni = expand.needs_input(pack)
        if ni:
            out += ["## NEEDS INPUT", ""] + [f"- {x}" for x in ni] + [""]
    out += [verify[-1].strip() if verify else DEFAULT_VERIFY, ""]
    return "\n".join(out)


def write_markdown(cfg: dict[str, Any], r: Result, now: dt.datetime | None = None) -> Path:
    """Write the record under paths.outputs and optionally open it in VS Code (`code -r`)."""
    now = now or dt.datetime.now()
    folder = resolve_path(cfg, cfg["paths"]["outputs"])
    folder.mkdir(parents=True, exist_ok=True)
    base = f"{now:%Y-%m-%d_%H%M}_{slugify(r.question)}"
    path, n = folder / f"{base}.md", 2
    while path.exists():
        path = folder / f"{base}_{n}.md"
        n += 1
    path.write_text(pagelinks.add_pdf_links(render_markdown(r, now), cfg), encoding="utf-8")   # links AFTER citecheck
    pack = r.extras.get("pack")
    if pack is not None and cfg.get("xref", {}).get("write_pack_file", True):
        from asme_rag import expand
        (path.parent / f"{path.stem}_refs.md").write_text(
            pagelinks.add_pdf_links(f"# Reference pack: {r.question}\n\n" + expand.render_pack(pack), cfg),
            encoding="utf-8")
    if cfg["paths"].get("open_in_vscode", False):
        try:
            subprocess.Popen(["code", "-r", str(path)], shell=(os.name == "nt"))   # "code" is code.cmd on Windows
        except OSError:
            pass            # VS Code not on PATH: the file is still written
    return path


def chat_loop(con, client, cfg: dict[str, Any], filters: dict[str, Any], governing: str | None,
              input_fn: Callable[[str], str] = input, out_fn: Callable[[str], None] = print,
              token_fn: Callable[[str], None] = lambda t: print(t, end="", flush=True)) -> list[Path]:
    """Interactive chat. Keeps the last 3 Q/A, streams tokens, writes one Markdown file per question.

    Commands: /quit, /clear (forget history). Design questions confirm the governing code once.
    """
    history: list[tuple[str, str]] = []
    paths: list[Path] = []
    gov = governing
    while True:
        try:
            q = input_fn("> ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/clear":
            history.clear()
            continue
        gov, status = confirm_governing(q, gov, lambda p: input_fn(p))
        if status == "needs_input":
            out_fn("NEEDS INPUT: governing code and edition")
            continue
        res = answer(con, client, cfg, q, history, filters, gov or "", on_token=token_fn, ask=input_fn)
        out_fn("")
        if res.extras.get("pack") is not None:
            from asme_rag import expand
            expand.prompt_expand(con, cfg, res.extras["pack"], input_fn)
        history.append((q, res.answer))
        history[:] = history[-3:]
        paths.append(write_markdown(cfg, res))
        out_fn(f"saved: {paths[-1]}")
    return paths
