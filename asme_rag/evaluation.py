"""Model comparison (`eval` command).

Purpose: run my question set against each chat model in config [eval] models and write a Markdown
table per model: right clause cited, 'not found' handled, citation check passed, time, tokens/s.
Inputs: tests/eval_questions.toml ([[question]] with question, expected_clause, optional
expected_page, expect_not_found), config, database, LM client. Output: outputs/eval/<date>_eval.md.
Data flow: rag.answer + citecheck -> evaluation. If a model is not loaded in LM Studio the run
pauses and asks me to load it (input_fn). Time includes retrieval. Tokens/s uses the server's usage
count when given, otherwise an estimate (words x 1.3, marked '~').
How to change: columns are built in run_eval(); thresholds none.
"""
from __future__ import annotations

import datetime as dt
import time
import tomllib
from pathlib import Path
from typing import Any, Callable

from asme_rag import citecheck, rag
from asme_rag.config import resolve_path
from asme_rag.lmclient import LMClient


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """Read [[question]] entries; require 'question'; default expect_not_found = false."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    qs = data.get("question", [])
    for q in qs:
        if "question" not in q:
            raise ValueError("every [[question]] needs a 'question' field")
        q.setdefault("expect_not_found", False)
        if not q["expect_not_found"] and "expected_clause" not in q:
            raise ValueError(f"question {q['question']!r}: give expected_clause or expect_not_found = true")
    return qs


def says_not_found(text: str) -> bool:
    """True if the answer declares 'not found' (at the start or as a standalone statement)."""
    t = text.lower()
    return t.lstrip().startswith("not found") or '"not found"' in t or "\nnot found" in t


def right_clause(results: list[citecheck.CiteResult], q: dict[str, Any]) -> bool:
    """A matched citation to the expected clause (and page if given)."""
    want = q.get("expected_clause", "").lower()
    for r in results:
        m = citecheck.CITATION_RE.match(r.text)
        if r.status != "matched" or not m:
            continue
        clause = citecheck.SUBPARA_SUFFIX_RE.sub("", m.group(3).strip()).lower()
        if clause == want and ("expected_page" not in q or int(m.group(4)) == int(q["expected_page"])):
            return True
    return False


def run_eval(con, cfg: dict[str, Any], questions: list[dict[str, Any]], models: list[str],
             input_fn: Callable[[str], str] = input, now: dt.datetime | None = None) -> str:
    """Run every question on every model; return the Markdown report."""
    now = now or dt.datetime.now()
    base = LMClient(cfg)
    out = [f"# Model comparison {now:%Y-%m-%d %H:%M}", "",
           f"Questions: {len(questions)}. Time includes retrieval. '~' = token count estimated.", ""]
    for model in models:
        loaded = model in base.list_models()
        while not loaded:
            reply = input_fn(f"Load model '{model}' in LM Studio, then press Enter (or type 'skip'): ")
            if reply.strip().lower() == "skip":
                break
            loaded = model in base.list_models()
        out_skip = not loaded
        out += [f"## {model}", ""]
        if out_skip:
            out += ["Model not loaded: skipped.", ""]
            continue
        client = LMClient(cfg)
        client.chat_model = model
        out += ["| Question | Right clause cited | Not-found handled | Citation check passed | Time s | Tokens/s |",
                "|---|---|---|---|---|---|"]
        tally = [0, 0, 0]
        for q in questions:
            t0 = time.perf_counter()
            res = rag.answer(con, client, cfg, q["question"], None, {"doc_id": q.get("doc")})
            dt_s = time.perf_counter() - t0
            results = citecheck.check_citations(res.answer, res.hits)
            nf = says_not_found(res.answer)
            if q["expect_not_found"]:
                clause_ok, nf_ok = "n/a", nf
            else:
                clause_ok, nf_ok = right_clause(results, q), not nf
            cc_ok = citecheck.passed(results, res.answer)
            toks = client.last_usage.get("completion_tokens")
            approx = "" if toks else "~"
            toks = toks or int(len(res.answer.split()) * 1.3)
            tps = f"{approx}{toks / dt_s:.1f}" if dt_s > 0 else "n/a"
            tally[0] += clause_ok is True or clause_ok == "n/a"
            tally[1] += bool(nf_ok)
            tally[2] += bool(cc_ok)
            mark = lambda v: v if v == "n/a" else ("yes" if v else "NO")
            out.append(f"| {q['question'][:60]} | {mark(clause_ok)} | {mark(nf_ok)} | {mark(cc_ok)} | {dt_s:.1f} | {tps} |")
        n = len(questions)
        out += ["", f"Totals: right clause {tally[0]}/{n}, not-found handled {tally[1]}/{n}, "
                    f"citation check passed {tally[2]}/{n}.", ""]
    return "\n".join(out)


def write_eval(cfg: dict[str, Any], md: str, now: dt.datetime | None = None) -> Path:
    """Write the report to outputs/eval/YYYY-MM-DD_HHMM_eval.md."""
    now = now or dt.datetime.now()
    folder = resolve_path(cfg, cfg["paths"]["outputs"]).parent / "eval"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{now:%Y-%m-%d_%H%M}_eval.md"
    path.write_text(md, encoding="utf-8")
    return path
