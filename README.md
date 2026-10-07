# Offline ASME BPVC RAG + Vacuum Chamber Design Assistant

Status: **phases 0 to 4d of the build prompt are complete** (scaffold, PDF parser, chunker + cross-reference
index, SQLite/FTS5/embeddings, grounded Q&A, citation check + model comparison, cross-reference expansion,
PDF page links). **Not built yet:** phase 5 (B16.5 / B36.10 / II-D / II-A table profiles and the
`flange_lookup`, `pipe_lookup`, `allowable_lookup`, `materials` commands), 5b (vision), 6 to 8 (the `chamber`
calculator and report), 9 (full docs), 10 (optional chat page). The `chamber` package is an empty placeholder.

**Honesty note.** All 84 tests pass, but they use SYNTHETIC page data and a fake LM Studio server. The code has
never been run against your real ASME PDFs or a real LM Studio model. Expect to tune the clause-ID regexes after
your first `ingest` (see "Tuning"). The tool is a design aid: you are the verifier. Every answer ends with `Verify:`.

## 1. Requirements

- Windows 11, Python 3.12, VS Code (the code is plain Python and also runs on Linux/macOS).
- LM Studio with its local server running (default `http://localhost:1234/v1`), with loaded:
  - a chat model (config default name `gemma-4-26b-a4b`; change it in `config.toml` to the name LM Studio shows),
  - the embedding model `text-embedding-nomic-embed-text-v1.5`.
- Python packages (`requirements.txt`): `PyMuPDF==1.24.14`, `numpy==2.1.3`. Nothing else to install.
- Your licensed PDFs in `data/pdfs/` (names in `data/pdfs/README.txt`).
- No internet is needed at run time. No cloud calls, no telemetry.

## 2. Setup (VS Code, Windows)

1. Unzip the folder anywhere. Open it in VS Code (`File > Open Folder`).
2. Install the recommended extensions when VS Code offers them (Python, Even Better TOML).
3. Create the environment: `Terminal > Run Task > Setup`, or in the integrated terminal:
   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
4. Select the interpreter `.venv` (`Ctrl+Shift+P > Python: Select Interpreter`).
5. In LM Studio load the chat and embedding models and start the local server.
6. Copy your PDFs into `data/pdfs/`.
7. Open `config.toml` and check `[lmstudio]` model names. `open_in_vscode = true` makes each output file open
   automatically (needs the `code` command on PATH: in VS Code run `Shell Command: Install 'code' command in PATH` on
   macOS/Linux; on Windows it is added by the installer).

On Linux/macOS replace `.venv\Scripts\python` with `.venv/bin/python`.

## 3. Run it

All commands: `.venv\Scripts\python -m asme_rag <command>` (or use `Terminal > Run Task`).

| Step | Command | What it does |
|---|---|---|
| 1 | `check` | Verifies config, LM Studio, loaded models, PDFs, database, and tests vision with a tiny image |
| 2 | `ingest` | Parses and chunks all PDFs, builds cross-reference links, then embeds. `--no-embed` skips embedding, `--doc ID` does one book |
| 3 | `inspect [--part UG]` | Structure and chunk counts. Use it to see whether clause detection worked |
| 4 | `page N --doc ID` | Reading order and bold/size flags of PDF page N (debugging the parser) |
| 5 | `clause UG-37 [--doc ID]` | Print a clause with metadata |
| 6 | `search "query" [--doc ID --code ... --edition 2025 -k 8]` | Retrieval only |
| 7 | `ask "question" [--governing "VIII-1 2025"] [--no-prompt]` | One grounded answer, saved to `outputs/chat/` |
| 8 | `chat` | Interactive, streaming, keeps the last 3 Q/A. `/quit`, `/clear` |
| 9 | `refs UG-37 [--doc ID]` | Everything a clause directly refers to (depth 1) |
| 10 | `rulemap "topic"` | Rule map before a design calculation |
| 11 | `eval` | Compares the chat models in `[eval] models` on your question set |
| - | `embed [--redo]` | Create embeddings only |

Example first session:
```
.venv\Scripts\python -m asme_rag check
.venv\Scripts\python -m asme_rag ingest
.venv\Scripts\python -m asme_rag inspect
.venv\Scripts\python -m asme_rag ask "What does UG-37 require for reinforcement of openings?"
```

VS Code tasks (`Terminal > Run Task`): Setup, RAG: Check / Ingest / Embed / Ask / Chat / Refs / Rulemap /
Materials / Calc / Report / Tests. "Ask", "Refs", "Rulemap" and "Materials" open an input box. Materials, Calc and
Report are placeholders until phases 5 to 8. Optional key binding for Ask (`File > Preferences > Keyboard Shortcuts (JSON)`):
```json
{ "key": "ctrl+alt+a", "command": "workbench.action.tasks.runTask", "args": "RAG: Ask" }
```

## 4. What you get

- **Answers** are written to `outputs/chat/YYYY-MM-DD_HHMM_<topic>.md` in this order: question, retrieved excerpts
  (verbatim), the model's answer, citation check, inline references, reference pack, `Verify:` line. The terminal prints
  only a status line and the path.
- **Citation check** runs on every answer. Any citation that does not match a retrieved excerpt (code, edition, clause,
  PDF page) is listed as `UNVERIFIED CITATION`. Nothing is silently corrected.
- **Not found:** if nothing is retrieved, the model is not called and the answer is "not found".
- **Cross-references** are collected by code, never by the model, to a maximum depth of 1 (constant `MAX_DEPTH_HARD` in
  `asme_rag/expand.py`). Deeper references are listed as numbered "Further references (not expanded)"; the tool then asks
  `Expand which?`. A book that is not ingested appears as `NOT INGESTED: <book>, <clause>`.
- **PDF links:** each citation links to its PDF page (`#page=N`). Test your PDF viewer; change `anchor_style` in `[links]` if needed.
- **Design questions** (words like "design", "calculate") first need the governing code and edition (default VIII-1 2025).

## 5. Tests

```
.venv\Scripts\python -m unittest discover -s tests -v
```
84 tests; they need no PDFs, no PyMuPDF and no LM Studio (a fake server is included in `tests/fake_lm_server.py`).

## 6. Tuning after your first ingest

- Clause detection regexes: `asme_rag/chunker.py` (`PROFILES`, marked `# EDIT HERE:`) or override per profile in
  `config.toml` with a `[chunker.code_book]` block containing `clause_id_regexes = [...]`.
- Reference patterns: `asme_rag/xref.py`. Header/footer/licence stripping: `[parser]` in `config.toml`.
- Use `inspect` and `page N --doc ID` to see what was detected, then adjust and re-run `ingest --doc ID`.
- Chat model: `[lmstudio] chat_model`. Retrieval size: `[retrieval] top_k`. Cross-reference limits: `[xref]`.

## 7. Known limitations

- Untested on real PDFs and real LM Studio (see the honesty note). `ingest` and `page` need PyMuPDF and were never run here.
- Printed page numbers are not stored; links and citations use the PDF page index only.
- Appendix and "Part" references are always listed as possible (fuzzy) and unresolved: they point at headings, not clause IDs.
- Equations and tables extracted from text may be garbled; such chunks are flagged `unreliable` and labelled in outputs.
- The "Inputs required" section of `rulemap` is model-generated and marked UNVERIFIED.
- `eval` needs your own `tests/eval_questions.toml` (see `tests/eval_questions.example.toml`).

## 8. Troubleshooting

- `check` says LM Studio not reachable: start the local server in LM Studio and check `base_url` in `config.toml`.
- `model ... NOT LOADED`: the name in `config.toml` must equal the id shown by LM Studio.
- `PDF missing`: file name or folder differs from `config.toml` `pdf_path`.
- `ModuleNotFoundError: fitz`: run `pip install -r requirements.txt` inside `.venv`.
- No hits / poor hits: run `inspect`; if parts or clauses are missing, tune the regexes (section 6).
- Output file does not open: put `code` on PATH or set `open_in_vscode = false`.
- Console shows odd characters: output is forced to UTF-8; use Windows Terminal or the VS Code terminal.

## 9. Layout

```
config.toml          every tunable (LM Studio, paths, regex overrides, xref, links, eval models)
requirements.txt     Python packages
asme_rag/            pdf_parser, chunker, xref, storage, ingest, rag, citecheck, evaluation, expand, pagelinks, lmclient, cli
chamber/             placeholder (phases 6 to 8)
tests/               unit tests + fake LM Studio server
data/pdfs/           put your PDFs here;  data/verified/  (future verified CSVs);  data/db/  SQLite database
outputs/chat/        answers;  outputs/eval/  model comparisons
.vscode/             settings, tasks, launch, recommended extensions
```
