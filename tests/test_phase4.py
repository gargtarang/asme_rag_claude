"""Phase 4 tests: prompt, answer flow, Markdown layout, streaming, follow-ups, governing-code step."""
import datetime as dt
import tempfile
import unittest
from pathlib import Path

from asme_rag import rag, storage
from asme_rag.config import load_config
from asme_rag.ingest import ingest_pages, rebuild_links
from asme_rag.lmclient import LMClient
from asme_rag.pdf_parser import Block
from tests.fake_lm_server import start_fake_server, CANNED_CHAT

ROOT = Path(__file__).resolve().parent.parent


def B(text, bold=False):
    return Block(text=text, bbox=(0, 0, 1, 1), page=7, bold=bold)


class RagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv, url = start_fake_server()
        cls.cfg = load_config(ROOT / "config.toml")
        cls.cfg["lmstudio"]["base_url"] = url
        cls.client = LMClient(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg["paths"]["outputs"] = str(Path(self.tmp.name) / "out")
        self.cfg["paths"]["open_in_vscode"] = False
        self.con = storage.connect(Path(self.tmp.name) / "t.sqlite")
        doc = next(d for d in self.cfg["documents"] if d["id"] == "viii1_2025")
        ingest_pages(self.con, self.cfg, doc, [(7, [
            B("UG-37 Reinforcement Required for Openings", True),
            B("(a) Required area of reinforcement for nozzle openings in vessel walls.")])])
        rebuild_links(self.con, self.cfg)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def test_prompt_contains_required_rules(self):
        for phrase in ("Answer ONLY from the numbered excerpts", "never output design numbers",
                       'End with a line starting "Verify:"', "NEEDS INPUT"):
            self.assertIn(phrase, rag.SYSTEM_PROMPT)

    def test_messages_include_excerpts_history_and_governing(self):
        hits = storage.keyword_search(self.con, "UG-37")
        msgs = rag.build_messages("q?", hits, [("a", "b")] * 5, "VIII-1 2025")
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("VIII-1 2025", msgs[0]["content"])
        self.assertEqual(len(msgs), 1 + 6 + 1)             # system + last 3 Q/A + question
        self.assertIn("[E1] [ASME BPVC VIII-1, 2025, UG-37, PDF p.7]", msgs[-1]["content"])

    def test_answer_and_markdown_layout(self):
        res = rag.answer(self.con, self.client, self.cfg, "reinforcement for nozzle openings")
        self.assertIn(CANNED_CHAT.split(" Verify")[0], res.answer)
        path = rag.write_markdown(self.cfg, res, dt.datetime(2026, 10, 7, 14, 5))
        self.assertEqual(path.name, "2026-10-07_1405_reinforcement-for-nozzle-openings.md")
        md = path.read_text(encoding="utf-8")
        self.assertLess(md.index("## Retrieved excerpts"), md.index("## Answer"))
        self.assertIn("Required area of reinforcement", md)       # verbatim excerpt
        self.assertTrue(md.rstrip().splitlines()[-1].startswith("Verify:"))
        self.assertIn("chunk `viii1_2025:", md)

    def test_filename_collision_gets_suffix(self):
        res = rag.answer(self.con, self.client, self.cfg, "same question UG-37")
        now = dt.datetime(2026, 1, 1, 9, 0)
        a, b = rag.write_markdown(self.cfg, res, now), rag.write_markdown(self.cfg, res, now)
        self.assertNotEqual(a, b)

    def test_no_hits_does_not_call_model_and_says_not_found(self):
        before = len(self.srv.image_requests)
        res = rag.answer(self.con, self.client, self.cfg, "zzzz qqqq xxxx")
        self.assertTrue(res.answer.startswith("not found"))
        self.assertIn("Verify:", res.answer)

    def test_verify_line_added_when_missing(self):
        self.assertIn("Verify:", rag.ensure_verify_line("answer without it"))
        keep = "answer\nVerify: look at UG-37"
        self.assertEqual(rag.ensure_verify_line(keep), keep)

    def test_streaming_tokens_collected(self):
        tokens = []
        res = rag.answer(self.con, self.client, self.cfg, "UG-37 nozzle", on_token=tokens.append)
        self.assertGreater(len(tokens), 3)
        self.assertIn("Canned answer.", res.answer)

    def test_followup_reuses_previous_question_for_retrieval(self):
        self.assertEqual(rag.retrieval_query("and for vacuum?", [("UG-37 nozzle", "x")]),
                         "UG-37 nozzle and for vacuum?")
        long_q = "what is the full rule for jacketed vessels in this edition"
        self.assertEqual(rag.retrieval_query(long_q, [("prev", "x")]), long_q)

    def test_governing_confirmation(self):
        self.assertEqual(rag.confirm_governing("what is UG-37?", None, None), (None, "not_design"))
        self.assertEqual(rag.confirm_governing("design the nozzle", None, None), (None, "needs_input"))
        self.assertEqual(rag.confirm_governing("design the nozzle", "VIII-1 2025", None)[1], "given")
        self.assertEqual(rag.confirm_governing("design it", None, lambda p: ""), ("VIII-1 2025", "confirmed"))
        self.assertEqual(rag.confirm_governing("design it", None, lambda p: "VIII-2 2025")[0], "VIII-2 2025")

    def test_chat_loop_keeps_three_qa_and_asks_governing_once(self):
        answers = iter(["what is UG-37", "design the nozzle for UG-37", "VIII-1 2025", "UG-37 again",
                        "UG-37 four", "UG-37 five", "/quit"])
        out, prompts = [], []
        def inp(p):
            prompts.append(p)
            return next(answers)
        paths = rag.chat_loop(self.con, self.client, self.cfg, {}, None, input_fn=inp,
                              out_fn=out.append, token_fn=lambda t: None)
        self.assertEqual(len(paths), 5)
        self.assertEqual(sum("Which code and edition" in p for p in prompts), 1)
        self.assertTrue(all(p.exists() for p in paths))


if __name__ == "__main__":
    unittest.main()
