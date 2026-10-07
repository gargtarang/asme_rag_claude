"""Regression tests: model timeout is reported (not a crash) and the prompt size is bounded."""
import tempfile
import time
import unittest
from pathlib import Path

from asme_rag import rag, storage
from asme_rag.config import load_config
from asme_rag.ingest import ingest_pages, rebuild_links
from asme_rag.lmclient import LMClient
from asme_rag.pdf_parser import Block
from tests.fake_lm_server import start_fake_server

ROOT = Path(__file__).resolve().parent.parent


class TimeoutAndBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv, url = start_fake_server()
        cls.cfg = load_config(ROOT / "config.toml")
        cls.cfg["lmstudio"]["base_url"] = url

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg["paths"].update(outputs=str(Path(self.tmp.name) / "out" / "chat"), open_in_vscode=False)
        self.con = storage.connect(Path(self.tmp.name) / "t.sqlite")
        doc = next(d for d in self.cfg["documents"] if d["id"] == "viii1_2025")
        blocks = []
        for i in range(10):
            blocks += [Block(f"UG-{i + 10} Clause {i}", (0, 0, 1, 1), 1, bold=True),
                       Block("nozzle reinforcement openings " + "filler " * 400, (0, 0, 1, 1), 1)]
        ingest_pages(self.con, self.cfg, doc, [(1, blocks)])
        rebuild_links(self.con, self.cfg)

    def tearDown(self):
        self.srv.script = None
        self.con.close()
        self.tmp.cleanup()

    def test_timeout_gives_model_error_and_still_saves_excerpts(self):
        self.srv.script = lambda req: time.sleep(1.5) or "late"
        client = LMClient(self.cfg)
        client.chat_timeout = 0.3
        res = rag.answer(self.con, client, self.cfg, "nozzle reinforcement openings")
        self.assertTrue(res.answer.startswith("MODEL ERROR"))
        self.assertIn("chat_timeout_s", res.answer)
        self.assertIn("model_error", res.extras)
        md = rag.write_markdown(self.cfg, res).read_text(encoding="utf-8")
        self.assertIn("## Retrieved excerpts", md)
        self.assertTrue(md.rstrip().splitlines()[-1].startswith("Verify:"))

    def test_timeout_also_when_streaming(self):
        self.srv.script = lambda req: time.sleep(1.5) or "late"
        client = LMClient(self.cfg)
        client.chat_timeout = 0.3
        res = rag.answer(self.con, client, self.cfg, "nozzle reinforcement openings", on_token=lambda t: None)
        self.assertTrue(res.answer.startswith("MODEL ERROR"))

    def test_prompt_is_bounded_and_numbering_matches_file(self):
        seen = {}
        self.srv.script = lambda req: seen.setdefault("p", req["messages"][-1]["content"]) and "ok\nVerify: x"
        self.cfg["retrieval"].update(prompt_excerpt_chars=500, prompt_total_chars=1500, top_k=8)
        res = rag.answer(self.con, LMClient(self.cfg), self.cfg, "nozzle reinforcement openings", use_xref=False)
        self.assertLess(len(seen["p"]), 2600)
        self.assertIn("truncated in the prompt", seen["p"])
        self.assertIn("prompt size limit", res.extras["prompt_note"])
        self.assertGreater(len(res.hits), seen["p"].count("[E"))       # file lists all, prompt only some
        md = rag.render_markdown(res)
        self.assertIn("### E1:", md)
        self.assertIn("filler " * 300, md)                             # full verbatim text kept in the file

    def test_default_timeouts(self):
        c = LMClient(self.cfg)
        self.assertEqual(c.chat_timeout, 900.0)
        self.assertEqual(c.timeout, 120.0)


if __name__ == "__main__":
    unittest.main()
