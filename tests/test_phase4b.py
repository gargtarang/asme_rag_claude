"""Phase 4b tests: citation check (matched / unmatched / malformed) and the eval command."""
import tempfile
import unittest
from pathlib import Path

from asme_rag import citecheck, evaluation, rag, storage
from asme_rag.config import load_config
from asme_rag.ingest import ingest_pages, rebuild_links
from asme_rag.lmclient import LMClient
from asme_rag.pdf_parser import Block
from tests import fake_lm_server
from tests.fake_lm_server import start_fake_server

ROOT = Path(__file__).resolve().parent.parent
GOOD = "[ASME BPVC VIII-1, 2025, UG-37, PDF p.7]"


def B(text, bold=False, page=7):
    return Block(text=text, bbox=(0, 0, 1, 1), page=page, bold=bold)


class Base(unittest.TestCase):
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
        self.cfg["paths"]["outputs"] = str(Path(self.tmp.name) / "out" / "chat")
        self.cfg["paths"]["open_in_vscode"] = False
        self.con = storage.connect(Path(self.tmp.name) / "t.sqlite")
        doc = next(d for d in self.cfg["documents"] if d["id"] == "viii1_2025")
        ingest_pages(self.con, self.cfg, doc, [(7, [
            B("UG-37 Reinforcement Required for Openings", True),
            B("(a) Required area of reinforcement for nozzle openings.")])])
        rebuild_links(self.con, self.cfg)
        self.hits = storage.keyword_search(self.con, "UG-37")

    def tearDown(self):
        self.srv.script = None
        self.con.close()
        self.tmp.cleanup()


class CiteTests(Base):
    def check(self, text):
        return citecheck.check_citations(text, self.hits)

    def test_matched(self):
        r = self.check(f"Needed {GOOD}.")
        self.assertEqual([x.status for x in r], ["matched"])
        self.assertEqual(r[0].excerpt, 1)

    def test_subparagraph_and_short_code_match(self):
        self.assertEqual(self.check("x [VIII-1, 2025, UG-37(a), PDF p.7]")[0].status, "matched")

    def test_unmatched_reasons(self):
        wrong_page = self.check("[ASME BPVC VIII-1, 2025, UG-37, PDF p.99]")[0]
        self.assertEqual(wrong_page.status, "unverified")
        self.assertIn("different PDF page: 7", wrong_page.reason)
        self.assertIn("not in retrieved", self.check("[ASME BPVC VIII-1, 2025, UG-99, PDF p.7]")[0].reason)
        self.assertIn("edition", self.check("[ASME BPVC VIII-1, 2019, UG-37, PDF p.7]")[0].reason)
        self.assertIn("code/book", self.check("[ASME BPVC VIII-2, 2025, UG-37, PDF p.7]")[0].reason)

    def test_malformed_citation_reported(self):
        r = self.check("see [UG-37, PDF p.7] and " + GOOD)
        self.assertEqual(sorted(x.status for x in r), ["matched", "unverified"])
        self.assertTrue(any("malformed" in x.reason for x in r))

    def test_render_and_passed(self):
        bad = self.check("[ASME BPVC VIII-1, 2025, UG-99, PDF p.7]")
        md = citecheck.render_check(bad, "x")
        self.assertIn("UNVERIFIED CITATION", md)
        self.assertFalse(citecheck.passed(bad, "x"))
        self.assertTrue(citecheck.passed(self.check(GOOD), GOOD))
        self.assertTrue(citecheck.passed([], "not found - look in Part UG"))
        self.assertFalse(citecheck.passed([], "answer with no citations"))
        self.assertIn("no parsable citations", citecheck.render_check([], "plain answer"))

    def test_model_generated_chunk_warns(self):
        self.hits[0].chunk["model_generated"] = 1
        r = self.check(GOOD)
        self.assertIn("model-generated", r[0].warning)

    def test_answer_pipeline_records_check_in_markdown(self):
        self.srv.script = lambda req: f"Reinforcement is required {GOOD}. Also [ASME BPVC VIII-1, 2025, UG-77, PDF p.3].\nVerify: UG-37"
        res = rag.answer(self.con, self.client, self.cfg, "UG-37 reinforcement")
        md = rag.render_markdown(res)
        self.assertIn("1 of 2 citations matched", md)
        self.assertIn("UNVERIFIED CITATION: [ASME BPVC VIII-1, 2025, UG-77, PDF p.3]", md)
        self.assertNotIn("Not run in this build", md)


class EvalTests(Base):
    def write_questions(self, text):
        p = Path(self.tmp.name) / "q.toml"
        p.write_text(text, encoding="utf-8")
        return p

    def test_load_questions_validation(self):
        p = self.write_questions('[[question]]\nquestion = "x"\n')
        with self.assertRaises(ValueError):
            evaluation.load_questions(p)
        p = self.write_questions('[[question]]\nquestion = "x"\nexpect_not_found = true\n')
        self.assertTrue(evaluation.load_questions(p)[0]["expect_not_found"])

    def test_eval_report(self):
        self.srv.script = lambda req: (f"Required {GOOD}.\nVerify: UG-37" if "UG-37" in req["messages"][-1]["content"]
                                       else 'not found - look elsewhere.\nVerify: x')
        qs = [{"question": "UG-37 reinforcement", "expected_clause": "UG-37", "expected_page": 7, "expect_not_found": False},
              {"question": "UG-37 wrong clause", "expected_clause": "UG-99", "expect_not_found": False},
              {"question": "zzzz qqqq", "expect_not_found": True}]
        md = evaluation.run_eval(self.con, self.cfg, qs, ["gemma-4-26b-a4b"], input_fn=lambda p: "skip")
        self.assertIn("## gemma-4-26b-a4b", md)
        rows = [ln for ln in md.splitlines() if ln.startswith("| UG-37 reinf")]
        self.assertIn("| yes | yes | yes |", rows[0])
        wrong = [ln for ln in md.splitlines() if ln.startswith("| UG-37 wrong")][0]
        self.assertIn("| NO |", wrong)
        nf = [ln for ln in md.splitlines() if ln.startswith("| zzzz")][0]
        self.assertIn("| n/a | yes | yes |", nf)
        self.assertIn("right clause 2/3", md)
        path = evaluation.write_eval(self.cfg, md)
        self.assertEqual(path.parent.name, "eval")

    def test_eval_pauses_for_unloaded_model_and_can_skip(self):
        asked = []
        md = evaluation.run_eval(self.con, self.cfg, [], ["not-loaded-model"],
                                 input_fn=lambda p: asked.append(p) or "skip")
        self.assertEqual(len(asked), 1)
        self.assertIn("Model not loaded: skipped.", md)

    def test_eval_waits_until_model_is_loaded(self):
        calls = []
        def load(prompt):
            calls.append(prompt)
            fake_lm_server.MODELS.append("later-model")
            return ""
        try:
            md = evaluation.run_eval(self.con, self.cfg, [], ["later-model"], input_fn=load)
        finally:
            fake_lm_server.MODELS.remove("later-model")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("skipped", md)


if __name__ == "__main__":
    unittest.main()
