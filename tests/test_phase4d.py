"""Phase 4d tests: PDF page links (anchor style, URL encoding, citecheck still matches) and VS Code tasks."""
import json
import tempfile
import unittest
from pathlib import Path

from asme_rag import citecheck, pagelinks, rag, storage
from asme_rag.config import load_config
from asme_rag.ingest import ingest_pages, rebuild_links
from asme_rag.lmclient import LMClient
from asme_rag.pdf_parser import Block
from tests.fake_lm_server import start_fake_server

ROOT = Path(__file__).resolve().parent.parent
CIT = "[ASME BPVC VIII-1, 2025, UG-37, PDF p.123]"


class LinkTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "config.toml")
        self.cfg["_root"] = "/proj"

    def test_link_added_with_encoded_path_and_page(self):
        out = pagelinks.add_pdf_links(f"Needed {CIT}.", self.cfg)
        self.assertIn(f"{CIT}(file:///proj/data/pdfs/ASME%20BPVC%20VIII-1_2025.pdf#page=123)", out)

    def test_link_text_unchanged_so_citecheck_still_matches(self):
        out = pagelinks.add_pdf_links(f"Needed {CIT}.", self.cfg)
        m = citecheck.CITATION_RE.search(out)
        self.assertEqual(m.group(0), CIT)
        self.assertEqual(m.group(4), "123")

    def test_anchor_style_configurable(self):
        self.cfg["links"]["anchor_style"] = "#page={n}&zoom=100"
        self.assertTrue(pagelinks.add_pdf_links(CIT, self.cfg).endswith("#page=123&zoom=100)"))

    def test_disabled_and_unknown_book_and_fences(self):
        self.cfg["links"]["pdf_page_links"] = False
        self.assertEqual(pagelinks.add_pdf_links(CIT, self.cfg), CIT)
        self.cfg["links"]["pdf_page_links"] = True
        unknown = "[ASME BPVC VIII-9, 2025, UG-1, PDF p.2]"
        self.assertEqual(pagelinks.add_pdf_links(unknown, self.cfg), unknown)
        fenced = f"```text\n{CIT}\n```\nafter {CIT}"
        out = pagelinks.add_pdf_links(fenced, self.cfg)
        self.assertEqual(out.count("file:///"), 1)
        self.assertTrue(out.startswith(f"```text\n{CIT}\n```"))

    def test_not_linked_twice_and_standard_books(self):
        once = pagelinks.add_pdf_links(CIT, self.cfg)
        self.assertEqual(pagelinks.add_pdf_links(once, self.cfg), once)
        b16 = pagelinks.add_pdf_links("[ASME B16.5, 2025, Table 2-1.1, PDF p.40]", self.cfg)
        self.assertIn("B16_5_2025.pdf#page=40", b16)
        self.assertIsNone(pagelinks.doc_for_citation(self.cfg, "ASME VIII-1", 1999))

    def test_base_dir_is_configurable(self):
        self.cfg["links"]["pdf_base_dir"] = "/elsewhere/my pdfs"
        self.assertIn("file:///elsewhere/my%20pdfs/ASME%20BPVC%20VIII-1_2025.pdf", pagelinks.add_pdf_links(CIT, self.cfg))


class OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv, url = start_fake_server()
        cls.cfg = load_config(ROOT / "config.toml")
        cls.cfg["lmstudio"]["base_url"] = url

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_written_file_has_links_and_check_ran_before_linking(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.cfg
            cfg["paths"].update(outputs=str(Path(tmp) / "out" / "chat"), open_in_vscode=False)
            con = storage.connect(Path(tmp) / "t.sqlite")
            doc = next(d for d in cfg["documents"] if d["id"] == "viii1_2025")
            ingest_pages(con, cfg, doc, [(7, [Block("UG-37 Openings", (0, 0, 1, 1), 7, bold=True),
                                              Block("Reinforcement text here.", (0, 0, 1, 1), 7)])])
            rebuild_links(con, cfg)
            self.srv.script = lambda req: f"Rule {CIT.replace('123', '7')}.\nVerify: UG-37"
            res = rag.answer(con, LMClient(cfg), cfg, "UG-37 reinforcement")
            self.assertEqual([r.status for r in res.extras["citation_results"]], ["matched"])
            md = rag.write_markdown(cfg, res).read_text(encoding="utf-8")
            self.assertIn("PDF p.7](file:///", md)
            self.assertIn("#page=7)", md)
            self.assertTrue(md.rstrip().splitlines()[-1].startswith("Verify:"))
            self.srv.script = None
            con.close()


class VSCodeTests(unittest.TestCase):
    def load(self, name):
        return json.loads((ROOT / ".vscode" / name).read_text(encoding="utf-8"))

    def test_tasks_and_inputs(self):
        t = self.load("tasks.json")
        labels = {x["label"] for x in t["tasks"]}
        for want in ("RAG: Ask", "RAG: Refs", "RAG: Rulemap", "RAG: Materials", "RAG: Calc", "RAG: Report", "RAG: Tests"):
            self.assertIn(want, labels)
        self.assertTrue(all(i["type"] == "promptString" for i in t["inputs"]))
        used = " ".join(x["command"] for x in t["tasks"])
        for i in t["inputs"]:
            self.assertIn("${input:" + i["id"] + "}", used)

    def test_other_vscode_files_parse(self):
        self.assertIn("ms-python.python", self.load("extensions.json")["recommendations"])
        self.assertTrue(self.load("settings.json")["python.testing.unittestEnabled"])
        self.assertTrue(self.load("launch.json")["configurations"])

    def test_open_in_vscode_default_true(self):
        self.assertTrue(load_config(ROOT / "config.toml")["paths"]["open_in_vscode"])


if __name__ == "__main__":
    unittest.main()
