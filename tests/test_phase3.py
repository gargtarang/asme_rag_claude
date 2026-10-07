"""Phase 3 tests: storage, FTS query building, filters, embeddings, hybrid retrieval (SYNTHETIC data)."""
import unittest
from pathlib import Path
import tempfile

from asme_rag import storage
from asme_rag.config import load_config
from asme_rag.ingest import ingest_pages, rebuild_links
from asme_rag.lmclient import LMClient
from asme_rag.pdf_parser import Block
from tests.fake_lm_server import start_fake_server

ROOT = Path(__file__).resolve().parent.parent


def B(text, bold=False, page=1):
    return Block(text=text, bbox=(0, 0, 1, 1), page=page, bold=bold)


class StorageTests(unittest.TestCase):
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
        self.con = storage.connect(Path(self.tmp.name) / "t.sqlite")
        docs = {d["id"]: d for d in self.cfg["documents"]}
        ingest_pages(self.con, self.cfg, docs["viii1_2025"], [(1, [
            B("Part UG", True), B("UG-37 Reinforcement Required for Openings", True),
            B("(a) Required area of reinforcement for nozzle openings. See Division 2, Part 4.12."),
            B("UG-28 Thickness of Shells Under External Pressure", True),
            B("Vacuum external pressure shells rules apply.")])])
        ingest_pages(self.con, self.cfg, docs["viii2_2025"], [(5, [
            B("4.5.1 Openings in Shells", True), B("Nozzle opening reinforcement for pressure vessels.")])])
        rebuild_links(self.con, self.cfg)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def test_build_fts_query(self):
        q = storage.build_fts_query("What is the reinforcement required by UG-37 for the nozzle?")
        self.assertIn('"UG-37"', q)
        self.assertIn('"reinforcement"', q)
        self.assertNotIn('"the"', q)
        self.assertEqual(storage.build_fts_query("the of a"), "")
        self.assertNotIn('""', storage.build_fts_query('say "hi" -- OR AND'))

    def test_keyword_search_by_clause_id_and_text(self):
        hits = storage.keyword_search(self.con, "UG-37")
        self.assertEqual(hits[0].chunk["clause_id"], "UG-37")
        hits = storage.keyword_search(self.con, "vacuum external pressure")
        self.assertEqual(hits[0].chunk["clause_id"], "UG-28")

    def test_filters_across_books(self):
        allhits = storage.keyword_search(self.con, "nozzle opening reinforcement")
        self.assertEqual({h.chunk["doc_id"] for h in allhits}, {"viii1_2025", "viii2_2025"})
        only2 = storage.keyword_search(self.con, "nozzle opening reinforcement", {"doc_id": "viii2_2025"})
        self.assertEqual({h.chunk["doc_id"] for h in only2}, {"viii2_2025"})
        self.assertEqual(storage.keyword_search(self.con, "nozzle", {"edition": 1999}), [])
        self.assertTrue(storage.keyword_search(self.con, "nozzle", {"code": "ASME BPVC", "division": "VIII-1"}))

    def test_reingest_replaces_document(self):
        docs = {d["id"]: d for d in self.cfg["documents"]}
        ingest_pages(self.con, self.cfg, docs["viii1_2025"], [(1, [B("UG-1 Scope", True), B("New text.")])])
        self.assertEqual(storage.keyword_search(self.con, "UG-37"), [])
        self.assertEqual(len(storage.get_clause(self.con, "ug-1", "viii1_2025")), 1)

    def test_embeddings_created_once_and_redo(self):
        n = storage.embed_chunks(self.con, self.client, self.cfg)
        self.assertGreater(n, 0)
        self.assertEqual(storage.embed_chunks(self.con, self.client, self.cfg), 0)
        self.assertEqual(storage.embed_chunks(self.con, self.client, self.cfg, redo=True), n)

    def test_vector_and_hybrid_search_merge(self):
        storage.embed_chunks(self.con, self.client, self.cfg)
        hits = storage.hybrid_search(self.con, self.client, self.cfg, "UG-37 nozzle reinforcement", k=5)
        self.assertTrue(hits)
        top = hits[0]
        self.assertEqual(top.chunk["clause_id"], "UG-37")
        self.assertIn("keyword", top.sources)
        self.assertNotIn("vec", top.chunk)

    def test_hybrid_falls_back_to_keyword_when_no_embeddings(self):
        hits = storage.hybrid_search(self.con, self.client, self.cfg, "UG-28 vacuum")
        self.assertEqual(hits[0].sources, ["keyword"])

    def test_links_stored_and_structure_counts(self):
        n = self.con.execute("select count(*) from links where kind='book'").fetchone()[0]
        self.assertEqual(n, 1)
        rows = storage.structure_counts(self.con)
        self.assertIn("viii1_2025", [r[0] for r in rows])
        self.assertEqual([r[0] for r in storage.structure_counts(self.con, "UG")], ["viii1_2025"])


if __name__ == "__main__":
    unittest.main()
