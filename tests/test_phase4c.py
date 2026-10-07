"""Phase 4c tests: depth-1 cross-reference expansion on a SYNTHETIC corpus."""
import tempfile
import unittest
from pathlib import Path

from asme_rag import citecheck, expand, rag, storage
from asme_rag.config import load_config
from asme_rag.ingest import ingest_pages, rebuild_links
from asme_rag.lmclient import LMClient
from asme_rag.pdf_parser import Block
from tests.fake_lm_server import start_fake_server

ROOT = Path(__file__).resolve().parent.parent


def B(text, page, bold=False, size=10.0, sup=None):
    return Block(text=text, bbox=(0, 0, 1, 1), page=page, size=size, bold=bold, superscripts=sup or [])


VIII1 = [
    (1, [B("Part UG", 1, True),
         B("UG-37 Reinforcement Required for Openings", 1, True),
         B("(a) Openings need reinforcement. See UG-36 and Table UG-37.1. See also UG-40 through UG-42. "
           "See Division 2, Part 4.5 and B31.3 Table A-1.", 1, sup=["1"]),
         B("1 Footnote: applies to circular openings.", 1, size=7),
         B("UG-36 Openings in Pressure Vessels", 1, True),
         B("(a) Openings follow UG-37 and the neck rules of UG-45. See Fig. UG-37.", 1),
         B("Table UG-37.1 Sample values\nrow1  10  20", 1),
         B("NOTE: (1) Table note applies to all rows.", 1)]),
    (2, [B("UG-40 Limits of Reinforcement", 2, True), B("Limit text for reinforcement.", 2),
         B("UG-41 Strength of Attachments", 2, True), B("Strength text.", 2),
         B("UG-42 Reinforcement of Multiple Openings", 2, True), B("Multiple openings text. See UG-99.", 2),
         B("UG-45 Nozzle Neck Thickness", 2, True), B("Neck thickness text. See UG-16.", 2),
         B("UG-16 General Minimum Thickness", 2, True), B("Minimum thickness text.", 2)]),
]
VIII2 = [(9, [B("4.5.1 Openings in Shells", 9, True), B("Opening rules of Division 2. See B16.5 Table 2-1.1.", 9)])]


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
        self.cfg["xref"] = {"max_depth": 1, "max_clauses": 40, "ask_when_larger_than": 40,
                            "include_footnotes": True, "show_unexpanded": True, "write_pack_file": True,
                            "prompt_to_expand": True, "prompt_max_items": 10}
        self.con = storage.connect(Path(self.tmp.name) / "t.sqlite")
        docs = {d["id"]: d for d in self.cfg["documents"]}
        ingest_pages(self.con, self.cfg, docs["viii1_2025"], VIII1)
        ingest_pages(self.con, self.cfg, docs["viii2_2025"], VIII2)
        rebuild_links(self.con, self.cfg)
        self.ug37 = storage.get_clause(self.con, "UG-37", "viii1_2025")[:1]

    def tearDown(self):
        self.srv.script = None
        self.con.close()
        self.tmp.cleanup()

    def clauses(self, pack, group):
        return [i.chunk["clause_id"] for i in pack.group(group)]


class ExpandTests(Base):
    def test_hard_depth_limit_rejected(self):
        self.cfg["xref"]["max_depth"] = 2
        with self.assertRaises(ValueError) as cm:
            expand.xref_settings(self.cfg)
        self.assertIn("MAX_DEPTH_HARD", str(cm.exception))

    def test_level0_and_level1(self):
        pack = expand.build_pack(self.con, self.cfg, self.ug37)
        self.assertEqual(self.clauses(pack, "level0"), ["UG-37"])
        lvl1 = self.clauses(pack, "level1")
        for want in ("UG-36", "Table UG-37.1", "UG-40", "UG-41", "UG-42", "4.5.1"):
            self.assertIn(want, lvl1)

    def test_range_includes_every_clause_between(self):
        pack = expand.build_pack(self.con, self.cfg, self.ug37)
        self.assertEqual([c for c in self.clauses(pack, "level1") if c in ("UG-40", "UG-41", "UG-42")],
                         ["UG-40", "UG-41", "UG-42"])

    def test_chain_beyond_depth_one_is_listed_not_expanded(self):
        pack = expand.build_pack(self.con, self.cfg, self.ug37)
        every = self.clauses(pack, "level0") + self.clauses(pack, "level1")
        self.assertNotIn("UG-45", every)
        self.assertNotIn("UG-16", every)
        targets = [f.target for f in pack.further]
        self.assertIn("UG-45", targets)
        self.assertIn("UG-99", targets)                       # unresolved target is still listed
        self.assertEqual([f.n for f in pack.further], list(range(1, len(pack.further) + 1)))

    def test_loop_does_not_repeat_items(self):
        pack = expand.build_pack(self.con, self.cfg, self.ug37)       # UG-36 refers back to UG-37
        ids = [i.chunk["id"] for i in pack.items]
        self.assertEqual(len(ids), len(set(ids)))

    def test_footnote_and_table_note_belong_to_their_item(self):
        pack = expand.build_pack(self.con, self.cfg, self.ug37)
        a = pack.group("level0")[0]
        self.assertTrue(any("circular openings" in n["text"] for n in a.notes))
        tbl = next(i for i in pack.items if i.chunk["clause_id"] == "Table UG-37.1")
        self.assertTrue(any("Table note applies" in n["text"] for n in tbl.notes))
        md = expand.render_pack(pack)
        self.assertIn("Footnote (PDF p.1)", md)
        self.assertIn("Table note:", md)

    def test_other_book_not_ingested_in_place_and_in_needs_input(self):
        pack = expand.build_pack(self.con, self.cfg, self.ug37)
        inline = expand.render_inline(pack, self.con)
        self.assertIn("NOT INGESTED: B31.3", inline)
        self.assertIn("see [ASME BPVC VIII-2, 2025, 4.5.1, PDF p.9]", inline)
        self.assertTrue(any("B31.3" in x for x in expand.needs_input(pack)))

    def test_stable_order_and_sections(self):
        a = [i.chunk["id"] for i in expand.build_pack(self.con, self.cfg, self.ug37).items]
        b = [i.chunk["id"] for i in expand.build_pack(self.con, self.cfg, self.ug37).items]
        self.assertEqual(a, b)
        md = expand.render_pack(expand.build_pack(self.con, self.cfg, self.ug37))
        order = [md.index(h) for h in ("## Level 0", "## Level 1", "## Further references (not expanded)", "## Unresolved")]
        self.assertEqual(order, sorted(order))
        self.assertIn("Why included:", md)

    def test_size_cap_and_ask(self):
        self.cfg["xref"]["max_clauses"] = 3
        pack = expand.build_pack(self.con, self.cfg, self.ug37)
        self.assertLessEqual(len(pack.items), 3)
        self.assertTrue(any("max_clauses" in n for n in pack.notes))
        self.cfg["xref"].update(max_clauses=40, ask_when_larger_than=2)
        asked = []
        pack = expand.build_pack(self.con, self.cfg, self.ug37, ask=lambda p: asked.append(p) or "2")
        self.assertEqual(len(asked), 1)
        self.assertEqual(len(pack.group("level1")), 2)
        pack = expand.build_pack(self.con, self.cfg, self.ug37, ask=lambda p: "clauses only")
        self.assertTrue(all(i.chunk["type"] == "text" for i in pack.items))

    def test_numbered_selection_expands_one_step_only(self):
        pack = expand.build_pack(self.con, self.cfg, self.ug37)
        n45 = next(f.n for f in pack.further if f.target == "UG-45")
        expand.expand_selection(self.con, self.cfg, pack, [n45])
        self.assertEqual(self.clauses(pack, "selected"), ["UG-45"])
        self.assertNotIn("UG-16", self.clauses(pack, "selected") + self.clauses(pack, "level1"))
        self.assertIn("UG-16", [f.target for f in pack.further])        # listed, not followed
        self.assertIn("## Selected expansions", expand.render_pack(pack))

    def test_parse_selection_and_prompt(self):
        self.assertEqual(expand.parse_selection("2, 5", 6), [2, 5])
        self.assertEqual(expand.parse_selection("all", 3), [1, 2, 3])
        self.assertEqual(expand.parse_selection("none", 3), [])
        self.assertEqual(expand.parse_selection("9,1", 3), [1])
        pack = expand.build_pack(self.con, self.cfg, self.ug37)
        before = len(pack.items)
        expand.prompt_expand(self.con, self.cfg, pack, lambda p: "none")
        self.assertEqual(len(pack.items), before)
        self.cfg["xref"]["prompt_to_expand"] = False
        expand.prompt_expand(self.con, self.cfg, pack, lambda p: self.fail("must not ask"))

    def test_refs_clause_is_a_new_anchor(self):
        anchors = storage.get_clause(self.con, "UG-36", "viii1_2025")
        pack = expand.build_pack(self.con, self.cfg, anchors)
        self.assertIn("UG-45", self.clauses(pack, "level1"))
        self.assertIn("UG-37", self.clauses(pack, "level1"))


class IntegrationTests(Base):
    def test_answer_gets_level1_excerpts_and_citecheck_covers_them(self):
        seen = {}
        def script(req):
            seen["prompt"] = req["messages"][-1]["content"]
            return "Limits apply [ASME BPVC VIII-1, 2025, UG-40, PDF p.2].\nVerify: UG-40"
        self.srv.script = script
        res = rag.answer(self.con, self.client, self.cfg, "UG-37 reinforcement required openings")
        self.assertIn("UG-40", seen["prompt"])
        results = res.extras["citation_results"]
        self.assertEqual([r.status for r in results], ["matched"])
        self.assertIn("pack", res.extras)

    def test_markdown_has_pack_section_and_pack_file(self):
        res = rag.answer(self.con, self.client, self.cfg, "UG-37 reinforcement required openings")
        path = rag.write_markdown(self.cfg, res)
        md = path.read_text(encoding="utf-8")
        self.assertIn("## References inline", md)
        self.assertIn("# Reference pack", md)
        self.assertIn("## NEEDS INPUT", md)
        self.assertTrue(md.rstrip().splitlines()[-1].startswith("Verify:"))
        refs = path.parent / f"{path.stem}_refs.md"
        self.assertTrue(refs.exists())
        self.assertIn("Level 1", refs.read_text(encoding="utf-8"))

    def test_pack_file_can_be_turned_off(self):
        self.cfg["xref"]["write_pack_file"] = False
        res = rag.answer(self.con, self.client, self.cfg, "UG-37 reinforcement required openings")
        path = rag.write_markdown(self.cfg, res)
        self.assertFalse((path.parent / f"{path.stem}_refs.md").exists())

    def test_chat_prompts_expand_which(self):
        asked = []
        answers = iter(["UG-37 reinforcement required openings", "1", "/quit"])
        def inp(p):
            asked.append(p)
            return next(answers)
        rag.chat_loop(self.con, self.client, self.cfg, {}, None, input_fn=inp,
                      out_fn=lambda t: None, token_fn=lambda t: None)
        self.assertTrue(any("Expand which?" in p for p in asked))

    def test_rulemap_sections(self):
        text, pack = expand.rulemap(self.con, self.client, self.cfg, "reinforcement required openings")
        for h in ("## Governing clauses (level 0)", "## Direct references (level 1)", "## Tables and charts needed",
                  "## Cross-book first hops", "UNVERIFIED", "## Further references (not expanded)", "## NEEDS INPUT"):
            self.assertIn(h, text)
        self.assertIn("Table UG-37.1", text)
        self.assertIn("Division 2, Part 4.5", text)


if __name__ == "__main__":
    unittest.main()
