"""Phase 2 tests: clause detection, chunk splitting, tables, notes, cross-references (SYNTHETIC text)."""
import unittest

from asme_rag.chunker import chunk_document, parse_table_rows, _split_long
from asme_rag.pdf_parser import Block
from asme_rag.xref import build_index

DOC = {"id": "viii1_2025", "code": "ASME BPVC", "division": "VIII-1", "edition": 2025,
       "parser_profile": "code_book"}
DOCS = [DOC, {"id": "viii2_2025", "code": "ASME BPVC", "division": "VIII-2", "edition": 2025,
              "parser_profile": "code_book"}]


def B(text, page=1, bold=False, size=10.0, sup=None):
    return Block(text=text, bbox=(0, 0, 1, 1), page=page, size=size, bold=bold, superscripts=sup or [])


def pages():
    return [(1, [
        B("Part UG", bold=True),
        B("UG-37 Reinforcement Required for Openings", bold=True),
        B("(a) Nomenclature applies.", sup=["1"]),
        B("(b) See UG-36 through UG-42 and Table UG-37.1.\nSee Division 2, Part 4.12 for details."),
        B("1 Footnote text for the marker.", size=7),
        B("Table UG-37.1 Sample", bold=True),
    ]), (2, [
        B("Table UG-37.1 Sample\nrow1  10  20\nrow2  30  40"),
        B("NOTE: (1) Applies to openings only."),
        B("UG-36 Openings in Pressure Vessels", bold=True),
        B("Body of UG-36."),
        B("FIG. UG-37 Reinforcement geometry", bold=True),
        B("Nonmandatory Appendix A", bold=True),
        B("1-1 Scope", bold=True),
        B("Appendix body."),
    ])]


class ChunkerTests(unittest.TestCase):
    def setUp(self):
        self.chunks = chunk_document(pages(), DOC)

    def by(self, **kw):
        return [c for c in self.chunks if all(getattr(c, k) == v for k, v in kw.items())]

    def test_clause_and_part_detection(self):
        c = self.by(clause_id="UG-37", type="text")[0]
        self.assertEqual(c.part_or_subsection, "Part UG")
        self.assertIn("UG-36 through UG-42", c.text)
        self.assertEqual(c.heading_path, ["Part UG", "UG-37"])

    def test_body_is_not_mistaken_for_heading(self):
        self.assertEqual(len(self.by(clause_id="UG-36", type="text")), 1)

    def test_table_figure_note_types(self):
        t = self.by(type="table")
        self.assertTrue(all(x.clause_id == "Table UG-37.1" for x in t))
        self.assertEqual(self.by(type="figure_caption")[0].clause_id, "Fig. UG-37")
        note = self.by(type="note")[0]
        self.assertEqual(note.linked_to, "Table UG-37.1")

    def test_table_rows_structured(self):
        tbl = [c for c in self.chunks if c.type == "table" and c.rows][0]
        self.assertEqual(tbl.rows[0], ["row1", "10", "20"])
        self.assertFalse(tbl.unreliable)

    def test_unreliable_table_flagged(self):
        rows, bad = parse_table_rows("Table X\na  b  c  d  e\nf")
        self.assertTrue(bad)

    def test_appendix_status_and_metadata(self):
        c = self.by(clause_id="1-1")[0]
        self.assertEqual(c.status, "non-mandatory")
        self.assertEqual(c.part_or_subsection, "Nonmandatory Appendix A")
        for f in ("doc_id", "code", "division", "edition", "pdf_page", "model_generated", "verified", "bold_flags"):
            self.assertTrue(hasattr(c, f))
        self.assertFalse(c.model_generated or c.verified)

    def test_footnote_chunk_and_garble_flag(self):
        self.assertEqual(len(self.by(type="footnote")), 1)
        g = chunk_document([(1, [B("UG-1 Title", bold=True), B("bad \ufffd text")])], DOC)
        self.assertTrue(g[0].unreliable)

    def test_long_clause_split_at_subparagraphs_keeps_parent(self):
        body = "\n".join(f"({chr(97 + i)}) " + "word " * 40 for i in range(10))
        ch = chunk_document([(1, [B("UG-99 Long", bold=True), B(body)])], DOC, max_chars=500)
        self.assertGreater(len(ch), 1)
        self.assertTrue(all(c.clause_id == "UG-99" for c in ch))
        self.assertTrue(all(len(c.text) <= 500 for c in ch))
        self.assertTrue(all(c.text.startswith("(") for c in ch[1:]))
        self.assertTrue(ch[0].text.startswith("UG-99"))

    def test_split_long_no_boundary_hard_split(self):
        self.assertTrue(all(len(p) <= 100 for p in _split_long("x" * 350, 100)))


class XrefTests(unittest.TestCase):
    def setUp(self):
        self.chunks = chunk_document(pages(), DOC)
        self.links = build_index(self.chunks, DOCS)

    def kinds(self, kind):
        return [l for l in self.links if l.kind == kind]

    def test_range_detected_and_not_split_into_singles(self):
        self.assertEqual(len(self.kinds("range")), 1)
        singles = [l.target for l in self.kinds("clause")]
        self.assertNotIn("UG-42", singles)

    def test_table_reference_resolved(self):
        t = [l for l in self.kinds("table") if l.target.lower() == "table ug-37.1"]
        self.assertTrue(t and t[0].status == "resolved")

    def test_other_book_reference_resolved_when_ingested_and_not_otherwise(self):
        b = self.kinds("book")[0]
        self.assertEqual((b.status, b.target_doc), ("resolved", "viii2_2025"))
        only1 = build_index(self.chunks, [DOC])
        b1 = [l for l in only1 if l.kind == "book"][0]
        self.assertEqual((b1.status, b1.target_doc), ("unresolved", "VIII-2"))

    def test_footnote_and_table_note_links(self):
        self.assertTrue(self.kinds("footnote"))
        self.assertTrue(self.kinds("table_note"))

    def test_deterministic_order(self):
        again = build_index(chunk_document(pages(), DOC), DOCS)
        self.assertEqual([(l.from_chunk, l.kind, l.target) for l in self.links],
                         [(l.from_chunk, l.kind, l.target) for l in again])

    def test_no_self_reference(self):
        ug37 = [c for c in self.chunks if c.clause_id == "UG-37" and c.type == "text"][0]
        self.assertFalse([l for l in self.links if l.from_chunk == ug37.id and l.target == "UG-37"])


if __name__ == "__main__":
    unittest.main()


class XrefMaskingTests(unittest.TestCase):
    def test_table_reference_does_not_create_a_fake_clause_link(self):
        ch = chunk_document([(1, [B("UG-1 Title", bold=True), B("See Table UG-37.1 and Fig. UG-37 only. B31.3 Table A-1.")])], DOC)
        links = build_index(ch, DOCS)
        self.assertEqual([l.target for l in links if l.kind == "clause"], [])
        self.assertEqual([l.target for l in links if l.kind == "table"], ["Table UG-37.1"])
        self.assertEqual([l.target for l in links if l.kind == "book"], ["B31.3 Table A-1"])
