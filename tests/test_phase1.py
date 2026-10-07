"""Phase 1 tests: parser on a SYNTHETIC two-column page (not a real ASME page)."""
import unittest

from asme_rag.pdf_parser import blocks_from_dict

W, H = 600.0, 800.0


def mk(text, x0, y0, x1, y1, size=10, flags=0, font="Times", direction=(1.0, 0.0)):
    """Build one synthetic text block with a single line and a single span."""
    span = {"text": text, "size": size, "flags": flags, "font": font}
    line = {"bbox": (x0, y0, x1, y1), "dir": direction, "spans": [span]}
    return {"type": 0, "bbox": (x0, y0, x1, y1), "lines": [line]}


def page(blocks):
    return {"width": W, "height": H, "blocks": blocks}


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.page = page([
            mk("RUNNING HEADER", 50, 10, 550, 25),                       # header zone
            mk("Full width title", 50, 60, 550, 80, size=14, flags=16),  # splits bands
            mk("Right col 1", 320, 100, 560, 120),
            mk("Left col 1", 40, 100, 280, 120),
            mk("Left col 2", 40, 130, 280, 150),
            mk("Right col 2", 320, 130, 560, 150),
            mk("DRAFT WATERMARK", 100, 300, 500, 500, direction=(0.7, 0.7)),
            mk("Chapter ........ 12", 40, 200, 280, 215),                # TOC line
            mk("Licensed to someone", 40, 700, 280, 715),                # strip-pattern line
            mk("Page 5", 250, 780, 350, 795),                            # footer zone
        ])

    def texts(self, **kw):
        cfg = {"strip_patterns": [r"^Licensed to"]}
        return [b.text for b in blocks_from_dict(self.page, 1, cfg, **kw)]

    def test_reading_order_two_columns(self):
        self.assertEqual(self.texts(), ["Full width title", "Left col 1", "Left col 2",
                                        "Right col 1", "Right col 2"])

    def test_header_footer_watermark_toc_strip_removed(self):
        out = " ".join(self.texts())
        for gone in ("HEADER", "Page 5", "WATERMARK", "Chapter", "Licensed"):
            self.assertNotIn(gone, out)

    def test_toc_kept_on_request(self):
        self.assertIn("Chapter ........ 12", self.texts(keep_toc=True))

    def test_bold_size_and_superscript_flags(self):
        pg = page([{"type": 0, "bbox": (40, 100, 280, 120), "lines": [{
            "bbox": (40, 100, 280, 120), "dir": (1.0, 0.0), "spans": [
                {"text": "Allowable stress", "size": 10, "flags": 0, "font": "Times"},
                {"text": "12", "size": 6, "flags": 1, "font": "Times"}]}]},
            mk("Heading", 40, 200, 280, 220, size=14, font="Times-Bold")])
        blocks = blocks_from_dict(pg, 3)
        self.assertEqual(blocks[0].superscripts, ["12"])
        self.assertTrue(blocks[1].bold)
        self.assertEqual(blocks[1].size, 14)
        self.assertEqual(blocks[0].page, 3)

    def test_image_blocks_skipped(self):
        pg = page([{"type": 1, "bbox": (0, 100, 100, 200)}, mk("Text", 40, 300, 280, 320)])
        self.assertEqual([b.text for b in blocks_from_dict(pg, 1)], ["Text"])

    def test_single_column_sorted_by_y(self):
        pg = page([mk("B", 40, 200, 280, 220), mk("A", 40, 100, 280, 120)])
        self.assertEqual([b.text for b in blocks_from_dict(pg, 1)], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
