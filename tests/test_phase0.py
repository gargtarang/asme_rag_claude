"""Phase 0 tests: config loading, fake server, LM client, `check` command."""
import io
import contextlib
import unittest
from pathlib import Path

from asme_rag.config import load_config
from asme_rag.lmclient import LMClient
from asme_rag.cli import cmd_check, tiny_png
from tests.fake_lm_server import start_fake_server, CANNED_CHAT, CANNED_IMAGE

ROOT = Path(__file__).resolve().parent.parent


class Phase0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv, cls.url = start_fake_server()
        cls.cfg = load_config(ROOT / "config.toml")
        cls.cfg["lmstudio"]["base_url"] = cls.url

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_config_has_six_documents_and_open_in_vscode(self):
        self.assertEqual(len(self.cfg["documents"]), 6)
        self.assertIs(self.cfg["paths"]["open_in_vscode"], True)
        self.assertEqual(self.cfg["xref"]["max_depth"], 1)

    def test_models_and_chat(self):
        c = LMClient(self.cfg)
        self.assertIn("gemma-4-26b-a4b", c.list_models())
        self.assertEqual(c.chat([{"role": "user", "content": "hi"}]), CANNED_CHAT)

    def test_streaming(self):
        c = LMClient(self.cfg)
        text = "".join(c.chat_stream([{"role": "user", "content": "hi"}])).strip()
        self.assertEqual(text, CANNED_CHAT)

    def test_embeddings_deterministic(self):
        c = LMClient(self.cfg)
        a, b = c.embed(["x", "y"]), c.embed(["x", "y"])
        self.assertEqual(a, b)
        self.assertEqual(len(a), 2)

    def test_image_payload_contains_image(self):
        c = LMClient(self.cfg)
        before = len(self.srv.image_requests)
        self.assertEqual(c.chat_with_image("describe", tiny_png()), CANNED_IMAGE)
        req = self.srv.image_requests[before]
        parts = req["messages"][0]["content"]
        self.assertTrue(any(p["type"] == "image_url" and
                            p["image_url"]["url"].startswith("data:image/png;base64,") for p in parts))

    def test_check_runs_against_fake_server(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cmd_check(self.cfg, None)
        out = buf.getvalue()
        self.assertIn("LM Studio: reachable", out)
        self.assertIn("vision: working", out)
        self.assertIn("MISSING", out)  # PDFs are not in the sandbox
        self.assertEqual(rc, 1)        # missing PDFs count as problems


if __name__ == "__main__":
    unittest.main()
