import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
PRELOADER = ROOT / "modal_build" / "birefnet_weights.py"
spec = importlib.util.spec_from_file_location("birefnet_weights", PRELOADER)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class BiRefNetWeightsTest(unittest.TestCase):
    def test_weight_identity_is_exact(self):
        self.assertEqual(mod.MODEL_NAME, "birefnet-general-lite")
        self.assertEqual(mod.MODEL_BYTES, 224_005_088)
        self.assertRegex(mod.MODEL_MD5, r"^[0-9a-f]{32}$")
        self.assertRegex(mod.MODEL_SHA256, r"^[0-9a-f]{64}$")
        self.assertEqual(mod.VOLUME_NAME, "modal-3d-birefnet-weights")

    def test_preloader_verifies_before_commit(self):
        source = PRELOADER.read_text(encoding="utf-8")
        self.assertIn("size != MODEL_BYTES", source)
        self.assertIn("md5 != MODEL_MD5", source)
        self.assertIn("sha256 != MODEL_SHA256", source)
        self.assertIn("WEIGHT_MANIFEST.write_text", source)
        self.assertIn("weights.commit()", source)
        self.assertLess(source.index("WEIGHT_MANIFEST.write_text"), source.index("weights.commit()"))


if __name__ == "__main__":
    unittest.main()
