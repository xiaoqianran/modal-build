import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
PRELOADER = ROOT / "build" / "embodiedgen_affordance_weights.py"
spec = importlib.util.spec_from_file_location("affordance_weights", PRELOADER)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class AffordanceWeightsTest(unittest.TestCase):
    def test_model_revisions_are_exact(self):
        self.assertRegex(mod.HUNYUAN3D_PART_MODEL_REVISION, r"^[0-9a-f]{40}$")
        self.assertRegex(mod.SONATA_MODEL_REVISION, r"^[0-9a-f]{40}$")
        self.assertRegex(mod.P3SAM_WEIGHT_SHA256, r"^[0-9a-f]{64}$")
        self.assertRegex(mod.SONATA_WEIGHT_SHA256, r"^[0-9a-f]{64}$")

    def test_preloader_records_hashes_and_commits_last(self):
        source = PRELOADER.read_text(encoding="utf-8")
        self.assertIn("actual_p3sam_sha256 != P3SAM_WEIGHT_SHA256", source)
        self.assertIn("actual_sonata_sha256 != SONATA_WEIGHT_SHA256", source)
        self.assertIn("weights.commit()", source)
        self.assertLess(source.index("WEIGHT_MANIFEST.write_text"), source.index("weights.commit()"))
        self.assertIn("revision=HUNYUAN3D_PART_MODEL_REVISION", source)
        self.assertIn("revision=SONATA_MODEL_REVISION", source)


if __name__ == "__main__":
    unittest.main()
