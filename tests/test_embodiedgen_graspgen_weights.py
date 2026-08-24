import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
PRELOADER = ROOT / "modal_build" / "embodiedgen_graspgen_weights.py"
spec = importlib.util.spec_from_file_location("graspgen_weights", PRELOADER)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class GraspGenWeightsTest(unittest.TestCase):
    def test_revision_is_exact(self):
        self.assertRegex(mod.GRASPGEN_MODELS_REVISION, r"^[0-9a-f]{40}$")
        self.assertRegex(mod.GRASPGEN_CONFIG_SHA256, r"^[0-9a-f]{64}$")
        self.assertRegex(mod.GRASPGEN_GEN_SHA256, r"^[0-9a-f]{64}$")
        self.assertRegex(mod.GRASPGEN_DIS_SHA256, r"^[0-9a-f]{64}$")

    def test_preloader_is_revision_pinned_and_atomic(self):
        source = PRELOADER.read_text()
        self.assertIn("revision=GRASPGEN_MODELS_REVISION", source)
        self.assertIn("graspgen_franka_panda_gen.pth", source)
        self.assertIn("graspgen_franka_panda_dis.pth", source)
        self.assertIn("actual_hashes[key] != expected", source)
        self.assertIn("weights.commit()", source)
        self.assertLess(source.index("GRASPGEN_MANIFEST.write_text"), source.index("weights.commit()"))


if __name__ == "__main__":
    unittest.main()
