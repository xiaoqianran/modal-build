import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


class HYWorld2ArtifactPlanTest(unittest.TestCase):
    def test_restricted_hy_artifacts_never_public_release(self):
        plan = json.loads((ROOT / "env/artifact-plan.json").read_text())
        by_name = {item["name"]: item for item in plan["artifacts"]}
        self.assertFalse(by_name["HY-World gsplat_maskgaussian"]["github_release"])
        self.assertFalse(by_name["HY-World recast navmesh binding"]["github_release"])

    def test_native_abi_is_blackwell_target(self):
        env = json.loads((ROOT / "env/hyworld2-py311-cu128-torch271-sm120-v1.json").read_text())
        self.assertEqual(env["python"], "3.11")
        self.assertEqual(env["cuda"], "12.8.1")
        self.assertEqual(env["torch"], "2.7.1")
        self.assertEqual(env["cuda_arch"], "12.0")
        self.assertEqual(env["target_gpu"], "RTX-PRO-6000")

    def test_publish_script_fails_closed(self):
        source = (ROOT / "scripts/publish_from_volume.sh").read_text()
        self.assertIn("manifest public_release=false", source)
        self.assertIn("exit 3", source)


if __name__ == "__main__":
    unittest.main()
