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

    def test_native_abi_has_hopper_target(self):
        env = json.loads((ROOT / "env/hyworld2-py311-cu128-torch271-sm90-v1.json").read_text())
        self.assertEqual(env["python"], "3.11")
        self.assertEqual(env["cuda"], "12.8.1")
        self.assertEqual(env["torch"], "2.7.1")
        self.assertEqual(env["cuda_arch"], "9.0")
        self.assertEqual(env["target_gpu"], "H100")
        self.assertTrue(all("sm90" in tag or "oss-source" in tag for tag in env["bundles"].values()))

    def test_stage3_h100_runtime_uses_minimal_public_native_bundle(self):
        env = json.loads((ROOT / "env/hyworld2-stage3-py311-cu128-torch271-sm90-v1.json").read_text())
        self.assertEqual(env["target_gpu"], "H100")
        self.assertEqual(env["cuda_arch"], "9.0")
        self.assertEqual(
            env["bundles"],
            {
                "stage3_native": "hyworld2-stage3-native-py311-cu128-torch271-sm90-v1",
                "oss_source": "hyworld2-oss-source-py311-v1",
            },
        )
        self.assertEqual(env["attention"]["default"], "torch-sdpa")

    def test_publish_script_fails_closed(self):
        source = (ROOT / "scripts/publish_from_volume.sh").read_text()
        self.assertIn("manifest public_release=false", source)
        self.assertIn("exit 3", source)

    def test_comfyui_runtime_records_incompatible_release_abi(self):
        env = json.loads(
            (ROOT / "env/hyworld2-comfyui-py312-cu130-torch291-sm120-v1.json").read_text()
        )
        self.assertEqual(env["target_gpu"], "RTX-PRO-6000")
        self.assertEqual(env["cuda_arch"], "12.0")
        self.assertFalse(
            env["artifact_compatibility"]["hyworld2-py311-cu128-torch271-sm120-v1"]
        )


if __name__ == "__main__":
    unittest.main()
