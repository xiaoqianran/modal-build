import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUNTIME = ROOT / "runtime" / "embodiedgen_affordance_semantic.py"
spec = importlib.util.spec_from_file_location("embodiedgen_affordance_semantic", RUNTIME)
semantic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(semantic)


class AffordanceSemanticContractTest(unittest.TestCase):
    def manifest(self):
        digest = "a" * 64
        return {
            "version": 1,
            "sourceJobId": "job-" + "1" * 32,
            "outputJobId": "job-" + "2" * 32,
            "category": "mug",
            "segmentation": {"path": "affordance/agentscape_part_segmentation.v1.json", "sha256": digest},
            "images": {
                "rgbGrid": {"path": "affordance/semantic_inputs/rgb.png", "sha256": digest, "mediaType": "image/png"},
                "maskGrid": {"path": "affordance/semantic_inputs/mask.png", "sha256": digest, "mediaType": "image/png"},
                "partAtlas": {"path": "affordance/semantic_inputs/atlas.png", "sha256": digest, "mediaType": "image/png"},
            },
            "parts": [{"id": "0", "maskColor": "Red"}, {"id": "1", "maskColor": "Green"}],
        }

    def response(self):
        return {
            "parts": [
                {
                    "id": "0",
                    "mask_color": "Red",
                    "part_name": "mug handle",
                    "graspable": True,
                    "grasp_scenarios": [{"scenario": "grasp handle", "confidence": 0.9}],
                    "functional_labels": ["provide side grip"],
                    "semantic_description": "Curved side handle for holding the mug.",
                },
                {
                    "id": "1",
                    "mask_color": "Green",
                    "part_name": "mug body",
                    "graspable": True,
                    "grasp_scenarios": [{"scenario": "grasp body", "confidence": 0.6}],
                    "functional_labels": ["contain liquid"],
                    "semantic_description": "Main cup body that contains liquid.",
                },
            ]
        }

    def test_worker_is_isolated_cpu_network_surface_and_uses_uv(self):
        source = RUNTIME.read_text(encoding="utf-8")
        self.assertIn('APP_NAME = "modal-3d-embodiedgen-affordance-semantic"', source)
        self.assertIn('.uv_pip_install("openai==1.101.0", "pillow==11.3.0")', source)
        self.assertIn('SEMANTIC_SECRET_NAME = "modal-3d-embodiedgen-affordance-semantic"', source)
        self.assertIn('SEMANTIC_INPUT_PATH = Path("affordance/semantic_inputs/semantic_inputs.v1.json")', source)
        self.assertIn('required_keys=["ENDPOINT", "API_KEY", "MODEL_NAME"]', source)
        worker = source[source.index("def annotate_semantics(") :]
        self.assertNotIn('gpu=', worker)
        self.assertNotIn('torch', source.lower())
        self.assertNotIn('nvcc', source.lower())

    def test_manifest_is_hash_bound_and_path_safe(self):
        normalized = semantic.validate_semantic_input_manifest(self.manifest())
        self.assertEqual([x["id"] for x in normalized["parts"]], ["0", "1"])
        bad = self.manifest()
        bad["images"]["rgbGrid"]["path"] = "../secret.png"
        with self.assertRaises(ValueError):
            semantic.validate_semantic_input_manifest(bad)
        bad = self.manifest()
        bad["segmentation"]["sha256"] = "bad"
        with self.assertRaises(ValueError):
            semantic.validate_semantic_input_manifest(bad)

    def test_semantic_response_is_strict_and_never_promotes_joint_action_truth(self):
        normalized = semantic.validate_semantic_response(self.response(), self.manifest()["parts"])
        self.assertEqual(len(normalized), 2)
        bad = self.response()
        bad["parts"][0]["joint"] = {"type": "revolute"}
        with self.assertRaisesRegex(ValueError, "forbidden executable fields"):
            semantic.validate_semantic_response(bad, self.manifest()["parts"])
        bad = self.response()
        bad["parts"][0]["mask_color"] = "Blue"
        with self.assertRaisesRegex(ValueError, "mask color mismatch"):
            semantic.validate_semantic_response(bad, self.manifest()["parts"])

    def test_numeric_zero_part_id_is_preserved_and_checked(self):
        response = self.response()
        response["parts"][0]["id"] = 0
        response["parts"][1]["id"] = 1
        normalized = semantic.validate_semantic_response(response, self.manifest()["parts"])
        self.assertEqual([item["id"] for item in normalized], ["0", "1"])
        response["parts"][0]["id"] = True
        with self.assertRaisesRegex(ValueError, "boolean part id"):
            semantic.validate_semantic_response(response, self.manifest()["parts"])

    def test_non_graspable_null_scenarios_normalize_to_empty_list(self):
        response = self.response()
        response["parts"][0]["graspable"] = False
        response["parts"][0]["grasp_scenarios"] = None
        normalized = semantic.validate_semantic_response(response, self.manifest()["parts"])
        self.assertEqual(normalized[0]["grasp_scenarios"], [])
        response = self.response()
        response["parts"][0]["grasp_scenarios"] = []
        with self.assertRaisesRegex(ValueError, "requires at least one grasp scenario"):
            semantic.validate_semantic_response(response, self.manifest()["parts"])

    def test_non_graspable_part_cannot_smuggle_grasp_scenarios(self):
        bad = self.response()
        bad["parts"][0]["graspable"] = False
        with self.assertRaisesRegex(ValueError, "must not contain grasp_scenarios"):
            semantic.validate_semantic_response(bad, self.manifest()["parts"])

    def test_prompt_and_output_provenance_exclude_credentials(self):
        source = RUNTIME.read_text(encoding="utf-8")
        self.assertRegex(semantic.PROMPT_REVISION, r"^[0-9a-f]{64}$")
        self.assertIn("Do not infer or output joints", semantic.SYSTEM_PROMPT)
        output_block = source[source.index('    output = {') : source.index('    output_path =', source.index('    output = {'))]
        self.assertNotIn("API_KEY", output_block)
        self.assertNotIn("ENDPOINT", output_block)
        self.assertIn('"promptRevision": PROMPT_REVISION', output_block)
        self.assertIn('"partAtlasSha256"', output_block)
        self.assertIn('isolated part atlas', semantic.SYSTEM_PROMPT.lower())
        self.assertIn('"requestIds": request_ids', output_block)

    def test_data_url_enforces_image_size_bounds(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.png"
            path.write_bytes(b"abc")
            value = semantic._data_url(path, "image/png")
            self.assertTrue(value.startswith("data:image/png;base64,"))
            path.write_bytes(b"")
            with self.assertRaises(ValueError):
                semantic._data_url(path, "image/png")


if __name__ == "__main__":
    unittest.main()
