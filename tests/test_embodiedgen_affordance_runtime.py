import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUNTIME = ROOT / "runtime" / "embodiedgen_affordance_l40s.py"

spec = importlib.util.spec_from_file_location("embodiedgen_affordance_runtime", RUNTIME)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class AffordanceRuntimeTest(unittest.TestCase):
    def test_every_external_revision_is_exact(self):
        for value in (
            runtime.EMBODIEDGEN_COMMIT,
            runtime.HUNYUAN3D_PART_COMMIT,
            runtime.HUNYUAN3D_PART_MODEL_REVISION,
            runtime.SONATA_MODEL_REVISION,
        ):
            self.assertRegex(value, r"^[0-9a-f]{40}$")

    def test_release_wheels_have_exact_sha256(self):
        self.assertEqual(len(runtime.AFFORDANCE_WHEELS), 3)
        for name, digest in runtime.AFFORDANCE_WHEELS.items():
            self.assertTrue(name.endswith(".whl"))
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        source = RUNTIME.read_text()
        self.assertIn("sha256sum -c -", source)
        self.assertNotIn("--clobber", source)

    def test_consumer_is_runtime_only_and_l40s(self):
        source = RUNTIME.read_text()
        self.assertIn("nvidia/cuda:12.6.3-runtime-ubuntu22.04", source)
        self.assertIn("! command -v nvcc", source)
        self.assertIn('.apt_install("git", "curl", "libgomp1", "clang")', source)
        self.assertIn("omegaconf==2.3.0", source)
        self.assertIn('gpu="L40S"', source)
        self.assertIn('"TORCH_CUDA_ARCH_LIST": "8.9"', source)

    def test_weights_are_revision_pinned_and_runtime_offline(self):
        source = RUNTIME.read_text()
        self.assertIn('weight_manifest.get("hunyuan3d_part_model_revision")', source)
        self.assertIn('weight_manifest.get("sonata_model_revision")', source)
        self.assertIn('"p3sam": P3SAM_WEIGHT_SHA256', source)
        self.assertIn('"sonata": SONATA_WEIGHT_SHA256', source)
        self.assertIn('"HF_HUB_OFFLINE": "1"', source)
        self.assertIn('/weights/affordance/sonata/sonata.pth', source)
        self.assertIn("base64.b64decode", source)
        self.assertIn("unexpected P3-SAM Sonata loader source", source)
        self.assertIn("grep -Fq 'custom_config={\\\"enable_flash\\\": False}'", source)

    def test_segmenter_has_no_gpt_dependency(self):
        source = RUNTIME.read_text()
        self.assertNotIn("GPT_CLIENT", source)
        self.assertNotIn("openai", source.lower())
        self.assertIn("from auto_mask import AutoMask", source)
        self.assertIn("is_parallel=False", source)
        self.assertIn("clean_mesh_flag=False", source)

    def test_segmenter_validates_geometry_and_outputs(self):
        source = RUNTIME.read_text()
        self.assertIn("face label count mismatch", source)
        self.assertIn("P3-SAM returned no valid parts", source)
        self.assertIn('"mesh_part_seg.glb"', source)
        self.assertIn('"part_segmentation.json"', source)
        self.assertIn("P3SAM_PART_SEGMENTATION_OK", source)

    def test_job_id_guard(self):
        self.assertTrue(runtime._valid_job_id("job-" + "a" * 32))
        for value in ("../x", "job-123", "bench-x", "job-" + "g" * 32):
            self.assertFalse(runtime._valid_job_id(value))

    def test_graspgen_raw_worker_is_pinned_and_gpt_free(self):
        source = RUNTIME.read_text()
        self.assertIn('GRASPGEN_COMMIT = "a56d518f3b76ea2a432b5b838b3c68027d29be49"', source)
        self.assertIn('GRASPGEN_MODELS_REVISION = "ec1ccbb5eec0680db669246ac312a3636f16ee43"', source)
        self.assertIn('image=grasp_image', source)
        self.assertIn('diffusers==0.11.1', source)
        self.assertIn('huggingface_hub==0.25.2', source)
        self.assertIn('enable_flash=False', source)
        self.assertIn("eval_utils/yourdfpy is not required", source)
        self.assertIn('GRASPGEN_RAW_GRASPS_OK', source)
        self.assertNotIn('GPT_CLIENT', source)

    def test_graspgen_worker_uses_urdf_collision_frame_and_validates_poses(self):
        source = RUNTIME.read_text()
        self.assertIn('def _load_urdf_collision_mesh', source)
        self.assertIn('collision.find("origin")', source)
        self.assertIn('euler_matrix(*rpy, axes="sxyz")', source)
        self.assertIn('source_frame": "urdf_link:sample_00"', source)
        self.assertIn('torch.isfinite(grasps)', source)
        self.assertIn('rotation_orthogonality_max_error', source)
        self.assertIn('raw_grasps.franka.v1.json', source)
        self.assertIn('evidence_level": "raw"', source)

    def test_segmenter_emits_compiler_native_glb_aligned_evidence(self):
        source = RUNTIME.read_text()
        self.assertIn('def _read_glb_document', source)
        self.assertIn('def _glb_accessor_array', source)
        self.assertIn('def _build_agentscape_segmentation_evidence', source)
        self.assertIn('verified-vertex-identity-triangle-index-set', source)
        self.assertIn('vertex identity drift exceeds tolerance', source)
        self.assertIn('source OBJ face is referenced more than once', source)
        self.assertIn('primary GLB does not cover all source OBJ faces', source)
        self.assertIn('agentscape_part_segmentation.v1.json', source)
        self.assertIn('"source": "embodiedgen/p3sam"', source)
        self.assertIn('"sourceNode": source_node', source)
        self.assertIn('"faceLabels": primitive_labels', source)


if __name__ == "__main__":
    unittest.main()
