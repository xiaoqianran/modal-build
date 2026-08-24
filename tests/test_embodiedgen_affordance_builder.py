import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
BUILDER = ROOT / "modal_build" / "embodiedgen_affordance.py"
spec = importlib.util.spec_from_file_location("embodiedgen_affordance_builder", BUILDER)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class AffordanceBuilderTest(unittest.TestCase):
    def test_source_pins_are_exact(self):
        self.assertRegex(builder.PINS["embodiedgen"], r"^[0-9a-f]{40}$")
        self.assertRegex(builder.PINS["graspgen"], r"^[0-9a-f]{40}$")
        self.assertRegex(builder.PINS["hunyuan3d_part"], r"^[0-9a-f]{40}$")
        self.assertEqual(builder.PINS["torch_scatter"], "2.1.2")
        self.assertRegex(builder.PINS["torch_scatter_wheel_sha256"], r"^[0-9a-f]{64}$")

    def test_builder_is_cpu_only_and_sm89(self):
        source = BUILDER.read_text()
        build_start = source.index("@app.function", source.index("def wheel_shared_objects"))
        build_end = source.index("@app.function", build_start + 1)
        build_section = source[build_start:build_end]
        self.assertNotIn('gpu=', build_section)
        self.assertIn('TORCH_CUDA_ARCH_LIST', build_section)
        self.assertIn('nvidia/cuda:12.6.3-devel-ubuntu22.04', source)
        self.assertIn('"TORCH_CUDA_ARCH_LIST": "8.9"', source)

    def test_bundle_has_three_compiled_wheels(self):
        source = BUILDER.read_text()
        for token in ("TORCH_SCATTER_WHEEL_URL", "pointnet2_ops", "chamfer3D"):
            self.assertIn(token, source)
        self.assertIn("torch-scatter wheel hash mismatch", source)
        self.assertIn("expected 3 wheels", source)
        self.assertNotIn("flash-attn", source)
        self.assertIn("wheel has no compiled shared object", source)


    def test_p3sam_no_flash_patch_uses_upstream_fallback(self):
        patch = ROOT / "patches" / "embodiedgen-v2.0.0" / "production" / "p3sam-no-flash.patch"
        body = patch.read_text()
        self.assertIn('custom_config={"enable_flash": False}', body)
        self.assertIn('embodied_gen/utils/monkey_patch/p3sam.py', body)
        self.assertNotIn('diff --git a/P3-SAM/model.py', body)
        self.assertNotIn('flash-attn', BUILDER.read_text())

    def test_embodiedgen_pin_guards_submodule_pins(self):
        source = BUILDER.read_text()
        self.assertIn('clone_at("https://github.com/HorizonRobotics/EmbodiedGen.git"', source)
        self.assertIn('"thirdparty/GraspGen": PINS["graspgen"]', source)
        self.assertIn('"thirdparty/Hunyuan3D-Part": PINS["hunyuan3d_part"]', source)
        self.assertIn("submodule pin mismatch", source)
        self.assertIn('"--depth=1"', source)

    def test_l40s_validator_executes_all_native_extensions(self):
        source = BUILDER.read_text()
        self.assertIn('gpu="L40S"', source)
        self.assertIn('from torch_scatter import scatter', source)
        self.assertIn('import pointnet2_ops._ext as pointnet2_ext', source)
        self.assertIn('pointnet2_ext.furthest_point_sampling', source)
        self.assertIn('import chamfer_3D', source)
        self.assertIn('chamfer_3D.forward', source)
        self.assertIn('AFFORDANCE_L40S_SMOKE_OK', source)
        self.assertIn('staged wheel hash mismatch', source)

    def test_staging_is_immutable_and_preemption_safe(self):
        source = BUILDER.read_text()
        self.assertIn("refusing to overwrite", source)
        self.assertIn("preempt CPU builders", source)
        self.assertLess(source.index('if output.exists()'), source.index('GraspGen pointnet2 CUDA extension'))
        self.assertGreater(source.index('output.mkdir(parents=True, exist_ok=False)'), source.index('expected 3 wheels'))
        self.assertIn("artifacts.commit()", source)


if __name__ == "__main__":
    unittest.main()
