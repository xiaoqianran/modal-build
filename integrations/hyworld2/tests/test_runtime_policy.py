import importlib.util
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
COMFY = ROOT / "comfyui_modal"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RuntimePolicyTest(unittest.TestCase):
    def test_default_gpu_is_rtx_pro_6000(self):
        old = os.environ.pop("MODAL_GPU", None)
        try:
            config = load_module("hyworld2_modal_config", COMFY / "modal_config.py")
        finally:
            if old is not None:
                os.environ["MODAL_GPU"] = old
        self.assertEqual(config.GPU, "RTX-PRO-6000")
        self.assertEqual(config.CUDA_ARCH, "12.0")
        self.assertNotIn("H100", config.GPU_ARCHITECTURES)

    def test_prompt_has_low_cost_smoke_controls(self):
        runner = load_module("hyworld2_world_runner", COMFY / "world_runner.py")
        prompt = runner.build_prompt(
            image_name="input.png",
            filename="smoke",
            target_size=252,
            use_gsplat=True,
        )
        inputs = prompt["3"]["inputs"]
        self.assertEqual(inputs["target_size"], 252)
        self.assertEqual(inputs["head_frame_chunk_size"], 1)
        self.assertEqual(inputs["head_compute_mode"], "depth+gs")
        self.assertEqual(prompt["4"]["class_type"], "VNCCS_SavePLY")

    def test_runtime_scales_to_zero(self):
        source = (COMFY / "modal_app.py").read_text(encoding="utf-8")
        self.assertIn("min_containers=0", source)
        self.assertIn("scaledown_window=SCALEDOWN_WINDOW_SECONDS", source)


if __name__ == "__main__":
    unittest.main()
