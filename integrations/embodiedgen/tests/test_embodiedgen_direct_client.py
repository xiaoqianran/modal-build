import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUNTIME = ROOT / "runtime" / "embodiedgen_v2_l40s.py"
CLIENT = ROOT / "runtime" / "embodiedgen_direct_client.py"


class DirectControlPlaneContractTest(unittest.TestCase):
    def test_modal_runtime_has_no_request_gateway_or_cpu_orchestrator(self):
        source = RUNTIME.read_text(encoding="utf-8")
        for forbidden in (
            "@modal.asgi_app",
            "def job_api(",
            "def run_job(",
            "def run_retexture_job(",
            "def run_affordance_job(",
        ):
            self.assertNotIn(forbidden, source)

    def test_direct_client_is_not_a_modal_app(self):
        source = CLIENT.read_text(encoding="utf-8")
        self.assertNotIn("modal.App(", source)
        self.assertNotIn("@app.function", source)
        self.assertNotIn("@modal.asgi_app", source)
        self.assertIn("modal.Cls.from_name", source)
        self.assertIn("modal.Function.from_name", source)
        self.assertIn("self.artifacts.batch_upload", source)

    def test_core_pipeline_is_orchestrated_locally_in_stage_order(self):
        source = CLIENT.read_text(encoding="utf-8")
        start = source.index("    def _run_core(")
        end = source.index("    def run_image(", start)
        body = source[start:end]
        order = [
            '"text2image"',
            '"rembg"',
            '"sam3d"',
            '"mesh"',
            '"texture"',
            '"finalize"',
        ]
        positions = [body.index(marker) for marker in order]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("text.generate.remote", body)
        self.assertIn("rembg.prepare.remote", body)
        self.assertIn("sam3d.generate.remote", body)
        self.assertIn("mesh.process.remote", body)
        self.assertIn("lite.remote", body)
        self.assertIn("finalize.remote", body)

    def test_retexture_goes_directly_to_gpu_worker(self):
        source = CLIENT.read_text(encoding="utf-8")
        start = source.index("    def run_retexture(")
        end = source.index("    def run_affordance(", start)
        body = source[start:end]
        self.assertIn('self._cls("RetextureWorker")', body)
        self.assertIn("worker.generate.remote", body)
        self.assertNotIn("run_retexture_job", body)

    def test_affordance_orchestration_is_local_but_compute_stages_remain_remote(self):
        source = CLIENT.read_text(encoding="utf-8")
        start = source.index("    def run_affordance(")
        end = source.index("    def get_job(", start)
        body = source[start:end]
        self.assertIn('self._fn("segment_job", AFFORDANCE_APP_NAME)', body)
        self.assertIn('self._fn("raw_grasp_job", AFFORDANCE_APP_NAME)', body)
        self.assertIn('self._fn("prepare_affordance_semantic_inputs")', body)
        self.assertIn('self._fn("annotate_semantics", AFFORDANCE_SEMANTIC_APP_NAME)', body)
        self.assertIn('self._fn("finalize_affordance_bundle")', body)
        self.assertNotIn("run_affordance_job", body)

    def test_result_download_reads_volume_directly(self):
        source = CLIENT.read_text(encoding="utf-8")
        start = source.index("    def download(")
        body = source[start:]
        self.assertIn("self.artifacts.read_file(remote)", body)
        self.assertIn("role not in available", body)


if __name__ == "__main__":
    unittest.main()
