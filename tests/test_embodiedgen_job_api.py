import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUNTIME = ROOT / "runtime" / "embodiedgen_v2_l40s.py"
spec = importlib.util.spec_from_file_location("embodiedgen_job_api", RUNTIME)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class JobIdentityTest(unittest.TestCase):
    def test_generated_ids_are_unique_and_safe(self):
        first = runtime.new_job_id()
        second = runtime.new_job_id()
        self.assertNotEqual(first, second)
        self.assertTrue(runtime.is_api_job_id(first))
        self.assertTrue(runtime.is_api_job_id(second))

    def test_api_root_rejects_path_escape_and_legacy_ids(self):
        for value in ("../x", "/tmp/x", "bench-20260823T121021-warm", "job-nothex"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime.api_job_root(value)

    def test_result_file_map_contains_only_relative_safe_paths(self):
        for value in runtime.ALL_RESULT_FILES.values():
            path = Path(value)
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)



class ArtifactLifecycleTest(unittest.TestCase):
    def test_prune_keeps_only_result_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "result").mkdir()
            (root / "result" / "model.glb").write_bytes(b"glb")
            (root / "validation_report.json").write_text("{}")
            (root / "state.pkl").write_bytes(b"state")
            (root / "nested").mkdir()
            (root / "nested" / "temp").write_text("x")

            removed = runtime.prune_job_intermediates(root)

            self.assertEqual(removed, ["nested", "state.pkl"])
            self.assertTrue((root / "result" / "model.glb").exists())
            self.assertTrue((root / "validation_report.json").exists())

    def test_recent_running_job_does_not_expire(self):
        now = 2_000_000.0
        self.assertFalse(
            runtime.api_job_expired(
                {"status": "running", "updated_epoch": now - runtime.API_ACTIVE_STALE_SECONDS + 1},
                now=now,
                fallback_mtime=0,
            )
        )

    def test_stuck_running_job_expires(self):
        now = 2_000_000.0
        self.assertTrue(
            runtime.api_job_expired(
                {"status": "running", "updated_epoch": now - runtime.API_ACTIVE_STALE_SECONDS - 1},
                now=now,
                fallback_mtime=0,
            )
        )

    def test_failed_and_success_ttls_are_different(self):
        now = 2_000_000.0
        failed = {"status": "failed", "updated_epoch": now - runtime.API_FAILED_TTL_SECONDS - 1}
        succeeded = {
            "status": "succeeded",
            "updated_epoch": now - runtime.API_FAILED_TTL_SECONDS - 1,
        }
        self.assertTrue(runtime.api_job_expired(failed, now=now, fallback_mtime=now))
        self.assertFalse(runtime.api_job_expired(succeeded, now=now, fallback_mtime=now))


class InputResolutionTest(unittest.TestCase):
    def test_api_job_requires_uploaded_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_id = runtime.new_job_id()
            with self.assertRaises(FileNotFoundError):
                runtime.resolve_job_input(job_id, Path(tmp), Path("fallback.jpg"))

    def test_legacy_job_can_use_benchmark_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            fallback = Path(tmp) / "fallback.jpg"
            path, source = runtime.resolve_job_input("bench-safe", Path(tmp), fallback)
            self.assertEqual(path, fallback)
            self.assertEqual(source, "benchmark-sample")

    def test_api_job_uses_uploaded_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploaded = root / "input_image"
            uploaded.write_bytes(b"image")
            path, source = runtime.resolve_job_input(runtime.new_job_id(), root, Path("fallback.jpg"))
            self.assertEqual(path, uploaded)
            self.assertEqual(source, "uploaded")



class TextJobTest(unittest.TestCase):
    def test_prompt_validation(self):
        self.assertEqual(runtime.normalize_text_prompt("  red mug  "), "red mug")
        with self.assertRaises(TypeError):
            runtime.normalize_text_prompt(None)
        for value in ("", "   ", "x" * (runtime.MAX_PROMPT_CHARS + 1)):
            with self.subTest(value=value[:16]), self.assertRaises(ValueError):
                runtime.normalize_text_prompt(value)

    def test_generated_text_input_is_labeled_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "input_image").write_bytes(b"png")
            (root / "prompt.txt").write_text("red mug\n")
            path, source = runtime.resolve_job_input(runtime.new_job_id(), root, Path("fallback.jpg"))
            self.assertEqual(path, root / "input_image")
            self.assertEqual(source, "generated-text")

    def test_text_worker_is_l40s_and_offline(self):
        source = RUNTIME.read_text(encoding="utf-8")
        pos = source.index("class Text2ImageWorker:")
        decorator = source[source.rfind("@app.cls(", 0, pos):pos]
        body_end = source.index("def _rembg_load", pos)
        body = source[pos:body_end]
        self.assertIn('image=image', decorator)
        self.assertIn('gpu="L40S"', decorator)
        self.assertIn('local_files_only=True', body)
        self.assertIn('HF_HUB_OFFLINE', body)
        self.assertIn('TRANSFORMERS_OFFLINE', body)

    def test_text_stage_runs_before_rembg(self):
        source = RUNTIME.read_text(encoding="utf-8")
        start = source.index("def run_job(")
        end = source.index("@app.function(", start)
        body = source[start:end]
        self.assertLess(body.index('"text2image"'), body.index('"rembg"'))

    def test_text_submit_uses_async_modal_interfaces(self):
        source = RUNTIME.read_text(encoding="utf-8")
        start = source.index("    async def submit_text_job(")
        end = source.index('    @web.get("/jobs/{job_id}")', start)
        submit = source[start:end]
        for blocking in ("artifacts.commit()", "job_states.put(", "run_job.spawn("):
            self.assertNotIn(blocking, submit)
        for async_call in ("artifacts.commit.aio()", "job_states.put.aio(", "run_job.spawn.aio("):
            self.assertIn(async_call, submit)


class AutoscaleDedupeTest(unittest.TestCase):
    class FakeTarget:
        def __init__(self):
            self.calls = []

        def update_autoscaler(self, **kwargs):
            self.calls.append(kwargs)

    def setUp(self):
        runtime._active_autoscale_profile = None

    def handles(self):
        return tuple(self.FakeTarget() for _ in range(5))

    def test_same_profile_only_updates_once_per_process(self):
        handles = self.handles()
        runtime.apply_autoscale_profile("cost_first", handles)
        runtime.apply_autoscale_profile("cost_first", handles)
        self.assertEqual([len(target.calls) for target in handles], [1, 1, 1, 1, 1])

    def test_profile_change_updates_all_stages(self):
        handles = self.handles()
        runtime.apply_autoscale_profile("min_cost", handles)
        runtime.apply_autoscale_profile("cost_first", handles)
        self.assertEqual([len(target.calls) for target in handles], [2, 2, 2, 2, 2])
        self.assertEqual(runtime._active_autoscale_profile, "cost_first")


class RetextureJobTest(unittest.TestCase):
    def test_retexture_worker_is_single_l40s_offline_pipeline(self):
        source = RUNTIME.read_text(encoding="utf-8")
        pos = source.index("class RetextureWorker:")
        decorator = source[source.rfind("@app.cls(", 0, pos):pos]
        end = source.index("def _rembg_load", pos)
        body = source[pos:end]
        self.assertIn('gpu="L40S"', decorator)
        self.assertIn('local_files_only=True', body)
        self.assertIn('HF_HUB_OFFLINE', body)
        self.assertIn('delight=False', body)
        self.assertIn('ip_adapt_scale=0.0', body)

    def test_retexture_validates_geometry_preservation(self):
        source = RUNTIME.read_text(encoding="utf-8")
        pos = source.index("class RetextureWorker:")
        end = source.index("def _rembg_load", pos)
        body = source[pos:end]
        self.assertIn('source_geometry_preserved', body)
        self.assertIn('np.allclose(src_mesh.bounds,objm.bounds', body)
        self.assertIn('len(src_mesh.faces)==len(objm.faces)', body)

    def test_retexture_endpoint_requires_succeeded_source_and_async_spawn(self):
        source = RUNTIME.read_text(encoding="utf-8")
        start = source.index('    async def submit_retexture_job(')
        end = source.index('    @web.get("/jobs/{job_id}")', start)
        body = source[start:end]
        self.assertIn('source_state.get("status") != "succeeded"', body)
        self.assertIn('await job_states.get.aio(', body)
        self.assertIn('await job_states.put.aio(', body)
        self.assertIn('await run_retexture_job.spawn.aio(', body)
        self.assertNotIn('run_retexture_job.spawn(', body)


class AffordanceJobTest(unittest.TestCase):
    def test_affordance_options_are_strict_and_defaulted(self):
        options = runtime.normalize_affordance_options({})
        self.assertEqual(options["profile"], "part-evidence-only")
        self.assertEqual(options["point_num"], 20000)
        self.assertEqual(options["topk"], 20)
        semantic = runtime.normalize_affordance_options({"profile": "semantic-evidence-v1", "category": " mug "})
        self.assertEqual(semantic["profile"], "semantic-evidence-v1")
        self.assertEqual(semantic["category"], "mug")
        with self.assertRaises(ValueError):
            runtime.normalize_affordance_options({"profile": "full"})
        with self.assertRaises(ValueError):
            runtime.normalize_affordance_options({"category": "mug"})
        with self.assertRaises(ValueError):
            runtime.normalize_affordance_options({"topk": 81, "num_grasps": 80})
        with self.assertRaises(TypeError):
            runtime.normalize_affordance_options({"seed": True})
        with self.assertRaises(ValueError):
            runtime.normalize_affordance_options({"semantic": True})

    def test_affordance_orchestrator_uses_separate_deployed_app_and_stage_order(self):
        source = RUNTIME.read_text(encoding="utf-8")
        self.assertIn('AFFORDANCE_APP_NAME = "modal-3d-embodiedgen-affordance"', source)
        self.assertIn('modal.Function.from_name(AFFORDANCE_APP_NAME, "segment_job")', source)
        self.assertIn('modal.Function.from_name(AFFORDANCE_APP_NAME, "raw_grasp_job")', source)
        start = source.index("def run_affordance_job(")
        end = source.index("@app.function(", start)
        body = source[start:end]
        self.assertLess(body.index('"segment"'), body.index('"grasp_raw"'))
        self.assertLess(body.index('"grasp_raw"'), body.index('"finalize"'))
        self.assertIn('AFFORDANCE_SEMANTIC_PROFILE', body)
        self.assertIn('"semantic_inputs"', body)
        self.assertIn('"semantic_annotate"', body)
        self.assertLess(body.index('"semantic_inputs"'), body.index('"semantic_annotate"'))
        self.assertIn('output_job_id=job_id', body)
        self.assertIn('files=sorted(affordance_result_files(profile))', body)

    def test_affordance_finalize_publishes_hash_bound_bundle(self):
        source = RUNTIME.read_text(encoding="utf-8")
        start = source.index("def finalize_affordance_bundle(")
        end = source.index("@app.function(", start)
        body = source[start:end]
        self.assertIn('"version": 1', body)
        self.assertIn('"provider": "embodiedgen"', body)
        self.assertIn('"role": role', body)
        self.assertIn('segmentation.get("artifact", {}).get("sha256") != primary_sha256', body)
        self.assertIn('raw_grasps.get("evidence_level") != "raw"', body)
        self.assertIn('AFFORDANCE_PART_EVIDENCE_BUNDLE_OK', body)
        self.assertIn('AFFORDANCE_SEMANTIC_EVIDENCE_BUNDLE_OK', body)
        self.assertIn('"part_semantics"', body)
        self.assertIn('part semantics IDs do not match segmentation IDs', body)
        self.assertIn('part semantics validation SHA mismatch', body)
        self.assertIn('forbidden executable fields', body)
        self.assertNotIn('sapien_grasps', body)

    def test_affordance_result_files_are_profile_scoped(self):
        base = runtime.affordance_result_files("part-evidence-only")
        semantic = runtime.affordance_result_files("semantic-evidence-v1")
        self.assertNotIn("part_semantics", base)
        self.assertIn("part_semantics", semantic)
        self.assertIn("semantic_inputs", semantic)
        self.assertEqual(base["affordance_bundle"], semantic["affordance_bundle"])
        with self.assertRaises(ValueError):
            runtime.affordance_result_files("unknown")

    def test_affordance_endpoint_requires_succeeded_source_and_async_dispatch(self):
        source = RUNTIME.read_text(encoding="utf-8")
        start = source.index('    async def submit_affordance_job(')
        end = source.index('    @web.get("/jobs/{job_id}")', start)
        body = source[start:end]
        self.assertIn('source_state.get("status") != "succeeded"', body)
        self.assertIn('await artifacts.reload.aio()', body)
        self.assertIn('await artifacts.commit.aio()', body)
        self.assertIn('await job_states.put.aio(', body)
        self.assertIn('await run_affordance_job.spawn.aio(', body)
        self.assertNotIn('run_affordance_job.spawn(', body)

    def test_job_file_urls_are_scoped_to_state_file_roles(self):
        source = RUNTIME.read_text(encoding="utf-8")
        self.assertIn('available = state.get("files") or sorted(RESULT_FILES)', source)
        self.assertIn('if name not in available:', source)
        self.assertIn('ALL_RESULT_FILES[name]', source)


class AffordanceSemanticInputTest(unittest.TestCase):
    def test_semantic_parts_are_bound_to_persisted_provider_palette(self):
        parts = runtime.semantic_parts_from_segmentation(
            {
                "palette": [
                    {"id": "0", "name": "Red", "rgb": [230, 25, 75]},
                    {"id": "1", "name": "Green", "rgb": [60, 180, 75]},
                ]
            },
            {"segments": [{"id": "1", "faceCount": 2}, {"id": "0", "faceCount": 3}]},
        )
        self.assertEqual(parts, [
            {"id": "1", "maskColor": "Green", "maskRgb": [60, 180, 75]},
            {"id": "0", "maskColor": "Red", "maskRgb": [230, 25, 75]},
        ])
        with self.assertRaises(ValueError):
            runtime.semantic_parts_from_segmentation(
                {"palette": [{"id": "0", "name": "Red", "rgb": [230, 25, 75]}]},
                {"segments": [{"id": "1", "faceCount": 2}]},
            )

    def test_semantic_input_renderer_is_isolated_from_gpt_and_hash_binds_outputs(self):
        source = RUNTIME.read_text(encoding="utf-8")
        start = source.index("def prepare_affordance_semantic_inputs(")
        end = source.index("def affordance_runtime_handles", start)
        body = source[start:end]
        self.assertIn('gpu="L40S"', source[source.rfind("@app.function", 0, start):start])
        self.assertIn('from embodied_gen.utils.vis_utils import render_grid', body)
        self.assertIn('output_subdir="rgb_views"', body)
        self.assertIn('staging / "mask_views"', body)
        self.assertIn('render_semantic_face_label_grid(', body)
        self.assertIn('compiler_segmentation,', body)
        self.assertIn('semantic_grid_diagnostics(mask_final)', body)
        self.assertIn('semantic_mask_palette_visibility(mask_final, parts)', body)
        self.assertIn('render_semantic_part_atlas(', body)
        self.assertIn('"partAtlas": {', body)
        self.assertIn('"renderer": "embodiedgen-rgb+nvdiffrast-face-id"', body)
        self.assertIn('semantic_mask_palette_visibility(mask_final, parts)', body)
        self.assertIn('"sha256": _sha256_file(rgb_final)', body)
        self.assertIn('"sha256": _sha256_file(mask_final)', body)
        self.assertIn('semantic_parts_from_segmentation', body)
        self.assertIn('AFFORDANCE_SEMANTIC_INPUTS_OK', body)
        self.assertNotIn('API_KEY', body)
        self.assertNotIn('MODEL_NAME', body)
        self.assertNotIn('openai', body.lower())

    def test_semantic_mask_uses_triangle_id_raster_not_materials(self):
        source = RUNTIME.read_text(encoding="utf-8")
        start = source.index("def render_semantic_face_label_grid(")
        end = source.index("def new_job_id", start)
        body = source[start:end]
        self.assertIn('rast[..., 3].long() - 1', body)
        self.assertIn('image[valid] = colors[ids[valid]]', body)
        self.assertIn('CameraSetting(', body)
        self.assertIn('elevation=(45.0, -45.0)', body)
        self.assertIn('distance=5.0', body)
        self.assertIn('fov=math.radians(30.0)', body)
        self.assertIn('render_semantic_face_label_grid', source)
        self.assertNotIn('map_Kd', body)
        self.assertNotIn('mask_obj', body)

    def test_semantic_part_atlas_isolated_views_cover_hidden_parts(self):
        source = RUNTIME.read_text(encoding="utf-8")
        start = source.index("def render_semantic_part_atlas(")
        end = source.index("def new_job_id", start)
        body = source[start:end]
        self.assertIn('views_per_part: int = 3', body)
        self.assertIn('torch.unique(', body)
        self.assertIn('normalize_vertices_array(part_vertices)', body)
        self.assertIn('semantic part atlas cannot render part', body)
        self.assertIn("part {part_id}  {part['maskColor']}", body)

    def test_semantic_category_is_bounded(self):
        self.assertEqual(runtime.normalize_semantic_category(" mug "), "mug")
        with self.assertRaises(ValueError):
            runtime.normalize_semantic_category("")
        with self.assertRaises(ValueError):
            runtime.normalize_semantic_category("x" * 161)


class AsyncControlPlaneTest(unittest.IsolatedAsyncioTestCase):
    class AsyncCall:
        def __init__(self, fn):
            self.aio = fn

    class AsyncItems:
        def __init__(self, data):
            self.data = data

        def aio(self):
            async def iterate():
                for item in list(self.data.items()):
                    yield item
            return iterate()

    class FakeTraffic:
        def __init__(self):
            self.data = {}
            self.put = AsyncControlPlaneTest.AsyncCall(self._put)
            self.pop = AsyncControlPlaneTest.AsyncCall(self._pop)
            self.items = AsyncControlPlaneTest.AsyncItems(self.data)

        async def _put(self, key, value):
            self.data[key] = value
            return True

        async def _pop(self, key, default=None):
            return self.data.pop(key, default)

    class FakeAutoscalerCall:
        def __init__(self, owner):
            self.owner = owner
            self.aio = self._aio

        async def _aio(self, **kwargs):
            self.owner.calls.append(kwargs)
            return kwargs

    class FakeTarget:
        def __init__(self):
            self.calls = []
            self.update_autoscaler = AsyncControlPlaneTest.FakeAutoscalerCall(self)

    async def test_async_profile_selection_uses_only_aio_dict_methods(self):
        original = runtime.traffic_events
        fake = self.FakeTraffic()
        runtime.traffic_events = fake
        try:
            first = await runtime.select_request_profile_aio("auto", now=1000.0)
            second = await runtime.select_request_profile_aio("auto", now=1001.0)
        finally:
            runtime.traffic_events = original
        self.assertEqual(first["selected_profile"], "min_cost")
        self.assertEqual(second["selected_profile"], "cost_first")
        self.assertEqual(second["recent_requests_60s"], 2)

    async def test_async_autoscale_updates_all_stages_via_aio(self):
        runtime._active_autoscale_profile = None
        handles = tuple(self.FakeTarget() for _ in range(5))
        await runtime.apply_autoscale_profile_aio("min_cost", handles)
        self.assertEqual([len(target.calls) for target in handles], [1, 1, 1, 1, 1])
        self.assertTrue(all(target.calls[0]["scaledown_window"] == 2 for target in handles))


class RuntimeIsolationTest(unittest.TestCase):
    def test_cpu_workers_use_lightweight_cpu_image(self):
        source = RUNTIME.read_text(encoding="utf-8")
        for marker in ("class RembgWorker:", "class MeshWorker:", "def cpu_finalize("):
            pos = source.index(marker)
            decorator = source[source.rfind("@app.", 0, pos):pos]
            self.assertIn("image=cpu_image", decorator, marker)
        self.assertIn("image=image,\n    gpu=\"L40S\"", source)

    def test_async_submit_uses_only_modal_aio_interfaces(self):
        source = RUNTIME.read_text(encoding="utf-8")
        start = source.index("    async def submit_job(")
        end = source.index("    @web.get(\"/jobs/{job_id}\")", start)
        submit = source[start:end]
        for blocking in ("artifacts.commit()", "job_states.put(", "run_job.spawn("):
            self.assertNotIn(blocking, submit)
        for async_call in ("artifacts.commit.aio()", "job_states.put.aio(", "run_job.spawn.aio("):
            self.assertIn(async_call, submit)

    def test_benchmark_fallback_is_preloaded_not_source_checkout(self):
        source = RUNTIME.read_text(encoding="utf-8")
        self.assertIn('/weights/examples/sample_00.jpg', source)
        rembg_start = source.index("def _rembg_load")
        rembg_end = source.index("def _rembg_prepare", rembg_start)
        self.assertNotIn('/workspace/EmbodiedGen', source[rembg_start:rembg_end])


if __name__ == "__main__":
    unittest.main()
