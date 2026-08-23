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
        for value in runtime.RESULT_FILES.values():
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


if __name__ == "__main__":
    unittest.main()
