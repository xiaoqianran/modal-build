import importlib.util
import unittest
from pathlib import Path

RUNTIME = Path(__file__).parents[1] / "runtime" / "embodiedgen_v2_l40s.py"
spec = importlib.util.spec_from_file_location("embodiedgen_runtime", RUNTIME)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class AutoscalePolicyTest(unittest.TestCase):
    def test_auto_stays_min_cost_for_isolated_request(self):
        profile, count = runtime.auto_profile_for_timestamps([1000.0], 1000.0)
        self.assertEqual((profile, count), ("min_cost", 1))

    def test_auto_promotes_on_second_request_within_60_seconds(self):
        profile, count = runtime.auto_profile_for_timestamps([941.0, 1000.0], 1000.0)
        self.assertEqual((profile, count), ("cost_first", 2))

    def test_auto_does_not_promote_on_old_request(self):
        profile, count = runtime.auto_profile_for_timestamps([939.9, 1000.0], 1000.0)
        self.assertEqual((profile, count), ("min_cost", 1))

    def test_future_timestamps_are_ignored(self):
        profile, count = runtime.auto_profile_for_timestamps([1001.0, 1000.0], 1000.0)
        self.assertEqual((profile, count), ("min_cost", 1))

    def test_auto_never_promotes_to_latency_profiles(self):
        timestamps = [1000.0 - i for i in range(20)]
        profile, count = runtime.auto_profile_for_timestamps(timestamps, 1000.0)
        self.assertEqual(profile, "cost_first")
        self.assertEqual(count, 20)

    def test_every_profile_defines_every_stage(self):
        expected = {"rembg", "sam3d", "mesh", "lite", "finalize"}
        for profile in runtime.AUTOSCALE_PROFILES.values():
            self.assertEqual(set(profile), expected)

    def test_cost_first_tail_cost_is_stable(self):
        summary = runtime.autoscale_profile_summary("cost_first")
        self.assertAlmostEqual(summary["idle_tail_total_usd"], 0.03065400, places=8)
        self.assertEqual(summary["scaledown_window_seconds"]["sam3d"], 30)
        self.assertEqual(summary["scaledown_window_seconds"]["finalize"], 2)

    def test_text_profile_cost_includes_text2image_tail(self):
        summary = runtime.text_autoscale_profile_summary("min_cost")
        self.assertEqual(summary["text2image_scaledown_window_seconds"], 2)
        self.assertAlmostEqual(summary["text2image_idle_tail_usd"], 0.00133067, places=8)
        self.assertAlmostEqual(summary["text_to_3d_idle_tail_total_usd"], 0.00433439, places=8)

    def test_text_cost_first_tail_is_separate_from_image_cost(self):
        image = runtime.autoscale_profile_summary("cost_first")
        text = runtime.text_autoscale_profile_summary("cost_first")
        self.assertAlmostEqual(image["idle_tail_total_usd"], 0.03065400, places=8)
        self.assertAlmostEqual(text["text2image_idle_tail_usd"], 0.01996000, places=8)
        self.assertAlmostEqual(text["text_to_3d_idle_tail_total_usd"], 0.05061400, places=8)

    def test_retexture_min_cost_tail_is_isolated(self):
        summary = runtime.retexture_autoscale_profile_summary("min_cost")
        self.assertEqual(summary["scaledown_window_seconds"], 2)
        self.assertAlmostEqual(summary["idle_tail_cost_usd"], 0.00133067, places=8)

    def test_unknown_profile_fails_closed(self):
        with self.assertRaises(ValueError):
            runtime.autoscale_profile_summary("unknown")


if __name__ == "__main__":
    unittest.main()
