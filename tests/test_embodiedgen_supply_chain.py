import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUNTIME = ROOT / "runtime" / "embodiedgen_v2_l40s.py"
BUILDER = ROOT / "modal_build" / "embodiedgen.py"

spec = importlib.util.spec_from_file_location("embodiedgen_supply_runtime", RUNTIME)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimePinsTest(unittest.TestCase):
    def test_git_sources_are_exact_commits(self):
        for value in (runtime.EMBODIEDGEN_COMMIT, runtime.CLIP_COMMIT, runtime.KOLORS_COMMIT):
            self.assertRegex(value, r"^[0-9a-f]{40}$")

    def test_release_hashes_are_sha256(self):
        self.assertEqual(
            runtime.RELEASE_WHEELS_SHA256,
            "4168abccbc9a0033825e3ad8b9a9e992795f6449107adf357a4dd4acafec398c",
        )
        self.assertEqual(
            runtime.RELEASE_EXTENSIONS_SHA256,
            "e5e1991ec465b399d46bca271af46394b054afd9eefdbcdcd8b5329f4c8e5bb3",
        )

    def test_runtime_verifies_release_archives_before_unzip(self):
        source = RUNTIME.read_text()
        self.assertIn("sha256sum -c -", source)
        verify_pos = source.index("sha256sum -c -")
        unzip_pos = source.index("unzip -q /tmp/wheels.zip")
        self.assertLess(verify_pos, unzip_pos)

    def test_all_direct_git_dependencies_are_pinned(self):
        source = RUNTIME.read_text()
        urls = re.findall(r"git\+https://github\.com/[^'\"]+", source)
        self.assertTrue(urls)
        for url in urls:
            self.assertIn("@", url.split("git+https://", 1)[1], url)


class ImmutableReleaseTest(unittest.TestCase):
    def test_builder_never_clobbers_release_assets(self):
        source = BUILDER.read_text()
        self.assertNotIn("--clobber", source)
        self.assertIn("refusing to overwrite immutable artifacts", source)
        self.assertIn("Bump TAG for a new release", source)


if __name__ == "__main__":
    unittest.main()
