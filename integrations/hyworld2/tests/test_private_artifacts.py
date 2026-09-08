import hashlib
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from integrations.hyworld2 import private_artifacts


class PrivateArtifactValidationTest(unittest.TestCase):
    def _make_bundle(self, root: Path, *, public_release: bool = False, omit_notice: bool = False):
        spec = private_artifacts.RESTRICTED_BUNDLES[0]
        archive = root / f"{spec.tag}.wheels.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("wheels/gsplat-1.whl", b"gsplat")
            bundle.writestr("wheels/recast-1.whl", b"recast")
            bundle.writestr("LICENSES/Tencent-HY-WORLD-2.0-License.txt", b"license")
            bundle.writestr("LICENSES/GLM-copying.txt", b"glm")
            if not omit_notice:
                bundle.writestr("LICENSES/NOTICE.txt", b"notice")
        archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        manifest = {
            "tag": spec.tag,
            "bundle_kind": "hyworld2-restricted-native",
            "public_release": public_release,
            "python": "3.11",
            "cuda": "12.8.1",
            "torch": "2.7.1",
            "torchvision": "0.22.1",
            "cuda_arch": "9.0",
            "target_gpu": "H100",
            "source": "Tencent-Hunyuan/HY-World-2.0",
            "source_revision": private_artifacts.HYWORLD2_REVISION,
            "archive_sha256": archive_sha,
            "wheels": [{"file": "gsplat-1.whl"}, {"file": "recast-1.whl"}],
            "smoke": ["gpu-sm90", "gsplat-distloss-gauss_masks-cuda", "recast-import"],
        }
        (root / f"{spec.tag}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / f"{spec.tag}.wheels.zip.sha256").write_text(
            f"{archive_sha}  {archive.name}\n", encoding="utf-8"
        )
        return spec

    def test_valid_restricted_bundle_passes_strict_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self._make_bundle(root)
            manifest = private_artifacts.validate_bundle(root, spec)
        self.assertFalse(manifest["public_release"])

    def test_public_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self._make_bundle(root, public_release=True)
            with self.assertRaises(private_artifacts.ArtifactError):
                private_artifacts.validate_bundle(root, spec)

    def test_license_payload_is_required(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self._make_bundle(root, omit_notice=True)
            with self.assertRaisesRegex(private_artifacts.ArtifactError, "license payload missing"):
                private_artifacts.validate_bundle(root, spec)

    def test_private_repo_check_fails_closed_for_public_repo(self):
        result = subprocess.CompletedProcess([], 0, stdout='{"visibility":"PUBLIC"}', stderr="")
        with (
            mock.patch.object(private_artifacts, "_gh", return_value=result),
            self.assertRaisesRegex(private_artifacts.ArtifactError, "visibility='PUBLIC'"),
        ):
            private_artifacts._assert_private_repository("owner/public")

    def test_default_ensure_never_compiles_when_both_caches_are_missing(self):
        spec = private_artifacts.RESTRICTED_BUNDLES[0]
        failed = subprocess.CalledProcessError(1, ["missing"])
        with (
            mock.patch.object(private_artifacts, "verify_volume", side_effect=failed),
            mock.patch.object(private_artifacts, "restore", side_effect=failed),
            mock.patch.object(private_artifacts, "_compile") as compile_mock,
            self.assertRaisesRegex(private_artifacts.ArtifactError, "--compile-if-missing"),
        ):
            private_artifacts.ensure(
                spec,
                repository="owner/private",
                volume="artifacts",
                compile_if_missing=False,
            )
        compile_mock.assert_not_called()

    def test_compile_fallback_requires_explicit_opt_in(self):
        spec = private_artifacts.RESTRICTED_BUNDLES[0]
        failed = subprocess.CalledProcessError(1, ["missing"])
        with (
            mock.patch.object(private_artifacts, "verify_volume", side_effect=failed),
            mock.patch.object(private_artifacts, "restore", side_effect=failed),
            mock.patch.object(private_artifacts, "_compile") as compile_mock,
            mock.patch.object(
                private_artifacts,
                "backup",
                return_value={"status": "backed-up", "tag": spec.tag},
            ),
        ):
            result = private_artifacts.ensure(
                spec,
                repository="owner/private",
                volume="artifacts",
                compile_if_missing=True,
            )
        compile_mock.assert_called_once_with(spec)
        self.assertEqual(result["status"], "compiled-and-backed-up")

    def test_all_selects_only_known_restricted_architectures(self):
        specs = private_artifacts._select_specs([], True)
        self.assertEqual({spec.cuda_arch for spec in specs}, {"9.0", "12.0"})


if __name__ == "__main__":
    unittest.main()
