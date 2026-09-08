from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PRIVATE_REPOSITORY = "xiaoqianran/modal-build-private"
DEFAULT_ARTIFACT_VOLUME = "modal-build-artifacts"
ARTIFACT_SUFFIXES = ("wheels.zip", "wheels.zip.sha256", "manifest.json")
HYWORLD2_REVISION = "df9988efb87bfc0f4947eb3889411cf957478b06"


@dataclass(frozen=True)
class RestrictedBundleSpec:
    tag: str
    cuda_arch: str
    target_gpu: str
    builder: str


RESTRICTED_BUNDLES = (
    RestrictedBundleSpec(
        tag="hyworld2-hy-native-py311-cu128-torch271-sm90-v1",
        cuda_arch="9.0",
        target_gpu="H100",
        builder="integrations/hyworld2/build/hyworld2_hy_native_sm90.py::build",
    ),
    RestrictedBundleSpec(
        tag="hyworld2-hy-native-py311-cu128-torch271-sm120-v1",
        cuda_arch="12.0",
        target_gpu="RTX-PRO-6000",
        builder="integrations/hyworld2/build/hyworld2_hy_native_sm120.py::build",
    ),
)
SPECS_BY_TAG = {spec.tag: spec for spec in RESTRICTED_BUNDLES}


class ArtifactError(RuntimeError):
    pass


def _tool(env_name: str, default: str) -> list[str]:
    return shlex.split(os.environ.get(env_name, default))


def _run(
    command: list[str],
    *,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture,
    )


def _gh_command() -> list[str]:
    configured = os.environ.get("HYWORLD2_GH_CMD")
    if configured:
        return shlex.split(configured)
    if shutil.which("gh"):
        return ["gh"]
    if os.name != "nt":
        windows_gh = Path("/mnt/c/Program Files/GitHub CLI/gh.exe")
        if windows_gh.is_file():
            return [str(windows_gh)]
    return ["gh"]


def _gh(*args: str, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run([*_gh_command(), *args], capture=capture, check=check)


def _gh_fs_path(path: Path) -> str:
    """Translate WSL paths when the authenticated GitHub CLI is a Windows executable."""
    command = _gh_command()
    if os.name != "nt" and command and command[0].lower().endswith(".exe"):
        result = _run(["wslpath", "-w", str(path)], capture=True)
        return result.stdout.strip()
    return str(path)


def _modal(
    *args: str, capture: bool = False, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return _run([*_tool("HYWORLD2_MODAL_CMD", "modal"), *args], capture=capture, check=check)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(root: Path, tag: str) -> dict[str, Path]:
    return {suffix: root / f"{tag}.{suffix}" for suffix in ARTIFACT_SUFFIXES}


def _parse_sidecar(path: Path) -> tuple[str, str]:
    parts = path.read_text(encoding="utf-8").strip().split(maxsplit=1)
    if len(parts) != 2 or len(parts[0]) != 64:
        raise ArtifactError(f"invalid SHA256 sidecar: {path}")
    return parts[0].lower(), parts[1].strip()


def validate_bundle(root: Path, spec: RestrictedBundleSpec) -> dict[str, object]:
    paths = _paths(root, spec.tag)
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise ArtifactError(f"missing artifact files for {spec.tag}: {missing}")

    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    expected_fields = {
        "tag": spec.tag,
        "bundle_kind": "hyworld2-restricted-native",
        "public_release": False,
        "python": "3.11",
        "cuda": "12.8.1",
        "torch": "2.7.1",
        "torchvision": "0.22.1",
        "cuda_arch": spec.cuda_arch,
        "target_gpu": spec.target_gpu,
        "source": "Tencent-Hunyuan/HY-World-2.0",
        "source_revision": HYWORLD2_REVISION,
    }
    for key, expected in expected_fields.items():
        if manifest.get(key) != expected:
            raise ArtifactError(
                f"manifest mismatch for {spec.tag}: {key}={manifest.get(key)!r}, expected {expected!r}"
            )

    archive = paths["wheels.zip"]
    actual_sha = _sha256(archive)
    manifest_sha = str(manifest.get("archive_sha256", "")).lower()
    sidecar_sha, sidecar_name = _parse_sidecar(paths["wheels.zip.sha256"])
    if manifest_sha != actual_sha:
        raise ArtifactError(f"manifest SHA256 mismatch for {spec.tag}: {manifest_sha} != {actual_sha}")
    if sidecar_sha != actual_sha or sidecar_name != archive.name:
        raise ArtifactError(f"sidecar SHA256 mismatch for {spec.tag}")

    wheels = manifest.get("wheels")
    if not isinstance(wheels, list) or len(wheels) != 2:
        raise ArtifactError(f"expected exactly two restricted wheels for {spec.tag}")
    smoke = set(manifest.get("smoke") or [])
    expected_smoke = {
        f"gpu-sm{spec.cuda_arch.replace('.', '')}",
        "gsplat-distloss-gauss_masks-cuda",
        "recast-import",
    }
    if not expected_smoke.issubset(smoke):
        raise ArtifactError(f"restricted CUDA smoke record incomplete for {spec.tag}: {sorted(smoke)}")

    with zipfile.ZipFile(archive) as bundle:
        corrupt = bundle.testzip()
        if corrupt is not None:
            raise ArtifactError(f"corrupt zip member in {spec.tag}: {corrupt}")
        names = set(bundle.namelist())
        required_license_suffixes = {
            "LICENSES/Tencent-HY-WORLD-2.0-License.txt",
            "LICENSES/NOTICE.txt",
            "LICENSES/GLM-copying.txt",
        }
        missing_licenses = sorted(required_license_suffixes - names)
        if missing_licenses:
            raise ArtifactError(f"license payload missing for {spec.tag}: {missing_licenses}")
        wheel_entries = [name for name in names if name.startswith("wheels/") and name.endswith(".whl")]
        if len(wheel_entries) != 2:
            raise ArtifactError(f"expected two wheel files inside {spec.tag}, got {wheel_entries}")

    return manifest


def _assert_private_repository(repository: str) -> None:
    result = _gh("repo", "view", repository, "--json", "visibility", capture=True)
    payload = json.loads(result.stdout)
    if payload.get("visibility") != "PRIVATE":
        raise ArtifactError(
            f"refusing restricted artifact operation: {repository} visibility={payload.get('visibility')!r}"
        )


def _download_volume(tag: str, destination: Path, volume: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for suffix, local_path in _paths(destination, tag).items():
        _modal("volume", "get", volume, f"{tag}.{suffix}", str(local_path), "--force")


def _upload_volume(tag: str, source: Path, volume: str) -> None:
    for suffix, local_path in _paths(source, tag).items():
        _modal("volume", "put", "-f", volume, str(local_path), f"{tag}.{suffix}")


def _download_release(tag: str, destination: Path, repository: str) -> None:
    _assert_private_repository(repository)
    destination.mkdir(parents=True, exist_ok=True)
    _gh(
        "release",
        "download",
        tag,
        "--repo",
        repository,
        "--dir",
        _gh_fs_path(destination),
        "--pattern",
        f"{tag}.*",
        "--clobber",
    )


def _release_exists(tag: str, repository: str) -> bool:
    result = _gh("release", "view", tag, "--repo", repository, capture=True, check=False)
    return result.returncode == 0


def _publish_release(tag: str, source: Path, repository: str) -> str:
    _assert_private_repository(repository)
    assets = [_gh_fs_path(path) for path in _paths(source, tag).values()]
    if _release_exists(tag, repository):
        _gh("release", "upload", tag, *assets, "--repo", repository, "--clobber")
    else:
        _gh(
            "release",
            "create",
            tag,
            *assets,
            "--repo",
            repository,
            "--target",
            "main",
            "--title",
            tag,
            "--notes",
            (
                "Private backup of HY-WORLD-derived native wheels. Redistribution remains governed "
                "by the Tencent HY-WORLD 2.0 Community License included in the bundle. Public build "
                "recipes and source revisions live in xiaoqianran/modal-build."
            ),
        )

    view = _gh(
        "release",
        "view",
        tag,
        "--repo",
        repository,
        "--json",
        "assets,url,isDraft,isPrerelease",
        capture=True,
    )
    payload = json.loads(view.stdout)
    if payload.get("isDraft") or payload.get("isPrerelease"):
        raise ArtifactError(f"unexpected draft/prerelease state for private artifact {tag}")
    assets_by_name = {item["name"]: item for item in payload.get("assets", [])}
    for path in _paths(source, tag).values():
        asset = assets_by_name.get(path.name)
        if asset is None:
            raise ArtifactError(f"release asset missing after upload: {path.name}")
        expected_digest = f"sha256:{_sha256(path)}"
        if asset.get("digest") != expected_digest:
            raise ArtifactError(
                f"GitHub asset digest mismatch for {path.name}: {asset.get('digest')} != {expected_digest}"
            )
    return str(payload["url"])


def backup(spec: RestrictedBundleSpec, *, repository: str, volume: str) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix=f"hyworld2-backup-{spec.cuda_arch}-") as temp_dir:
        root = Path(temp_dir)
        _download_volume(spec.tag, root, volume)
        manifest = validate_bundle(root, spec)
        url = _publish_release(spec.tag, root, repository)
        return {
            "status": "backed-up",
            "tag": spec.tag,
            "sha256": str(manifest["archive_sha256"]),
            "repository": repository,
            "release": url,
        }


def restore(spec: RestrictedBundleSpec, *, repository: str, volume: str) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix=f"hyworld2-restore-{spec.cuda_arch}-") as temp_dir:
        root = Path(temp_dir) / "release"
        _download_release(spec.tag, root, repository)
        manifest = validate_bundle(root, spec)
        _upload_volume(spec.tag, root, volume)

        verify_root = Path(temp_dir) / "volume-verify"
        _download_volume(spec.tag, verify_root, volume)
        verify_manifest = validate_bundle(verify_root, spec)
        if verify_manifest["archive_sha256"] != manifest["archive_sha256"]:
            raise ArtifactError(f"post-upload verification changed archive for {spec.tag}")
        return {
            "status": "restored",
            "tag": spec.tag,
            "sha256": str(manifest["archive_sha256"]),
            "repository": repository,
            "volume": volume,
        }


def verify_volume(spec: RestrictedBundleSpec, *, volume: str) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix=f"hyworld2-verify-{spec.cuda_arch}-") as temp_dir:
        root = Path(temp_dir)
        _download_volume(spec.tag, root, volume)
        manifest = validate_bundle(root, spec)
        return {
            "status": "ready",
            "tag": spec.tag,
            "sha256": str(manifest["archive_sha256"]),
            "volume": volume,
        }


def _compile(spec: RestrictedBundleSpec) -> None:
    _modal("run", spec.builder)


def ensure(
    spec: RestrictedBundleSpec,
    *,
    repository: str,
    volume: str,
    compile_if_missing: bool,
) -> dict[str, str]:
    try:
        return verify_volume(spec, volume=volume)
    except (ArtifactError, subprocess.CalledProcessError) as volume_error:
        print(f"Volume cache unavailable for {spec.tag}: {volume_error}")

    try:
        return restore(spec, repository=repository, volume=volume)
    except (ArtifactError, subprocess.CalledProcessError) as restore_error:
        if not compile_if_missing:
            raise ArtifactError(
                f"{spec.tag} is unavailable in both Volume and private Release; "
                "refusing GPU compilation without --compile-if-missing"
            ) from restore_error
        print(f"Private restore unavailable for {spec.tag}; compiling explicitly requested fallback")

    _compile(spec)
    result = backup(spec, repository=repository, volume=volume)
    result["status"] = "compiled-and-backed-up"
    return result


def _select_specs(tags: Iterable[str], all_bundles: bool) -> list[RestrictedBundleSpec]:
    if all_bundles:
        return list(RESTRICTED_BUNDLES)
    selected = []
    for tag in tags:
        try:
            selected.append(SPECS_BY_TAG[tag])
        except KeyError as exc:
            raise ArtifactError(f"unknown restricted HYWorld2 tag: {tag}") from exc
    if not selected:
        raise ArtifactError("provide at least one tag or --all")
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Back up and restore HYWorld2 restricted native bundles using a private GitHub Release."
    )
    parser.add_argument(
        "command",
        choices=("backup", "restore", "verify", "ensure"),
        help="artifact operation",
    )
    parser.add_argument("tags", nargs="*", help="restricted bundle tags")
    parser.add_argument("--all", action="store_true", help="operate on both sm90 and sm120 bundles")
    parser.add_argument(
        "--repository",
        default=os.environ.get("HYWORLD2_PRIVATE_ARTIFACT_REPO", DEFAULT_PRIVATE_REPOSITORY),
    )
    parser.add_argument(
        "--volume",
        default=os.environ.get("HYWORLD2_ARTIFACT_VOLUME", DEFAULT_ARTIFACT_VOLUME),
    )
    parser.add_argument(
        "--compile-if-missing",
        action="store_true",
        help="allow expensive GPU compilation only after both caches are unavailable",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        specs = _select_specs(args.tags, args.all)
        results: list[dict[str, str]] = []
        for spec in specs:
            if args.command == "backup":
                result = backup(spec, repository=args.repository, volume=args.volume)
            elif args.command == "restore":
                result = restore(spec, repository=args.repository, volume=args.volume)
            elif args.command == "verify":
                result = verify_volume(spec, volume=args.volume)
            else:
                result = ensure(
                    spec,
                    repository=args.repository,
                    volume=args.volume,
                    compile_if_missing=bool(args.compile_if_missing),
                )
            results.append(result)
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0
    except (ArtifactError, json.JSONDecodeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
