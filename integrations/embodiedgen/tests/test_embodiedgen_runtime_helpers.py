import importlib.util
import tempfile
import unittest
from pathlib import Path

RUNTIME = Path(__file__).parents[1] / "runtime" / "embodiedgen_v2_l40s.py"
spec = importlib.util.spec_from_file_location("embodiedgen_runtime_helpers", RUNTIME)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class MeshSimplificationTest(unittest.TestCase):
    def test_small_mesh_bypasses_simplifier(self):
        vertices = [0, 1, 2, 3]
        faces = [0, 1, 2, 3]

        def must_not_run(*_args, **_kwargs):
            raise AssertionError("simplifier must not run for <= target faces")

        out_vertices, out_faces, simplified = runtime.simplify_mesh_if_needed(
            vertices, faces, must_not_run, target_faces=4
        )
        self.assertIs(out_vertices, vertices)
        self.assertIs(out_faces, faces)
        self.assertFalse(simplified)

    def test_oversized_mesh_calls_simplifier_with_target(self):
        calls = []

        def fake_simplify(vertices, faces, **kwargs):
            calls.append(kwargs)
            return ["v"], ["f"]

        vertices, faces, simplified = runtime.simplify_mesh_if_needed(
            list(range(6)), list(range(5)), fake_simplify, target_faces=4
        )
        self.assertTrue(simplified)
        self.assertEqual(vertices, ["v"])
        self.assertEqual(faces, ["f"])
        self.assertEqual(calls[0]["target_count"], 4)
        self.assertEqual(calls[0]["agg"], 7.0)
        self.assertFalse(calls[0]["preserve_border"])


class ObjBundleTest(unittest.TestCase):
    def test_copy_obj_bundle_includes_mtl_and_texture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "job"
            out = Path(tmp) / "result"
            root.mkdir()
            (root / "sample.obj").write_text("mtllib material.mtl\nv 0 0 0\n")
            (root / "material.mtl").write_text("newmtl material_0\nmap_Kd material_0.png\n")
            (root / "material_0.png").write_bytes(b"png")

            runtime.copy_obj_bundle(root / "sample.obj", out)

            self.assertTrue((out / "sample.obj").exists())
            self.assertTrue((out / "material.mtl").exists())
            self.assertTrue((out / "material_0.png").exists())
            self.assertEqual(runtime.missing_obj_material_dependencies(out / "sample.obj"), [])

    def test_nested_mtl_texture_path_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "job"
            out = Path(tmp) / "result"
            (root / "materials" / "textures").mkdir(parents=True)
            (root / "sample.obj").write_text("mtllib materials/material.mtl\n")
            (root / "materials" / "material.mtl").write_text("map_Kd textures/albedo.png\n")
            (root / "materials" / "textures" / "albedo.png").write_bytes(b"png")

            runtime.copy_obj_bundle(root / "sample.obj", out)

            self.assertTrue((out / "materials" / "material.mtl").exists())
            self.assertTrue((out / "materials" / "textures" / "albedo.png").exists())
            self.assertEqual(runtime.missing_obj_material_dependencies(out / "sample.obj"), [])

    def test_missing_texture_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.obj").write_text("mtllib material.mtl\n")
            (root / "material.mtl").write_text("map_Kd missing.png\n")
            self.assertEqual(
                runtime.missing_obj_material_dependencies(root / "sample.obj"),
                ["missing.png"],
            )

    def test_obj_reference_cannot_escape_job_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.obj").write_text("mtllib ../outside.mtl\n")
            with self.assertRaises(RuntimeError):
                runtime.obj_material_dependencies(root / "sample.obj")


class ValidationTest(unittest.TestCase):
    def valid_checks(self):
        return {
            "ply_vertices": 1,
            "obj_vertices": 1,
            "obj_faces": 1,
            "glb_geometries": 1,
            "urdf_mesh_exists": True,
            "video_exists": True,
            "obj_material_refs_ok": True,
        }

    def test_all_required_outputs_pass(self):
        self.assertTrue(runtime.validation_passes(self.valid_checks()))

    def test_missing_video_fails(self):
        checks = self.valid_checks()
        checks["video_exists"] = False
        self.assertFalse(runtime.validation_passes(checks))

    def test_broken_obj_material_fails(self):
        checks = self.valid_checks()
        checks["obj_material_refs_ok"] = False
        self.assertFalse(runtime.validation_passes(checks))


if __name__ == "__main__":
    unittest.main()
