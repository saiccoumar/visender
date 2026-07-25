"""End-to-end tests that run a real Blender.

Skipped when no Blender binary is found (set ``$BLENDER`` to point at one).
These are the only tests that exercise scene building, lighting, camera setup
and the render itself, so they are deliberately cheap: EEVEE, 8 samples, tiny
resolution. Correctness of *pixels* is not asserted -- only that every stage
runs, produces its artefacts, and that the scene contains what it should.

``-P`` runs a script in Blender's Python. The in-Blender assertions live in
``_scene_probe.py`` next to this file, which imports ``blender_import`` as a
library and prints a JSON report we assert on out here.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PROBE = HERE / "_scene_probe.py"

pytestmark = pytest.mark.slow


def run_blender(blender: str, script: Path, argv: list[str], cwd=None) -> str:
    env = dict(os.environ, MUJOCO_GL="egl", BLENDER_USER_SCRIPTS="")
    proc = subprocess.run(
        [blender, "-b", "--factory-startup", "--python-exit-code", "1",
         "--python", str(script), "--", *argv],
        capture_output=True, text=True, env=env, cwd=cwd, timeout=600)
    if proc.returncode != 0:
        pytest.fail(f"blender failed ({proc.returncode})\n"
                    f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    return proc.stdout


def probe(blender: str, bundle: Path, extra: list[str]) -> dict:
    out = run_blender(blender, PROBE, ["--bundle", str(bundle), *extra])
    report = [ln for ln in out.splitlines() if ln.startswith("PROBE_JSON:")]
    assert report, f"probe produced no report:\n{out}"
    return json.loads(report[-1][len("PROBE_JSON:"):])


BLENDER_IMPORT = Path(__file__).resolve().parents[1] / "viser2blender" / "blender_import.py"


# --------------------------------------------------------------------------- #
# The module has to at least import inside Blender.
# --------------------------------------------------------------------------- #

def test_blender_import_module_loads_inside_blender(blender, tmp_path):
    script = tmp_path / "smoke.py"
    script.write_text(
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('bi', r'{BLENDER_IMPORT}')\n"
        "m = importlib.util.module_from_spec(spec)\n"
        # Registered before exec: @dataclass looks its own module up by name.
        "sys.modules['bi'] = m\n"
        "spec.loader.exec_module(m)\n"
        "print('LOADED', m.SUPPORTED_BUNDLE_VERSION)\n")
    assert "LOADED 1" in run_blender(blender, script, [])


# --------------------------------------------------------------------------- #
# Scene building
# --------------------------------------------------------------------------- #

def test_every_node_kind_builds(blender, make_bundle):
    report = probe(blender, make_bundle(), [])
    created = report["created"]
    assert set(created) == {"/mesh", "/group/box", "/group/ball", "/cloud", "/lines"}
    assert report["objects_by_type"]["MESH"] >= 3
    assert report["objects_by_type"].get("LIGHT", 0) >= 1  # the DirectionalLight
    assert report["camera"] is not None
    assert report["mesh_vertex_counts"]["/mesh"] == 4


def test_camera_matches_the_manifest_pose(blender, make_bundle):
    report = probe(blender, make_bundle(), [])
    cam = report["camera"]
    assert cam["location"] == pytest.approx([4.0, -4.0, 3.0], abs=1e-4)
    # Looking at the origin from (4,-4,3): the camera's -Z axis points inward.
    assert cam["direction"][0] < 0 and cam["direction"][1] > 0 and cam["direction"][2] < 0
    assert cam["resolution"] == [1600, 900]
    assert cam["sensor_fit"] == "VERTICAL"


def test_resolution_and_scale_overrides_apply(blender, make_bundle):
    report = probe(blender, make_bundle(),
                   ["--resolution", "640", "360", "--scale", "50"])
    assert report["camera"]["resolution"] == [640, 360]
    assert report["camera"]["resolution_percentage"] == 50


def test_orbit_and_dolly_move_the_camera_but_keep_the_look_at(blender, make_bundle):
    base = probe(blender, make_bundle(), [])["camera"]["location"]
    orbited = probe(blender, make_bundle(), ["--orbit", "90", "0"])["camera"]["location"]
    dollied = probe(blender, make_bundle(), ["--dolly", "2.0"])["camera"]["location"]

    def norm(v):
        return sum(c * c for c in v) ** 0.5

    assert norm(orbited) == pytest.approx(norm(base), rel=1e-4)  # same radius
    assert orbited != pytest.approx(base, abs=1e-3)              # actually moved
    assert norm(dollied) == pytest.approx(2 * norm(base), rel=1e-4)


def test_auto_camera_frames_the_scene_when_the_bundle_has_none(blender, make_bundle):
    bundle = make_bundle(camera=False)
    assert probe(blender, bundle, [])["camera"] is None
    report = probe(blender, bundle, ["--auto-camera"])
    assert report["camera"] is not None
    # It must frame the geometry, not the backdrop plane sized from it.
    with_backdrop = probe(blender, bundle, ["--auto-camera", "--backdrop"])
    assert with_backdrop["camera"]["location"] == \
        pytest.approx(report["camera"]["location"], rel=1e-3)


def test_keep_default_cube_is_off_by_default(blender, make_bundle):
    assert "Cube" not in probe(blender, make_bundle(), [])["object_names"]
    assert "Cube" in probe(blender, make_bundle(), ["--keep-default-cube"])["object_names"]


# --------------------------------------------------------------------------- #
# Materials
# --------------------------------------------------------------------------- #

def test_material_override_replaces_the_node_material(blender, make_bundle):
    report = probe(blender, make_bundle(), ["--material", "/mesh=use:aluminium"])
    assert report["materials"]["/mesh"] == ["aluminium"]


def test_subtree_material_override_hits_every_child(blender, make_bundle):
    report = probe(blender, make_bundle(), ["--material", "/group/*=use:copper"])
    assert report["materials"]["/group/box"] == ["copper"]
    assert report["materials"]["/group/ball"] == ["copper"]
    assert report["materials"]["/mesh"] != ["copper"]


def test_a_named_material_is_one_shared_datablock(blender, make_bundle):
    """One datablock per name, not one per object."""
    report = probe(blender, make_bundle(),
                   ["--material", "/group/box=use:copper",
                    "--material", "/group/ball=use:copper"])
    assert report["material_datablock_counts"]["copper"] == 1


def test_unknown_named_material_fails_loudly(blender, make_bundle, tmp_path):
    proc = subprocess.run(
        [blender, "-b", "--factory-startup", "--python", str(PROBE), "--",
         "--bundle", str(make_bundle()), "--material", "/mesh=use:unobtainium"],
        capture_output=True, text=True, timeout=600)
    assert "unobtainium" in proc.stdout + proc.stderr


def test_config_split_rule_assigns_per_polygon_materials(blender, make_bundle, tmp_path):
    config = tmp_path / "c.json"
    config.write_text(json.dumps({
        "material_rules": [{
            "node": "/group/box",
            "split": [
                {"where": {"normal_z": ">0.9"}, "use": "aluminium"},
                {"default": True, "use": "matte_black"},
            ],
        }],
    }))
    report = probe(blender, make_bundle(), ["--config", str(config)])
    slots = report["materials"]["/group/box"]
    assert "aluminium" in slots and "matte_black" in slots
    used = report["polygon_material_names"]["/group/box"]
    assert set(used) == {"aluminium", "matte_black"}   # both rules actually fired


# --------------------------------------------------------------------------- #
# Lighting and world
# --------------------------------------------------------------------------- #

def test_key_light_is_placed_relative_to_the_camera(blender, make_bundle):
    report = probe(blender, make_bundle(), ["--key-light=-40,35"])
    assert "key_light" in report["object_names"]


def test_key_light_follows_an_orbited_camera(blender, make_bundle):
    """Camera-relative lighting must read the *effective* camera."""
    a = probe(blender, make_bundle(), ["--key-light=-40,35"])["light_directions"]
    b = probe(blender, make_bundle(),
              ["--key-light=-40,35", "--orbit", "90", "0"])["light_directions"]
    assert a["key_light"] != pytest.approx(b["key_light"], abs=1e-3)


def test_three_point_rig_adds_three_lights(blender, make_bundle):
    names = probe(blender, make_bundle(), ["--three-point"])["object_names"]
    assert {"key_light", "fill_light"} <= set(names)


def test_auto_light_only_fires_when_the_bundle_has_none(blender, make_bundle):
    with_lights = probe(blender, make_bundle(lights=True), ["--auto-light"])
    assert "key_light" not in with_lights["object_names"]
    without = probe(blender, make_bundle(lights=False), ["--auto-light"])
    assert "key_light" in without["object_names"]


def test_dim_authored_scales_imported_light_energy(blender, make_bundle):
    full = probe(blender, make_bundle(), [])["light_energies"]
    dimmed = probe(blender, make_bundle(), ["--dim-authored", "0.25"])["light_energies"]
    name = next(iter(full))
    assert dimmed[name] == pytest.approx(full[name] * 0.25, rel=1e-4)


def test_backdrop_plane_is_added_below_the_scene(blender, make_bundle):
    report = probe(blender, make_bundle(), ["--backdrop"])
    assert "backdrop" in report["object_names"]
    assert report["backdrop_z"] <= report["scene_bounds"][0][2] + 1e-6


def test_studio_world_and_hdri_free_operation(blender, make_bundle):
    report = probe(blender, make_bundle(), ["--studio-world", "--world-strength", "0.9"])
    assert report["world_nodes"]  # a node tree was built


# --------------------------------------------------------------------------- #
# Full pipeline: render, .blend, sidecar
# --------------------------------------------------------------------------- #

def test_full_render_writes_image_blend_and_sidecar(blender, make_bundle, tmp_path):
    out = tmp_path / "out" / "shot.png"
    bundle = make_bundle()
    run_blender(blender, BLENDER_IMPORT, [
        "--bundle", str(bundle), "--render", str(out),
        "--engine", "EEVEE", "--samples", "8", "--resolution", "160", "90",
        "--save-blend", "--backdrop", "--auto-light",
    ])
    assert out.exists() and out.stat().st_size > 0
    assert out.with_suffix(".blend").exists()

    sidecar = out.with_suffix(".png.yaml")
    assert sidecar.exists()
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(sidecar.read_text())
    assert doc["bundle"] == str(bundle)
    assert doc["bundle_version"] == 1
    assert doc["render_time_seconds"] >= 0
    assert doc["resolved_config"]["samples"] == 8
    assert doc["resolved_config"]["resolution"] == [160, 90]
    assert "_resolved_config" not in doc["resolved_config"]
    assert "config_file_values" not in doc  # no --config was passed


def test_sidecar_records_effective_settings_not_the_config_file(blender, make_bundle,
                                                                tmp_path):
    """A CLI flag that beat the config must be what provenance reports."""
    config = tmp_path / "c.json"
    config.write_text(json.dumps({"samples": 999, "engine": "EEVEE"}))
    out = tmp_path / "shot.png"
    run_blender(blender, BLENDER_IMPORT, [
        "--bundle", str(make_bundle()), "--render", str(out), "--config", str(config),
        "--samples", "8", "--resolution", "160", "90", "--auto-light",
    ])
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(out.with_suffix(".png.yaml").read_text())
    assert doc["resolved_config"]["samples"] == 8       # effective
    assert doc["config_file_values"]["samples"] == 999  # what the file asked for


def test_transparent_film_produces_rgba(blender, make_bundle, tmp_path):
    out = tmp_path / "shot.png"
    run_blender(blender, BLENDER_IMPORT, [
        "--bundle", str(make_bundle()), "--render", str(out), "--transparent",
        "--engine", "EEVEE", "--samples", "8", "--resolution", "64", "64",
    ])
    png = out.read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert png[25] == 6  # IHDR colour type 6 = RGBA


def test_list_nodes_short_circuits_before_building(blender, make_bundle):
    out = run_blender(blender, BLENDER_IMPORT,
                      ["--bundle", str(make_bundle()), "--list-nodes"])
    assert "/mesh" in out and "PointCloudProps" in out


def test_missing_bundle_argument_is_a_clean_error(blender, tmp_path):
    proc = subprocess.run(
        [blender, "-b", "--factory-startup", "--python", str(BLENDER_IMPORT), "--"],
        capture_output=True, text=True, timeout=600)
    assert "no bundle" in proc.stdout + proc.stderr


def test_v2b_end_to_end_through_the_cli(blender, make_bundle, tmp_path):
    """The wrapper, the config, and Blender, all for real."""
    bundle = make_bundle()
    cfg = tmp_path / "scene.yaml"
    cfg.write_text(
        f"bundle: {bundle}\n"
        f"blender: {blender}\n"
        "output: out/{profile}.png\n"
        "profiles:\n"
        "  test: {engine: EEVEE, samples: 8, resolution: [160, 90]}\n"
        "lighting: {auto: true}\n"
        "backdrop: {enabled: true, color: [205, 205, 210]}\n"
    )
    from viser2blender import cli
    assert cli.main(["render", str(cfg), "--profile", "test"]) == 0
    assert (tmp_path / "out" / "test.png").exists()
