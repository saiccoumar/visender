"""Regression tests for YAML -> flat-Settings config resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visender import config as cfg


def write(tmp_path: Path, text: str, name: str = "scene.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


# --------------------------------------------------------------------------- #
# Flattening: every nested section must land on its Settings field.
# --------------------------------------------------------------------------- #

def test_nested_sections_flatten_to_settings_fields(tmp_path):
    path = write(tmp_path, """
bundle: my_bundle
world: {studio: true, strength: 0.9}
camera: {fit: fit_all, auto: true, orbit: [10, -5], dolly: 1.2, dof: 2.5, fstop: 4.0, scale: 50}
film: {exposure: 0.5, look: "AgX - Punchy", transparent: true}
backdrop: {enabled: true, color: [205, 205, 210], shadow_catcher: true}
lighting:
  sun_scale: 2.0
  point_scale: 12.0
  dim_authored: 0.15
  three_point: false
  auto: true
  key: {az: -70.7, el: 33.5, energy: 4.0, angle: 12.0, color: [255, 250, 244]}
save_blend: true
""")
    flat, blender = cfg.resolve(path)

    assert blender is None
    assert flat["studio_world"] is True
    assert flat["world_strength"] == 0.9
    assert flat["fit"] == "fit_all"
    assert flat["auto_camera"] is True
    assert flat["orbit"] == [10, -5]
    assert flat["dolly"] == 1.2
    assert flat["dof"] == 2.5
    assert flat["fstop"] == 4.0
    assert flat["scale"] == 50
    assert flat["exposure"] == 0.5
    assert flat["look"] == "AgX - Punchy"
    assert flat["transparent"] is True
    assert flat["backdrop"] is True
    assert flat["backdrop_color"] == "205,205,210"
    assert flat["shadow_catcher"] is True
    assert flat["sun_scale"] == 2.0
    assert flat["point_scale"] == 12.0
    assert flat["dim_authored"] == 0.15
    assert flat["auto_light"] is True
    assert flat["key_light"] == "-70.7,33.5"
    assert flat["key_energy"] == 4.0
    assert flat["key_angle"] == 12.0
    assert flat["key_color"] == "255,250,244"
    assert flat["save_blend"] is True

    # Every emitted key must be a real Settings field, or Blender silently drops it.
    assert set(flat) <= cfg.SETTINGS_FIELDS


def test_absent_sections_emit_no_keys(tmp_path):
    """An empty config must not stamp defaults over the Blender-side defaults."""
    flat, _ = cfg.resolve(write(tmp_path, "bundle: b\n"))
    assert set(flat) == {"bundle"}


def test_false_and_zero_survive_flattening(tmp_path):
    """``_put`` filters None only -- falsey values are real user choices."""
    flat, _ = cfg.resolve(write(tmp_path, """
bundle: b
world: {studio: false, strength: 0.0}
lighting: {dim_authored: 0.0}
film: {exposure: 0.0, transparent: false}
"""))
    assert flat["studio_world"] is False
    assert flat["world_strength"] == 0.0
    assert flat["dim_authored"] == 0.0
    assert flat["exposure"] == 0.0
    assert flat["transparent"] is False


def test_key_light_needs_both_az_and_el(tmp_path):
    flat, _ = cfg.resolve(write(tmp_path, """
bundle: b
lighting: {key: {az: -40, energy: 3.0}}
"""))
    assert "key_light" not in flat
    assert flat["key_energy"] == 3.0


# --------------------------------------------------------------------------- #
# Paths are relative to the config file, never to the CWD.
# --------------------------------------------------------------------------- #

def test_relative_paths_resolve_against_config_dir(tmp_path, monkeypatch):
    sub = tmp_path / "cfgdir"
    sub.mkdir()
    path = write(sub, """
bundle: bundles/mine
blender: ../blender/blender
output: out/img.png
world: {hdri: hdris/studio.exr}
""")
    monkeypatch.chdir(tmp_path.parent)
    flat, blender = cfg.resolve(path)
    assert flat["bundle"] == str(sub / "bundles/mine")
    assert flat["hdri"] == str(sub / "hdris/studio.exr")
    assert flat["render"] == str(sub / "out/img.png")
    assert Path(blender).resolve() == (tmp_path / "blender/blender").resolve()


def test_absolute_paths_are_left_alone(tmp_path):
    flat, _ = cfg.resolve(write(tmp_path, f"bundle: {tmp_path}/abs\n"))
    assert flat["bundle"] == f"{tmp_path}/abs"


def test_missing_config_is_a_config_error(tmp_path):
    with pytest.raises(cfg.ConfigError):
        cfg.resolve(tmp_path / "nope.yaml")


def test_non_mapping_top_level_is_an_error(tmp_path):
    with pytest.raises(cfg.ConfigError):
        cfg.resolve(write(tmp_path, "- a\n- b\n"))


# --------------------------------------------------------------------------- #
# Typos must be hard errors, at every level.
# --------------------------------------------------------------------------- #

def test_unknown_top_level_key_errors_with_suggestion(tmp_path):
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.resolve(write(tmp_path, "bundle: b\nlightning: {}\n"))
    assert "lightning" in str(exc.value)
    assert "lighting" in str(exc.value)


def test_unknown_section_key_errors(tmp_path):
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.resolve(write(tmp_path, "bundle: b\ncamera: {fitt: fit_all}\n"))
    assert "camera" in str(exc.value) and "fitt" in str(exc.value)


def test_unknown_lighting_key_subkey_errors(tmp_path):
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.resolve(write(tmp_path, "bundle: b\nlighting: {key: {azimuth: 3}}\n"))
    assert "lighting.key" in str(exc.value)


def test_unknown_profile_key_errors(tmp_path):
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.resolve(write(tmp_path, "bundle: b\nprofiles: {final: {sampels: 10}}\n"),
                    profile="final")
    assert "sampels" in str(exc.value)


def test_unknown_profile_name_errors(tmp_path):
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.resolve(write(tmp_path, "bundle: b\nprofiles: {draft: {samples: 8}}\n"),
                    profile="final")
    assert "draft" in str(exc.value)


# --------------------------------------------------------------------------- #
# Precedence: config file < profile.
# --------------------------------------------------------------------------- #

def test_profile_overrides_config_values(tmp_path):
    src = """
bundle: b
world: {strength: 0.5}
profiles:
  final: {world_strength: 1.5, samples: 4096, engine: CYCLES}
"""
    base, _ = cfg.resolve(write(tmp_path, src))
    assert base["world_strength"] == 0.5 and "samples" not in base

    final, _ = cfg.resolve(write(tmp_path, src), profile="final")
    assert final["world_strength"] == 1.5
    assert final["samples"] == 4096
    assert final["engine"] == "CYCLES"


def test_profile_values_get_the_same_coercion_as_sections(tmp_path):
    """A profile speaks flat keys but must still get colour/path normalisation."""
    flat, _ = cfg.resolve(write(tmp_path, """
bundle: b
profiles:
  hero:
    key_color: [255, 250, 244]
    backdrop_color: [205, 205, 210]
    point_color: [10, 20, 30]
    hdri: hdris/studio.exr
    save_blend: out/hero.blend
"""), profile="hero")
    assert flat["key_color"] == "255,250,244"
    assert flat["backdrop_color"] == "205,205,210"
    assert flat["point_color"] == "10,20,30"
    assert flat["hdri"] == str(tmp_path / "hdris/studio.exr")
    assert flat["save_blend"] == str(tmp_path / "out/hero.blend")


def test_profile_save_blend_true_is_not_path_resolved(tmp_path):
    flat, _ = cfg.resolve(write(tmp_path, """
bundle: b
profiles: {hero: {save_blend: true}}
"""), profile="hero")
    assert flat["save_blend"] is True


# --------------------------------------------------------------------------- #
# Output templating.
# --------------------------------------------------------------------------- #

def test_output_template_expands_profile_and_bundle_name(tmp_path):
    flat, _ = cfg.resolve(write(tmp_path, """
bundle: bundles/pen_grip_151402
output: out/{bundle_name}_{profile}.png
profiles: {final: {samples: 64}}
"""), profile="final")
    assert flat["render"] == str(tmp_path / "out/pen_grip_151402_final.png")


def test_output_template_profile_defaults_to_the_word_default(tmp_path):
    flat, _ = cfg.resolve(write(tmp_path, "bundle: b\noutput: out/{profile}.png\n"))
    assert flat["render"] == str(tmp_path / "out/default.png")


def test_output_argument_beats_config_output(tmp_path):
    flat, _ = cfg.resolve(write(tmp_path, "bundle: b\noutput: out/config.png\n"),
                          output="out/cli.png")
    assert flat["render"] == str(tmp_path / "out/cli.png")


def test_output_template_supports_date_and_time(tmp_path):
    flat, _ = cfg.resolve(write(tmp_path, "bundle: b\noutput: out/{date}_{time}.png\n"))
    stem = Path(flat["render"]).stem
    date, _, time = stem.partition("_")
    assert len(date) == 8 and date.isdigit()
    assert len(time) == 6 and time.isdigit()


# --------------------------------------------------------------------------- #
# Material rules and aliases.
# --------------------------------------------------------------------------- #

def test_material_rules_expand_aliases(tmp_path):
    flat, _ = cfg.resolve(write(tmp_path, """
bundle: b
aliases:
  robot: /allegro
  hand: "{robot}/hand"
materials:
  library: {my_metal: {base_color: [158, 163, 168], metallic: 1.0}}
  rules:
    - {node: "{hand}/link_0", use: aluminium}
    - {node: "{robot}/*", use: my_metal}
"""))
    assert [r["node"] for r in flat["material_rules"]] == \
        ["/allegro/hand/link_0", "/allegro/*"]
    assert flat["material_library"]["my_metal"]["metallic"] == 1.0


def test_unknown_alias_errors(tmp_path):
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.resolve(write(tmp_path, """
bundle: b
materials: {rules: [{node: "{nope}/x", use: aluminium}]}
"""))
    assert "unknown alias" in str(exc.value)


def test_cyclic_alias_errors_instead_of_hanging(tmp_path):
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.resolve(write(tmp_path, """
bundle: b
aliases: {a: "{b}", b: "{a}"}
materials: {rules: [{node: "{a}/x", use: aluminium}]}
"""))
    assert "did not terminate" in str(exc.value)


def test_material_rules_survive_split_specs(tmp_path):
    flat, _ = cfg.resolve(write(tmp_path, """
bundle: b
materials:
  rules:
    - node: /table
      split:
        - {where: {normal_z: ">0.9", world_z: [0.0, 0.1]}, use: aluminium}
        - {default: true, use: matte_black}
"""))
    rule = flat["material_rules"][0]
    assert rule["split"][0]["where"]["normal_z"] == ">0.9"
    assert rule["split"][1]["default"] is True


# --------------------------------------------------------------------------- #
# Scaffolding / manifest helpers.
# --------------------------------------------------------------------------- #

def test_scaffold_round_trips_through_resolve(tmp_path, make_bundle):
    bundle = make_bundle()
    text = cfg.scaffold(bundle)
    path = write(tmp_path, text, "scaffold.yaml")
    flat, _ = cfg.resolve(path, profile="draft")
    assert flat["bundle"] == str(bundle)
    assert flat["engine"] == "EEVEE"
    assert flat["samples"] == 64
    assert flat["resolution"] == [1280, 720]
    # Mesh-bearing nodes are offered as commented rule stubs.
    assert "/mesh" in text and "/group/box" in text


def test_load_manifest_reads_scene_json(make_bundle):
    bundle = make_bundle()
    manifest = cfg.load_manifest(bundle)
    assert manifest["format"] == "visender"
    assert manifest == json.loads((bundle / "scene.json").read_text())
