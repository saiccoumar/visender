"""Regression tests for the pure-logic half of ``blender_import``.

Everything here runs against the stubbed bpy from ``conftest`` (or the real one
when pytest is run inside Blender): argument parsing, settings precedence,
material specs, split predicates and the provenance emitter never touch Blender
data. Scene building is covered in ``test_blender_integration.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


# --------------------------------------------------------------------------- #
# Settings precedence: default < config < explicit CLI flag
# --------------------------------------------------------------------------- #

def settings_from(bi, argv, config=None):
    args = bi.parse_args(argv)
    return bi.Settings.from_args(args, bi._explicit_flags(argv), config)


def test_defaults_come_from_the_dataclass(bi):
    s = settings_from(bi, ["--bundle", "b"])
    assert s.samples == 128
    assert s.engine == "CYCLES"
    assert s.fit == "keep_vertical"
    assert s.point_scale == pytest.approx(4.0 * math.pi)


def test_config_fills_flags_the_user_did_not_pass(bi):
    s = settings_from(bi, ["--bundle", "b"], {"samples": 4096, "engine": "EEVEE"})
    assert s.samples == 4096
    assert s.engine == "EEVEE"


def test_explicit_flag_beats_config(bi):
    s = settings_from(bi, ["--bundle", "b", "--samples", "32"],
                      {"samples": 4096, "engine": "EEVEE"})
    assert s.samples == 32       # CLI wins
    assert s.engine == "EEVEE"   # config still fills the rest


def test_explicit_flag_beats_config_even_when_it_equals_the_default(bi):
    """The SUPPRESS-per-action trick exists exactly for this case."""
    s = settings_from(bi, ["--bundle", "b", "--samples", "128"], {"samples": 4096})
    assert s.samples == 128


def test_explicit_store_true_flags_are_detected(bi):
    explicit = bi._explicit_flags(["--bundle", "b", "--gpu", "--backdrop"])
    assert {"bundle", "gpu", "backdrop"} <= explicit
    assert "samples" not in explicit
    # An explicit --gpu must not be undone by a config that says false.
    s = settings_from(bi, ["--bundle", "b", "--gpu"], {"gpu": False})
    assert s.gpu is True


def test_every_action_carries_a_suppressed_default_in_the_sentinel_parser(bi):
    """Guards the documented gotcha: a per-arg ``default=`` outranks a
    parser-level SUPPRESS, so an unpassed flag would look explicit."""
    import argparse
    parser = bi.build_parser()
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    ns, _ = parser.parse_known_args([])
    assert vars(ns) == {}


def test_config_only_keys_reach_settings(bi):
    s = settings_from(bi, ["--bundle", "b"], {
        "material_rules": [{"node": "/table", "use": "aluminium"}],
        "material_library": {"my_metal": {"metallic": 1.0}},
    })
    assert s.material_rules[0]["node"] == "/table"
    assert s.material_library["my_metal"]["metallic"] == 1.0


def test_unknown_config_keys_are_ignored_not_fatal(bi):
    s = settings_from(bi, ["--bundle", "b"], {"not_a_field": 1, "samples": 7})
    assert s.samples == 7
    assert not hasattr(s, "not_a_field")


def test_json_lists_are_normalised_to_tuples(bi):
    s = settings_from(bi, ["--bundle", "b"],
                      {"resolution": [1280, 720], "orbit": [10.0, -5.0]})
    assert s.resolution == (1280, 720)
    assert s.orbit == (10.0, -5.0)


def test_resolved_config_is_recorded_for_the_sidecar(bi):
    cfg = {"samples": 64}
    s = settings_from(bi, ["--bundle", "b"], cfg)
    assert s._resolved_config == cfg
    assert settings_from(bi, ["--bundle", "b"])._resolved_config is None


def test_optional_value_flags_take_a_bare_form(bi):
    assert settings_from(bi, ["--bundle", "b", "--dof"]).dof is True
    assert settings_from(bi, ["--bundle", "b", "--dof", "2.5"]).dof == "2.5"
    assert settings_from(bi, ["--bundle", "b", "--save-blend"]).save_blend is True
    assert settings_from(bi, ["--bundle", "b", "--save-blend", "x.blend"]).save_blend \
        == "x.blend"


def test_material_flag_is_repeatable(bi):
    s = settings_from(bi, ["--bundle", "b", "--material", "/a=1,2,3",
                           "--material", "/b=use:aluminium"])
    assert s.material == ["/a=1,2,3", "/b=use:aluminium"]


def test_fit_rejects_an_unknown_policy(bi):
    with pytest.raises(SystemExit):
        bi.parse_args(["--bundle", "b", "--fit", "stretch"])


def test_load_config_reads_the_json_written_by_bliser(bi, tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"samples": 9}))
    assert bi.load_config(str(path)) == {"samples": 9}


# --------------------------------------------------------------------------- #
# Colour handling
# --------------------------------------------------------------------------- #

def test_parse_rgb_and_srgb_to_linear(bi):
    assert bi._parse_rgb("205,205,210") == (205, 205, 210)
    assert bi._srgb_to_linear((255, 255, 255)) == (1.0, 1.0, 1.0)
    assert bi._srgb_to_linear((0, 0, 0)) == (0.0, 0.0, 0.0)
    mid = bi._srgb_to_linear((128, 128, 128))[0]
    assert 0.2 < mid < 0.3  # gamma 2.2, not linear


def test_parse_rgb_rejects_the_wrong_arity(bi):
    with pytest.raises(SystemExit):
        bi._parse_rgb("205,205")


# --------------------------------------------------------------------------- #
# --material specs
# --------------------------------------------------------------------------- #

def test_positional_material_spec(bi):
    (path, subtree, props), = bi.parse_material_overrides(["/table=30,90,200"])
    assert (path, subtree) == ("/table", False)
    assert props == {"color": (30.0, 90.0, 200.0)}


def test_positional_material_spec_with_alpha_and_roughness(bi):
    (_, _, props), = bi.parse_material_overrides(["/t=30,90,200,0.5,0.2"])
    assert props["opacity"] == 0.5
    assert props["roughness"] == 0.2


def test_subtree_material_spec(bi):
    (path, subtree, _), = bi.parse_material_overrides(["/pen/*=30,30,40"])
    assert (path, subtree) == ("/pen", True)


def test_keyvalue_material_spec_is_full_pbr(bi):
    (_, _, props), = bi.parse_material_overrides(
        ["/t=base_color:158,163,168;metallic:1.0;roughness:0.22"])
    assert props == {"base_color": (158, 163, 168), "metallic": 1.0, "roughness": 0.22}


def test_named_material_spec(bi):
    (_, _, props), = bi.parse_material_overrides(["/t=use:aluminium"])
    assert props == {"use": "aluminium"}


def test_material_spec_errors(bi):
    with pytest.raises(SystemExit):
        bi.parse_material_overrides(["/table"])       # no '='
    with pytest.raises(SystemExit):
        bi.parse_material_overrides(["/table=30,90"])  # not enough channels


def test_later_material_rules_win(bi):
    rules = bi.parse_material_overrides(["/pen/*=1,1,1", "/pen/tip=2,2,2"])
    assert bi._override_for("/pen/tip", rules) == {"color": (2.0, 2.0, 2.0)}
    assert bi._override_for("/pen/body", rules) == {"color": (1.0, 1.0, 1.0)}
    assert bi._override_for("/table", rules) is None


def test_subtree_match_does_not_leak_to_sibling_prefixes(bi):
    rules = bi.parse_material_overrides(["/pen/*=1,1,1"])
    assert bi._override_for("/pencil/body", rules) is None
    assert bi._override_for("/pen", rules) == {"color": (1.0, 1.0, 1.0)}


def test_rule_targets_matches_node_and_subtree(bi):
    created = {"/a": ["A"], "/a/b": ["B"], "/ab": ["C"]}
    assert bi._rule_targets(created, "/a") == ["A"]
    assert sorted(bi._rule_targets(created, "/a/*")) == ["A", "B"]
    assert bi._rule_targets(created, "/missing") == []


# --------------------------------------------------------------------------- #
# Geometric split predicates
# --------------------------------------------------------------------------- #

def test_cmp_pred_operators(bi):
    assert bi._cmp_pred(">0.9")(0.95) and not bi._cmp_pred(">0.9")(0.9)
    assert bi._cmp_pred(">=0.9")(0.9)
    assert bi._cmp_pred("<-0.5")(-0.6) and not bi._cmp_pred("<-0.5")(-0.4)
    assert bi._cmp_pred("<=0")(0.0)


def test_cmp_pred_rejects_junk(bi):
    for spec in ("0.9", "=0.9", ">", ">abc"):
        with pytest.raises(SystemExit):
            bi._cmp_pred(spec)


def test_compile_where_combines_predicates_with_and(bi):
    match = bi._compile_where({"normal_z": ">0.9", "world_z": [0.0, 0.1], "area": [0.0, 1.0]})
    assert match([0, 0, 0.05], [0, 0, 1.0], 0.5, "")
    assert not match([0, 0, 0.5], [0, 0, 1.0], 0.5, "")   # out of world_z band
    assert not match([0, 0, 0.05], [0, 0, 0.1], 0.5, "")  # normal too shallow
    assert not match([0, 0, 0.05], [0, 0, 1.0], 5.0, "")  # area too large


def test_compile_where_matches_material_name_by_regex(bi):
    match = bi._compile_where({"material_name": "wood"})
    assert match([0, 0, 0], [0, 0, 1], 1.0, "oak_wood_01")
    assert not match([0, 0, 0], [0, 0, 1], 1.0, "steel")
    assert not match([0, 0, 0], [0, 0, 1], 1.0, "")


def test_compile_where_empty_matches_everything(bi):
    assert bi._compile_where({})([0, 0, 0], [0, 0, 1], 1.0, "")


def test_compile_where_rejects_unknown_predicates(bi):
    with pytest.raises(SystemExit) as exc:
        bi._compile_where({"normal_w": ">0.5"})
    assert "normal_w" in str(exc.value)


def test_compile_where_binds_each_axis_separately(bi):
    """A late-binding bug in the loop would make every axis test the last one."""
    match = bi._compile_where({"normal_x": ">0.9", "normal_y": "<0.1"})
    assert match([0, 0, 0], [1.0, 0.0, 0.0], 1.0, "")
    assert not match([0, 0, 0], [0.0, 1.0, 0.0], 1.0, "")


# --------------------------------------------------------------------------- #
# Bundle version gate
# --------------------------------------------------------------------------- #

def test_check_bundle_version_accepts_the_current_format(bi):
    bi.check_bundle_version({"format": "bliser", "version": 1}, Path("b"))


def test_check_bundle_version_rejects_a_foreign_directory(bi):
    with pytest.raises(SystemExit):
        bi.check_bundle_version({}, Path("b"))


def test_check_bundle_version_rejects_a_newer_bundle(bi):
    with pytest.raises(SystemExit) as exc:
        bi.check_bundle_version({"format": "bliser", "version": 99}, Path("b"))
    assert "Update bliser" in str(exc.value)


def test_check_bundle_version_warns_on_an_older_bundle(bi, capsys, monkeypatch):
    monkeypatch.setattr(bi, "SUPPORTED_BUNDLE_VERSION", 2)
    bi.check_bundle_version({"format": "bliser", "version": 1}, Path("b"))
    assert "WARNING" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Output paths and the provenance sidecar
# --------------------------------------------------------------------------- #

def test_derive_blend_path(bi):
    S = bi.Settings
    assert bi._derive_blend_path(S(save_blend=None)) is None
    assert bi._derive_blend_path(S(save_blend=False)) is None
    assert bi._derive_blend_path(S(save_blend="out/x.blend")) == "out/x.blend"
    assert bi._derive_blend_path(S(save_blend=True, render="out/img.png")) == "out/img.blend"
    assert bi._derive_blend_path(S(save_blend=True, bundle="/path/to/mine")) == "mine.blend"


def test_short_name(bi):
    assert bi._short("/a/b/c") == "a_b_c"
    assert bi._short("/") == "root"


def test_to_yaml_emits_parsable_yaml(bi):
    yaml = pytest.importorskip("yaml")
    doc = {
        "bundle": "/path/with spaces/b",
        "render_time_seconds": 1.25,
        "resolution": [1280, 720],
        "gpu": True,
        "look": None,
        "empty": {},
        "resolved_config": {"samples": 64, "engine": "EEVEE", "orbit": [1, 2]},
    }
    parsed = yaml.safe_load(bi._to_yaml(doc))
    assert parsed == doc


def test_to_yaml_quotes_strings_that_would_otherwise_be_typed(bi):
    yaml = pytest.importorskip("yaml")
    parsed = yaml.safe_load(bi._to_yaml({"a": "true", "b": "12", "c": "a: b"}))
    assert parsed == {"a": "true", "b": "12", "c": "a: b"}
