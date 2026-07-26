"""Regression tests for the ``visender`` wrapper.

The wrapper's whole job is to build one Blender command line, so these tests
capture ``subprocess.run`` and assert on the command and the resolved JSON it
hands over rather than launching Blender.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from visender import cli


@pytest.fixture
def fake_run(monkeypatch):
    """Capture the Blender invocation instead of running it."""
    captured: dict = {}

    class _Result:
        returncode = 0

    def _run(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        idx = cmd.index("--config")
        captured["config"] = json.loads(Path(cmd[idx + 1]).read_text())
        return _Result()

    monkeypatch.setattr(cli.subprocess, "run", _run)
    return captured


@pytest.fixture
def cfg_file(tmp_path, make_bundle):
    bundle = make_bundle()

    def _write(body: str) -> Path:
        path = tmp_path / "scene.yaml"
        path.write_text(f"bundle: {bundle}\n" + body)
        return path

    _write.bundle = bundle
    return _write


# --------------------------------------------------------------------------- #
# Blender discovery
# --------------------------------------------------------------------------- #

def test_locate_blender_prefers_explicit_then_config_then_env(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    from_cfg = tmp_path / "cfg"
    from_env = tmp_path / "env"
    for p in (explicit, from_cfg, from_env):
        p.write_text("")
    monkeypatch.setenv("BLENDER", str(from_env))

    assert cli.locate_blender(str(explicit), str(from_cfg)) == str(explicit)
    assert cli.locate_blender(None, str(from_cfg)) == str(from_cfg)
    assert cli.locate_blender(None, None) == str(from_env)


def test_locate_blender_skips_paths_that_do_not_exist(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.write_text("")
    monkeypatch.delenv("BLENDER", raising=False)
    # A stale --blender path must fall through to the config value, not fail.
    assert cli.locate_blender(str(tmp_path / "ghost"), str(real)) == str(real)


def test_locate_blender_error_lists_what_it_tried(monkeypatch):
    monkeypatch.delenv("BLENDER", raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)
    monkeypatch.setattr(cli.glob, "glob", lambda _: [])
    with pytest.raises(SystemExit) as exc:
        cli.locate_blender(None, None)
    assert "could not locate Blender" in str(exc.value)


def test_locate_blender_globs_opt_newest_last(monkeypatch):
    monkeypatch.delenv("BLENDER", raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)
    monkeypatch.setattr(cli.glob, "glob", lambda _: [
        "/opt/blender/blender-4.2.0-linux-x64/blender",
        "/opt/blender/blender-5.2.0-linux-x64/blender",
    ])
    assert cli.locate_blender(None, None).startswith("/opt/blender/blender-5.2.0")


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #

def test_render_builds_the_blender_command(cfg_file, fake_run, monkeypatch, tmp_path):
    blender = tmp_path / "blender"
    blender.write_text("")
    rc = cli.cmd_render([str(cfg_file("world: {studio: true}\n")),
                         "--blender", str(blender)])
    assert rc == 0
    cmd = fake_run["cmd"]
    assert cmd[:3] == [str(blender), "-b", "--python"]
    assert cmd[3].endswith("blender_import.py")
    assert cmd[4] == "--"
    assert fake_run["config"]["studio_world"] is True
    assert fake_run["config"]["bundle"] == str(cfg_file.bundle)


def test_render_forwards_unknown_flags_to_blender_side(cfg_file, fake_run, tmp_path):
    blender = tmp_path / "blender"
    blender.write_text("")
    cli.cmd_render([str(cfg_file("")), "--blender", str(blender),
                    "--samples", "32", "--engine", "EEVEE"])
    cmd = fake_run["cmd"]
    # Passthrough flags land after --config so the Blender side sees them as
    # explicit and lets them win over the config JSON.
    assert cmd[-4:] == ["--samples", "32", "--engine", "EEVEE"]
    assert "samples" not in fake_run["config"]


def test_render_config_temp_file_is_cleaned_up(cfg_file, fake_run, tmp_path):
    blender = tmp_path / "blender"
    blender.write_text("")
    cli.cmd_render([str(cfg_file("")), "--blender", str(blender)])
    json_path = fake_run["cmd"][fake_run["cmd"].index("--config") + 1]
    assert not Path(json_path).exists()


def test_quality_shorthand_fills_only_unset_keys(cfg_file, fake_run, tmp_path):
    blender = tmp_path / "blender"
    blender.write_text("")
    cli.cmd_render([str(cfg_file("")), "--blender", str(blender), "--quality", "final"])
    assert fake_run["config"]["engine"] == "CYCLES"
    assert fake_run["config"]["samples"] == 8192
    assert fake_run["config"]["gpu"] is True


def test_quality_does_not_clobber_a_config_value(cfg_file, fake_run, tmp_path):
    blender = tmp_path / "blender"
    blender.write_text("")
    cli.cmd_render([str(cfg_file("camera: {scale: 25}\n")), "--blender", str(blender),
                    "--quality", "draft"])
    assert fake_run["config"]["scale"] == 25
    assert fake_run["config"]["engine"] == "EEVEE"


def test_quality_and_profile_together_is_an_error(cfg_file, tmp_path):
    path = cfg_file("profiles: {final: {samples: 16}}\n")
    with pytest.raises(SystemExit) as exc:
        cli.cmd_render([str(path), "--profile", "final", "--quality", "final"])
    assert "--quality" in str(exc.value)


def test_render_without_a_bundle_is_an_error(tmp_path):
    path = tmp_path / "scene.yaml"
    path.write_text("output: out.png\n")
    with pytest.raises(SystemExit) as exc:
        cli.cmd_render([str(path)])
    assert "no 'bundle:'" in str(exc.value)


def test_render_output_override_reaches_the_config(cfg_file, fake_run, tmp_path):
    blender = tmp_path / "blender"
    blender.write_text("")
    cli.cmd_render([str(cfg_file("output: out/config.png\n")), "--blender", str(blender),
                    "--output", "out/override.png"])
    assert fake_run["config"]["render"].endswith("out/override.png")


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #

def _blender(tmp_path):
    path = tmp_path / "blender"
    path.write_text("")
    return str(path)


def test_export_from_a_config_saves_a_blend_and_renders_nothing(
        cfg_file, fake_run, tmp_path):
    path = cfg_file("output: out/gundam.png\nworld: {studio: true}\n")
    assert cli.cmd_export([str(path), "--blender", _blender(tmp_path)]) == 0
    cfg = fake_run["config"]
    assert "render" not in cfg                          # nothing renders
    assert cfg["save_blend"].endswith("out/gundam.blend")   # named off the output
    assert cfg["studio_world"] is True                  # the shot's look survives


def test_export_output_override_wins_over_the_derived_name(cfg_file, fake_run, tmp_path):
    path = cfg_file("output: out/gundam.png\n")
    cli.cmd_export([str(path), "--blender", _blender(tmp_path),
                    "-o", str(tmp_path / "hero.blend")])
    assert fake_run["config"]["save_blend"] == str(tmp_path / "hero.blend")


def test_export_from_a_bare_bundle_names_the_blend_after_it(
        make_bundle, fake_run, tmp_path, monkeypatch):
    bundle = make_bundle()
    monkeypatch.chdir(tmp_path)
    cli.cmd_export([str(bundle), "--blender", _blender(tmp_path)])
    cfg = fake_run["config"]
    assert cfg["bundle"] == str(bundle)
    assert cfg["save_blend"] == str(tmp_path / f"{bundle.name}.blend")


def test_export_keys_a_recorded_bundle_by_default(make_bundle, fake_run, tmp_path):
    bundle = make_bundle()
    manifest = json.loads((bundle / "scene.json").read_text())
    manifest["animation"] = {"fps": 24, "frame_count": 2, "nodes": ["/mesh"]}
    (bundle / "scene.json").write_text(json.dumps(manifest))

    cli.cmd_export([str(bundle), "--blender", _blender(tmp_path),
                    "-o", str(tmp_path / "a.blend")])
    assert fake_run["config"]["animation"] is True

    cli.cmd_export([str(bundle), "--blender", _blender(tmp_path), "--still",
                    "-o", str(tmp_path / "a.blend")])
    assert fake_run["config"]["animation"] is False


def test_export_of_a_still_bundle_is_not_animated(make_bundle, fake_run, tmp_path):
    cli.cmd_export([str(make_bundle()), "--blender", _blender(tmp_path),
                    "-o", str(tmp_path / "a.blend")])
    assert fake_run["config"]["animation"] is False


def test_export_without_a_bundle_is_an_error(tmp_path):
    path = tmp_path / "scene.yaml"
    path.write_text("output: out.png\n")
    with pytest.raises(SystemExit) as exc:
        cli.cmd_export([str(path)])
    assert "no 'bundle:'" in str(exc.value)


# --------------------------------------------------------------------------- #
# list-nodes / init / dispatch
# --------------------------------------------------------------------------- #

def test_list_nodes_reports_every_node_and_sizes(make_bundle, capsys):
    bundle = make_bundle()
    assert cli.cmd_list_nodes([str(bundle)]) == 0
    out = capsys.readouterr().out
    for name in ("/mesh", "/group/box", "/cloud", "/lines", "/sun"):
        assert name in out
    assert "4 verts" in out   # tetrahedron
    assert "64 pts" in out    # point cloud


def test_node_count_is_blank_for_nodes_without_an_npz(make_bundle):
    bundle = make_bundle()
    assert cli._node_count(bundle, {"name": "/group/box", "kind": "BoxProps"}) == ""


def test_node_count_survives_a_corrupt_asset(make_bundle):
    bundle = make_bundle()
    (bundle / "assets" / "mesh.npz").write_bytes(b"not an npz")
    assert cli._node_count(bundle, {"asset": "assets/mesh.npz"}) == ""


def test_init_emits_a_config_for_the_bundle(make_bundle, capsys):
    bundle = make_bundle()
    assert cli.cmd_init([str(bundle)]) == 0
    out = capsys.readouterr().out
    assert f"bundle: {bundle}" in out
    assert "profiles:" in out


def test_main_dispatches_and_rejects_unknown_subcommands(capsys, make_bundle):
    assert cli.main(["list-nodes", str(make_bundle())]) == 0
    assert cli.main(["frobnicate"]) == 2
    assert "unknown subcommand" in capsys.readouterr().err


def test_main_with_no_args_exits_nonzero(capsys):
    assert cli.main([]) == 2
    assert cli.main(["--help"]) == 0


def test_quality_table_keys_are_settings_fields():
    """A shorthand that names a non-field would be silently dropped."""
    from visender import config as cfg
    for name, table in cli.QUALITY.items():
        assert set(table) <= cfg.SETTINGS_FIELDS, name
