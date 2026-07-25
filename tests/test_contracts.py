"""Structural invariants that hold the two interpreters together.

``blender_import`` runs inside Blender's bundled Python; ``config``/``cli`` run
in the solver env. They never import each other, so nothing but these checks
stops the two halves from drifting apart.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from bliser import cli, config as cfg

REPO = Path(__file__).resolve().parents[1]
BLENDER_IMPORT_PY = REPO / "bliser" / "blender_import.py"

# Blender's Python ships the stdlib plus numpy/bpy/mathutils and nothing else.
ALLOWED_BLENDER_SIDE_IMPORTS = {
    "bpy", "mathutils", "numpy", "np",
    "argparse", "dataclasses", "datetime", "json", "math", "re", "sys", "time",
    "pathlib", "typing", "__future__", "os", "collections", "itertools",
    "functools", "textwrap", "shutil", "glob",
    # The provenance sidecar reads __version__ from the package if it happens to
    # be on Blender's path; the import is guarded and must stay that way.
    "bliser",
}


def blender_side_tree() -> ast.Module:
    return ast.parse(BLENDER_IMPORT_PY.read_text())


def test_blender_side_imports_nothing_pip_only():
    """A stray ``import yaml`` here would break every render, not a test."""
    offenders = []
    for node in ast.walk(blender_side_tree()):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        offenders += [n for n in names if n and n not in ALLOWED_BLENDER_SIDE_IMPORTS]
    assert not offenders, f"blender_import may not import {sorted(set(offenders))}"


def test_blender_side_never_imports_the_solver_half():
    text = BLENDER_IMPORT_PY.read_text()
    assert "from bliser import config" not in text
    assert "import yaml" not in text


def settings_field_names() -> set[str]:
    """Field names of ``blender_import.Settings``, read without importing bpy."""
    for node in blender_side_tree().body:
        if isinstance(node, ast.ClassDef) and node.name == "Settings":
            return {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
                and not stmt.target.id.startswith("_")
            }
    raise AssertionError("Settings dataclass not found")


def test_settings_fields_match_the_config_side_copy():
    """``config.SETTINGS_FIELDS`` duplicates the dataclass on purpose; drift
    means a config key is silently dropped on the Blender side."""
    assert cfg.SETTINGS_FIELDS == settings_field_names()


def argparse_dests() -> set[str]:
    """Every ``dest`` ``_add_arguments`` creates, derived from the flag names."""
    dests = set()
    for node in ast.walk(blender_side_tree()):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        explicit = next((kw.value.value for kw in node.keywords if kw.arg == "dest"), None)
        if explicit:
            dests.add(explicit)
            continue
        flag = node.args[0].value
        dests.add(flag.lstrip("-").replace("-", "_"))
    return dests


def test_every_settings_field_is_reachable_from_the_command_line_or_config():
    """A field with neither a flag nor a config path can never be set."""
    config_only = {"material_rules", "material_library"}
    unreachable = settings_field_names() - argparse_dests() - config_only
    assert not unreachable, f"no way to set {sorted(unreachable)}"


def test_every_cli_flag_lands_on_a_settings_field():
    """A flag whose dest is not a field is parsed and then thrown away."""
    non_settings = {"config", "list_nodes"}  # control flow, not render settings
    stray = argparse_dests() - settings_field_names() - non_settings
    assert not stray, f"flags that reach no Settings field: {sorted(stray)}"


def test_config_sections_only_produce_known_settings_keys(tmp_path):
    """Exercise every schema key at once and check the flat output stays legal."""
    # A plausible value per key shape: paths and named specs are strings, rule
    # containers are empty, everything else is a number.
    special = {
        "key": "{az: 0, el: 0}",
        "hdri": "some.exr",
        "look": "AgX",
        "fit": "fit_all",
        "library": "{}",
        "rules": "[]",
        "color": "[1, 2, 3]",
    }
    body = ["bundle: b", "output: o.png", "save_blend: true"]
    for section, keys in cfg.SECTION_KEYS.items():
        entries = [f"{key}: {special.get(key, 1)}" for key in sorted(keys)]
        body.append(f"{section}: {{{', '.join(entries)}}}")
    path = tmp_path / "all.yaml"
    path.write_text("\n".join(body) + "\n")
    flat, _ = cfg.resolve(path)
    assert set(flat) <= cfg.SETTINGS_FIELDS


def test_section_keys_cover_every_documented_top_level_section():
    structural = {"bundle", "blender", "output", "aliases", "profiles", "save_blend"}
    assert set(cfg.SECTION_KEYS) | structural == cfg.TOP_LEVEL_KEYS


def test_docs_mention_every_top_level_key():
    doc = (REPO / "docs" / "config.md")
    if not doc.exists():
        pytest.skip("docs/config.md not present")
    text = doc.read_text()
    missing = [k for k in cfg.TOP_LEVEL_KEYS if k not in text]
    assert not missing, f"undocumented config keys: {sorted(missing)}"


def test_pyproject_keeps_pyyaml_out_of_the_base_dependencies():
    """Blender's Python has no yaml; making it a base dep breaks installs."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover
        tomllib = pytest.importorskip("tomli")
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    base = " ".join(data["project"]["dependencies"]).lower()
    assert "yaml" not in base
    assert "yaml" in " ".join(data["project"]["optional-dependencies"]["cli"]).lower()
    assert data["project"]["scripts"]["bliser"] == "bliser.cli:main"


def test_cli_locates_blender_import_next_to_itself():
    assert Path(cli._blender_import_path()) == BLENDER_IMPORT_PY
    assert BLENDER_IMPORT_PY.exists()
