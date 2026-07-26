"""Config resolution for the ``visender`` wrapper. Runs in the *solver* env.

This module may import anything pip provides (it imports ``yaml``). It is never
imported by :mod:`visender.blender_import`, which runs inside Blender's
bundled Python. Its job is to turn a human-friendly ``pen_grip.yaml`` into a
flat, fully-resolved JSON dict whose keys match ``blender_import.Settings``
field names, which the Blender side then simply loads.

See ``docs/config.md`` for the full schema.
"""

from __future__ import annotations

import datetime as _dt
import difflib
import json
from pathlib import Path
from typing import Any

import yaml

# Top-level keys the schema recognises. Anything else is a hard error (with a
# "did you mean" suggestion) so a typo in a config file cannot silently do
# nothing.
TOP_LEVEL_KEYS = {
    "bundle", "blender", "output", "aliases", "profiles",
    "world", "camera", "film", "backdrop", "lighting", "materials", "save_blend",
    "animation",
}

# Recognised keys inside each nested section, validated the same way as the top
# level: a misspelled sub-key must be an error, not a silent no-op.
SECTION_KEYS: dict[str, set[str]] = {
    "world": {"studio", "strength", "hdri"},
    "camera": {"fit", "auto", "orbit", "dolly", "dof", "fstop", "scale"},
    "film": {"exposure", "look", "transparent"},
    "backdrop": {"enabled", "color", "shadow_catcher"},
    "lighting": {"sun_scale", "point_scale", "dim_authored", "three_point", "auto", "key"},
    "materials": {"library", "rules"},
    "animation": {"enabled", "start", "end", "step", "fps"},
}
_KEY_SECTION_KEYS = {"az", "el", "energy", "angle", "color"}

# Flat ``blender_import.Settings`` field names, which is the vocabulary a
# ``profiles:`` entry speaks. Duplicated here on purpose: this module must not
# import ``blender_import`` (that needs ``bpy``). Keep it in step when a field is
# added there; drift then shows up as a clear "unknown profile key" the first
# time a config uses the new field, instead of as a value the Blender side
# silently drops.
SETTINGS_FIELDS = {
    "bundle", "render", "samples", "resolution", "gpu", "engine", "hdri",
    "world_strength", "sun_scale", "point_scale", "keep_default_cube", "material",
    "studio_world", "backdrop", "backdrop_color", "exposure", "adaptive_threshold",
    "max_bounces", "key_light", "key_energy", "key_angle", "key_color",
    "dim_authored", "three_point", "auto_light", "fit", "auto_camera", "orbit",
    "dolly", "dof", "fstop", "scale", "save_blend", "transparent",
    "shadow_catcher", "look", "point_size", "point_color", "line_scale",
    "material_rules", "material_library",
    "animation", "frame_start", "frame_end", "frame_step", "fps",
}

# Path-valued keys are resolved relative to the config file's directory.
_PATH_KEYS = {"bundle", "blender", "output", "hdri"}

# Flat settings keys the Blender side parses out of an "R,G,B" string, so a
# profile may write them as a list and still work.
_FLAT_COLOR_KEYS = {"backdrop_color", "key_color", "point_color"}

# Flat settings keys holding a path, resolved relative to the config file.
_FLAT_PATH_KEYS = {"bundle", "hdri", "render"}

_MAX_ALIAS_DEPTH = 10


class ConfigError(SystemExit):
    """A user-facing config error (subclasses SystemExit so it exits cleanly)."""


def resolve(config_path: str | Path, *, profile: str | None = None,
            output: str | None = None) -> tuple[dict, str | None]:
    """Load and resolve a config file.

    Returns ``(resolved, blender_path)`` where ``resolved`` is a flat dict of
    ``blender_import.Settings`` fields (config file < selected profile, with CLI
    layered on afterwards by the caller), and ``blender_path`` is the configured
    Blender binary or None.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise ConfigError(f"config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: top level must be a mapping")

    _validate_keys(raw, TOP_LEVEL_KEYS, "top-level")
    for section, allowed in SECTION_KEYS.items():
        body = raw.get(section)
        if isinstance(body, dict):
            _validate_keys(body, allowed, f"{section}")
    key_section = (raw.get("lighting") or {}).get("key")
    if isinstance(key_section, dict):
        _validate_keys(key_section, _KEY_SECTION_KEYS, "lighting.key")

    base_dir = config_path.parent
    aliases = raw.get("aliases", {}) or {}

    flat: dict[str, Any] = {}

    # --- profile (config < profile) --------------------------------------- #
    profiles = raw.get("profiles", {}) or {}
    if profile is not None and profile not in profiles:
        raise ConfigError(f"unknown profile {profile!r}; have {sorted(profiles)}")
    prof = profiles.get(profile, {}) if profile else {}
    if not isinstance(prof, dict):
        raise ConfigError(f"profile {profile!r} must be a mapping")
    _validate_keys(prof, SETTINGS_FIELDS, f"profile {profile!r}")

    # --- flatten nested sections into Settings fields --------------------- #
    if "bundle" in raw:
        flat["bundle"] = _resolve_path(raw["bundle"], base_dir)

    world = raw.get("world", {}) or {}
    _put(flat, "studio_world", world.get("studio"))
    _put(flat, "world_strength", world.get("strength"))
    _put(flat, "hdri", _resolve_path(world.get("hdri"), base_dir) if world.get("hdri") else None)

    cam = raw.get("camera", {}) or {}
    _put(flat, "fit", cam.get("fit"))
    _put(flat, "auto_camera", cam.get("auto"))
    _put(flat, "orbit", cam.get("orbit"))
    _put(flat, "dolly", cam.get("dolly"))
    _put(flat, "dof", cam.get("dof"))
    _put(flat, "fstop", cam.get("fstop"))
    _put(flat, "scale", cam.get("scale"))

    film = raw.get("film", {}) or {}
    _put(flat, "exposure", film.get("exposure"))
    _put(flat, "look", film.get("look"))
    _put(flat, "transparent", film.get("transparent"))

    bd = raw.get("backdrop", {}) or {}
    _put(flat, "backdrop", bd.get("enabled"))
    if bd.get("color") is not None:
        flat["backdrop_color"] = _rgb_str(bd["color"])
    _put(flat, "shadow_catcher", bd.get("shadow_catcher"))

    lit = raw.get("lighting", {}) or {}
    _put(flat, "sun_scale", lit.get("sun_scale"))
    _put(flat, "point_scale", lit.get("point_scale"))
    _put(flat, "dim_authored", lit.get("dim_authored"))
    _put(flat, "three_point", lit.get("three_point"))
    _put(flat, "auto_light", lit.get("auto"))
    key = lit.get("key", {}) or {}
    if isinstance(key, dict) and key:
        if "az" in key and "el" in key:
            flat["key_light"] = f"{key['az']},{key['el']}"
        _put(flat, "key_energy", key.get("energy"))
        _put(flat, "key_angle", key.get("angle"))
        if key.get("color") is not None:
            flat["key_color"] = _rgb_str(key["color"])

    mats = raw.get("materials", {}) or {}
    if mats.get("library"):
        flat["material_library"] = mats["library"]
    if mats.get("rules"):
        flat["material_rules"] = _expand_rule_aliases(mats["rules"], aliases)

    anim = raw.get("animation", {}) or {}
    _put(flat, "animation", anim.get("enabled"))
    _put(flat, "frame_start", anim.get("start"))
    _put(flat, "frame_end", anim.get("end"))
    _put(flat, "frame_step", anim.get("step"))
    _put(flat, "fps", anim.get("fps"))

    _put(flat, "save_blend", raw.get("save_blend"))

    # --- apply profile overrides ------------------------------------------ #
    # Profiles speak flat Settings keys, but they still need the same coercion
    # the nested sections get: a colour written as a list has to reach Blender as
    # "R,G,B", and a path has to be relative to the config file, not the CWD.
    for k, v in prof.items():
        flat[k] = _coerce_flat(k, v, base_dir)

    # --- output path ------------------------------------------------------ #
    out_template = output or raw.get("output")
    if out_template:
        flat["render"] = _resolve_path(
            _expand_output(out_template, raw, profile, base_dir), base_dir)

    blender_path = raw.get("blender")
    if blender_path:
        blender_path = str(_resolve_path(blender_path, base_dir))

    return flat, blender_path


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _put(flat: dict, key: str, value: Any) -> None:
    if value is not None:
        flat[key] = value


def _coerce_flat(key: str, value: Any, base_dir: Path) -> Any:
    """Normalise a flat settings value written by a profile."""
    if key in _FLAT_COLOR_KEYS and value is not None and not isinstance(value, str):
        return _rgb_str(value)
    if key in _FLAT_PATH_KEYS and isinstance(value, str):
        return _resolve_path(value, base_dir)
    # ``save_blend`` is True | path; only the path form needs resolving.
    if key == "save_blend" and isinstance(value, str):
        return _resolve_path(value, base_dir)
    return value


def _validate_keys(mapping: dict, allowed: set[str], where: str) -> None:
    for key in mapping:
        if key not in allowed:
            hint = difflib.get_close_matches(key, allowed, n=1)
            suggestion = f" (did you mean {hint[0]!r}?)" if hint else ""
            raise ConfigError(f"unknown {where} key {key!r}{suggestion}")


def _resolve_path(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    p = Path(value)
    return str(p if p.is_absolute() else (base_dir / p))


def _rgb_str(color) -> str:
    if isinstance(color, str):
        return color
    return ",".join(str(int(c)) for c in color)


def _expand_aliases(text: str, aliases: dict) -> str:
    """Recursively expand ``{name}`` tokens, capped so a cycle errors."""
    for _ in range(_MAX_ALIAS_DEPTH):
        try:
            expanded = text.format_map(_AliasMap(aliases))
        except KeyError as exc:
            raise ConfigError(f"unknown alias {exc} in {text!r}")
        if expanded == text:
            return expanded
        text = expanded
    raise ConfigError(f"alias expansion did not terminate (cycle?) in {text!r}")


class _AliasMap(dict):
    """format_map source that turns an unknown ``{name}`` into a KeyError."""
    def __missing__(self, key):
        raise KeyError(repr(key))


def _expand_rule_aliases(rules: list, aliases: dict) -> list:
    out = []
    for rule in rules:
        rule = dict(rule)
        if "node" in rule and isinstance(rule["node"], str):
            rule["node"] = _expand_aliases(rule["node"], aliases)
        out.append(rule)
    return out


def _expand_output(template: str, raw: dict, profile: str | None, base_dir: Path) -> str:
    now = _dt.datetime.now()
    bundle = raw.get("bundle", "")
    return template.format(
        profile=profile or "default",
        bundle_name=Path(str(bundle)).name,
        date=now.strftime("%Y%m%d"),
        time=now.strftime("%H%M%S"),
    )


def scaffold(bundle: str | Path) -> str:
    """Return a starter YAML config for ``bundle`` (used by ``visender init``)."""
    bundle = Path(bundle)
    manifest = json.loads((bundle / "scene.json").read_text())
    lines = [
        f"bundle: {bundle}",
        "# blender: /opt/blender/blender-5.2.0-linux-x64/blender",
        f"output: renders/out/{bundle.name}_{{profile}}.png",
        "",
        "profiles:",
        "  draft: {engine: EEVEE, samples: 64, resolution: [1280, 720]}",
        "  final: {engine: CYCLES, samples: 4096, resolution: [3840, 2160], gpu: true}",
        "",
        "world: {studio: true, strength: 0.9}",
        "backdrop: {enabled: true, color: [205, 205, 210]}",
        "lighting: {dim_authored: 0.15, key: {az: -40, el: 35}}",
        "save_blend: true",
        "",
        "materials:",
        "  # library: {my_metal: {base_color: [158,163,168], metallic: 1.0, roughness: 0.22}}",
        "  rules:",
    ]
    for node in manifest["nodes"]:
        if node["kind"] in ("GlbProps", "MeshProps", "BoxProps", "IcosphereProps",
                            "BatchedMeshesProps"):
            lines.append(f"    # - {{node: {node['name']!r}, use: aluminium}}  # {node['kind']}")
    return "\n".join(lines) + "\n"


def load_manifest(bundle: str | Path) -> dict:
    return json.loads((Path(bundle) / "scene.json").read_text())
