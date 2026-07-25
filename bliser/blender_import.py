"""Rebuild a viser scene bundle inside Blender. Runs in Blender's Python.

    blender --python blender_import.py -- --bundle my_scene.viserbundle
    blender -b --python blender_import.py -- --bundle my_scene.viserbundle \
            --render cover.png --samples 256

    # or, driven by a resolved JSON config written by the ``bliser`` wrapper:
    blender -b --python blender_import.py -- --config /tmp/resolved.json

This module deliberately imports nothing from the rest of the package, and
nothing outside Blender's bundled interpreter: only ``bpy``, ``numpy``,
``mathutils`` and the standard library. Blender's Python has no ``viser``, no
``yaml``, no ``pydantic``. The two halves of bliser communicate through
files on disk (the bundle, and an optional resolved-config JSON), never through
a shared interpreter.

Axis note: viser hands GLB bytes to three.js untouched, and trimesh writes them
in the scene's own Z-up frame. Blender's glTF importer always rotates +Y-up to
+Z-up on the way in, so every imported payload is parented under an empty that
undoes that rotation. Without it, meshes land on their side.

Extending the pipeline: subclass :class:`Pipeline` and override any of
``after_build``, ``before_world`` or ``before_render``, or pass callables of the
same name to :meth:`Pipeline.from_argv`. Each hook can reach ``self.created``, a
map from viser node path to the Blender objects built for it.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

# The bundle format version this importer understands. Bundles carry their own
# ``version``; we fail on anything newer and warn on anything older.
SUPPORTED_BUNDLE_VERSION = 1

# Undoes the glTF importer's mandatory +Y-up -> +Z-up conversion, which it bakes
# into vertex data (not object matrices). Measured against Blender 5.2: the
# importer maps a source vertex (x, y, z) to (x, -z, y), i.e. it bakes a +90 deg
# rotation about X, so we rotate by -90 deg to send it back to viser's Z-up frame.
GLTF_UNROTATE = Matrix.Rotation(math.radians(-90.0), 4, "X")

# Name of the object ``make_backdrop`` creates. Framing and bounds queries skip
# it by name, so keep the two in step.
BACKDROP_NAME = "backdrop"


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

# Sentinel marking "the user did not pass this flag", so a resolved-config JSON
# can fill it in without clobbering an explicit command-line override.
_UNSET = object()


@dataclasses.dataclass
class Settings:
    """The single resolved settings object every stage reads from.

    Attribute names match the argparse ``dest`` names one-to-one, so existing
    call sites and scene drivers that pass an ``argparse.Namespace`` keep
    working unchanged -- a ``Settings`` is a drop-in for ``args``.
    """

    bundle: str = ""
    render: str | None = None
    samples: int = 128
    resolution: tuple[int, int] | None = None
    gpu: bool = False
    engine: str = "CYCLES"
    hdri: str | None = None
    world_strength: float = 1.0
    sun_scale: float = 1.0
    point_scale: float = 4.0 * math.pi
    keep_default_cube: bool = False
    material: list[str] = dataclasses.field(default_factory=list)
    studio_world: bool = False
    backdrop: bool = False
    backdrop_color: str = "200,200,205"
    exposure: float = 0.0
    adaptive_threshold: float = 0.005
    max_bounces: int | None = None

    # Phase 3 -- lighting
    key_light: str | None = None          # "AZ,EL" camera-relative degrees
    key_energy: float | None = None
    key_angle: float | None = None        # soft-shadow angle, degrees
    key_color: str | None = None          # "R,G,B" 0-255
    dim_authored: float | None = None
    three_point: bool = False
    auto_light: bool = False

    # Phase 5 -- camera / framing
    fit: str = "keep_vertical"
    auto_camera: bool = False
    orbit: tuple[float, float] | None = None
    dolly: float = 1.0
    dof: Any = None                        # True | float distance | None
    fstop: float = 2.8
    scale: int = 100

    # Phase 6 -- output
    save_blend: Any = None                 # True | path | None
    transparent: bool = False
    shadow_catcher: bool = False
    look: str | None = None

    # Phase 7 -- point clouds / splines
    point_size: float | None = None
    point_color: str | None = None
    line_scale: float = 1.0

    # Config-only surfaces (no CLI flag): richer material rules and library.
    material_rules: list[dict] = dataclasses.field(default_factory=list)
    material_library: dict = dataclasses.field(default_factory=dict)

    # Bookkeeping for the provenance sidecar.
    _resolved_config: dict | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace, explicit: set[str], config: dict | None) -> "Settings":
        """Fold CLI args and an optional resolved-config dict into settings.

        Precedence inside the Blender process: an explicitly passed CLI flag
        wins over a config value, which wins over the argparse default.
        """
        s = cls()
        field_names = {f.name for f in dataclasses.fields(cls)}

        # 1. argparse defaults / explicit values for everything on the Namespace.
        for name in field_names:
            if hasattr(args, name):
                setattr(s, name, getattr(args, name))

        # 2. Config fills anything the user did not pass explicitly.
        if config:
            s._resolved_config = config
            for key, value in config.items():
                if key in field_names and key not in explicit:
                    setattr(s, key, value)
            # material rules / library live only in config.
            s.material_rules = list(config.get("material_rules", s.material_rules))
            s.material_library = dict(config.get("material_library", s.material_library))

        # Normalise a couple of container types that JSON hands back as lists.
        if isinstance(s.resolution, list):
            s.resolution = tuple(s.resolution)
        if isinstance(s.orbit, list):
            s.orbit = tuple(s.orbit)
        return s


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def _add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bundle", help="Bundle directory from export_scene")
    p.add_argument("--config", default=None, help="Resolved-JSON config (written by bliser). "
                   "Values fill in any flag not passed explicitly on the command line.")
    p.add_argument("--list-nodes", action="store_true",
                   help="Print each node's path/kind/size from scene.json and exit.")
    p.add_argument("--render", default=None, help="Render straight to this image path")
    p.add_argument("--samples", type=int, default=128, help="Cycles samples")
    p.add_argument("--resolution", type=int, nargs=2, metavar=("W", "H"), default=None,
                   help="Override the resolution (defaults to the exporting browser window). "
                   "Blender's own -x/-y are ignored, since the camera is built afterwards.")
    p.add_argument("--gpu", action="store_true",
                   help="Render Cycles on the GPU. Background renders are CPU-only otherwise, "
                   "even when the GUI preferences select a GPU.")
    p.add_argument("--engine", default="CYCLES",
                   help="CYCLES (default) or EEVEE; resolved against this Blender's engine list")
    p.add_argument("--hdri", default=None, help=".exr/.hdr for the world background")
    p.add_argument("--world-strength", type=float, default=1.0, help="World lighting strength")
    p.add_argument("--sun-scale", type=float, default=1.0,
                   help="viser directional intensity -> Blender sun W/m^2")
    p.add_argument("--point-scale", type=float, default=4.0 * math.pi,
                   help="viser point/spot intensity (candela) -> Blender watts")
    p.add_argument("--keep-default-cube", action="store_true")
    p.add_argument("--material", action="append", default=[],
                   metavar="NODE=SPEC",
                   help="Override the material of an imported node (by its viser /path). "
                   "Positional form: NODE=R,G,B[,A][,ROUGHNESS] (0-255 sRGB colour). "
                   "Key=value form for full PBR: "
                   "'/table=base_color:158,163,168;metallic:1.0;roughness:0.22'. "
                   "A named material may be used directly: '/table=use:aluminium'. "
                   "Repeatable. A trailing '/*' matches a whole subtree.")
    p.add_argument("--studio-world", action="store_true",
                   help="Light the scene with a soft procedural vertical gradient world "
                   "(no file needed) instead of a flat colour. Ignored when --hdri is given.")
    p.add_argument("--backdrop", action="store_true",
                   help="Add a large neutral ground plane just under the scene so shadows land "
                   "on a floor and the frame reads like a studio shot.")
    p.add_argument("--backdrop-color", type=str, default="200,200,205",
                   help="R,G,B (0-255 sRGB) for the --backdrop floor. Default a light grey.")
    p.add_argument("--exposure", type=float, default=0.0,
                   help="Stops of exposure applied in colour management (+ brighter).")
    p.add_argument("--adaptive-threshold", type=float, default=0.005,
                   help="Cycles adaptive-sampling noise threshold; lower is cleaner and "
                   "slower (e.g. 0.001 for a final hero frame). Ignored under EEVEE.")
    p.add_argument("--max-bounces", type=int, default=None,
                   help="Cycles total light bounces (higher = more accurate GI, slower). "
                   "Ignored under EEVEE.")

    # Phase 3 -- lighting -------------------------------------------------- #
    p.add_argument("--key-light", default=None, metavar="AZ,EL",
                   help="Add a soft key light in camera space. Azimuth/elevation in degrees: "
                   "az 0 = behind camera, +az = toward frame-right, +el = above.")
    p.add_argument("--key-energy", type=float, default=None, help="Key light energy (watts).")
    p.add_argument("--key-angle", type=float, default=None,
                   help="Key light angular size in degrees (wide = soft shadows).")
    p.add_argument("--key-color", default=None, metavar="R,G,B", help="Key light colour, 0-255 sRGB.")
    p.add_argument("--dim-authored", type=float, default=None, metavar="F",
                   help="Scale every imported light's energy by F (e.g. 0.15 to knock back "
                   "the scene's own suns before adding a key).")
    p.add_argument("--three-point", action="store_true",
                   help="Add a camera-relative key+fill+rim rig.")
    p.add_argument("--auto-light", action="store_true",
                   help="Add a three-point rig ONLY if the bundle contains no lights at all "
                   "(otherwise the scene renders near-black under Cycles).")

    # Phase 5 -- camera / framing ------------------------------------------ #
    p.add_argument("--fit", choices=("keep_vertical", "keep_horizontal", "fit_all"),
                   default="keep_vertical",
                   help="Aspect-fit policy when the render aspect differs from the bundle "
                   "camera's. keep_vertical (default) preserves vertical FOV (may crop "
                   "horizontally); keep_horizontal preserves horizontal FOV; fit_all widens "
                   "FOV so nothing composed is lost (letterboxes).")
    p.add_argument("--auto-camera", action="store_true",
                   help="If the bundle has no camera, frame the scene bounds in a 3/4 view "
                   "instead of rendering Blender's default camera pointed at nothing.")
    p.add_argument("--orbit", type=float, nargs=2, metavar=("AZ", "EL"), default=None,
                   help="Rotate the camera about its look-at point by AZ/EL degrees.")
    p.add_argument("--dolly", type=float, default=1.0,
                   help="Scale the camera distance from its look-at point (1.0 = unchanged).")
    p.add_argument("--dof", nargs="?", const=True, default=None, metavar="DISTANCE",
                   help="Enable depth of field. Optional focus DISTANCE in metres; defaults "
                   "to |look_at - position| from the manifest.")
    p.add_argument("--fstop", type=float, default=2.8, help="Depth-of-field f-number.")
    p.add_argument("--scale", type=int, default=100,
                   help="resolution_percentage: render at this %% of the composed resolution "
                   "(same composition/aspect). 50 halves both axes for a quick preview.")

    # Phase 6 -- output ---------------------------------------------------- #
    p.add_argument("--save-blend", nargs="?", const=True, default=None, metavar="PATH",
                   help="Save a .blend after building (before the render, so it survives a "
                   "render crash). With no PATH, derived from the output image path.")
    p.add_argument("--transparent", action="store_true",
                   help="Render with a transparent film (RGBA output).")
    p.add_argument("--shadow-catcher", action="store_true",
                   help="Make the --backdrop plane a shadow catcher (Cycles only).")
    p.add_argument("--look", default=None,
                   help="Colour-management look, e.g. 'AGX - Punchy'. Validated against the "
                   "live enum.")

    # Phase 7 -- point clouds / splines ------------------------------------ #
    p.add_argument("--point-size", type=float, default=None,
                   help="Point-cloud instance radius in metres (default derived from bounds).")
    p.add_argument("--point-color", default=None, metavar="R,G,B",
                   help="Force a flat colour on all point clouds instead of per-point colour.")
    p.add_argument("--line-scale", type=float, default=1.0,
                   help="Scale the spline/line tube radius (radius = line_width*0.00025*scale).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="blender_import")
    _add_arguments(p)
    return p


def parse_args(argv: list[str]) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _explicit_flags(argv: list[str]) -> set[str]:
    """Which argparse dests were actually present on the command line.

    Parsing a second time with every default suppressed leaves only the
    user-supplied options on the namespace, so config values can fill the rest.
    Per-argument ``default=`` values override a parser-level SUPPRESS, so we
    stamp SUPPRESS onto every action explicitly.
    """
    sentinel = build_parser()
    for action in sentinel._actions:
        action.default = argparse.SUPPRESS
    ns, _ = sentinel.parse_known_args(argv)
    return set(vars(ns).keys())


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- #
# Colour helpers
# --------------------------------------------------------------------------- #

def _srgb_to_linear(color) -> tuple[float, float, float]:
    return tuple((c / 255.0) ** 2.2 for c in color)


def _parse_rgb(spec: str) -> tuple[float, float, float]:
    parts = [int(x) for x in spec.split(",")]
    if len(parts) != 3:
        raise SystemExit(f"expected R,G,B (got {spec!r})")
    return tuple(parts)


# --------------------------------------------------------------------------- #
# Materials (Phase 4)
# --------------------------------------------------------------------------- #

# Built-in named materials; a config's ``materials.library`` may add to or
# override these by name. Colours are 0-255 sRGB, everything else 0-1 unless
# noted.
BUILTIN_LIBRARY: dict[str, dict] = {
    "aluminium":     {"base_color": (158, 163, 168), "metallic": 1.0, "roughness": 0.22},
    "matte_black":   {"base_color": (18, 18, 20), "metallic": 0.0, "roughness": 0.8},
    "plastic_white": {"base_color": (235, 235, 235), "metallic": 0.0, "roughness": 0.45},
    "glass":         {"base_color": (255, 255, 255), "metallic": 0.0, "roughness": 0.0,
                      "transmission": 1.0, "ior": 1.45, "opacity": 1.0},
    "rubber":        {"base_color": (25, 25, 28), "metallic": 0.0, "roughness": 0.95},
    "copper":        {"base_color": (184, 115, 84), "metallic": 1.0, "roughness": 0.3},
}

# Datablock cache so a named library material is created once and shared across
# every object that references it, rather than one material per object.
_LIBRARY_CACHE: dict[str, Any] = {}


def reset_material_cache() -> None:
    _LIBRARY_CACHE.clear()


def _set_bsdf_inputs(bsdf, props: dict) -> None:
    """Apply a full-PBR props dict to a Principled BSDF node."""
    color = props.get("base_color") or props.get("color") or (200, 200, 200)
    rgb = _srgb_to_linear(color)
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)

    opacity = props.get("opacity")
    alpha = 1.0 if opacity is None else float(opacity)
    bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Roughness"].default_value = float(props.get("roughness", 0.5))

    def _maybe(name: str, key: str) -> None:
        if key in props and name in bsdf.inputs:
            bsdf.inputs[name].default_value = float(props[key])

    _maybe("Metallic", "metallic")
    _maybe("IOR", "ior")
    _maybe("Transmission Weight", "transmission")
    # "Specular" was renamed "Specular IOR Level" in 4.x; try both.
    for spec_name in ("Specular IOR Level", "Specular"):
        if "specular" in props and spec_name in bsdf.inputs:
            bsdf.inputs[spec_name].default_value = float(props["specular"])
            break

    if "emission_color" in props:
        ec = _srgb_to_linear(props["emission_color"])
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (*ec, 1.0)
    if "emission_strength" in props and "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = float(props["emission_strength"])


def _set_blend_method(mat, alpha: float) -> None:
    """Set alpha blending in a version-tolerant way.

    ``blend_method`` was reworked in 4.2 and changed again later, so only touch
    it if the attribute exists on this build (5.2 dropped it for EEVEE Next but
    still exposes it on some configs).
    """
    if alpha < 1.0 and hasattr(mat, "blend_method"):
        try:
            mat.blend_method = "BLEND"
        except (TypeError, AttributeError):
            pass


def make_material(name: str, props: dict):
    """Principled BSDF from a (possibly full-PBR) props dict."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    _set_bsdf_inputs(bsdf, props)
    opacity = props.get("opacity")
    _set_blend_method(mat, 1.0 if opacity is None else float(opacity))
    return mat


def resolve_named_material(name: str, library: dict):
    """Return the shared datablock for a named library material, creating once."""
    if name in _LIBRARY_CACHE:
        return _LIBRARY_CACHE[name]
    props = library.get(name, BUILTIN_LIBRARY.get(name))
    if props is None:
        known = sorted(set(BUILTIN_LIBRARY) | set(library))
        raise SystemExit(f"unknown material {name!r}; known: {known}")
    mat = make_material(name, props)
    _LIBRARY_CACHE[name] = mat
    return mat


def material_from_spec(name: str, spec: dict, library: dict):
    """A spec is either ``{'use': 'aluminium'}`` or a raw PBR props dict."""
    if "use" in spec:
        return resolve_named_material(spec["use"], library)
    return make_material(name, spec)


def apply_shading(obj, props: dict, mat) -> None:
    if mat is not None and hasattr(obj.data, "materials"):
        obj.data.materials.append(mat)
    obj.visible_shadow = bool(props.get("cast_shadow", True))
    if props.get("flat_shading"):
        for poly in obj.data.polygons:
            poly.use_smooth = False
    elif hasattr(obj.data, "polygons"):
        for poly in obj.data.polygons:
            poly.use_smooth = True


def parse_material_overrides(specs: list[str]) -> list[tuple[str, bool, dict]]:
    """Turn ``--material`` strings into ``(path, subtree, props)`` rules.

    Positional form (unchanged, scripts depend on it)::

        '/table=30,90,200'      -> flat blue on exactly /table
        '/pen/*=30,30,40'       -> the same colour on every node under /pen

    Key=value form (full PBR)::

        '/table=base_color:158,163,168;metallic:1.0;roughness:0.22'
        '/table=use:aluminium'  -> a named library material
    """
    rules: list[tuple[str, bool, dict]] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--material needs NODE=SPEC (got {spec!r})")
        path, _, rest = spec.partition("=")
        subtree = path.endswith("/*")
        path = path[:-2] if subtree else path
        props = _parse_material_value(rest, spec)
        rules.append((path.rstrip("/") or "/", subtree, props))
    return rules


# Which PBR keys are scalars (0-1) vs colours (R,G,B).
_COLOR_KEYS = {"base_color", "color", "emission_color"}


def _parse_material_value(rest: str, spec: str) -> dict:
    if ":" in rest:  # key=value form
        props: dict = {}
        for pair in rest.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            key, _, val = pair.partition(":")
            key = key.strip()
            if key == "use":
                props["use"] = val.strip()
            elif key in _COLOR_KEYS:
                props[key] = tuple(int(x) for x in val.split(","))
            else:
                props[key] = float(val)
        return props
    # positional R,G,B[,A][,ROUGH]
    nums = [float(x) for x in rest.split(",") if x.strip() != ""]
    if len(nums) < 3:
        raise SystemExit(f"--material needs at least R,G,B (got {spec!r})")
    props = {"color": tuple(nums[:3])}
    if len(nums) >= 4:
        props["opacity"] = nums[3]
    if len(nums) >= 5:
        props["roughness"] = nums[4]
    return props


def _override_for(name: str, rules) -> dict | None:
    """The last matching rule's props for ``name`` (later rules win)."""
    match = None
    for path, subtree, props in rules:
        if name == path or (subtree and (name == path or name.startswith(path + "/"))):
            match = props
    return match


def apply_material_override(objs, node_name: str, props: dict, library: dict) -> None:
    """Replace the material on every mesh in ``objs`` with an override.

    A ``{'use': name}`` spec shares the named datablock; anything else builds a
    one-off material for this node.
    """
    if "use" in props:
        mat = resolve_named_material(props["use"], library)
    else:
        mat = make_material(f"{_short(node_name)}_override", props)
    for obj in objs:
        if obj.type == "MESH" and hasattr(obj.data, "materials"):
            obj.data.materials.clear()
            obj.data.materials.append(mat)


# ---- geometric split selectors (Phase 4.3) --------------------------------- #

_CMP = {
    ">": lambda a, b: a > b, "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
}


def _cmp_pred(spec: str) -> Callable[[float], bool]:
    """Compile a comparison string like ``'>0.9'`` or ``'<=-0.5'``."""
    m = re.match(r"\s*(>=|<=|>|<)\s*(-?[0-9.]+)\s*$", spec)
    if not m:
        raise SystemExit(f"bad comparison {spec!r} (want e.g. '>0.9')")
    op, val = m.group(1), float(m.group(2))
    return lambda x: _CMP[op](x, val)


_WHERE_KEYS = {"normal_x", "normal_y", "normal_z", "world_x", "world_y", "world_z",
               "area", "material_name"}


def _compile_where(where: dict) -> Callable[[Vector, Vector, float, str], bool]:
    """Compile a ``where`` predicate dict into one callable.

    Compiled once per rule rather than per polygon: a hero mesh can carry tens
    of thousands of faces, and re-parsing ``'>0.9'`` for each of them dominated
    the split.
    """
    unknown = set(where) - _WHERE_KEYS
    if unknown:
        raise SystemExit(f"unknown split predicate(s) {sorted(unknown)}; "
                         f"known: {sorted(_WHERE_KEYS)}")

    tests: list[Callable[[Vector, Vector, float, str], bool]] = []
    for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
        if f"normal_{axis}" in where:
            pred, i = _cmp_pred(where[f"normal_{axis}"]), idx
            tests.append(lambda c, n, a, m, pred=pred, i=i: pred(n[i]))
        if f"world_{axis}" in where:
            lo, hi = where[f"world_{axis}"]
            tests.append(lambda c, n, a, m, lo=lo, hi=hi, i=idx: lo <= c[i] <= hi)
    if "area" in where:
        lo, hi = where["area"]
        tests.append(lambda c, n, a, m, lo=lo, hi=hi: lo <= a <= hi)
    if "material_name" in where:
        rx = re.compile(where["material_name"])
        tests.append(lambda c, n, a, m, rx=rx: bool(rx.search(m or "")))

    def matches(centre: Vector, normal: Vector, area: float, mat_name: str) -> bool:
        return all(t(centre, normal, area, mat_name) for t in tests)

    return matches


def apply_split(objs, rules: list[dict], library: dict, node_name: str) -> None:
    """Assign per-polygon materials on merged meshes from geometry predicates.

    Each rule is ``{'where': {...}, 'use': name}`` or ``{'default': True,
    'use': name}``. Predicates evaluate in world space per polygon (centre and
    matrix_world-transformed normal). Rules apply in order; last match wins;
    ``default`` catches the remainder.
    """
    matchers = [None if rule.get("default") else _compile_where(rule.get("where", {}))
                for rule in rules]

    for obj in objs:
        if obj.type != "MESH" or not hasattr(obj.data, "polygons"):
            continue
        mesh = obj.data

        # Snapshot the imported material names *before* appending our own slots,
        # so a ``material_name`` predicate can never match a split material this
        # call just created (which is what a slot-less mesh would otherwise hit).
        original_names = [m.name if m else "" for m in mesh.materials]

        # One material slot per distinct rule, appended once.
        slot_index: dict[int, int] = {}
        for i, rule in enumerate(rules):
            mat = material_from_spec(f"{_short(node_name)}_split{i}",
                                     {"use": rule["use"]} if isinstance(rule.get("use"), str)
                                     else rule.get("use", {}), library)
            mesh.materials.append(mat)
            slot_index[i] = len(mesh.materials) - 1

        default_slot = next((slot_index[i] for i, r in enumerate(rules) if r.get("default")), None)
        m3 = obj.matrix_world.to_3x3()
        counts = [0] * len(rules)
        for poly in mesh.polygons:
            centre = obj.matrix_world @ poly.center
            normal = (m3 @ poly.normal).normalized()
            existing = (original_names[poly.material_index]
                        if poly.material_index < len(original_names) else "")
            chosen = default_slot
            chosen_rule = None
            for i, matcher in enumerate(matchers):
                if matcher is None:  # the default rule
                    continue
                if matcher(centre, normal, poly.area, existing):
                    chosen = slot_index[i]
                    chosen_rule = i
            if chosen is not None:
                poly.material_index = chosen
            if chosen_rule is not None:
                counts[chosen_rule] += 1
            elif chosen == default_slot and default_slot is not None:
                for i, r in enumerate(rules):
                    if r.get("default"):
                        counts[i] += 1
        summary = ", ".join(
            f"rule[{i}]{'(default)' if rules[i].get('default') else ''}={counts[i]}"
            for i in range(len(rules)))
        print(f"[bliser] split {node_name}: {summary}")


# --------------------------------------------------------------------------- #
# Geometry builders
# --------------------------------------------------------------------------- #

def clear_scene(keep_cube: bool) -> None:
    reset_material_cache()
    if keep_cube:
        return
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def add_mesh_object(name: str, verts, faces, matrix: Matrix, props: dict):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.matrix_world = matrix
    apply_shading(obj, props, make_material(f"{name}_mat", props))
    return obj


def import_glb(node: dict, bundle: Path, matrix: Matrix):
    path = bundle / node["asset"]
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [o for o in bpy.data.objects if o not in before]
    roots = [o for o in imported if o.parent is None]

    scale = float(node["props"].get("scale", 1.0))
    holder = bpy.data.objects.new(_short(node["name"]), None)
    holder.empty_display_size = 0.05
    bpy.context.collection.objects.link(holder)
    holder.matrix_world = matrix @ GLTF_UNROTATE @ Matrix.Scale(scale, 4)

    for root in roots:
        local = root.matrix_basis.copy()
        root.parent = holder
        root.matrix_parent_inverse = Matrix.Identity(4)
        root.matrix_basis = local
    # matrix_world is cached; without this the new parenting is not reflected
    # until the next depsgraph evaluation (and anything we read back is stale).
    bpy.context.view_layer.update()

    cast = bool(node["props"].get("cast_shadow", True))
    for obj in imported:
        obj.visible_shadow = cast
    return holder


def add_light(node: dict, matrix: Matrix, settings) -> None:
    kind = node["kind"]
    props = node["props"]
    name = _short(node["name"])
    intensity = float(props.get("intensity", 1.0))

    if kind == "DirectionalLightProps":
        light = bpy.data.lights.new(name, type="SUN")
        light.energy = intensity * settings.sun_scale
        light.angle = math.radians(1.0)  # crisp shadows, like three.js
    elif kind == "PointLightProps":
        light = bpy.data.lights.new(name, type="POINT")
        light.energy = intensity * settings.point_scale
        light.shadow_soft_size = 0.02
        distance = float(props.get("distance", 0.0))
        if distance > 0.0:
            light.use_custom_distance = True
            light.cutoff_distance = distance
    elif kind == "SpotLightProps":
        light = bpy.data.lights.new(name, type="SPOT")
        light.energy = intensity * settings.point_scale
        light.spot_size = 2.0 * float(props.get("angle", 0.5))
        light.spot_blend = float(props.get("penumbra", 0.0))
    elif kind == "RectAreaLightProps":
        light = bpy.data.lights.new(name, type="AREA")
        light.shape = "RECTANGLE"
        light.size = float(props.get("width", 1.0))
        light.size_y = float(props.get("height", 1.0))
        light.energy = intensity * settings.point_scale
    else:
        return  # Ambient/hemisphere are folded into the world, not objects.

    color = props.get("color") or (255, 255, 255)
    light.color = _srgb_to_linear(color)
    if hasattr(light, "use_shadow"):
        light.use_shadow = bool(props.get("cast_shadow", True))

    obj = bpy.data.objects.new(name, light)
    bpy.context.collection.objects.link(obj)
    # Both three.js and Blender aim lights down local -Z, so the pose transfers
    # with no correction.
    obj.matrix_world = matrix


def add_curve(node: dict, bundle: Path, matrix: Matrix, settings) -> list:
    """Splines and line segments become beveled curves, so they render as
    tubes with real thickness instead of viser's screen-space lines.

    Per-segment colours are preserved by emitting one curve object per
    contiguous run of equal colour -- a trajectory coloured by time or cost is
    not flattened to one hue. (Legacy Blender curves carry no point-colour
    attribute, so a split into runs is the version-robust way to do this.)
    """
    data = np.load(bundle / node["asset"])
    points = np.asarray(data["points"], np.float64)
    props = node["props"]
    name = _short(node["name"])

    # viser line width is in pixels; there is no exact metric equivalent, so
    # approximate a tube radius that reads similarly at cover-image scale.
    radius = max(float(props.get("line_width", 2.0)) * 0.00025 * float(settings.line_scale), 1e-4)
    closed = bool(props.get("closed", False))
    cast = bool(props.get("cast_shadow", False))

    is_segments = node["kind"] == "LineSegmentsProps"
    colors = data.get("colors") if hasattr(data, "get") else (
        data["colors"] if "colors" in getattr(data, "files", []) else None)
    flat = props.get("color")

    # Build (polyline_points, colour) runs. For line segments, group consecutive
    # same-colour segments; for a spline, group contiguous same-colour points.
    runs = _colour_runs(points, colors, is_segments, flat)

    objs = []
    for idx, (line, colour) in enumerate(runs):
        curve = bpy.data.curves.new(f"{name}_{idx}", type="CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = radius
        curve.bevel_resolution = 4
        curve.fill_mode = "FULL"
        spline = curve.splines.new("POLY")
        spline.points.add(len(line) - 1)
        for i, p in enumerate(line):
            spline.points[i].co = (float(p[0]), float(p[1]), float(p[2]), 1.0)
        spline.use_cyclic_u = closed
        obj = bpy.data.objects.new(f"{name}_{idx}" if len(runs) > 1 else name, curve)
        bpy.context.collection.objects.link(obj)
        obj.matrix_world = matrix
        obj.data.materials.append(make_material(f"{name}_{idx}_mat", {"color": colour}))
        # In viser these are screen-space lines that cast no shadow; the tube we
        # build here would, introducing a shadow the browser view never showed.
        obj.visible_shadow = cast
        objs.append(obj)
    return objs


def _colour_runs(points, colors, is_segments: bool, flat):
    """Split a polyline/segment list into (points, colour) runs of equal colour."""
    if flat is not None or colors is None:
        colour = tuple(flat) if flat is not None else (255, 255, 255)
        line = points.reshape(-1, 3) if is_segments else points
        return [(line, colour)]

    cols = np.asarray(colors, np.int64).reshape(-1, 3)
    if is_segments:
        # points come in pairs; colour a segment by its first endpoint.
        segs = points.reshape(-1, 2, 3)
        seg_cols = cols.reshape(-1, 2, 3)[:, 0, :]
        runs = []
        start = 0
        for i in range(1, len(segs) + 1):
            if i == len(segs) or not np.array_equal(seg_cols[i], seg_cols[start]):
                # concatenate contiguous segments into one polyline where they chain.
                pts = segs[start:i].reshape(-1, 3)
                runs.append((pts, tuple(int(c) for c in seg_cols[start])))
                start = i
        return runs

    # spline: contiguous same-colour points; overlap a boundary point so runs join.
    runs = []
    start = 0
    for i in range(1, len(points) + 1):
        if i == len(points) or not np.array_equal(cols[i], cols[start]):
            end = min(i + 1, len(points))  # include the next point to avoid a gap
            runs.append((points[start:end], tuple(int(c) for c in cols[start])))
            start = i
    return runs


def add_point_cloud(node: dict, bundle: Path, matrix: Matrix, settings) -> list:
    """Render a point cloud as colour-carrying instanced geometry.

    Vertices alone render nothing, so build a geometry-nodes modifier that
    instances a small sphere on every point and drives its Base Color from a
    ``Col`` mesh attribute holding the exported per-point colours.
    """
    data = np.load(bundle / node["asset"])
    points = np.asarray(data["points"], np.float64)
    colors = data.get("colors") if hasattr(data, "get") else (
        data["colors"] if "colors" in getattr(data, "files", []) else None)
    name = _short(node["name"])

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(p) for p in points], [], [])
    mesh.update()

    # Per-point colour attribute named Col (linearised from exported sRGB).
    if settings.point_color is not None:
        flat = np.array(_srgb_to_linear(_parse_rgb(settings.point_color)))
        linear = np.broadcast_to(flat, (len(points), 3))
    elif colors is not None:
        linear = (np.asarray(colors, np.float64).reshape(-1, 3) / 255.0) ** 2.2
    else:
        linear = np.broadcast_to(np.array([0.8, 0.8, 0.8]), (len(points), 3))
    attr = mesh.attributes.new("Col", "FLOAT_COLOR", "POINT")
    for i, c in enumerate(linear):
        attr.data[i].color = (c[0], c[1], c[2], 1.0)

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.matrix_world = matrix

    # Default point size: a fraction of the bounds diagonal / cube-root count, so
    # it is never wrong by orders of magnitude regardless of the scene scale.
    if settings.point_size is not None:
        size = float(settings.point_size)
    else:
        span = float(np.linalg.norm(points.max(0) - points.min(0))) if len(points) else 1.0
        size = max(span / max(len(points) ** (1 / 3), 1) * 0.5, 1e-4)

    _point_cloud_geonodes(obj, size, name)
    obj.visible_shadow = bool(node["props"].get("cast_shadow", True))
    return [obj]


def _point_cloud_geonodes(obj, size: float, name: str) -> None:
    """Mesh-to-points -> instance-on-points icosphere, Col -> Base Color."""
    ng = bpy.data.node_groups.new(f"{name}_pointcloud", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = ng.nodes, ng.links
    gin = nodes.new("NodeGroupInput")
    gout = nodes.new("NodeGroupOutput")
    to_points = nodes.new("GeometryNodeMeshToPoints")

    ico = nodes.new("GeometryNodeMeshIcoSphere")
    if "Radius" in ico.inputs:
        ico.inputs["Radius"].default_value = size
    if "Subdivisions" in ico.inputs:
        ico.inputs["Subdivisions"].default_value = 1

    inst = nodes.new("GeometryNodeInstanceOnPoints")
    realize = nodes.new("GeometryNodeRealizeInstances")

    # Carry Col through to the instances and store it on the realized geometry so
    # a material Attribute node can read it.
    links.new(gin.outputs[0], to_points.inputs["Mesh"])
    links.new(to_points.outputs["Points"], inst.inputs["Points"])
    links.new(ico.outputs["Mesh"], inst.inputs["Instance"])
    links.new(inst.outputs["Instances"], realize.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], gout.inputs[0])

    mod = obj.modifiers.new(name, "NODES")
    mod.node_group = ng

    # Material reads the Col attribute (instances inherit it via the point domain).
    mat = bpy.data.materials.new(f"{name}_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_type = "GEOMETRY"
    at.attribute_name = "Col"
    nt.links.new(at.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.append(mat)


def add_box(node: dict, matrix: Matrix):
    dx, dy, dz = node["props"]["dimensions"]
    verts = np.array(
        [
            (x, y, z)
            for x in (-dx / 2, dx / 2)
            for y in (-dy / 2, dy / 2)
            for z in (-dz / 2, dz / 2)
        ]
    )
    faces = [
        (0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
        (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3),
    ]
    return add_mesh_object(_short(node["name"]), verts, faces, matrix, node["props"])


def add_icosphere(node: dict, matrix: Matrix):
    props = node["props"]
    bpy.ops.mesh.primitive_ico_sphere_add(
        radius=float(props["radius"]),
        subdivisions=int(props.get("subdivisions", 3)),
    )
    obj = bpy.context.active_object
    obj.name = _short(node["name"])
    obj.matrix_world = matrix
    apply_shading(obj, props, make_material(f"{obj.name}_mat", props))
    return obj


# --------------------------------------------------------------------------- #
# Camera (Phase 5)
# --------------------------------------------------------------------------- #

def scene_bounds(include_backdrop: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
    """World-space (mins, maxs) over every mesh, or None if nothing is built.

    The backdrop plane is excluded by default: it is sized *from* these bounds,
    so counting it would let framing (``--auto-camera``) chase a plane many
    times larger than the subject.
    """
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not include_backdrop and obj.name == BACKDROP_NAME:
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            mins = np.minimum(mins, world)
            maxs = np.maximum(maxs, world)
    if not np.isfinite(mins).all():
        return None
    return mins, maxs


def camera_basis(cam: dict) -> tuple[Vector, Vector, Vector, Vector]:
    """(eye, forward, right, up) from a manifest camera dict."""
    eye = Vector(cam["position"])
    forward = (Vector(cam["look_at"]) - eye).normalized()
    up = Vector(cam["up"]).normalized()
    right = forward.cross(up).normalized()
    true_up = right.cross(forward)
    return eye, forward, right, true_up


def _apply_orbit_dolly(cam: dict, orbit, dolly: float) -> dict:
    """Return a copy of ``cam`` rotated/dollied about its look-at point."""
    if (orbit is None or (orbit[0] == 0 and orbit[1] == 0)) and dolly == 1.0:
        return cam
    eye = Vector(cam["position"])
    target = Vector(cam["look_at"])
    up = Vector(cam["up"]).normalized()
    offset = eye - target
    if orbit is not None:
        az, el = math.radians(orbit[0]), math.radians(orbit[1])
        # world-Z azimuth, then elevation about the current right axis.
        offset = Matrix.Rotation(az, 4, "Z") @ offset
        fwd = (-offset).normalized()
        right = fwd.cross(up).normalized()
        offset = Matrix.Rotation(el, 4, right) @ offset
    offset = offset * float(dolly)
    out = dict(cam)
    out["position"] = list(target + offset)
    return out


def effective_camera(cam: dict | None, settings) -> dict | None:
    """The camera as it will actually be rendered (orbit/dolly applied).

    Every stage that reasons about the view -- framing *and* camera-relative
    lighting -- must go through this, or ``--orbit`` moves the camera out from
    under lights that were placed relative to it.
    """
    if cam is None:
        return None
    return _apply_orbit_dolly(cam, getattr(settings, "orbit", None),
                              getattr(settings, "dolly", 1.0))


def setup_camera(cam: dict, resolution: tuple[int, int] | None = None,
                 settings: Settings | None = None) -> None:
    settings = settings or Settings()
    cam = effective_camera(cam, settings)

    data = bpy.data.cameras.new("viser_camera")
    data.angle_y = float(cam["fov"])
    data.clip_start = max(float(cam["near"]), 1e-4)
    data.clip_end = float(cam["far"])
    obj = bpy.data.objects.new("viser_camera", data)
    bpy.context.collection.objects.link(obj)

    eye, forward, right, true_up = camera_basis(cam)
    # Blender cameras look down -Z with +Y up.
    rot = Matrix((right, true_up, -forward)).transposed().to_4x4()
    obj.matrix_world = Matrix.Translation(eye) @ rot

    scene = bpy.context.scene
    scene.camera = obj
    width, height = resolution or (
        int(cam.get("image_width") or 1920),
        int(cam.get("image_height") or 1080),
    )
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = int(settings.scale)

    _apply_aspect_fit(data, cam, int(width), int(height), settings.fit)

    if settings.dof:
        data.dof.use_dof = True
        dist = float(settings.dof) if settings.dof is not True else (
            Vector(cam["look_at"]) - eye).length
        data.dof.focus_distance = dist
        data.dof.aperture_fstop = float(settings.fstop)


def _apply_aspect_fit(data, cam: dict, width: int, height: int, fit: str) -> None:
    """Honour ``camera.fit`` and warn when the render crops the composed frame."""
    bundle_aspect = float(cam.get("aspect") or (cam.get("image_width", 1) /
                                                max(cam.get("image_height", 1), 1)))
    render_aspect = width / max(height, 1)

    if fit == "keep_vertical":
        data.sensor_fit = "VERTICAL"
    elif fit == "keep_horizontal":
        data.sensor_fit = "HORIZONTAL"
    elif fit == "fit_all":
        # Widen whichever axis would otherwise crop so nothing is lost.
        data.sensor_fit = "VERTICAL" if render_aspect <= bundle_aspect else "HORIZONTAL"

    if abs(render_aspect - bundle_aspect) / bundle_aspect > 0.02:
        axis = "horizontally" if render_aspect < bundle_aspect else "vertically"
        if fit == "keep_horizontal":
            axis = "vertically" if render_aspect < bundle_aspect else "horizontally"
        note = "letterboxing" if fit == "fit_all" else f"cropping {axis}"
        print(f"[bliser] WARNING: render aspect {render_aspect:.3f} differs from "
              f"bundle camera aspect {bundle_aspect:.3f}; {note} (fit={fit}).")


def auto_camera_spec(settings: Settings) -> dict | None:
    """A manifest-shaped camera framing the scene bounds in a 3/4 view.

    Returned rather than applied so the lighting stage can see the view it will
    be lighting; call this once the geometry is built and before the backdrop.
    """
    bounds = scene_bounds()
    if bounds is None:
        print("[bliser] --auto-camera: nothing built to frame.")
        return None
    mins, maxs = bounds
    center = (mins + maxs) / 2.0
    radius = float(np.linalg.norm(maxs - mins)) / 2.0 or 1.0

    az, el = math.radians(45.0), math.radians(25.0)
    direction = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    fov = math.radians(39.6)  # ~ default 50mm vertical FOV
    distance = radius / math.sin(fov / 2.0) * 1.15  # 15% margin
    eye = center + direction * distance

    cam = {
        "position": list(eye),
        "look_at": list(center),
        "up": [0.0, 0.0, 1.0],
        "fov": fov,
        "near": max(distance - radius * 2, 1e-3),
        "far": distance + radius * 4,
        "aspect": None,
    }
    res = settings.resolution or (1920, 1080)
    cam["aspect"] = res[0] / res[1]
    cam["image_width"], cam["image_height"] = int(res[0]), int(res[1])
    return cam


def make_auto_camera(settings: Settings) -> None:
    """Frame the scene bounds in a 3/4 view when the bundle has no camera."""
    cam = auto_camera_spec(settings)
    if cam is None:
        return
    setup_camera(cam, settings.resolution or (1920, 1080), settings)


# --------------------------------------------------------------------------- #
# Lighting (Phase 3)
# --------------------------------------------------------------------------- #

def dim_authored_lights(factor: float) -> None:
    """Scale every imported light's energy (call before adding a key light)."""
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            obj.data.energy *= float(factor)


def _cam_relative_direction(basis, azimuth: float, elevation: float) -> Vector:
    """Unit direction from az/el degrees in camera space.

    az 0 = behind camera (toward viewer), +az toward frame-right, +el up.
    """
    eye, forward, right, up = basis
    az, el = math.radians(azimuth), math.radians(elevation)
    # Start behind the camera (-forward), rotate toward right by az, up by el.
    d = (-forward) * (math.cos(az) * math.cos(el)) \
        + right * (math.sin(az) * math.cos(el)) \
        + up * math.sin(el)
    return d.normalized()


def add_camera_relative_light(cam: dict, azimuth: float, elevation: float, *,
                              energy: float = 3.0, angle_deg: float = 8.0,
                              color=(255, 255, 255), name: str = "key_light") -> None:
    """Add a sun light aimed from a camera-relative az/el direction."""
    basis = camera_basis(cam)
    direction = _cam_relative_direction(basis, azimuth, elevation)

    light = bpy.data.lights.new(name, type="SUN")
    light.energy = float(energy)
    light.angle = math.radians(float(angle_deg))
    light.color = _srgb_to_linear(color)
    obj = bpy.data.objects.new(name, light)
    bpy.context.collection.objects.link(obj)
    # A sun aims down its local -Z; point that along `direction` (light travels
    # from the placed direction toward the subject, so -Z = -direction).
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector(-direction).to_track_quat("-Z", "Y")


def three_point_rig(cam: dict, *, key_energy: float = 3.0, color=(255, 255, 255)) -> None:
    """Camera-relative key + fill (opposite, ~1/4 energy, wider) + rim (behind)."""
    add_camera_relative_light(cam, -40, 35, energy=key_energy, angle_deg=8.0,
                              color=color, name="key_light")
    add_camera_relative_light(cam, 45, 15, energy=key_energy * 0.25, angle_deg=25.0,
                              color=color, name="fill_light")
    add_camera_relative_light(cam, 160, 45, energy=key_energy * 0.6, angle_deg=5.0,
                              color=color, name="rim_light")


def apply_lighting(manifest: dict, settings: Settings) -> None:
    """Lighting stage: dim authored lights, add a key or a full rig.

    Runs after the manifest is loaded but before the camera object is created;
    lights are placed in world space from the camera basis *as it will be
    rendered*, so ``--orbit``/``--dolly`` keep a key light camera-relative.
    """
    cam = effective_camera(manifest.get("camera"), settings)

    if settings.dim_authored is not None:
        dim_authored_lights(float(settings.dim_authored))

    n_lights = sum(1 for n in manifest["nodes"] if n["kind"].endswith("LightProps")
                   and not n["kind"].startswith(("Ambient", "Hemisphere")))

    if cam is None:
        if settings.three_point or (settings.auto_light and n_lights == 0):
            print("[bliser] lighting rig needs a camera; skipping (no camera in bundle).")
        return

    key_color = _parse_rgb(settings.key_color) if settings.key_color else (255, 255, 255)
    key_energy = settings.key_energy if settings.key_energy is not None else 3.0

    if settings.three_point or (settings.auto_light and n_lights == 0):
        three_point_rig(cam, key_energy=key_energy, color=key_color)
    elif settings.key_light is not None:
        az, el = (float(x) for x in settings.key_light.split(","))
        add_camera_relative_light(
            cam, az, el, energy=key_energy,
            angle_deg=settings.key_angle if settings.key_angle is not None else 8.0,
            color=key_color)


# --------------------------------------------------------------------------- #
# World / backdrop
# --------------------------------------------------------------------------- #

def make_backdrop(color: str, shadow_catcher: bool = False, engine: str = "CYCLES") -> None:
    """Drop a large neutral ground plane just under the built scene so shadows
    catch on a floor. Sized and placed from the current mesh bounds."""
    bounds = scene_bounds()
    if bounds is None:
        return  # nothing built
    mins, maxs = bounds

    center = (mins + maxs) / 2.0
    span = float(np.max(maxs[:2] - mins[:2]))
    bpy.ops.mesh.primitive_plane_add(size=max(span * 8.0, 4.0))
    plane = bpy.context.active_object
    plane.name = BACKDROP_NAME
    # A hair below the lowest point so it never z-fights coincident geometry.
    plane.location = (center[0], center[1], float(mins[2]) - 1e-3)

    rgb = [int(c) for c in color.split(",")]
    plane.data.materials.append(make_material("backdrop_mat", {"color": rgb}))
    plane.visible_shadow = True

    if shadow_catcher:
        if engine.upper().startswith("CYCLES") and hasattr(plane, "is_shadow_catcher"):
            plane.is_shadow_catcher = True
        else:
            print("[bliser] shadow catcher is Cycles-only; using an opaque plane.")


def setup_world(manifest: dict, settings) -> None:
    world = bpy.data.worlds.new("viser_world")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    bg = nodes["Background"]
    bg.inputs["Strength"].default_value = settings.world_strength

    if settings.hdri:
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(settings.hdri)
        links.new(env.outputs["Color"], bg.inputs["Color"])
        return

    if getattr(settings, "studio_world", False):
        # A soft vertical gradient keyed off the view direction's Z: warm-dark
        # floor, neutral horizon, cool-bright zenith. Fills shadows and grades
        # the frame without any external HDRI file.
        geo = nodes.new("ShaderNodeNewGeometry")
        sep = nodes.new("ShaderNodeSeparateXYZ")
        rng = nodes.new("ShaderNodeMapRange")  # z in [-1,1] -> t in [0,1]
        rng.inputs["From Min"].default_value = -1.0
        rng.inputs["From Max"].default_value = 1.0
        ramp = nodes.new("ShaderNodeValToRGB")
        el = ramp.color_ramp.elements
        el[0].position, el[0].color = 0.0, (0.06, 0.055, 0.05, 1.0)  # floor
        el[1].position, el[1].color = 1.0, (0.55, 0.60, 0.70, 1.0)   # zenith
        mid = ramp.color_ramp.elements.new(0.5)
        mid.color = (0.30, 0.31, 0.34, 1.0)                          # horizon
        links.new(geo.outputs["Incoming"], sep.inputs["Vector"])
        links.new(sep.outputs["Z"], rng.inputs["Value"])
        links.new(rng.outputs["Result"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = settings.world_strength
        return

    # Ambient/hemisphere lights have no object form in Blender; fold whatever
    # the scene had into a flat world colour.
    color, strength = (0.05, 0.05, 0.06), settings.world_strength
    for node in manifest["nodes"]:
        props = node["props"]
        if node["kind"] == "AmbientLightProps":
            color = _srgb_to_linear(props["color"])
            strength = float(props["intensity"]) * settings.world_strength
        elif node["kind"] == "HemisphereLightProps":
            sky = np.array(props["sky_color"], float) / 255.0
            ground = np.array(props["ground_color"], float) / 255.0
            color = tuple(((sky + ground) / 2.0) ** 2.2)
            strength = float(props["intensity"]) * settings.world_strength
    bg.inputs["Color"].default_value = (*color, 1.0)
    bg.inputs["Strength"].default_value = strength

    if manifest.get("environment_map"):
        print(
            f"[bliser] scene used the '{manifest['environment_map']}' viser "
            "environment map; pass --hdri <file.exr> to match it in Blender."
        )


def _short(name: str) -> str:
    return name.strip("/").replace("/", "_") or "root"


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def build(manifest: dict, bundle: Path, settings) -> dict[str, list]:
    """Build every node and return a map of viser node path -> created objects."""
    overrides = parse_material_overrides(getattr(settings, "material", []))
    library = getattr(settings, "material_library", {}) or {}
    created_map: dict[str, list] = {}

    for node in manifest["nodes"]:
        matrix = Matrix([list(row) for row in node["matrix"]])
        kind = node["kind"]
        created = None
        if kind in ("GlbProps", "BatchedGlbProps"):
            holder = import_glb(node, bundle, matrix)
            created = [o for o in holder.children_recursive if o.type == "MESH"]
        elif kind in ("MeshProps", "SkinnedMeshProps", "BatchedMeshesProps"):
            data = np.load(bundle / node["asset"])
            created = [add_mesh_object(
                _short(node["name"]), data["vertices"], data["faces"], matrix, node["props"]
            )]
        elif kind == "BoxProps":
            created = [add_box(node, matrix)]
        elif kind == "IcosphereProps":
            created = [add_icosphere(node, matrix)]
        elif kind == "PointCloudProps":
            created = add_point_cloud(node, bundle, matrix, settings)
        elif kind.endswith("SplineProps") or kind == "LineSegmentsProps":
            created = add_curve(node, bundle, matrix, settings)
        elif kind.endswith("LightProps"):
            add_light(node, matrix, settings)
        else:
            print(f"[bliser] unhandled node kind: {kind} ({node['name']})")

        if created:
            created_map[node["name"]] = created
            props = _override_for(node["name"], overrides)
            if props is not None:
                apply_material_override(created, node["name"], props, library)

    # Config-driven material rules. A rule's ``node`` may end in ``/*`` to match a
    # whole subtree (like the CLI ``--material``); its action is one of ``split``,
    # ``use`` (a named library material), or ``color`` (a flat/PBR props dict).
    for rule in getattr(settings, "material_rules", []) or []:
        node = rule["node"]
        objs = _rule_targets(created_map, node)
        if not objs:
            print(f"[bliser] material rule: no objects for node {node!r}")
            continue
        if "split" in rule:
            apply_split(objs, rule["split"], library, node)
        elif "use" in rule:
            apply_material_override(objs, node, {"use": rule["use"]}, library)
        elif "color" in rule:
            props = {"color": tuple(rule["color"])}
            for k in ("opacity", "roughness", "metallic", "ior", "transmission",
                      "specular", "emission_strength"):
                if k in rule:
                    props[k] = rule[k]
            apply_material_override(objs, node, props, library)

    return created_map


def _rule_targets(created_map: dict[str, list], node: str) -> list:
    """Objects a config rule addresses; a trailing ``/*`` matches the subtree."""
    if node.endswith("/*"):
        prefix = node[:-2].rstrip("/")
        objs: list = []
        for name, created in created_map.items():
            if name == prefix or name.startswith(prefix + "/"):
                objs.extend(created)
        return objs
    return created_map.get(node, [])


# --------------------------------------------------------------------------- #
# Render engine
# --------------------------------------------------------------------------- #

def resolve_engine(requested: str) -> str:
    """Map an engine name onto whatever this Blender build calls it."""
    available = [
        item.identifier
        for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    ]
    if requested in available:
        return requested
    key = requested.upper().replace("BLENDER_", "").replace("_NEXT", "")
    for candidate in available:
        if key in candidate.upper():
            return candidate
    raise SystemExit(f"engine {requested!r} unavailable; this Blender has {available}")


def enable_gpu() -> None:
    """Point Cycles at every available GPU."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue  # not compiled into this build
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == backend]
        if gpus:
            for device in prefs.devices:
                device.use = device.type == backend
            bpy.context.scene.cycles.device = "GPU"
            print(f"[bliser] Cycles on {backend}: "
                  f"{', '.join(d.name for d in gpus)}")
            return
    print("[bliser] --gpu requested but no GPU found; rendering on CPU.")


def set_render_engine(scene, requested: str) -> str:
    """Select the engine, enabling the Cycles add-on if it is asked for."""
    if requested.strip().upper().startswith("CYCLES"):
        try:
            bpy.ops.preferences.addon_enable(module="cycles")
        except Exception as exc:  # already on, or genuinely absent
            print(f"[bliser] cycles addon_enable: {exc}")
        try:
            scene.render.engine = "CYCLES"
            return "CYCLES"
        except Exception as exc:
            raise SystemExit(f"Cycles requested but unavailable: {exc}")
    engine = resolve_engine(requested)
    scene.render.engine = engine
    return engine


def _apply_look(scene, look: str) -> None:
    """Set a colour-management look, validating against the live enum."""
    valid = [item.identifier for item in
             scene.view_settings.bl_rna.properties["look"].enum_items]
    if look in valid:
        scene.view_settings.look = look
        return
    # Blender prefixes looks with the view transform, e.g. "AgX - Punchy"; try a
    # case-insensitive / suffix match before giving up.
    for cand in valid:
        if look.lower() in cand.lower():
            scene.view_settings.look = cand
            return
    raise SystemExit(f"look {look!r} unavailable; valid looks: {valid}")


def render(path: str, settings) -> None:
    scene = bpy.context.scene
    engine = set_render_engine(scene, settings.engine)
    if engine == "CYCLES":
        scene.cycles.samples = settings.samples
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = float(settings.adaptive_threshold)
        scene.cycles.use_denoising = True
        for denoiser in ("OPTIX", "OPENIMAGEDENOISE"):
            try:
                scene.cycles.denoiser = denoiser
                break
            except TypeError:
                continue
        if settings.max_bounces is not None:
            scene.cycles.max_bounces = int(settings.max_bounces)
            scene.cycles.diffuse_bounces = int(settings.max_bounces)
            scene.cycles.glossy_bounces = int(settings.max_bounces)
            scene.cycles.transmission_bounces = int(settings.max_bounces)
        if settings.gpu:
            enable_gpu()
    elif hasattr(scene, "eevee"):
        ee = scene.eevee
        ee.taa_render_samples = settings.samples
        for attr in ("use_raytracing", "use_fast_gi", "use_shadows"):
            if hasattr(ee, attr):
                setattr(ee, attr, True)

    scene.view_settings.exposure = float(getattr(settings, "exposure", 0.0))
    if getattr(settings, "look", None):
        _apply_look(scene, settings.look)

    transparent = bool(getattr(settings, "transparent", False))
    scene.render.film_transparent = transparent
    if transparent:
        scene.render.image_settings.color_mode = "RGBA"

    out = Path(path).absolute()
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out)
    bpy.ops.render.render(write_still=True)
    print(f"[bliser] rendered -> {path}")


# --------------------------------------------------------------------------- #
# Provenance + misc
# --------------------------------------------------------------------------- #

def check_bundle_version(manifest: dict, bundle: Path) -> None:
    if manifest.get("format") != "bliser":
        raise SystemExit(f"{bundle} is not a bliser bundle")
    version = int(manifest.get("version", 1))
    if version > SUPPORTED_BUNDLE_VERSION:
        raise SystemExit(
            f"{bundle} is format version {version}, but this importer only understands "
            f"up to {SUPPORTED_BUNDLE_VERSION}. Update bliser.")
    if version < SUPPORTED_BUNDLE_VERSION:
        print(f"[bliser] WARNING: bundle is format version {version} "
              f"(importer is {SUPPORTED_BUNDLE_VERSION}); some fields may be missing.")


def list_nodes(manifest: dict) -> None:
    """Print name/kind/size per node (no bpy needed for the logic, but usable here)."""
    print(f"{'kind':<24} {'size':>10}  name")
    for node in manifest["nodes"]:
        size = ""
        props = node.get("props", {})
        if "dimensions" in props:
            size = "box"
        print(f"{node['kind']:<24} {size:>10}  {node['name']}")


def write_sidecar(output: str, settings: Settings, bundle: Path,
                  manifest: dict, render_time: float) -> None:
    """Write ``<output>.yaml`` provenance next to the rendered image.

    Blender's Python has no yaml module, so emit a small, hand-rolled YAML
    document (flat key/value plus the resolved config as JSON-in-YAML).
    """
    out = Path(output)
    sidecar = out.with_suffix(out.suffix + ".yaml")
    # The *effective* settings, i.e. after CLI flags have won over the config.
    # Recording ``_resolved_config`` alone would claim the config's value for
    # anything overridden on the command line.
    effective = {k: v for k, v in dataclasses.asdict(settings).items()
                 if not k.startswith("_")}
    doc = {
        "bundle": str(bundle),
        "bundle_version": manifest.get("version"),
        "bliser_version": _package_version(),
        "blender_version": bpy.app.version_string,
        "render_time_seconds": round(render_time, 2),
        "rendered_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "resolved_config": effective,
    }
    if settings._resolved_config is not None:
        doc["config_file_values"] = settings._resolved_config
    sidecar.write_text(_to_yaml(doc))
    print(f"[bliser] wrote provenance -> {sidecar}")


def _package_version() -> str:
    try:
        from bliser import __version__  # may fail: not on Blender's path
        return __version__
    except Exception:
        return "unknown"


def _to_yaml(doc: dict, indent: int = 0) -> str:
    """Minimal YAML emitter (stdlib-only) for the provenance sidecar."""
    pad = "  " * indent
    lines = []
    for key, value in doc.items():
        if isinstance(value, dict):
            if not value:
                lines.append(f"{pad}{key}: {{}}")
                continue
            lines.append(f"{pad}{key}:")
            lines.append(_to_yaml(value, indent + 1).rstrip("\n"))
        elif isinstance(value, (list, tuple)):
            lines.append(f"{pad}{key}: {json.dumps(list(value))}")
        elif isinstance(value, str):
            lines.append(f"{pad}{key}: {json.dumps(value)}")
        elif isinstance(value, bool):
            lines.append(f"{pad}{key}: {'true' if value else 'false'}")
        elif value is None:
            lines.append(f"{pad}{key}: null")
        else:
            lines.append(f"{pad}{key}: {value}")
    return "\n".join(lines) + "\n"


def save_blend(path: str) -> None:
    Path(path).absolute().parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(path).absolute()))
    print(f"[bliser] saved .blend -> {path}")


def _derive_blend_path(settings: Settings) -> str | None:
    if settings.save_blend in (None, False):
        return None
    if settings.save_blend is not True:
        return str(settings.save_blend)
    if settings.render:
        return str(Path(settings.render).with_suffix(".blend"))
    return str(Path(settings.bundle).name + ".blend")


# --------------------------------------------------------------------------- #
# Pipeline (Phase 2)
# --------------------------------------------------------------------------- #

class Pipeline:
    """The render pipeline as overridable stages.

    Stage order::

        clear -> build -> after_build -> plan_camera -> backdrop -> lighting
              -> before_world -> world -> camera -> before_render -> render

    ``plan_camera`` only resolves ``--auto-camera`` into ``self.manifest`` so
    that later stages can read the view; the camera object itself is created in
    ``camera``.

    Subclass and override ``after_build`` / ``before_world`` / ``before_render``,
    or pass callables of those names to :meth:`from_argv`. Hooks read and write
    ``self.settings``, ``self.manifest``, ``self.bundle`` and ``self.created``
    (viser node path -> list of Blender objects).
    """

    def __init__(self, settings: Settings, *, after_build: Callable | None = None,
                 before_world: Callable | None = None, before_render: Callable | None = None):
        self.settings = settings
        self.bundle = Path(settings.bundle)
        self.manifest = json.loads((self.bundle / "scene.json").read_text())
        check_bundle_version(self.manifest, self.bundle)
        self.created: dict[str, list] = {}
        self._cb = {
            "after_build": after_build,
            "before_world": before_world,
            "before_render": before_render,
        }

    # -- hooks (overridable; default to the optional callback) ------------- #
    def after_build(self) -> None:
        if self._cb["after_build"]:
            self._cb["after_build"](self)

    def before_world(self) -> None:
        if self._cb["before_world"]:
            self._cb["before_world"](self)

    def before_render(self) -> None:
        if self._cb["before_render"]:
            self._cb["before_render"](self)

    # -- stages ------------------------------------------------------------ #
    def clear(self) -> None:
        clear_scene(self.settings.keep_default_cube)

    def build(self) -> None:
        self.created = build(self.manifest, self.bundle, self.settings)

    def plan_camera(self) -> None:
        """Resolve --auto-camera into ``self.manifest['camera']``.

        Done here -- after build, before the backdrop is sized and before any
        camera-relative light is placed -- so the lighting stage sees the view
        and the framing does not try to frame the backdrop.
        """
        if self.manifest.get("camera") or not self.settings.auto_camera:
            return
        cam = auto_camera_spec(self.settings)
        if cam is not None:
            self.manifest["camera"] = cam

    def backdrop(self) -> None:
        if self.settings.backdrop:
            make_backdrop(self.settings.backdrop_color,
                          shadow_catcher=self.settings.shadow_catcher,
                          engine=self.settings.engine)

    def lighting(self) -> None:
        apply_lighting(self.manifest, self.settings)

    def world(self) -> None:
        setup_world(self.manifest, self.settings)

    def camera(self) -> None:
        if self.manifest.get("camera"):
            setup_camera(self.manifest["camera"], self.settings.resolution, self.settings)
        elif self.settings.auto_camera:
            pass  # plan_camera already reported that there was nothing to frame
        else:
            print("[bliser] no camera in bundle (no browser was connected). "
                  "Pass --auto-camera to frame the scene automatically.")
            if self.settings.resolution:
                bpy.context.scene.render.resolution_x = self.settings.resolution[0]
                bpy.context.scene.render.resolution_y = self.settings.resolution[1]

    def render(self) -> None:
        blend = _derive_blend_path(self.settings)
        if blend:
            save_blend(blend)
        if not self.settings.render:
            return
        start = time.time()
        render(self.settings.render, self.settings)
        write_sidecar(self.settings.render, self.settings, self.bundle,
                      self.manifest, time.time() - start)

    # -- driver ------------------------------------------------------------ #
    def run(self) -> "Pipeline":
        self.clear()
        self.build()
        self.after_build()
        self.plan_camera()
        self.backdrop()
        self.lighting()
        self.before_world()
        self.world()
        self.camera()
        self.before_render()
        self.render()
        print(f"[bliser] built {len(self.manifest['nodes'])} nodes from {self.bundle}")
        return self

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_argv(cls, argv: list[str] | None = None, **hooks) -> "Pipeline":
        if argv is None:
            argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
        args = parse_args(argv)
        explicit = _explicit_flags(argv)
        config = load_config(args.config) if args.config else None
        settings = Settings.from_args(args, explicit, config)
        if not settings.bundle:
            raise SystemExit("no bundle: pass --bundle PATH or --config with a bundle.")
        return cls(settings, **hooks)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)
    explicit = _explicit_flags(argv)
    config = load_config(args.config) if args.config else None
    settings = Settings.from_args(args, explicit, config)

    if not settings.bundle:
        raise SystemExit("no bundle: pass --bundle PATH or --config with a bundle.")

    # --list-nodes short-circuits before building anything.
    if getattr(args, "list_nodes", False):
        manifest = json.loads((Path(settings.bundle) / "scene.json").read_text())
        check_bundle_version(manifest, Path(settings.bundle))
        list_nodes(manifest)
        return

    Pipeline(settings).run()


if __name__ == "__main__":
    main()
