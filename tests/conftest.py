"""Shared fixtures.

Two tiers of test live here:

* Solver-side tests import ``bliser.config`` / ``cli`` / ``_export``
  normally -- they only need numpy + pyyaml.
* ``blender_import`` imports ``bpy`` and ``mathutils`` at module scope, so it
  cannot be imported by a plain pytest run. The ``bi`` fixture loads it against
  minimal stubs when bpy is absent, which is enough for every pure-logic
  function (arg parsing, settings precedence, material specs, YAML emission).
  Anything that actually touches Blender data is covered by the integration
  tier in ``test_blender_integration.py``, which shells out to a real Blender.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import math
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
BLENDER_IMPORT_PY = REPO / "bliser" / "blender_import.py"


# --------------------------------------------------------------------------- #
# blender_import under stubs
# --------------------------------------------------------------------------- #

class _StubVector(list):
    """Just enough mathutils.Vector for the pure-logic paths."""

    def __init__(self, values=(0.0, 0.0, 0.0)):
        super().__init__(float(v) for v in values)

    @property
    def length(self):
        return math.sqrt(sum(v * v for v in self))

    def normalized(self):
        n = self.length or 1.0
        return _StubVector([v / n for v in self])

    def cross(self, other):
        a, b = self, other
        return _StubVector([a[1] * b[2] - a[2] * b[1],
                            a[2] * b[0] - a[0] * b[2],
                            a[0] * b[1] - a[1] * b[0]])

    def __add__(self, other):
        return _StubVector([a + b for a, b in zip(self, other)])

    def __sub__(self, other):
        return _StubVector([a - b for a, b in zip(self, other)])

    def __mul__(self, k):
        return _StubVector([a * float(k) for a in self])

    __rmul__ = __mul__

    def __neg__(self):
        return _StubVector([-a for a in self])


class _StubMatrix:
    def __init__(self, rows=None):
        self.rows = rows

    @classmethod
    def Rotation(cls, *args, **kwargs):
        return cls()

    @classmethod
    def Translation(cls, *args, **kwargs):
        return cls()

    def __matmul__(self, other):
        return other

    def to_4x4(self):
        return self

    def to_3x3(self):
        return self

    def transposed(self):
        return self


def _install_stubs() -> None:
    bpy = types.ModuleType("bpy")
    bpy.data = types.SimpleNamespace()
    bpy.ops = types.SimpleNamespace()
    bpy.context = types.SimpleNamespace()
    bpy.types = types.SimpleNamespace()
    bpy.app = types.SimpleNamespace(version_string="stub")
    sys.modules.setdefault("bpy", bpy)

    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = _StubVector
    mathutils.Matrix = _StubMatrix
    mathutils.Quaternion = object
    sys.modules.setdefault("mathutils", mathutils)


@pytest.fixture(scope="session")
def bi():
    """The ``blender_import`` module, loaded by path (stubbed bpy if needed)."""
    try:
        import bpy  # noqa: F401  (running inside Blender)
    except ImportError:
        _install_stubs()
    spec = importlib.util.spec_from_file_location("_bliser_blender_import",
                                                  BLENDER_IMPORT_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Synthetic bundles
# --------------------------------------------------------------------------- #

def _identity():
    return np.eye(4).tolist()


@pytest.fixture
def make_bundle(tmp_path):
    """Write a minimal but representative bundle and return its path.

    Covers one node of every kind the importer branches on, so a change that
    breaks a builder shows up as a failure rather than as a silent skip.
    """

    built: dict[tuple, Path] = {}

    def _make(name: str = "bundle", *, camera: bool = True, version: int = 1,
              fmt: str = "bliser", lights: bool = True) -> Path:
        # Same arguments -> same bundle, so a test may call this repeatedly (e.g.
        # to run Blender twice with different flags) without rebuilding.
        key = (name, camera, version, fmt, lights)
        if key in built:
            return built[key]
        out = tmp_path / ("bundle_%d" % len(built) if built else name)
        built[key] = out
        assets = out / "assets"
        assets.mkdir(parents=True)

        # A unit tetrahedron mesh.
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.uint32)
        np.savez_compressed(assets / "mesh.npz", vertices=verts, faces=faces)

        pts = np.random.default_rng(0).random((64, 3)).astype(np.float32)
        cols = np.full((64, 3), 200, np.uint8)
        np.savez_compressed(assets / "cloud.npz", points=pts, colors=cols)

        # viser line segments are (N, 2, 3): a point *and* a colour per endpoint.
        seg = np.array([[[0, 0, 0], [1, 1, 1]], [[1, 1, 1], [2, 0, 1]]], np.float32)
        seg_cols = np.full((2, 2, 3), 30, np.uint8)
        seg_cols[1] = 200  # two colour runs, so the run-splitting path is exercised
        np.savez_compressed(assets / "lines.npz", points=seg, colors=seg_cols)

        nodes = [
            {"name": "/mesh", "kind": "MeshProps", "matrix": _identity(),
             "props": {"flat_shading": False, "cast_shadow": True,
                       "color": [200, 200, 200], "opacity": 1.0},
             "asset": "assets/mesh.npz"},
            {"name": "/group/box", "kind": "BoxProps", "matrix": _identity(),
             "props": {"dimensions": [1.0, 1.0, 1.0], "color": [180, 60, 60],
                       "opacity": 1.0}},
            {"name": "/group/ball", "kind": "IcosphereProps", "matrix": _identity(),
             "props": {"radius": 0.5, "subdivisions": 2, "color": [60, 180, 60],
                       "opacity": 1.0}},
            {"name": "/cloud", "kind": "PointCloudProps", "matrix": _identity(),
             "props": {"point_size": 0.02}, "asset": "assets/cloud.npz"},
            {"name": "/lines", "kind": "LineSegmentsProps", "matrix": _identity(),
             "props": {"line_width": 2.0}, "asset": "assets/lines.npz"},
        ]
        if lights:
            nodes.append(
                {"name": "/sun", "kind": "DirectionalLightProps",
                 "matrix": _identity(),
                 "props": {"color": [255, 255, 255], "intensity": 1.0}})

        manifest = {
            "format": fmt,
            "version": version,
            "up_direction": "+z",
            "nodes": nodes,
            "camera": {
                "position": [4.0, -4.0, 3.0], "look_at": [0.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0], "fov": 0.69, "aspect": 16 / 9,
                "near": 0.1, "far": 100.0,
                "image_width": 1600, "image_height": 900,
            } if camera else None,
            "environment_map": None,
            "extras": {},
        }
        (out / "scene.json").write_text(json.dumps(manifest, indent=1))
        return out

    return _make


# --------------------------------------------------------------------------- #
# Real Blender (integration tier)
# --------------------------------------------------------------------------- #

def find_blender() -> str | None:
    for candidate in (os.environ.get("BLENDER"), shutil.which("blender")):
        if candidate and Path(candidate).exists():
            return candidate
    globbed = sorted(glob.glob("/opt/blender/blender-*/blender"))
    return globbed[-1] if globbed else None


@pytest.fixture(scope="session")
def blender() -> str:
    binary = find_blender()
    if binary is None:
        pytest.skip("no Blender binary found (set $BLENDER to run integration tests)")
    return binary
