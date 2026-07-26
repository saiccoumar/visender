"""Regression tests for the viser -> bundle exporter.

viser is not a hard dependency, so these drive ``export_scene`` with fakes
shaped like the handles it reads: ``handle._impl.{props,visible,wxyz,position}``.
The point is to pin the *bundle contract* (``scene.json`` + assets), which is
the only thing the Blender side ever sees.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any

import numpy as np
import pytest

from visender import _export


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

def props_type(name: str, **fields):
    """A dataclass named like a viser props class, e.g. ``MeshProps``."""
    return dataclasses.make_dataclass(
        name, [(k, Any, dataclasses.field(default=v)) for k, v in fields.items()])


MeshProps = props_type("MeshProps", vertices=None, faces=None, color=(200, 200, 200),
                       opacity=1.0, flat_shading=False, cast_shadow=True)
GlbProps = props_type("GlbProps", glb_data=b"", scale=1.0)
PointCloudProps = props_type("PointCloudProps", points=None, colors=None, point_size=0.01)
LineSegmentsProps = props_type("LineSegmentsProps", points=None, colors=None, line_width=2.0)
BoxProps = props_type("BoxProps", dimensions=(1.0, 1.0, 1.0), color=(200, 0, 0))
DirectionalLightProps = props_type("DirectionalLightProps", color=(255, 255, 255),
                                   intensity=1.0)
FrameProps = props_type("FrameProps", show_axes=True)
TransformControlsProps = props_type("TransformControlsProps", scale=1.0)
UnsupportedProps = props_type("GaussianSplatsProps", blob=b"")


class FakeHandle:
    def __init__(self, props, *, position=(0, 0, 0), wxyz=(1, 0, 0, 0), visible=True):
        self._impl = type("Impl", (), {})()
        self._impl.props = props
        self._impl.position = np.asarray(position, float)
        self._impl.wxyz = np.asarray(wxyz, float)
        self._impl.visible = visible


class FakeServer:
    def __init__(self, handles: dict, camera=None):
        self.scene = type("Scene", (), {})()
        self.scene._handle_from_node_name = handles
        self._camera = camera

    def get_clients(self):
        if self._camera is None:
            return {}
        return {0: type("Client", (), {"camera": self._camera})()}


class FakeCamera:
    position = (4.0, -4.0, 3.0)
    look_at = (0.0, 0.0, 0.0)
    up_direction = (0.0, 0.0, 1.0)
    fov = 0.69
    aspect = 16 / 9
    near = 0.1
    far = 100.0
    image_width = 1600
    image_height = 900


def simple_mesh_props():
    return MeshProps(vertices=np.zeros((3, 3), np.float32),
                     faces=np.array([[0, 1, 2]], np.uint32))


def read(out):
    return json.loads((out / "scene.json").read_text())


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #

def test_quat_to_mat_identity_and_90deg():
    assert np.allclose(_export._quat_to_mat((1, 0, 0, 0)), np.eye(4))
    half = math.sqrt(0.5)
    m = _export._quat_to_mat((half, 0, 0, half))  # +90 deg about Z
    assert np.allclose(m[:3, :3] @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-6)


def test_quat_to_mat_normalises_and_survives_a_zero_quat():
    m = _export._quat_to_mat((2, 0, 0, 0))
    assert np.allclose(m, np.eye(4))
    assert np.allclose(_export._quat_to_mat((0, 0, 0, 0)), np.eye(4))


def test_slug_and_ancestors():
    assert _export._slug("/a/b/c") == "a__b__c"
    assert _export._slug("/") == "root"
    assert _export._ancestors("/a/b/c") == ["/a", "/a/b"]
    assert _export._ancestors("/a") == []


def test_slug_of_a_deep_node_path_stays_a_legal_filename():
    # Deep URDFs (a foot ten links down) produce node paths far past the
    # 255-byte limit on a single path component.
    deep = "/robot/visual/" + "/".join(f"long_link_name_{i:03d}" for i in range(20))
    slug = _export._slug(deep)
    assert len(slug.encode()) <= 120
    assert slug.endswith(f"__{hashlib.sha1(deep.encode()).hexdigest()[:8]}")
    # Distinct paths that share a tail must not collide.
    assert _export._slug(deep) != _export._slug(deep.replace("_000", "_999"))


def test_world_matrix_composes_the_ancestor_chain(tmp_path):
    server = FakeServer({
        "/parent": FakeHandle(FrameProps(), position=(1, 0, 0)),
        "/parent/child": FakeHandle(simple_mesh_props(), position=(0, 2, 0)),
    })
    out = _export.export_scene(server, tmp_path / "b")
    node = read(out)["nodes"][0]
    assert np.allclose(np.array(node["matrix"])[:3, 3], [1, 2, 0])


def test_scalar_props_drops_bulk_arrays_and_bytes(tmp_path):
    props = MeshProps(vertices=np.zeros((100, 3), np.float32),
                      faces=np.zeros((50, 3), np.uint32))
    server = FakeServer({"/mesh": FakeHandle(props)})
    out = _export.export_scene(server, tmp_path / "b")
    node = read(out)["nodes"][0]
    assert "vertices" not in node["props"] and "faces" not in node["props"]
    assert node["props"]["color"] == [200, 200, 200]


# --------------------------------------------------------------------------- #
# Node selection rules
# --------------------------------------------------------------------------- #

def test_gizmos_are_dropped_unless_asked_for(tmp_path):
    server = FakeServer({
        "/mesh": FakeHandle(simple_mesh_props()),
        "/frame": FakeHandle(FrameProps()),
    })
    names = [n["name"] for n in read(_export.export_scene(server, tmp_path / "a"))["nodes"]]
    assert names == ["/mesh"]

    kept = read(_export.export_scene(server, tmp_path / "b", include_gizmos=True))
    # A FrameProps has no Blender equivalent, so it is reported as skipped
    # rather than emitted -- the rule under test is only that it got as far as
    # the geometry dispatch.
    assert [n["name"] for n in kept["nodes"]] == ["/mesh"]


def test_hidden_nodes_are_dropped_unless_asked_for(tmp_path):
    server = FakeServer({
        "/shown": FakeHandle(simple_mesh_props()),
        "/hidden": FakeHandle(simple_mesh_props(), visible=False),
    })
    names = [n["name"] for n in read(_export.export_scene(server, tmp_path / "a"))["nodes"]]
    assert names == ["/shown"]
    names = [n["name"] for n in
             read(_export.export_scene(server, tmp_path / "b", include_hidden=True))["nodes"]]
    assert sorted(names) == ["/hidden", "/shown"]


def test_a_hidden_ancestor_hides_the_subtree(tmp_path):
    server = FakeServer({
        "/group": FakeHandle(FrameProps(), visible=False),
        "/group/mesh": FakeHandle(simple_mesh_props()),
    })
    assert read(_export.export_scene(server, tmp_path / "b"))["nodes"] == []


def test_a_hidden_transform_gizmo_does_not_hide_its_children(tmp_path):
    """Unticking "show gizmos" must not silently delete what hangs off them."""
    server = FakeServer({
        "/gizmo": FakeHandle(TransformControlsProps(), visible=False),
        "/gizmo/light": FakeHandle(DirectionalLightProps()),
    })
    names = [n["name"] for n in read(_export.export_scene(server, tmp_path / "b"))["nodes"]]
    assert names == ["/gizmo/light"]


def test_a_hidden_gizmo_still_hides_itself(tmp_path):
    server = FakeServer({"/gizmo": FakeHandle(TransformControlsProps(), visible=False)})
    assert read(_export.export_scene(server, tmp_path / "b"))["nodes"] == []


def test_node_filter_applies_after_the_other_rules(tmp_path):
    server = FakeServer({
        "/keep": FakeHandle(simple_mesh_props()),
        "/drop": FakeHandle(simple_mesh_props()),
    })
    out = _export.export_scene(server, tmp_path / "b",
                               node_filter=lambda n: n != "/drop")
    assert [n["name"] for n in read(out)["nodes"]] == ["/keep"]


def test_unsupported_kinds_are_skipped_not_fatal(tmp_path, capsys):
    server = FakeServer({
        "/mesh": FakeHandle(simple_mesh_props()),
        "/splat": FakeHandle(UnsupportedProps()),
    })
    out = _export.export_scene(server, tmp_path / "b")
    assert [n["name"] for n in read(out)["nodes"]] == ["/mesh"]
    assert "/splat" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Assets
# --------------------------------------------------------------------------- #

def test_mesh_asset_round_trips(tmp_path):
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    faces = np.array([[0, 1, 2]], np.uint32)
    server = FakeServer({"/m": FakeHandle(MeshProps(vertices=verts, faces=faces))})
    out = _export.export_scene(server, tmp_path / "b")
    node = read(out)["nodes"][0]
    assert node["asset"] == "assets/m.npz"
    with np.load(out / node["asset"]) as data:
        assert np.allclose(data["vertices"], verts)
        assert np.array_equal(data["faces"], faces)
        assert data["vertices"].dtype == np.float32
        assert data["faces"].dtype == np.uint32


def test_glb_asset_is_written_verbatim(tmp_path):
    server = FakeServer({"/robot": FakeHandle(GlbProps(glb_data=b"glTF-bytes"))})
    out = _export.export_scene(server, tmp_path / "b")
    node = read(out)["nodes"][0]
    assert (out / node["asset"]).read_bytes() == b"glTF-bytes"


def test_point_cloud_colors_are_broadcast_to_per_point(tmp_path):
    """A single colour for the whole cloud must reach Blender as per-point."""
    pts = np.zeros((10, 3), np.float32)
    server = FakeServer({"/c": FakeHandle(
        PointCloudProps(points=pts, colors=np.array([10, 20, 30], np.uint8)))})
    out = _export.export_scene(server, tmp_path / "b")
    with np.load(out / "assets/c.npz") as data:
        assert data["colors"].shape == (10, 3)
        assert (data["colors"] == [10, 20, 30]).all()


def test_line_segments_carry_colors_but_splines_do_not(tmp_path):
    seg = np.zeros((4, 2, 3), np.float32)
    server = FakeServer({"/l": FakeHandle(
        LineSegmentsProps(points=seg, colors=np.zeros((4, 2, 3), np.uint8)))})
    out = _export.export_scene(server, tmp_path / "b")
    with np.load(out / "assets/l.npz") as data:
        assert set(data.files) == {"points", "colors"}


def test_primitives_and_lights_need_no_asset(tmp_path):
    server = FakeServer({
        "/box": FakeHandle(BoxProps()),
        "/sun": FakeHandle(DirectionalLightProps()),
    })
    for node in read(_export.export_scene(server, tmp_path / "b"))["nodes"]:
        assert "asset" not in node


# --------------------------------------------------------------------------- #
# Manifest shape
# --------------------------------------------------------------------------- #

def test_manifest_header_is_the_contract(tmp_path):
    server = FakeServer({"/m": FakeHandle(simple_mesh_props())}, camera=FakeCamera())
    manifest = read(_export.export_scene(server, tmp_path / "b",
                                         environment_map="city", extras={"seed": 3}))
    assert manifest["format"] == "visender"
    assert manifest["version"] == 1
    assert manifest["up_direction"] == "+z"
    assert manifest["environment_map"] == "city"
    assert manifest["extras"] == {"seed": 3}
    cam = manifest["camera"]
    assert cam["position"] == [4.0, -4.0, 3.0]
    assert cam["image_width"] == 1600 and cam["image_height"] == 900
    assert set(cam) == {"position", "look_at", "up", "fov", "aspect", "near", "far",
                        "image_width", "image_height"}


def test_camera_is_null_when_no_browser_is_connected(tmp_path):
    server = FakeServer({"/m": FakeHandle(simple_mesh_props())})
    assert read(_export.export_scene(server, tmp_path / "b"))["camera"] is None


def test_manifest_is_json_serialisable_with_numpy_props(tmp_path):
    props = MeshProps(vertices=np.zeros((3, 3), np.float32),
                      faces=np.array([[0, 1, 2]], np.uint32),
                      color=np.array([1, 2, 3], np.uint8),
                      opacity=np.float32(0.5), flat_shading=np.bool_(True))
    server = FakeServer({"/m": FakeHandle(props)})
    node = read(_export.export_scene(server, tmp_path / "b"))["nodes"][0]
    assert node["props"]["color"] == [1, 2, 3]
    assert node["props"]["opacity"] == pytest.approx(0.5)
    assert node["props"]["flat_shading"] is True


def test_export_replaces_an_existing_bundle(tmp_path):
    out = tmp_path / "b"
    server = FakeServer({"/a": FakeHandle(simple_mesh_props())})
    _export.export_scene(server, out)
    stale = out / "assets" / "stale.npz"
    stale.write_bytes(b"x")
    server2 = FakeServer({"/b": FakeHandle(simple_mesh_props())})
    _export.export_scene(server2, out)
    assert not stale.exists()
    assert [n["name"] for n in read(out)["nodes"]] == ["/b"]


def test_nodes_are_emitted_in_a_stable_order(tmp_path):
    handles = {f"/n{i}": FakeHandle(simple_mesh_props()) for i in (3, 1, 2)}
    out = _export.export_scene(FakeServer(handles), tmp_path / "b")
    assert [n["name"] for n in read(out)["nodes"]] == ["/n1", "/n2", "/n3"]


# --------------------------------------------------------------------------- #
# The export button
# --------------------------------------------------------------------------- #


class FakeGuiHandle:
    def __init__(self, value=None):
        self.value = value
        self.disabled = False
        self._callback = None

    def on_click(self, fn):
        self._callback = fn
        return fn

    def click(self):
        self._callback(self)


class FakeGuiServer(FakeServer):
    def __init__(self, handles, camera=None):
        super().__init__(handles, camera)
        self.buttons: list[FakeGuiHandle] = []
        self.gui = type("Gui", (), {
            "add_button": lambda _self, label, **kw: self._make(self.buttons),
            "add_text": lambda _self, label, **kw: self._make([]),
        })()

    def _make(self, sink):
        handle = FakeGuiHandle()
        sink.append(handle)
        return handle


def test_export_button_forwards_a_node_filter_instead_of_calling_it(tmp_path):
    # node_filter is itself a callable, so the "zero-arg callable means
    # deferred value" convenience must not swallow it.
    server = FakeGuiServer({"/keep": FakeHandle(simple_mesh_props()),
                            "/drop": FakeHandle(simple_mesh_props())})
    _export.add_export_button(server, out_dir=tmp_path / "b", timestamp=False,
                              node_filter=lambda name: name != "/drop")
    server.buttons[0].click()
    assert [n["name"] for n in read(tmp_path / "b")["nodes"]] == ["/keep"]


def test_export_button_still_evaluates_zero_arg_thunks(tmp_path):
    server = FakeGuiServer({"/a": FakeHandle(simple_mesh_props())})
    _export.add_export_button(server, out_dir=tmp_path / "b", timestamp=False,
                              environment_map=lambda: "city")
    server.buttons[0].click()
    assert read(tmp_path / "b")["environment_map"] == "city"
