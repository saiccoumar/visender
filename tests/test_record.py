"""The recording half: Recorder sampling and the animation block it writes.

Driven by the same fakes as ``test_export.py`` -- viser is not a dependency of
the test run. What is pinned here is the *animated bundle contract*: which
nodes get a track, what shape the npz has, and the invariant that makes it
cheap (a node that never moves carries no track at all).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from visender import _record
from tests.test_export import (
    BoxProps,
    FakeCamera,
    FakeHandle,
    FakeServer,
    MeshProps,
    simple_mesh_props,
)


def two_node_server(camera=None):
    handles = {
        "/still": FakeHandle(BoxProps(), position=(1.0, 0.0, 0.0)),
        "/mover": FakeHandle(simple_mesh_props(), position=(0.0, 0.0, 0.0)),
    }
    return FakeServer(handles, camera=camera), handles


def test_capture_samples_world_matrices_at_the_time_of_the_call():
    server, handles = two_node_server()
    rec = _record.Recorder(server, camera=False)
    rec.capture()
    handles["/mover"]._impl.position = np.array([0.0, 0.0, 5.0])
    rec.capture()

    assert rec.frame_count == 2
    first, second = rec.payload()["frames"]
    assert first["matrices"]["/mover"][2][3] == 0.0
    assert second["matrices"]["/mover"][2][3] == 5.0
    # The still node is sampled too -- it is dropped later, at save time.
    assert "/still" in first["matrices"]


def test_child_tracks_are_world_matrices():
    """A moving parent must move its children's tracks, exactly as the static
    exporter composes them -- otherwise a URDF root move would go nowhere."""
    handles = {
        "/base": FakeHandle(BoxProps(), position=(0.0, 0.0, 0.0)),
        "/base/child": FakeHandle(simple_mesh_props(), position=(0.0, 0.0, 2.0)),
    }
    server = FakeServer(handles)
    rec = _record.Recorder(server, camera=False)
    rec.capture()
    handles["/base"]._impl.position = np.array([10.0, 0.0, 0.0])
    rec.capture()

    frames = rec.payload()["frames"]
    assert frames[0]["matrices"]["/base/child"][0][3] == 0.0
    assert frames[1]["matrices"]["/base/child"][0][3] == 10.0
    assert frames[1]["matrices"]["/base/child"][2][3] == 2.0


def test_save_writes_an_animation_block_and_drops_static_nodes(tmp_path):
    server, handles = two_node_server()
    rec = _record.Recorder(server, fps=30.0, camera=False)
    for z in (0.0, 1.0, 2.0):
        handles["/mover"]._impl.position = np.array([0.0, 0.0, z])
        rec.capture()

    bundle = rec.save(tmp_path / "take")
    manifest = json.loads((bundle / "scene.json").read_text())
    anim = manifest["animation"]

    assert anim["fps"] == 30.0
    assert anim["frame_count"] == 3
    assert anim["nodes"] == ["/mover"], "a node that never moved got a track"
    assert anim["has_camera"] is False

    data = np.load(bundle / anim["asset"])
    assert data["matrices"].shape == (3, 1, 4, 4)
    assert data["matrices"][:, 0, 2, 3].tolist() == [0.0, 1.0, 2.0]
    # The static bundle contract is untouched: nodes and assets still there.
    assert {n["name"] for n in manifest["nodes"]} == {"/still", "/mover"}


def test_camera_track_is_written_when_a_browser_is_connected(tmp_path):
    server, handles = two_node_server(camera=FakeCamera())
    rec = _record.Recorder(server, fps=24.0)
    rec.capture()
    handles["/mover"]._impl.position = np.array([0.0, 0.0, 1.0])
    rec.capture()

    anim = json.loads((rec.save(tmp_path / "take") / "scene.json").read_text())["animation"]
    assert anim["has_camera"] is True
    assert anim["camera_static"]["near"] == FakeCamera.near
    data = np.load(tmp_path / "take" / anim["asset"])
    assert data["camera_position"].shape == (2, 3)
    assert data["camera_fov"].shape == (2,)


def test_an_explicit_camera_overrides_the_browsers(tmp_path):
    """An authored camera move must reach the bundle as authored -- not as
    whatever the browser happened to be looking at when the frame was taken."""
    server, handles = two_node_server(camera=FakeCamera())
    rec = _record.Recorder(server, fps=24.0)
    for x in (0.0, 10.0):
        handles["/mover"]._impl.position = np.array([x, 0.0, 0.0])
        rec.capture(camera={"position": [x, -5.0, 2.0], "look_at": [0.0, 0.0, 1.0],
                            "up": [0.0, 0.0, 1.0], "fov": 0.5 + x / 100})

    bundle = rec.save(tmp_path / "take")
    anim = json.loads((bundle / "scene.json").read_text())["animation"]
    data = np.load(bundle / anim["asset"])
    assert anim["has_camera"] is True
    assert data["camera_position"][:, 0].tolist() == [0.0, 10.0]
    assert data["camera_fov"].tolist() == pytest.approx([0.5, 0.6])
    # Intrinsics it did not state fall back to the live camera.
    assert anim["camera_static"]["near"] == FakeCamera.near


def test_an_explicit_camera_works_with_no_browser_attached(tmp_path):
    server, _ = two_node_server(camera=None)
    rec = _record.Recorder(server, fps=24.0)
    for i in range(2):
        rec.capture(camera={"position": [i, 0.0, 0.0], "look_at": [0.0, 0.0, 0.0],
                            "up": [0.0, 0.0, 1.0], "fov": 0.7})
    anim = json.loads((rec.save(tmp_path / "take") / "scene.json").read_text())["animation"]
    assert anim["has_camera"] is True
    assert anim["camera_static"]["far"] == _record._CAMERA_DEFAULTS["far"]


def test_a_node_filtered_out_of_the_bundle_gets_no_track(tmp_path):
    """Tracks are intersected with what was actually exported: keying a node
    the Blender side never built would be a silent no-op at render time."""
    server, handles = two_node_server()
    rec = _record.Recorder(server, camera=False)
    for z in (0.0, 1.0):
        handles["/mover"]._impl.position = np.array([0.0, 0.0, z])
        rec.capture()

    bundle = rec.save(tmp_path / "take", node_filter=lambda name: name != "/mover")
    anim = json.loads((bundle / "scene.json").read_text())["animation"]
    assert anim["nodes"] == []


def test_saving_nothing_is_an_error(tmp_path):
    server, _ = two_node_server()
    with pytest.raises(RuntimeError, match="nothing recorded"):
        _record.Recorder(server).save(tmp_path / "take")


def test_fps_must_be_positive():
    server, _ = two_node_server()
    with pytest.raises(ValueError):
        _record.Recorder(server, fps=0)


def test_live_capture_collects_frames_and_stops():
    server, _ = two_node_server()
    rec = _record.Recorder(server, fps=120.0, camera=False)
    rec.start()
    assert rec.recording
    deadline = __import__("time").monotonic() + 2.0
    while rec.frame_count < 3 and __import__("time").monotonic() < deadline:
        __import__("time").sleep(0.01)
    count = rec.stop()
    assert count >= 3
    assert not rec.recording
    assert rec.frame_count == count, "sampling continued after stop()"
