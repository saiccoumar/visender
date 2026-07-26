"""Record a viser scene over time and save it as an animated bundle.

A bundle is normally one instant: every node's world matrix as it stood when
you clicked *Export*. A :class:`Recorder` samples those same matrices
repeatedly and stores the stack alongside the (single) set of assets, so the
Blender side can key every object instead of just placing it::

    rec = visender.Recorder(server, fps=24)
    for cfg in trajectory:
        viser_urdf.update_cfg(cfg)
        rec.capture()
    rec.save("out/gundam_video")

Assets are written once, from the scene as it stands at ``save()`` time --
geometry is assumed not to change during a take, only poses (and the camera).
That is what a robot trajectory is, and it keeps a 500-frame take the same size
on disk as a still.

There are two ways to drive it:

* **Frame-stepped** (above): you advance the scene, then call
  :meth:`Recorder.capture`. Deterministic, and the wall-clock time a frame took
  to compute is irrelevant -- ``fps`` is purely how fast it plays back.
* **Live** (:meth:`Recorder.start` / :func:`add_record_button`): a background
  thread samples at ``fps`` while you drag things in the browser.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ._export import _camera_dict, _local_matrix, _ancestors, export_scene


def _sample_world_matrices(server) -> dict[str, np.ndarray]:
    """Every scene node's world matrix, right now.

    Composed the same way :func:`visender.export_scene` composes them (down the
    full ``/a/b/c`` path), so a track lines up with the static matrix in
    ``scene.json`` for the frame it was captured on.
    """
    handles = dict(server.scene._handle_from_node_name)  # noqa: SLF001
    local: dict[str, np.ndarray] = {}
    for name, handle in handles.items():
        local[name] = _local_matrix(handle)

    out: dict[str, np.ndarray] = {}
    for name in handles:
        m = np.eye(4)
        for path in [*_ancestors(name), name]:
            if path in local:
                m = m @ local[path]
        out[name] = m
    return out


# Enough of a camera to render from when a take was authored with no browser
# attached. Only the intrinsics: an explicit camera always brings its own pose.
_CAMERA_DEFAULTS = {
    "position": [4.0, -4.0, 3.0], "look_at": [0.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0],
    "fov": 0.75, "aspect": 16 / 9, "near": 0.1, "far": 1000.0,
    "image_width": 1920, "image_height": 1080,
}


def _merge_camera(explicit: dict, live: dict | None) -> dict:
    """An authored camera pose over whatever intrinsics are available."""
    camera = dict(_CAMERA_DEFAULTS)
    if live:
        camera.update(live)
    camera.update(explicit)
    return camera


class Recorder:
    """Samples node poses (and the browser camera) into an animation track.

    Args:
        server: A live :class:`viser.ViserServer`.
        fps: Playback rate stored in the bundle. In frame-stepped mode this is
            purely a playback property; in live mode it is also the sampling
            rate.
        camera: Also record the connected browser camera each frame, so a
            camera move made in the browser is rendered as a camera move.
    """

    def __init__(self, server, *, fps: float = 24.0, camera: bool = True):
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps!r}")
        self.server = server
        self.fps = float(fps)
        self.record_camera = bool(camera)
        self._frames: list[dict[str, Any]] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- sampling ---------------------------------------------------------- #
    def capture(self, camera: dict | None = None) -> int:
        """Sample the scene once. Returns the new frame count.

        Args:
            camera: Use this camera for the frame instead of sampling the
                browser's. Pass ``{"position": ..., "look_at": ..., "up": ...,
                "fov": ...}`` to record an *authored* camera move -- a
                keyframed one, say. Anything it leaves out (near/far, aspect,
                image size) is filled from the live browser camera, or from
                defaults when no browser is attached.
        """
        frame: dict[str, Any] = {"matrices": _sample_world_matrices(self.server)}
        if camera is not None:
            frame["camera"] = _merge_camera(camera, _camera_dict(self.server))
        elif self.record_camera:
            frame["camera"] = _camera_dict(self.server)
        self._frames.append(frame)
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def duration(self) -> float:
        """Playback length in seconds at the recorder's fps."""
        return len(self._frames) / self.fps

    # -- live capture ------------------------------------------------------ #
    def start(self) -> None:
        """Begin sampling at ``fps`` on a background thread."""
        if self._thread is not None:
            return
        self._stop.clear()

        def loop() -> None:
            period = 1.0 / self.fps
            next_at = time.monotonic()
            while not self._stop.is_set():
                self.capture()
                next_at += period
                # Sampling is cheap, but if a frame did overrun, drop the debt
                # rather than sprinting to catch up (which would compress time).
                self._stop.wait(max(0.0, next_at - time.monotonic()))
                next_at = max(next_at, time.monotonic())

        self._thread = threading.Thread(target=loop, daemon=True,
                                        name="visender-recorder")
        self._thread.start()

    def stop(self) -> int:
        """Stop live sampling. Returns the frame count."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return len(self._frames)

    @property
    def recording(self) -> bool:
        return self._thread is not None

    # -- output ------------------------------------------------------------ #
    def payload(self) -> dict:
        """The animation block handed to :func:`visender.export_scene`."""
        return {"fps": self.fps, "frames": list(self._frames)}

    def save(self, out_dir: str | Path, **export_kwargs) -> Path:
        """Write an animated bundle: assets as they stand now, plus the track.

        Extra keyword arguments are forwarded to
        :func:`visender.export_scene` (``node_filter``, ``environment_map`` and
        the rest), so a recording is filtered exactly like a still.
        """
        if not self._frames:
            raise RuntimeError(
                "nothing recorded: call capture() (or start()/stop()) first.")
        return export_scene(self.server, out_dir,
                            animation=self.payload(), **export_kwargs)


def add_record_button(
    server,
    *,
    out_dir: str | Path = "blender_recording",
    fps: float = 24.0,
    label: str = "Record",
    timestamp: bool = True,
    **export_kwargs,
):
    """Drop record/stop/export controls into the viser GUI.

    Press **Record**, move the scene (or the camera) in the browser, press
    **Stop**, then **Export recording** to write an animated bundle. Everything
    after ``timestamp`` is forwarded to :func:`visender.export_scene`.

    Returns:
        The :class:`Recorder`, so a script can also drive it programmatically.
    """
    recorder = Recorder(server, fps=fps)

    rate = server.gui.add_number("Record FPS", initial_value=float(fps),
                                 min=1.0, max=240.0, step=1.0)
    toggle = server.gui.add_button(label, icon=_icon("PLAYER_RECORD"))
    export = server.gui.add_button("Export recording", icon=_icon("CAMERA"))
    status = server.gui.add_text("Recording", initial_value="0 frames",
                                 disabled=True)
    export.disabled = True

    def refresh() -> None:
        status.value = (f"{recorder.frame_count} frames "
                        f"({recorder.duration:.1f}s @ {recorder.fps:g} fps)")
        export.disabled = recorder.frame_count == 0

    @toggle.on_click
    def _(_) -> None:
        if recorder.recording:
            recorder.stop()
            toggle.label, rate.disabled = label, False
        else:
            recorder.fps = float(rate.value)
            recorder.clear()
            recorder.start()
            toggle.label, rate.disabled = "Stop", True
        refresh()

    @export.on_click
    def _(_) -> None:
        if recorder.recording:
            recorder.stop()
            toggle.label, rate.disabled = label, False
        target = Path(out_dir)
        if timestamp:
            target = target.with_name(f"{target.name}_{time.strftime('%H%M%S')}")
        export.disabled = True
        try:
            recorder.save(target, **{k: _thunk(v) for k, v in export_kwargs.items()})
            status.value = str(target)
        except Exception as exc:  # surface it in the browser, not just stdout
            status.value = f"FAILED: {exc}"
            raise
        finally:
            export.disabled = False

    return recorder


def _thunk(value):
    from ._export import _thunk as thunk

    return thunk(value)


def _icon(name: str):
    try:
        from viser import Icon

        return getattr(Icon, name, None)
    except Exception:
        return None
