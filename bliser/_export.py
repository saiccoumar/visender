"""Snapshot a live viser scene into a self-contained bundle on disk.

The bundle is a directory::

    my_scene.viserbundle/
        scene.json          manifest: nodes, world transforms, materials, camera
        assets/
            robot_link0.glb     GLB payloads (add_glb / add_mesh_trimesh)
            paper.npz           raw vertex/face/point arrays

``scene.json`` is the contract between this module (which needs viser) and
:mod:`bliser.blender_import` (which needs bpy). Neither imports the
other.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

# Nodes that exist purely to author the scene in the browser. They still act as
# parent frames (a light hanging off a transform gizmo inherits its pose), so
# their transforms are always read -- we just never emit geometry for them.
#
# Hiding one is an authoring action ("Show gizmos" off, to take the shot), not a
# statement about its children: a key light parented to a gizmo must survive.
# So these never gate the visibility of a subtree -- see ``visible`` below.
GIZMO_CONTROL_PROPS = frozenset({"TransformControlsProps"})

GIZMO_PROPS = frozenset(
    {
        "TransformControlsProps",
        "LabelProps",
        "Gui3DProps",
        "GridProps",
        "FrameProps",
        "BatchedAxesProps",
        "CameraFrustumProps",
        "ImageProps",
    }
)


def _quat_to_mat(wxyz) -> np.ndarray:
    """wxyz quaternion -> 4x4 homogeneous matrix."""
    w, x, y, z = (float(v) for v in wxyz)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(4)
    w, x, y, z = w / n, x / n, y / n, z / n
    m = np.eye(4)
    m[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return m


def _local_matrix(handle) -> np.ndarray:
    m = _quat_to_mat(handle._impl.wxyz)
    m[:3, 3] = np.asarray(handle._impl.position, dtype=float)
    return m


def _slug(name: str) -> str:
    return name.strip("/").replace("/", "__") or "root"


def _ancestors(name: str) -> list[str]:
    """Parent paths of ``/a/b/c``, outermost first: ['/a', '/a/b']."""
    parts = name.strip("/").split("/")
    return ["/" + "/".join(parts[:i]) for i in range(1, len(parts))]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _scalar_props(props) -> dict:
    """Every prop that is small enough to inline into scene.json."""
    out = {}
    for field in dataclasses.fields(props):
        value = getattr(props, field.name)
        if isinstance(value, (bytes, bytearray)):
            continue
        if isinstance(value, np.ndarray) and value.size > 16:
            continue
        out[field.name] = _jsonable(value)
    return out


def _camera_dict(server) -> dict | None:
    """Pose of the first connected browser client, so Blender opens on the
    same view the shot was framed in."""
    clients = server.get_clients()
    if not clients:
        return None
    cam = next(iter(clients.values())).camera
    return {
        "position": _jsonable(cam.position),
        "look_at": _jsonable(cam.look_at),
        "up": _jsonable(cam.up_direction),
        "fov": float(cam.fov),  # vertical, radians
        "aspect": float(cam.aspect),
        "near": float(cam.near),
        "far": float(cam.far),
        "image_width": int(cam.image_width),
        "image_height": int(cam.image_height),
    }


def export_scene(
    server,
    out_dir: str | Path,
    *,
    include_gizmos: bool = False,
    include_hidden: bool = False,
    node_filter: Callable[[str], bool] | None = None,
    environment_map: str | None = None,
    extras: dict | None = None,
) -> Path:
    """Write every asset and pose in ``server``'s scene to ``out_dir``.

    Args:
        server: A live :class:`viser.ViserServer`.
        out_dir: Bundle directory. Replaced if it already exists.
        include_gizmos: Emit transform controls, labels, grids and frames as
            geometry. Off by default -- those are alignment aids, not subjects.
        include_hidden: Emit nodes hidden in the browser (a node is hidden if
            it or any ancestor has ``visible = False``).
        node_filter: Return False to drop a node by its ``/path`` name. Applied
            after the gizmo and visibility rules.
        environment_map: The preset last passed to
            ``scene.configure_environment_map``. viser does not retain it, so
            pass it here if you want Blender's world to match.
        extras: Arbitrary JSON-safe dict stored under ``"extras"``.

    Returns:
        Path to the bundle directory.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    assets = out_dir / "assets"
    assets.mkdir(parents=True)

    handles = dict(server.scene._handle_from_node_name)

    def visible(name: str) -> bool:
        for path in [*_ancestors(name), name]:
            h = handles.get(path)
            if h is None or h._impl.visible:
                continue
            # A hidden transform gizmo means "I am done aligning this", not
            # "drop everything under me". Unticking a "Show gizmos" checkbox
            # must not silently delete the lights parented to it.
            if path != name and type(h._impl.props).__name__ in GIZMO_CONTROL_PROPS:
                continue
            return False
        return True

    def world_matrix(name: str) -> np.ndarray:
        m = np.eye(4)
        for path in [*_ancestors(name), name]:
            h = handles.get(path)
            if h is not None:
                m = m @ _local_matrix(h)
        return m

    nodes: list[dict] = []
    skipped: list[str] = []
    for name, handle in sorted(handles.items()):
        props = handle._impl.props
        kind = type(props).__name__

        if kind in GIZMO_PROPS and not include_gizmos:
            continue
        if not include_hidden and not visible(name):
            continue
        if node_filter is not None and not node_filter(name):
            continue

        node = {
            "name": name,
            "kind": kind,
            "matrix": world_matrix(name).tolist(),
            "props": _scalar_props(props),
        }

        # Bulk payloads go beside the manifest; the manifest just points at them.
        slug = _slug(name)
        if kind in ("GlbProps", "BatchedGlbProps"):
            path = assets / f"{slug}.glb"
            path.write_bytes(props.glb_data)
            node["asset"] = path.relative_to(out_dir).as_posix()
        elif kind in ("MeshProps", "SkinnedMeshProps", "BatchedMeshesProps"):
            path = assets / f"{slug}.npz"
            np.savez_compressed(
                path,
                vertices=np.asarray(props.vertices, np.float32),
                faces=np.asarray(props.faces, np.uint32),
            )
            node["asset"] = path.relative_to(out_dir).as_posix()
        elif kind == "PointCloudProps":
            path = assets / f"{slug}.npz"
            np.savez_compressed(
                path,
                points=np.asarray(props.points, np.float32),
                colors=np.broadcast_to(
                    np.asarray(props.colors, np.uint8), props.points.shape
                ).copy(),
            )
            node["asset"] = path.relative_to(out_dir).as_posix()
        elif kind in (
            "CatmullRomSplineProps",
            "CubicBezierSplineProps",
            "LineSegmentsProps",
        ):
            path = assets / f"{slug}.npz"
            arrays = {"points": np.asarray(props.points, np.float32)}
            if kind == "LineSegmentsProps":
                arrays["colors"] = np.asarray(props.colors, np.uint8)
            np.savez_compressed(path, **arrays)
            node["asset"] = path.relative_to(out_dir).as_posix()
        elif kind in ("BoxProps", "IcosphereProps") or kind.endswith("LightProps"):
            pass  # Fully described by its scalar props.
        else:
            skipped.append(f"{name} ({kind})")
            continue

        nodes.append(node)

    manifest = {
        "format": "bliser",
        "version": 1,
        "up_direction": "+z",
        "nodes": nodes,
        "camera": _camera_dict(server),
        "environment_map": environment_map,
        "extras": extras or {},
    }
    (out_dir / "scene.json").write_text(json.dumps(manifest, indent=1))

    print(f"[bliser] wrote {len(nodes)} nodes -> {out_dir}")
    if skipped:
        print(f"[bliser] no Blender equivalent, skipped: {', '.join(skipped)}")
    return out_dir


def add_export_button(
    server,
    *,
    out_dir: str | Path = "blender_export",
    label: str = "Export to Blender",
    timestamp: bool = True,
    **export_kwargs,
):
    """Drop an "Export to Blender" button into the viser GUI.

    Everything after ``label`` is forwarded to :func:`export_scene`, so the
    button inherits the same filtering options::

        import bliser
        bliser.add_export_button(server, out_dir="renders/pen_grip",
                                        environment_map="city")

    Any keyword may instead be a *zero-argument* callable, evaluated on click.
    Use it to read GUI state that changes after the button is created::

        bliser.add_export_button(
            server, environment_map=lambda: gui_env.value)

    Callables that require arguments are passed through untouched, so an
    option that is itself a function — ``node_filter=lambda name: ...`` —
    still works.

    Args:
        timestamp: Suffix each export with ``_<HHMMSS>`` so repeated clicks
            accumulate takes instead of overwriting one.

    Returns:
        The viser button handle.
    """
    button = server.gui.add_button(label, icon=_icon())
    status = server.gui.add_text("Last export", initial_value="(none)", disabled=True)

    @button.on_click
    def _(_) -> None:
        import time

        target = Path(out_dir)
        if timestamp:
            target = target.with_name(f"{target.name}_{time.strftime('%H%M%S')}")
        resolved = {k: _thunk(v) for k, v in export_kwargs.items()}
        button.disabled = True
        try:
            export_scene(server, target, **resolved)
            status.value = str(target)
        except Exception as exc:  # surface the failure in the browser, not just stdout
            status.value = f"FAILED: {exc}"
            raise
        finally:
            button.disabled = False

    return button


def _thunk(value):
    """Evaluate ``value`` if it is a zero-argument callable, else return it.

    ``node_filter`` is itself a callable, so "callable means deferred value"
    is not enough — only something callable with *no* arguments is a thunk.
    """
    if not callable(value):
        return value
    import inspect

    try:
        inspect.signature(value).bind()
    except (TypeError, ValueError):
        return value
    return value()


def _icon():
    try:
        from viser import Icon

        return Icon.CAMERA
    except Exception:
        return None
