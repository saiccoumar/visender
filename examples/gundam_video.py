"""visender example: pose the RX-78, keyframe it, record the take as video.

    python examples/gundam_video.py [--shot examples/gundam_video_shot.yaml]

**You author the motion in the browser.** Click a body part to grab the joint
that drives it and dial it in (exactly like `gundam.py`), then press *Capture
keyframe* to pin that pose at a time on the timeline. A handful of poses is a
shot: the script samples them onto exact 1/fps steps with linear interpolation
in between, so five poses become a few hundred configurations.

The loop is:

1. **Pose** tab — click the robot, move joints.
2. **Keyframes** tab — set a time, *Capture new keyframe*. Repeat.
3. Fix any keyframe individually: select it, *Edit this keyframe* to load its
   stored pose and camera, adjust, then *Update from current view* to write
   both back at the same time. There is also *Revert*, *Retime*, *Duplicate*,
   *Insert breakdown after* (seeded with the pose already passing through, so
   it changes nothing until you move something), *Delete*, and — per joint —
   *Reset joint to base pose*.
4. **Preview** tab — scrub the interpolated frames, or *Play* them at the
   shot's fps. This is the review step, and it is free: nothing is written.
5. *Export shot* — writes the previewed take as an animated bundle
   Blender renders as a movie.

**The camera is keyframed too.** Capturing or updating a keyframe stores the
view you are looking from — scrubbing and playback fly the browser along it,
and the exported bundle carries it into Blender. Only keyframes that actually
hold a camera join that track, so you can frame two or three moments and leave
the rest to the body; before the first and after the last it holds. Untick
*Save camera with keyframe* to author the body alone, or *Clear this
keyframe's camera* to drop one.

The camera does **not** interpolate the way the joints do. Joint angles are
independent scalars and stay linear, which is honest about what was authored.
The camera is a path and an aim: its eye follows a cubic spline, and its aim is
interpolated as a *rotation* (spherical cubic — squad), so the view neither
corners at the keys nor swings fast through the middle of a turn. Set
`camera_interpolation: linear` in the shot file for the old behaviour.

*Save shot YAML* writes the keyframes to disk so a take is reproducible and
re-editable; `--shot` loads one back. The YAML is also perfectly editable by
hand if you would rather type radians — press *Reload shot* to pick up the
edit. See `docs/animation.md` for the render side.

Interpolation is linear in joint space, and every keyframe is a *complete*
pose: a joint it does not name sits at `base_pose` for that instant rather than
holding whatever came before. That way each keyframe reads on its own, and the
saved YAML only lists what you actually changed.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import viser
import yaml
from viser.extras import ViserUrdf

import visender

# The scene, the URDF loader, the click-to-pose panel and the light panel are
# gundam.py's; this script adds the keyframe editor and the timeline. Works
# however the file is invoked, not just from a shell sitting in examples/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gundam import (  # noqa: E402
    DEFAULT_URDF,
    HOME_POSE,
    add_joint_picker,
    add_light_panel,
    home_cfg,
    load_urdf,
)

DEFAULT_SHOT = Path(__file__).parent / "gundam_video_shot.yaml"
DEFAULT_OUT = Path(__file__).parent / "out/gundam_video"

# Two keyframes closer together than this are the same keyframe: capturing at
# t=1.0 when one already sits at t=1.0 replaces it rather than stacking a
# second pose the interpolator would have to choose between.
_SAME_TIME = 1e-6

# What the FPS slider can reach. A shot file or --fps may sit outside it (or
# between its steps); the slider clamps for display and the shot keeps the real
# value until you actually drag it.
FPS_MIN, FPS_MAX = 1.0, 120.0


def _fps_on_slider(fps: float) -> float:
    return float(min(FPS_MAX, max(FPS_MIN, round(float(fps)))))


# --------------------------------------------------------------------------- #
# Shot file
# --------------------------------------------------------------------------- #

class Shot:
    """A shot: ``fps``, a base pose, and keyframes sorted by time.

    Round-trips through YAML, so a take authored by clicking can be re-opened,
    diffed and hand-edited.
    """

    def __init__(self, fps: float = 24.0, keyframes: list[dict] | None = None,
                 base: dict[str, float] | None = None,
                 camera_interpolation: str = "cubic"):
        if float(fps) <= 0:
            raise SystemExit(f"fps must be positive, got {fps!r}")
        if camera_interpolation not in ("cubic", "linear"):
            raise SystemExit("camera_interpolation must be 'cubic' or 'linear', "
                             f"got {camera_interpolation!r}")
        self.fps = float(fps)
        self.base = dict(HOME_POSE if base is None else base)
        self.camera_interpolation = camera_interpolation
        self.keyframes = sorted(list(keyframes or []), key=lambda k: float(k["t"]))

    @property
    def duration(self) -> float:
        if not self.keyframes:
            return 0.0
        return float(self.keyframes[-1]["t"]) - float(self.keyframes[0]["t"])

    @property
    def times(self) -> list[float]:
        return [float(k["t"]) for k in self.keyframes]

    # -- editing ----------------------------------------------------------- #
    def put(self, t: float, joints: dict[str, float],
            camera: dict | None = None) -> int:
        """Add a keyframe at ``t`` (replacing one already there). Returns its index."""
        t = float(t)
        self.keyframes = [k for k in self.keyframes
                          if abs(float(k["t"]) - t) > _SAME_TIME]
        keyframe: dict = {"t": t, "joints": dict(joints)}
        if camera:
            keyframe["camera"] = dict(camera)
        self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda k: float(k["t"]))
        return self.times.index(t)

    def update(self, index: int, joints: dict[str, float],
               camera: dict | None = None) -> None:
        """Rewrite a keyframe's pose in place, keeping its time.

        ``camera=None`` leaves whatever camera the keyframe already had, so
        re-posing a keyframe never silently loses the shot you framed for it.
        """
        self.keyframes[index]["joints"] = dict(joints)
        if camera:
            self.keyframes[index]["camera"] = dict(camera)

    def clear_camera(self, index: int) -> None:
        self.keyframes[index].pop("camera", None)

    def delete(self, index: int) -> None:
        del self.keyframes[index]

    def retime(self, index: int, t: float) -> int:
        keyframe = self.keyframes.pop(index)
        return self.put(t, keyframe["joints"])

    def drop_joint(self, index: int, joint: str) -> None:
        """Stop a keyframe naming a joint, so it falls back to the base pose."""
        self.keyframes[index]["joints"].pop(joint, None)

    def gap_after(self, index: int) -> float:
        """A sensible time offset for a keyframe inserted after this one:
        halfway to the next, or one more of the previous gap at the end."""
        times = self.times
        if index + 1 < len(times):
            return (times[index + 1] - times[index]) / 2
        if index > 0:
            return times[index] - times[index - 1]
        return 0.5

    # -- disk -------------------------------------------------------------- #
    @classmethod
    def load(cls, path: Path) -> "Shot":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - {"fps", "keyframes", "base_pose",
                              "camera_interpolation"}
        if unknown:
            raise SystemExit(f"{path}: unknown key(s) {sorted(unknown)}; expected "
                             "fps, base_pose, camera_interpolation, keyframes")
        for k in raw.get("keyframes") or []:
            if "t" not in k:
                raise SystemExit(f"{path}: a keyframe has no 't'")
            stray = set(k) - {"t", "joints", "camera"}
            if stray:
                raise SystemExit(f"{path}: keyframe t={k['t']} has unknown "
                                 f"key(s) {sorted(stray)}")
            if "camera" in k:
                missing = {"position", "look_at"} - set(k["camera"] or {})
                if missing:
                    raise SystemExit(f"{path}: keyframe t={k['t']} camera is "
                                     f"missing {sorted(missing)}")
        return cls(raw.get("fps", 24), raw.get("keyframes"), raw.get("base_pose"),
                   raw.get("camera_interpolation", "cubic"))

    def dump(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        def entry(k: dict) -> dict:
            out = {"t": round(float(k["t"]), 4)}
            if k.get("camera"):
                out["camera"] = {
                    key: ([round(float(c), 4) for c in value]
                          if isinstance(value, (list, tuple, np.ndarray))
                          else round(float(value), 5))
                    for key, value in k["camera"].items()
                }
            out["joints"] = {n: round(float(v), 4) for n, v in k["joints"].items()}
            return out

        doc = {
            "fps": self.fps,
            "camera_interpolation": self.camera_interpolation,
            "base_pose": {k: round(float(v), 4) for k, v in self.base.items()},
            "keyframes": [entry(k) for k in self.keyframes],
        }
        header = ("# Shot file for examples/gundam_video.py — written from the "
                  "browser.\n#\n#   python examples/gundam_video.py --shot "
                  f"{path.name}\n#\n# Angles are radians. Every keyframe is a "
                  "complete pose: a joint it does not\n# name sits at "
                  "`base_pose` at that instant. Hand-edit freely, then press\n"
                  "# 'Reload shot' in the browser.\n\n")
        path.write_text(header + yaml.safe_dump(doc, sort_keys=False))
        return path

    def write_fps(self, path: Path) -> Path:
        """Push just ``fps`` back into an existing shot file, in place.

        Dialling the slider is meant to change the shot's rate on disk, not to
        be a back-door 'Save' -- so this rewrites the one top-level ``fps:``
        line and leaves every keyframe, the base pose and the hand-written
        comments exactly as they were. Keyframe edits still need Save.

        A file that does not exist yet (or somehow has no ``fps:``) is written
        in full instead, since there is nothing to preserve.
        """
        path = Path(path)
        if not path.exists():
            return self.dump(path)
        lines = path.read_text().splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.startswith("fps:"):
                lines[i] = f"fps: {self.fps:g}\n"
                path.write_text("".join(lines))
                return path
        return self.dump(path)


def sample(shot: Shot, urdf) -> np.ndarray:
    """Interpolate the shot onto ``fps`` frames: (n_frames, n_joints).

    Every joint is resolved against the base pose first, so a keyframe only has
    to name what it changes, and the result is clipped to the URDF's own limits
    (which makes a shot survive an ``--urdf`` swap, and a mistyped angle merely
    saturate instead of exploding the model).
    """
    key_times, key_cfgs = keyframe_cfgs(shot, urdf)
    if key_times is None:
        return clip_to_limits(base_cfg(shot, urdf)[None, :], urdf)

    # Frame times land on exact 1/fps steps, and the last keyframe is always
    # hit: a 4.5 s shot at 24 fps is 109 frames, not 108 plus a lost pose.
    n_frames = max(1, int(round(shot.duration * shot.fps)) + 1)
    times = key_times[0] + np.arange(n_frames) / shot.fps
    times[-1] = key_times[-1]
    return clip_to_limits(interpolate(times, key_times, key_cfgs), urdf)


def base_cfg(shot: Shot, urdf) -> np.ndarray:
    """The base pose as a configuration vector."""
    index = {name: i for i, name in enumerate(urdf.actuated_joint_names)}
    cfg = np.zeros(len(index))
    for name, value in shot.base.items():
        if name in index:
            cfg[index[name]] = float(value)
    return cfg


def keyframe_cfgs(shot: Shot, urdf):
    """``(times, cfgs)`` for the shot's keyframes, or ``(None, None)`` if empty.

    Each keyframe resolves against the base pose, so it only has to name the
    joints it changes.
    """
    if not shot.keyframes:
        return None, None
    names = urdf.actuated_joint_names
    index = {name: i for i, name in enumerate(names)}
    base = base_cfg(shot, urdf)

    times, cfgs = [], []
    for k in shot.keyframes:
        cfg = base.copy()
        for name, value in (k.get("joints") or {}).items():
            if name not in index:
                raise SystemExit(
                    f"keyframe t={k['t']}: no actuated joint {name!r}. "
                    f"Names come from the URDF, e.g. {names[:3]}...")
            cfg[index[name]] = float(value)
        times.append(float(k["t"]))
        cfgs.append(cfg)
    return np.asarray(times), np.asarray(cfgs)


def interpolate(times, key_times, key_cfgs) -> np.ndarray:
    """Linear interpolation of the keyframe poses at arbitrary times."""
    times = np.atleast_1d(times)
    return np.stack(
        [np.interp(times, key_times, key_cfgs[:, j])
         for j in range(key_cfgs.shape[1])],
        axis=1,
    )


def clip_to_limits(cfgs: np.ndarray, urdf) -> np.ndarray:
    """Clip to the URDF's own limits.

    Which is what makes a shot survive an ``--urdf`` swap, and a mistyped angle
    merely saturate instead of exploding the model.
    """
    names = urdf.actuated_joint_names
    lower = np.array([_limit(urdf, n, "lower") for n in names])
    upper = np.array([_limit(urdf, n, "upper") for n in names])
    return np.clip(cfgs, lower, upper)


def pose_at(shot: Shot, urdf, t: float) -> np.ndarray:
    """The interpolated pose at an arbitrary time, off the frame grid.

    Inserting a breakdown keyframe uses this: seeded with the pose the shot
    already passes through at that instant, adding one changes nothing until
    you actually move a joint.
    """
    key_times, key_cfgs = keyframe_cfgs(shot, urdf)
    if key_times is None:
        return clip_to_limits(base_cfg(shot, urdf), urdf)
    return clip_to_limits(interpolate([float(t)], key_times, key_cfgs)[0], urdf)


def keyframe_pose(shot: Shot, urdf, index: int) -> np.ndarray:
    """The pose *stored* in a keyframe -- not the nearest sampled frame.

    A keyframe at t=0.15 with fps=24 sits between frames 3 and 4, so reading it
    off the timeline would hand back an interpolated neighbour and editing that
    would quietly rewrite the pose.
    """
    _, key_cfgs = keyframe_cfgs(shot, urdf)
    return clip_to_limits(key_cfgs[index], urdf)


def _limit(urdf, name: str, which: str) -> float:
    limit = urdf.joint_map[name].limit
    if limit is None:
        return -np.inf if which == "lower" else np.inf
    return float(getattr(limit, which))


# --------------------------------------------------------------------------- #
# Smooth interpolation
# --------------------------------------------------------------------------- #
# Joint angles interpolate linearly -- a robot pose is a set of independent
# scalars and linear is honest about what was authored. A camera is not: the
# eye traces a path through space and the lens points along a direction, and
# doing either linearly gives a corner at every key. So the camera gets a cubic
# spline for its position and a spherical one for its aim.


def hermite(times: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
    """Cubic Hermite spline through ``values`` at ``times``, evaluated at ``t``.

    Tangents are Catmull-Rom finite differences taken over the *actual* key
    spacing, so keys 0.1 s apart and keys 2 s apart both behave; the ends use a
    one-sided difference. The curve passes exactly through every key (it is an
    interpolating spline, not an approximating one), and with only two keys the
    tangents are the secant, so it degenerates to the linear result rather than
    inventing an ease.

    A smooth spline can overshoot between keys -- that is what makes a camera
    move feel like a camera move rather than a series of ramps, but it is why
    the *joints* do not use it.
    """
    values = np.asarray(values, float)
    n = len(times)
    if n == 1:
        return values[0]
    t = float(np.clip(t, times[0], times[-1]))

    tangents = np.empty_like(values)
    tangents[0] = (values[1] - values[0]) / (times[1] - times[0])
    tangents[-1] = (values[-1] - values[-2]) / (times[-1] - times[-2])
    for i in range(1, n - 1):
        tangents[i] = (values[i + 1] - values[i - 1]) / (times[i + 1] - times[i - 1])

    i = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, n - 2))
    h = times[i + 1] - times[i]
    s = (t - times[i]) / h
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * values[i]
            + (s3 - 2 * s2 + s) * h * tangents[i]
            + (-2 * s3 + 3 * s2) * values[i + 1]
            + (s3 - s2) * h * tangents[i + 1])


# ---- unit quaternions (wxyz), enough for a spherical spline ---------------- #

def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_log(q):
    """Log of a unit quaternion -> the rotation vector half-angle, as xyz."""
    v = np.asarray(q[1:], float)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.zeros(3)
    return v / norm * math.atan2(norm, float(np.clip(q[0], -1.0, 1.0)))


def quat_exp(v):
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[math.cos(theta)], np.asarray(v) / theta * math.sin(theta)])


def quat_slerp(q0, q1, s: float):
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:                      # take the short way round
        q1, dot = -q1, -dot
    if dot > 0.9995:                   # nearly parallel: lerp and renormalise
        out = q0 + s * (q1 - q0)
        return out / np.linalg.norm(out)
    theta = math.acos(np.clip(dot, -1.0, 1.0))
    sin_theta = math.sin(theta)
    return (math.sin((1 - s) * theta) * q0 + math.sin(s * theta) * q1) / sin_theta


def squad(quats: np.ndarray, times: np.ndarray, t: float) -> np.ndarray:
    """Spherical cubic interpolation (Shoemake's squad) through ``quats``.

    Slerping key to key would already keep the *rate* sensible, but it is only
    C0: the angular velocity jumps at every key, which is exactly the flick
    that makes a keyed camera look mechanical. Squad blends the slerp with a
    second one through control quaternions derived from each key's neighbours,
    which makes the rotation C1 -- smooth through the keys, not just between
    them.
    """
    n = len(quats)
    if n == 1:
        return quats[0]
    t = float(np.clip(t, times[0], times[-1]))
    i = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, n - 2))
    s = (t - times[i]) / (times[i + 1] - times[i])

    def control(j):
        """The tangent quaternion at key j, from its two neighbours."""
        if j == 0 or j == n - 1:
            return quats[j]
        inv = quat_conj(quats[j])
        prev = quat_log(quat_mul(inv, quats[j - 1]))
        nxt = quat_log(quat_mul(inv, quats[j + 1]))
        return quat_mul(quats[j], quat_exp(-(prev + nxt) / 4.0))

    a, b = control(i), control(i + 1)
    return quat_slerp(quat_slerp(quats[i], quats[i + 1], s),
                      quat_slerp(a, b, s), 2 * s * (1 - s))


def camera_quat(position, look_at, up) -> np.ndarray:
    """Orientation of a camera at ``position`` aimed at ``look_at``.

    Same basis the Blender importer builds (looking down local -Z, +Y up), so
    the orientation this interpolates is the one that gets rendered.
    """
    forward = np.asarray(look_at, float) - np.asarray(position, float)
    norm = np.linalg.norm(forward)
    forward = forward / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    up = np.asarray(up, float)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:   # aim parallel to up: pick any right
        right = np.cross(forward, [0.0, 1.0, 0.0])
        if np.linalg.norm(right) < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return _quat_from_matrix(np.column_stack([right, true_up, -forward]))


def camera_forward(q) -> np.ndarray:
    """The aim direction of an orientation from :func:`camera_quat`."""
    w, x, y, z = q
    # Third column of the rotation matrix, negated (the camera looks down -Z).
    return -np.array([2 * (x * z + y * w),
                      2 * (y * z - x * w),
                      1 - 2 * (x * x + y * y)])


def _quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> wxyz, branching on the largest diagonal term so the
    square root never lands on a near-zero denominator."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q = [0.25 / s, (m[2, 1] - m[1, 2]) * s,
             (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
             (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
             0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
             (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q, float)
    return q / np.linalg.norm(q)


def align_hemispheres(quats: np.ndarray) -> np.ndarray:
    """q and -q are the same orientation but interpolate opposite ways round;
    flip each key into the same hemisphere as the one before it."""
    out = [np.asarray(quats[0], float)]
    for q in quats[1:]:
        q = np.asarray(q, float)
        out.append(-q if np.dot(out[-1], q) < 0 else q)
    return np.array(out)


# --------------------------------------------------------------------------- #
# Camera track
# --------------------------------------------------------------------------- #
# The pose and the camera are separate tracks over the same timeline. A keyframe
# only joins the camera track if it actually stores a camera, so you can frame
# the shot at two or three moments and leave the rest of the keyframes to the
# body -- exactly how you would key a camera in any animation tool.

CAMERA_VECTORS = ("position", "look_at", "up")


def camera_track(shot: Shot):
    """``(times, {field: array})`` over the keyframes that carry a camera."""
    keyed = [k for k in shot.keyframes if k.get("camera")]
    if not keyed:
        return None, None
    times = np.array([float(k["t"]) for k in keyed])
    track = {field: np.array([[float(v) for v in k["camera"][field]] for k in keyed])
             for field in CAMERA_VECTORS if field in keyed[0]["camera"]}
    if "fov" in keyed[0]["camera"]:
        track["fov"] = np.array([float(k["camera"]["fov"]) for k in keyed])
    return times, track


def camera_at(shot: Shot, t: float) -> dict | None:
    """The camera at an arbitrary time, interpolated between the camera keys.

    Before the first camera key and after the last one it holds, so a camera
    keyed only in the middle of a take still gives every frame a pose.

    With ``camera_interpolation: cubic`` (the default) the eye follows a cubic
    spline and the *aim* is interpolated as a rotation rather than as a point:
    slerping the orientation, not lerping the look-at, is what stops the view
    from swinging fast through the middle of a turn and snapping at the keys.
    The look-at is rebuilt from the interpolated orientation and its own
    splined distance, so it stays a look-at for viser and for the bundle.
    """
    times, track = camera_track(shot)
    if times is None:
        return None
    t = float(t)

    if shot.camera_interpolation == "linear" or len(times) == 1:
        camera: dict = {}
        for field, values in track.items():
            if values.ndim == 1:
                camera[field] = float(np.interp(t, times, values))
            else:
                camera[field] = [float(np.interp(t, times, values[:, i]))
                                 for i in range(values.shape[1])]
        return camera

    position = hermite(times, track["position"], t)
    camera = {"position": [float(v) for v in position]}

    if "look_at" in track:
        up = track.get("up", np.tile([0.0, 0.0, 1.0], (len(times), 1)))
        quats = align_hemispheres(np.array(
            [camera_quat(p, l, u) for p, l, u in
             zip(track["position"], track["look_at"], up)]))
        # How far ahead the aim point sits is a distance, not a rotation, so it
        # splines with the position rather than riding along with the slerp.
        distance = np.linalg.norm(track["look_at"] - track["position"], axis=1)
        aim = camera_forward(squad(quats, times, t))
        camera["look_at"] = [float(v) for v in
                             position + aim * float(hermite(times, distance, t))]
    if "up" in track:
        up = hermite(times, track["up"], t)
        norm = float(np.linalg.norm(up))
        camera["up"] = [float(v) for v in (up / norm if norm > 1e-9 else up)]
    if "fov" in track:
        camera["fov"] = float(hermite(times, track["fov"], t))
    return camera


def read_camera(server) -> dict | None:
    """The connected browser's camera, in the shot file's shape."""
    clients = server.get_clients()
    if not clients:
        return None
    cam = next(iter(clients.values())).camera
    return {
        "position": [float(v) for v in cam.position],
        "look_at": [float(v) for v in cam.look_at],
        "up": [float(v) for v in cam.up_direction],
        "fov": float(cam.fov),
    }


def apply_camera(server, camera: dict | None) -> None:
    """Push a camera onto every connected browser.

    ``position`` is set first on purpose: viser's setter drags ``look_at``
    along by the same offset, so writing the aim afterwards is what makes the
    result the pose that was asked for rather than a translated version of the
    old one.
    """
    if not camera:
        return
    for client in server.get_clients().values():
        cam = client.camera
        if "position" in camera:
            cam.position = np.array(camera["position"], float)
        if "look_at" in camera:
            cam.look_at = np.array(camera["look_at"], float)
        if "up" in camera:
            cam.up_direction = np.array(camera["up"], float)
        if "fov" in camera:
            cam.fov = float(camera["fov"])


def pose_to_joints(cfg: np.ndarray, urdf, base: dict[str, float]) -> dict[str, float]:
    """The joints a keyframe needs to name: those that differ from the base pose.

    Storing all 39 every time would work and be unreadable. This keeps a
    captured keyframe as small as what you actually moved -- and identical in
    meaning, since unnamed joints resolve to the base pose anyway.
    """
    out = {}
    for i, name in enumerate(urdf.actuated_joint_names):
        if abs(float(cfg[i]) - float(base.get(name, 0.0))) > 1e-6:
            out[name] = float(cfg[i])
    return out


# --------------------------------------------------------------------------- #
# Scene
# --------------------------------------------------------------------------- #

def build_scene(server: viser.ViserServer, shot_path: Path, urdf_path: Path,
                out_dir: Path) -> None:
    if not urdf_path.exists():
        raise SystemExit(f"No URDF at {urdf_path}")

    server.scene.set_up_direction("+z")
    server.scene.add_box(
        "/pad", dimensions=(30.0, 30.0, 0.6), position=(0.0, 0.0, -0.3),
        color=(58, 60, 66),
    )

    urdf = load_urdf(urdf_path)
    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot")
    cfg = home_cfg(urdf)
    viser_urdf.update_cfg(cfg)

    # A missing shot file is the normal way to start: an empty timeline you fill
    # by posing and capturing.
    shot = Shot.load(shot_path) if shot_path.exists() else Shot()
    if not shot_path.exists():
        print(f"[shot] {shot_path} does not exist yet — starting with an empty "
              "timeline. 'Save shot YAML' will create it.")

    tabs = server.gui.add_tab_group()
    pose_tab = tabs.add_tab("Pose")
    keys_tab = tabs.add_tab("Keyframes")
    preview_tab = tabs.add_tab("Preview")
    lights_tab = tabs.add_tab("Lights")

    # ------------------------------------------------------------------ pose
    with pose_tab:
        server.gui.add_markdown(
            "Click a body part to select the joint that drives it, then dial "
            "the slider. When the pose looks right, capture it in the "
            "**Keyframes** tab.")
    refresh_pose = add_joint_picker(server, urdf, viser_urdf, cfg,
                                    container=pose_tab)

    # ------------------------------------------------------------- keyframes
    with keys_tab:
        # A slider, because fps is a thing you dial while watching the frame
        # count next to it rather than a number you know in advance. What you
        # land on is written straight back to the shot YAML. A hand-edited file
        # may still ask for something off the grid -- 23.976, say -- so the
        # slider clamps and rounds for *display*; the shot keeps the real value
        # until you actually drag it.
        fps_input = server.gui.add_slider(
            "FPS", min=FPS_MIN, max=FPS_MAX, step=1.0,
            initial_value=_fps_on_slider(shot.fps))
        with_camera = server.gui.add_checkbox("Save camera with keyframe",
                                              initial_value=True)
        listing = server.gui.add_markdown("")

        add_folder = server.gui.add_folder("Add")
        with add_folder:
            at_time = server.gui.add_number("Time (s)", initial_value=0.0,
                                            min=0.0, step=0.05)
            capture = server.gui.add_button("● Capture new keyframe",
                                            icon=_icon("PLUS"))

        edit_folder = server.gui.add_folder("Edit keyframe")
        with edit_folder:
            picker = server.gui.add_dropdown("Keyframe", options=("—",),
                                             initial_value="—")
            detail = server.gui.add_markdown("")
            edit_key = server.gui.add_button("✎ Edit this keyframe",
                                             icon=_icon("EDIT"))
            update_key = server.gui.add_button("✔ Update from current view",
                                               icon=_icon("CHECK"))
            revert_key = server.gui.add_button("↺ Revert")
            new_time = server.gui.add_number("Retime to (s)", initial_value=0.0,
                                             min=0.0, step=0.05)
            retime = server.gui.add_button("Move keyframe to that time")
            insert_key = server.gui.add_button("+ Insert breakdown after")
            duplicate_key = server.gui.add_button("+ Duplicate")
            delete = server.gui.add_button("✕ Delete")

        joint_folder = server.gui.add_folder("Joints in this keyframe")
        with joint_folder:
            joint_picker = server.gui.add_dropdown("Joint", options=("—",),
                                                   initial_value="—")
            drop_joint = server.gui.add_button("Reset joint to base pose")
            clear_camera = server.gui.add_button("Clear this keyframe's camera")

        file_folder = server.gui.add_folder("File")
        with file_folder:
            save_path = server.gui.add_text("Shot YAML", initial_value=str(shot_path))
            save = server.gui.add_button("Save shot YAML",
                                         icon=_icon("DEVICE_FLOPPY"))
            reload_button = server.gui.add_button("Reload shot")
        keys_status = server.gui.add_text("Status", initial_value="(unsaved)",
                                          disabled=True)

    # ---------------------------------------------------------------- preview
    with preview_tab:
        info = server.gui.add_markdown("")
        timeline = server.gui.add_slider("Frame", min=0, max=0, step=1,
                                         initial_value=0)
        at_seconds = server.gui.add_text("Time", initial_value="0.000s",
                                         disabled=True)
        play = server.gui.add_button("▶ Play")
        loop = server.gui.add_checkbox("Loop", initial_value=True)
        record = server.gui.add_button("⤓ Export shot", icon=_icon("CAMERA"))
        rec_status = server.gui.add_text("Last export", initial_value="(none)",
                                         disabled=True)
        rec_detail = server.gui.add_markdown("")

    # ``editing`` is the index of the keyframe currently open for editing, or
    # None when the timeline is in charge of the pose. The two are mutually
    # exclusive on purpose: re-sampling the shot rewrites the pose from the
    # timeline, which would quietly undo an edit in progress.
    state = {"shot": shot, "frames": sample(shot, urdf), "playing": False,
             "editing": None, "suppress": False}

    # -- keeping the four panels in step ------------------------------------ #
    def time_of(frame: int) -> float:
        """Seconds at a timeline frame. A take need not start at t=0."""
        s = state["shot"]
        return (s.times[0] if s.keyframes else 0.0) + frame / s.fps

    def frame_of(t: float) -> int:
        s = state["shot"]
        origin = s.times[0] if s.keyframes else 0.0
        return int(round((float(t) - origin) * s.fps))

    def label_of(i: int) -> str:
        k = state["shot"].keyframes[i]
        mark = " 📷" if k.get("camera") else ""
        return f"{i}: t={float(k['t']):.2f}s ({len(k['joints'])} joints){mark}"

    def selected_index() -> int | None:
        if not state["shot"].keyframes or picker.value == "—":
            return None
        try:
            return int(picker.value.split(":")[0])
        except ValueError:
            return None

    def rebuild(select: int | None = None) -> None:
        """Re-sample the shot and refresh every widget that describes it.

        Leaves the pose alone while a keyframe is open for editing -- otherwise
        re-sampling would overwrite the very pose being edited.
        """
        s = state["shot"]
        state["frames"] = sample(s, urdf)
        frames = state["frames"]

        labels = [label_of(i) for i in range(len(s.keyframes))] or ["—"]
        state["suppress"] = True
        try:
            picker.options = tuple(labels)
            if select is not None and select < len(labels):
                picker.value = labels[select]
            elif picker.value not in labels:
                picker.value = labels[0]
            timeline.max = max(0, len(frames) - 1)
            timeline.value = min(timeline.value, timeline.max)
        finally:
            state["suppress"] = False

        if s.keyframes:
            cams = sum(1 for k in s.keyframes if k.get("camera"))
            listing.content = "\n".join(
                [f"**{len(s.keyframes)} keyframes**, {cams} with a camera  "]
                + [f"`t={float(k['t']):6.2f}s`  {len(k['joints'])} joints"
                   + ("  📷" if k.get("camera") else "")
                   for k in s.keyframes])
        else:
            listing.content = ("*No keyframes yet.* Pose the robot, set a time, "
                               "then **Capture new keyframe**.")
        info.content = (f"**{len(frames)} frames** @ {s.fps:g} fps "
                        f"= {len(frames) / s.fps:.2f}s  \n"
                        f"{len(s.keyframes)} keyframes")
        refresh_detail()
        if state["editing"] is None:
            show(timeline.value)

    def refresh_detail() -> None:
        """The selected keyframe's own contents, and the joint dropdown."""
        i = selected_index()
        if i is None:
            detail.content = "*Nothing selected.*"
            state["suppress"] = True
            try:
                joint_picker.options = ("—",)
            finally:
                state["suppress"] = False
            return
        k = state["shot"].keyframes[i]
        editing = state["editing"] == i
        lines = [f"**#{i} — t = {float(k['t']):.2f}s**"
                 + ("  \n*editing: Update writes back here*" if editing else ""),
                 f"{len(k['joints'])} joints"
                 + (", camera saved" if k.get("camera") else ", no camera")]
        detail.content = "  \n".join(lines)
        names = tuple(k["joints"]) or ("—",)
        state["suppress"] = True
        try:
            joint_picker.options = names
        finally:
            state["suppress"] = False
        new_time.value = round(float(k["t"]), 3)

    def show(i: int) -> None:
        """Put frame ``i`` into the viewport, the editing pose and the camera.

        The timeline and the joint sliders drive one configuration vector, so
        scrubbing to a moment and then adjusting a joint is how you fix a pose
        in place -- Update writes it back to the selected keyframe.
        """
        frames = state["frames"]
        if len(frames) == 0:
            return
        i = int(np.clip(i, 0, len(frames) - 1))
        cfg[:] = frames[i]
        refresh_pose()
        apply_camera(server, camera_at(state["shot"], time_of(i)))
        at_seconds.value = f"{time_of(i):.3f}s"

    def exit_edit() -> None:
        if state["editing"] is not None:
            state["editing"] = None
            keys_status.value = "left edit mode"
            refresh_detail()

    @timeline.on_update
    def _(_) -> None:
        if state["suppress"]:
            return
        # Touching the timeline means you are done editing that one keyframe.
        exit_edit()
        show(timeline.value)
        if not state["playing"]:
            # Scrubbing is also how you pick the time to capture at.
            at_time.value = round(time_of(timeline.value), 3)

    @picker.on_update
    def _(_) -> None:
        if not state["suppress"]:
            exit_edit()
            refresh_detail()

    # -- adding -------------------------------------------------------------- #
    @capture.on_click
    def _(_) -> None:
        state["playing"] = False
        s = state["shot"]
        camera = read_camera(server) if with_camera.value else None
        i = s.put(at_time.value, pose_to_joints(cfg, urdf, s.base), camera)
        keys_status.value = (f"captured t={float(at_time.value):.2f}s"
                             + (" with camera" if camera else "") + " (unsaved)")
        # Park the timeline on the pose just captured: re-sampling rewrites the
        # pose from the timeline, so landing anywhere else would throw away the
        # pose you are still working on.
        state["editing"] = None
        rebuild(select=i)
        timeline.value = max(0, min(frame_of(at_time.value), timeline.max))
        show(timeline.value)

    # -- editing one keyframe ------------------------------------------------ #
    @edit_key.on_click
    def _(_) -> None:
        i = selected_index()
        if i is None:
            return
        state["playing"] = False
        state["editing"] = i
        s = state["shot"]
        # The *stored* pose, not the nearest sampled frame: a keyframe between
        # two frames would otherwise come back subtly interpolated, and the
        # first Update would write that drift into the shot.
        cfg[:] = keyframe_pose(s, urdf, i)
        refresh_pose()
        apply_camera(server, s.keyframes[i].get("camera"))
        state["suppress"] = True
        try:
            timeline.value = max(0, min(frame_of(float(s.keyframes[i]["t"])),
                                        timeline.max))
        finally:
            state["suppress"] = False
        at_seconds.value = f"{float(s.keyframes[i]['t']):.3f}s"
        keys_status.value = (f"editing #{i} — pose it, frame it, then Update")
        refresh_detail()

    @update_key.on_click
    def _(_) -> None:
        i = state["editing"] if state["editing"] is not None else selected_index()
        if i is None:
            return
        s = state["shot"]
        camera = read_camera(server) if with_camera.value else None
        s.update(i, pose_to_joints(cfg, urdf, s.base), camera)
        keys_status.value = (f"updated #{i}"
                             + (" (pose + camera)" if camera else " (pose)")
                             + " (unsaved)")
        rebuild(select=i)

    @revert_key.on_click
    def _(_) -> None:
        i = state["editing"] if state["editing"] is not None else selected_index()
        if i is None:
            return
        s = state["shot"]
        cfg[:] = keyframe_pose(s, urdf, i)
        refresh_pose()
        apply_camera(server, s.keyframes[i].get("camera"))
        keys_status.value = f"reverted #{i} to what is stored"

    @retime.on_click
    def _(_) -> None:
        i = selected_index()
        if i is None:
            return
        state["editing"] = None
        j = state["shot"].retime(i, new_time.value)
        keys_status.value = f"moved #{i} to t={float(new_time.value):.2f}s (unsaved)"
        rebuild(select=j)

    @insert_key.on_click
    def _(_) -> None:
        """A breakdown keyframe that changes nothing until you edit it.

        Seeded with the pose (and camera) the shot already passes through at
        that instant, so inserting one is free: the motion is identical until
        you move something.
        """
        i = selected_index()
        if i is None:
            return
        s = state["shot"]
        t = float(s.keyframes[i]["t"]) + s.gap_after(i)
        joints = pose_to_joints(pose_at(s, urdf, t), urdf, s.base)
        j = s.put(t, joints, camera_at(s, t))
        state["editing"] = None
        keys_status.value = f"inserted #{j} at t={t:.2f}s (unsaved)"
        rebuild(select=j)

    @duplicate_key.on_click
    def _(_) -> None:
        i = selected_index()
        if i is None:
            return
        s = state["shot"]
        k = s.keyframes[i]
        t = float(k["t"]) + s.gap_after(i)
        j = s.put(t, k["joints"], k.get("camera"))
        state["editing"] = None
        keys_status.value = f"duplicated #{i} to t={t:.2f}s (unsaved)"
        rebuild(select=j)

    @delete.on_click
    def _(_) -> None:
        i = selected_index()
        if i is None:
            return
        state["editing"] = None
        state["shot"].delete(i)
        keys_status.value = f"deleted #{i} (unsaved)"
        rebuild(select=max(0, i - 1))

    # -- editing one joint of one keyframe ----------------------------------- #
    @drop_joint.on_click
    def _(_) -> None:
        i = selected_index()
        if i is None or joint_picker.value == "—":
            return
        joint = joint_picker.value
        state["shot"].drop_joint(i, joint)
        keys_status.value = f"#{i}: {joint} back to the base pose (unsaved)"
        if state["editing"] == i:
            cfg[:] = keyframe_pose(state["shot"], urdf, i)
            refresh_pose()
        rebuild(select=i)

    @clear_camera.on_click
    def _(_) -> None:
        i = selected_index()
        if i is None:
            return
        state["shot"].clear_camera(i)
        keys_status.value = f"#{i}: camera cleared (unsaved)"
        rebuild(select=i)

    @fps_input.on_update
    def _(_) -> None:
        if state["suppress"]:
            return
        state["playing"] = False
        # The frame the timeline sits on is at a different index once the rate
        # changes, so read the moment in seconds first and restore *that* --
        # re-sampling the shot under you should not also jump you elsewhere in
        # it.
        at = time_of(timeline.value)
        state["shot"].fps = float(fps_input.value)
        rebuild()
        timeline.value = max(0, min(frame_of(at), timeline.max))
        show(timeline.value)
        # The rate you settle on is the shot's rate, so it goes straight back
        # to the YAML -- no separate Save for this one field.
        try:
            written = state["shot"].write_fps(Path(save_path.value))
        except OSError as exc:
            keys_status.value = f"fps NOT saved: {exc}"
            return
        keys_status.value = f"fps {state['shot'].fps:g} → {written}"

    @save.on_click
    def _(_) -> None:
        try:
            written = state["shot"].dump(Path(save_path.value))
        except OSError as exc:
            keys_status.value = f"FAILED: {exc}"
            return
        keys_status.value = f"saved {written}"

    @reload_button.on_click
    def _(_) -> None:
        state["playing"] = False
        try:
            state["shot"] = Shot.load(Path(save_path.value))
        except (SystemExit, OSError, yaml.YAMLError) as exc:
            # A bad edit (or a path typo) must not take the server down.
            keys_status.value = f"FAILED: {exc}"
            return
        state["suppress"] = True
        try:
            fps_input.value = _fps_on_slider(state["shot"].fps)
        finally:
            state["suppress"] = False
        keys_status.value = f"loaded {save_path.value}"
        rebuild(select=0)

    # -- playback: the review step, at the shot's real rate ------------------ #
    @play.on_click
    def _(_) -> None:
        if state["playing"]:
            state["playing"] = False
            return
        if len(state["frames"]) < 2:
            info.content += "  \n*Nothing to play yet — capture two keyframes.*"
            return
        state["playing"] = True
        play.label = "❚❚ Pause"

        def run() -> None:
            period = 1.0 / state["shot"].fps
            while state["playing"]:
                i = timeline.value + 1
                if i > timeline.max:
                    if not loop.value:
                        break
                    i = 0
                timeline.value = i          # fires on_update -> show()
                time.sleep(period)
            state["playing"] = False
            play.label = "▶ Play"

        threading.Thread(target=run, daemon=True).start()

    # -- export -------------------------------------------------------------- #
    @record.on_click
    def _(_) -> None:
        state["playing"] = False
        if len(state["shot"].keyframes) < 2:
            rec_status.value = "need at least two keyframes"
            return
        state["editing"] = None
        s = state["shot"]
        recorder = visender.Recorder(server, fps=s.fps)
        keyed_camera = camera_track(s)[0] is not None
        record.disabled = True
        try:
            # Frame-stepped, not wall-clock: every frame is sampled after the
            # pose it belongs to is in place, however long that took.
            #
            # This is the previewed take and nothing else: the same frames the
            # timeline scrubs, written straight out. No start/stop pass over
            # the live scene, so what you exported is what you just watched.
            #
            # A keyframed camera is passed to the recorder explicitly rather
            # than pushed to the browser and read back: the authored value is
            # what should land in the bundle, with no round trip to depend on
            # (and it exports with no browser attached at all).
            for i in range(len(state["frames"])):
                show(i)
                camera = camera_at(s, time_of(i)) if keyed_camera else None
                recorder.capture(camera=camera)
            target = out_dir.with_name(f"{out_dir.name}_{time.strftime('%H%M%S')}")
            recorder.save(
                target,
                node_filter=lambda name: not name.endswith(("/bulb", "/_selection")),
            )
            # Say which camera went into the take. The fallback to the live
            # browser camera keeps an uncamera'd shot exportable, but it is
            # silent, and a take exported that way looks
            # nothing like the keyframed one no matter what the shot file says.
            keys = sum(1 for k in s.keyframes if k.get("camera"))
            source = (f"**keyframed** ({keys} camera keys, "
                      f"{s.camera_interpolation})" if keyed_camera else
                      "**live from the browser** — no keyframe has a camera, so "
                      "this take got whatever the view happened to be doing")
            rec_status.value = str(target)
            rec_detail.content = (f"{len(state['frames'])} frames @ {s.fps:g} fps  \n"
                                  f"camera: {source}  \n"
                                  f"Point `bundle:` at this take to render it.")
            print(f"[shot] exported {target} — {len(state['frames'])} frames, "
                  f"camera {'keyframed' if keyed_camera else 'live from browser'}")
            timeline.value = 0
        except Exception as exc:
            rec_status.value = f"FAILED: {exc}"
            raise
        finally:
            record.disabled = False

    gizmos = add_light_panel(server, lights_tab)
    rebuild(select=0 if shot.keyframes else None)
    for ctrl in gizmos:
        ctrl.visible = True


def _icon(name: str):
    try:
        from viser import Icon

        return getattr(Icon, name, None)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=Path, default=DEFAULT_SHOT,
                        help="Keyframe YAML to open (created on save if absent).")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Bundle directory prefix; each take is timestamped.")
    args = parser.parse_args()

    server = viser.ViserServer()
    build_scene(server, args.shot, args.urdf, args.out)

    print("Pose the robot, capture keyframes, preview the take, then "
          "'Export shot' to write an animated bundle.")
    server.sleep_forever()


if __name__ == "__main__":
    main()
