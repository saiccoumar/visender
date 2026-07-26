# Recording a viser scene and rendering it as video

A bundle is normally one instant. `visender.Recorder` samples the same world
matrices repeatedly and stores the stack next to the (single) set of assets, so
Blender keys every object instead of just placing it.

Geometry is captured **once**, at save time; only poses and the camera vary per
frame. That is what a robot trajectory is, and it means a 500-frame take costs
about what a still costs on disk.

## 1. Record

Frame-stepped — you advance the scene, then sample. Deterministic, and how long
a frame took to compute is irrelevant (`fps` is purely playback):

```python
rec = visender.Recorder(server, fps=24)
for cfg in trajectory:
    viser_urdf.update_cfg(cfg)
    rec.capture()
rec.save("renders/take1")
```

Live — a background thread samples at `fps` while you drag things (and fly the
camera) in the browser:

```python
visender.add_record_button(server, out_dir="renders/take", fps=24)
```

That adds **Record** / **Export recording** buttons and a frame counter.
`save()` and `add_record_button` take every `export_scene` option
(`node_filter`, `environment_map`, …), so a recording is filtered exactly like
a still.

| option | default | what it does |
| --- | --- | --- |
| `fps` | `24` | Playback rate stored in the bundle; also the sampling rate in live mode. |
| `camera` | `True` | Sample the connected browser camera each frame, so a camera move renders as a camera move. |

## 2. Examine before you render

```bash
visender inspect renders/take1
```

```
recording: 109 frames @ 24 fps = 4.54s, blender frames 1..109
           124 animated nodes, camera animated
```

Rendering is the expensive step, so check the take first — and check it in the
browser, where scrubbing is free. `examples/gundam_video.py` puts a timeline
slider and a **Play** button over the exact frames it will record.

That example is also where the motion is *authored*: click a body part to grab
the joint that drives it, dial it in, and press **Capture new keyframe** to pin
the pose at a time. Any keyframe can then be edited on its own — select it,
**Edit this keyframe** to load its stored pose and camera, adjust, **Update
from current view** to write both back — alongside revert, retime, duplicate,
insert-breakdown, delete, and per-joint **Reset joint to base pose**. **Save
shot YAML** writes the take to disk; `--shot` opens it again.

**The camera is keyframed with the pose.** Capturing or updating a keyframe
stores the view you are looking from — scrubbing and playback fly the browser
along it, and recording carries it into Blender. Only keyframes that hold a
camera join that track, so you can frame two or three moments and leave the
rest to the body; before the first key and after the last it holds.

It does not interpolate the way the joints do. Joint angles are independent
scalars and stay linear; the camera is a path and an aim, so its eye follows a
cubic spline (Catmull-Rom tangents over the real key spacing, so uneven keys
behave) and its aim is interpolated as a *rotation* — spherical cubic, squad —
rather than by lerping the look-at point. Lerping a look-at swings the view
fast through the middle of a turn and flicks at every key; squad is C1, so the
angular velocity is continuous *through* the keys and not merely between them.
The look-at is rebuilt afterwards from the interpolated orientation and its own
splined distance, so it stays a look-at for viser and for the bundle. Two
camera keys reduce exactly to the linear result rather than inventing an ease,
and `camera_interpolation: linear` in the shot file turns the whole thing off.
Smooth splines can overshoot between keys — that is what makes a camera move
read as a camera move, and it is why the joints do not use one.

An authored
camera goes to the recorder explicitly rather than being pushed to the browser
and read back, so what lands in the bundle is the value you authored, with no
round trip to depend on:

```python
rec.capture(camera={"position": [...], "look_at": [...], "up": [...], "fov": 0.7})
```

Anything that dict leaves out (near/far, aspect, image size) is filled from the
live browser camera, or from defaults when no browser is attached.

## 3. Render

```yaml
animation:
  enabled: true      # without this an animated bundle renders frame 1 as a still
  # start / end      # trim without re-recording
  # step: 2          # every Nth frame — a cheap way to judge timing
  # fps: 30          # retime the take
output: out/take_{profile}.mp4
```

```bash
visender render take.yaml --profile draft
```

The **output suffix picks the container**: `.mp4` / `.mkv` / `.mov` / `.webm`
encode a movie; anything else writes a numbered sequence
(`out/take_draft_0001.png`, …), which survives an interrupted render and can be
re-encoded by hand. `--transparent` only survives in `.webm` or a sequence.

## What is in the bundle

`scene.json` gains one optional key; everything else is unchanged, and a
still-only reader ignores it:

```json
"animation": {"fps": 24.0, "frame_count": 109, "asset": "assets/animation.npz",
              "nodes": ["/robot/visual/..."], "has_camera": true}
```

`animation.npz` holds `matrices` — `(frames, tracked_nodes, 4, 4)` world
matrices in float32 — plus `camera_position` / `camera_look_at` / `camera_up` /
`camera_fov` when a browser was attached.

## Things worth knowing

- **Only nodes that move get a track.** A node whose world matrix never changes
  is already placed by its static `matrix` in `scene.json`. A URDF with 300
  welded shells and 124 moving ones stores 124.
- **A node must exist for the whole take.** One added halfway through gets no
  track: there is no honest pose for the first half.
- **Keys are baked and interpolated linearly.** Every frame was sampled, so
  Bezier easing between adjacent keys would be invented overshoot, not
  smoothing. Quaternions are flipped into a common hemisphere first, or a 2°
  turn renders as a 358° spin.
- **`--orbit` / `--dolly` apply per frame**, so an art-directed offset rides
  along with a recorded camera move instead of being lost by it.
- **Camera-relative lights follow the camera** through the whole take, which is
  usually what you want in a moving shot — `lighting.dim_authored` plus a `key`
  keeps the subject lit at every framing, not just the one you exported from.
- **Budget the final.** One Cycles frame costs what a hero still costs. Judge
  timing with an EEVEE `draft` profile and `step: 2` first.
