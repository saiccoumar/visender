# viser2blender

Snapshot a live viser scene — every mesh, pose, light and the camera you framed
it with — and rebuild it in Blender so cover images can be rendered with Cycles
instead of three.js.

The two halves never share an interpreter: **export** needs `viser` and runs in
your solver env; **import** needs `bpy` and runs inside Blender. They talk only
through a bundle directory on disk.

## Install

```bash
pip install -e external/viser2blender
```

## Export: add a button to any viser script

```python
import viser2blender

server = viser.ViserServer()
...  # build the scene as usual

viser2blender.add_export_button(server, out_dir="renders/pen_grip",
                                environment_map="city")
```

Pose the scene in the browser, then click **Export to Blender**. Each click
writes a fresh timestamped bundle (`renders/pen_grip_142530/`), so you can
export several takes and pick the best one later.

There is a plain function too, if you want to export without a GUI:

```python
viser2blender.export_scene(server, "renders/take1")
```

Useful options (both entry points take them):

| option | default | what it does |
| --- | --- | --- |
| `include_gizmos` | `False` | Emit transform controls, labels, grids, frames. Off by default — they are alignment aids, not subjects. |
| `include_hidden` | `False` | Emit nodes hidden in the browser. |
| `node_filter` | `None` | `lambda name: ...` — return `False` to drop a node by path. |
| `environment_map` | `None` | The preset you passed to `configure_environment_map`. viser does not retain it, so repeat it here. |

**Gizmos are skipped, but their transforms are never ignored.** A light parented
to a transform control (`/ctrl/key/light`) keeps the pose you dragged it to —
world transforms are composed down the full `/a/b/c` path regardless of which
nodes get emitted.

## Import: rebuild in Blender

Nothing to install on the Blender side. `blender_import.py` imports only `bpy`
and numpy, both of which ship with Blender, and it is loaded by file path rather
than as a package — `pip install -e` above only sets up the *export* half.

In every form below, the bare `--` is required: Blender consumes the arguments
before it, and everything after it is passed through to the script.

### Open it in the Blender app

To art-direct the scene by hand — swap materials, add a backdrop, tweak lights:

```bash
blender --python external/viser2blender/viser2blender/blender_import.py \
        -- --bundle renders/pen_grip_142530
```

Blender opens with the scene already built and the camera framed on whatever
view your browser was showing at export time. Press <kbd>F12</kbd> to render,
<kbd>Numpad 0</kbd> to look through the imported camera, and save as a normal
`.blend` when you like it.

If `blender` is not on your `$PATH` (a downloaded tarball rather than a distro
package), call it by full path:

```bash
~/blender-4.5.0-linux-x64/blender --python ... -- --bundle ...
```

You can also skip the shell entirely: in Blender, open the **Scripting** tab →
**New**, paste this, and hit **Run Script** (<kbd>Alt</kbd>+<kbd>P</kbd>):

```python
import sys
sys.path.insert(0, "/abs/path/to/external/viser2blender/viser2blender")
sys.argv = ["blender", "--", "--bundle", "/abs/path/to/renders/pen_grip_142530"]

import blender_import
blender_import.main()
```

Use absolute paths here — Blender's working directory is not your shell's.

### Render from the CLI

`-b` (background) renders without opening a window, which is what you want on a
headless box or in a batch:

```bash
blender -b --python external/viser2blender/viser2blender/blender_import.py \
        -- --bundle renders/pen_grip_142530 \
           --render cover.png --samples 256
```

`--render` takes the output path; the image format follows its extension. Cycles
is the default engine — for a fast preview, EEVEE takes seconds instead of
minutes:

```bash
blender -b --python external/viser2blender/viser2blender/blender_import.py \
        -- --bundle renders/pen_grip_142530 \
           --render preview.png --engine EEVEE --samples 32
```

A realistic first cover-image pass, once you have an HDRI to stand in for viser's
environment map, and with the lights turned up to taste:

```bash
blender -b --python external/viser2blender/viser2blender/blender_import.py \
        -- --bundle renders/pen_grip_142530 \
           --render cover.png --samples 512 \
           --hdri ~/hdris/studio_small_08_4k.exr \
           --world-strength 0.8 --sun-scale 2.0
```

Resolution comes from the browser window the scene was exported from. Override
it with `--resolution 3840 2160`. Use the script's flag, **not** Blender's own
`-x`/`-y` — the camera is built after Blender has parsed those, so it overwrites
them. Vertical FOV is preserved, so a different aspect ratio widens or crops the
frame horizontally rather than rescaling it.

Background renders are CPU-only unless you pass `--gpu`, which is not implied by
having selected a GPU in the GUI preferences:

```bash
blender -b --python external/viser2blender/viser2blender/blender_import.py \
        -- --bundle renders/pen_grip_142530 \
           --render cover.png --samples 512 --gpu
```

| flag | default | what it does |
| --- | --- | --- |
| `--bundle` | *required* | Bundle directory from the export step. |
| `--render PATH` | — | Render straight to an image. |
| `--engine` | `CYCLES` | `CYCLES` or `EEVEE` (resolved against your Blender's engine list). |
| `--samples` | `128` | Render samples. |
| `--resolution W H` | browser size | Override the output resolution. |
| `--gpu` | off | Render Cycles on the GPU (OPTIX/CUDA/HIP/METAL/ONEAPI, first found). |
| `--hdri FILE` | — | `.exr`/`.hdr` for the world background. |
| `--world-strength` | `1.0` | World lighting strength. |
| `--sun-scale` | `1.0` | viser directional intensity → Blender sun W/m². |
| `--point-scale` | `4π` | viser point/spot intensity (candela) → Blender watts. |

## What transfers

| viser | Blender |
| --- | --- |
| `add_mesh_trimesh`, `add_glb` | imported GLB under a posed empty |
| `add_mesh_simple`, `add_box`, `add_icosphere` | mesh + Principled BSDF |
| `add_spline_catmull_rom`, `add_spline_cubic_bezier`, `add_line_segments` | beveled curve (a real tube, not a screen-space line) |
| `add_light_directional` / `point` / `spot` / `rect_area` | sun / point / spot / area |
| `add_light_ambient` / `hemisphere` | folded into the world shader |
| browser camera | camera with matching position, aim and vertical FOV |
| `add_point_cloud` | vertex-only mesh (add your own geometry-nodes instancing) |

Anything else is named on stdout and skipped rather than silently dropped.

## Things worth knowing

- **The camera comes from a connected browser.** Export with the tab open, and
  Blender opens on the view you framed. With no client attached the bundle has
  no camera and Blender falls back to its default view.
- **Light intensity is not physically portable.** three.js and Cycles disagree
  about units, so `--sun-scale` / `--point-scale` are the knobs to taste. Poses
  transfer exactly; brightness is a starting point.
- **Line widths are pixels in viser, metres in Blender.** Splines are given a
  tube radius of `line_width * 0.00025`, which reads about right at cover-image
  scale. Adjust `bevel_depth` in Blender if not.
- **Environment maps do not transfer.** viser's presets are three.js built-ins.
  Pass `--hdri` with your own file to match.
- **Colours are converted sRGB → linear** (`^2.2`) so Blender matches what the
  browser showed.

## Verified behaviour

Two axis/caching subtleties are handled, both confirmed against Blender 5.2 by
comparing world-space vertex bounds to trimesh ground truth:

- Blender's glTF importer bakes a −90° X rotation into *vertex data* (not object
  matrices) to convert +Y-up → +Z-up. viser hands GLB bytes to three.js
  untouched and trimesh writes them Z-up, so that rotation is undone on import —
  otherwise meshes land on their side.
- `matrix_world` is cached, so the depsgraph is updated after re-parenting.
  Without it the transforms silently do not apply.
