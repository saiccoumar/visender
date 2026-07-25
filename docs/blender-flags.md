# Raw `blender_import.py` flags

`v2b render` is a convenience layer over exactly these flags; everything here
still works without it. Nothing is installed on the Blender side —
`blender_import.py` imports only `bpy` and numpy, both of which ship with
Blender, and it is loaded by file path rather than as a package.

In every form below the bare `--` is required: Blender consumes the arguments
before it, and everything after it is passed to the script.

## Open the scene in the Blender app

```bash
blender --python viser2blender/blender_import.py -- --bundle renders/pen_grip_142530
```

Blender opens with the scene built and the camera framed on whatever view your
browser was showing at export time. <kbd>F12</kbd> renders, <kbd>Numpad 0</kbd>
looks through the imported camera.

If `blender` is not on your `$PATH`, call it by full path
(`~/blender-5.2.0-linux-x64/blender --python ... -- --bundle ...`).

You can also skip the shell: in Blender's **Scripting** tab → **New**, paste
this and hit <kbd>Alt</kbd>+<kbd>P</kbd>. Use absolute paths — Blender's working
directory is not your shell's.

```python
import sys
sys.path.insert(0, "/abs/path/to/viser2blender/viser2blender")
sys.argv = ["blender", "--", "--bundle", "/abs/path/to/renders/pen_grip_142530"]

import blender_import
blender_import.main()
```

## Render from the CLI

`-b` (background) renders without opening a window:

```bash
blender -b --python viser2blender/blender_import.py \
        -- --bundle renders/pen_grip_142530 \
           --render cover.png --samples 512 --gpu \
           --hdri ~/hdris/studio_small_08_4k.exr \
           --world-strength 0.8 --sun-scale 2.0
```

Two things that catch people out:

- Background renders are **CPU-only unless you pass `--gpu`**. Selecting a GPU
  in the GUI preferences does not carry over.
- Override resolution with the script's `--resolution W H`, **not** Blender's
  own `-x`/`-y` — the camera is built after Blender parses those, so it
  overwrites them. Vertical FOV is preserved, so a different aspect ratio
  widens or crops horizontally rather than rescaling.

## Scene

| flag | default | what it does |
| --- | --- | --- |
| `--bundle` | *required* | Bundle directory from the export step. |
| `--render PATH` | — | Render straight to an image; format follows the extension. |
| `--engine` | `CYCLES` | `CYCLES` or `EEVEE` (resolved against your Blender's engine list). |
| `--samples` | `128` | Render samples. |
| `--resolution W H` | browser size | Override the output resolution. |
| `--gpu` | off | Render Cycles on the GPU (OPTIX/CUDA/HIP/METAL/ONEAPI, first found). |
| `--config PATH` | — | Resolved-JSON config; fills any flag not passed explicitly (written by `v2b`, but hand-writable). |
| `--list-nodes` | off | Print each node's path/kind/size and exit without building. |

## World & materials

| flag | default | what it does |
| --- | --- | --- |
| `--hdri FILE` | — | `.exr`/`.hdr` for the world background. |
| `--studio-world` | off | Soft procedural vertical gradient world (no file needed). Ignored when `--hdri` is set. |
| `--world-strength` | `1.0` | World lighting strength. |
| `--backdrop` | off | Large neutral ground plane just under the scene, so shadows land on a floor. |
| `--backdrop-color` | `200,200,205` | R,G,B (0–255 sRGB) for the backdrop. |
| `--shadow-catcher` | off | Make the backdrop a shadow catcher (Cycles only). |
| `--material NODE=SPEC` | — | Override a node's material by its viser `/path`. Positional `R,G,B[,A][,ROUGH]`, or key=value PBR `'base_color:158,163,168;metallic:1.0'`, or `'use:aluminium'`. Repeatable; trailing `/*` matches a subtree. |
| `--exposure` | `0.0` | Stops of exposure in colour management (+ brighter). |
| `--look` | — | Colour-management look, e.g. `'AgX - Punchy'`. Validated against the live enum. |

## Lighting (camera-relative)

| flag | default | what it does |
| --- | --- | --- |
| `--sun-scale` | `1.0` | viser directional intensity → Blender sun W/m². |
| `--point-scale` | `4π` | viser point/spot intensity (candela) → Blender watts. |
| `--dim-authored F` | — | Scale every imported light's energy (e.g. `0.15` before adding a key). |
| `--key-light AZ,EL` | — | Soft key in **camera space**: az 0 = behind camera, +az frame-right, +el up. |
| `--key-energy` / `--key-angle` / `--key-color` | — | Key watts / soft-shadow angle (deg) / colour. |
| `--three-point` | off | Camera-relative key + fill + rim. |
| `--auto-light` | off | Three-point rig **only if** the bundle has no lights (else it renders near-black under Cycles). |

## Camera & framing

| flag | default | what it does |
| --- | --- | --- |
| `--fit` | `keep_vertical` | Aspect policy: `keep_vertical` / `keep_horizontal` / `fit_all` (letterbox, lose nothing). Warns whenever the render aspect crops the composed frame. |
| `--auto-camera` | off | Frame the scene bounds in a 3/4 view when the bundle has no camera. Resolved before lighting, so `--auto-light`/`--three-point` light the view it picks. |
| `--orbit AZ EL` / `--dolly F` | — / `1.0` | Rotate / scale-distance the camera about its look-at point. Camera-space lights orbit with it, so a key stays put relative to the frame. |
| `--dof [DIST] --fstop F` | — / `2.8` | Depth of field; focus defaults to `\|look_at − position\|`. |
| `--scale` | `100` | `resolution_percentage` — preview at 50% for the same composition. |

## Output

| flag | default | what it does |
| --- | --- | --- |
| `--save-blend [PATH]` | off | Save a `.blend` after building (before the render, so it survives a crash). Path derived from the output if omitted. |
| `--transparent` | off | Transparent film, RGBA output. |
| `--point-size` / `--point-color` | auto / — | Point-cloud instance radius (m) / forced flat colour. |
| `--line-scale` | `1.0` | Scale spline/line tube radius. |

Under EEVEE the render path also turns on ray-traced GI/AO/reflections
(`use_raytracing`), which the legacy flat look leaves off — the single biggest
quality win short of Cycles.
