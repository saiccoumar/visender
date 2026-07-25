# viser2blender

Sai Coumar

![Render](cover.png)


### Overview 
viser2blender is a codegen tool that lets you snapshot a live viser scene and rebuild it in Blender to create beautiful demonstration renderings with Cycles instead of three.js. It saves a viser scene to an intermediate bundle representation which is then routed into blenderpy for automatic environment regeneration.


The two halves never share an interpreter. **Export** needs `viser` and runs in
your solver env; **import** needs `bpy` and runs inside Blender. They talk only
through a bundle directory on disk.

A runnable end-to-end example — a Panda on a desk with joint sliders, draggable
props and lights, a render config and a pre-exported bundle — is in
[`examples/`](examples/README.md).

## Install
As a prerequisite, blender must be installed. Blender 5.2's tar.gz can be downloadeed here: [Blender 5.2 Linux Download](https://www.blender.org/download/release/Blender5.2/blender-5.2.0-linux-x64.tar.xz/)

The path to the blender executable must be set to the $BLENDER variable for v2b to run.

```bash
pip install -e '.[cli]'
```

The `cli` extra pulls in `pyyaml`, which the `v2b` wrapper needs to read config
files. A plain `pip install -e .` is enough for the export half and the raw
`blender --python` path. Nothing is installed on the Blender side.

## 1. Export from viser

Add a button to any viser script:

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

Both entry points take the same options:

| option | default | what it does |
| --- | --- | --- |
| `include_gizmos` | `False` | Emit transform controls, labels, grids, frames. Off by default — they are alignment aids, not subjects. |
| `include_hidden` | `False` | Emit nodes hidden in the browser. |
| `node_filter` | `None` | `lambda name: ...` — return `False` to drop a node by path. |
| `environment_map` | `None` | The preset you passed to `configure_environment_map`. viser does not retain it, so repeat it here. |

**Gizmos are skipped, but their transforms are never ignored.** A light parented
to a transform control (`/ctrl/key/light`) keeps the pose you dragged it to —
world transforms are composed down the full `/a/b/c` path regardless of which
nodes get emitted. This is the intended way to art-direct a shot.

## 2. Render with `v2b`

The `v2b` command quickly automates the blenderpy environment generation. 

`v2b` can be run with the following commands:

```bash
v2b render renders/pen_grip.yaml --profile final
v2b render renders/pen_grip.yaml --profile draft --output preview.png
v2b list-nodes renders/pen_grip_142530            # node paths + vertex/point counts
v2b init renders/pen_grip_142530 > pen_grip.yaml  # scaffold a starter config
```

A minimal config:

```yaml
bundle: renders/pen_grip_142530
output: renders/out/pen_grip_{profile}.png
profiles:
  draft: {engine: EEVEE,  samples: 64,   resolution: [1280, 720]}
  final: {engine: CYCLES, samples: 8192, resolution: [3840, 2160], gpu: true}
world:     {studio: true, strength: 0.9}
backdrop:  {enabled: true, color: [205, 205, 210]}
lighting:  {dim_authored: 0.15, key: {az: -40, el: 35}}
save_blend: true
```

Precedence is **config file < profile < explicit CLI flag** — any raw
`blender_import.py` flag passed after the config is forwarded straight through
and wins over the file. Every successful render also writes a `<output>.yaml`
provenance sidecar: the *effective* settings after overrides, the config file's
own values, bundle path, viser2blender/Blender versions and render time.

Before a config exists, `--quality draft|preview|final` is a shorthand (mutually
exclusive with `--profile`):

```bash
v2b render pen_grip.yaml --quality draft
```

The full schema — profiles, aliases, per-polygon material splits, camera
framing, lighting rig — is in [`docs/config.md`](docs/config.md).

## Rendering without `v2b`

`v2b` is a convenience layer over `blender_import.py`, which you can drive
directly:

```bash
blender -b --python viser2blender/blender_import.py \
        -- --bundle renders/pen_grip_142530 --render cover.png --samples 512
```

Drop the `-b` to open the scene in the Blender GUI instead, already built and
framed. Every flag is documented in
[`docs/blender-flags.md`](docs/blender-flags.md).

## What transfers

| viser | Blender |
| --- | --- |
| `add_mesh_trimesh`, `add_glb` | imported GLB under a posed empty |
| `add_mesh_simple`, `add_box`, `add_icosphere` | mesh + Principled BSDF |
| `add_spline_catmull_rom`, `add_spline_cubic_bezier`, `add_line_segments` | beveled curve (a real tube), per-segment colours preserved via a `Col` attribute |
| `add_light_directional` / `point` / `spot` / `rect_area` | sun / point / spot / area |
| `add_light_ambient` / `hemisphere` | folded into the world shader |
| `add_point_cloud` | geometry-nodes instanced spheres, coloured from the exported per-point `Col` |
| browser camera | camera with matching position, aim and vertical FOV |

Anything else is named on stdout and skipped rather than silently dropped.

## Things worth knowing

- **The camera comes from a connected browser.** Export with the tab open and
  Blender opens on the view you framed. With no client attached the bundle has
  no camera, and Blender falls back to its default view.
- **Light intensity is not physically portable.** three.js and Cycles disagree
  about units, so `--sun-scale` / `--point-scale` are knobs to taste. Poses
  transfer exactly; brightness is a starting point.
- **Line widths are pixels in viser, metres in Blender.** Splines get a tube
  radius of `line_width * 0.00025`, which reads about right at cover-image
  scale. Adjust `bevel_depth` in Blender if not.
- **Environment maps do not transfer.** viser's presets are three.js built-ins;
  pass `--hdri` with your own file to match.
- **Colours are converted sRGB → linear** (`^2.2`), so Blender matches what the
  browser showed.
- **Axes are corrected on import.** Blender's glTF importer bakes a
  `(x, y, z) → (x, -z, y)` rotation (+90° about X) into imported *vertex data*;
  the import undoes it so meshes match viser's Z-up frame. Measured against
  Blender 5.2 by comparing world-space vertex bounds to trimesh ground truth.
  Relatedly, `matrix_world` is cached, so the depsgraph is updated after
  re-parenting — without it the transforms silently do not apply.

## Tests

```bash
pip install pytest                      # plus the [cli] extra for pyyaml
pytest                                  # everything, ~20 s
pytest -m "not slow"                    # skip the tests that launch Blender
BLENDER=/path/to/blender pytest         # pick a specific Blender
```

Three tiers, in `tests/`:

- **Solver side** (`test_config.py`, `test_cli.py`, `test_export.py`) — config
  resolution, the `v2b` wrapper (Blender stubbed out) and the bundle format the
  exporter writes, driven by fakes shaped like viser handles.
- **Contracts** (`test_contracts.py`) — the invariants that keep the two
  interpreters in step: `blender_import` imports nothing pip-only,
  `config.SETTINGS_FIELDS` matches the `Settings` dataclass (checked by parsing
  the source, not importing it), every CLI flag lands on a field, pyyaml stays
  out of the base dependencies.
- **Blender side** (`test_blender_import_logic.py`,
  `test_blender_integration.py`) — pure logic runs under a stub `bpy`; scene
  building, materials, lighting, camera and a real EEVEE render run inside
  Blender via `tests/_scene_probe.py`, which reports the built scene as JSON.
  Skipped automatically when no Blender binary is found.
