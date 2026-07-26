# Example: Gundam RX-78 on a landing pad

This example walks a user through the visender pipeline: viser scene → bundle → cycles render — on an example scene

![Render](../cover.png)


Files:

| | |
| --- | --- |
| [`gundam.py`](gundam.py) | the viser scene, with click-to-pose joints, light gizmos and the export button |
| [`gundam.yaml`](gundam.yaml) | the render config (`visender render`) |
| `gundam_model/` | the vendored GGC RX-78 URDF and meshes (CC BY-NC-SA) |


## 1. Install

```bash
uv sync --extra cli --extra export     # from the visender repo root
source .venv/bin/activate
```

The scene loads a URDF, which needs `yourdfpy` — it is in the `dev` dependency
group, installed by a plain `uv sync`. With pip: `pip install -e '.[cli]' yourdfpy`.

## 2. Configure scene and export a bundle 

```bash
python examples/gundam.py            # or --urdf /path/to/other.urdf
```

Open the printed URL. Then:

- **Click a body part** to select the nearest actuated joint above it — the
  panel names the joint and the link, axes mark the selected frame, and a slider
  (clamped to the URDF's limits) poses it. **Deselect** clears the selection;
  **Reset pose** returns the whole robot to its home config.
- The **Lights** tab adds and removes lights: pick a type (directional / point /
  ambient) and a colour, hit **Add light**, and each one gets a folder with its
  own colour picker, intensity and **Remove** button. The scene starts with the
  key/fill pair the render config expects.
- **Drag the gizmos** on the lights. A directional light is aimed by *rotating*
  its gizmo; a point light only cares where it sits, so its rotation handles are
  disabled, and an ambient light has no gizmo at all.
- Orbit until you like the framing, then click **Export to Blender**. That
  writes `examples/out/gundam_<HHMMSS>/`; point the config's `bundle:` at it.

The camera in the bundle is the view your *browser* was showing, so this step is
how you compose the shot.

Any URDF whose visual meshes resolve will work — `--urdf` defaults to the
vendored RX-78, and `package://` paths resolve against the URDF's own directory.
Note that the model is ~18 m tall with its feet at z = 0, so the props, lights
and distances in the script are all sized in those metres.

## 3. Render

```bash
visender render examples/gundam.yaml --profile draft   # EEVEE, seconds
visender render examples/gundam.yaml --profile final   # Cycles, GPU
```

Output lands in `examples/out/`, next to a `.png.yaml` provenance sidecar
recording the settings that produced it.

If you want to render without doing step 2 first, set `bundle: bundle_panda_desk`
in the config — that shipped bundle was exported with no browser attached, so it
has no camera and `camera.auto` frames the scene instead. Export from a real
browser session and that block becomes unnecessary.

## Where to go next

- Add `save_blend: true` to the config and open the `.blend` to art-direct by hand.
- `visender init <bundle> > my.yaml` scaffolds a config for a bundle of your own.
- The full config schema is in [`../docs/config.md`](../docs/config.md); the raw
  `blender --python blender_import.py -- --bundle ...` path is in
  [`../docs/blender-flags.md`](../docs/blender-flags.md).
