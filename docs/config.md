# visender config schema

A `*.yaml` config replaces the pile of shell flags a figure would otherwise
need, and gives every figure one entry point: `visender render config.yaml`.

The config is read in your **solver** env (it needs `pyyaml`, installed by the
`cli` extra: `pip install 'visender[cli]'`). `visender` resolves it to a flat
JSON dict and hands that to Blender via `blender_import.py --config`; Blender's
Python never sees the YAML. See [`blender-flags.md`](blender-flags.md) for the
no-wrapper path.

## Precedence

For any single setting:

```
config file  <  selected profile  <  explicit CLI flag
```

`visender render config.yaml --profile final --samples 4096` takes `samples` from the
CLI, everything else from the `final` profile, and the rest from the file. A flag
you do not pass is left to the profile/file — the Blender side distinguishes
"you passed `--samples`" from "128 is the default", so a config value is never
clobbered by a default.

Unknown keys are errors, not no-ops: a typo at the top level, inside any nested
section, or in a `profiles:` entry fails with a "did you mean" suggestion.
`profiles:` entries are keyed by flat `Settings`/flag name (`backdrop_color`,
not `backdrop.color`) and get the same coercion as the nested sections, so a
colour may be written as a list and a path resolves relative to the config file.

## Paths

Every path (`bundle`, `blender`, `output`, `world.hdri`) resolves **relative to
the config file's directory**, not your shell's CWD. Absolute paths are left
alone.

## Full schema

```yaml
bundle: renders/allegro_pen_grip_151402   # required: the exported bundle dir
blender: /opt/blender/blender-5.2.0-linux-x64/blender   # optional; see lookup order
output: renders/out/pen_grip_{profile}.png              # {profile} {bundle_name} {date} {time}

aliases:                        # expand {name} inside materials.rules node paths
  allegro: /robot/visual/.../base_link

profiles:                       # named overlays selected with --profile
  draft: {engine: EEVEE,  samples: 256,   resolution: [2560, 1440]}
  final: {engine: CYCLES, samples: 16384, resolution: [7680, 4320],
          adaptive_threshold: 0.001, max_bounces: 24, gpu: true}

world:                          # -> --studio-world / --world-strength / --hdri
  studio: true
  strength: 0.9
  hdri: null

camera:                         # -> --fit / --auto-camera / --orbit / --dolly / --dof ...
  fit: keep_vertical            # keep_vertical | keep_horizontal | fit_all
  auto: false                   # frame scene bounds when the bundle has no camera
  orbit: [0, 0]                 # [azimuth, elevation] degrees about look_at
  dolly: 1.0                    # scale camera distance
  dof: null                     # true | focus-distance metres
  fstop: 2.8
  scale: 100                    # resolution_percentage

film:                           # -> --exposure / --look / --transparent
  exposure: 0.35
  look: "AgX - Punchy"          # validated against Blender's live look enum
  transparent: false            # RGBA film for compositing

backdrop:                       # -> --backdrop / --backdrop-color / --shadow-catcher
  enabled: true
  color: [205, 205, 210]
  shadow_catcher: false         # Cycles only; falls back to opaque under EEVEE

lighting:                       # -> --dim-authored / --key-light / --three-point / --auto-light
  dim_authored: 0.15            # scale every imported light's energy
  key: {az: -40, el: 35, energy: 3.0, angle: 8.0, color: [255, 255, 255]}
  three_point: false            # key + fill + rim, all camera-relative
  auto: false                   # three-point ONLY if the bundle has no lights

materials:
  library:                      # name -> PBR dict; overrides the built-ins
    my_metal: {base_color: [158, 163, 168], metallic: 1.0, roughness: 0.22}
  rules:                        # applied in order; last match wins
    - node: /pen
      use: matte_black          # a whole-node named material
    - node: "{allegro}"         # alias-expanded
      split:                    # per-polygon assignment on a merged mesh
        - where: {normal_z: ">0.9", world_z: [-0.06, 0.09]}
          use: matte_black
        - default: true
          use: my_metal

animation:                      # -> --animation / --frame-start / --frame-end ...
  enabled: true                 # render the recorded frame range, not one still
  start: 1                      # first frame (1-based; default the first recorded)
  end: null                     # last frame (default the last recorded)
  step: 1                       # render every Nth frame -- a cheap long-take preview
                                # (playback rate divides to match, so timing is kept)
  fps: null                     # override the playback rate stored in the bundle

save_blend: true                # save a .blend beside the output before rendering
```

### Animation

`animation.enabled` only means anything for a bundle recorded with
`visender.Recorder` (see [`docs/animation.md`](animation.md)). The **output
suffix picks the container**: `.mp4`/`.mkv`/`.mov`/`.webm` encode a movie,
anything else writes a numbered PNG sequence (`out/gundam_0001.png`, …). A
still bundle rendered with `animation.enabled` is an error, and an animated
bundle rendered *without* it renders frame 1 as a still and says so.

### Built-in materials

`aluminium`, `matte_black`, `plastic_white`, `glass`, `rubber`, `copper`. A
`materials.library` entry of the same name overrides the built-in. A PBR dict
accepts: `base_color`/`color` (0–255 sRGB), `metallic`, `roughness`, `ior`,
`transmission`, `specular`, `emission_color`, `emission_strength`, `opacity`
(all 0–1 unless a colour).

### Split predicates

All evaluated in **world space**, per polygon (polygon centre; normal
transformed by `matrix_world.to_3x3()`):

| predicate | form | meaning |
| --- | --- | --- |
| `normal_x/y/z` | `">0.9"`, `"<-0.5"` | comparison on the world normal component |
| `world_x/y/z` | `[min, max]` | polygon centre inside a band |
| `area` | `[min, max]` | polygon area (isolate small hardware) |
| `material_name` | regex | match against the imported GLB material name |

A rule with `default: true` catches every polygon no earlier rule claimed. A
face-count summary per rule is printed so you can see a predicate matched what
you meant.

## Blender lookup order

`--blender` flag → `blender:` in config → `$BLENDER` → `which blender` → newest
`/opt/blender/blender-*/blender`. If none resolves, `visender` fails and lists what it
tried.

## Provenance sidecar

Every successful render writes `<output>.yaml` next to the image: the bundle
path, `visender` and Blender versions, the wall-clock render time, and
`resolved_config` — the **effective** settings, after CLI flags have won over the
config, so it reproduces the frame from the image alone. When a config file was
used, its own values are also recorded under `config_file_values`, so you can see
what the command line changed.

## `visender` subcommands

```
visender render config.yaml --profile final [--output cover.png] [--quality draft] [blender flags]
visender list-nodes <bundle>      # node path / kind / vertex-or-point count
visender inspect <bundle>         # nodes, camera, and what (if anything) was recorded
visender init <bundle> > cfg.yaml # scaffold a starter config from a bundle
```

`--quality draft|preview|final` is a shorthand for when no profile is written
yet; it is mutually exclusive with `--profile`. Any unrecognised flag after the
config is forwarded straight to Blender, so every raw `blender_import.py` flag
still works through the wrapper.
