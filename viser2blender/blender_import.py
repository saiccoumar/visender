"""Rebuild a viser scene bundle inside Blender. Runs in Blender's Python.

    blender --python blender_import.py -- --bundle my_scene.viserbundle
    blender -b --python blender_import.py -- --bundle my_scene.viserbundle \
            --render cover.png --samples 256

This module deliberately imports nothing from the rest of the package: Blender's
interpreter has bpy and numpy but not viser.

Axis note: viser hands GLB bytes to three.js untouched, and trimesh writes them
in the scene's own Z-up frame. Blender's glTF importer always rotates +Y-up to
+Z-up on the way in, so every imported payload is parented under an empty that
undoes that rotation. Without it, meshes land on their side.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

# Undoes the glTF importer's mandatory +Y-up -> +Z-up conversion, which it bakes
# into vertex data (not object matrices). Measured against Blender 5.2: the
# importer maps a source vertex (x, y, z) to (x, -z, y), i.e. it bakes a +90 deg
# rotation about X, so we rotate by -90 deg to send it back to viser's Z-up frame.
GLTF_UNROTATE = Matrix.Rotation(math.radians(-90.0), 4, "X")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="blender_import")
    p.add_argument("--bundle", required=True, help="Bundle directory from export_scene")
    p.add_argument("--render", default=None, help="Render straight to this image path")
    p.add_argument("--samples", type=int, default=128, help="Cycles samples")
    p.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help="Override the resolution (defaults to the exporting browser window). "
        "Blender's own -x/-y are ignored, since the camera is built afterwards.",
    )
    p.add_argument(
        "--gpu",
        action="store_true",
        help="Render Cycles on the GPU. Background renders are CPU-only otherwise, "
        "even when the GUI preferences select a GPU.",
    )
    p.add_argument(
        "--engine",
        default="CYCLES",
        help="CYCLES (default) or EEVEE; resolved against this Blender's engine list",
    )
    p.add_argument("--hdri", default=None, help=".exr/.hdr for the world background")
    p.add_argument(
        "--world-strength", type=float, default=1.0, help="World lighting strength"
    )
    p.add_argument(
        "--sun-scale",
        type=float,
        default=1.0,
        help="viser directional intensity -> Blender sun W/m^2",
    )
    p.add_argument(
        "--point-scale",
        type=float,
        default=4.0 * math.pi,
        help="viser point/spot intensity (candela) -> Blender watts",
    )
    p.add_argument("--keep-default-cube", action="store_true")
    return p.parse_args(argv)


def clear_scene(keep_cube: bool) -> None:
    if keep_cube:
        return
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def make_material(name: str, props: dict):
    """Principled BSDF from viser's flat colour + opacity."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    color = props.get("color") or (200, 200, 200)
    rgb = [(c / 255.0) ** 2.2 for c in color]  # viser colours are sRGB
    opacity = props.get("opacity")
    alpha = 1.0 if opacity is None else float(opacity)
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Roughness"].default_value = 0.5
    if alpha < 1.0:
        mat.blend_method = "BLEND"
    return mat


def apply_shading(obj, props: dict, mat) -> None:
    if mat is not None and hasattr(obj.data, "materials"):
        obj.data.materials.append(mat)
    obj.visible_shadow = bool(props.get("cast_shadow", True))
    if props.get("flat_shading"):
        for poly in obj.data.polygons:
            poly.use_smooth = False
    elif hasattr(obj.data, "polygons"):
        for poly in obj.data.polygons:
            poly.use_smooth = True


def add_mesh_object(name: str, verts, faces, matrix: Matrix, props: dict):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.matrix_world = matrix
    apply_shading(obj, props, make_material(f"{name}_mat", props))
    return obj


def import_glb(node: dict, bundle: Path, matrix: Matrix):
    path = bundle / node["asset"]
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [o for o in bpy.data.objects if o not in before]
    roots = [o for o in imported if o.parent is None]

    scale = float(node["props"].get("scale", 1.0))
    holder = bpy.data.objects.new(_short(node["name"]), None)
    holder.empty_display_size = 0.05
    bpy.context.collection.objects.link(holder)
    holder.matrix_world = matrix @ GLTF_UNROTATE @ Matrix.Scale(scale, 4)

    for root in roots:
        local = root.matrix_basis.copy()
        root.parent = holder
        root.matrix_parent_inverse = Matrix.Identity(4)
        root.matrix_basis = local
    # matrix_world is cached; without this the new parenting is not reflected
    # until the next depsgraph evaluation (and anything we read back is stale).
    bpy.context.view_layer.update()

    cast = bool(node["props"].get("cast_shadow", True))
    for obj in imported:
        obj.visible_shadow = cast
    return holder


def add_light(node: dict, matrix: Matrix, args) -> None:
    kind = node["kind"]
    props = node["props"]
    name = _short(node["name"])
    intensity = float(props.get("intensity", 1.0))

    if kind == "DirectionalLightProps":
        light = bpy.data.lights.new(name, type="SUN")
        light.energy = intensity * args.sun_scale
        light.angle = math.radians(1.0)  # crisp shadows, like three.js
    elif kind == "PointLightProps":
        light = bpy.data.lights.new(name, type="POINT")
        light.energy = intensity * args.point_scale
        light.shadow_soft_size = 0.02
        distance = float(props.get("distance", 0.0))
        if distance > 0.0:
            light.use_custom_distance = True
            light.cutoff_distance = distance
    elif kind == "SpotLightProps":
        light = bpy.data.lights.new(name, type="SPOT")
        light.energy = intensity * args.point_scale
        light.spot_size = 2.0 * float(props.get("angle", 0.5))
        light.spot_blend = float(props.get("penumbra", 0.0))
    elif kind == "RectAreaLightProps":
        light = bpy.data.lights.new(name, type="AREA")
        light.shape = "RECTANGLE"
        light.size = float(props.get("width", 1.0))
        light.size_y = float(props.get("height", 1.0))
        light.energy = intensity * args.point_scale
    else:
        return  # Ambient/hemisphere are folded into the world, not objects.

    color = props.get("color") or (255, 255, 255)
    light.color = [(c / 255.0) ** 2.2 for c in color]
    if hasattr(light, "use_shadow"):
        light.use_shadow = bool(props.get("cast_shadow", True))

    obj = bpy.data.objects.new(name, light)
    bpy.context.collection.objects.link(obj)
    # Both three.js and Blender aim lights down local -Z, so the pose transfers
    # with no correction.
    obj.matrix_world = matrix


def add_curve(node: dict, bundle: Path, matrix: Matrix) -> None:
    """Splines and line segments become beveled curves, so they render as
    tubes with real thickness instead of viser's screen-space lines."""
    data = np.load(bundle / node["asset"])
    points = data["points"]
    props = node["props"]
    name = _short(node["name"])

    # viser line width is in pixels; there is no exact metric equivalent, so
    # approximate a tube radius that reads similarly at cover-image scale.
    radius = max(float(props.get("line_width", 2.0)) * 0.00025, 1e-4)

    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 4
    curve.fill_mode = "FULL"

    polylines = points.reshape(-1, 2, 3) if node["kind"] == "LineSegmentsProps" else [points]
    for line in polylines:
        spline = curve.splines.new("POLY")
        spline.points.add(len(line) - 1)
        for i, p in enumerate(line):
            spline.points[i].co = (float(p[0]), float(p[1]), float(p[2]), 1.0)
        spline.use_cyclic_u = bool(props.get("closed", False))

    color = props.get("color")
    if color is None:  # LineSegments store per-point colours; take the first.
        colors = data.get("colors")
        color = tuple(int(c) for c in colors.reshape(-1, 3)[0]) if colors is not None else (255, 255, 255)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.matrix_world = matrix
    obj.data.materials.append(make_material(f"{name}_mat", {"color": color}))


def add_point_cloud(node: dict, bundle: Path, matrix: Matrix) -> None:
    data = np.load(bundle / node["asset"])
    points = data["points"]
    name = _short(node["name"])
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(p) for p in points], [], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.matrix_world = matrix
    # Vertices alone render nothing; a geometry-nodes or particle setup is left
    # to the user, who will want to art-direct point size anyway.


def add_box(node: dict, matrix: Matrix):
    dx, dy, dz = node["props"]["dimensions"]
    verts = np.array(
        [
            (x, y, z)
            for x in (-dx / 2, dx / 2)
            for y in (-dy / 2, dy / 2)
            for z in (-dz / 2, dz / 2)
        ]
    )
    faces = [
        (0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
        (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3),
    ]
    return add_mesh_object(_short(node["name"]), verts, faces, matrix, node["props"])


def add_icosphere(node: dict, matrix: Matrix):
    props = node["props"]
    bpy.ops.mesh.primitive_ico_sphere_add(
        radius=float(props["radius"]),
        subdivisions=int(props.get("subdivisions", 3)),
    )
    obj = bpy.context.active_object
    obj.name = _short(node["name"])
    obj.matrix_world = matrix
    apply_shading(obj, props, make_material(f"{obj.name}_mat", props))
    return obj


def setup_camera(cam: dict, resolution: tuple[int, int] | None = None) -> None:
    data = bpy.data.cameras.new("viser_camera")
    data.sensor_fit = "VERTICAL"
    data.angle_y = float(cam["fov"])
    data.clip_start = max(float(cam["near"]), 1e-4)
    data.clip_end = float(cam["far"])
    obj = bpy.data.objects.new("viser_camera", data)
    bpy.context.collection.objects.link(obj)

    # Build the pose from position/look_at/up rather than viser's quaternion, so
    # we never have to reason about which camera convention each side uses.
    eye = Vector(cam["position"])
    forward = (Vector(cam["look_at"]) - eye).normalized()
    up = Vector(cam["up"]).normalized()
    right = forward.cross(up).normalized()
    true_up = right.cross(forward)
    # Blender cameras look down -Z with +Y up.
    rot = Matrix((right, true_up, -forward)).transposed().to_4x4()
    obj.matrix_world = Matrix.Translation(eye) @ rot

    scene = bpy.context.scene
    scene.camera = obj
    width, height = resolution or (
        int(cam.get("image_width") or 1920),
        int(cam.get("image_height") or 1080),
    )
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    # Vertical FOV is preserved, so a resolution with a different aspect ratio
    # than the browser widens or crops the frame horizontally.


def setup_world(manifest: dict, args) -> None:
    world = bpy.data.worlds.new("viser_world")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    bg = nodes["Background"]
    bg.inputs["Strength"].default_value = args.world_strength

    if args.hdri:
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(args.hdri)
        links.new(env.outputs["Color"], bg.inputs["Color"])
        return

    # Ambient/hemisphere lights have no object form in Blender; fold whatever
    # the scene had into a flat world colour.
    color, strength = (0.05, 0.05, 0.06), args.world_strength
    for node in manifest["nodes"]:
        props = node["props"]
        if node["kind"] == "AmbientLightProps":
            color = tuple((c / 255.0) ** 2.2 for c in props["color"])
            strength = float(props["intensity"]) * args.world_strength
        elif node["kind"] == "HemisphereLightProps":
            sky = np.array(props["sky_color"], float) / 255.0
            ground = np.array(props["ground_color"], float) / 255.0
            color = tuple(((sky + ground) / 2.0) ** 2.2)
            strength = float(props["intensity"]) * args.world_strength
    bg.inputs["Color"].default_value = (*color, 1.0)
    bg.inputs["Strength"].default_value = strength

    if manifest.get("environment_map"):
        print(
            f"[viser2blender] scene used the '{manifest['environment_map']}' viser "
            "environment map; pass --hdri <file.exr> to match it in Blender."
        )


def _short(name: str) -> str:
    return name.strip("/").replace("/", "_") or "root"


def build(manifest: dict, bundle: Path, args) -> None:
    for node in manifest["nodes"]:
        matrix = Matrix([list(row) for row in node["matrix"]])
        kind = node["kind"]
        if kind in ("GlbProps", "BatchedGlbProps"):
            import_glb(node, bundle, matrix)
        elif kind in ("MeshProps", "SkinnedMeshProps", "BatchedMeshesProps"):
            data = np.load(bundle / node["asset"])
            add_mesh_object(
                _short(node["name"]), data["vertices"], data["faces"], matrix, node["props"]
            )
        elif kind == "BoxProps":
            add_box(node, matrix)
        elif kind == "IcosphereProps":
            add_icosphere(node, matrix)
        elif kind == "PointCloudProps":
            add_point_cloud(node, bundle, matrix)
        elif kind.endswith("SplineProps") or kind == "LineSegmentsProps":
            add_curve(node, bundle, matrix)
        elif kind.endswith("LightProps"):
            add_light(node, matrix, args)
        else:
            print(f"[viser2blender] unhandled node kind: {kind} ({node['name']})")


def resolve_engine(requested: str) -> str:
    """Map an engine name onto whatever this Blender build calls it.

    The EEVEE identifier moved around across releases (BLENDER_EEVEE in 4.1 and
    again in 5.x, BLENDER_EEVEE_NEXT in 4.2-4.5), so match against the live enum
    instead of trusting a hard-coded name.
    """
    available = [
        item.identifier
        for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    ]
    if requested in available:
        return requested
    key = requested.upper().replace("BLENDER_", "").replace("_NEXT", "")
    for candidate in available:
        if key in candidate.upper():
            return candidate
    raise SystemExit(f"engine {requested!r} unavailable; this Blender has {available}")


def enable_gpu() -> None:
    """Point Cycles at every available GPU.

    Two settings are needed and neither is implied by the other: the addon
    preference picks the backend (and is not read from a saved GUI config in
    background mode), while scene.cycles.device opts this scene in.
    """
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue  # not compiled into this build
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type != "CPU"]
        if gpus:
            for device in prefs.devices:
                device.use = device.type != "CPU"
            bpy.context.scene.cycles.device = "GPU"
            print(f"[viser2blender] Cycles on {backend}: "
                  f"{', '.join(d.name for d in gpus)}")
            return
    print("[viser2blender] --gpu requested but no GPU found; rendering on CPU.")


def render(path: str, args) -> None:
    scene = bpy.context.scene
    engine = resolve_engine(args.engine)
    scene.render.engine = engine
    if engine == "CYCLES":
        scene.cycles.samples = args.samples
        if args.gpu:
            enable_gpu()
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.samples
    scene.render.film_transparent = False
    scene.render.filepath = str(Path(path).absolute())
    bpy.ops.render.render(write_still=True)
    print(f"[viser2blender] rendered -> {path}")


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parse_args(argv)

    bundle = Path(args.bundle)
    manifest = json.loads((bundle / "scene.json").read_text())
    if manifest.get("format") != "viser2blender":
        raise SystemExit(f"{bundle} is not a viser2blender bundle")

    clear_scene(args.keep_default_cube)
    build(manifest, bundle, args)
    setup_world(manifest, args)
    if manifest.get("camera"):
        setup_camera(manifest["camera"], args.resolution)
    else:
        print("[viser2blender] no camera in bundle (no browser was connected).")
        if args.resolution:
            bpy.context.scene.render.resolution_x = args.resolution[0]
            bpy.context.scene.render.resolution_y = args.resolution[1]

    print(f"[viser2blender] built {len(manifest['nodes'])} nodes from {bundle}")
    if args.render:
        render(args.render, args)


if __name__ == "__main__":
    main()
