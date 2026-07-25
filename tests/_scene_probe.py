"""Run the viser2blender pipeline inside Blender and report the scene as JSON.

Executed by ``test_blender_integration.py`` via ``blender -b --python``. Runs
every stage except the render itself (so the tests stay cheap), then prints one
``PROBE_JSON:`` line describing what ended up in the scene.

Usage::

    blender -b --python _scene_probe.py -- --bundle DIR [blender_import flags...]
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import bpy

HERE = Path(__file__).resolve().parent
BLENDER_IMPORT = HERE.parent / "viser2blender" / "blender_import.py"

_spec = importlib.util.spec_from_file_location("v2b_blender_import", BLENDER_IMPORT)
bi = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bi
_spec.loader.exec_module(bi)


class ProbePipeline(bi.Pipeline):
    def render(self) -> None:  # never render, never save
        pass


def vec(v) -> list:
    return [round(float(c), 6) for c in v]


def report(pipeline) -> dict:
    # Nothing has drawn a frame, so object matrices are still stale w.r.t. the
    # loc/rot fields the builders set. Flush them before reading matrix_world.
    bpy.context.view_layer.update()
    scene = bpy.context.scene
    objects = list(bpy.data.objects)

    by_type: dict[str, int] = {}
    for obj in objects:
        by_type[obj.type] = by_type.get(obj.type, 0) + 1

    materials: dict[str, list] = {}
    poly_materials: dict[str, list] = {}
    vertex_counts: dict[str, int] = {}
    for node, created in pipeline.created.items():
        slots, polys, verts = [], set(), 0
        for obj in created:
            if obj.type != "MESH":
                continue
            slots += [m.name if m else "" for m in obj.data.materials]
            verts += len(obj.data.vertices)
            for poly in obj.data.polygons:
                if poly.material_index < len(obj.data.materials):
                    mat = obj.data.materials[poly.material_index]
                    polys.add(mat.name if mat else "")
        materials[node] = slots
        poly_materials[node] = sorted(polys)
        vertex_counts[node] = verts

    datablock_counts: dict[str, int] = {}
    for mat in bpy.data.materials:
        stem = mat.name.split(".")[0]
        datablock_counts[stem] = datablock_counts.get(stem, 0) + 1

    lights = [o for o in objects if o.type == "LIGHT"]
    cam = scene.camera
    camera = None
    if cam is not None:
        camera = {
            "location": vec(cam.matrix_world.translation),
            # A Blender camera looks down its local -Z.
            "direction": vec(cam.matrix_world.to_quaternion() @
                             bi.Vector((0.0, 0.0, -1.0))),
            "resolution": [scene.render.resolution_x, scene.render.resolution_y],
            "resolution_percentage": scene.render.resolution_percentage,
            "sensor_fit": cam.data.sensor_fit,
            "angle_y": round(float(cam.data.angle_y), 6),
            "use_dof": bool(cam.data.dof.use_dof),
            "focus_distance": round(float(cam.data.dof.focus_distance), 6),
        }

    bounds = bi.scene_bounds()
    backdrop = bpy.data.objects.get(bi.BACKDROP_NAME)

    return {
        "created": sorted(pipeline.created),
        "object_names": sorted(o.name for o in objects),
        "objects_by_type": by_type,
        "mesh_vertex_counts": vertex_counts,
        "materials": materials,
        "polygon_material_names": poly_materials,
        "material_datablock_counts": datablock_counts,
        "light_energies": {o.name: round(float(o.data.energy), 6) for o in lights},
        "light_directions": {
            o.name: vec(o.matrix_world.to_quaternion() @ bi.Vector((0.0, 0.0, -1.0)))
            for o in lights
        },
        "camera": camera,
        "scene_bounds": [vec(bounds[0]), vec(bounds[1])] if bounds else None,
        "backdrop_z": round(float(backdrop.location.z), 6) if backdrop else None,
        "world_nodes": sorted(n.bl_idname for n in scene.world.node_tree.nodes)
        if scene.world and scene.world.node_tree else [],
        "engine": scene.render.engine,
    }


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = bi.parse_args(argv)
    settings = bi.Settings.from_args(
        args, bi._explicit_flags(argv),
        bi.load_config(args.config) if args.config else None)
    pipeline = ProbePipeline(settings).run()
    print("PROBE_JSON:" + json.dumps(report(pipeline)))


main()
