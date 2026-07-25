"""bliser example: a Panda on a desk, with draggable props and lights.

    python examples/panda_desk.py [--urdf /path/to/panda.urdf]

Frame it in the browser, click **Export to Blender**, then render the bundle
with `bliser render examples/panda_desk.yaml`. The scene is *posable*: joint
sliders drive the URDF, and four transform gizmos let you drag two props and
aim the lights before exporting.

What that demonstrates about the exporter:

- **Gizmos are skipped; their transforms are not.** The cube lives at
  `/ctrl/cube/mesh` and the key light at `/ctrl/key/light`. The
  `TransformControls` nodes themselves never reach Blender, but the pose you
  dragged them to is composed down the path, so the prop and the light land
  exactly where you put them.
- **`node_filter`** drops the little glowing bulb markers, which exist only so
  the lights are visible in the browser.
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

import bliser

# A Panda that ships with the vamp resources in this monorepo. Override with
# --urdf; any URDF whose meshes resolve will do.
DEFAULT_URDF = Path.home() / "Work/vamp-shru/resources/panda/panda.urdf"

# Elbow-up ready pose — the one every Panda figure is composed around.
HOME_CFG = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def load_urdf(path: Path):
    """Load a URDF, resolving ``package://`` against the URDF's own directory."""
    import logging

    import yourdfpy

    # yourdfpy re-warns about the Panda's mimicked finger joint on *every*
    # update_cfg — once per slider drag, which buries anything useful.
    logging.getLogger("yourdfpy").setLevel(logging.ERROR)

    root = path.parent

    def handler(fname: str) -> str:
        return os.path.join(root, fname.removeprefix("package://"))

    return yourdfpy.URDF.load(
        str(path), load_collision_meshes=False, filename_handler=handler
    )


def build_scene(server: viser.ViserServer, urdf_path: Path = DEFAULT_URDF) -> None:
    """Populate ``server`` with the desk, robot, props, lights and export button."""
    if not urdf_path.exists():
        raise SystemExit(
            f"No URDF at {urdf_path}\n"
            "Pass --urdf /path/to/robot.urdf (visual meshes must resolve)."
        )

    server.scene.set_up_direction("+z")

    # The desk the robot is bolted to. Its top surface is z = 0, so the URDF
    # base sits flush on it.
    server.scene.add_box(
        "/desk", dimensions=(1.6, 1.1, 0.04), position=(0.15, 0.0, -0.02),
        color=(58, 60, 66),
    )

    # ------------------------------------------------------------------ robot
    urdf = load_urdf(urdf_path)
    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot")
    viser_urdf.update_cfg(HOME_CFG)

    # Joint sliders, so the pose in the render is one you dialled in. Limits
    # come from the URDF; the slider set adapts to whatever robot you loaded.
    with server.gui.add_folder("Joints"):
        sliders: list[viser.GuiInputHandle[float]] = []
        for i, name in enumerate(urdf.actuated_joint_names):
            lower, upper = urdf.joint_map[name].limit.lower, urdf.joint_map[name].limit.upper
            initial = float(HOME_CFG[i]) if i < len(HOME_CFG) else 0.0
            slider = server.gui.add_slider(
                name.replace("panda_", ""), min=float(lower), max=float(upper),
                step=1e-3, initial_value=np.clip(initial, lower, upper),
            )
            slider.on_update(lambda _: viser_urdf.update_cfg(
                np.array([s.value for s in sliders])))
            sliders.append(slider)

        reset = server.gui.add_button("Reset pose")

        @reset.on_click
        def _(_) -> None:
            for slider, value in zip(sliders, HOME_CFG):
                slider.value = float(value)

    # ------------------------------------------------- props on transform ctrls
    # Each prop hangs off a gizmo: drag it in the browser, and the pose rides
    # along into Blender even though the gizmo itself is never exported.
    cube_ctrl = server.scene.add_transform_controls(
        "/ctrl/cube", scale=0.18, position=(0.45, 0.22, 0.03))
    server.scene.add_box("/ctrl/cube/mesh", dimensions=(0.06, 0.06, 0.06),
                         color=(214, 108, 42))

    ball_ctrl = server.scene.add_transform_controls(
        "/ctrl/ball", scale=0.18, disable_rotations=True,
        position=(0.42, -0.24, 0.04))
    server.scene.add_icosphere("/ctrl/ball/mesh", radius=0.04,
                               color=(70, 130, 200))

    # ------------------------------------------------------- lighting gizmos
    # A directional light is aimed by *rotating* its gizmo; a point light only
    # cares where you put it, so its gizmo has rotation disabled. The `bulb`
    # spheres are browser-only markers — node_filter drops them on export.
    key_ctrl = server.scene.add_transform_controls(
        "/ctrl/key", scale=0.25, position=(0.9, -0.7, 1.1),
        wxyz=(0.8642, -0.3558, 0.3558, 0.1017))
    server.scene.add_light_directional("/ctrl/key/light", intensity=2.2)
    server.scene.add_icosphere("/ctrl/key/bulb", radius=0.035,
                               color=(255, 236, 180))

    fill_ctrl = server.scene.add_transform_controls(
        "/ctrl/fill", scale=0.2, disable_rotations=True,
        position=(-0.35, 0.55, 0.75))
    server.scene.add_light_point("/ctrl/fill/light", intensity=8.0,
                                 color=(200, 220, 255))
    server.scene.add_icosphere("/ctrl/fill/bulb", radius=0.03,
                               color=(200, 220, 255))

    # ------------------------------------------------------------------ export
    bliser.add_export_button(
        server,
        out_dir=Path(__file__).parent / "out/panda_desk",
        node_filter=lambda name: not name.endswith("/bulb"),
    )

    for ctrl in (cube_ctrl, ball_ctrl, key_ctrl, fill_ctrl):
        ctrl.visible = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    args = parser.parse_args()

    server = viser.ViserServer()
    build_scene(server, args.urdf)

    print("Pose the robot, drag the props and lights, then click "
          "'Export to Blender'.")
    server.sleep_forever()


if __name__ == "__main__":
    main()
