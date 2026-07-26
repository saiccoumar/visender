"""visender example: an 18 m RX-78 on a landing pad, posed by clicking on it.

    python examples/gundam.py [--urdf /path/to/robot.urdf]

Frame it in the browser, click **Export to Blender**, then render the bundle
with `visender render examples/gundam.yaml`. The scene is *posable*: click any
body part to select the joint that drives it, dial the slider, deselect. The
**Lights** tab adds and removes coloured directional/point/ambient lights, each
on its own gizmo, before exporting.

What that demonstrates about the exporter:

- **Gizmos are skipped; their transforms are not.** A light lives at
  `/ctrl/light_<n>/light`. The `TransformControls` nodes themselves never reach
  Blender, but the pose you dragged them to is composed down the path, so the
  light lands exactly where you aimed it.
- **`node_filter`** drops the little glowing bulb markers and the selection
  axes, which exist only so the lights and the picked joint are visible in the
  browser.

The model is the GUNDAM GLOBAL CHALLENGE test model vendored under
`examples/gundam_model/` (CC BY-NC-SA; see `meshes/LICENSE`). It stands ~18 m
tall with its feet at z = 0, so every prop, light and distance in this scene is
sized in *those* metres — a mech-scale version of the same setup.
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

import visender

# The GGC RX-78 test model vendored next to this script. Override with --urdf;
# any URDF whose visual meshes resolve will do.
DEFAULT_URDF = Path(__file__).parent / "gundam_model/GGC_TestModel_rx78_20170112.urdf"

# A relaxed standing pose: arms out of the torso silhouette, head turned
# slightly into the key light. Joints not named here stay at 0, and everything
# is clipped to the URDF's own limits, so this survives an --urdf swap.
HOME_POSE = {
    "head_neck_y": 0.30,
    "head_neck_p": 0.10,
    "larm_shoulder_p": -0.25,
    "larm_shoulder_r": 0.22,
    "larm_elbow_p": -0.55,
    "rarm_shoulder_p": -0.25,
    "rarm_shoulder_r": -0.22,
    "rarm_elbow_p": -0.55,
    "lleg_crotch_r": 0.04,
    "rleg_crotch_r": -0.04,
}


def load_urdf(path: Path):
    """Load a URDF, resolving ``package://`` against the URDF's own directory."""
    import logging

    import yourdfpy

    # yourdfpy re-warns about every mimicked finger joint on *each* update_cfg —
    # this model has dozens of them, which buries anything useful.
    logging.getLogger("yourdfpy").setLevel(logging.ERROR)

    root = path.parent

    def handler(fname: str) -> str:
        return os.path.join(root, fname.removeprefix("package://"))

    return yourdfpy.URDF.load(
        str(path), load_collision_meshes=False, filename_handler=handler
    )


def home_cfg(urdf) -> np.ndarray:
    """HOME_POSE as a configuration vector, clipped to the URDF's own limits."""
    values = []
    for name in urdf.actuated_joint_names:
        limit = urdf.joint_map[name].limit
        value = HOME_POSE.get(name, 0.0)
        if limit is not None:
            value = float(np.clip(value, limit.lower, limit.upper))
        values.append(value)
    return np.array(values)


def joint_driving_link(urdf, link: str) -> str | None:
    """The nearest actuated joint above ``link``, or None if it is rigid to the base.

    Clicking a forearm mesh should grab the elbow, not one of the fixed joints
    that glue the forearm's shells together — so walk up the tree until an
    actuated joint turns up.
    """
    actuated = set(urdf.actuated_joint_names)
    parent_joint = {joint.child: name for name, joint in urdf.joint_map.items()}
    while link in parent_joint:
        name = parent_joint[link]
        if name in actuated:
            return name
        link = urdf.joint_map[name].parent
    return None


def child_joints_of(urdf, joint: str) -> list[str]:
    """The actuated joints immediately below ``joint`` in the kinematic tree.

    Descends through fixed joints the same way :func:`joint_driving_link`
    climbs through them, so one step down lands on the next thing you can pose.
    A branch point (the torso, say) returns several.
    """
    actuated = set(urdf.actuated_joint_names)
    below: dict[str, list[str]] = {}
    for name, j in urdf.joint_map.items():
        below.setdefault(j.parent, []).append(name)

    found: list[str] = []
    stack = [urdf.joint_map[joint].child]
    while stack:
        for name in below.get(stack.pop(), []):
            if name in actuated:
                found.append(name)
            else:
                stack.append(urdf.joint_map[name].child)
    return found


def add_joint_picker(server, urdf, viser_urdf, cfg: np.ndarray, container=None) -> None:
    """Click a link in the viewport to pose the joint that drives it.

    Clicking only ever reaches joints whose links you can see and hit, so the
    panel also walks the kinematic chain: **▲ Up** steps to the parent joint
    (the hip above a knee, say), **▼ Down** to the joint below, and a dropdown
    appears at branch points like the torso.

    ``cfg`` is the live configuration vector: the slider writes into it, so the
    pose survives selecting a different joint, and `Reset pose` restores the
    home configuration it was seeded with.

    Returns a ``refresh()`` callable that pushes ``cfg`` back into the viewport
    and the open slider, for callers that write the vector themselves.
    """
    home = cfg.copy()
    joint_index = {name: i for i, name in enumerate(urdf.actuated_joint_names)}

    # Meshes hang off a frame per link, e.g.
    # `/robot/visual/<link>/<...>/<geometry>.dae`, so the owning link is the
    # deepest path segment that names a link, and the frame to parent the
    # selection axes to is the path up to and including that segment.
    def link_of(node_name: str) -> str | None:
        for segment in reversed(node_name.split("/")):
            if segment in urdf.link_map:
                return segment
        return None

    # Every link's frame path, harvested once from the nodes ViserUrdf created.
    # Navigating with the arrows lands on joints whose links were never clicked,
    # so the axes cannot be derived from the click path alone.
    frame_of: dict[str, str] = {}
    for node in (*viser_urdf._joint_frames, *viser_urdf._meshes):  # noqa: SLF001
        segments = node.name.split("/")
        for i, segment in enumerate(segments):
            if segment in urdf.link_map:
                frame_of.setdefault(segment, "/".join(segments[: i + 1]))

    # Widgets are (re)created inside click callbacks, long after the `with tab:`
    # block that set up the panel has exited — so re-enter the container each
    # time, or they land at the top level of the GUI instead of in the tab.
    panel = container if container is not None else contextlib.nullcontext()

    with panel:
        PROMPT = "**Click a body part to select its joint.**"
        status = server.gui.add_markdown(PROMPT)
    selected: str | None = None
    widgets: list = []
    marker: viser.SceneNodeHandle | None = None

    def clear() -> None:
        nonlocal selected, widgets, marker
        for handle in widgets:
            handle.remove()
        if marker is not None:
            marker.remove()
        selected, widgets, marker = None, [], None
        status.content = PROMPT

    def select_joint(joint: str) -> None:
        nonlocal selected, marker
        clear()
        selected = joint
        index = joint_index[joint]
        link = urdf.joint_map[joint].child
        limit = urdf.joint_map[joint].limit
        status.content = f"**{joint}**  \n`{link}`"

        frame = frame_of.get(link)
        if frame is not None:
            marker = server.scene.add_frame(
                f"{frame}/_selection", show_axes=True, axes_length=3.0,
                axes_radius=0.08, origin_radius=0.2,
            )

        with panel:
            slider = server.gui.add_slider(
                joint, min=float(limit.lower), max=float(limit.upper),
                step=1e-3, initial_value=float(cfg[index]),
            )

        @slider.on_update
        def _(_) -> None:
            cfg[index] = slider.value
            viser_urdf.update_cfg(cfg)

        widgets.append(slider)

        # ---- walk the chain, for joints no mesh click can reach comfortably
        parent = joint_driving_link(urdf, urdf.joint_map[joint].parent)
        children = child_joints_of(urdf, joint)
        with panel:
            up = server.gui.add_button("▲ Up (parent joint)")
            if len(children) > 1:
                # A branch point: name the options rather than pick one for you.
                down = server.gui.add_dropdown(
                    "▼ Down (child joint)",
                    options=("—", *children), initial_value="—",
                )
            else:
                down = server.gui.add_button("▼ Down (child joint)")
            deselect = server.gui.add_button("Deselect")

        up.disabled = parent is None
        if parent is not None:
            up.on_click(lambda _, parent=parent: select_joint(parent))
        widgets.append(up)

        if len(children) > 1:
            down.on_update(
                lambda _: select_joint(down.value) if down.value != "—" else None
            )
        else:
            down.disabled = not children
            if children:
                down.on_click(lambda _, child=children[0]: select_joint(child))
        widgets.append(down)

        deselect.on_click(lambda _: clear())
        widgets.append(deselect)

    def select_node(node_name: str) -> None:
        nonlocal marker
        link = link_of(node_name)
        joint = joint_driving_link(urdf, link) if link is not None else None
        if joint is not None:
            select_joint(joint)
            return

        # A part that is rigid to the base still drops the previous selection —
        # otherwise the panel would describe one link while the axes and the
        # slider still belonged to another.
        clear()
        status.content = f"`{link}` is fixed to the base — nothing to pose."
        # No joint to offer, but still give the note a way out.
        with panel:
            deselect = server.gui.add_button("Deselect")
        deselect.on_click(lambda _: clear())
        widgets.append(deselect)

    for mesh in viser_urdf._meshes:  # noqa: SLF001 — no public accessor yet
        mesh.on_click(lambda event: select_node(event.target.name))

    with panel:
        reset = server.gui.add_button("Reset pose")

    def refresh() -> None:
        """Re-read ``cfg`` into the viewport and the open slider.

        Anything that writes ``cfg`` from outside this panel -- a timeline
        scrub in ``gundam_video.py``, say -- must call this, or the slider goes
        on displaying the value it was created with while driving a joint that
        has since moved.
        """
        viser_urdf.update_cfg(cfg)
        if selected is not None:
            widgets[0].value = float(cfg[joint_index[selected]])

    @reset.on_click
    def _(_) -> None:
        cfg[:] = home
        refresh()

    return refresh


# Sensible starting energies per light kind at this scale. Point-light falloff
# is quadratic, so what lit a desk from 1 m away is nothing from 20 m up.
LIGHT_DEFAULTS = {
    "directional": {"intensity": 2.2, "position": (16.0, -12.0, 22.0),
                    "color": (255, 236, 180)},
    "point": {"intensity": 3000.0, "position": (-10.0, 14.0, 15.0),
              "color": (200, 220, 255)},
    "ambient": {"intensity": 0.4, "position": (0.0, 0.0, 0.0),
                "color": (180, 200, 220)},
}


def add_light_panel(server, container) -> list:
    """Add/remove lights from the GUI, each with its own color and intensity.

    Every light gets a folder with a color picker, an intensity slider and a
    **Remove** button; directional and point lights also get a transform gizmo
    to aim or place them. The gizmos never reach Blender — the poses do.

    Returns the list of live gizmo handles (used to force them visible).
    """
    gizmos: list = []
    counter = 0

    def add_light(kind: str, color=None, intensity=None, position=None,
                  wxyz=(1.0, 0.0, 0.0, 0.0)):
        nonlocal counter
        counter += 1
        defaults = LIGHT_DEFAULTS[kind]
        color = tuple(color if color is not None else defaults["color"])
        intensity = float(intensity if intensity is not None
                          else defaults["intensity"])
        position = tuple(position if position is not None else defaults["position"])
        base = f"/ctrl/light_{counter}"

        # An ambient light has no position to speak of, so it gets no gizmo.
        # A directional light is aimed by *rotating* its gizmo; a point light
        # only cares where you put it, so its rotation handles are disabled.
        gizmo = None
        if kind != "ambient":
            gizmo = server.scene.add_transform_controls(
                base, scale=4.0, position=position, wxyz=wxyz,
                disable_rotations=kind == "point",
            )
            gizmos.append(gizmo)

        add_fn = {
            "directional": server.scene.add_light_directional,
            "point": server.scene.add_light_point,
            "ambient": server.scene.add_light_ambient,
        }[kind]
        light = add_fn(f"{base}/light", color=color, intensity=intensity)
        if gizmo is None:
            light.position = position

        # A browser-only marker so the light is visible in the viewport;
        # node_filter drops every `/bulb` on export.
        bulb = None
        if gizmo is not None:
            bulb = server.scene.add_icosphere(
                f"{base}/bulb", radius=0.7, color=color)

        with container:
            folder = server.gui.add_folder(f"{kind} {counter}")
        with folder:
            swatch = server.gui.add_rgb("Color", initial_value=color)
            energy = server.gui.add_number(
                "Intensity", initial_value=intensity, min=0.0,
                step=0.1 if kind != "point" else 50.0,
            )
            remove = server.gui.add_button("Remove")

        @swatch.on_update
        def _(_) -> None:
            light.color = swatch.value
            if bulb is not None:
                bulb.color = swatch.value

        energy.on_update(lambda _: setattr(light, "intensity", float(energy.value)))

        @remove.on_click
        def _(_) -> None:
            for handle in (bulb, light, gizmo):
                if handle is not None:
                    handle.remove()
            if gizmo in gizmos:
                gizmos.remove(gizmo)
            folder.remove()

        return light

    with container:
        kind_input = server.gui.add_dropdown(
            "Type", options=("directional", "point", "ambient"),
            initial_value="directional",
        )
        new_color = server.gui.add_rgb("New light color", initial_value=(255, 255, 255))
        add_button = server.gui.add_button("Add light")

    add_button.on_click(
        lambda _: add_light(kind_input.value, color=new_color.value))

    # The scene starts with the same key/fill pair the render config expects.
    add_light("directional", wxyz=(0.8642, -0.3558, 0.3558, 0.1017))
    add_light("point")
    return gizmos


def build_scene(server: viser.ViserServer, urdf_path: Path = DEFAULT_URDF) -> None:
    """Populate ``server`` with the pad, robot, props, lights and export button."""
    if not urdf_path.exists():
        raise SystemExit(
            f"No URDF at {urdf_path}\n"
            "Pass --urdf /path/to/robot.urdf (visual meshes must resolve)."
        )

    server.scene.set_up_direction("+z")

    # The landing pad the robot stands on. Its top surface is z = 0, so the
    # URDF's feet sit flush on it.
    server.scene.add_box(
        "/pad", dimensions=(30.0, 30.0, 0.6), position=(0.0, 0.0, -0.3),
        color=(58, 60, 66),
    )

    # ------------------------------------------------------------------ robot
    urdf = load_urdf(urdf_path)
    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot")
    cfg = home_cfg(urdf)
    viser_urdf.update_cfg(cfg)

    # ------------------------------------------------------------------- gui
    tabs = server.gui.add_tab_group()
    pose_tab = tabs.add_tab("Pose")
    lights_tab = tabs.add_tab("Lights")

    # Click-to-pose: pick a body part in the viewport, dial its joint, deselect.
    add_joint_picker(server, urdf, viser_urdf, cfg, container=pose_tab)

    gizmos = add_light_panel(server, lights_tab)

    # ------------------------------------------------------------------ export
    visender.add_export_button(
        server,
        out_dir=Path(__file__).parent / "out/gundam",
        node_filter=lambda name: not name.endswith(("/bulb", "/_selection")),
    )

    for ctrl in gizmos:
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
