"""Snapshot a live viser scene and rebuild it in Blender for rendering.

Add a button to any viser sandbox::

    import visender
    visender.add_export_button(server, out_dir="renders/pen_grip")

Then, once you have clicked it::

    blender --python external/visender/visender/blender_import.py \
            -- --bundle renders/pen_grip_142530

See :func:`export_scene` for the programmatic entry point.
"""

from ._export import add_export_button, export_scene

__all__ = ["add_export_button", "export_scene"]
__version__ = "0.1.0"
