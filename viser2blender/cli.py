"""``v2b`` command-line wrapper. Runs in the *solver* env (may import yaml).

    v2b render pen_grip.yaml --profile final [--output cover.png] [blender flags...]
    v2b list-nodes <bundle>
    v2b init <bundle> > pen_grip.yaml

The wrapper resolves a YAML config to a flat JSON dict and hands it to Blender
via ``blender_import.py --config``. It never imports ``blender_import`` (that
module imports ``bpy``); it locates the file by path instead. Precedence is
config file < profile < explicit CLI flag: config+profile are written to the
JSON, and any extra Blender flags passed on the command line are forwarded
straight through, where the Blender side lets an explicit flag win over config.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from . import config as _config
except ModuleNotFoundError as exc:  # pragma: no cover - install-shape guard
    if exc.name != "yaml":
        raise
    raise SystemExit(
        "v2b needs pyyaml to read a config file: pip install 'viser2blender[cli]' "
        "(or pip install pyyaml). Blender's own Python does not need it -- "
        "blender_import.py never imports yaml."
    ) from exc

# Built-in quality shorthands mapping onto typical profile settings, for when no
# config profile has been written yet.
QUALITY = {
    "draft":   {"engine": "EEVEE", "samples": 64, "scale": 100},
    "preview": {"engine": "CYCLES", "samples": 256, "adaptive_threshold": 0.01},
    "final":   {"engine": "CYCLES", "samples": 8192, "adaptive_threshold": 0.001,
                "max_bounces": 24, "gpu": True},
}


def _blender_import_path() -> str:
    return str(Path(__file__).parent / "blender_import.py")


def locate_blender(explicit: str | None, config_value: str | None) -> str:
    """Blender binary: --blender > config > $BLENDER > which > /opt glob."""
    tried = []
    for candidate in (explicit, config_value, os.environ.get("BLENDER")):
        tried.append(candidate)
        if candidate and Path(candidate).exists():
            return candidate
    which = shutil.which("blender")
    tried.append("shutil.which('blender')")
    if which:
        return which
    globbed = sorted(glob.glob("/opt/blender/blender-*/blender"))
    tried.append("/opt/blender/blender-*/blender")
    if globbed:
        return globbed[-1]  # newest by lexical sort
    raise SystemExit("could not locate Blender. Tried: "
                     + ", ".join(str(t) for t in tried if t is not None)
                     + ". Pass --blender PATH or set 'blender:' in the config.")


def cmd_render(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="v2b render")
    p.add_argument("config", help="YAML config file")
    p.add_argument("--profile", default=None)
    p.add_argument("--quality", choices=sorted(QUALITY), default=None,
                   help="Shorthand profile when no config profile is written.")
    p.add_argument("--output", default=None, help="Override the output image path.")
    p.add_argument("--blender", default=None, help="Path to the Blender binary.")
    args, passthrough = p.parse_known_args(argv)

    resolved, cfg_blender = _config.resolve(
        args.config, profile=args.profile, output=args.output)

    if args.quality:
        if args.profile is not None:
            raise SystemExit("--quality is a stand-in for a config profile; pass one or the "
                             "other, not both (--profile "
                             f"{args.profile!r} and --quality {args.quality!r}).")
        for k, v in QUALITY[args.quality].items():
            resolved.setdefault(k, v)

    if "bundle" not in resolved:
        raise SystemExit(f"{args.config}: no 'bundle:' set.")

    blender = locate_blender(args.blender, cfg_blender)

    with tempfile.NamedTemporaryFile("w", suffix=".v2b.json", delete=False) as fh:
        json.dump(resolved, fh)
        json_path = fh.name

    cmd = [blender, "-b", "--python", _blender_import_path(), "--",
           "--config", json_path, *passthrough]
    print(f"[v2b] {' '.join(cmd)}", flush=True)
    try:
        return subprocess.run(cmd).returncode
    finally:
        try:
            os.unlink(json_path)
        except OSError:
            pass


def cmd_list_nodes(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="v2b list-nodes")
    p.add_argument("bundle")
    args = p.parse_args(argv)
    bundle = Path(args.bundle)
    manifest = _config.load_manifest(bundle)
    print(f"{'kind':<24} {'count':>10}  name")
    for node in manifest["nodes"]:
        print(f"{node['kind']:<24} {_node_count(bundle, node):>10}  {node['name']}")
    return 0


def _node_count(bundle: Path, node: dict) -> str:
    """Vertex/point count for a node, read from its npz asset if present."""
    asset = node.get("asset", "")
    if not asset.endswith(".npz"):
        return ""
    try:
        import numpy as np
        with np.load(bundle / asset) as data:
            for key in ("vertices", "points"):
                if key in data.files:
                    return f"{len(data[key])} {'verts' if key == 'vertices' else 'pts'}"
    except Exception:
        return ""
    return ""


def cmd_init(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="v2b init")
    p.add_argument("bundle")
    args = p.parse_args(argv)
    sys.stdout.write(_config.scaffold(args.bundle))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: v2b {render|list-nodes|init} ...", file=sys.stderr)
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    dispatch = {"render": cmd_render, "list-nodes": cmd_list_nodes, "init": cmd_init}
    if cmd not in dispatch:
        print(f"unknown subcommand {cmd!r}; expected one of {sorted(dispatch)}", file=sys.stderr)
        return 2
    return dispatch[cmd](rest)


if __name__ == "__main__":
    raise SystemExit(main())
