#!/usr/bin/env python3
"""Install the isolated annealer and build its pinned native geometry adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import venv

FLOORPLANNER_PIN = "7835a4ef645f7c9faccc4a74c678574285a7c82b"
DEFAULT_CHECKOUT = Path(__file__).resolve().parents[3] / "surface-code-floorplanner"


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def preserve_runtime_paths(checkout: Path) -> None:
    """The upstream adapter helper drops native rpath flags preceding the archive."""
    link = shlex.split(
        (
            checkout / ".deps/liblsqecc/build/CMakeFiles/lsqecc_slicer.dir/link.txt"
        ).read_text()
    )
    rpaths = [arg for arg in link if arg.startswith("-Wl,-rpath,")]
    if not rpaths:
        return
    metadata_path = checkout / ".deps/adapter/build.json"
    metadata = json.loads(metadata_path.read_text())
    for command in metadata["commands"]:
        if "-c" not in command:
            command[command.index("-o") : command.index("-o")] = rpaths
            run(*command)
    for binary in ("liblsqecc_adapter", "export_layout"):
        path = checkout / ".deps/adapter" / binary
        metadata["sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata["runtime_path_fix"] = (
        "Preserved native CMake rpath flags in adapter/exporter links"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floorplanner-dir", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--prefix", default="", help="Optional CMake dependency prefixes."
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    checkout = args.floorplanner_dir.resolve()
    if not checkout.exists():
        run(
            "git",
            "clone",
            "https://github.com/akataba/surface-code-floorplanner.git",
            str(checkout),
        )
        run("git", "-C", str(checkout), "checkout", FLOORPLANNER_PIN)
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain"], text=True
    ).strip()
    if actual != FLOORPLANNER_PIN or dirty:
        parser.error("Existing floorplanner must be clean and at " + FLOORPLANNER_PIN)
    environment = checkout / ".venv"
    python = environment / "bin" / "python"
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(environment)
    run(
        str(python),
        "-m",
        "pip",
        "install",
        "--cache-dir",
        str(environment / "pip-cache"),
        "-e",
        str(checkout),
    )
    run(
        str(python),
        str(checkout / "scripts/bootstrap.py"),
        "--jobs",
        str(args.jobs),
        "--prefix",
        args.prefix,
    )
    run(
        str(python),
        str(checkout / "scripts/build_liblsqecc_adapter.py"),
        "--source",
        str(checkout / ".deps/liblsqecc"),
        "--build",
        str(checkout / ".deps/liblsqecc/build"),
        "--out",
        str(checkout / ".deps/adapter"),
    )
    preserve_runtime_paths(checkout)
    run(str(checkout / ".deps/adapter/liblsqecc_adapter"), "--help")
    print(f"\nFloorplanner ready: {checkout}")
    print(f"Run {sys.executable} tools/lattice_surgery/run_layout_comparison.py")


if __name__ == "__main__":
    main()
