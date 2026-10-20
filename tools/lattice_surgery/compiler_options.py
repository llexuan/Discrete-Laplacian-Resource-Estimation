"""Shared presets for LLI generation and physical-resource estimation."""

from pathlib import Path


def output_prefix(preset: str) -> str:
    return {"edpc": "EDPC", "direct": "EDPC", "viewer": "compact"}.get(preset, preset)


def backend_options(
    preset: str,
    disttime: int = 1,
    staggered: bool = False,
    geometry_file: Path | None = None,
) -> list[str]:
    if disttime < 1:
        raise ValueError("--disttime must be positive")
    if geometry_file is not None and preset != "edpc":
        raise ValueError("Custom floorplans require the edpc preprocessing preset")
    if preset == "viewer":
        return ["-L", "compact", "-P", "stream"]
    if preset not in ("direct", "edpc", "compact", "compact_no_clogging"):
        raise ValueError(f"Unknown preset: {preset}")
    options = ["-L", "edpc" if preset == "direct" else preset]
    if geometry_file is not None:
        options += ["--geometry-file", str(geometry_file.resolve())]
    options += ["-P", "wave", "--local", "--disttime", str(disttime)]
    if not staggered:
        options += ["--nostagger"]
    if preset == "direct":
        options += ["--notwists"]
    else:
        options += [
            "--cnotcorrections",
            "never",
            "-r",
            "graph_search",
            "-g",
            "djikstra",
        ]
    return options
