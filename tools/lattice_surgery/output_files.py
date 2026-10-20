"""Shared locations for readable reports and supporting machine artifacts."""

from pathlib import Path

from compiler_options import output_prefix


def resource_paths(output: Path, method: str) -> dict[str, Path]:
    prefix = output_prefix(method)
    return {
        "lli": output / "artifacts" / f"{prefix}.lli",
        "sliced": output / "artifacts" / f"{prefix}_sliced.lli",
        "json": output / "data" / f"{prefix}_resource_estimate.json",
        "report": output / f"{prefix}_resource_report.txt",
    }


def prepare_output(output: Path) -> None:
    for folder in (output, output / "data", output / "artifacts"):
        folder.mkdir(parents=True, exist_ok=True)
