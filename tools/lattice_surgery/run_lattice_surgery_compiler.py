#!/usr/bin/env python3
# pyright: reportMissingImports=false
# """Compile OpenQASM with liblsqecc or the legacy Python compiler."""

# """.venv/bin/python \
#   tools/lattice_surgery/run_lattice_surgery_compiler.py \
#   --preset viewer"""

# """.venv/bin/python \
#   tools/lattice_surgery/run_lattice_surgery_compiler.py"""


from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QASM = Path(__file__).resolve().with_name("input_circuit.qasm")
COMPILER_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compiler"
DEFAULT_SLICER = (
    PROJECT_ROOT.parent / "liblsqecc" / "build" / "lsqecc_slicer"
)
DEFAULT_LEGACY_PYTHON = Path(
    os.environ.get(
        "LSQECC_PYTHON",
        "/opt/anaconda3/envs/lsqecc310/bin/python",
    )
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile OpenQASM 2 into lattice-surgery patch slices."
    )
    parser.add_argument(
        "--backend",
        choices=("fast", "python"),
        default="fast",
        help="Use liblsqecc (default) or the legacy Python compiler.",
    )
    parser.add_argument(
        "--preset",
        choices=("direct", "viewer"),
        default="direct",
        help=(
            "Fast-backend settings: upgraded EDPC/wave direct compilation "
            "(default), or compact/stream viewer-style compilation."
        ),
    )
    parser.add_argument(
        "--slicer",
        type=Path,
        default=Path(os.environ.get("LSQECC_SLICER", DEFAULT_SLICER)),
        help="Path to the liblsqecc lsqecc_slicer executable.",
    )
    parser.add_argument(
        "--legacy-python",
        type=Path,
        default=DEFAULT_LEGACY_PYTHON,
        help="Python interpreter containing the legacy lsqecc package.",
    )
    parser.add_argument(
        "--qasm",
        type=Path,
        default=DEFAULT_QASM,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=COMPILER_OUTPUT_DIR / "compiler_report.txt",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=COMPILER_OUTPUT_DIR / "compiler_summary.json",
    )
    parser.add_argument(
        "--slices",
        type=Path,
        default=COMPILER_OUTPUT_DIR / "compiler_slices.json",
        help="Where to write the complete patch layout for every time slice.",
    )
    parser.add_argument(
        "--no-litinski-transform",
        action="store_true",
        help="Disable the Litinski transform for the Python backend only.",
    )
    parser.add_argument(
        "--no-intermediate-report",
        action="store_true",
        help=(
            "Do not prepend the legacy Pauli-rotation/Litinski report when "
            "using the fast backend."
        ),
    )
    return parser


def _count_slices(payload: object) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("slices", "slice_list", "patches"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
    raise ValueError("Could not identify the slice array in compiler output.")


def _git_revision(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _legacy_intermediate_report(
    args: argparse.Namespace, qasm_path: Path
) -> tuple[str, bool]:
    if args.no_intermediate_report:
        return (
            "Legacy Pauli-rotation/Litinski report disabled by command line.\n",
            False,
        )

    legacy_python = args.legacy_python.resolve()
    if not legacy_python.is_file():
        return (
            f"Legacy compiler Python not found at {legacy_python}.\n",
            False,
        )

    with tempfile.TemporaryDirectory(prefix="lsqecc-legacy-report-") as tmp:
        tmp_dir = Path(tmp)
        legacy_report = tmp_dir / "report.txt"
        command = [
            str(legacy_python),
            str(Path(__file__).resolve()),
            "--backend",
            "python",
            "--qasm",
            str(qasm_path),
            "--report",
            str(legacy_report),
            "--summary",
            str(tmp_dir / "summary.json"),
            "--slices",
            str(tmp_dir / "slices.json"),
        ]
        if args.no_litinski_transform:
            command.append("--no-litinski-transform")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not legacy_report.is_file():
            details = result.stderr or result.stdout or "unknown error"
            return (
                "Legacy intermediate report generation failed:\n"
                f"{details.strip()}\n",
                False,
            )
        return legacy_report.read_text(encoding="utf-8"), True


def _run_fast(
    args: argparse.Namespace,
    qasm_path: Path,
    report_path: Path,
    summary_path: Path,
    slices_path: Path,
) -> None:
    slicer = args.slicer.resolve()
    if not slicer.is_file():
        raise FileNotFoundError(
            f"Fast slicer not found at {slicer}. Run setup_compiler.sh."
        )

    command = [
        str(slicer),
        "-I",
        "qasm",
        "-i",
        str(qasm_path),
        "-o",
        str(slices_path),
        "-f",
        "stats",
        "--graceful",
    ]
    if args.preset == "direct":
        command.extend(
            [
                "-L",
                "edpc",
                "--disttime",
                "1",
                "--nostagger",
                "--notwists",
                "--local",
                "-P",
                "wave",
            ]
        )
    else:
        command.extend(["-L", "compact", "-P", "stream"])

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    fast_report = (
        f"command: {shlex.join(command)}\n"
        f"exit_code: {result.returncode}\n\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\n"
    )
    if result.returncode != 0:
        report_path.write_text(fast_report, encoding="utf-8")
        raise RuntimeError(
            f"lsqecc_slicer failed; inspect {report_path} for details."
        )

    intermediate_report, intermediate_included = (
        _legacy_intermediate_report(args, qasm_path)
    )
    report_text = (
        "=== Pauli rotation and Litinski transformation report ===\n\n"
        f"{intermediate_report.rstrip()}\n\n"
        "=== Fast liblsqecc direct slicing report ===\n\n"
        f"{fast_report}"
    )
    report_path.write_text(report_text.rstrip() + "\n", encoding="utf-8")

    slice_payload = json.loads(slices_path.read_text(encoding="utf-8"))
    num_slices = _count_slices(slice_payload)
    repo = slicer.parent.parent
    summary = {
        "compiler": "latticesurgery-com/liblsqecc",
        "backend": "lsqecc_slicer",
        "backend_revision": _git_revision(repo),
        "preset": args.preset,
        "source_qasm": str(qasm_path),
        "num_lattice_surgery_slices": num_slices,
        "command": command,
        "intermediate_report_included": intermediate_included,
        "legacy_python": str(args.legacy_python.resolve()),
        "report_path": str(report_path),
        "slices_path": str(slices_path),
        "viewer_compatible_json": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("backend : liblsqecc/lsqecc_slicer")
    print(f"preset  : {args.preset}")
    print(f"slices  : {num_slices}")


def _run_python(
    args: argparse.Namespace,
    qasm_path: Path,
    report_path: Path,
    summary_path: Path,
    slices_path: Path,
) -> None:
    try:
        from lsqecc.pipeline.lattice_surgery_compilation_pipeline import (
            compile_str,
        )
        from lsqecc.pipeline.json_pipeline import _SliceArrayJSONEncoder
        from lsqecc.resource_estimation import llops_resource_estimator
        from lsqecc.simulation.logical_patch_state_simulation import SimulatorType
    except ImportError as exc:
        raise RuntimeError(
            "lsqecc is not installed in this interpreter. Run this script with "
            "`conda run -n lsqecc310 python ...` after setup_compiler.sh."
        ) from exc

    qasm_text = qasm_path.read_text(encoding="utf-8")
    original_estimate = llops_resource_estimator.estimate
    llops_resource_estimator.estimate = lambda _computation: None
    try:
        slices, compilation_text = compile_str(
            qasm_text,
            apply_litinski_transform=not args.no_litinski_transform,
            simulation_type=SimulatorType.NOOP,
        )
    finally:
        llops_resource_estimator.estimate = original_estimate

    report_path.write_text(compilation_text, encoding="utf-8")
    slices_path.write_text(
        _SliceArrayJSONEncoder().encode(slices) + "\n",
        encoding="utf-8",
    )
    summary = {
        "compiler": "latticesurgery-com/lattice-surgery-compiler",
        "backend": "python_compile_str",
        "source_qasm": str(qasm_path),
        "apply_litinski_transform": not args.no_litinski_transform,
        "simulation_type": "NOOP",
        "upstream_resource_estimator_disabled": True,
        "num_lattice_surgery_slices": len(slices),
        "report_path": str(report_path),
        "slices_path": str(slices_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("backend : lattice-surgery-compiler/compile_str")
    print(f"slices  : {len(slices)}")


def main() -> None:
    args = build_parser().parse_args()
    qasm_path = args.qasm.resolve()
    report_path = args.report.resolve()
    summary_path = args.summary.resolve()
    slices_path = args.slices.resolve()
    qasm_text = qasm_path.read_text(encoding="utf-8")
    if not qasm_text.lstrip().startswith("OPENQASM 2.0;"):
        raise ValueError(f"{qasm_path} is not an OpenQASM 2 program.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    slices_path.parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "fast":
        _run_fast(args, qasm_path, report_path, summary_path, slices_path)
    else:
        _run_python(args, qasm_path, report_path, summary_path, slices_path)

    print(f"report  : {report_path}")
    print(f"layouts : {slices_path}")
    print(f"summary : {summary_path}")


if __name__ == "__main__":
    main()
