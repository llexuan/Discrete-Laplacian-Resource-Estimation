#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Generate unsliced and sliced LLI from a hand-written OpenQASM circuit."""

import argparse
import os
from pathlib import Path
import subprocess

from compiler_options import backend_options, output_prefix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QASM = Path(__file__).resolve().with_name("input_circuit.qasm")
COMPILER_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compiler"
DEFAULT_SLICER = PROJECT_ROOT.parent / "liblsqecc" / "build" / "lsqecc_slicer"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile OpenQASM 2 into layout-independent and time-sliced "
            "lattice-surgery instructions."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("fast", "python"),
        default="fast",
        help="Use liblsqecc (default) or emit only the legacy text report.",
    )
    parser.add_argument(
        "--preset",
        choices=("direct", "viewer", "edpc", "compact", "compact_no_clogging"),
        default="direct",
        help=(
            "Historical direct/viewer configurations, or matched edpc/compact "
            "configurations for layout comparisons."
        ),
    )
    parser.add_argument(
        "--slicer",
        type=Path,
        default=Path(os.environ.get("LSQECC_SLICER", DEFAULT_SLICER)),
        help="Path to the liblsqecc lsqecc_slicer executable.",
    )
    parser.add_argument("--qasm", type=Path, default=DEFAULT_QASM)
    parser.add_argument(
        "--geometry-file",
        type=Path,
        help="Adapter geometry; pass its matching mapped QASM.",
    )
    parser.add_argument("--disttime", type=int, default=1)
    parser.add_argument("--staggered", action="store_true")
    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="Wall-clock seconds per compiler call.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=COMPILER_OUTPUT_DIR / "legacy_report.txt",
        help="Text output for --backend python only; fast layouts generate LLI only.",
    )
    parser.add_argument(
        "--lli",
        type=Path,
        help="Where to write layout-independent LLI (default: <layout>.lli).",
    )
    parser.add_argument(
        "--sliced-lli",
        type=Path,
        help="Where to write scheduled LLI (default: <layout>_sliced.lli).",
    )
    parser.add_argument(
        "--no-litinski-transform",
        action="store_true",
        help="Disable the Litinski transform in the legacy text report.",
    )
    return parser


def _backend_options(preset: str) -> list[str]:
    return backend_options(preset)


def _run_command(command: list[str], output_name: str, timeout: float = 30) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0 or result.stderr.strip():
        details = result.stderr or result.stdout or "unknown error"
        raise RuntimeError(
            f"{output_name} generation failed with exit code "
            f"{result.returncode}:\n{details.strip()}"
        )
    return result.stdout.rstrip() + "\n"


def _run_python_report(
    args: argparse.Namespace,
    qasm_path: Path,
    report_path: Path,
) -> None:
    try:
        from lsqecc.pipeline.lattice_surgery_compilation_pipeline import (
            compile_str,
        )
        from lsqecc.resource_estimation import llops_resource_estimator
        from lsqecc.simulation.logical_patch_state_simulation import SimulatorType
    except ImportError as exc:
        raise RuntimeError(
            "The legacy lsqecc package is not installed in this interpreter."
        ) from exc

    original_estimate = llops_resource_estimator.estimate
    llops_resource_estimator.estimate = lambda _computation: None
    try:
        _slices, compilation_text = compile_str(
            qasm_path.read_text(encoding="utf-8"),
            apply_litinski_transform=not args.no_litinski_transform,
            simulation_type=SimulatorType.NOOP,
        )
    finally:
        llops_resource_estimator.estimate = original_estimate
    report_path.write_text(compilation_text.rstrip() + "\n", encoding="utf-8")


def _run_fast(
    args: argparse.Namespace,
    qasm_path: Path,
    lli_path: Path,
    sliced_lli_path: Path,
) -> None:
    slicer = args.slicer.resolve()
    if not slicer.is_file():
        raise FileNotFoundError(
            f"Fast slicer not found at {slicer}. Run setup_compiler.sh."
        )

    prefix = [
        str(slicer),
        "-I",
        "qasm",
        "-i",
        str(qasm_path),
    ]
    options = backend_options(
        args.preset, args.disttime, args.staggered, args.geometry_file
    )
    lli_command = [
        *prefix,
        "--printlli",
        "before",
        "--graceful",
        *options,
    ]
    sliced_command = [
        *prefix,
        "--printlli",
        "sliced",
        "--graceful",
        *options,
    ]

    lli_path.write_text(
        _run_command(lli_command, "Unsliced LLI", args.timeout),
        encoding="utf-8",
    )
    sliced_lli_path.write_text(
        _run_command(sliced_command, "Sliced LLI", args.timeout),
        encoding="utf-8",
    )

    print("backend      : liblsqecc/lsqecc_slicer")
    print(f"preset       : {args.preset}")
    print(f"unsliced LLI : {lli_path}")
    print(f"sliced LLI   : {sliced_lli_path}")


def main() -> None:
    args = build_parser().parse_args()
    qasm_path = args.qasm.resolve()
    prefix = output_prefix(args.preset)
    lli_path = (args.lli or COMPILER_OUTPUT_DIR / f"{prefix}.lli").resolve()
    sliced_lli_path = (
        args.sliced_lli or COMPILER_OUTPUT_DIR / f"{prefix}_sliced.lli"
    ).resolve()

    qasm_text = qasm_path.read_text(encoding="utf-8")
    if not qasm_text.lstrip().startswith("OPENQASM 2.0;"):
        raise ValueError(f"{qasm_path} is not an OpenQASM 2 program.")

    lli_path.parent.mkdir(parents=True, exist_ok=True)
    sliced_lli_path.parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "python":
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _run_python_report(args, qasm_path, report_path)
        print(f"report       : {report_path}")
        return

    _run_fast(
        args,
        qasm_path,
        lli_path,
        sliced_lli_path,
    )


if __name__ == "__main__":
    main()
