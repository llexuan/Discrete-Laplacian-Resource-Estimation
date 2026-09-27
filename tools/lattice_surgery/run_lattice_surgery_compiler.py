#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Generate unsliced and sliced LLI from a hand-written OpenQASM circuit."""

import argparse
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
        choices=("direct", "viewer"),
        default="direct",
        help="Use EDPC/wave direct compilation or compact/stream scheduling.",
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
    parser.add_argument("--qasm", type=Path, default=DEFAULT_QASM)
    parser.add_argument(
        "--report",
        type=Path,
        default=COMPILER_OUTPUT_DIR / "compiler_report.txt",
    )
    parser.add_argument(
        "--lli",
        type=Path,
        default=COMPILER_OUTPUT_DIR / "compiler.lli",
        help="Where to write layout-independent LLI before slicing.",
    )
    parser.add_argument(
        "--sliced-lli",
        type=Path,
        default=COMPILER_OUTPUT_DIR / "compiler_sliced.lli",
        help="Where to write LLI grouped by scheduled timestep.",
    )
    parser.add_argument(
        "--no-litinski-transform",
        action="store_true",
        help="Disable the Litinski transform in the legacy text report.",
    )
    parser.add_argument(
        "--no-intermediate-report",
        action="store_true",
        help="Do not include the legacy Pauli/Litinski report.",
    )
    return parser


def _backend_options(preset: str) -> list[str]:
    if preset == "direct":
        return [
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
    return ["-L", "compact", "-P", "stream"]


def _run_command(command: list[str], output_name: str) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
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


def _legacy_intermediate_report(
    args: argparse.Namespace,
    qasm_path: Path,
) -> str:
    if args.no_intermediate_report:
        return "Legacy Pauli/Litinski report disabled.\n"

    legacy_python = args.legacy_python.resolve()
    if not legacy_python.is_file():
        return f"Legacy compiler Python not found at {legacy_python}.\n"

    with tempfile.TemporaryDirectory(prefix="lsqecc-legacy-report-") as tmp:
        legacy_report = Path(tmp) / "report.txt"
        command = [
            str(legacy_python),
            str(Path(__file__).resolve()),
            "--backend",
            "python",
            "--qasm",
            str(qasm_path),
            "--report",
            str(legacy_report),
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
            return f"Legacy intermediate report failed:\n{details.strip()}\n"
        return legacy_report.read_text(encoding="utf-8")


def _run_fast(
    args: argparse.Namespace,
    qasm_path: Path,
    report_path: Path,
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
    options = _backend_options(args.preset)
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
        _run_command(lli_command, "Unsliced LLI"),
        encoding="utf-8",
    )
    sliced_lli_path.write_text(
        _run_command(sliced_command, "Sliced LLI"),
        encoding="utf-8",
    )

    legacy_report = _legacy_intermediate_report(args, qasm_path)
    report = (
        "=== Pauli rotation and Litinski transformation report ===\n\n"
        f"{legacy_report.rstrip()}\n\n"
        "=== liblsqecc LLI compilation commands ===\n\n"
        f"Unsliced LLI: {shlex.join(lli_command)}\n"
        f"Sliced LLI:   {shlex.join(sliced_command)}\n"
    )
    report_path.write_text(report, encoding="utf-8")

    print("backend      : liblsqecc/lsqecc_slicer")
    print(f"preset       : {args.preset}")
    print(f"unsliced LLI : {lli_path}")
    print(f"sliced LLI   : {sliced_lli_path}")


def main() -> None:
    args = build_parser().parse_args()
    qasm_path = args.qasm.resolve()
    report_path = args.report.resolve()
    lli_path = args.lli.resolve()
    sliced_lli_path = args.sliced_lli.resolve()

    qasm_text = qasm_path.read_text(encoding="utf-8")
    if not qasm_text.lstrip().startswith("OPENQASM 2.0;"):
        raise ValueError(f"{qasm_path} is not an OpenQASM 2 program.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lli_path.parent.mkdir(parents=True, exist_ok=True)
    sliced_lli_path.parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "python":
        _run_python_report(args, qasm_path, report_path)
        print(f"report       : {report_path}")
        return

    _run_fast(
        args,
        qasm_path,
        report_path,
        lli_path,
        sliced_lli_path,
    )
    print(f"report       : {report_path}")


if __name__ == "__main__":
    main()
