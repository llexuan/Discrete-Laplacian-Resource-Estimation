#!/usr/bin/env python3
"""Compile an exported QASM file inside the isolated lsqecc environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPILER_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compiler"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Python lattice-surgery compiler on OpenQASM 2."
    )
    parser.add_argument(
        "--qasm",
        type=Path,
        default=COMPILER_OUTPUT_DIR / "synthesized_rotation.qasm",
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
        "--no-litinski-transform",
        action="store_true",
        help="Disable the compiler's Litinski transformation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        from lsqecc.pipeline.lattice_surgery_compilation_pipeline import (
            compile_str,
        )
        from lsqecc.simulation.logical_patch_state_simulation import SimulatorType
    except ImportError as exc:
        raise RuntimeError(
            "lsqecc is not installed in this interpreter. Run this script with "
            "`conda run -n lsqecc310 python ...` after setup_compiler.sh."
        ) from exc

    qasm_path = args.qasm.resolve()
    report_path = args.report.resolve()
    summary_path = args.summary.resolve()
    qasm_text = qasm_path.read_text(encoding="utf-8")
    if not qasm_text.lstrip().startswith("OPENQASM 2.0;"):
        raise ValueError(f"{qasm_path} is not an OpenQASM 2 program.")

    slices, compilation_text = compile_str(
        qasm_text,
        apply_litinski_transform=not args.no_litinski_transform,
        simulation_type=SimulatorType.NOOP,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(compilation_text, encoding="utf-8")
    summary = {
        "compiler": "latticesurgery-com/lattice-surgery-compiler",
        "source_qasm": str(qasm_path),
        "apply_litinski_transform": not args.no_litinski_transform,
        "simulation_type": "NOOP",
        "num_lattice_surgery_slices": len(slices),
        "report_path": str(report_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"slices  : {len(slices)}")
    print(f"report  : {report_path}")
    print(f"summary : {summary_path}")


if __name__ == "__main__":
    main()
