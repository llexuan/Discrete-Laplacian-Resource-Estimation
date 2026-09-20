#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Compile an exported QASM file inside the isolated lsqecc environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QASM = Path(__file__).resolve().with_name("input_circuit.qasm")
COMPILER_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compiler"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Python lattice-surgery compiler on OpenQASM 2."
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
        help="Disable the compiler's Litinski transformation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
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

    qasm_path = args.qasm.resolve()
    report_path = args.report.resolve()
    summary_path = args.summary.resolve()
    slices_path = args.slices.resolve()
    qasm_text = qasm_path.read_text(encoding="utf-8")
    if not qasm_text.lstrip().startswith("OPENQASM 2.0;"):
        raise ValueError(f"{qasm_path} is not an OpenQASM 2 program.")

    # The pinned compiler passes LatticeSurgeryComputation to an estimator that
    # expects LogicalLatticeComputation, causing AttributeError after slice
    # generation. Disable only that optional estimate for this smoke-test
    # runner; compilation and slice generation are unchanged.
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

    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    slices_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(compilation_text, encoding="utf-8")
    slices_path.write_text(
        _SliceArrayJSONEncoder().encode(slices) + "\n",
        encoding="utf-8",
    )
    summary = {
        "compiler": "latticesurgery-com/lattice-surgery-compiler",
        "source_qasm": str(qasm_path),
        "apply_litinski_transform": not args.no_litinski_transform,
        "simulation_type": "NOOP",
        "upstream_resource_estimator_disabled": True,
        "num_lattice_surgery_slices": len(slices),
        "report_path": str(report_path),
        "slices_path": str(slices_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"slices  : {len(slices)}")
    print(f"report  : {report_path}")
    print(f"layouts : {slices_path}")
    print(f"summary : {summary_path}")


if __name__ == "__main__":
    main()
