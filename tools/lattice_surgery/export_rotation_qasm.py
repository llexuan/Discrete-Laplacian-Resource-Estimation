#!/usr/bin/env python3
"""Export one pyLIQTR-synthesized rotation as OpenQASM 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qiskit import qasm2

from laplacian_qsvt.synthesis.rotation_synthesis import (
    RotationSynthesis,
    synthesize_ry,
    synthesize_rz,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "compiler" / "synthesized_rotation.qasm"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize one rotation in the project environment and export an "
            "OpenQASM 2 input for the isolated lattice-surgery compiler."
        )
    )
    parser.add_argument("--axis", choices=("ry", "rz"), default="rz")
    parser.add_argument("--theta", type=float, default=0.3, help="Angle in radians.")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="pyLIQTR projective operator-norm tolerance.",
    )
    parser.add_argument("--output-qasm", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=None,
        help="Metadata JSON path (default: QASM path with .json suffix).",
    )
    return parser


def synthesize(axis: str, theta: float, epsilon: float) -> RotationSynthesis:
    if axis == "ry":
        return synthesize_ry(theta, epsilon)
    return synthesize_rz(theta, epsilon)


def main() -> None:
    args = build_parser().parse_args()
    result = synthesize(args.axis, args.theta, args.epsilon)
    qasm_path = args.output_qasm.resolve()
    metadata_path = (
        args.output_metadata.resolve()
        if args.output_metadata is not None
        else qasm_path.with_suffix(".json")
    )
    qasm_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # OpenQASM 2 has no global-phase statement. Dropping the circuit's global
    # phase is physically harmless and is recorded explicitly in the metadata.
    qasm_path.write_text(qasm2.dumps(result.circuit), encoding="utf-8")
    metadata = {
        "axis": args.axis,
        "theta_radians": args.theta,
        "requested_epsilon": args.epsilon,
        "projective_error": result.projective_error,
        "backend": "pyLIQTR",
        "backend_version": result.backend_version,
        "gate_counts": result.gate_counts,
        "t_count": result.t_count,
        "clifford_count": result.clifford_count,
        "depth": result.depth,
        "t_depth": result.t_depth,
        "dropped_global_phase_radians": float(result.circuit.global_phase),
        "qasm_path": str(qasm_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"QASM     : {qasm_path}")
    print(f"metadata : {metadata_path}")
    print(f"gates    : {result.gate_counts}")
    print(f"T-count  : {result.t_count}")
    print(f"error    : {result.projective_error:.3e}")


if __name__ == "__main__":
    main()
