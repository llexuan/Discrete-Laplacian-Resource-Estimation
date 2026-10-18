#!/usr/bin/env python3
"""Export build_circuit() from a Python file as synthesized Clifford+T QASM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import runpy

from qiskit import QuantumCircuit, qasm2, transpile

from laplacian_qsvt.synthesis.rotation_synthesis import synthesize_ry, synthesize_rz
from run_layout_comparison import normalize_qasm

CT_BASIS = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "cx", "cz"]


def export_circuit(circuit: QuantumCircuit, synthesis_tol: float) -> tuple[str, dict]:
    if not isinstance(circuit, QuantumCircuit):
        raise ValueError("build_circuit() must return a Qiskit QuantumCircuit")
    if not math.isfinite(synthesis_tol) or not 0 < synthesis_tol < 1:
        raise ValueError("--synthesis-tol must be finite and between 0 and 1")
    if circuit.num_clbits or circuit.parameters:
        raise ValueError("Input must have bound parameters and no classical bits")
    if any(
        item.operation.name in ("measure", "reset", "initialize")
        for item in circuit.data
    ):
        raise ValueError(
            "Input must be a unitary circuit without measurements/reset/initialize"
        )
    decomposed = transpile(
        circuit,
        basis_gates=CT_BASIS + ["rz", "ry"],
        optimization_level=0,
        seed_transpiler=0,
    )
    rotations = sum(item.operation.name in ("rz", "ry") for item in decomposed.data)
    epsilon = synthesis_tol / max(1, rotations)
    emitted = QuantumCircuit(circuit.num_qubits)
    emitted.global_phase = decomposed.global_phase
    measured_error = 0.0
    for item in decomposed.data:
        name = item.operation.name
        qubits = [decomposed.find_bit(qubit).index for qubit in item.qubits]
        if name in ("rz", "ry"):
            synthesize = synthesize_rz if name == "rz" else synthesize_ry
            result = synthesize(float(item.operation.params[0]), epsilon)
            emitted.compose(result.circuit, qubits=qubits, inplace=True)
            measured_error += result.projective_error
        elif name != "barrier":
            emitted.append(item.operation, qubits)
    qasm, qubits = normalize_qasm(qasm2.dumps(emitted))
    return qasm, {
        "logical_qubits": qubits,
        "synthesis_tolerance": synthesis_tol,
        "rotation_count": rotations,
        "epsilon_per_rotation": epsilon,
        "measured_synthesis_error_bound": measured_error,
        "gate_counts": dict(emitted.count_ops()),
        "dropped_global_phase_radians": float(emitted.global_phase),
        "scope": "Unitary equivalence up to global phase, with a telescoping rotation-error bound; scheduling semantics are checked separately.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-python",
        type=Path,
        default=Path(__file__).with_name("input_circuit.py"),
    )
    parser.add_argument(
        "--output-qasm",
        type=Path,
        default=Path(__file__).with_name("input_circuit.qasm"),
    )
    parser.add_argument("--synthesis-tol", type=float, default=1e-6)
    args = parser.parse_args()
    namespace = runpy.run_path(str(args.input_python.resolve()))
    builder = namespace.get("build_circuit")
    if not callable(builder):
        parser.error("Input Python file must define build_circuit()")
    qasm, metadata = export_circuit(builder(), args.synthesis_tol)
    path = args.output_qasm.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(qasm)
    metadata["source_python"] = str(args.input_python.resolve())
    path.with_suffix(".synthesis.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"QASM: {path}")
    print(f"Synthesis error bound: {metadata['measured_synthesis_error_bound']:.3e}")


if __name__ == "__main__":
    main()
