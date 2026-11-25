"""Explicit Clifford+T compilation with a whole-sequence rotation budget.

Controls are decomposed before approximating rotations: no approximation's
global phase is silently promoted to a relative phase by a later control.
"""

from __future__ import annotations

from collections import Counter
import math

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.passes.synthesis import HLSConfig

from .rotation_synthesis import synthesize_rz

CT_BASIS = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "cx", "cz"]
ROTATION_BASIS = CT_BASIS + ["rz"]


def expand_to_rotations(circuit: QuantumCircuit) -> QuantumCircuit:
    """Keep resident qubits dirty and borrow them only with exact uncomputation."""
    return transpile(
        circuit,
        basis_gates=ROTATION_BASIS,
        optimization_level=0,
        seed_transpiler=0,
        hls_config=HLSConfig(mcx=["n_dirty_i15", "noaux_v24"]),
        qubits_initially_zero=False,
    )


def exact_eighth_turn(theta: float) -> int | None:
    """Recognize only machine-precision multiples of pi/4, not small angles."""
    turns = round(theta / (math.pi / 4))
    # Keep the tiny recognition error in the separately reported roundoff sum.
    if abs(theta - turns * math.pi / 4) <= 8 * math.ulp(max(abs(theta), 1.0)):
        return turns
    return None


def arbitrary_rotation_count(circuit: QuantumCircuit) -> int:
    return sum(
        instruction.operation.name == "rz"
        and exact_eighth_turn(float(instruction.operation.params[0])) is None
        for instruction in circuit.data
    )


def compile_rotations(circuit: QuantumCircuit, epsilon: float) -> tuple[QuantumCircuit, dict]:
    """Replace every arbitrary Rz; preserve the entire unitary, including phase."""
    if not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must be finite and strictly between zero and one")
    out = QuantumCircuit(circuit.num_qubits, name=circuit.name + "_CT")
    out.global_phase = circuit.global_phase
    errors = 0.0
    roundoff = 0.0
    rotations = Counter()
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        if operation.name == "rz":
            theta = float(operation.params[0])
            turns = exact_eighth_turn(theta)
            if turns is not None:
                roundoff += abs(theta - turns * math.pi / 4) / 2
                out.global_phase -= turns * math.pi / 8
                # diag(1, exp(i*turns*pi/4)) with at most two gates.
                exponent = turns % 8
                if exponent >= 4:
                    out.z(qubits[0])
                    exponent -= 4
                if exponent >= 2:
                    out.s(qubits[0])
                    exponent -= 2
                if exponent:
                    out.t(qubits[0])
            else:
                # Use the exact inverse of the same synthesized positive-angle
                # circuit for -theta, preserving conjugate/uncompute pairs.
                result = synthesize_rz(abs(theta), epsilon)
                approximation = result.circuit if theta >= 0 else result.circuit.inverse()
                out.compose(approximation, qubits=qubits, inplace=True)
                errors += result.projective_error
                rotations[repr(theta)] += 1
        elif operation.name in CT_BASIS:
            out.append(operation, qubits)
        elif operation.name != "barrier":
            raise ValueError(f"Unexpected operation after rotation expansion: {operation.name}")
    return out, {
        "arbitrary_rotations": sum(rotations.values()),
        "distinct_rz_angles": len(rotations),
        "epsilon_per_rotation": epsilon,
        "summed_operator_error_bound": errors + roundoff,
        "exact_angle_roundoff_bound": roundoff,
        "global_phase_preserved": True,
        "mcx_workspace": "borrowed resident qubits, returned exactly; no clean-state assumption",
    }


def circuit_resources(circuit: QuantumCircuit) -> dict:
    counts = {str(gate): int(count) for gate, count in circuit.count_ops().items()}
    t_count = counts.get("t", 0) + counts.get("tdg", 0)
    return {
        "logical_qubits": circuit.num_qubits,
        "gate_counts": counts,
        "clifford_gate_count": sum(counts.values()) - t_count,
        "t_count": t_count,
        "total_gates": sum(counts.values()),
        "total_depth": int(circuit.depth()),
        "t_depth": int(circuit.depth(filter_function=lambda inst: inst.operation.name in ("t", "tdg"))),
    }
