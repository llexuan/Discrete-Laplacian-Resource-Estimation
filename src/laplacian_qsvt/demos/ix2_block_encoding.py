#!/usr/bin/env python3
"""
Implement the paper's Fig. 6 / Fig. 7 circuit structure in Qiskit.

This version uses a block-encoding for H = (I + X) / 2.

Registers (1 qubit each):
  q: qubitization control qubit 
  a: block-encoding ancilla qubit 
  s: system qubit 
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from ..paths import CIRCUITS_DIR, ensure_parent


def load_phase_metadata(
    path: str | Path,
) -> tuple[str, int, int, int, list[float], float, np.ndarray | None]:
    """Load and validate QSP phase data written by the phase-synthesis scripts."""
    phase_path = Path(path)
    payload = json.loads(phase_path.read_text(encoding="utf-8"))

    if "phases" not in payload:
        raise ValueError(f"{phase_path} has no 'phases' array.")
    phases_array = np.asarray(payload["phases"], dtype=float)
    if phases_array.ndim != 1 or phases_array.size == 0:
        raise ValueError(f"{phase_path} must contain a non-empty 1D 'phases' array.")
    if not np.all(np.isfinite(phases_array)):
        raise ValueError(f"{phase_path} contains non-finite QSP phases.")

    effective_degree = int(
        payload.get("effective_degree", phases_array.size - 1)
    )
    degree = int(payload.get("degree", effective_degree))
    parity = int(payload.get("parity", effective_degree % 2))
    target_function = str(payload.get("target_function", "unknown"))
    max_scale = float(
        payload.get("max_scale", payload.get("returned_scale", 1.0))
    )

    raw_coeffs = payload.get("cheb_coeffs")
    cheb_coeffs = None
    if raw_coeffs is not None:
        cheb_coeffs = np.asarray(raw_coeffs, dtype=float)
        if cheb_coeffs.ndim != 1 or cheb_coeffs.size == 0:
            raise ValueError(
                f"{phase_path} must contain a non-empty 1D 'cheb_coeffs' array."
            )
        if not np.all(np.isfinite(cheb_coeffs)):
            raise ValueError(
                f"{phase_path} contains non-finite Chebyshev coefficients."
            )

    return (
        target_function,
        degree,
        effective_degree,
        parity,
        phases_array.tolist(),
        max_scale,
        cheb_coeffs,
    )


def chebyshev_matrix_eval(
    coefficients: list[float] | np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """Evaluate sum_k coefficients[k] T_k(matrix) using Clenshaw recurrence."""
    coeffs = np.asarray(coefficients)
    mat = np.asarray(matrix)
    if coeffs.ndim != 1 or coeffs.size == 0:
        raise ValueError("Chebyshev coefficients must be a non-empty 1D array.")
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Chebyshev matrix evaluation requires a square matrix.")

    dtype = np.result_type(coeffs.dtype, mat.dtype, np.float64)
    coeffs = coeffs.astype(dtype, copy=False)
    mat = mat.astype(dtype, copy=False)
    identity = np.eye(mat.shape[0], dtype=dtype)

    if coeffs.size == 1:
        return coeffs[0] * identity

    b_k_plus_1 = np.zeros_like(mat, dtype=dtype)
    b_k_plus_2 = np.zeros_like(mat, dtype=dtype)
    for coefficient in coeffs[:0:-1]:
        b_k = (
            2.0 * (mat @ b_k_plus_1)
            - b_k_plus_2
            + coefficient * identity
        )
        b_k_plus_2 = b_k_plus_1
        b_k_plus_1 = b_k

    return mat @ b_k_plus_1 - b_k_plus_2 + coeffs[0] * identity


def build_u_h_for_i_plus_x_over_2() -> QuantumCircuit:
    """
    Build a 2-qubit block-encoding unitary U_H on (a, s) such that:
      <0_a| U_H |0_a> = (I + X) / 2.
    """
    qc = QuantumCircuit(2, name="U_H")
    # PREP
    qc.h(0)  # ancilla a
    # SELECT: I when a=0, X when a=1
    qc.cx(0, 1)  # a controls X on s
    # PREP^dag
    qc.h(0)
    return qc


def build_u_h_prime(u_h: QuantumCircuit) -> QuantumCircuit:
    """
    Build U_H' from Eq. (32):
      U_H' = |0><0|_q ⊗ U_H + |1><1|_q ⊗ U_H^dag.

    Qubit ordering in this function:
      [q, a, s]
    """
    qc = QuantumCircuit(3, name="U_H'")
    u_h_gate = u_h.to_gate(label="U_H")
    u_h_dag_gate = u_h.inverse().to_gate(label="U_H†")

    # Apply U_H on branch q=0 (X sandwich turns |0>-control into |1>-control).
    qc.x(0)
    qc.append(u_h_gate.control(1), [0, 1, 2])
    qc.x(0)

    # Apply U_H^dag on branch q=1.
    qc.append(u_h_dag_gate.control(1), [0, 1, 2])
    return qc


def apply_u_r(circ: QuantumCircuit, q: int, a: int) -> None:
    """
    Apply reflection U_R = 2 |+_q,0_a><+_q,0_a| - I (up to global phase).

    Implemented by:
      1) map |+_q,0_a> -> |0,0> via H on q,
      2) reflect about |0,0>,
      3) map back via H on q.
    """
    circ.h(q)
    circ.x(q)
    circ.x(a)
    circ.cz(q, a)
    circ.x(q)
    circ.x(a)
    circ.h(q)


def build_w_operator(u_h: QuantumCircuit) -> QuantumCircuit:
    """
    Build W :
      W = (U_R ⊗ I_s) S U_H'
    with S = X_q ⊗ I_{a,s}.

    Circuit order follows right-to-left operator application:
      first U_H', then S, then U_R.
    """
    qc = QuantumCircuit(3, name="W")
    qc.compose(build_u_h_prime(u_h), [0, 1, 2], inplace=True)
    qc.x(0)  # S = X_q
    apply_u_r(qc, q=0, a=1)
    return qc


def apply_phi_block(
    circ: QuantumCircuit,
    w_gate,
    phi_2j: float,
    phi_2j_plus_1: float,
    q: int = 0,
    a: int = 1,
    s: int = 2,
) -> None:
    """
    Build one Fig. 6 block Phi_j(phi_{2j+1}, phi_{2j}) on [q,a,s].

    This follows the structure shown in Fig. 6:
      Rz(-phi_{2j}) H Rz(-pi/2) H Rz(phi_{2j}) W
      Rz(-phi_{2j+1}) H Rz(pi/2) H Rz(phi_{2j+1}) W^dag
    """
    circ.rz(-phi_2j, q)
    circ.h(q)
    circ.rz(-math.pi / 2, q)
    circ.h(q)
    circ.rz(phi_2j, q)
    circ.append(w_gate, [q, a, s])

    circ.rz(-phi_2j_plus_1, q)
    circ.h(q)
    circ.rz(math.pi / 2, q)
    circ.h(q)
    circ.rz(phi_2j_plus_1, q)
    circ.append(w_gate.inverse(), [q, a, s])


def extract_block_from_u_h(u_h: QuantumCircuit) -> np.ndarray:
    """
    Return the encoded operator <0_a|U_H|0_a> on system s.

    For two qubits ordered [a, s], ancilla is qubit 0 (LSB),
    so basis indices with a=0 are [0, 2].
    """
    U = Operator(u_h).data
    idx = [0, 2]
    return U[np.ix_(idx, idx)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Fig. 6 / Fig. 7 style QSP circuits with H=(I+X)/2 block-encoding."
    )
    parser.add_argument("--phi-even", type=float, default=-0.15354338744702095)
    parser.add_argument("--phi-odd", type=float, default=0.8126827401045922)
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="fig6_fig7_ix",
        help="Prefix for saved ASCII circuit files.",
    )
    args = parser.parse_args()

    u_h = build_u_h_for_i_plus_x_over_2()
    w = build_w_operator(u_h)

    fig6_block = QuantumCircuit(3, name="Phi_j")
    apply_phi_block(fig6_block, w.to_gate(label="W"), args.phi_even, args.phi_odd)

    encoded = extract_block_from_u_h(u_h)
    target = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)


    print("=== Block-encoding check for U_H ===")
    print("Target (I+X)/2:")
    print(target)
    print("<0_a|U_H|0_a>:")
    print(encoded)
    print("||difference||_max =", np.max(np.abs(encoded - target)))

    print("\n=== Fig. 7-like W circuit ===")
    print(w.draw(output="text"))

    print("\n=== Fig. 6-like Phi_j block ===")
    print(fig6_block.draw(output="text"))

    w_path = CIRCUITS_DIR / f"{args.save_prefix}_W.txt"
    phi_path = CIRCUITS_DIR / f"{args.save_prefix}_Phi_j.txt"
    ensure_parent(w_path).write_text(
        str(w.draw(output="text")) + "\n", encoding="utf-8"
    )
    ensure_parent(phi_path).write_text(
        str(fig6_block.draw(output="text")) + "\n", encoding="utf-8"
    )
    print(f"\nSaved: {w_path} and {phi_path}")


if __name__ == "__main__":
    main()
