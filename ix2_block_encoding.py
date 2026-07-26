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
import math
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


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
        type=Path,
        default=Path("fig6_fig7_ix"),
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

    w_path = Path(f"{args.save_prefix}_W.txt")
    phi_path = Path(f"{args.save_prefix}_Phi_j.txt")
    w_path.write_text(str(w.draw(output="text")) + "\n", encoding="utf-8")
    phi_path.write_text(str(fig6_block.draw(output="text")) + "\n", encoding="utf-8")
    print(f"\nSaved: {w_path} and {phi_path}")


if __name__ == "__main__":
    main()
