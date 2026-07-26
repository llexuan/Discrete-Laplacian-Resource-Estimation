#!/usr/bin/env python3
"""
QSVT matrix inversion of the 2D periodic discrete Laplacian.

The dimension is fixed to D = 2 (K = ceil(log2 2) = 1 selector qubit,
m = 2 + K = 3 block-encoding ancillas). The user only chooses the number of
grid/data qubits per dimension via -d.

Registers (Qiskit little-endian, qubit 0 = least significant):
  q            : qubitization control qubit                (index 0)
  l0, l1       : block-encoding ancillas                   (indices 1, 2)
  k0           : dimension selector (K = 1)                 (index 3)
  system       : 2 grid sub-registers j^(0), j^(1)          (indices 4 ..)
                 each of n qubits (grid size per dim N = 2**n)

Usage:
  python inversion_phase_rotations.py --epsilon 0.49 --kappa 2.5
  python laplacian_inversion_encode.py -d=1          # 2D, N=2 per dim
  python laplacian_inversion_encode.py -d=2          # 2D, N=4 per dim
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ["MPLCONFIGDIR"] = str(Path(".mplconfig").resolve())
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

from ix2_block_encoding import chebyshev_matrix_eval, load_phase_metadata

# Single-qubit reference kets.
_KET0 = np.array([1.0, 0.0], dtype=complex)
_KET_PLUS = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
DEFAULT_PHASES_JSON = Path("phases_inversion.json")


def num_k_qubits(dims: int) -> int:
    """Selector-register size K = ceil(log2 D) (0 for D = 1)."""
    if dims <= 1:
        return 0
    return int(math.ceil(math.log2(dims)))


def signal_state(m: int) -> np.ndarray:
    """
    Qubitization "signal" state |Pi> = |+>_q ⊗ |0>^{⊗ m} 
    """
    vec = _KET_PLUS  # q is the least-significant factor
    for _ in range(m):
        vec = np.kron(_KET0, vec)  # prepend one |0> ancilla (more significant)
    return vec


def build_projector_rotation(phi: float, m: int) -> UnitaryGate:
    """
    Projector-controlled phase rotation on (q + m ancilla):
        R(phi) = exp( i * phi * (2 |Pi><Pi| - I) ),  |Pi> = |+>_q |0>^{⊗ m}.
    """
    dim = 2 ** (1 + m)
    sig = signal_state(m)
    pi_proj = np.outer(sig, sig.conj())
    mat = expm(1j * phi * (2.0 * pi_proj - np.eye(dim, dtype=complex)))
    return UnitaryGate(mat, label="R(phi)")


def build_reflection(m: int) -> UnitaryGate:
    """Reflection U_R = I - 2 |Pi><Pi| on (q + m ancilla)."""
    dim = 2 ** (1 + m)
    sig = signal_state(m)
    pi_proj = np.outer(sig, sig.conj())
    mat = np.eye(dim, dtype=complex) - 2.0 * pi_proj
    return UnitaryGate(mat, label="U_R")


def shift_matrices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Cyclic shift permutation matrices on N = 2**n grid points (s0 = LSB):
      S^+ |j> = |(j + 1) mod N>,   S^- |j> = |(j - 1) mod N>.
    """
    N = 2**n
    s_plus = np.zeros((N, N), dtype=complex)
    for j in range(N):
        s_plus[(j + 1) % N, j] = 1.0
    s_minus = s_plus.T.conj().copy()  # decrement = inverse of increment
    return s_plus, s_minus


def scaled_1d_laplacian(n: int) -> np.ndarray:
    """Scaled 1D periodic Laplacian L~_p^(1) = (1/4)(S^+ + S^- - 2 I)."""
    s_plus, s_minus = shift_matrices(n)
    N = 2**n
    return 0.25 * (s_plus + s_minus - 2.0 * np.eye(N, dtype=complex))


def scaled_periodic_laplacian(n: int, dims: int) -> np.ndarray:
    """
    Scaled D-dimensional periodic Laplacian (equal grids, omega_d = 1/D):

        L~_p^(D) = (1/D) sum_d ( I ⊗ ... ⊗ L~_p^(1) ⊗ ... ⊗ I ),
    """
    l1d = scaled_1d_laplacian(n)
    N = 2**n
    ident = np.eye(N, dtype=complex)
    omega = 1.0 / dims
    total = np.zeros((N**dims, N**dims), dtype=complex)
    for d in range(dims):
        mats = [l1d if dd == d else ident for dd in range(dims - 1, -1, -1)]
        term = mats[0]
        for mat in mats[1:]:
            term = np.kron(term, mat)
        total = total + omega * term
    return total


def build_prep_k() -> UnitaryGate:
    """
    Selector-register state preparation for the fixed 2D case.
        U_prep_k |0> = (|0> + |1>) / sqrt(2) = |+>,

    """
    u_prep = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    return UnitaryGate(u_prep, label="U_prep_k")


def build_u_laplacian_periodic(n: int, dims: int) -> QuantumCircuit:
    """
    Fig. 1 (D=1) / (D>1) block encoding of L~_p^(D).
    """
    K = num_k_qubits(dims)
    m = 2 + K
    total = m + dims * n
    qc = QuantumCircuit(total, name="U_L")

    l0, l1 = 0, 1
    kq = list(range(2, 2 + K))

    def sysreg(d: int) -> list[int]:
        base = 2 + K + d * n
        return list(range(base, base + n))

    s_plus, s_minus = shift_matrices(n)
    s_plus_gate = UnitaryGate(s_plus, label="S+")
    s_minus_gate = UnitaryGate(s_minus, label="S-")

    # PREP selector register (D > 1 only).
    prep = build_prep_k() if K > 0 else None
    if prep is not None:
        qc.append(prep, kq)

    # PREP block-encoding ancillas: H then Z -> |->_{l0} |->_{l1}.
    qc.h(l0)
    qc.h(l1)
    qc.z(l0)
    qc.z(l1)

    # SELECT: for each dimension d, apply the 1D shifts on j^(d), also
    # controlled on k = d. S^- fires when l1 = 0 (and k = d); S^+ when l0 = 1
    n_ctrl = 1 + K
    for d in range(dims):
        sq = sysreg(d)
        qc.append(s_minus_gate.control(n_ctrl, ctrl_state=(d << 1)), [l1] + kq + sq)
        qc.append(s_plus_gate.control(n_ctrl, ctrl_state=(d << 1) | 1), [l0] + kq + sq)

    # UNPREP block-encoding ancillas.
    qc.h(l0)
    qc.h(l1)

    # UNPREP selector register.
    if prep is not None:
        qc.append(prep.inverse(), kq)
    return qc


def build_u_prime(u_l: QuantumCircuit, m: int, n_sys: int) -> QuantumCircuit:
    """
    U' = |0><0|_q ⊗ U_L + |1><1|_q ⊗ U_L^dag on (q + m ancilla + system).
    """
    total = 1 + m + n_sys
    qc = QuantumCircuit(total, name="U'")
    u_l_gate = u_l.to_gate(label="U_L")
    u_l_dag_gate = u_l.inverse().to_gate(label="U_L†")
    targets = list(range(1, total))  # ancilla + system

    # U_L on the q = 0 branch (X sandwich turns control-on-1 into control-on-0).
    qc.x(0)
    qc.append(u_l_gate.control(1), [0] + targets)
    qc.x(0)
    # U_L^dag on the q = 1 branch.
    qc.append(u_l_dag_gate.control(1), [0] + targets)
    return qc


def build_w_operator(u_l: QuantumCircuit, m: int, n_sys: int) -> QuantumCircuit:
    """
    Qubitization walk W = U_R S U', with S = X_q and U_R = I - 2|Pi><Pi|.
    """
    total = 1 + m + n_sys
    qc = QuantumCircuit(total, name="W")
    qc.compose(build_u_prime(u_l, m, n_sys), range(total), inplace=True)
    qc.x(0)  # S = X_q
    qc.append(build_reflection(m), list(range(0, 1 + m)))
    return qc


def build_qsp_sequence(
    w_gate, phases: list[float], m: int, n_sys: int
) -> QuantumCircuit:
    """
    U = R(phi_0) W R(phi_1) W ... W R(phi_d).
    """
    total = 1 + m + n_sys
    circ = QuantumCircuit(total, name="U_qsp")
    proj_qubits = list(range(0, 1 + m))
    num = len(phases)
    for k, phi in enumerate(phases):
        circ.append(build_projector_rotation(phi, m), proj_qubits)
        if k < num - 1:
            circ.append(w_gate, range(total))
    return circ


def extract_block(full_qsp: QuantumCircuit, m: int, n_sys: int) -> np.ndarray:
    """
    Projected QSP block on the system register:

        (<+_q, 0^m| ⊗ I_sys) U_qsp (|+_q, 0^m> ⊗ I_sys).
    """
    U = Operator(full_qsp).data
    sig = signal_state(m)[:, None]  # (2**(1+m), 1)
    i_sys = np.eye(2**n_sys, dtype=complex)
    right = np.kron(i_sys, sig)
    left = np.kron(i_sys, sig.conj().T)
    return left @ U @ right


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QSVT matrix inversion of the 2D periodic Laplacian via a "
        "Fig.-4 block encoding and P^{MI} phases."
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="System/grid qubits per dimension (grid size per dim N = 2**d).",
    )
    parser.add_argument(
        "--save-prefix",
        type=Path,
        default=Path("lap_inv"),
        help="Prefix for saved ASCII circuit files.",
    )
    parser.add_argument(
        "--output-block-metrics",
        type=Path,
        default=Path("block_metrics_laplacian.json"),
        help="Where to write block-encoding error metrics as JSON.",
    )
    args = parser.parse_args()

    phases_json = DEFAULT_PHASES_JSON

    n = int(args.d)
    dims = 2  # dimension fixed to 2D
    if n < 1:
        raise ValueError("-d must be >= 1.")

    K = num_k_qubits(dims)
    m = 2 + K
    n_sys = dims * n
    total_qubits = 1 + m + n_sys

    (
        target_function,
        degree,
        effective_degree,
        parity,
        phases,
        max_scale,
        cheb_coeffs,
    ) = load_phase_metadata(phases_json)
    if cheb_coeffs is None:
        raise ValueError(
            f"{phases_json} has no 'cheb_coeffs'. Run "
            "inversion_phase_rotations.py to produce the P^{MI} polynomial."
        )
    phases_used = list(phases)

    # Extra context from the inversion metadata (for the eigenvalue report).
    meta = json.loads(phases_json.read_text(encoding="utf-8"))
    kappa = float(meta.get("kappa", 0.0))
    epsilon = float(meta.get("epsilon", 0.0))
    poly_scale = 1.0 / (2.0 * kappa) if kappa else float("nan")

    # Block encoding + qubitization walk + QSVT phase sequence.
    u_l = build_u_laplacian_periodic(n, dims)
    w = build_w_operator(u_l, m, n_sys)
    full_qsp = build_qsp_sequence(w.to_gate(label="W"), phases_used, m, n_sys)

    # Classical references.
    lap = scaled_periodic_laplacian(n, dims)
    poly_target_matrix = chebyshev_matrix_eval(cheb_coeffs, lap)

    # Sanity: the bare block encoding really encodes L~_p (extract <0^m|U_L|0^m>).
    u_l_op = Operator(u_l).data
    anc0 = np.zeros(2**m, dtype=complex)
    anc0[0] = 1.0  # all ancillas |0>
    left_be = np.kron(np.eye(2**n_sys, dtype=complex), anc0.conj()[None, :])
    right_be = np.kron(np.eye(2**n_sys, dtype=complex), anc0[:, None])
    be_block = left_be @ u_l_op @ right_be
    be_err = float(np.max(np.abs(be_block - lap)))

    # QSVT readout: P(A) appears in the imaginary part; sign fixed by parity.
    par = parity if parity is not None else (effective_degree % 2)
    sign = -1.0 if (par % 2 == 1) else 1.0
    qsp_block = extract_block(full_qsp, m, n_sys)
    p_recovered = sign * qsp_block.imag.astype(complex)

    block_err = p_recovered - poly_target_matrix.real
    block_err_max_entry = float(np.max(np.abs(block_err)))
    block_err_spectral = float(np.linalg.norm(block_err, ord=2))

    print("=== QSVT inversion of the 2D periodic Laplacian ===")
    print("source_json      :", phases_json)
    print("target_function  :", target_function)
    print(f"kappa            : {kappa:g}")
    print(f"epsilon          : {epsilon:g}")
    print(f"poly_scale 1/2k  : {poly_scale:.8f}")
    print(f"dimensions D     : {dims}")
    print(f"qubits/dim (-d)  : {n}   (grid per dim N = {2**n})")
    print(f"selector qubits K: {K}   (= ceil(log2 D))")
    print(f"ancillas m       : {m}   (= 2 + K)")
    print(f"system qubits    : {n_sys}   (= D * d, full grid = {2**n_sys})")
    print(f"total_qubits     : {total_qubits}   (q + {m} ancilla + {n_sys} system)")
    print("degree           :", degree)
    print("effective_degree :", effective_degree)
    print("parity           :", parity)
    print("total_phases     :", len(phases_used))
    print("num_W_applications:", len(phases_used) - 1)
    print(
        f"block_encoding_check  : max|<0^m|U_L|0^m> - L~_p^(D)| = {be_err:.6e} "
        "(machine precision)"
    )
    print(
        f"block_error_max_entry : {block_err_max_entry:.6e} (machine precision error)"
    )
    print(
        f"block_error_2_spectral: {block_err_spectral:.6e} (machine precision error)"
    )

    # Eigenvalue-level view: recovered P^{MI}(lambda) vs ideal scale/lambda.
    from numpy.polynomial.chebyshev import chebval

    evals = np.linalg.eigvalsh(lap)
    uniq, counts = np.unique(np.round(evals, 8), return_counts=True)
    forbidden_hit = False
    print("\nEigenvalue inversion report (lambda in [-1, 0]):")
    print(
        f"  {'lambda':>10}  {'mult':>4}  {'P^MI(lambda)':>14}  "
        f"{'scale/lambda':>14}  region"
    )
    for lam, mult in zip(uniq, counts):
        pv = float(chebval(lam, cheb_coeffs))
        if abs(lam) < 1e-12:
            ideal = 0.0
            region = "zero mode"
        else:
            ideal = poly_scale / lam
            if kappa and abs(lam) >= 1.0 / kappa:
                region = "valid"
            else:
                region = "forbidden"
                forbidden_hit = True
        print(f"  {lam:>10.5f}  {mult:>4d}  {pv:>14.6e}  {ideal:>14.6e}  {region}")

    if forbidden_hit:
        print(
            f"\n[warning] Some non-zero eigenvalues fall inside |lambda| < 1/kappa "
            f"= {1.0 / kappa:.4f}. The block encoding is still exact, but P^{{MI}} "
            f"does not invert those modes accurately. Regenerate phases with a "
            f"larger --kappa (need 1/kappa <= min non-zero |lambda|)."
        )

    args.output_block_metrics.write_text(
        json.dumps(
            {
                "source_json": str(phases_json),
                "operator": "scaled_periodic_laplacian",
                "target_function": target_function,
                "dimensions": dims,
                "qubits_per_dim": n,
                "selector_qubits": K,
                "ancilla_qubits": m,
                "num_system_qubits": n_sys,
                "grid_size": 2**n_sys,
                "total_qubits": total_qubits,
                "kappa": kappa,
                "epsilon": epsilon,
                "poly_scale": poly_scale,
                "degree": degree,
                "effective_degree": effective_degree,
                "parity": parity,
                "total_phases": len(phases_used),
                "block_encoding_check_error": be_err,
                "block_max_entry_error": block_err_max_entry,
                "block_spectral_error": block_err_spectral,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    u_l_path = Path(f"laplacian_{args.save_prefix}_U_L.txt")
    w_path = Path(f"laplacian_{args.save_prefix}_W.txt")
    k_line = (
        f"  q_3..q_{2 + K} = k selector register (K = {K})\n" if K > 0 else ""
    )
    u_l_text = (
        "=== Block encoding U_L of the scaled D-dim periodic Laplacian ===\n"
        f"D = {dims}, grid per dim N = {2**n}; <0^{m}| U_L |0^{m}> = L~_p^(D).\n\n"
        "registers (inside U_L):\n"
        "  q_0 = l0 = block-encoding ancilla\n"
        "  q_1 = l1 = block-encoding ancilla\n"
        + (f"  q_2..q_{1 + K} = k selector register (K = {K})\n" if K > 0 else "")
        + f"  next {n_sys} qubits = D system sub-registers (n = {n} each)\n\n"
        "sequence: [U_prep_k] ; H,Z on l0,l1 ; per-dim S^- (l1=0,k=d) & "
        "S^+ (l0=1,k=d) ; H on l0,l1 ; [U_prep_k^dag]\n\n"
        f"{u_l.draw(output='text')}\n"
    )
    u_l_path.write_text(u_l_text, encoding="utf-8")

    w_text = (
        "=== Fig. 7-style qubitization walk W for the periodic Laplacian ===\n"
        "Combined walk built from U' plus the operators X_q and U_R.\n\n"
        "definition:\n"
        "  W = U_R * X_q * U'\n"
        "  U' = |0><0|_q ⊗ U_L + |1><1|_q ⊗ U_L^dag\n"
        f"  U_R = I - 2|Pi><Pi|, |Pi> = |+>_q |0>^(⊗{m})\n\n"
        "registers:\n"
        "  q_0 = q  = qubitization control\n"
        "  q_1 = l0 = block-encoding ancilla\n"
        "  q_2 = l1 = block-encoding ancilla\n"
        + k_line
        + f"  remaining {n_sys} qubits = system grid register\n\n"
        "Qiskit drawing of W:\n"
        f"{w.draw(output='text')}\n"
    )
    w_path.write_text(w_text, encoding="utf-8")

    # QSP/QSVT phase sequence drawing. The full sequence (all phases + walks) is
    # far too wide to read, so draw a faithful but truncated illustrative slice
    # built by the same build_qsp_sequence routine.
    qsp_path = Path(f"laplacian_{args.save_prefix}_QSP.txt")
    n_illus = min(4, len(phases_used))
    qsp_illus = build_qsp_sequence(
        w.to_gate(label="W"), phases_used[:n_illus], m, n_sys
    )
    qsp_text = (
        "=== QSVT phase sequence U_qsp for the 2D periodic Laplacian ===\n"
        "This is the circuit built by build_qsp_sequence().\n\n"
        "definition:\n"
        "  U_qsp = R(phi_0) W R(phi_1) W ... W R(phi_d)\n"
        "  R(phi) = exp( i phi (2|Pi><Pi| - I) ) acts on q + m ancilla\n"
        f"  |Pi> = |+>_q |0>^(⊗{m}),  W = U_R X_q U' is the walk operator\n\n"
        "registers:\n"
        "  q_0 = q  = qubitization control\n"
        "  q_1 = l0 = block-encoding ancilla\n"
        "  q_2 = l1 = block-encoding ancilla\n"
        + k_line
        + f"  remaining {n_sys} qubits = system grid register\n\n"
        f"full sequence uses {len(phases_used)} R(phi) rotations and "
        f"{len(phases_used) - 1} W walks.\n"
        f"Illustrative slice below shows the first {n_illus} rotations / "
        f"{max(n_illus - 1, 0)} walks:\n\n"
        f"{qsp_illus.draw(output='text')}\n"
    )
    qsp_path.write_text(qsp_text, encoding="utf-8")

    print("\nCircuit drawings written to text files.")
    print(f"Saved: {u_l_path}, {w_path}, {qsp_path}, and {args.output_block_metrics}")


if __name__ == "__main__":
    main()
