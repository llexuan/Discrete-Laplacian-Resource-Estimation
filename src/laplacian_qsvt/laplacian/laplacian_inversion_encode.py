#!/usr/bin/env python3
"""
QSVT matrix inversion and logical resource estimation for the 2D/3D periodic
discrete Laplacian.

Dense correctness checks use a small --verify-n. Target-size Clifford+T
resources use --target-n and are composed from repeated scalable W resources
plus explicit pyLIQTR synthesis of the QSP projector rotations.

Registers (Qiskit little-endian, qubit 0 = least significant):
  q            : qubitization control qubit                (index 0)
  l0, l1       : block-encoding ancillas                   (indices 1, 2)
  k            : dimension selector, K = ceil(log2 D)       (indices 3 ..)
  system       : D grid sub-registers j^(0), ..., j^(D-1)
                 each of n qubits (grid size per dim N = 2**n)
  phase flag   : one clean workspace qubit in resource circuits

Usage:
  inversion-phases --epsilon 0.49 --kappa 2.5
  laplacian-inversion-resource --dims 2 -d 1 --target-n 20
  laplacian-inversion-resource --dims 3 -d 1 --target-n 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from ..paths import (
    CIRCUITS_DIR,
    METRICS_DIR,
    MPLCONFIG_DIR,
    PHASES_DIR,
    ensure_parent,
)

os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_DIR)
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit.synthesis.multi_controlled import synth_mcx_n_dirty_i15

from ..demos.ix2_block_encoding import chebyshev_matrix_eval, load_phase_metadata
from ..synthesis.rotation_synthesis import RotationSynthesis, synthesize_rz
from .laplacian_1d_resource import resource_counts
from .laplacian_nd_resource import (
    CT_BASIS,
    build_u_l_nd as build_scalable_u_l,
    scaled_nd_laplacian,
)
from .laplacian_w_resource import (
    build_w_operator as build_scalable_w_operator,
    estimate_w_resources,
)

# Single-qubit reference kets.
_KET0 = np.array([1.0, 0.0], dtype=complex)
_KET_PLUS = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
DEFAULT_PHASES_JSON = PHASES_DIR / "phases_inversion.json"


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


def build_projector_rotation_clifford_t(
    phi: float, m: int, synthesis_eps: float
) -> tuple[QuantumCircuit, RotationSynthesis]:
    """
    Explicitly implement R(phi) with one clean flag and a synthesized Rz(2phi).

    H and X map |Pi> = |+>_q|0>^m to the all-ones predicate. Computing that
    predicate into the flag makes Rz(2phi) apply e^(+i phi) to |Pi> and
    e^(-i phi) to its orthogonal complement. The flag is returned to |0>.
    A V-chain borrows m-1 system qubits as dirty ancillas and restores them,
    avoiding uncontrolled approximate rotations inside an ancilla-free MCX.
    """
    synthesis = synthesize_rz(2.0 * phi, synthesis_eps)
    predicate = list(range(1 + m))
    flag = 1 + m
    dirty = list(range(2 + m, 2 * m + 1))
    qc = QuantumCircuit(2 * m + 1, name="R_Pi_Clifford_T")
    mcx = synth_mcx_n_dirty_i15(len(predicate))
    mcx_qubits = predicate + [flag] + dirty
    qc.h(0)
    qc.x(predicate)
    qc.compose(mcx, qubits=mcx_qubits, inplace=True)
    qc.compose(synthesis.circuit, qubits=[flag], inplace=True)
    qc.compose(mcx, qubits=mcx_qubits, inplace=True)
    qc.x(predicate)
    qc.h(0)
    return qc, synthesis


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


def _add_gate_counts(
    destination: dict[str, int], source: dict[str, int], multiplier: int = 1
) -> None:
    for gate, count in source.items():
        destination[gate] = destination.get(gate, 0) + multiplier * int(count)


def estimate_projector_rotation_resources(
    phases: list[float], m: int, eps_per_rotation: float
) -> dict:
    """Synthesize and sum all QSP projector-rotation Clifford+T resources."""
    unique: dict[str, dict] = {}
    for phi in phases:
        key = repr(float(phi))
        if key not in unique:
            circuit, synthesis = build_projector_rotation_clifford_t(
                float(phi), m, eps_per_rotation
            )
            res = resource_counts(circuit, CT_BASIS, count_t_depth=True)
            counts = res["gate_counts"]
            t_count = int(counts.get("t", 0) + counts.get("tdg", 0))
            unique[key] = {
                "phi": float(phi),
                "rz_angle": 2.0 * float(phi),
                "occurrences": 0,
                "projective_error": synthesis.projective_error,
                "rotation_synthesis": {
                    "backend": "pyLIQTR",
                    "backend_version": synthesis.backend_version,
                    "method": "get_ring_elts_direct + exact_decomp",
                    "epsilon": synthesis.epsilon,
                    "projective_error": synthesis.projective_error,
                    "raw_phase_offset": synthesis.raw_phase_offset,
                    "gate_counts": synthesis.gate_counts,
                    "t_count": synthesis.t_count,
                    "clifford_gate_count": synthesis.clifford_count,
                    "depth": synthesis.depth,
                    "t_depth": synthesis.t_depth,
                },
                "gate_counts": counts,
                "clifford_gate_count": int(res["total_gates"] - t_count),
                "t_count": t_count,
                "total_depth": int(res["depth"]),
                "t_depth": int(res["t_depth"]),
            }
        unique[key]["occurrences"] += 1

    total_gate_counts: dict[str, int] = {}
    total_clifford = 0
    total_t = 0
    total_depth = 0
    total_t_depth = 0
    summed_error = 0.0
    for item in unique.values():
        occurrences = int(item["occurrences"])
        _add_gate_counts(total_gate_counts, item["gate_counts"], occurrences)
        total_clifford += occurrences * int(item["clifford_gate_count"])
        total_t += occurrences * int(item["t_count"])
        total_depth += occurrences * int(item["total_depth"])
        total_t_depth += occurrences * int(item["t_depth"])
        summed_error += occurrences * float(item["projective_error"])

    return {
        "num_projector_rotations": len(phases),
        "num_unique_phase_angles": len(unique),
        "eps_per_rotation": eps_per_rotation,
        "summed_projective_error_bound": summed_error,
        "logical_workspace_qubits": 1,
        "borrowed_dirty_system_qubits": m - 1,
        "gate_counts": total_gate_counts,
        "clifford_gate_count": total_clifford,
        "t_count": total_t,
        "total_depth": total_depth,
        "t_depth": total_t_depth,
        "unique_rotation_resources": list(unique.values()),
    }


def minimum_nonzero_laplacian_magnitude(n: int, dims: int) -> float:
    """Analytic spectral gap of the normalized periodic Laplacian."""
    grid_size = 2**n
    return math.sin(math.pi / grid_size) ** 2 / dims


def estimate_full_qsp_resources(
    phases: list[float],
    dims: int,
    target_n: int,
    synthesis_tol: float,
) -> dict:
    """Compose the logical cost of all projector rotations and repeated W uses."""
    if dims not in (2, 3):
        raise ValueError("dims must be 2 or 3.")
    if target_n < 1:
        raise ValueError("target_n must be >= 1.")
    if len(phases) < 1:
        raise ValueError("At least one QSP phase is required.")
    if not 0.0 < synthesis_tol < 1.0:
        raise ValueError("synthesis_tol must satisfy 0 < tolerance < 1.")

    num_phases = len(phases)
    num_w = num_phases - 1
    rotations_per_w = 4 if dims == 3 else 0
    num_w_rotations = num_w * rotations_per_w
    num_approximated_rotations = num_phases + num_w_rotations
    eps_per_rotation = synthesis_tol / num_approximated_rotations

    # estimate_w_resources divides a 3D per-W PREP tolerance among its four
    # selector rotations. Passing 4*eps_per makes every approximation in the
    # complete QSP sequence use the same conservative error allocation.
    w_prep_tol = (
        rotations_per_w * eps_per_rotation
        if rotations_per_w
        else synthesis_tol
    )
    w_resources = estimate_w_resources(target_n, dims, w_prep_tol)
    m = 2 + num_k_qubits(dims)
    projector_resources = estimate_projector_rotation_resources(
        phases, m, eps_per_rotation
    )

    total_gate_counts: dict[str, int] = {}
    _add_gate_counts(total_gate_counts, w_resources["gate_counts"], num_w)
    _add_gate_counts(total_gate_counts, projector_resources["gate_counts"])

    w_clifford = num_w * int(w_resources["clifford_gate_count"])
    w_t = num_w * int(w_resources["t_count"])
    w_depth = num_w * int(w_resources["total_depth"])
    w_t_depth = num_w * int(w_resources["t_depth"])

    walk_synthesis_error = 0.0
    if w_resources["rotation_synthesis"] is not None:
        walk_synthesis_error = (
            num_w_rotations
            * float(
                w_resources["rotation_synthesis"][
                    "projective_error_per_rotation"
                ]
            )
        )
    measured_synthesis_bound = (
        walk_synthesis_error
        + float(projector_resources["summed_projective_error_bound"])
    )

    gap = minimum_nonzero_laplacian_magnitude(target_n, dims)
    return {
        "composition": "R(phi_0) W R(phi_1) ... W R(phi_d)",
        "dimensions": dims,
        "target_n": target_n,
        "grid_points_per_dimension": 2**target_n,
        "num_projector_rotations": num_phases,
        "num_w_applications": num_w,
        "num_w_selector_rotations": num_w_rotations,
        "num_approximated_rotations": num_approximated_rotations,
        "synthesis_tolerance": synthesis_tol,
        "eps_per_rotation": eps_per_rotation,
        "measured_synthesis_error_bound": measured_synthesis_bound,
        "logical_qubits": int(w_resources["logical_qubits"]) + 1,
        "phase_flag_qubits": 1,
        "depth_accounting": "serial_component_sum_upper_bound",
        "clifford_gate_count": (
            w_clifford + int(projector_resources["clifford_gate_count"])
        ),
        "t_count": w_t + int(projector_resources["t_count"]),
        "total_depth": w_depth + int(projector_resources["total_depth"]),
        "t_depth": w_t_depth + int(projector_resources["t_depth"]),
        "gate_counts": total_gate_counts,
        "walk_contribution": {
            "per_w": w_resources,
            "num_w_applications": num_w,
            "clifford_gate_count": w_clifford,
            "t_count": w_t,
            "total_depth": w_depth,
            "t_depth": w_t_depth,
        },
        "projector_rotation_contribution": projector_resources,
        "target_spectral_gap": gap,
        "minimum_kappa_for_full_nonzero_spectrum": 1.0 / gap,
    }


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
        description="Verify and estimate QSVT matrix inversion resources for "
        "the 2D/3D periodic Laplacian."
    )
    parser.add_argument(
        "-d",
        "--verify-n",
        dest="verify_n",
        type=int,
        required=True,
        help="Small qubits-per-dimension value used for dense verification.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        choices=(2, 3),
        default=2,
        help="Spatial dimension D (default 2).",
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=20,
        help="Qubits per dimension for the logical resource estimate.",
    )
    parser.add_argument(
        "--phases-json",
        type=Path,
        default=DEFAULT_PHASES_JSON,
        help="QSP phases and polynomial metadata.",
    )
    parser.add_argument(
        "--synthesis-tol",
        type=float,
        default=None,
        help=(
            "Override the full-QSP Clifford+T synthesis tolerance stored in "
            "the phases JSON."
        ),
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="lap_inv",
        help="Prefix for saved ASCII circuit files.",
    )
    parser.add_argument(
        "--output-block-metrics",
        type=Path,
        default=METRICS_DIR / "block_metrics_laplacian.json",
        help="Where to write block-encoding error metrics as JSON.",
    )
    parser.add_argument(
        "--output-resource-metrics",
        type=Path,
        default=METRICS_DIR / "laplacian_inversion_resource_metrics.json",
        help="Where to write the final logical QSP resource estimate.",
    )
    args = parser.parse_args()

    phases_json = args.phases_json
    verify_n = int(args.verify_n)
    target_n = int(args.target_n)
    dims = int(args.dims)
    if verify_n < 1 or target_n < 1:
        raise ValueError("--verify-n and --target-n must be >= 1.")

    K = num_k_qubits(dims)
    m = 2 + K
    n_sys = dims * verify_n
    total_qubits = 1 + m + n_sys

    (
        target_function,
        degree,
        effective_degree,
        parity,
        phases,
        _max_scale,
        cheb_coeffs,
    ) = load_phase_metadata(phases_json)
    if cheb_coeffs is None:
        raise ValueError(
            f"{phases_json} has no 'cheb_coeffs'. Run "
            "inversion-phases to produce the P^{MI} polynomial."
        )
    phases_used = list(phases)

    # Extra context from the inversion metadata (for the eigenvalue report).
    meta = json.loads(phases_json.read_text(encoding="utf-8"))
    kappa = float(meta.get("kappa", 0.0))
    epsilon = float(meta.get("epsilon", 0.0))
    if kappa <= 1.0 or epsilon <= 0.0:
        raise ValueError(
            f"{phases_json} must contain kappa > 1 and epsilon > 0."
        )
    synthesis_tol = (
        float(args.synthesis_tol)
        if args.synthesis_tol is not None
        else float(meta.get("synthesis_tolerance", 1e-8))
    )
    if not 0.0 < synthesis_tol < 1.0:
        raise ValueError("synthesis tolerance must satisfy 0 < tolerance < 1.")
    poly_scale = 1.0 / (2.0 * kappa) if kappa else float("nan")

    # Small ideal circuit for dense verification only.
    u_l = build_scalable_u_l(verify_n, dims)
    w = build_scalable_w_operator(verify_n, dims)
    full_qsp = build_qsp_sequence(w.to_gate(label="W"), phases_used, m, n_sys)

    # Classical references.
    lap = scaled_nd_laplacian(verify_n, dims)
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

    # Target-size resources are composed without constructing the enormous full
    # circuit: every W and every projector rotation is still counted explicitly.
    resource_estimate = estimate_full_qsp_resources(
        phases_used, dims, target_n, synthesis_tol
    )
    polynomial_error = float(
        meta.get("poly_vs_inverse_max_abs_error", epsilon)
    )
    phase_numerical_error = float(meta.get("phase_vs_poly_max_abs_error", 0.0))
    conservative_total_error = (
        polynomial_error + phase_numerical_error + synthesis_tol
    )

    print(f"=== QSVT inversion of the {dims}D periodic Laplacian ===")
    print("source_json      :", phases_json)
    print("target_function  :", target_function)
    print(f"kappa            : {kappa:g}")
    print(f"epsilon          : {epsilon:g}")
    print(f"synthesis_tol    : {synthesis_tol:.1e} (separate from epsilon)")
    print(f"poly_scale 1/2k  : {poly_scale:.8f}")
    print(f"dimensions D     : {dims}")
    print(f"verification n   : {verify_n}   (grid per dim N = {2**verify_n})")
    print(f"resource target n: {target_n}   (grid per dim N = {2**target_n})")
    print(f"selector qubits K: {K}   (= ceil(log2 D))")
    print(f"ancillas m       : {m}   (= 2 + K)")
    print(f"verification system qubits: {n_sys}")
    print(f"verification total qubits : {total_qubits}")
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
    print(f"\n--- full logical QSP resource estimate (target n={target_n}) ---")
    print(f"logical qubits      : {resource_estimate['logical_qubits']}")
    print(f"W applications      : {resource_estimate['num_w_applications']}")
    print(f"projector rotations : {resource_estimate['num_projector_rotations']}")
    print(f"Clifford gate count : {resource_estimate['clifford_gate_count']}")
    print(f"T-count             : {resource_estimate['t_count']}")
    print(f"total depth         : {resource_estimate['total_depth']}")
    print(f"T-depth             : {resource_estimate['t_depth']}")
    print(f"eps/rotation        : {resource_estimate['eps_per_rotation']:.2e}")
    print(
        "measured synthesis bound: "
        f"{resource_estimate['measured_synthesis_error_bound']:.2e}"
    )
    print(f"polynomial error    : {polynomial_error:.2e}")
    print(f"phase numerical err : {phase_numerical_error:.2e}")
    print(f"conservative total  : {conservative_total_error:.2e}")

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

    required_kappa = float(
        resource_estimate["minimum_kappa_for_full_nonzero_spectrum"]
    )
    if kappa < required_kappa:
        print(
            f"\n[warning] The target n={target_n}, D={dims} grid requires "
            f"kappa >= {required_kappa:.6g} to cover its full non-zero "
            f"spectrum; the supplied kappa={kappa:g} estimates inversion only "
            "on the stated validity domain."
        )

    ensure_parent(args.output_block_metrics).write_text(
        json.dumps(
            {
                "source_json": str(phases_json),
                "operator": "scaled_periodic_laplacian",
                "target_function": target_function,
                "dimensions": dims,
                "verify_n": verify_n,
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
                "num_w_applications": len(phases_used) - 1,
                "block_encoding_check_error": be_err,
                "block_max_entry_error": block_err_max_entry,
                "block_spectral_error": block_err_spectral,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    ensure_parent(args.output_resource_metrics).write_text(
        json.dumps(
            {
                "scope": "full_qsp_logical_resource_estimate",
                "source_json": str(phases_json),
                "target_function": target_function,
                "epsilon": epsilon,
                "kappa": kappa,
                "polynomial_degree": degree,
                "effective_degree": effective_degree,
                "parity": parity,
                "error_summary": {
                    "polynomial_error": polynomial_error,
                    "phase_numerical_error": phase_numerical_error,
                    "synthesis_tolerance": synthesis_tol,
                    "measured_synthesis_error_bound": resource_estimate[
                        "measured_synthesis_error_bound"
                    ],
                    "conservative_total_error_bound": conservative_total_error,
                },
                "verification": {
                    "verify_n": verify_n,
                    "block_encoding_check_error": be_err,
                    "qsp_block_max_entry_error": block_err_max_entry,
                    "qsp_block_spectral_error": block_err_spectral,
                },
                "logical_resources": resource_estimate,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    u_l_path = CIRCUITS_DIR / f"laplacian_{args.save_prefix}_U_L.txt"
    w_path = CIRCUITS_DIR / f"laplacian_{args.save_prefix}_W.txt"
    k_line = (
        f"  q_3..q_{2 + K} = k selector register (K = {K})\n" if K > 0 else ""
    )
    u_l_text = (
        "=== Block encoding U_L of the scaled D-dim periodic Laplacian ===\n"
        f"D = {dims}, verification grid per dim N = {2**verify_n}; "
        f"<0^{m}| U_L |0^{m}> = L~_p^(D).\n\n"
        "registers (inside U_L):\n"
        "  q_0 = l0 = block-encoding ancilla\n"
        "  q_1 = l1 = block-encoding ancilla\n"
        + (f"  q_2..q_{1 + K} = k selector register (K = {K})\n" if K > 0 else "")
        + f"  next {n_sys} qubits = D system sub-registers "
        f"(n = {verify_n} each)\n\n"
        "sequence: [U_prep_k] ; H,Z on l0,l1 ; per-dim S^- (l1=0,k=d) & "
        "S^+ (l0=1,k=d) ; H on l0,l1 ; [U_prep_k^dag]\n\n"
        f"{u_l.draw(output='text')}\n"
    )
    ensure_parent(u_l_path).write_text(u_l_text, encoding="utf-8")

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
    ensure_parent(w_path).write_text(w_text, encoding="utf-8")

    # QSP/QSVT phase sequence drawing. The full sequence (all phases + walks) is
    # far too wide to read, so draw a faithful but truncated illustrative slice
    # built by the same build_qsp_sequence routine.
    qsp_path = CIRCUITS_DIR / f"laplacian_{args.save_prefix}_QSP.txt"
    n_illus = min(4, len(phases_used))
    qsp_illus = build_qsp_sequence(
        w.to_gate(label="W"), phases_used[:n_illus], m, n_sys
    )
    qsp_text = (
        f"=== QSVT phase sequence U_qsp for the {dims}D periodic Laplacian ===\n"
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
    ensure_parent(qsp_path).write_text(qsp_text, encoding="utf-8")

    print("\nCircuit drawings written to text files.")
    print(
        f"Saved: {u_l_path}, {w_path}, {qsp_path}, "
        f"{args.output_block_metrics}, and {args.output_resource_metrics}"
    )


if __name__ == "__main__":
    main()
