#!/usr/bin/env python3
"""
Nested QSVT inversion of the 1D preconditioned periodic Laplacian (conditioned kernel).

    full sequence = QFT^dag . [R(phi_0) W_cond R(phi_1) ... W_cond R(phi_d)] . QFT

W_cond is the repository's qubitization walk with U_L replaced by U_Lcond, the
inner sinc^2 QSVT over U_X. The outer P^MI phases (kappa = 2.5) are reused, and
the encoded operator's spectrum P_2d(2 m_k / N) lies in [0.405, 1].

Scope: the conditioned kernel L_cond^{-1} (A = C L_cond, C = diag((pi m_k/N)^2)),
assuming C^{-1} b is prepared directly.

Registers (Qiskit little-endian):
  q          : outer qubitization control                  (index 0)
  a, p, J    : U_X ancillas, J = ceil(log2(q_sys - 1))      (indices 1 .. m)
  system     : q_sys grid qubits, frequency bits reversed by the swap-free QFT
  phase flag : clean flag shared by inner and outer phase rotations

Usage:
  sinc2-phases --d 5
  preconditioned-inversion-resource --verify-n 4 --target-n 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from ..paths import CIRCUITS_DIR, METRICS_DIR, MPLCONFIG_DIR, PHASES_DIR, ensure_parent

os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_DIR)
from numpy.polynomial import chebyshev as C
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from ..demos.ix2_block_encoding import chebyshev_matrix_eval, load_phase_metadata
from ..laplacian.laplacian_1d_resource import resource_counts
from ..laplacian.laplacian_inversion_encode import estimate_projector_rotation_resources
from ..laplacian.laplacian_nd_resource import CT_BASIS
from .gates import RotationEmitter, eighth_turns
from .sinc2 import (
    DEFAULT_PHASES_JSON as DEFAULT_INNER_JSON,
    block_phase_factor,
    p2d,
    sinc2_phases,
    sinc2_target,
)
from .u_x import (
    KET0,
    KET_PLUS,
    ancilla_zero_isometry,
    build_prep,
    build_select,
    build_u_x,
    draw_prep_select,
    frequency_of_system_states,
    isometry,
    walk_layout,
    x_hat_values,
)
from .walk import (
    append_signal_rotation,
    build_inner_qsvt,
    build_qft,
    build_reflection,
    build_walk,
)

CASE = "preconditioned_1d_conditioned_kernel"
CASE_TITLE = "Preconditioned 1D Laplacian, conditioned kernel (nested sinc^2 block encoding)"
DEFAULT_OUTER_JSON = PHASES_DIR / "phases_inversion.json"
DEFAULT_SYNTHESIS_TOL = 1e-7


# --- dense verification -------------------------------------------------------
def _matrix(lay, build) -> np.ndarray:
    qc = lay.circuit("piece")
    build(qc)
    return Operator(qc).data


def inner_matrix(lay, psi: list[float], sign_control: bool) -> np.ndarray:
    """Dense U_in from the circuit pieces of build_inner_qsvt, multiplied in order."""
    emitter = RotationEmitter()
    u_x = Operator(build_u_x(lay, emitter, prep=build_prep(lay, emitter))).data
    total = np.eye(2**lay.total, dtype=complex)
    for index, angle in enumerate(reversed(psi)):
        total = _matrix(
            lay, lambda qc: append_signal_rotation(qc, lay, emitter, angle, sign_control)
        ) @ total
        if index + 1 < len(psi):
            total = u_x @ total
    return total


def phase_factor_angle(num_u_x: int) -> float:
    c = block_phase_factor(num_u_x)
    return math.atan2(c.imag, c.real)


def walk_matrix(lay, psi: list[float]) -> np.ndarray:
    """Dense W_cond = U_R X_q Rz(-2 arg c)_q U_in(q-signed), piece by piece."""
    alpha = phase_factor_angle(len(psi) - 1)

    def tail(qc):
        qc.rz(-2.0 * alpha, lay.q)
        qc.x(lay.q)
        qc.compose(build_reflection(lay), inplace=True)

    return _matrix(lay, tail) @ inner_matrix(lay, psi, True)


def sequence_block(lay, walk: np.ndarray, phases: list[float]) -> np.ndarray:
    """<Pi, 0_flag| R(phi_0) W ... W R(phi_d) |Pi, 0_flag> on the system."""
    inputs = ancilla_zero_isometry(lay, plus_q=True)
    fixed = {lay.q: KET_PLUS, **{qubit: KET0 for qubit in lay.anc}}
    signal = isometry(lay.total, [*lay.system, lay.flag], fixed)
    states = inputs.copy()
    for index, phi in enumerate(phases):
        projected = signal @ (signal.conj().T @ states)
        states = np.exp(-1j * phi) * states + (np.exp(1j * phi) - np.exp(-1j * phi)) * projected
        if index + 1 < len(phases):
            states = walk @ states
    return inputs.conj().T @ states


def dft_matrix(n: int) -> np.ndarray:
    N = 2**n
    k = np.arange(N)
    return np.exp(2j * np.pi * np.outer(k, k) / N) / math.sqrt(N)


def anti_hermitian_part(block: np.ndarray) -> np.ndarray:
    return (block - block.conj().T) / 2j


def dense_checks(
    n: int, psi: list[float], d: int, phases: list[float], cheb: list[float], parity: int, kappa: float
) -> dict:
    """Exact small-size checks of U_X, U_Lcond, W_cond, QFT and the full sequence."""
    started = time.monotonic()
    natural = walk_layout(n, "natural")
    emitter = RotationEmitter()
    u_x = Operator(build_u_x(natural, emitter, prep=build_prep(natural, emitter))).data
    embed = ancilla_zero_isometry(natural)
    theta_nat = x_hat_values(n)[frequency_of_system_states(natural)]
    u_x_block = embed.conj().T @ u_x @ embed
    inner_block = block_phase_factor(len(psi) - 1) * (
        embed.conj().T @ inner_matrix(natural, psi, False) @ embed
    )

    lay = walk_layout(n)
    theta = x_hat_values(n)[frequency_of_system_states(lay)]
    walk = walk_matrix(lay, psi)
    plus = ancilla_zero_isometry(lay, plus_q=True)
    signal_block = plus.conj().T @ walk @ plus
    # U' = X_q U_R W, and its q = 1 branch must be the adjoint of its q = 0 branch.
    tail = _matrix(lay, lambda qc: (qc.compose(build_reflection(lay), inplace=True), qc.x(lay.q)))
    u_prime = (tail @ walk).reshape([2] * (2 * lay.total))
    axis_out, axis_in = lay.total - 1 - lay.q, 2 * lay.total - 1 - lay.q
    branches = [
        np.take(np.take(u_prime, b, axis=axis_in), b, axis=axis_out).reshape(
            2 ** (lay.total - 1), 2 ** (lay.total - 1)
        )
        for b in (0, 1)
    ]
    off = np.take(np.take(u_prime, 0, axis=axis_in), 1, axis=axis_out)

    sign = -1.0 if parity % 2 else 1.0
    block = sequence_block(lay, walk, phases)
    recovered = sign * anti_hermitian_part(block)
    encoded = p2d(theta, d)
    poly_target = np.diag(C.chebval(encoded, cheb))

    qft = Operator(build_qft(lay, RotationEmitter())).data
    qft_ct = Operator(build_qft(lay, RotationEmitter(1e-6))).data
    position = sign * anti_hermitian_part(qft.conj().T @ block @ qft)
    dft = dft_matrix(n)
    l_cond_poly = dft.conj().T @ np.diag(p2d(x_hat_values(n), d)) @ dft
    l_cond = dft.conj().T @ np.diag(sinc2_target(x_hat_values(n))) @ dft
    kernel = np.linalg.inv(l_cond) / (2.0 * kappa)
    rows = frequency_of_system_states(lay)
    return {
        "verify_n": n,
        "dense_qubits": lay.total,
        "u_x_block_error": float(np.max(np.abs(u_x_block - np.diag(theta_nat)))),
        "u_x_hermiticity_error": float(np.max(np.abs(u_x - u_x.conj().T))),
        "u_lcond_hermitian_part_error": float(
            np.max(np.abs((inner_block + inner_block.conj().T) / 2 - np.diag(p2d(theta_nat, d))))
        ),
        "walk_signal_block_error": float(np.max(np.abs(signal_block + np.diag(encoded)))),
        "u_prime_adjoint_branch_error": float(np.max(np.abs(branches[1] - branches[0].conj().T))),
        "u_prime_offdiagonal_max": float(np.max(np.abs(off))),
        "qft_bit_reversed_dft_error": float(np.max(np.abs(qft - dft[rows]))),
        "qft_clifford_t_error_at_1e-6": float(np.linalg.norm(qft_ct - qft, 2)),
        "qsp_block_max_entry_error": float(np.max(np.abs(recovered - poly_target))),
        "qsp_block_spectral_error": float(np.linalg.norm(recovered - poly_target, 2)),
        "position_vs_classical_polynomial_error": float(
            np.max(np.abs(position - chebyshev_matrix_eval(cheb, l_cond_poly)))
        ),
        "position_vs_exact_conditioned_kernel_error": float(
            np.linalg.norm(position - kernel, 2)
        ),
        "wall_seconds": time.monotonic() - started,
    }


# --- target-size spectral and error analysis ---------------------------------
def spectral_summary(n: int, d: int, cheb: list[float], kappa: float, inner_phase_error: float) -> dict:
    """Exact mode-by-mode analysis; every operator here is diagonal in k."""
    m = np.arange(0, 2 ** (n - 1) + 1)
    theta = 2.0 * m / 2**n
    exact = sinc2_target(theta)
    encoded = p2d(theta, d)
    out_poly = C.chebval(encoded, cheb)
    out_exact = C.chebval(exact, cheb)
    ideal = 1.0 / (2.0 * kappa * exact)
    domain = np.linspace(1.0 / kappa, 1.0, 20001)
    slope = float(np.max(np.abs(C.chebval(domain, C.chebder(cheb)))))
    return {
        "modes_checked": int(len(m)),
        "encoded_min_eigenvalue": float(np.min(encoded)),
        "encoded_max_eigenvalue": float(np.max(encoded)),
        "exact_min_eigenvalue": float(np.min(exact)),
        "inner_polynomial_error": float(np.max(np.abs(encoded - exact))),
        "inner_error_after_inversion": float(np.max(np.abs(out_poly - out_exact))),
        "outer_polynomial_error_on_spectrum": float(np.max(np.abs(out_exact - ideal))),
        "kernel_error": float(np.max(np.abs(out_poly - ideal))),
        "outer_slope_on_domain": slope,
        "inner_phase_error_after_inversion_bound": slope * inner_phase_error,
    }


# --- target-size logical resources -------------------------------------------
def _sum_counts(target: dict, source: dict, weight: int = 1) -> None:
    for gate, count in source.items():
        target[gate] = target.get(gate, 0) + weight * int(count)


def _summary(circuit: QuantumCircuit) -> dict:
    res = resource_counts(circuit, CT_BASIS, count_t_depth=True)
    counts = res["gate_counts"]
    t_count = int(counts.get("t", 0) + counts.get("tdg", 0))
    return {
        "logical_qubits": int(res["num_qubits"]),
        "gate_counts": counts,
        "clifford_gate_count": int(res["total_gates"] - t_count),
        "t_count": t_count,
        "total_depth": int(res["depth"]),
        "t_depth": int(res["t_depth"]),
    }


def per_rotation_tolerance(n: int, psi: list[float], phases: list[float], synthesis_tol: float) -> tuple[float, dict]:
    """Split the synthesis budget over every approximated rotation occurrence."""
    lay = walk_layout(n)
    walk_counter = RotationEmitter()
    build_walk(lay, psi, walk_counter)
    qft_counter = RotationEmitter()
    build_qft(lay, qft_counter)
    outer = sum(1 for phi in phases if eighth_turns(2.0 * phi) is None)
    counts = {
        "per_w": len(walk_counter.records),
        "per_qft": len(qft_counter.records),
        "outer_projectors": outer,
    }
    counts["total"] = (len(phases) - 1) * counts["per_w"] + 2 * counts["per_qft"] + outer
    # Decade rounding keeps pyLIQTR away from fractional tolerances and above
    # the double-precision floor of its error check (~1e-13).
    eps = 10.0 ** math.floor(math.log10(synthesis_tol / counts["total"]))
    return eps, counts


def target_components(n: int, psi: list[float], eps: float) -> dict:
    """Clifford+T W_cond and QFT boundaries exactly as counted and exported."""
    lay = walk_layout(n)
    emitter = RotationEmitter(eps)
    walk, inventory = build_walk(lay, psi, emitter)
    qft_emitter = RotationEmitter(eps)
    qft = build_qft(lay, qft_emitter)
    return {
        "layout": lay,
        "W": walk,
        "W_records": emitter.records,
        "W_inventory": inventory,
        "QFT": qft,
        "QFT_dg": qft.inverse(),
        "QFT_records": qft_emitter.records,
    }


def estimate_logical_resources(
    n: int, psi: list[float], phases: list[float], synthesis_tol: float
) -> dict:
    """Compose W_cond, every projector rotation and the two QFTs at size n."""
    eps, rotation_counts = per_rotation_tolerance(n, psi, phases, synthesis_tol)
    parts = target_components(n, psi, eps)
    lay = parts["layout"]
    num_w = len(phases) - 1
    walk = _summary(parts["W"])
    inventory = parts["W_inventory"]
    w_error = sum(r["projective_error"] for r in parts["W_records"])
    prep_records = inventory["prep_records"]
    signal_records = inventory["signal_records"]
    walk["rotation_synthesis"] = {
        "backend": "pyLIQTR",
        "method": "get_ring_elts_direct + exact_decomp",
        "eps_per_rotation": eps,
        "rotations_per_w": len(parts["W_records"]),
        "prep_rotations": len(prep_records),
        "prep_uses_per_w": inventory["prep_uses"],
        "signal_rotations_per_w": len(signal_records),
        "summed_projective_error_per_w": w_error,
        "prep_angles": [r["angle"] for r in prep_records],
        "signal_rz_angles": [r["angle"] for r in signal_records],
    }

    emitter = RotationEmitter(eps)
    anatomy = {
        "PREP": _summary(build_prep(lay, emitter)),
        "SELECT": _summary(build_select(lay)),
        "U_X": _summary(build_u_x(lay, emitter)),
        "U_Lcond (q-signed inner QSVT)": _summary(
            build_inner_qsvt(lay, psi, RotationEmitter(eps), sign_control=True)[0]
        ),
        "U_R": _summary(build_reflection(lay)),
    }
    walk["anatomy"] = anatomy

    qft_error = sum(r["projective_error"] for r in parts["QFT_records"])
    boundary = []
    for identifier, circuit in (("QFT", parts["QFT"]), ("QFT_dg", parts["QFT_dg"])):
        item = _summary(circuit)
        item.update(
            id=identifier,
            occurrences=1,
            synthesized_rotations=len(parts["QFT_records"]),
            summed_projective_error=qft_error,
            system_qubits=list(lay.system),
        )
        boundary.append(item)

    projectors = estimate_projector_rotation_resources(phases, lay.m, eps)

    gate_counts: dict[str, int] = {}
    _sum_counts(gate_counts, walk["gate_counts"], num_w)
    _sum_counts(gate_counts, projectors["gate_counts"])
    for item in boundary:
        _sum_counts(gate_counts, item["gate_counts"])

    def total(key: str) -> int:
        return num_w * walk[key] + projectors[key] + sum(item[key] for item in boundary)

    walks = {
        "per_w": walk,
        "num_w_applications": num_w,
        **{key: num_w * walk[key] for key in ("clifford_gate_count", "t_count", "total_depth", "t_depth")},
    }
    boundaries = {
        "components": boundary,
        **{key: sum(item[key] for item in boundary) for key in ("clifford_gate_count", "t_count", "total_depth", "t_depth")},
    }
    measured = num_w * w_error + 2 * qft_error + projectors["summed_projective_error_bound"]
    return {
        "composition": "QFT^dag . R(phi_0) W_cond R(phi_1) ... W_cond R(phi_d) . QFT",
        "dimensions": 1,
        "target_n": n,
        "grid_points_per_dimension": 2**n,
        "num_projector_rotations": len(phases),
        "num_w_applications": num_w,
        "num_w_selector_rotations": num_w * rotation_counts["per_w"],
        "num_approximated_rotations": rotation_counts["total"],
        "approximated_rotation_breakdown": rotation_counts,
        "synthesis_tolerance": synthesis_tol,
        "eps_per_rotation": eps,
        "selector_eps_per_rotation": eps,
        "measured_synthesis_error_bound": measured,
        "signal_ancillas": lay.m,
        "system_qubits": n,
        "walk_register_qubits": 1 + lay.m + n,
        "logical_qubits": lay.total,
        "phase_flag_qubits": 1,
        "phase_flag_shared_with_inner_qsvt": True,
        "depth_accounting": "serial_component_sum_upper_bound",
        "clifford_gate_count": total("clifford_gate_count"),
        "t_count": total("t_count"),
        "total_depth": total("total_depth"),
        "t_depth": total("t_depth"),
        "gate_counts": gate_counts,
        "walk_contribution": walks,
        "projector_rotation_contribution": projectors,
        "boundary_contribution": boundaries,
        "block_encoding": {
            "kind": CASE,
            "inner_psi": list(psi),
            "inner_degree": len(psi) - 1,
            "u_x_calls_per_w": len(psi) - 1,
            "index_qubits": lay.num_index,
            "select_mcx_gates": n - 1,
            "select_mcx_controls": lay.num_index + 2,
            "signal_mcx_controls": lay.m,
            "fourier_bit_order": "bit_reversed (swap-free QFT)",
        },
        "registers": {
            "q": lay.q,
            "a": lay.a,
            "p": lay.p,
            "J": list(lay.J),
            "system": list(lay.system),
            "phase_flag": lay.flag,
        },
    }


# --- reports and drawings ------------------------------------------------------
def render_resource_report(metrics: dict) -> str:
    """Readable logical report, in the repository's inversion report format."""
    logical = metrics["logical_resources"]
    walks = logical["walk_contribution"]
    one_walk = walks["per_w"]
    projectors = logical["projector_rotation_contribution"]
    boundary = logical["boundary_contribution"]
    errors = metrics["error_summary"]
    spectrum = metrics["spectral_summary"]
    verification = metrics["verification"]
    block = logical["block_encoding"]
    synth = one_walk["rotation_synthesis"]
    lines = [
        "=== Preconditioned 1D Laplacian inverse (conditioned kernel): logical resource report ===",
        "",
        f"Sequence              : {logical['composition']}",
        "Block encoding        : nested; U_Lcond = inner sinc^2 QSVT over U_X = PREP^dag SELECT_X PREP",
        "Accounting            : compiled Clifford+T component costs composed across the full sequence",
        "Scope                 : conditioned kernel L_cond^{-1}; C^{-1} b assumed prepared; logical gates before layout/QEC",
        f"Spatial dimensions    : {logical['dimensions']}",
        f"Qubits per dimension  : {logical['target_n']}",
        f"Grid points per axis  : {logical['grid_points_per_dimension']:,}",
        f"Outer polynomial degree: {metrics['polynomial_degree']} (P^MI, kappa = {metrics['kappa']:g})",
        f"Inner polynomial degree: {block['inner_degree']} (P_2d, d = {metrics['inner']['d']})",
        f"W applications        : {logical['num_w_applications']:,}",
        f"U_X calls per W       : {block['u_x_calls_per_w']}",
        f"U_X calls in total    : {logical['num_w_applications'] * block['u_x_calls_per_w']:,}",
        f"Projector rotations   : {logical['num_projector_rotations']:,}",
        f"Unique phase angles   : {projectors['num_unique_phase_angles']:,}",
        "",
        "--- entire sequence: logical resource counts ---",
        f"Logical qubits        : {logical['logical_qubits']:,}",
        f"Clifford gate count   : {logical['clifford_gate_count']:,}",
        f"T-count (T + T-dagger): {logical['t_count']:,}",
        f"Total gate count      : {logical['clifford_gate_count'] + logical['t_count']:,}",
        f"Total depth upper bound: {logical['total_depth']:,}",
        f"T-depth upper bound   : {logical['t_depth']:,}",
        f"Depth accounting      : {logical['depth_accounting']}",
        "",
        "--- contribution breakdown ---",
        f"{'Component':<31} {'Clifford gates':>16} {'T/T-dagger':>14} {'Depth bound':>15} {'T-depth bound':>15}",
    ]
    rows = [
        ("One W_cond", one_walk),
        ("All W applications", walks),
        ("All projector rotations", projectors),
        ("QFT + QFT^dag boundaries", boundary),
        ("Entire sequence", logical),
    ]
    for label, part in rows:
        lines.append(
            f"{label:<31} {part['clifford_gate_count']:>16,} {part['t_count']:>14,} "
            f"{part['total_depth']:>15,} {part['t_depth']:>15,}"
        )
    lines += [
        "",
        "--- anatomy of one W_cond (each part compiled separately) ---",
        f"{'Part':<31} {'Uses per W':>11} {'Clifford gates':>16} {'T/T-dagger':>12}",
    ]
    uses = {
        "PREP": 2 * block["u_x_calls_per_w"],
        "SELECT": block["u_x_calls_per_w"],
        "U_X": block["u_x_calls_per_w"],
        "U_Lcond (q-signed inner QSVT)": 1,
        "U_R": 1,
    }
    for label, part in one_walk["anatomy"].items():
        lines.append(
            f"{label:<31} {uses[label]:>11} {part['clifford_gate_count']:>16,} {part['t_count']:>12,}"
        )
    lines += [
        "",
        "--- registers ---",
        f"Logical qubits per W  : {one_walk['logical_qubits']} (includes the shared phase flag)",
        f"Signal ancillas m     : {logical['signal_ancillas']} = a + p + J, |J| = ceil(log2(q-1)) = {block['index_qubits']}",
        "Phase flag            : 1, shared by the inner and outer phase rotations",
        "Borrowed system qubits: dirty-ancilla MCX ladders (already included; not additional qubits)",
        f"SELECT                : {block['select_mcx_gates']} MCX gates with {block['select_mcx_controls']} controls on one shared ancilla a",
        f"Inner phase rotations : {synth['signal_rotations_per_w']} per W, each MCX({block['signal_mcx_controls']} -> flag) Rz MCX, sign set by q",
        f"Frequency bit order   : {block['fourier_bit_order']}",
        "",
        "--- full-sequence logical gate counts ---",
    ]
    lines += [f"{gate:<22}: {count:,}" for gate, count in sorted(logical["gate_counts"].items())]
    lines += [
        "",
        "--- approximation and synthesis errors ---",
        f"Kernel polynomial error     : {errors['polynomial_error']:.6e}  (max_k |P^MI(P_2d) - 1/(2 kappa sinc^2)|, all {spectrum['modes_checked']:,} distinct modes)",
        f"  inner P_2d vs sinc^2      : {spectrum['inner_polynomial_error']:.6e}  (bound {metrics['inner']['truncation_bound']:.6e})",
        f"  inner error after P^MI    : {spectrum['inner_error_after_inversion']:.6e}",
        f"  outer P^MI on L_cond      : {spectrum['outer_polynomial_error_on_spectrum']:.6e}",
        f"Phase numerical error       : {errors['phase_numerical_error']:.6e}  (outer + inner propagated)",
        f"Synthesis tolerance         : {errors['synthesis_tolerance']:.6e}",
        f"Measured synthesis bound    : {errors['measured_synthesis_error_bound']:.6e}",
        f"Conservative total bound    : {errors['conservative_total_error_bound']:.6e}",
        f"Synthesized rotations       : {logical['num_approximated_rotations']:,}",
        f"  per W / per QFT / outer   : {synth['rotations_per_w']} / {logical['approximated_rotation_breakdown']['per_qft']} / {logical['approximated_rotation_breakdown']['outer_projectors']}",
        f"  per W = {synth['prep_uses_per_w']} PREP uses x {synth['prep_rotations']} + {synth['signal_rotations_per_w']} inner phases",
        f"Tolerance per rotation      : {logical['eps_per_rotation']:.6e}",
        "",
        "--- inversion domain and spectral coverage ---",
        f"Kappa                 : {metrics['kappa']:g}",
        f"Polynomial epsilon    : {metrics['epsilon']:g}",
        f"Declared domain       : 1/kappa <= |lambda| <= 1 (|lambda| >= {1.0 / metrics['kappa']:.6e})",
        f"Encoded spectrum      : [{spectrum['encoded_min_eigenvalue']:.6f}, {spectrum['encoded_max_eigenvalue']:.6f}] (P_2d(2 m_k / N))",
        "Zero mode             : L_cond has eigenvalue 1 at k = 0; the singular mode lives in C, outside this kernel",
        f"Target spectral gap   : {logical['target_spectral_gap']:.6e}",
        f"Kappa for all modes   : {logical['minimum_kappa_for_full_nonzero_spectrum']:.6e} (covered by kappa = {metrics['kappa']:g})",
        "",
        "--- dense correctness checks at small verification size ---",
        f"Verification n        : {verification['verify_n']} ({verification['dense_qubits']} dense qubits; not the target size)",
        f"U_X block error       : {verification['u_x_block_error']:.6e}",
        f"U_X Hermiticity error : {verification['u_x_hermiticity_error']:.6e}",
        f"U_Lcond Herm-part error: {verification['u_lcond_hermitian_part_error']:.6e}  (Herm <0|U_Lcond|0> = P_2d(X))",
        f"W signal block error  : {verification['walk_signal_block_error']:.6e}  (<Pi|W|Pi> = -P_2d(X))",
        f"U' adjoint-branch error: {verification['u_prime_adjoint_branch_error']:.6e}",
        f"QFT vs bit-reversed DFT: {verification['qft_bit_reversed_dft_error']:.6e}",
        f"QSP block max-entry error: {verification['qsp_block_max_entry_error']:.6e}",
        f"QSP block spectral error: {verification['qsp_block_spectral_error']:.6e}",
        f"Position basis vs classical P^MI(L_cond,2d): {verification['position_vs_classical_polynomial_error']:.6e}",
        f"Position basis vs exact (1/2kappa) L_cond^-1: {verification['position_vs_exact_conditioned_kernel_error']:.6e}",
        "",
        f"Outer phase source    : {metrics['source_json']}",
        f"Inner phase source    : {metrics['inner_phases_json']}",
    ]
    return "\n".join(lines) + "\n"


def write_drawings(n_target: int, n_small: int, psi: list[float], folder: Path) -> list[Path]:
    """PREP and SELECT at the target and a readable size; boxed U_X/W structure."""
    paths = []
    target, small = draw_prep_select(n_target), draw_prep_select(n_small)
    for part in ("PREP", "SELECT"):
        path = folder / f"preconditioned_1d_{part}.txt"
        ensure_parent(path).write_text(
            f"##### target size q = {n_target} #####\n{target[part]}\n"
            f"##### readable instance q = {n_small} (same builder) #####\n{small[part]}",
            encoding="utf-8",
        )
        paths.append(path)
        try:
            figure = target[f"{part.lower()}_circuit"].draw(
                output="mpl", fold=-1, idle_wires=part == "SELECT", scale=1.2
            )
            png = folder / f"preconditioned_1d_{part}.png"
            figure.savefig(png, dpi=150, bbox_inches="tight")
            paths.append(png)
        except Exception as error:  # Optional; text drawings are authoritative.
            print(f"[note] {part} PNG skipped: {error}")

    lay = walk_layout(n_small)
    emitter = RotationEmitter()
    prep = build_prep(lay, emitter, logical=True)
    u_x = QuantumCircuit(lay.total, name="U_X")
    u_x.append(prep.to_gate(label="PREP"), range(lay.total))
    u_x.append(build_select(lay, logical=True).to_gate(label="SELECT_X"), range(lay.total))
    u_x.append(prep.inverse().to_gate(label="PREP_dg"), range(lay.total))
    boxed = u_x.to_gate(label="U_X")
    inner = QuantumCircuit(lay.total, name="U_Lcond")
    for index, angle in enumerate(reversed(psi)):
        append_signal_rotation(inner, lay, RotationEmitter(), angle, True, logical=True)
        if index + 1 < len(psi):
            inner.append(boxed, range(lay.total))
    walk = QuantumCircuit(lay.total, name="W_cond")
    walk.append(inner.to_gate(label="U_Lcond (q-signed)"), range(lay.total))
    walk.rz(-2.0 * phase_factor_angle(len(psi) - 1), lay.q)
    walk.x(lay.q)
    walk.append(build_reflection(lay, logical=True).to_gate(label="U_R"), range(lay.total))
    registers = (
        f"registers (q = {n_small} readable instance):\n"
        f"  q_0 = q (outer qubitization control), q_1 = a, q_2 = p, "
        f"q_3..q_{2 + lay.num_index} = J\n"
        f"  q_{3 + lay.num_index}..q_{2 + lay.num_index + n_small} = system "
        f"(frequency bit k_j on q_{3 + lay.num_index + n_small - 1}-j), q_{lay.flag} = phase flag\n\n"
    )
    structure = folder / "preconditioned_1d_U_X_W.txt"
    ensure_parent(structure).write_text(
        "=== U_X, inner sinc^2 QSVT and W_cond for the preconditioned 1D Laplacian ===\n"
        + registers
        + "U_X = PREP^dag . SELECT_X . PREP\n\n"
        + f"{u_x.draw(output='text', fold=-1)}\n\n"
        + f"U_Lcond: {len(psi)} phase rotations R(+-psi) around {len(psi) - 1} U_X calls\n\n"
        + f"{inner.draw(output='text', fold=250)}\n\n"
        + "W_cond = U_R . X_q . Rz(-2 arg c)_q . U_Lcond(q-signed)\n\n"
        + f"{walk.draw(output='text', fold=-1)}\n",
        encoding="utf-8",
    )
    paths.append(structure)
    return paths


# --- command line --------------------------------------------------------------
def load_inner(path: Path, d: int | None) -> dict:
    if path.is_file():
        meta = json.loads(path.read_text(encoding="utf-8"))
        if d is None or meta["d"] == d:
            return meta
    meta = sinc2_phases(5 if d is None else d)
    ensure_parent(path).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-d", "--verify-n", dest="verify_n", type=int, default=4)
    parser.add_argument("--target-n", type=int, default=20)
    parser.add_argument("--phases-json", type=Path, default=DEFAULT_OUTER_JSON)
    parser.add_argument("--inner-phases-json", type=Path, default=DEFAULT_INNER_JSON)
    parser.add_argument("--inner-d", type=int, default=None, help="Odd sinc^2 truncation index (default: phases file, else 5).")
    parser.add_argument("--synthesis-tol", type=float, default=DEFAULT_SYNTHESIS_TOL)
    parser.add_argument("--output-circuits-dir", type=Path, default=CIRCUITS_DIR)
    parser.add_argument(
        "--output-resource-metrics", type=Path,
        default=METRICS_DIR / "preconditioned_1d_inversion_resource_metrics.json",
    )
    args = parser.parse_args()
    report_path = args.output_resource_metrics.with_name(
        args.output_resource_metrics.stem.removesuffix("_metrics") + "_report.txt"
    )
    if not 0.0 < args.synthesis_tol < 1.0:
        raise ValueError("--synthesis-tol must satisfy 0 < tolerance < 1.")

    inner = load_inner(args.inner_phases_json, args.inner_d)
    psi = [float(x) for x in inner["psi"]]
    target_function, degree, effective_degree, parity, phases, _scale, cheb = load_phase_metadata(
        args.phases_json
    )
    meta = json.loads(args.phases_json.read_text(encoding="utf-8"))
    kappa, epsilon = float(meta["kappa"]), float(meta["epsilon"])
    parity = parity if parity is not None else effective_degree % 2
    phases = [float(phi) for phi in phases]

    print(f"Inner sinc^2 polynomial: degree {inner['degree']} ({len(psi)} phases)", flush=True)
    paths = write_drawings(args.target_n, args.verify_n, psi, args.output_circuits_dir)
    print("Circuit drawings: " + ", ".join(str(p) for p in paths), flush=True)

    print(f"Dense verification at n = {args.verify_n} ...", flush=True)
    verification = dense_checks(args.verify_n, psi, inner["d"], phases, cheb, parity, kappa)
    for key, value in verification.items():
        print(f"  {key:<44}: {value}")

    print(f"Spectral analysis at n = {args.target_n} ...", flush=True)
    spectrum = spectral_summary(args.target_n, inner["d"], cheb, kappa, inner["phase_vs_poly_max_abs_error"])

    print(f"Logical Clifford+T resources at n = {args.target_n} ...", flush=True)
    started = time.monotonic()
    logical = estimate_logical_resources(args.target_n, psi, phases, args.synthesis_tol)
    logical["target_spectral_gap"] = spectrum["encoded_min_eigenvalue"]
    logical["minimum_kappa_for_full_nonzero_spectrum"] = 1.0 / spectrum["encoded_min_eigenvalue"]
    print(f"  composed in {time.monotonic() - started:.1f} s", flush=True)

    phase_numerical = float(meta.get("phase_vs_poly_max_abs_error", 0.0)) + spectrum[
        "inner_phase_error_after_inversion_bound"
    ]
    errors = {
        "polynomial_error": spectrum["kernel_error"],
        "inner_polynomial_error": spectrum["inner_polynomial_error"],
        "inner_error_after_inversion": spectrum["inner_error_after_inversion"],
        "outer_polynomial_error": spectrum["outer_polynomial_error_on_spectrum"],
        "phase_numerical_error": phase_numerical,
        "synthesis_tolerance": args.synthesis_tol,
        "measured_synthesis_error_bound": logical["measured_synthesis_error_bound"],
        "conservative_total_error_bound": spectrum["kernel_error"] + phase_numerical + args.synthesis_tol,
    }
    metrics = {
        "scope": "full_qsp_logical_resource_estimate",
        "case": CASE,
        "case_title": CASE_TITLE,
        "source_json": str(args.phases_json),
        "inner_phases_json": str(args.inner_phases_json),
        "target_function": target_function,
        "epsilon": epsilon,
        "kappa": kappa,
        "polynomial_degree": degree,
        "effective_degree": effective_degree,
        "parity": parity,
        "inner": {key: inner[key] for key in (
            "d", "degree", "truncation_bound", "poly_vs_sinc2_max_abs_error",
            "phase_vs_poly_max_abs_error", "minimum_on_unit_interval",
        )},
        "error_summary": errors,
        "spectral_summary": spectrum,
        "verification": verification,
        "logical_resources": logical,
    }
    ensure_parent(args.output_resource_metrics).write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    report = render_resource_report(metrics)
    ensure_parent(report_path).write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved: {args.output_resource_metrics} and {report_path}")


if __name__ == "__main__":
    main()
