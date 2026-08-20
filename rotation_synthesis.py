"""Explicit single-qubit Clifford+T synthesis through pyLIQTR.

The project uses Qiskit circuits, while pyLIQTR's public circuit transformer
uses Cirq.  This adapter calls pyLIQTR's underlying ring-approximation and
exact-decomposition routines directly and asks them to return Qiskit gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
import math
import random

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    HGate,
    RYGate,
    SGate,
    SdgGate,
    TGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.quantum_info import Operator


@dataclass(frozen=True)
class RotationSynthesis:
    """An explicit Clifford+T approximation of one numeric Ry rotation."""

    circuit: QuantumCircuit
    backend_version: str
    epsilon: float
    projective_error: float
    raw_phase_offset: float
    gate_counts: dict[str, int]
    t_count: int
    clifford_count: int
    depth: int
    t_depth: int


def projective_operator_norm_error(
    target: np.ndarray, actual: np.ndarray
) -> tuple[float, float]:
    """Return min_phi ||actual - exp(i phi) target||_2 and the best phi."""
    target = np.asarray(target, dtype=np.complex128)
    actual = np.asarray(actual, dtype=np.complex128)
    if target.shape != (2, 2) or actual.shape != (2, 2):
        raise ValueError("Expected two 2x2 single-qubit operators.")

    relative = target.conj().T @ actual
    overlap = np.trace(relative)
    if abs(overlap) > 32 * np.finfo(float).eps:
        phase = overlap / abs(overlap)
    else:
        phase = np.sqrt(np.linalg.det(relative))
        phase /= abs(phase)

    error = np.linalg.norm(actual - phase * target, ord=2)
    return float(error), float(np.angle(phase))


@lru_cache(maxsize=None)
def _synthesize_ry_cached(theta_text: str, epsilon_text: str) -> RotationSynthesis:
    try:
        import gmpy2
        from gmpy2 import mpfr
        from pyLIQTR.gate_decomp.exact_decomp import exact_decomp
        from pyLIQTR.gate_decomp.gate_approximation import get_ring_elts_direct
    except ImportError as exc:
        raise RuntimeError(
            "Explicit 3D rotation synthesis requires pyLIQTR and gmpy2. "
            "Install requirements.txt with Python 3.8--3.12.2."
        ) from exc

    theta_mp = mpfr(theta_text)
    epsilon_mp = mpfr(epsilon_text)
    if not gmpy2.is_finite(theta_mp):
        raise ValueError("Rotation angle must be finite.")
    if not gmpy2.is_finite(epsilon_mp) or not 0 < epsilon_mp < 1:
        raise ValueError("Rotation tolerance must satisfy 0 < epsilon < 1.")

    # pyLIQTR approximates Rz(theta) over D[omega], then exactly decomposes the
    # approximation.  exact_decomp expects return objects in this exact order.
    # pyLIQTR's number-theory solver uses Python randomness. Save and restore
    # the caller's state while fixing this synthesis result for reproducibility.
    random_state = random.getstate()
    random.seed(f"pyLIQTR-Ry:{theta_text}:{epsilon_text}")
    try:
        u, t, k = get_ring_elts_direct(theta_mp, prec=0, eps=epsilon_mp)
    finally:
        random.setstate(random_state)
    rz_gates, reported_t_count = exact_decomp(
        u,
        t,
        k,
        (
            SGate(),
            SdgGate(),
            HGate(),
            XGate(),
            YGate(),
            ZGate(),
            TGate(),
        ),
        circuit_order=True,
    )

    circuit = QuantumCircuit(1, name="Ry_Clifford_T")
    # Circuit order Sdg-H-Rz-H-S gives matrix order S-H-Rz-H-Sdg = Ry.
    circuit.sdg(0)
    circuit.h(0)
    for gate in rz_gates:
        circuit.append(gate, [0])
    circuit.h(0)
    circuit.s(0)

    target = Operator(RYGate(float(theta_mp))).data
    actual = Operator(circuit).data
    error, phase_offset = projective_operator_norm_error(target, actual)

    # pyLIQTR's SO(3)-based decomposition is defined up to global phase.
    # Restore that phase so block-encoding matrix comparisons remain meaningful.
    circuit.global_phase -= phase_offset

    counts = {str(name): int(count) for name, count in circuit.count_ops().items()}
    t_count = counts.get("t", 0) + counts.get("tdg", 0)
    if t_count != int(reported_t_count):
        raise RuntimeError(
            "pyLIQTR's reported T-count does not match its emitted gate sequence."
        )
    clifford_count = int(sum(counts.values()) - t_count)
    t_depth = int(
        circuit.depth(
            filter_function=lambda instr: instr.operation.name in ("t", "tdg")
        )
    )

    return RotationSynthesis(
        circuit=circuit,
        backend_version=version("pyLIQTR"),
        epsilon=float(epsilon_mp),
        projective_error=error,
        raw_phase_offset=phase_offset,
        gate_counts=counts,
        t_count=t_count,
        clifford_count=clifford_count,
        depth=int(circuit.depth()),
        t_depth=t_depth,
    )


def synthesize_ry(theta: float, epsilon: float) -> RotationSynthesis:
    """Synthesize Ry(theta), returning a fresh copy of the cached circuit."""
    if not math.isfinite(theta):
        raise ValueError("Rotation angle must be finite.")
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("Rotation tolerance must satisfy 0 < epsilon < 1.")

    result = _synthesize_ry_cached(repr(theta), repr(epsilon))
    return RotationSynthesis(
        circuit=result.circuit.copy(),
        backend_version=result.backend_version,
        epsilon=result.epsilon,
        projective_error=result.projective_error,
        raw_phase_offset=result.raw_phase_offset,
        gate_counts=dict(result.gate_counts),
        t_count=result.t_count,
        clifford_count=result.clifford_count,
        depth=result.depth,
        t_depth=result.t_depth,
    )
