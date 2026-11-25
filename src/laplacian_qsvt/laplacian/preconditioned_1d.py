"""Explicit Fourier-basis block encodings for the conditioned 1D Laplacian.

All signal ancillas precede the system register (Qiskit little endian).  The
frequency SELECT uses one shared flag: its two MSB branches are implemented
explicitly, including the padding branch.  No system-sized matrix is formed.

The real polynomial is extracted coherently.  In particular, taking the
imaginary part of a simulated matrix is never substituted for a circuit.
"""
from __future__ import annotations

import contextlib
import io
import math
import os

import numpy as np
from numpy.polynomial import chebyshev as C

from ..paths import MPLCONFIG_DIR

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
from pyqsp import angle_sequence
from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import RYGate, XGate
from qiskit.quantum_info import Operator


def folded_layout(q: int) -> dict:
    if q < 2:
        raise ValueError("The folded-frequency construction requires q >= 2.")
    selector = (q - 1).bit_length()
    return {"system_qubits": q, "selector_qubits": selector,
            "ancilla_count": selector + 1, "flag": 0,
            "selector": list(range(1, selector + 1)),
            "system": list(range(selector + 1, selector + 1 + q))}


def _positive_state_preparation(probabilities: np.ndarray) -> QuantumCircuit:
    """Real R_y tree preparing nonnegative amplitudes on a small selector."""
    count = len(probabilities).bit_length() - 1
    qc = QuantumCircuit(count, name="PREP_frequency_weights")
    # Prepare MSB first; each rotation is conditional on its existing prefix.
    for target in reversed(range(count)):
        controls = list(range(target + 1, count))
        for prefix in range(1 << len(controls)):
            lo = prefix << (target + 1)
            half = 1 << target
            zero = float(np.sum(probabilities[lo:lo + half]))
            one = float(np.sum(probabilities[lo + half:lo + 2 * half]))
            if zero + one == 0.0:
                continue
            angle = 2.0 * math.atan2(math.sqrt(one), math.sqrt(zero))
            if not controls:
                qc.ry(angle, target)
            else:
                qc.append(RYGate(angle).control(len(controls), ctrl_state=prefix),
                          controls + [target])
    return qc


def build_folded_frequency_block_encoding(q: int) -> QuantumCircuit:
    """Hermitian unitary with zero-ancilla block diag(2 min(k,N-k)/N)."""
    layout = folded_layout(q)
    flag, selector, system = layout["flag"], layout["selector"], layout["system"]
    weights = np.zeros(1 << len(selector))
    for j in range(q - 1):
        weights[j] = 2.0 ** (j - q + 1)
    weights[q - 1] = 2.0 ** (1 - q)
    prep = _positive_state_preparation(weights)
    qc = QuantumCircuit(q + layout["ancilla_count"], name="U_folded_frequency")
    qc.compose(prep, selector, inplace=True)
    b = system[-1]
    for j in range(q - 1):
        controls = selector + [system[j], b]
        # b=0: flip when k_j=0, so <0|SELECT|0> = k_j.
        # b=1: flip when k_j=1, so <0|SELECT|0> = 1-k_j.
        for bit in (0, 1):
            state = j | (bit << len(selector)) | (bit << (len(selector) + 1))
            qc.append(XGate().control(len(controls), ctrl_state=state),
                      controls + [flag])
    # Padding contributes zero to Y and identity to I-Y.
    qc.append(XGate().control(len(selector) + 1, ctrl_state=q - 1),
              selector + [b, flag])
    qc.compose(prep.inverse(), selector, inplace=True)
    qc.metadata = {**layout, "basis": "Fourier", "normalization": 1.0,
                   "zero_mode_extension": "theta(0)=0", "hermitian": True}
    return qc


def make_inner_polynomial(epsilon: float = 1e-7,
                          safety_scale: float = 1.0 - 1e-6) -> dict:
    """Bounded even lower Taylor polynomial and symmetric-QSP phases.

    The alternating-series bound is uniform on [-1,1].  Odd truncation
    indices give 0 <= P <= sinc^2(pi*x/2) <= 1.  Safety scaling is recorded
    as block-encoding normalization, rather than hidden approximation error.
    """
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must lie strictly between zero and one.")
    if not 0 < safety_scale < 1:
        raise ValueError("safety_scale must lie strictly between zero and one.")
    truncation = 1
    while True:
        log_bound = (math.log(2) + (2 * truncation + 2) * math.log(math.pi)
                     - math.lgamma(2 * truncation + 5))
        bound = math.exp(log_bound)
        if bound <= epsilon:
            break
        truncation += 2
        if truncation > 99:
            raise ValueError("Requested inner tolerance is unsupported.")
    power = np.zeros(2 * truncation + 1)
    for j in range(truncation + 1):
        power[2 * j] = 2 * (-1) ** j * math.pi ** (2 * j) / math.factorial(2 * j + 2)
    coeffs = C.poly2cheb(power) * safety_scale
    with contextlib.redirect_stdout(io.StringIO()):
        result = angle_sequence.QuantumSignalProcessingPhases(
            coeffs, method="sym_qsp", chebyshev_basis=True)
    phases = np.asarray(result[0] if isinstance(result, tuple) else result, dtype=float)
    grid = np.linspace(-1, 1, 4001)
    response = angle_sequence.ComputeQSPResponse(
        grid, phases, signal_operator="Wx", sym_qsp=True)
    response_error = float(np.max(np.abs(np.imag(response["pdat"]) - C.chebval(grid, coeffs))))
    return {"target_function": "sinc_squared_pi_x_over_2", "epsilon": epsilon,
            "truncation_index": truncation, "degree": 2 * truncation,
            "power_coefficients_unscaled": power.tolist(), "cheb_coeffs": coeffs.tolist(),
            "phases": phases.tolist(), "parity": 0, "safety_scale": safety_scale,
            "normalization": 1.0 / safety_scale,
            "uniform_polynomial_error_bound_unscaled": bound,
            "phase_vs_poly_sampled_max_abs_error": response_error,
            "phase_validation_grid_points": len(grid),
            "encoded_spectral_lower_bound": safety_scale * (4 / math.pi ** 2 - bound),
            "encoded_spectral_upper_bound": safety_scale}


def build_signed_projector_rotation(phi: float, signal_ancillas: int,
                                    system_qubits: int) -> QuantumCircuit:
    """exp(i phi Z_e (2 Pi-I)), with e=0, clean predicate f=1.

    Signal ancillas are 2..m+1 and system qubits follow them.  Computing the
    predicate into f permits a single Rz and exact reversible Boolean gates.
    """
    m = signal_ancillas
    if m < 1:
        raise ValueError("At least one signal ancilla is required.")
    qc = QuantumCircuit(m + system_qubits + 2, name="signed_R_Pi")
    predicate = list(range(2, 2 + m))
    qc.x(predicate)
    qc.mcx(predicate, 1)
    qc.x(predicate)
    qc.cx(0, 1)
    qc.rz(2.0 * phi, 1)
    qc.cx(0, 1)
    qc.x(predicate)
    qc.mcx(predicate, 1)
    qc.x(predicate)
    return qc


def reflection_form_phases(phases: list[float]) -> list[float]:
    """Convert Wx phases to Hermitian-reflection queries (global i**degree)."""
    if len(phases) < 2:
        raise ValueError("A nonconstant polynomial is required.")
    result = [float(value) - math.pi / 2 for value in phases]
    result[0] += math.pi / 4
    result[-1] += math.pi / 4
    return result


def build_real_polynomial_block_encoding(signal: QuantumCircuit,
                                         signal_ancillas: int,
                                         phases: list[float]) -> QuantumCircuit:
    """Coherently encode the real sym_qsp polynomial of a real Hermitian signal.

    Wx = i R(-pi/4) U R(-pi/4).  Thus symmetric phases give a symmetric
    complex V.  The extraction circuit implements [[Im V, Re V],
    [Re V, -Im V]], a real Hermitian unitary.  One extra branch and one clean
    phase-predicate workspace become part of the returned signal register.
    """
    if not np.allclose(phases, list(reversed(phases)), atol=1e-12, rtol=0):
        raise ValueError("Hermitian real extraction requires palindromic phases.")
    system_qubits = signal.num_qubits - signal_ancillas
    qc = QuantumCircuit(signal.num_qubits + 2, name="U_real_polynomial")
    qc.s(0)
    qc.h(0)
    qc.rz(-(len(phases) - 1) * math.pi, 0)
    converted = reflection_form_phases(phases)
    for index, phi in enumerate(converted):
        qc.compose(build_signed_projector_rotation(phi, signal_ancillas, system_qubits),
                   inplace=True)
        if index + 1 < len(converted):
            qc.compose(signal, list(range(2, signal.num_qubits + 2)), inplace=True)
    qc.h(0)
    qc.sdg(0)
    qc.x(0)
    qc.metadata = {"ancilla_count": signal_ancillas + 2,
                   "system_qubits": system_qubits, "basis": "Fourier",
                   "degree": len(phases) - 1, "hermitian": True,
                   "coherent_real_extraction": True,
                   "signal_primitive": "Hermitian reflection-form query"}
    return qc


def build_preconditioned_block_encoding(q: int, polynomial: dict | None = None) -> QuantumCircuit:
    polynomial = make_inner_polynomial() if polynomial is None else polynomial
    folded = build_folded_frequency_block_encoding(q)
    qc = build_real_polynomial_block_encoding(
        folded, folded.metadata["ancilla_count"], polynomial["phases"])
    qc.name = "U_conditioned_1D"
    qc.metadata.update({"normalization": polynomial["normalization"],
                        "zero_mode_extension": "continuous sinc squared value 1",
                        "inner_polynomial": polynomial})
    return qc


def build_preconditioned_walk(q: int, polynomial: dict | None = None) -> QuantumCircuit:
    """Hermitian signal query used by the reflection-form outer QSP circuit.

    This is U_cond itself.  Projector phase shifts replace the conventional
    reflection-times-U walk representation without adding generic controls.
    """
    qc = build_preconditioned_block_encoding(q, polynomial)
    qc.name = "W_conditioned_1D_reflection_form"
    return qc


def exact_conditioned_eigenvalues(q: int) -> np.ndarray:
    frequencies = np.arange(1 << q)
    folded = np.minimum(frequencies, (1 << q) - frequencies)
    return np.sinc(folded / (1 << q)) ** 2


def frequency_sector_circuit(circuit: QuantumCircuit, q: int,
                             frequency: int) -> QuantumCircuit:
    """Restrict the actual circuit to one unchanged computational frequency.

    This removes known system controls; it does not substitute the desired
    eigenvalue for a circuit.  Any system-targeting operation is rejected.
    """
    ancillas = circuit.num_qubits - q
    if not 0 <= frequency < (1 << q):
        raise ValueError("Frequency lies outside the system register.")
    result = QuantumCircuit(ancillas, name=f"{circuit.name}_k{frequency}")
    result.global_phase = circuit.global_phase
    for instruction in circuit.data:
        operation = instruction.operation
        wires = [circuit.find_bit(bit).index for bit in instruction.qubits]
        if all(wire < ancillas for wire in wires):
            result.append(operation, wires)
            continue
        if not isinstance(operation, ControlledGate):
            raise ValueError(f"System operation {operation.name} is not a fixed control.")
        controls = wires[:operation.num_ctrl_qubits]
        targets = wires[operation.num_ctrl_qubits:]
        if any(wire >= ancillas for wire in targets):
            raise ValueError("The frequency-sector shortcut requires unchanged system bits.")
        remaining, state = [], 0
        enabled = True
        for position, wire in enumerate(controls):
            expected = (operation.ctrl_state >> position) & 1
            if wire >= ancillas:
                enabled &= ((frequency >> (wire - ancillas)) & 1) == expected
            else:
                state |= expected << len(remaining)
                remaining.append(wire)
        if not enabled:
            continue
        reduced = operation.base_gate
        if remaining:
            reduced = reduced.control(len(remaining), ctrl_state=state)
        result.append(reduced, remaining + targets)
    return result


def verify_conditioned_inverse(q: int, inner: dict, outerphase: dict | list[float],
                               cheb_coeffs: list[float] | None = None) -> dict:
    """Validate actual small-grid gates, including coherent outer extraction.

    Matrix propagation reuses the compiled ideal component matrices; the
    long outer circuit is not materialized.  Matrix entries are read directly
    as complex amplitudes, with no post hoc real/imaginary-part extraction.
    """
    if not 2 <= q <= 6:
        raise ValueError("Circuit verification is intended for 2 <= q <= 6.")
    folded = build_folded_frequency_block_encoding(q)
    signal = build_preconditioned_block_encoding(q, inner)
    m = signal.metadata["ancilla_count"]
    if isinstance(outerphase, dict):
        phases = list(outerphase["phases"])
        coefficients = outerphase["cheb_coeffs"]
    else:
        phases = list(outerphase)
        if cheb_coeffs is None:
            raise ValueError("Outer polynomial coefficients are required.")
        coefficients = cheb_coeffs
    converted = reflection_form_phases(phases)
    rotations = [Operator(build_signed_projector_rotation(phi, m, 0)).data
                 for phi in converted]
    start = QuantumCircuit(m + 2)
    start.s(0)
    start.h(0)
    start.rz(-(len(phases) - 1) * math.pi, 0)
    finish = QuantumCircuit(m + 2)
    finish.h(0)
    finish.sdg(0)
    finish.x(0)
    initial = Operator(start).data[:, 0]
    finish_matrix = Operator(finish).data
    failures = {"folded_signal_max_error": 0.0, "inner_signal_vs_polynomial_max_error": 0.0,
                "inner_unitary_imaginary_max": 0.0, "inner_hermiticity_max_error": 0.0,
                "inner_involution_max_error": 0.0, "outer_signal_vs_polynomial_max_error": 0.0}
    actual_inner, actual_outer = [], []
    for k in range(1 << q):
        theta = 2 * min(k, (1 << q) - k) / (1 << q)
        folded_matrix = Operator(frequency_sector_circuit(folded, q, k)).data
        matrix = Operator(frequency_sector_circuit(signal, q, k)).data
        value = matrix[0, 0]
        target = C.chebval(theta, inner["cheb_coeffs"])
        failures["folded_signal_max_error"] = max(failures["folded_signal_max_error"],
                                                   abs(folded_matrix[0, 0] - theta))
        failures["inner_signal_vs_polynomial_max_error"] = max(
            failures["inner_signal_vs_polynomial_max_error"], abs(value - target))
        failures["inner_unitary_imaginary_max"] = max(
            failures["inner_unitary_imaginary_max"], float(np.max(np.abs(matrix.imag))))
        failures["inner_hermiticity_max_error"] = max(
            failures["inner_hermiticity_max_error"], float(np.max(np.abs(matrix - matrix.conj().T))))
        failures["inner_involution_max_error"] = max(
            failures["inner_involution_max_error"], float(np.max(np.abs(matrix @ matrix - np.eye(len(matrix))))))
        embedded = np.kron(matrix, np.eye(4))
        state = initial.copy()
        for j, rotation in enumerate(rotations):
            state = rotation @ state
            if j + 1 < len(rotations):
                state = embedded @ state
        response = (finish_matrix @ state)[0]
        failures["outer_signal_vs_polynomial_max_error"] = max(
            failures["outer_signal_vs_polynomial_max_error"],
            abs(response - C.chebval(value, coefficients)))
        actual_inner.append(float(value.real))
        actual_outer.append(float(response.real))
    passed = max(failures.values()) < 1e-8
    return {"verification_q": q, "grid_points": 1 << q, "passed": bool(passed),
            "method": "actual gate restriction to each invariant frequency sector",
            "coherent_real_extraction_verified": bool(passed),
            **{key: float(value) for key, value in failures.items()},
            "inner_signal_eigenvalues": actual_inner, "outer_signal_eigenvalues": actual_outer,
            "zero_mode": "continuous extension tested; application inversion excludes zero mode"}
