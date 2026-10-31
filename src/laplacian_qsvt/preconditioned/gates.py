"""Circuit primitives shared by U_X, the inner QSVT and the QFT boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit.synthesis.multi_controlled import synth_mcx_n_dirty_i15

from ..synthesis.rotation_synthesis import synthesize_rz

ANGLE_TOL = 1e-12
_EXACT_RZ = {
    0: (),
    1: ("t",),
    2: ("s",),
    3: ("s", "t"),
    4: ("z",),
    5: ("z", "t"),
    6: ("sdg",),
    7: ("tdg",),
}


def eighth_turns(theta: float) -> int | None:
    """Return k when theta = k*pi/4, i.e. Rz(theta) is exactly Clifford+T."""
    k = round(theta / (math.pi / 4))
    return k if abs(theta - k * math.pi / 4) < ANGLE_TOL else None


@dataclass
class RotationEmitter:
    """Emit Rz/Ry as ideal gates or as explicit Clifford+T sequences.

    ``synthesis_eps=None`` keeps native rotations (dense checks and drawings).
    Otherwise angles k*pi/4 are emitted exactly and every other angle uses
    pyLIQTR at ``synthesis_eps``. Each approximated angle is recorded once per
    emission so the caller can weight its error by circuit reuse.
    """

    synthesis_eps: float | None = None
    records: list[dict] = field(default_factory=list)

    def rz(self, qc: QuantumCircuit, theta: float, qubit: int) -> None:
        k = eighth_turns(theta)
        if self.synthesis_eps is None:
            qc.rz(theta, qubit)
            if k is None:
                self.records.append({"angle": float(theta), "projective_error": 0.0})
            return
        if k is not None:
            for name in _EXACT_RZ[k % 8]:
                getattr(qc, name)(qubit)
            qc.global_phase -= k * math.pi / 8
            return
        synthesis = synthesize_rz(float(theta), self.synthesis_eps)
        qc.compose(synthesis.circuit, qubits=[qubit], inplace=True)
        self.records.append(
            {
                "angle": float(theta),
                "epsilon": synthesis.epsilon,
                "projective_error": synthesis.projective_error,
                "t_count": synthesis.t_count,
                "clifford_count": synthesis.clifford_count,
            }
        )

    def ry(self, qc: QuantumCircuit, theta: float, qubit: int) -> None:
        if self.synthesis_eps is None:
            qc.ry(theta, qubit)
            if eighth_turns(theta) is None:
                self.records.append({"angle": float(theta), "projective_error": 0.0})
            return
        # Circuit order Sdg-H-Rz-H-S gives matrix order S-H-Rz-H-Sdg = Ry.
        qc.sdg(qubit)
        qc.h(qubit)
        self.rz(qc, theta, qubit)
        qc.h(qubit)
        qc.s(qubit)


def append_mcx(
    qc: QuantumCircuit,
    controls: list[int],
    target: int,
    dirty: list[int],
    ctrl_state: list[int] | None = None,
    logical: bool = False,
) -> None:
    """Multi-controlled X with optional open controls.

    ``logical=True`` appends one MCXGate for readable drawings. Otherwise
    three or more controls use Iten et al.'s exact dirty-ancilla ladder,
    borrowing ``len(controls) - 2`` qubits from ``dirty``.
    """
    states = [1] * len(controls) if ctrl_state is None else list(ctrl_state)
    if logical:
        value = sum(bit << i for i, bit in enumerate(states))
        qc.append(MCXGate(len(controls), ctrl_state=value), list(controls) + [target])
        return
    opened = [c for c, bit in zip(controls, states) if bit == 0]
    if opened:
        qc.x(opened)
    k = len(controls)
    if k == 0:
        qc.x(target)
    elif k == 1:
        qc.cx(controls[0], target)
    elif k == 2:
        qc.ccx(controls[0], controls[1], target)
    else:
        borrowed = [d for d in dirty if d not in controls and d != target][: k - 2]
        if len(borrowed) < k - 2:
            raise ValueError(f"MCX with {k} controls needs {k - 2} dirty qubits.")
        if k == 3:
            # Qiskit's 3-control form uses pi/8 phases; four Toffolis stay Clifford+T.
            c0, c1, c2 = controls
            d = borrowed[0]
            qc.ccx(c2, d, target)
            qc.ccx(c0, c1, d)
            qc.ccx(c2, d, target)
            qc.ccx(c0, c1, d)
        else:
            qc.compose(
                synth_mcx_n_dirty_i15(k),
                qubits=list(controls) + [target] + borrowed,
                inplace=True,
            )
    if opened:
        qc.x(opened)


def gray_code_angles(angles: list[float]) -> list[float]:
    """Mottonen transform: Gray-code rotation angles of a uniformly controlled Ry."""
    size = len(angles)
    gray = [i ^ (i >> 1) for i in range(size)]
    return [
        sum(
            (-1) ** bin(j & gray[i]).count("1") * angles[j] for j in range(size)
        )
        / size
        for i in range(size)
    ]


def append_multiplexed_ry(
    qc: QuantumCircuit,
    emitter: RotationEmitter,
    target: int,
    controls: list[int],
    angles: list[float],
) -> None:
    """Apply Ry(angles[c]) to ``target`` when ``controls`` (little-endian) read c."""
    if not controls:
        if abs(angles[0]) > ANGLE_TOL:
            emitter.ry(qc, angles[0], target)
        return
    size = 2 ** len(controls)
    gray = [i ^ (i >> 1) for i in range(size)]
    for i, theta in enumerate(gray_code_angles(angles)):
        if abs(theta) > ANGLE_TOL:
            emitter.ry(qc, theta, target)
        changed = gray[i] ^ gray[(i + 1) % size]
        qc.cx(controls[changed.bit_length() - 1], target)


def _fill_dont_cares(table: list[float | None]) -> list[float]:
    """Choose unconstrained multiplexor angles that zero Gray-code rotations."""
    size = len(table)
    free = [i for i, value in enumerate(table) if value is None]
    fixed = np.array([0.0 if value is None else value for value in table])
    if not free:
        return fixed.tolist()
    gray = [i ^ (i >> 1) for i in range(size)]
    sign = np.array(
        [[(-1) ** bin(j & gray[i]).count("1") for j in range(size)] for i in range(size)],
        dtype=float,
    )
    zeroed: list[int] = []
    solution = np.zeros(len(free))
    for row in reversed(range(size)):
        rows = zeroed + [row]
        a = sign[np.ix_(rows, free)]
        b = -sign[rows] @ fixed
        x, *_ = np.linalg.lstsq(a, b, rcond=None)
        if np.max(np.abs(a @ x - b)) < 1e-12:
            zeroed, solution = rows, x
    fixed[free] = solution
    return fixed.tolist()


def state_preparation_tree(
    weights: list[float], num_bits: int
) -> list[tuple[int, list[int], list[float]]]:
    """Pruned Grover-Rudolph tree for a state with real amplitudes sqrt(weights).

    Returns (target bit, control bits, angles) levels from the most significant
    bit down. Each level keeps the smallest set of higher bits its angles
    depend on; unreachable prefixes are free and chosen to cancel rotations.
    """
    if len(weights) != 2**num_bits:
        raise ValueError("Need one weight per basis state.")
    levels = []
    for bit in reversed(range(num_bits)):
        higher = list(range(bit + 1, num_bits))
        table: list[float | None] = []
        for prefix in range(2 ** len(higher)):
            base = prefix << (bit + 1)
            w0 = sum(weights[base + low] for low in range(2**bit))
            w1 = sum(weights[base + (1 << bit) + low] for low in range(2**bit))
            table.append(
                None if w0 + w1 <= 0 else 2.0 * math.atan2(math.sqrt(w1), math.sqrt(w0))
            )
        chosen = None
        for size in range(len(higher) + 1):
            for subset in combinations(range(len(higher)), size):
                reduced: dict[int, float] = {}
                consistent = True
                for prefix, angle in enumerate(table):
                    if angle is None:
                        continue
                    key = sum(((prefix >> s) & 1) << i for i, s in enumerate(subset))
                    if key in reduced and abs(reduced[key] - angle) > ANGLE_TOL:
                        consistent = False
                        break
                    reduced[key] = angle
                if consistent:
                    chosen = (subset, reduced)
                    break
            if chosen is not None:
                break
        subset, reduced = chosen
        angles = _fill_dont_cares([reduced.get(i) for i in range(2 ** len(subset))])
        levels.append((bit, [higher[s] for s in subset], angles))
    return levels
