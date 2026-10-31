"""Inner sinc^2 QSVT, the outer qubitization walk W_cond and the QFT boundaries.

Inner block encoding (ancillas a, p, J; Pi_X = |0^m><0^m|):
    U_in(psi) = R(psi_0) U_X R(psi_1) U_X ... U_X R(psi_D),  R(psi) = exp(i psi (2 Pi_X - I))
    U_Lcond   = c U_in,  c = -i^{D+1},  Herm <0^m| U_Lcond |0^m> = P_2d(X)

Each R(psi) uses the resident phase flag: MCX(0^m -> flag) Rz(2 psi) MCX.
Because U_X is Hermitian and psi is symmetric, U_in^dag = U_in(-psi), so

    U' = |0><0|_q c U_in(psi) + |1><1|_q c* U_in(-psi)
       = Rz(-2 arg c)_q . [U_in with every Rz sign flipped by CX(q -> flag)].

No gate of U_X needs a q control. The outer walk is unchanged:
    W = U_R X_q U',  U_R = I - 2|Pi><Pi|,  |Pi> = |+>_q |0^m>.
"""

from __future__ import annotations

import cmath
import math

from qiskit import QuantumCircuit

from .gates import RotationEmitter, append_mcx
from .sinc2 import block_phase_factor
from .u_x import Layout, build_prep, build_u_x


def append_signal_rotation(
    qc: QuantumCircuit,
    lay: Layout,
    emitter: RotationEmitter,
    psi: float,
    sign_control: bool,
    logical: bool = False,
) -> None:
    """R(psi) on Pi_X, or R(+-psi) selected by q when ``sign_control``."""
    dirty = list(lay.system)
    append_mcx(qc, lay.anc, lay.flag, dirty, [0] * lay.m, logical)
    if sign_control:
        qc.cx(lay.q, lay.flag)
    emitter.rz(qc, 2.0 * psi, lay.flag)
    if sign_control:
        qc.cx(lay.q, lay.flag)
    append_mcx(qc, lay.anc, lay.flag, dirty, [0] * lay.m, logical)


def build_inner_qsvt(
    lay: Layout,
    psi: list[float],
    emitter: RotationEmitter,
    sign_control: bool = False,
    logical: bool = False,
) -> tuple[QuantumCircuit, dict]:
    """U_in(psi) and its rotation inventory (PREP is built once and reused)."""
    prep_emitter = RotationEmitter(emitter.synthesis_eps)
    prep = build_prep(lay, prep_emitter, logical)
    u_x = build_u_x(lay, prep_emitter, logical, prep=prep)
    signal_emitter = RotationEmitter(emitter.synthesis_eps)
    qc = lay.circuit("U_Lcond")
    # Operator order R(psi_0) U_X ... R(psi_D): the last phase acts first in time.
    for index, angle in enumerate(reversed(psi)):
        append_signal_rotation(qc, lay, signal_emitter, angle, sign_control, logical)
        if index + 1 < len(psi):
            qc.compose(u_x, inplace=True)
    inventory = {
        "num_u_x": len(psi) - 1,
        "prep_uses": 2 * (len(psi) - 1),
        "prep_records": prep_emitter.records,
        "signal_records": signal_emitter.records,
    }
    emitter.records.extend(
        prep_emitter.records * inventory["prep_uses"] + signal_emitter.records
    )
    return qc, inventory


def build_reflection(lay: Layout, logical: bool = False) -> QuantumCircuit:
    """U_R = I - 2|Pi><Pi| on q + (a, p, J), as a dirty-ancilla MCZ."""
    qc = lay.circuit("U_R")
    reflected = [lay.q, *lay.anc]
    qc.h(lay.q)
    qc.x(reflected)
    qc.h(lay.q)
    append_mcx(qc, lay.anc, lay.q, list(lay.system), None, logical)
    qc.h(lay.q)
    qc.x(reflected)
    qc.h(lay.q)
    return qc


def build_walk(
    lay: Layout,
    psi: list[float],
    emitter: RotationEmitter,
    logical: bool = False,
) -> tuple[QuantumCircuit, dict]:
    """W_cond = U_R X_q U' on the full resident register (phase flag included)."""
    inner, inventory = build_inner_qsvt(lay, psi, emitter, sign_control=True, logical=logical)
    qc = lay.circuit("W_cond")
    qc.compose(inner, inplace=True)
    alpha = cmath.phase(block_phase_factor(len(psi) - 1))
    emitter.rz(qc, -2.0 * alpha, lay.q)
    qc.x(lay.q)
    qc.compose(build_reflection(lay, logical), inplace=True)
    return qc, inventory


def build_qft(
    lay: Layout, emitter: RotationEmitter, logical: bool = False
) -> QuantumCircuit:
    """Swap-free QFT on the system register; frequency bits leave bit reversed.

    Each CP(lambda) = P(lambda/2)_c CX P(-lambda/2)_t CX P(lambda/2)_t is
    emitted with Rz gates and its exact global phase lambda/4.
    """
    qc = QuantumCircuit(lay.n, name="QFT")
    for target in reversed(range(lay.n)):
        qc.h(target)
        for control in reversed(range(target)):
            lam = math.pi / 2 ** (target - control)
            if logical:
                qc.cp(lam, control, target)
                continue
            emitter.rz(qc, lam / 2, control)
            qc.cx(control, target)
            emitter.rz(qc, -lam / 2, target)
            qc.cx(control, target)
            emitter.rz(qc, lam / 2, target)
            qc.global_phase += lam / 4
    return qc
