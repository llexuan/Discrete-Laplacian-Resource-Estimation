"""Exact LCU block encoding U_X of X = diag(2 m_k / N) in the Fourier basis.

For frequency k = k_{q-1} 2^{q-1} + r, 2 m_k / N = (1 - k_{q-1}) Y + k_{q-1} (1 - Y)
with Y = sum_{j<q-1} lambda_j P_j, lambda_j = 2^{j-q+1} and P_j = (I - Z_j)/2.

    U_X = PREP^dag . SELECT_X . PREP

PREP |0>_{pJ} = sqrt(2/N) |1>_p |0>_J + sum_j sqrt(lambda_j) |0>_p |j>_J.
SELECT_X = CX(k_{q-1} -> a) . X_a . sum_j |j><j|_J |0><0|_p CX(k_j -> a),
with every branch targeting one shared ancilla a. The block <0_{a p J}| U_X |0_{a p J}>
equals X exactly, and U_X is Hermitian because SELECT_X is.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from .gates import (
    RotationEmitter,
    append_mcx,
    append_multiplexed_ry,
    state_preparation_tree,
)

KET0 = np.array([1.0, 0.0], dtype=complex)
KET_PLUS = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)


def index_qubits(n: int) -> int:
    """|J| = ceil(log2(q - 1)) qubits index the q - 1 projectors P_j."""
    if n < 3:
        raise ValueError("The preconditioned block encoding needs q >= 3 system qubits.")
    return int(math.ceil(math.log2(n - 1)))


@dataclass(frozen=True)
class Layout:
    """Global qubit indices shared by U_X, the inner QSVT and the outer walk.

    Little-endian order: q, a, p, J_0..J_{L-1}, system s_0..s_{q-1}, flag.
    ``k_bits[j]`` is the system qubit holding frequency bit k_j.
    """

    n: int
    num_index: int
    q: int | None
    a: int
    p: int
    J: tuple[int, ...]
    system: tuple[int, ...]
    flag: int | None
    total: int
    k_bits: tuple[int, ...]
    registers: tuple | None = None

    @property
    def anc(self) -> list[int]:
        return [self.a, self.p, *self.J]

    @property
    def m(self) -> int:
        return 2 + self.num_index

    def circuit(self, name: str) -> QuantumCircuit:
        if self.registers:
            return QuantumCircuit(*self.registers, name=name)
        return QuantumCircuit(self.total, name=name)


def walk_layout(n: int, fourier_order: str = "bit_reversed") -> Layout:
    """Resident register of the outer walk; 'bit_reversed' matches a swap-free QFT."""
    L = index_qubits(n)
    system = tuple(range(3 + L, 3 + L + n))
    if fourier_order == "bit_reversed":
        k_bits = tuple(reversed(system))
    elif fourier_order == "natural":
        k_bits = system
    else:
        raise ValueError("fourier_order must be 'bit_reversed' or 'natural'.")
    return Layout(
        n=n,
        num_index=L,
        q=0,
        a=1,
        p=2,
        J=tuple(range(3, 3 + L)),
        system=system,
        flag=3 + L + n,
        total=4 + L + n,
        k_bits=k_bits,
    )


def drawing_layout(n: int) -> Layout:
    """Named a/p/J/k registers only, for PREP and SELECT drawings."""
    L = index_qubits(n)
    registers = (
        QuantumRegister(1, "a"),
        QuantumRegister(1, "p"),
        QuantumRegister(L, "J"),
        QuantumRegister(n, "k"),
    )
    system = tuple(range(2 + L, 2 + L + n))
    return Layout(
        n=n,
        num_index=L,
        q=None,
        a=0,
        p=1,
        J=tuple(range(2, 2 + L)),
        system=system,
        flag=None,
        total=2 + L + n,
        k_bits=system,
        registers=registers,
    )


def lcu_weights(n: int) -> list[float]:
    """lambda_j = 2^{j-q+1} for j = 0..q-2; the padding weight is 2/N."""
    return [2.0 ** (j - n + 1) for j in range(n - 1)]


def index_register_weights(n: int) -> list[float]:
    """Marginal J distribution of PREP: padding merges into J = 0."""
    L = index_qubits(n)
    weights = [0.0] * 2**L
    for j, value in enumerate(lcu_weights(n)):
        weights[j] = value
    weights[0] += 2.0 / 2**n
    return weights


def build_prep(lay: Layout, emitter: RotationEmitter, logical: bool = False) -> QuantumCircuit:
    """PREP: pruned Gray-code state tree on J, then split p only when J = 0."""
    qc = lay.circuit("PREP")
    for bit, control_bits, angles in state_preparation_tree(
        index_register_weights(lay.n), lay.num_index
    ):
        append_multiplexed_ry(
            qc, emitter, lay.J[bit], [lay.J[c] for c in control_bits], angles
        )
    # Ry(-pi/4) X Ry(pi/4)|0> = (|0> + |1>)/sqrt2, so p carries 2/N when J = 0.
    emitter.ry(qc, math.pi / 4, lay.p)
    append_mcx(
        qc, list(lay.J), lay.p, list(lay.system), [0] * lay.num_index, logical
    )
    emitter.ry(qc, -math.pi / 4, lay.p)
    return qc


def build_select(lay: Layout, logical: bool = False) -> QuantumCircuit:
    """SELECT_X on one shared ancilla a (the note's separate a_j break Eq. 6)."""
    qc = lay.circuit("SELECT")
    qc.x(lay.a)
    extra = [] if lay.q is None else [lay.q]
    for j in range(lay.n - 1):
        controls = [*lay.J, lay.p, lay.k_bits[j]]
        states = [(j >> i) & 1 for i in range(lay.num_index)] + [0, 1]
        dirty = [s for s in lay.system if s != lay.k_bits[j]] + extra
        append_mcx(qc, controls, lay.a, dirty, states, logical)
    qc.cx(lay.k_bits[lay.n - 1], lay.a)
    return qc


def build_u_x(
    lay: Layout,
    emitter: RotationEmitter,
    logical: bool = False,
    prep: QuantumCircuit | None = None,
) -> QuantumCircuit:
    """U_X = PREP^dag SELECT PREP; PREP^dag is the exact inverse circuit."""
    prep = build_prep(lay, emitter, logical) if prep is None else prep
    qc = lay.circuit("U_X")
    qc.compose(prep, inplace=True)
    qc.compose(build_select(lay, logical), inplace=True)
    qc.compose(prep.inverse(), inplace=True)
    return qc


def frequency_m(n: int) -> np.ndarray:
    """m_k = min(k, N - k)."""
    k = np.arange(2**n)
    return np.minimum(k, 2**n - k)


def x_hat_values(n: int) -> np.ndarray:
    """theta_k = 2 m_k / N in [0, 1]."""
    return 2.0 * frequency_m(n) / 2**n


def frequency_of_system_states(lay: Layout) -> np.ndarray:
    """Frequency k read by the circuit for each system basis index."""
    positions = [lay.system.index(qubit) for qubit in lay.k_bits]
    states = np.arange(2**lay.n)
    return sum(((states >> pos) & 1) << j for j, pos in enumerate(positions))


def isometry(total: int, free: list[int], fixed: dict[int, np.ndarray]) -> np.ndarray:
    """Columns embed each basis state of ``free`` with ``fixed`` single-qubit kets."""
    if sorted(list(free) + list(fixed)) != list(range(total)):
        raise ValueError("Every qubit must be either free or fixed.")
    columns = []
    for state in range(2 ** len(free)):
        vector = np.ones(1, dtype=complex)
        for qubit in reversed(range(total)):
            if qubit in fixed:
                ket = fixed[qubit]
            else:
                ket = np.zeros(2, dtype=complex)
                ket[(state >> free.index(qubit)) & 1] = 1.0
            vector = np.kron(vector, ket)
        columns.append(vector)
    return np.stack(columns, axis=1)


def ancilla_zero_isometry(lay: Layout, plus_q: bool = False) -> np.ndarray:
    """Embed the system with a, p, J, flag in |0> and q in |0> or |+>."""
    fixed = {qubit: KET0 for qubit in lay.anc}
    if lay.flag is not None:
        fixed[lay.flag] = KET0
    if lay.q is not None:
        fixed[lay.q] = KET_PLUS if plus_q else KET0
    return isometry(lay.total, list(lay.system), fixed)


def prep_state_error(lay: Layout, prep_matrix: np.ndarray) -> float:
    """max |PREP|0> - |gamma>| up to the irrelevant branch signs."""
    column = prep_matrix[:, 0]
    target = np.zeros_like(column)
    target[1 << lay.p] = math.sqrt(2.0 / 2**lay.n)
    for j, value in enumerate(lcu_weights(lay.n)):
        index = sum(((j >> i) & 1) << qubit for i, qubit in enumerate(lay.J))
        target[index] = math.sqrt(value)
    return float(np.max(np.abs(np.abs(column) - np.abs(target))))


def draw_prep_select(n: int) -> dict[str, str]:
    """Readable gate-level PREP and SELECT circuits on named registers."""
    lay = drawing_layout(n)
    emitter = RotationEmitter()
    prep = build_prep(lay, emitter, logical=True)
    select = build_select(lay, logical=True)
    weights = lcu_weights(n)
    L = lay.num_index
    header = (
        f"q = {n} system qubits (N = 2^{n}); |J| = ceil(log2(q-1)) = {L}; "
        f"ancillas a, p, J = {2 + L}\n"
        f"lambda_j = 2^(j-q+1), j = 0..{n - 2}; padding weight 2/N = {2.0 / 2**n:.6e}; "
        f"sum = {sum(weights) + 2.0 / 2**n:.15f}\n"
    )
    prep_text = (
        "=== PREP for U_X (preconditioned 1D Laplacian) ===\n"
        + header
        + "PREP|0>_{pJ} = sqrt(2/N)|1>_p|0>_J + sum_j sqrt(lambda_j)|0>_p|j>_J\n\n"
        "construction:\n"
        "  1. pruned Gray-code state tree on J with marginal weights w(0) = lambda_0 + 2/N,\n"
        "     w(j) = lambda_j (j >= 1); each level keeps only the higher J bits it depends on\n"
        "  2. Ry(pi/4) . MCX(J = 0 -> p) . Ry(-pi/4) splits w(0) equally into lambda_0 and 2/N\n"
        f"non-Clifford+T rotations in PREP: {len(emitter.records)} "
        "(the two Ry(+-pi/4) are exact Clifford+T)\n"
        "resource circuits replace MCX with an exact dirty-ancilla ladder borrowing system qubits.\n"
        "idle wires (a and the k register) are hidden.\n\n"
        f"{prep.draw(output='text', fold=-1, idle_wires=False)}\n"
    )
    select_text = (
        "=== SELECT_X for U_X (preconditioned 1D Laplacian) ===\n"
        + header
        + "SELECT_X = CX(k_{q-1} -> a) . X_a . prod_j [ CX(k_j -> a) controlled on J = j, p = 0 ]\n\n"
        "construction:\n"
        "  X_a              : the note's U_Pj = X_a CNOT(k_j -> a) and the padding X_a, merged\n"
        f"  {n - 1} MCX gates      : target a; controls J = j (open = 0), p = 0 (open), k_j (closed)\n"
        "  CX(k_{q-1} -> a) : SELECT_X = CX_a SELECT_Y, which needs one shared ancilla a\n"
        "block: <0_a 0_p 0_J| PREP^dag SELECT_X PREP |0_a 0_p 0_J> = diag(2 m_k / N)\n"
        "resource circuits replace MCX with an exact dirty-ancilla ladder borrowing system qubits.\n\n"
        f"{select.draw(output='text', fold=-1)}\n"
    )
    return {"PREP": prep_text, "SELECT": select_text, "prep_circuit": prep, "select_circuit": select}
