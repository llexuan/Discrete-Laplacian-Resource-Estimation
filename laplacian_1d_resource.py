#!/usr/bin/env python3
"""
1D periodic
block-encoding basic unit U_L^(1), built to SCALE (no dense 2^n matrices).

Two independent register sizes:
  --verify-n : one small n -> dense matrix verification + wraparound.
  --target-n : large (20)        -> transpile + resource counts, no dense math.

Usage:
  python laplacian_1d_resource.py                 # verify n=3, target n=20
  python laplacian_1d_resource.py --verify-n 4 --target-n 20
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ["MPLCONFIGDIR"] = str(Path(".mplconfig").resolve())
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator



# classical normalized 1D operator (small-n reference only).

def shift_matrices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense cyclic shifts on N = 2**n points (for small-n verification only)."""
    N = 2**n
    s_plus = np.zeros((N, N), dtype=complex)
    for j in range(N):
        s_plus[(j + 1) % N, j] = 1.0
    s_minus = s_plus.T.conj().copy()
    return s_plus, s_minus


def scaled_1d_laplacian(n: int) -> np.ndarray:
    """L~_p^(1) = (1/4)(S^+ + S^- - 2 I)."""
    s_plus, s_minus = shift_matrices(n)
    N = 2**n
    return 0.25 * (s_plus + s_minus - 2.0 * np.eye(N, dtype=complex))


# Scalable reversible increment / decrement (S^+ / S^-).
def build_increment(n: int) -> QuantumCircuit:
    """
    Reversible modular incrementer on n qubits (little-endian, qubit 0 = LSB):

        S^+ |j> = |(j + 1) mod 2**n>.
    """
    qc = QuantumCircuit(n, name="S+")
    for target in range(n - 1, 0, -1):
        qc.mcx(list(range(target)), target)
    qc.x(0)
    return qc


def build_decrement(n: int) -> QuantumCircuit:
    """S^- |j> = |(j - 1) mod 2**n>, the inverse of the incrementer."""
    dec = build_increment(n).inverse()
    dec.name = "S-"
    return dec


def mcx_control_counts(n: int, ext_controls: int = 1) -> list[int]:
    """
    Control counts of the multi-controlled-X gates inside a CONTROLLED shift.

    The bare cascade holds C^rX for r = 1..n-1 plus a single X on the LSB;
    `ext_controls` external controls (l0/l1, plus any selector qubits) raise
    every count by that amount.
    """
    return sorted([r + ext_controls for r in range(1, n)] + [ext_controls])


def apply_controlled_shift(
    qc: QuantumCircuit,
    ctrl_qubits: list[int],
    ctrl_state: int,
    sys_qubits: list[int],
    direction: int,
) -> None:
    """
    Append a controlled modular shift S^+ (direction=+1) or S^- (direction=-1).

    The external controls are pushed into every multi-controlled X of the
    increment cascade instead of wrapping the shift as one composite controlled
    gate. Both are the same unitary, but Qiskit synthesises a controlled
    *composite* gate through a generic route whose cost explodes with n, while
    the explicit cascade stays a plain sequence of C^kX gates.

    `ctrl_state` follows Qiskit's convention: bit i selects the required value
    of ctrl_qubits[i], and a 0 bit is an open control, realised by an X pair.
    """
    open_ctrls = [
        q for i, q in enumerate(ctrl_qubits) if not (ctrl_state >> i) & 1
    ]
    n = len(sys_qubits)
    for q in open_ctrls:
        qc.x(q)
    # S^- is the reverse of S^+, and every C^kX is its own inverse.
    if direction < 0:
        qc.mcx(ctrl_qubits, sys_qubits[0])
        targets = range(1, n)
    else:
        targets = range(n - 1, 0, -1)
    for target in targets:
        qc.mcx(ctrl_qubits + sys_qubits[:target], sys_qubits[target])
    if direction > 0:
        qc.mcx(ctrl_qubits, sys_qubits[0])
    for q in open_ctrls:
        qc.x(q)



# Step 2: high-level 1D block encoding U_L^(1).
def build_u_l_1d(n: int, mcx_workspace: int = 2) -> QuantumCircuit:
    """
    block encoding of L~_p^(1) on (l0, l1, j0..j_{n-1}) plus a small workspace.

    Qubit order: 0 = l0, 1 = l1, 2..n+1 = system j, then `mcx_workspace`
    borrowable ancillas. Valid at any n (including n = 20).

    The workspace carries no data and is returned to |0>, but its presence lets
    the C^kX cascade use an ancilla-assisted decomposition that is linear in the
    number of controls instead of the ancilla-free one, whose cost grows
    exponentially: at n = 20 the two forms differ by 13.75e6 versus 3.0e3 T
    gates. Two ancillas already reach the linear regime and more do not help.
    The multidimensional unit gets the same benefit for free, since the idle
    registers of the other dimensions can be borrowed.
    """
    qc = QuantumCircuit(2 + n + mcx_workspace, name="U_L")
    sysq = list(range(2, 2 + n))

    # PREP: H then Z on each ancilla -> |->_{l0} |->_{l1}.
    qc.h(0)
    qc.h(1)
    qc.z(0)
    qc.z(1)

    # SELECT: S^- fires when l1 = 0 (open control); S^+ fires when l0 = 1.
    apply_controlled_shift(qc, [1], 0, sysq, -1)
    apply_controlled_shift(qc, [0], 1, sysq, +1)

    # UNPREP: final Hadamards on the ancilla register.
    qc.h(0)
    qc.h(1)
    return qc


def build_u_l_1d_schematic(n: int, mcx_workspace: int = 2) -> QuantumCircuit:
    """
    Same unitary as build_u_l_1d, drawn with opaque S+/S- boxes.

    This form is for presentation only. Resource estimation uses build_u_l_1d,
    whose explicit C^kX cascades expose the shift implementation to Qiskit.
    """
    qc = QuantumCircuit(2 + n + mcx_workspace, name="U_L")
    sysq = list(range(2, 2 + n))
    inc = build_increment(n).to_gate(label="S+")
    dec = build_decrement(n).to_gate(label="S-")

    qc.h(0)
    qc.h(1)
    qc.z(0)
    qc.z(1)
    qc.append(dec.control(1, ctrl_state=0), [1] + sysq)
    qc.append(inc.control(1, ctrl_state=1), [0] + sysq)
    qc.h(0)
    qc.h(1)
    return qc


# Verification helpers (dense, small n only).
def verify_shifts(n: int) -> dict:
    """Check the gate-based increment/decrement against the dense shifts."""
    s_plus, s_minus = shift_matrices(n)
    inc_op = Operator(build_increment(n)).data
    dec_op = Operator(build_decrement(n)).data
    inc_err = float(np.max(np.abs(inc_op - s_plus)))
    dec_err = float(np.max(np.abs(dec_op - s_minus)))

    # Explicit wraparound basis-state checks.
    N = 2**n

    def apply(op: np.ndarray, j: int) -> int:
        vec = np.zeros(N, dtype=complex)
        vec[j] = 1.0
        out = op @ vec
        return int(np.argmax(np.abs(out)))

    wrap_inc = apply(inc_op, N - 1)  # expect 0
    wrap_dec = apply(dec_op, 0)  # expect N - 1
    return {
        "increment_max_err": inc_err,
        "decrement_max_err": dec_err,
        "wrap_increment_ok": wrap_inc == 0,
        "wrap_decrement_ok": wrap_dec == (N - 1),
    }


def verify_block_encoding(n: int, mcx_workspace: int = 2) -> float:
    """
    Check <00|_{l0 l1} U_L |00> = L~_p^(1) densely for small n.

    The workspace is projected onto |0..0> on both sides, which also confirms
    that the ancilla-assisted C^kX decompositions hand it back untouched.
    """
    u_l_op = Operator(build_u_l_1d(n, mcx_workspace)).data
    lap = scaled_1d_laplacian(n)
    anc0 = np.zeros(4, dtype=complex)
    anc0[0] = 1.0  # |l1=0, l0=0>
    ws0 = np.zeros(2**mcx_workspace, dtype=complex)
    ws0[0] = 1.0
    # Little-endian qubit order: ancillas lowest, workspace highest.
    right = np.kron(
        np.kron(ws0[:, None], np.eye(2**n, dtype=complex)), anc0[:, None]
    )
    block = right.conj().T @ u_l_op @ right
    return float(np.max(np.abs(block - lap)))



# Resource estimation (target n, no dense math).
def resource_counts(
    qc: QuantumCircuit, basis_gates: list[str], count_t_depth: bool = False
) -> dict:
    """Transpile to a logical basis and report gate counts / depth."""
    t = transpile(qc, basis_gates=basis_gates, optimization_level=0)
    ops = dict(t.count_ops())
    result = {
        "basis_gates": basis_gates,
        "gate_counts": ops,
        "total_gates": int(sum(ops.values())),
        "depth": int(t.depth()),
        "num_qubits": int(t.num_qubits),
    }
    if count_t_depth:
        # T-depth: number of sequential layers containing a T or T-dagger gate.
        result["t_depth"] = int(
            t.depth(
                filter_function=lambda instr: instr.operation.name in ("t", "tdg")
            )
        )
    return result


def high_level_counts(n: int) -> dict:
    """Structural (pre-decomposition) operation breakdown of U_L^(1)."""
    ctrl_counts = mcx_control_counts(n)
    return {
        "H_gates": 4,
        "Z_gates": 2,
        "controlled_S_plus": 1,
        "controlled_S_minus": 1,
        "mcx_control_counts_per_shift": ctrl_counts,
        "note": (
            "Each controlled shift is a cascade of C^kX for k = 1..n "
            "(k=1 is a CNOT). Two shifts => two such cascades."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scalable 1D periodic Laplacian block-encoding basic unit: "
        "dense verification at small n, resource counts at target n."
    )
    parser.add_argument(
        "--verify-n",
        type=int,
        default=3,
        help="Register size used for one dense correctness check (default 3).",
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=20,
        help="Register size for the resource estimate (default 20 ~ 10^6 pts).",
    )
    parser.add_argument(
        "--mcx-workspace",
        type=int,
        default=2,
        help="Borrowable ancillas for the C^kX cascades (default 2). 0 forces "
        "the ancilla-free decomposition, whose cost explodes with n.",
    )
    parser.add_argument(
        "--save-prefix",
        type=Path,
        default=Path("1d"),
        help="Prefix for the saved ASCII circuit drawing.",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path("laplacian_1d_resource_metrics.json"),
        help="Where to write verification + resource metrics as JSON.",
    )
    args = parser.parse_args()

    if args.verify_n < 1:
        raise ValueError("--verify-n must be >= 1.")
    if args.target_n < 1:
        raise ValueError("--target-n must be >= 1.")
    if args.mcx_workspace < 0:
        raise ValueError("--mcx-workspace must be >= 0.")

    print("=== 1D periodic Laplacian block-encoding basic unit (U_L^(1)) ===")
    print("scope            : classical L~_p^(1) + U_L^(1)")
    print("shift model      : reversible modular increment/decrement (scalable)")
    print(f"verification n   : n = {args.verify_n}")
    print(f"target size      : n = {args.target_n} (N = 2**{args.target_n})")

    # --- Step 1+2 verification at the selected small n. ---
    print("\n--- correctness verification (dense, small n) ---")
    n = args.verify_n
    shifts = verify_shifts(n)
    be_err = verify_block_encoding(n, args.mcx_workspace)
    verify_report = {n: {**shifts, "block_encoding_err": be_err}}
    print(
        f"n={n}: S+ err={shifts['increment_max_err']:.2e}, "
        f"S- err={shifts['decrement_max_err']:.2e}, "
        f"wrap+={shifts['wrap_increment_ok']}, "
        f"wrap-={shifts['wrap_decrement_ok']}, "
        f"<00|U_L|00>-L~ err={be_err:.2e}"
    )

    # --- Structural / high-level counts at target n. ---
    hlc = high_level_counts(args.target_n)
    print(f"\n--- high-level operation structure (n={args.target_n}) ---")
    print(f"H gates                 : {hlc['H_gates']}")
    print(f"Z gates                 : {hlc['Z_gates']}")
    print(f"controlled S^+ / S^-    : 1 / 1")
    print(f"MCX ancilla workspace   : {args.mcx_workspace} (borrowable, returned to |0>)")
    print(
        f"per-shift MCX controls  : C^kX for k in "
        f"[{hlc['mcx_control_counts_per_shift'][0]}.."
        f"{hlc['mcx_control_counts_per_shift'][-1]}]"
    )

    # --- Decomposed resource counts at target n  ---
    print(f"\n--- decomposed resource counts (n={args.target_n}) ---")
    u_l_target = build_u_l_1d(args.target_n, args.mcx_workspace)

    # Coarse logical reference (arbitrary 1-qubit rotations + CNOT). Fast, but
    # hides non-Clifford cost inside 'u'.
    res_ucx = resource_counts(u_l_target, ["u", "cx"])
    print(f"[basis u, cx] total gates : {res_ucx['total_gates']}")
    print(f"[basis u, cx] gate counts : {res_ucx['gate_counts']}")
    print(f"[basis u, cx] depth       : {res_ucx['depth']}")
    print(f"[basis u, cx] qubits      : {res_ucx['num_qubits']}")

    # Fault-tolerant estimate (Clifford+T). This is the primary resource metric.
    ct_basis = ["h", "t", "tdg", "s", "sdg", "x", "z", "cx"]
    res_ct = resource_counts(u_l_target, ct_basis, count_t_depth=True)
    gc = res_ct["gate_counts"]
    t_count = int(gc.get("t", 0) + gc.get("tdg", 0))
    clifford_count = int(res_ct["total_gates"] - t_count)
    res_ct["t_count"] = t_count
    res_ct["clifford_gate_count"] = clifford_count

    print(f"\n--- fault-tolerant resource estimate (Clifford+T, n={args.target_n}) ---")
    print(f"logical qubits      : {res_ct['num_qubits']}")
    print(f"Clifford gate count : {clifford_count}")
    print(f"T-count             : {t_count}")
    print(f"total depth         : {res_ct['depth']}")
    print(f"T-depth             : {res_ct['t_depth']}")

    # --- Persist metrics + a small-n circuit drawing. ---
    metrics = {
        "scope": "1d_periodic_laplacian_block_encoding_steps_1_2",
        "verify_n": args.verify_n,
        "target_n": args.target_n,
        "grid_points_target": 2**args.target_n,
        "mcx_workspace": args.mcx_workspace,
        "verification": {str(k): v for k, v in verify_report.items()},
        "high_level_counts": hlc,
        "resource_counts_u_cx": res_ucx,
        "resource_counts_clifford_t": res_ct,
        "fault_tolerant_summary": {
            "logical_qubits": res_ct["num_qubits"],
            "clifford_gate_count": clifford_count,
            "t_count": t_count,
            "total_depth": res_ct["depth"],
            "t_depth": res_ct["t_depth"],
        },
    }
    args.output_metrics.write_text(json.dumps(metrics, indent=2) + "\n", "utf-8")

    draw_n = args.verify_n
    draw_path = Path(f"laplacian_{args.save_prefix}_U_L.txt")
    draw_text = (
        "=== 1D periodic Laplacian block encoding U_L^(1) (scalable form) ===\n"
        f"drawn at n = {draw_n} (grid N = {2**draw_n}); "
        "<00|_{l0 l1} U_L |00> = L~_p^(1).\n\n"
        "registers:\n"
        "  q_0 = l0 = block-encoding ancilla\n"
        "  q_1 = l1 = block-encoding ancilla\n"
        f"  q_2..q_{1 + draw_n} = system grid register (n = {draw_n})\n"
        f"  q_{2 + draw_n}.. = MCX ancilla workspace "
        f"({args.mcx_workspace}, borrowed and returned to |0>)\n\n"
        "--- schematic form (shifts as opaque S+/S- boxes) ---\n"
        "This is the compact mathematical view of the controlled shifts.\n\n"
        f"{build_u_l_1d_schematic(draw_n, args.mcx_workspace).draw(output='text')}\n\n"
        "--- counted form (shifts expanded into C^kX cascades) ---\n"
        f"Each shift contains MCX control counts {mcx_control_counts(draw_n)} "
        f"at n = {draw_n}; this is the circuit passed to the transpiler.\n\n"
        f"{build_u_l_1d(draw_n, args.mcx_workspace).draw(output='text')}\n"
    )
    draw_path.write_text(draw_text, encoding="utf-8")

    print("\nSaved:", args.output_metrics, "and", draw_path)


if __name__ == "__main__":
    main()
