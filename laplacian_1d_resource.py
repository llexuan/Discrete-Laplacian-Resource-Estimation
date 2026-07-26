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


def mcx_control_counts(n: int) -> list[int]:
    """
    Control counts of the multi-controlled-X gates inside a CONTROLLED shift
    (S^+ or S^- with one extra external control l0/l1).
    """
    return list(range(1, n + 1))



# Step 2: high-level 1D block encoding U_L^(1).
def build_u_l_1d(n: int) -> QuantumCircuit:
    """
    block encoding of L~_p^(1) on (l0, l1, j0..j_{n-1}).

    Qubit order: 0 = l0, 1 = l1, 2.. = system j. Uses the scalable
    increment/decrement gates so it is valid at any n (including n = 20).
    """
    qc = QuantumCircuit(2 + n, name="U_L")
    sysq = list(range(2, 2 + n))

    inc = build_increment(n).to_gate(label="S+")
    dec = build_decrement(n).to_gate(label="S-")

    # PREP: H then Z on each ancilla -> |->_{l0} |->_{l1}.
    qc.h(0)
    qc.h(1)
    qc.z(0)
    qc.z(1)

    # SELECT: S^- fires when l1 = 0 (open control); S^+ fires when l0 = 1.
    qc.append(dec.control(1, ctrl_state=0), [1] + sysq)
    qc.append(inc.control(1, ctrl_state=1), [0] + sysq)

    # UNPREP: final Hadamards on the ancilla register.
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


def verify_block_encoding(n: int) -> float:
    """Check <00|_{l0 l1} U_L |00> = L~_p^(1) densely for small n."""
    u_l_op = Operator(build_u_l_1d(n)).data
    lap = scaled_1d_laplacian(n)
    anc0 = np.zeros(4, dtype=complex)
    anc0[0] = 1.0  # |l1=0, l0=0>
    left = np.kron(np.eye(2**n, dtype=complex), anc0.conj()[None, :])
    right = np.kron(np.eye(2**n, dtype=complex), anc0[:, None])
    block = left @ u_l_op @ right
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

    print("=== 1D periodic Laplacian block-encoding basic unit (U_L^(1)) ===")
    print("scope            : classical L~_p^(1) + U_L^(1)")
    print("shift model      : reversible modular increment/decrement (scalable)")
    print(f"verification n   : n = {args.verify_n}")
    print(f"target size      : n = {args.target_n} (N = 2**{args.target_n})")

    # --- Step 1+2 verification at the selected small n. ---
    print("\n--- correctness verification (dense, small n) ---")
    n = args.verify_n
    shifts = verify_shifts(n)
    be_err = verify_block_encoding(n)
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
    print(
        f"per-shift MCX controls  : C^kX for k in "
        f"[{hlc['mcx_control_counts_per_shift'][0]}.."
        f"{hlc['mcx_control_counts_per_shift'][-1]}]"
    )

    # --- Decomposed resource counts at target n  ---
    print(f"\n--- decomposed resource counts (n={args.target_n}) ---")
    u_l_target = build_u_l_1d(args.target_n)

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
    draw_circ = build_u_l_1d(draw_n)
    draw_path = Path(f"laplacian_{args.save_prefix}_U_L.txt")
    draw_text = (
        "=== 1D periodic Laplacian block encoding U_L^(1) (scalable form) ===\n"
        f"drawn at n = {draw_n} (grid N = {2**draw_n}); "
        "<00|_{l0 l1} U_L |00> = L~_p^(1).\n\n"
        "registers:\n"
        "  q_0 = l0 = block-encoding ancilla\n"
        "  q_1 = l1 = block-encoding ancilla\n"
        f"  q_2.. = system grid register (n = {draw_n})\n\n"
        "S^+ / S^- are reversible modular increment / decrement circuits.\n\n"
        f"{draw_circ.draw(output='text')}\n"
    )
    draw_path.write_text(draw_text, encoding="utf-8")

    print("\nSaved:", args.output_metrics, "and", draw_path)


if __name__ == "__main__":
    main()
