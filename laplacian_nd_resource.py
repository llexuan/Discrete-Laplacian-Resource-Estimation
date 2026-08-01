#!/usr/bin/env python3
"""
Multidimensional (2D / 3D) periodic Laplacian block-encoding basic unit
U_L^(D), built to SCALE by reusing the tested 1D shift primitives.

This estimates ONE multidimensional block-encoding segment.

It adds, on top of the 1D unit:
  * a K = ceil(log2 D) dimension-selector register,
  * a selector-state preparation  |sel> = (1/sqrt D) sum_d |d>,
  * selector-controlled shifts on each of the D system registers,
  * the inverse selector preparation.

2D:  U_prep_k is a single Hadamard (exactly Clifford), so the whole unit is
     exactly Clifford+T.
3D:  |sel> = (|00>+|01>+|10>)/sqrt3 needs one arbitrary Ry rotation, which is
     NOT exactly Clifford+T. Its T-cost is estimated from --prep-tol using the
     Ross-Selinger single-qubit synthesis scaling (~ 3 log2(1/eps) T gates).

Usage:
  python laplacian_nd_resource.py --dims 2                 # verify n=2, target n=20
  python laplacian_nd_resource.py --dims 3 --prep-tol 1e-8
  python laplacian_nd_resource.py --dims 2 --verify-n 2 --target-n 6
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ["MPLCONFIGDIR"] = str(Path(".mplconfig").resolve())
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate
from qiskit.quantum_info import Operator

# Reuse the tested 1D primitives + counting utility.
from laplacian_1d_resource import (
    apply_controlled_shift,
    build_decrement,
    build_increment,
    mcx_control_counts,
    resource_counts,
    shift_matrices,
)


CT_BASIS = ["h", "t", "tdg", "s", "sdg", "x", "z", "cx"]


# --- selector geometry -------------------------------------------------------
def num_k_qubits(dims: int) -> int:
    """K = ceil(log2 D) selector qubits (K=1 for 2D, K=2 for 3D)."""
    return max(1, math.ceil(math.log2(dims)))


def build_prep_selector(dims: int, omit_rotation: bool = False) -> QuantumCircuit:
    """
    Prepare |sel> = (1/sqrt D) sum_{d=0}^{D-1} |d> on K selector qubits.
    """
    k = num_k_qubits(dims)
    prep = QuantumCircuit(k, name="U_prep_k")
    if dims == 2:
        prep.h(0)
    elif dims == 3:
        theta1 = 2.0 * math.acos(math.sqrt(2.0 / 3.0))
        if not omit_rotation:
            prep.ry(theta1, 1)           # split amplitude 2/3 : 1/3 on sel1
        prep.ch(1, 0, ctrl_state=0)      # equalize |0>,|1> of sel0 when sel1=0
    else:
        raise ValueError("dims must be 2 or 3.")
    return prep


def count_prep_rotations(dims: int) -> int:
    """Number of non-Clifford single-qubit rotations in ONE selector prep."""
    count = 0
    for instr in build_prep_selector(dims).data:
        if instr.operation.name in ("rx", "ry", "rz"):
            if not is_clifford_angle(float(instr.operation.params[0])):
                count += 1
    return count


# --- Step: high-level D-dimensional block encoding U_L^(D). ------------------
def sys_registers(n: int, dims: int, offset: int) -> list[list[int]]:
    """The D system registers of n qubits each, starting at `offset`."""
    return [list(range(offset + d * n, offset + (d + 1) * n)) for d in range(dims)]


def append_mc_h(qc: QuantumCircuit, controls: list[int], target: int) -> None:
    """
    Append a multi-controlled Hadamard, C^k H, in exact Clifford+T.
    """
    if not controls:
        qc.h(target)
        return
    # Global phases of the T and Tdg halves cancel, so this is exact.
    for sign in (+1, -1):
        if sign < 0:
            qc.mcx(controls, target)
        qc.sdg(target)
        qc.h(target)
        qc.t(target) if sign > 0 else qc.tdg(target)
        qc.h(target)
        qc.s(target)


def apply_u_l_nd(
    qc: QuantumCircuit,
    n: int,
    dims: int,
    anc: list[int],
    sel: list[int],
    sys_regs: list[list[int]],
    extra_controls: list[int] = (),
    omit_rotation: bool = False,
) -> None:
    """
    Append U_L^(D) onto `qc` at the given qubit positions.
    """
    l0, l1 = anc
    ec = list(extra_controls)
    k = len(sel)
    # Every extra control is closed, so its ctrl_state bits are all 1. They sit
    # above the ancilla bit (1) and the selector bits (k).
    ec_state = ((1 << len(ec)) - 1) << (1 + k)

    def add_prep(prep: QuantumCircuit) -> None:
        for instr in prep.data:
            qc.append(
                instr.operation,
                [sel[prep.find_bit(b).index] for b in instr.qubits],
            )

    prep_sel = build_prep_selector(dims, omit_rotation=omit_rotation)

    # PREP ancillas: H then Z on each -> |->_{l0} |->_{l1}.
    append_mc_h(qc, ec, l0)
    append_mc_h(qc, ec, l1)
    if ec:
        qc.append(ZGate().control(len(ec)), ec + [l0])
        qc.append(ZGate().control(len(ec)), ec + [l1])
    else:
        qc.z(l0)
        qc.z(l1)

    # PREP selector.
    add_prep(prep_sel)

    # SELECT: for each dimension, selector-and-ancilla-controlled shifts.
    for d in range(dims):
        # control state bits: bit0 = ancilla, bits 1..k = selector value d.
        cs_minus = (d << 1) | ec_state           # l1 = 0 (open) and sel = d
        cs_plus = 1 | (d << 1) | ec_state        # l0 = 1        and sel = d
        apply_controlled_shift(qc, [l1] + sel + ec, cs_minus, sys_regs[d], -1)
        apply_controlled_shift(qc, [l0] + sel + ec, cs_plus, sys_regs[d], +1)

    # UNPREP selector, then final Hadamards on the ancillas.
    add_prep(prep_sel.inverse())
    append_mc_h(qc, ec, l0)
    append_mc_h(qc, ec, l1)


def build_u_l_nd(n: int, dims: int, exact_bulk: bool = False) -> QuantumCircuit:
    """
    Block encoding of L~_p^(D) = 1/(4D) [ sum_d (S_d^+ + S_d^-) - 2D I ].
    """
    k = num_k_qubits(dims)
    qc = QuantumCircuit(2 + k + dims * n, name=f"U_L_{dims}D")
    apply_u_l_nd(
        qc,
        n,
        dims,
        anc=[0, 1],
        sel=list(range(2, 2 + k)),
        sys_regs=sys_registers(n, dims, 2 + k),
        omit_rotation=exact_bulk,
    )
    return qc


def build_u_l_nd_schematic(n: int, dims: int) -> QuantumCircuit:
    """
  For visualisation only.
    """
    k = num_k_qubits(dims)
    qc = QuantumCircuit(2 + k + dims * n, name=f"U_L_{dims}D")
    sel = list(range(2, 2 + k))
    qc.h(0)
    qc.h(1)
    qc.z(0)
    qc.z(1)
    qc.append(build_prep_selector(dims).to_gate(label="U_prep_k"), sel)
    for d in range(dims):
        sys_d = list(range(2 + k + d * n, 2 + k + (d + 1) * n))
        inc = build_increment(n).to_gate(label="S+")
        dec = build_decrement(n).to_gate(label="S-")
        qc.append(dec.control(1 + k, ctrl_state=d << 1), [1] + sel + sys_d)
        qc.append(inc.control(1 + k, ctrl_state=1 | (d << 1)), [0] + sel + sys_d)
    qc.append(
        build_prep_selector(dims).inverse().to_gate(label="U_prep_k+"), sel
    )
    qc.h(0)
    qc.h(1)
    return qc


# --- classical reference (dense, small n only). -----------------------------
def scaled_nd_laplacian(n: int, dims: int) -> np.ndarray:
    """L~_p^(D) as a dense (2**n)**D matrix (small-n verification only)."""
    big_n = 2**n
    s_plus, s_minus = shift_matrices(n)
    eye = np.eye(big_n, dtype=complex)
    dim_total = big_n**dims

    def embed(op: np.ndarray, d: int) -> np.ndarray:
        # numpy kron is MSB-first; dim 0 is the LSB register.
        mats = [op if dim == d else eye for dim in range(dims - 1, -1, -1)]
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    lap = np.zeros((dim_total, dim_total), dtype=complex)
    for d in range(dims):
        lap += embed(s_plus, d) + embed(s_minus, d)
    lap -= 2.0 * dims * np.eye(dim_total, dtype=complex)
    return lap / (4.0 * dims)


def verify_block_encoding_nd(n: int, dims: int) -> float:
    """Check <0..0|_{l,sel} U_L^(D) |0..0>_{l,sel} = L~_p^(D) densely."""
    u_op = Operator(build_u_l_nd(n, dims)).data
    k = num_k_qubits(dims)
    low = k + 2  # ancilla + selector qubits are the low-order bits
    dim_total = (2**n) ** dims
    idx = [s << low for s in range(dim_total)]  # states with l=sel=0
    block = u_op[np.ix_(idx, idx)]
    lap = scaled_nd_laplacian(n, dims)
    return float(np.max(np.abs(block - lap)))


# --- rotation synthesis model (3D selector prep only). ----------------------
def is_clifford_angle(theta: float) -> bool:
    """True if theta is a multiple of pi/2 (i.e. exactly Clifford)."""
    ratio = theta / (math.pi / 2.0)
    return abs(ratio - round(ratio)) < 1e-9


def ross_selinger_t_count(eps: float) -> int:
    """
    Ross-Selinger single-qubit z-rotation synthesis scaling: the T-count to
    approximate an arbitrary rotation to accuracy eps is ~ 3 log2(1/eps).
    """
    if not 0.0 < eps < 1.0:
        raise ValueError("Rotation tolerance must satisfy 0 < eps < 1.")
    return int(math.ceil(3.0 * math.log2(1.0 / eps)))


# --- resource estimation (target n, no dense math). -------------------------
def estimate_resources(n: int, dims: int, prep_tol: float) -> dict:
    """Fault-tolerant (Clifford+T) resource estimate for U_L^(D) at size n."""
    qc = build_u_l_nd(n, dims)

    if dims == 2:
        # Coarse logical reference: arbitrary one-qubit rotations + CNOT.
        res_ucx = resource_counts(qc, ["u", "cx"])
        # Everything decomposes exactly into Clifford+T.
        res = resource_counts(qc, CT_BASIS, count_t_depth=True)
        gc = res["gate_counts"]
        t_count = int(gc.get("t", 0) + gc.get("tdg", 0))
        clifford = int(res["total_gates"] - t_count)
        return {
            "logical_qubits": int(res["num_qubits"]),
            "clifford_gate_count": clifford,
            "t_count": t_count,
            "total_depth": int(res["depth"]),
            "t_depth": int(res["t_depth"]),
            "gate_counts": gc,
            "resource_counts_u_cx": res_ucx,
            "n_arbitrary_rotations": 0,
            "rotation_synthesis": None,
        }

    # 3D: everything except the selector-prep Ry decomposes exactly into
    # Clifford+T.  Count that exact bulk, then add the arbitrary rotations
    # (one per prep, one per unprep) via a synthesis model.
    bulk = build_u_l_nd(n, dims, exact_bulk=True)
    res = resource_counts(bulk, CT_BASIS, count_t_depth=True)
    gc = res["gate_counts"]
    exact_t = int(gc.get("t", 0) + gc.get("tdg", 0))
    exact_clifford = int(res["total_gates"] - exact_t)

    n_rot = 2 * count_prep_rotations(dims)  # prep + inverse unprep
    eps_per = prep_tol / max(n_rot, 1)
    t_per = ross_selinger_t_count(eps_per)
    synth_t = n_rot * t_per
    # Synthesized sequences interleave T's with Clifford gates (~1:1).
    synth_clifford = synth_t

    return {
        "logical_qubits": int(res["num_qubits"]),
        "clifford_gate_count": exact_clifford + synth_clifford,
        "t_count": exact_t + synth_t,
        "total_depth": int(res["depth"] + synth_t),
        "t_depth": int(res["t_depth"] + synth_t),
        "gate_counts": gc,
        "resource_counts_u_cx": None,
        "n_arbitrary_rotations": int(n_rot),
        "rotation_synthesis": {
            "model": "ross_selinger ~ 3 log2(1/eps) T per rotation",
            "prep_tol": prep_tol,
            "eps_per_rotation": eps_per,
            "t_per_rotation": int(t_per),
            "exact_t": exact_t,
            "synthesized_t": int(synth_t),
            "exact_clifford": int(exact_clifford),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multidimensional periodic Laplacian block-encoding unit "
        "U_L^(D): dense verification at small n, FT resource counts at target n."
    )
    parser.add_argument(
        "--dims",
        type=int,
        choices=(2, 3),
        default=2,
        help="Spatial dimension D (2 or 3).",
    )
    parser.add_argument(
        "--verify-n",
        type=int,
        default=2,
        help="Per-dimension register size for the dense block check (default 2).",
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=20,
        help="Per-dimension register size for the resource estimate (default 20).",
    )
    parser.add_argument(
        "--prep-tol",
        type=float,
        default=1e-8,
        help="3D selector-preparation synthesis tolerance (default 1e-8).",
    )
    parser.add_argument(
        "--save-prefix",
        type=Path,
        default=Path("nd"),
        help="Prefix for the saved ASCII circuit drawing.",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=None,
        help="Where to write metrics JSON (default laplacian_<D>d_resource_metrics.json).",
    )
    args = parser.parse_args()

    if args.verify_n < 1:
        raise ValueError("--verify-n must be >= 1.")
    if args.target_n < 1:
        raise ValueError("--target-n must be >= 1.")
    if not 0.0 < args.prep_tol < 1.0:
        raise ValueError("--prep-tol must satisfy 0 < tolerance < 1.")

    dims = args.dims
    k = num_k_qubits(dims)
    out_path = args.output_metrics or Path(f"laplacian_{dims}d_resource_metrics.json")

    print(f"=== {dims}D periodic Laplacian block-encoding basic unit (U_L^({dims})) ===")
    print("scope            : classical L~_p^(D) + U_L^(D), single segment")
    print("shift model      : reversible modular increment/decrement (scalable)")
    print(f"selector qubits  : K = {k}")
    print(f"verification n   : n = {args.verify_n} (per dimension)")
    print(
        f"target size      : n = {args.target_n} per dim "
        f"(N = 2**{args.target_n} per axis, grid = 2**{args.target_n * dims})"
    )
    if dims == 3:
        print(f"prep tolerance   : {args.prep_tol:.1e} (3D selector synthesis)")

    # --- correctness verification (dense, small n). ---
    print("\n--- correctness verification (dense, small n) ---")
    be_err = verify_block_encoding_nd(args.verify_n, dims)
    precision_label = " [machine precision]" if be_err <= 1e-12 else ""
    print(
        f"n={args.verify_n}: <0..0|U_L^({dims})|0..0> - L~_p^({dims}) "
        f"max err = {be_err:.2e}{precision_label}"
    )

    # --- register / qubit breakdown at target n. ---
    logical_before = 2 + k + dims * args.target_n
    print(f"\n--- register layout (n={args.target_n}) ---")
    print(f"block-encoding ancillas : 2")
    print(f"selector qubits         : {k}")
    print(f"system qubits           : {dims} x {args.target_n} = {dims * args.target_n}")
    print(f"logical qubits (pre-decomp) : {logical_before}")

    target_ctrl_counts = mcx_control_counts(args.target_n, 1 + k)
    print(f"controlled shifts       : {2 * dims} (S^+ and S^- per dimension)")
    print(
        f"per-shift MCX controls  : C^kX for k in "
        f"[{target_ctrl_counts[0]}..{target_ctrl_counts[-1]}] "
        f"(1 ancilla + {k} selector control(s) on every gate)"
    )

    est = estimate_resources(args.target_n, dims, args.prep_tol)
    if dims == 2:
        res_ucx = est["resource_counts_u_cx"]
        print(f"\n--- decomposed resource counts (n={args.target_n}) ---")
        print(f"[basis u, cx] total gates : {res_ucx['total_gates']}")
        print(f"[basis u, cx] gate counts : {res_ucx['gate_counts']}")
        print(f"[basis u, cx] depth       : {res_ucx['depth']}")
        print(f"[basis u, cx] qubits      : {res_ucx['num_qubits']}")

    # --- fault-tolerant resource estimate at target n. ---
    print(f"\n--- fault-tolerant resource estimate (Clifford+T, n={args.target_n}) ---")
    print(f"logical qubits      : {est['logical_qubits']}")
    print(f"Clifford gate count : {est['clifford_gate_count']}")
    print(f"T-count             : {est['t_count']}")
    print(f"total depth         : {est['total_depth']}")
    print(f"T-depth             : {est['t_depth']}")
    if dims == 3:
        rs = est["rotation_synthesis"]
        print(f"arbitrary rotations : {est['n_arbitrary_rotations']} "
              f"(eps/rot = {rs['eps_per_rotation']:.2e}, "
              f"{rs['t_per_rotation']} T/rot)")

    # --- persist metrics + the target-n circuit drawing. ---
    metrics = {
        "scope": f"{dims}d_periodic_laplacian_block_encoding_unit",
        "dims": dims,
        "selector_qubits": k,
        "verify_n": args.verify_n,
        "target_n": args.target_n,
        "prep_tol": args.prep_tol if dims == 3 else None,
        "grid_points_target": 2 ** (args.target_n * dims),
        "block_encoding_err": be_err,
        "logical_qubits_pre_decomp": logical_before,
        "controlled_shifts": 2 * dims,
        "mcx_control_counts_per_shift": target_ctrl_counts,
        "fault_tolerant_summary": {
            "logical_qubits": est["logical_qubits"],
            "clifford_gate_count": est["clifford_gate_count"],
            "t_count": est["t_count"],
            "total_depth": est["total_depth"],
            "t_depth": est["t_depth"],
        },
        "n_arbitrary_rotations": est["n_arbitrary_rotations"],
        "rotation_synthesis": est["rotation_synthesis"],
        "resource_counts_u_cx": est["resource_counts_u_cx"],
        "gate_counts": est["gate_counts"],
    }
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", "utf-8")

    # Draw the same circuit size used for resource estimation. Verification
    # remains at verify_n because dense matrix construction only scales to
    # small n, but the ASCII circuit should visibly follow --target-n.
    draw_n = args.target_n
    draw_path = Path(f"laplacian_{args.save_prefix}_{dims}d_U_L.txt")
    draw_text = (
        f"=== {dims}D periodic Laplacian block encoding U_L^({dims}) "
        "(scalable form) ===\n"
        f"drawn at n = {draw_n} per dimension; "
        f"<0..0|_{{l,sel}} U_L |0..0> = L~_p^({dims}).\n\n"
        "registers:\n"
        "  q_0, q_1        = block-encoding ancillas l0, l1\n"
        f"  q_2..q_{2 + k - 1}        = selector (K = {k})\n"
        f"  q_{2 + k}..           = {dims} system registers of n = {draw_n} qubits\n\n"
        "U_prep_k prepares (1/sqrt D) sum_d |d>; shifts are selector-controlled.\n\n"
        "--- schematic form (shifts as opaque S+/S- boxes) ---\n\n"
        f"{build_u_l_nd_schematic(draw_n, dims).draw(output='text')}\n\n"
        "--- counted form: each shift is a C^kX cascade carrying the ancilla\n"
        f"    and the K = {k} selector qubits as extra controls, so k runs over\n"
        f"    {mcx_control_counts(draw_n, 1 + k)} at n = {draw_n} ---\n\n"
        f"{build_u_l_nd(draw_n, dims).draw(output='text')}\n"
    )
    draw_path.write_text(draw_text, encoding="utf-8")

    print(
        f"\nSaved: {out_path} and {draw_path} "
        f"(circuit drawing uses target n = {draw_n})"
    )


if __name__ == "__main__":
    main()
