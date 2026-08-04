#!/usr/bin/env python3
"""
Qubitization signal unitary (walk operator) W for the periodic Laplacian.

W = U_R . X_q . U'          (rightmost factor applied first)

  U'  = |0><0|_q (x) U_L  +  |1><1|_q (x) U_L^dag
  X_q = Pauli-X on the qubitization control qubit q
  U_R = I - 2|Pi><Pi|,   |Pi> = |+>_q |0>^m,   m = 2 + K

U_L^(D) is the D-dimensional Laplacian block-encoding unit from
laplacian_nd_resource.py, and the signal property that makes W useful is

  <Pi| W |Pi> = -L~_p^(D),

which is what the dense check at small n confirms here.

Usage:
  python laplacian_w_resource.py --dims 2
  python laplacian_w_resource.py --dims 3 --prep-tol 1e-8
  python laplacian_w_resource.py --dims 2 --target-n 4
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
from qiskit.quantum_info import Operator

from laplacian_1d_resource import mcx_control_counts, resource_counts
from laplacian_nd_resource import (
    CT_BASIS,
    apply_u_l_nd,
    build_u_l_nd,
    count_prep_rotations,
    num_k_qubits,
    ross_selinger_t_count,
    scaled_nd_laplacian,
    sys_registers,
)

# --- register layout ---------------------------------------------------------
def w_layout(n: int, dims: int) -> dict:
    """
    Qubit map of W: q, then the U_L registers in their usual order.

      q  = 0                    qubitization control
      l0 = 1, l1 = 2            block-encoding ancillas
      sel= 3 .. 3+K-1           dimension selector
      sys= 3+K ..               D system registers of n qubits each
    """
    k = num_k_qubits(dims)
    return {
        "q": 0,
        "anc": [1, 2],
        "sel": list(range(3, 3 + k)),
        "sys_regs": sys_registers(n, dims, 3 + k),
        "k": k,
        "m": 2 + k,
        "total": 3 + k + dims * n,
    }


# --- U': controlled U_L on one branch, controlled U_L^dag on the other -------
def build_controlled_u_l(n: int, dims: int, omit_rotation: bool = False) -> QuantumCircuit:
    """q-controlled U_L^(D), with the control pushed into the individual gates."""
    lay = w_layout(n, dims)
    qc = QuantumCircuit(lay["total"], name="C-U_L")
    apply_u_l_nd(
        qc,
        n,
        dims,
        anc=lay["anc"],
        sel=lay["sel"],
        sys_regs=lay["sys_regs"],
        extra_controls=[lay["q"]],
        omit_rotation=omit_rotation,
    )
    return qc


def build_u_prime(n: int, dims: int, omit_rotation: bool = False) -> QuantumCircuit:
    """U' = |0><0|_q (x) U_L + |1><1|_q (x) U_L^dag."""
    lay = w_layout(n, dims)
    cu_l = build_controlled_u_l(n, dims, omit_rotation)

    qc = QuantumCircuit(lay["total"], name="U'")
    # U_L on the q = 0 branch: the X pair turns the closed q control into an open one.
    qc.x(lay["q"])
    qc.compose(cu_l, inplace=True)
    qc.x(lay["q"])
    # U_L^dag on the q = 1 branch. QuantumCircuit.inverse() reverses and inverts
    # the explicit gate list, so the cheap C^kX structure survives; taking the
    # inverse of a composite gate instead would hide it from the transpiler.
    qc.compose(cu_l.inverse(), inplace=True)
    return qc


# --- U_R: reflection about |Pi> = |+>_q |0>^m -------------------------------
def build_reflection_circuit(m: int) -> QuantumCircuit:
    """
    U_R = I - 2|Pi><Pi| on q plus the m block-encoding/selector ancillas.
    """
    qc = QuantumCircuit(1 + m, name="U_R")
    allq = list(range(1 + m))
    qc.h(0)
    qc.x(allq)
    qc.h(m)
    qc.mcx(allq[:-1], m)
    qc.h(m)
    qc.x(allq)
    qc.h(0)
    return qc


# --- W ----------------------------------------------------------------------
def build_w_operator(n: int, dims: int, omit_rotation: bool = False) -> QuantumCircuit:
    """W = U_R . X_q . U' as one flat, transpiler-friendly circuit."""
    lay = w_layout(n, dims)
    qc = QuantumCircuit(lay["total"], name=f"W_{dims}D")
    qc.compose(build_u_prime(n, dims, omit_rotation), inplace=True)
    qc.x(lay["q"])  # S = X_q
    qc.compose(
        build_reflection_circuit(lay["m"]),
        qubits=[lay["q"]] + lay["anc"] + lay["sel"],
        inplace=True,
    )
    return qc


def build_w_schematic(n: int, dims: int) -> QuantumCircuit:
    """
    Same unitary as build_w_operator, drawn with opaque U_L / U_L^dag / U_R boxes.

    Presentation only: the counted circuit is build_w_operator, whose explicit
    C^{k+1}X cascades are what the transpiler actually sees.
    """
    lay = w_layout(n, dims)
    qc = QuantumCircuit(lay["total"], name=f"W_{dims}D")
    u_l = build_u_l_nd(n, dims)
    targets = list(range(1, lay["total"]))
    qc.append(u_l.to_gate(label="U_L").control(1, ctrl_state=0), [lay["q"]] + targets)
    qc.append(u_l.inverse().to_gate(label="U_L+").control(1), [lay["q"]] + targets)
    qc.x(lay["q"])
    qc.append(
        build_reflection_circuit(lay["m"]).to_gate(label="U_R"),
        [lay["q"]] + lay["anc"] + lay["sel"],
    )
    return qc


# --- classical references (dense, small n only) -----------------------------
def signal_state(m: int) -> np.ndarray:
    """|Pi> = |+>_q (x) |0>^m, with q as the least significant qubit."""
    vec = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
    ket0 = np.array([1.0, 0.0], dtype=complex)
    for _ in range(m):
        vec = np.kron(ket0, vec)
    return vec

def verify_reflection(m: int) -> float:
    """Check the gate sequence for U_R against the dense I - 2|Pi><Pi|."""
    sig = signal_state(m)
    target = np.eye(2 ** (1 + m), dtype=complex) - 2.0 * np.outer(sig, sig.conj())
    got = Operator(build_reflection_circuit(m)).data
    return float(np.max(np.abs(got - target)))


def verify_u_prime(n: int, dims: int) -> float:
    """Check U' against |0><0|_q (x) U_L + |1><1|_q (x) U_L^dag."""
    u_l = Operator(build_u_l_nd(n, dims)).data
    p0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    p1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    target = np.kron(u_l, p0) + np.kron(u_l.conj().T, p1)
    got = Operator(build_u_prime(n, dims)).data
    return float(np.max(np.abs(got - target)))


def verify_w_signal_block(n: int, dims: int) -> tuple[float, float]:
    """
    Check the qubitization signal property <Pi| W |Pi> = -L~_p^(D).
    """
    w = Operator(build_w_operator(n, dims)).data
    m = 2 + num_k_qubits(dims)
    dim_sys = (2**n) ** dims
    # The 1+m ancilla qubits are the low-order bits, so they are the right factor.
    proj = np.kron(np.eye(dim_sys, dtype=complex), signal_state(m).reshape(-1, 1))
    block = proj.conj().T @ w @ proj
    sig_err = float(np.max(np.abs(block + scaled_nd_laplacian(n, dims))))
    unitary_err = float(np.max(np.abs(w.conj().T @ w - np.eye(w.shape[0]))))
    return sig_err, unitary_err


# --- resource estimation (target n, no dense math) --------------------------
def count_w_rotations(dims: int) -> int:
    return 4 * count_prep_rotations(dims)


def in_w_register(sub: QuantumCircuit, qubits: list[int], n: int, dims: int) -> QuantumCircuit:
    """
    Embed a component of W into the full W register, leaving the rest idle.
    """
    lay = w_layout(n, dims)
    qc = QuantumCircuit(lay["total"], name=sub.name)
    qc.compose(sub, qubits=qubits, inplace=True)
    return qc


def estimate_w_resources(n: int, dims: int, prep_tol: float) -> dict:
    """Fault-tolerant (Clifford+T) resource estimate for W at size n."""
    lay = w_layout(n, dims)
    n_rot = count_w_rotations(dims)
    # Drop the arbitrary rotations so the bulk is exactly Clifford+T, then price
    # them separately; for 2D there are none and the bulk is the whole circuit.
    bulk = build_w_operator(n, dims, omit_rotation=n_rot > 0)
    res = resource_counts(bulk, CT_BASIS, count_t_depth=True)
    gc = res["gate_counts"]
    exact_t = int(gc.get("t", 0) + gc.get("tdg", 0))
    exact_clifford = int(res["total_gates"] - exact_t)

    eps_per = prep_tol / max(n_rot, 1)
    t_per = ross_selinger_t_count(eps_per) if n_rot else 0
    synth_t = n_rot * t_per
    # Synthesized sequences interleave T's with Clifford gates (~1:1).
    synth_clifford = synth_t

    out = {
        "logical_qubits": int(res["num_qubits"]),
        "clifford_gate_count": exact_clifford + synth_clifford,
        "t_count": exact_t + synth_t,
        "total_depth": int(res["depth"] + synth_t),
        "t_depth": int(res["t_depth"] + synth_t),
        "gate_counts": gc,
        "n_arbitrary_rotations": int(n_rot),
        "rotation_synthesis": None,
        "resource_counts_u_cx": None,
        "reflection": resource_counts(
            in_w_register(
                build_reflection_circuit(lay["m"]),
                [lay["q"]] + lay["anc"] + lay["sel"],
                n,
                dims,
            ),
            CT_BASIS,
            count_t_depth=True,
        ),
    }
    if n_rot:
        out["rotation_synthesis"] = {
            "model": "ross_selinger ~ 3 log2(1/eps) T per rotation",
            "prep_tol": prep_tol,
            "eps_per_rotation": eps_per,
            "t_per_rotation": int(t_per),
            "exact_t": exact_t,
            "synthesized_t": int(synth_t),
            "exact_clifford": int(exact_clifford),
        }
    else:
        # Coarse logical reference, matching the U_L report.
        out["resource_counts_u_cx"] = resource_counts(bulk, ["u", "cx"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qubitization walk operator W = U_R X_q U' for the periodic "
        "Laplacian: dense verification at small n, FT resource counts at target n."
    )
    parser.add_argument("--dims", type=int, choices=(2, 3), default=2,
                        help="Spatial dimension D (2 or 3).")
    parser.add_argument("--verify-n", type=int, default=2,
                        help="Per-dimension register size for the dense checks.")
    parser.add_argument("--target-n", type=int, default=20,
                        help="Per-dimension register size for the resource estimate.")
    parser.add_argument("--prep-tol", type=float, default=1e-8,
                        help="3D selector-preparation synthesis tolerance.")
    parser.add_argument("--output-metrics", type=Path, default=None,
                        help="Where to write metrics JSON.")
    args = parser.parse_args()

    if args.verify_n < 1 or args.target_n < 1:
        raise ValueError("--verify-n and --target-n must be >= 1.")
    if not 0.0 < args.prep_tol < 1.0:
        raise ValueError("--prep-tol must satisfy 0 < tolerance < 1.")

    dims, n_t = args.dims, args.target_n
    lay = w_layout(n_t, dims)
    k, m = lay["k"], lay["m"]
    out_path = args.output_metrics or Path(f"laplacian_{dims}d_w_resource_metrics.json")

    print(f"=== {dims}D periodic Laplacian qubitization walk operator W ===")
    print("definition       : W = U_R . X_q . U'")
    print("U'               : |0><0|_q (x) U_L  +  |1><1|_q (x) U_L^dag")
    print(f"U_R              : I - 2|Pi><Pi|, |Pi> = |+>_q |0>^{m}")
    print(f"selector qubits  : K = {k}")
    print(f"verification n   : n = {args.verify_n} (per dimension)")
    print(f"target size      : n = {n_t} per dim (grid = 2**{n_t * dims})")
    if dims == 3:
        print(f"prep tolerance   : {args.prep_tol:.1e} (3D selector synthesis)")

    # --- correctness verification (dense, small n). ---
    print("\n--- correctness verification (dense, small n) ---")
    ref_err = verify_reflection(m)
    up_err = verify_u_prime(args.verify_n, dims)
    sig_err, uni_err = verify_w_signal_block(args.verify_n, dims)

    def label(err: float) -> str:
        return " [machine precision]" if err <= 1e-12 else ""

    print(f"U_R  - (I - 2|Pi><Pi|)                 max err = {ref_err:.2e}{label(ref_err)}")
    print(f"U'   - (|0><0|U_L + |1><1|U_L^dag)     max err = {up_err:.2e}{label(up_err)}")
    print(f"W^dag W - I                            max err = {uni_err:.2e}{label(uni_err)}")
    print(f"<Pi|W|Pi> - (-L~_p^({dims}))              max err = {sig_err:.2e}{label(sig_err)}")

    # --- register / qubit breakdown at target n. ---
    print(f"\n--- register layout (n={n_t}) ---")
    print(f"qubitization control    : 1 (q)")
    print(f"block-encoding ancillas : 2")
    print(f"selector qubits         : {k}")
    print(f"system qubits           : {dims} x {n_t} = {dims * n_t}")
    print(f"logical qubits (pre-decomp) : {lay['total']}")

    ctrl_counts = mcx_control_counts(n_t, 1 + k + 1)
    print(f"controlled shifts       : {4 * dims} "
          f"(S^+ and S^- per dimension, in U_L and in U_L^dag)")
    print(f"per-shift MCX controls  : C^kX for k in "
          f"[{ctrl_counts[0]}..{ctrl_counts[-1]}] "
          f"(1 ancilla + {k} selector + 1 qubitization control on every gate)")
    print(f"U_R multi-controlled X  : 1 x C^{m}X")

    est = estimate_w_resources(n_t, dims, args.prep_tol)

    if est["resource_counts_u_cx"] is not None:
        res_ucx = est["resource_counts_u_cx"]
        print(f"\n--- decomposed resource counts (n={n_t}) ---")
        print(f"[basis u, cx] total gates : {res_ucx['total_gates']}")
        print(f"[basis u, cx] gate counts : {res_ucx['gate_counts']}")
        print(f"[basis u, cx] depth       : {res_ucx['depth']}")
        print(f"[basis u, cx] qubits      : {res_ucx['num_qubits']}")

    print(f"\n--- fault-tolerant resource estimate (Clifford+T, n={n_t}) ---")
    print(f"logical qubits      : {est['logical_qubits']}")
    print(f"Clifford gate count : {est['clifford_gate_count']}")
    print(f"T-count             : {est['t_count']}")
    print(f"total depth         : {est['total_depth']}")
    print(f"T-depth             : {est['t_depth']}")
    if est["rotation_synthesis"] is not None:
        rs = est["rotation_synthesis"]
        print(f"arbitrary rotations : {est['n_arbitrary_rotations']} "
              f"(eps/rot = {rs['eps_per_rotation']:.2e}, "
              f"{rs['t_per_rotation']} T/rot)")

    ref = est["reflection"]
    ref_t = int(ref["gate_counts"].get("t", 0) + ref["gate_counts"].get("tdg", 0))
    print(f"\n--- U_R alone (Clifford+T) ---")
    print(f"T-count / Clifford / depth : "
          f"{ref_t} / {ref['total_gates'] - ref_t} / {ref['depth']}")

    # --- persist metrics + a small-n circuit drawing. ---
    metrics = {
        "scope": f"{dims}d_periodic_laplacian_qubitization_walk_operator",
        "definition": "W = U_R . X_q . U'",
        "dims": dims,
        "selector_qubits": k,
        "signal_ancillas_m": m,
        "verify_n": args.verify_n,
        "target_n": n_t,
        "prep_tol": args.prep_tol if dims == 3 else None,
        "verification": {
            "reflection_err": ref_err,
            "u_prime_err": up_err,
            "unitarity_err": uni_err,
            "signal_block_err": sig_err,
        },
        "logical_qubits_pre_decomp": lay["total"],
        "controlled_shifts": 4 * dims,
        "mcx_control_counts_per_shift": ctrl_counts,
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
        "reflection_counts": est["reflection"],
        "gate_counts": est["gate_counts"],
    }
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", "utf-8")

    # Draw the same circuit size used for resource estimation. Dense
    # verification remains at verify_n because it only scales to small n.
    draw_n = args.target_n
    draw_lay = w_layout(draw_n, dims)
    draw_path = Path(f"laplacian_w_{dims}d_W.txt")
    sel_lo, sel_hi = draw_lay["sel"][0], draw_lay["sel"][-1]
    sel_label = f"q_{sel_lo}" if sel_lo == sel_hi else f"q_{sel_lo}..q_{sel_hi}"
    draw_counts = mcx_control_counts(draw_n, 1 + k + 1)
    draw_text = (
        f"=== {dims}D periodic Laplacian qubitization walk operator W ===\n"
        f"drawn at n = {draw_n} per dimension (grid = 2**{draw_n * dims}); "
        f"<Pi|W|Pi> = -L~_p^({dims}).\n\n"
        "registers:\n"
        f"  q_{draw_lay['q']} = q = qubitization control\n"
        f"  q_{draw_lay['anc'][0]}, q_{draw_lay['anc'][1]} = block-encoding ancillas l0, l1\n"
        f"  {sel_label} = dimension selector (K = {draw_lay['k']})\n"
        f"  q_{draw_lay['sys_regs'][0][0]}.. = {dims} system registers of n = {draw_n} qubits\n\n"
        "--- schematic form (U_L, U_L+ and U_R as opaque boxes) ---\n"
        "Compact mathematical view: U' is controlled-U_L on the q=0 branch and\n"
        "controlled-U_L+ on the q=1 branch, then X_q, then the reflection U_R.\n\n"
        f"{build_w_schematic(draw_n, dims).draw(output='text')}\n\n"
        "--- counted form (every gate carries the q control explicitly) ---\n"
        f"Each shift is a cascade with MCX control counts {draw_counts} at "
        f"n = {draw_n} (the estimate at n = {n_t} uses {ctrl_counts}); this is "
        "the form passed to the transpiler.\n\n"
        f"{build_w_operator(draw_n, dims).draw(output='text')}\n"
    )
    draw_path.write_text(draw_text, encoding="utf-8")

    print("\nSaved:", out_path, "and", draw_path)


if __name__ == "__main__":
    main()
