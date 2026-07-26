#!/usr/bin/env python3
"""
Find one-qubit QSP phase angles for approximating a target f(a) on a in [0, 1].

Usage:
  python one_qubit_phase_rotations.py --degree 12 "cos(x)"
  python one_qubit_phase_rotations.py --degree 20 "exp(-x**2)"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Keep matplotlib cache writable inside the project directory.
os.environ["MPLCONFIGDIR"] = str(Path(".mplconfig").resolve())
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyqsp import angle_sequence
from pyqsp.poly import PolyTaylorSeries

DEFAULT_MAX_SCALE = 0.9
DEFAULT_NUM_POINTS = 401


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use pyqsp to synthesize one-qubit phase angles for P(a) ~ f(a)."
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=12,
        help="Polynomial degree used for the Chebyshev approximation.",
    )
    parser.add_argument(
        "target_function",
        nargs="?",
        default="cos(x)",
        help='Target function expression in variable x, e.g. "cos(x)" or "exp(-x**2)".',
    )
    parser.add_argument(
        "--output-phases",
        type=Path,
        default=Path("phases_target.json"),
        help="Where to write synthesized phases as JSON.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("qsp_target_fit.png"),
        help="Where to save the verification plot.",
    )
    return parser


def make_target_function(expr: str):
    allowed_names = {
        "np": np,
        "x": 0.0,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
        "e": np.e,
    }

    def target(x):
        local_dict = dict(allowed_names)
        local_dict["x"] = x
        return eval(expr, {"__builtins__": {}}, local_dict)

    # Fail fast for invalid expressions before synthesis.
    try:
        test_vals = target(np.array([0.0, 0.5, 1.0]))
        np.asarray(test_vals, dtype=float)
    except Exception as exc:
        raise ValueError(
            f'Invalid target expression "{expr}". Use x as the variable, e.g. "cos(x)".'
        ) from exc

    return target


def build_eval_grid(num_points: int) -> np.ndarray:
    if num_points < 2:
        raise ValueError("num_points must be >= 2.")
    return np.linspace(0.0, 1.0, num_points)


def format_chebyshev_series(coeffs: np.ndarray, variable: str = "x", tol: float = 1e-14) -> str:
    """
    Format coefficients c_k into a readable Chebyshev series:
      P(x) = sum_k c_k T_k(x)
    """
    terms = []
    for k, c in enumerate(np.asarray(coeffs, dtype=float)):
        if abs(c) < tol:
            continue
        if k == 0:
            basis = "1"
        elif k == 1:
            basis = f"T1({variable})"
        else:
            basis = f"T{k}({variable})"
        terms.append(f"{c:.16e}*{basis}")
    if not terms:
        return "0"
    return " + ".join(terms)


def main() -> None:
    args = build_parser().parse_args()

    if args.degree < 1:
        raise ValueError("--degree must be >= 1.")
    target_expr = args.target_function.strip()
    target = make_target_function(target_expr)

    max_scale = DEFAULT_MAX_SCALE
    num_points = DEFAULT_NUM_POINTS

    # Build a bounded Chebyshev approximation for sym_qsp angle synthesis.
    poly_cheb, scale = PolyTaylorSeries().taylor_series(
        func=target,
        degree=args.degree,
        ensure_bounded=True,
        return_scale=True,
        max_scale=max_scale,
        chebyshev_basis=True,
        cheb_samples=2 * args.degree,
    )
    cheb_coeffs = (
        np.asarray(poly_cheb.coef, dtype=float)
        if hasattr(poly_cheb, "coef")
        else np.asarray(poly_cheb, dtype=float)
    )

    # For sym_qsp, the polynomial response appears in imag(<0|U_phi(a)|0>).
    phases, _, parity = angle_sequence.QuantumSignalProcessingPhases(
        poly_cheb,
        method="sym_qsp",
        chebyshev_basis=True,
    )

    phases = np.asarray(phases, dtype=float)
    effective_degree = len(phases) - 1

    a_grid = build_eval_grid(num_points=num_points)
    qsp_response = angle_sequence.ComputeQSPResponse(
        a_grid,
        phases,
        signal_operator="Wx",
        sym_qsp=True,
    )
    p_of_a = np.imag(qsp_response["pdat"])
    if callable(poly_cheb):
        p_target_poly = np.asarray(poly_cheb(a_grid), dtype=float)
    else:
        p_target_poly = np.polynomial.chebyshev.chebval(a_grid, poly_cheb)
    target_scaled = scale * target(a_grid)

    poly_max_abs_err = float(np.max(np.abs(p_target_poly - target_scaled)))
    poly_rms_err = float(np.sqrt(np.mean((p_target_poly - target_scaled) ** 2)))
    phase_max_abs_err = float(np.max(np.abs(p_of_a - p_target_poly)))
    phase_rms_err = float(np.sqrt(np.mean((p_of_a - p_target_poly) ** 2)))
    max_abs_err = float(np.max(np.abs(p_of_a - target_scaled)))
    rms_err = float(np.sqrt(np.mean((p_of_a - target_scaled) ** 2)))

    args.output_phases.write_text(
        json.dumps(
            {
                "degree": args.degree,
                "effective_degree": int(effective_degree),
                "target_function": target_expr,
                "max_scale": max_scale,
                "returned_scale": float(scale),
                "parity": int(parity),
                "evaluation_domain": "[0,1]",
                "poly_max_abs_error": poly_max_abs_err,
                "poly_rms_error": poly_rms_err,
                "phase_max_abs_error": phase_max_abs_err,
                "phase_rms_error": phase_rms_err,
                "total_max_abs_error": max_abs_err,
                "total_rms_error": rms_err,
                "phases": phases.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        a_grid,
        target_scaled,
        label=f"scaled target: scale*({target_expr})",
        linewidth=2.0,
    )
    plt.plot(a_grid, p_of_a, "--", label="QSP P(a) = imag(<0|U|0>)", linewidth=2.0)
    plt.xlabel("a")
    plt.ylabel("value")
    plt.title(f"QSP fit on [0,1], degree={args.degree}, parity={parity}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=160)

    print("=== one-qubit phase-rotation synthesis (pyqsp) ===")
    print(f"target         : {target_expr}")
    print(f"degree         : {args.degree}")
    print(f"effective_degree: {effective_degree}")
    print(f"max_scale      : {max_scale}")
    print(f"returned_scale : {float(scale):.8f}")
    print(f"parity         : {parity}")
    print("evaluation_domain: [0,1]")
    print(f"num_phases     : {len(phases)}")
    print(f"poly_max_abs_error : {poly_max_abs_err:.6e}")
    print(f"poly_rms_error     : {poly_rms_err:.6e}")
    print(f"phase_max_abs_error: {phase_max_abs_err:.6e}")
    print(f"phase_rms_error   : {phase_rms_err:.6e}")
    print(f"max_abs_error  : {max_abs_err:.6e}")
    print(f"rms_error      : {rms_err:.6e}")
    print(f"phases_json    : {args.output_phases}")
    print(f"plot_png       : {args.output_plot}")
    print("\nchebyshev_coeffs (c_k for sum c_k T_k(x)):")
    print(np.array2string(cheb_coeffs, precision=10, separator=", "))
    print("\nchebyshev_series:")
    print(format_chebyshev_series(cheb_coeffs, variable="x"))
    print("\nphases (radians):")
    print(np.array2string(phases, precision=10, separator=", "))


if __name__ == "__main__":
    main()
