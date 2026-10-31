"""Inner QSVT polynomial P_2d(theta) ~ sinc^2(pi theta / 2) and its phases.

    P_2d(theta) = 2 sum_{k=0}^{d} (-1)^k pi^{2k} theta^{2k} / (2k + 2)!

For odd d the alternating partial sum lies below sinc^2, so |P_2d| <= 1 on
[-1, 1] without rescaling, and the truncation error is at most
2 pi^{2d+2} / (2d + 4)!.

Phases follow the repository's QSP convention (pyqsp sym_qsp, Wx signal,
polynomial in Im <0|U|0>). ``psi`` are the reflection-free phases used by
the inner circuit R(psi_0) U_X R(psi_1) ... U_X R(psi_{2d}).

Usage:
  sinc2-phases --d 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from ..paths import MPLCONFIG_DIR, PHASES_DIR, PLOTS_DIR, ensure_parent

os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_DIR)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import chebyshev as C
from pyqsp import angle_sequence

DEFAULT_PHASES_JSON = PHASES_DIR / "phases_sinc2.json"


def sinc2_target(theta: np.ndarray) -> np.ndarray:
    """Eigenvalue function of L_cond: sinc^2(pi theta / 2)."""
    x = 0.5 * np.pi * np.asarray(theta, dtype=float)
    return np.sinc(x / np.pi) ** 2


def p2d_monomial(d: int) -> np.ndarray:
    """Monomial coefficients of P_2d (degree 2d)."""
    coef = np.zeros(2 * d + 1)
    for k in range(d + 1):
        coef[2 * k] = 2.0 * (-1) ** k * math.pi ** (2 * k) / math.factorial(2 * k + 2)
    return coef


def p2d(theta: np.ndarray, d: int) -> np.ndarray:
    return np.polynomial.polynomial.polyval(np.asarray(theta, dtype=float), p2d_monomial(d))


def truncation_bound(d: int) -> float:
    """Alternating-series remainder 2 pi^{2d+2} / (2d+4)!."""
    return 2.0 * math.pi ** (2 * d + 2) / math.factorial(2 * d + 4)


def minimum_odd_d(eps_sinc: float) -> int:
    """Smallest odd d whose truncation bound is at most eps_sinc."""
    d = 1
    while truncation_bound(d) > eps_sinc:
        d += 2
    return d


def reflection_free_phases(phi: list[float]) -> list[float]:
    """Absorb each walk reflection I - 2 Pi = i R(pi/2) into the phases.

    R(phi_0) W R(phi_1) ... W R(phi_D) with W = (I - 2 Pi) U_X equals
    i^D R(-pi/4) [R(psi_0) U_X R(psi_1) ... U_X R(psi_D)] R(pi/4), whose block
    on Pi is i^D times that of the bracket.
    """
    psi = [p + math.pi / 2 for p in phi]
    psi[0] -= math.pi / 4
    psi[-1] -= math.pi / 4
    return psi


def block_phase_factor(num_u_x: int) -> complex:
    """c with Herm(c <0|R(psi_0) U_X ... R(psi_D)|0>) = P_2d(X); c = -i^{D+1}."""
    return -(1j ** (num_u_x + 1))


def sinc2_phases(d: int, num_points: int = 2001) -> dict:
    """Phases and errors for the inner sinc^2 QSVT polynomial."""
    if d < 1 or d % 2 == 0:
        raise ValueError("d must be odd and >= 1 so that |P_2d| <= 1.")
    cheb = C.poly2cheb(p2d_monomial(d))
    ret = angle_sequence.QuantumSignalProcessingPhases(
        cheb, method="sym_qsp", chebyshev_basis=True
    )
    phases = np.asarray(ret[0] if isinstance(ret, tuple) else ret, dtype=float)
    x = np.linspace(-1.0, 1.0, num_points)
    response = angle_sequence.ComputeQSPResponse(
        x, phases, signal_operator="Wx", sym_qsp=True
    )
    p_qsp = np.imag(response["pdat"])
    p_poly = p2d(x, d)
    target = sinc2_target(x)
    return {
        "target_function": "sinc2_pi_theta_over_2",
        "polynomial": "P_2d(theta) = 2 sum_{k=0}^{d} (-1)^k pi^{2k} theta^{2k} / (2k+2)!",
        "d": d,
        "degree": 2 * d,
        "parity": 0,
        "num_u_x_applications": 2 * d,
        "num_signal_rotations": len(phases),
        "truncation_bound": truncation_bound(d),
        "poly_vs_sinc2_max_abs_error": float(np.max(np.abs(p_poly - target))),
        "phase_vs_poly_max_abs_error": float(np.max(np.abs(p_qsp - p_poly))),
        "max_abs_poly_on_domain": float(np.max(np.abs(p_poly))),
        "minimum_on_unit_interval": float(np.min(p_poly[x >= 0])),
        "phases_symmetric": bool(np.allclose(phases, phases[::-1], atol=1e-14)),
        "block_phase_factor": [block_phase_factor(2 * d).real, block_phase_factor(2 * d).imag],
        "monomial_coeffs": p2d_monomial(d).tolist(),
        "cheb_coeffs": np.asarray(cheb, dtype=float).tolist(),
        "phases": phases.tolist(),
        "psi": reflection_free_phases(phases.tolist()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--d", type=int, default=None, help="Odd truncation index (degree 2d).")
    group.add_argument(
        "--eps-sinc", type=float, default=None,
        help="Choose the smallest odd d with truncation bound <= eps-sinc.",
    )
    parser.add_argument("--output-phases", type=Path, default=DEFAULT_PHASES_JSON)
    parser.add_argument("--output-plot", type=Path, default=PLOTS_DIR / "qsp_sinc2_fit.png")
    args = parser.parse_args()

    d = args.d if args.d is not None else (
        minimum_odd_d(args.eps_sinc) if args.eps_sinc is not None else 5
    )
    payload = sinc2_phases(d)
    ensure_parent(args.output_phases).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    x = np.linspace(0.0, 1.0, 801)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    top.plot(x, sinc2_target(x), linewidth=2.0, label="sinc^2(pi theta/2) (L_cond eigenvalues)")
    top.plot(x, p2d(x, d), "--", linewidth=1.8, label=f"P_{2 * d}(theta), d = {d}")
    top.axhline(0.4, color="gray", linestyle=":", label="outer inversion domain edge 1/kappa = 0.4")
    top.set_ylabel("value")
    top.legend(loc="lower left")
    top.grid(alpha=0.3)
    bottom.semilogy(x, np.abs(p2d(x, d) - sinc2_target(x)) + 1e-18)
    bottom.axhline(truncation_bound(d), color="k", linestyle=":", label="truncation bound")
    bottom.set_xlabel("theta = 2 m_k / N")
    bottom.set_ylabel("|P_2d - sinc^2|")
    bottom.legend(loc="upper left")
    bottom.grid(alpha=0.3)
    fig.suptitle(f"Inner sinc^2 QSVT polynomial: degree {2 * d}, {len(payload['phases'])} phases")
    fig.tight_layout()
    fig.savefig(ensure_parent(args.output_plot), dpi=160)

    print("=== inner sinc^2 QSVT phases (pyqsp sym_qsp) ===")
    for key in (
        "d", "degree", "num_u_x_applications", "num_signal_rotations", "truncation_bound",
        "poly_vs_sinc2_max_abs_error", "phase_vs_poly_max_abs_error",
        "max_abs_poly_on_domain", "minimum_on_unit_interval", "phases_symmetric",
    ):
        print(f"{key:<28}: {payload[key]}")
    print(f"Saved: {args.output_phases} and {args.output_plot}")


if __name__ == "__main__":
    main()
