"""
Matrix-inversion QSP phases.

The target matrix-inversion polynomial:

    P^{MI}_{eps,kappa}(x)
      = (1/(2*kappa)) * P^{1/x}_{eps/2, 2*kappa}(x) * P^{rect}_{eps',kappa}(x)

The user controls:
  --epsilon  target approximation error on the domain of validity
  --kappa    condition number; valid domain [1/kappa, 1] U [-1, -1/kappa]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path(".mplconfig").resolve())
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.special
from numpy.polynomial import chebyshev as C
from pyqsp import angle_sequence

DEFAULT_MAX_SCALE = 0.9
DEFAULT_NUM_POINTS = 601
DEFAULT_PEAK_MARGIN = 1e-3
MAX_RECT_DEGREE_RETRIES = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthesize QSP phase angles for the 1/x matrix-inversion polynomial."
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="Target approximation error on the domain of validity.",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        required=True,
        help="Condition number. Valid domain: [1/kappa,1] U [-1,-1/kappa].",
    )
    parser.add_argument(
        "--rect-degree",
        type=int,
        default=None,
        help="Override the even rectangular-filter degree.",
    )
    parser.add_argument(
        "--no-rect",
        action="store_true",
        help="Skip the rectangular filter and use only the 1/x approximation.",
    )
    parser.add_argument(
        "--output-phases",
        type=Path,
        default=Path("phases_inversion.json"),
        help="Where to write synthesized phases as JSON.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("qsp_inversion_fit.png"),
        help="Where to save the verification plot.",
    )
    return parser


def binomial_to_log_gamma_half(n: int, k: int) -> float:
    """Return 2**(-n) * C(n, k) as a float."""
    return math.exp(
        -n * math.log(2.0)
        + (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))
    )


def cks_b(a: float, c: float) -> int:
    """Eq. C1 exponent: b(a, c) = ceil(c**2 * log(c/a))."""
    return int(math.ceil((c**2) * math.log(c / a)))


def cks_D(a: float, c: float) -> int:
    """Eq. C2 truncation degree: D(a,c) = ceil(sqrt(b * log(4b/a)))."""
    b = cks_b(a, c)
    return int(math.ceil(math.sqrt(b * math.log((4.0 * b) / a))))


def rect_sharpness(delta: float, epsilon: float) -> float:
    """Eq. 49 sharpness k for erf step."""
    return (math.sqrt(2.0) / delta) * math.sqrt(math.log(2.0 / (math.pi * epsilon**2)))


def theta_degree(k: float, epsilon: float) -> int:
    """Step-function degree bound: n = ceil(sqrt((k^2 + log(1/eps)) log(1/eps)))."""
    log_inv = math.log(1.0 / epsilon)
    return int(math.ceil(math.sqrt((k**2 + log_inv) * log_inv)))


def build_one_over_x_cheb(b: int, D: int) -> C.Chebyshev:
    """
    Build truncated odd Chebyshev approximation to 1/x:

      P_g(x) = 4 * sum_{j=0}^{D} (-1)^j
               [ 2^{-2b} * sum_{i=j+1}^{b} binom(2b, b+i) ] * T_{2j+1}(x)
    """
    coef = np.zeros(2 * D + 2, dtype=float)
    for j in range(D + 1):
        inner_sum = 0.0
        for i in range(j + 1, b + 1):
            inner_sum += binomial_to_log_gamma_half(2 * b, b + i)
        coef[2 * j + 1] = 4.0 * ((-1.0) ** j) * inner_sum
    return C.Chebyshev(coef)


def build_rect_cheb(degree: int, delta: float, kappa: float, epsilon: float) -> C.Chebyshev:
    """
    Build an even Chebyshev fit of the smooth rectangular filter.

    Target:
      P_rect(x) = [1 + 0.5 * (Theta(x-3/(4kappa)) + Theta(-x-3/(4kappa)))] / (1+eps/2)
    where Theta is an erf-smoothed step.
    """
    if degree % 2 != 0:
        raise ValueError("rect-degree must be even.")

    k = rect_sharpness(delta, epsilon)

    def rect(x: np.ndarray) -> np.ndarray:
        return (
            1.0
            + 0.5
            * (
                scipy.special.erf((x - 3.0 / (4.0 * kappa)) * k)
                + scipy.special.erf((-x - 3.0 / (4.0 * kappa)) * k)
            )
        ) / (1.0 + epsilon / 2.0)

    samples = C.chebpts1(max(2 * degree, degree + 1))
    coefs = C.chebfit(samples, rect(samples), degree)
    coefs[1::2] = 0.0  # enforce even parity
    return C.Chebyshev(coefs)


def bound_poly(poly: C.Chebyshev, max_scale: float) -> tuple[C.Chebyshev, float, float]:
    """Rescale poly so max|poly(x)| over [-1,1] is max_scale."""
    grid = np.linspace(-1.0, 1.0, 4001)
    peak_grid = float(np.max(np.abs(poly(grid))))

    def neg_abs(x: np.ndarray) -> float:
        return -abs(float(poly(x[0])))

    try:
        res = scipy.optimize.shgo(neg_abs, bounds=[(-1.0, 1.0)])
        peak_opt = abs(float(poly(res.x[0])))
    except Exception:
        peak_opt = 0.0

    peak = max(peak_grid, peak_opt)
    if peak <= 0.0:
        return poly, 1.0, 0.0
    scale = max_scale / peak
    return C.Chebyshev(scale * poly.coef), float(scale), float(peak)


def main() -> None:
    args = build_parser().parse_args()

    if args.kappa <= 1.0:
        raise ValueError("--kappa must be > 1.")
    if args.epsilon <= 0.0:
        raise ValueError("--epsilon must be > 0.")
    if args.rect_degree is not None and args.rect_degree % 2 != 0:
        raise ValueError("--rect-degree must be even.")

    epsilon = float(args.epsilon)
    kappa = float(args.kappa)

    # Paper-style split: inverse-poly budget uses eps/4 at 2*kappa.
    eps_inv = epsilon / 4.0
    kappa_inv = 2.0 * kappa

    b = cks_b(eps_inv, kappa_inv)
    D_inv = cks_D(eps_inv, kappa_inv)
    if b < 1:
        raise ValueError("Derived b is < 1; reduce epsilon or increase kappa.")

    poly_inv = build_one_over_x_cheb(b, D_inv)
    use_rect = not args.no_rect
    fixed_scale = 1.0 / (2.0 * kappa)

    eps_rect = float("nan")
    delta_rect = float("nan")
    rect_degree = None

    if use_rect:
        # Eq. C6.
        eps_rect = min(2.0 * epsilon / (5.0 * kappa), kappa / (2.0 * D_inv))
        delta_rect = 1.0 / (4.0 * kappa)
        if args.rect_degree is None:
            k_rect = rect_sharpness(delta_rect, eps_rect)
            n_theta = theta_degree(k_rect, eps_rect)
            rect_degree = n_theta if n_theta % 2 == 0 else n_theta + 1
        else:
            rect_degree = int(args.rect_degree)

        # Keep the paper's fixed 1/(2kappa) scaling and raise degree if needed.
        check_grid = np.linspace(-1.0, 1.0, DEFAULT_NUM_POINTS)
        validated = False
        peak = float("inf")
        for _ in range(MAX_RECT_DEGREE_RETRIES + 1):
            poly_rect = build_rect_cheb(rect_degree, delta_rect, kappa, eps_rect)
            poly_mi = C.Chebyshev(C.chebmul(poly_inv.coef, poly_rect.coef))
            poly_mi = C.Chebyshev(fixed_scale * poly_mi.coef)
            peak = float(np.max(np.abs(poly_mi(check_grid))))
            if peak <= DEFAULT_MAX_SCALE - DEFAULT_PEAK_MARGIN:
                validated = True
                break
            if args.rect_degree is not None:
                raise ValueError(
                    f"Given --rect-degree={args.rect_degree} yields max |P^MI|={peak:.6f} > "
                    f"{DEFAULT_MAX_SCALE - DEFAULT_PEAK_MARGIN:.6f}. Increase --rect-degree."
                )
            rect_degree += 2

        if not validated:
            # Fallback: keep the shape but apply a safe extra scaling.
            poly_mi, extra, peak = bound_poly(poly_mi, DEFAULT_MAX_SCALE - DEFAULT_PEAK_MARGIN)
            print(
                "[warning] Could not validate fixed 1/(2kappa) scaling after degree retries; "
                f"applied extra scaling={extra:.8f} (peak before={peak:.6f})."
            )
    else:
        poly_mi = C.Chebyshev(fixed_scale * poly_inv.coef)

    cheb_coeffs = np.asarray(poly_mi.coef, dtype=float)
    degree = int(len(cheb_coeffs) - 1)

    phase_ret = angle_sequence.QuantumSignalProcessingPhases(
        cheb_coeffs,
        method="sym_qsp",
        chebyshev_basis=True,
    )
    if isinstance(phase_ret, tuple):
        phases = np.asarray(phase_ret[0], dtype=float)
        parity = int(phase_ret[2]) if len(phase_ret) > 2 else int(degree % 2)
    else:
        phases = np.asarray(phase_ret, dtype=float)
        parity = int(degree % 2)
    effective_degree = int(len(phases) - 1)

    x = np.linspace(-1.0, 1.0, DEFAULT_NUM_POINTS)
    qsp_response = angle_sequence.ComputeQSPResponse(
        x, phases, signal_operator="Wx", sym_qsp=True
    )
    p_of_x = np.imag(qsp_response["pdat"])
    p_target_poly = np.asarray(poly_mi(x), dtype=float)

    phase_vs_poly_max_abs_err = float(np.max(np.abs(p_of_x - p_target_poly)))
    phase_vs_poly_rms_err = float(np.sqrt(np.mean((p_of_x - p_target_poly) ** 2)))

    with np.errstate(divide="ignore", invalid="ignore"):
        scaled_inverse = np.divide(
            fixed_scale,
            x,
            out=np.zeros_like(x, dtype=float),
            where=np.abs(x) > 0.0,
        )
        g_exact = np.divide(
            fixed_scale * (1.0 - (1.0 - x**2) ** b),
            x,
            out=np.zeros_like(x, dtype=float),
            where=np.abs(x) > 0.0,
        )

    valid = np.abs(x) >= (1.0 / kappa)
    poly_vs_inverse_max_abs_error = float(np.max(np.abs(p_target_poly[valid] - scaled_inverse[valid])))
    poly_vs_inverse_rms_error = float(
        np.sqrt(np.mean((p_target_poly[valid] - scaled_inverse[valid]) ** 2))
    )
    domain_max_abs_error = float(np.max(np.abs(p_of_x[valid] - scaled_inverse[valid])))
    domain_rms_error = float(np.sqrt(np.mean((p_of_x[valid] - scaled_inverse[valid]) ** 2)))

    payload = {
        "target_function": "matrix_inversion_1_over_x",
        "epsilon": epsilon,
        "kappa": kappa,
        "eps_inverse": eps_inv,
        "kappa_inverse": kappa_inv,
        "b_target_exponent": int(b),
        "D_truncation_degree": int(D_inv),
        "inverse_poly_degree": int(2 * D_inv + 1),
        "use_rect_filter": bool(use_rect),
        "eps_rect": None if not use_rect else float(eps_rect),
        "rect_delta": None if not use_rect else float(delta_rect),
        "rect_degree": None if not use_rect else int(rect_degree),
        "degree": degree,
        "effective_degree": effective_degree,
        "max_scale": float(DEFAULT_MAX_SCALE),
        "returned_scale": float(fixed_scale),
        "parity": int(parity),
        "domain_of_validity": f"[1/{kappa:g}, 1] U [-1, -1/{kappa:g}]",
        "phase_vs_poly_max_abs_error": phase_vs_poly_max_abs_err,
        "phase_vs_poly_rms_error": phase_vs_poly_rms_err,
        "poly_vs_inverse_max_abs_error": poly_vs_inverse_max_abs_error,
        "poly_vs_inverse_rms_error": poly_vs_inverse_rms_error,
        "domain_max_abs_error": domain_max_abs_error,
        "domain_rms_error": domain_rms_error,
        "cheb_coeffs": cheb_coeffs.tolist(),
        "phases": phases.tolist(),
    }
    args.output_phases.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    plt.figure(figsize=(9, 5))
    plt.plot(x, p_target_poly, linewidth=2.0, label="P^{MI}(x) (target polynomial)")
    plt.plot(x, p_of_x, "--", linewidth=1.8, label="QSP P(x) = imag(<0|U|0>)")
    plt.plot(x, g_exact, color="k", linewidth=1.4, label="scaled g(x) = scale*(1-(1-x^2)^b)/x")
    plt.axvline(1.0 / kappa, color="gray", alpha=0.4, linestyle=":")
    plt.axvline(-1.0 / kappa, color="gray", alpha=0.4, linestyle=":")
    plt.ylim(-1.1, 1.1)
    plt.xlabel("x")
    plt.ylabel("value")
    title_extra = f", rect_degree={rect_degree}" if use_rect else ", no rect filter"
    plt.title(f"1/x QSP fit: epsilon={epsilon:g}, kappa={kappa:g}{title_extra}")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper center")
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=160)

    print("=== matrix-inversion (1/x) phase synthesis (pyqsp) ===")
    print(f"epsilon            : {epsilon:g}")
    print(f"kappa              : {kappa:g}")
    print(f"eps_inverse        : {eps_inv:g}   (= epsilon/4)")
    print(f"kappa_inverse      : {kappa_inv:g}   (= 2*kappa)")
    print(f"b (target exponent): {b}   -> exact g degree = {2*b - 1}")
    print(f"D (truncation deg) : {D_inv}   -> 1/x poly degree = {2*D_inv + 1}")
    print(f"use_rect_filter    : {use_rect}")
    if use_rect:
        print(f"eps_rect           : {eps_rect:g}")
        print(f"rect_delta (Delta) : {delta_rect:g}   (= 1/(4*kappa))")
        print(f"rect_degree        : {rect_degree}")
    print(f"polynomial_degree  : {degree}")
    print(f"effective_degree   : {effective_degree}")
    print(f"returned_scale     : {fixed_scale:.8f}")
    print(f"parity             : {parity}")
    print(f"domain_of_validity : [1/{kappa:g}, 1] U [-1, -1/{kappa:g}]")
    print(f"num_phases         : {len(phases)}")
    print(f"phase_vs_poly_max_abs_error   : {phase_vs_poly_max_abs_err:.6e}")
    print(f"phase_vs_poly_rms_error       : {phase_vs_poly_rms_err:.6e}")
    print(f"poly_vs_inverse_max_abs_error : {poly_vs_inverse_max_abs_error:.6e}")
    print(f"poly_vs_inverse_rms_error     : {poly_vs_inverse_rms_error:.6e}")
    print(f"domain_max_abs_error          : {domain_max_abs_error:.6e}")
    print(f"domain_rms_error              : {domain_rms_error:.6e}")
    print(f"phases_json        : {args.output_phases}")
    print(f"plot_png           : {args.output_plot}")
    print("\nchebyshev_coeffs:")
    print(np.array2string(cheb_coeffs, precision=10, separator=", "))
    print("\nphases (radians):")
    print(np.array2string(phases, precision=10, separator=", "))


if __name__ == "__main__":
    main()

