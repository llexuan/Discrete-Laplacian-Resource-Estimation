#!/usr/bin/env python3
"""
Combine polynomial, phase, and block-encoding errors using:
1) a conservative sum bound metric, and
2) tighter proxy / direct metrics when available.

python error_analysis.py --epsilon-target 1e-12
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute combined max error from phase and block metrics."
    )
    parser.add_argument(
        "--phase-metrics",
        type=Path,
        default=Path("phases_target.json"),
        help="Path to phase/polynomial metrics JSON (from one_qubit_phase_rotations.py).",
    )
    parser.add_argument(
        "--block-metrics",
        type=Path,
        default=Path("block_metrics.json"),
        help="Path to block-encoding metrics JSON (optional).",
    )
    parser.add_argument(
        "--epsilon-block-2",
        type=float,
        default=None,
        help="Override block spectral error directly. If omitted, read from --block-metrics; if unavailable, defaults to 0.",
    )
    parser.add_argument(
        "--epsilon-target",
        type=float,
        required=True,
        help="Target threshold epsilon_target for accept/reject check.",
    )
    parser.add_argument(
        "--epsilon-exec",
        type=float,
        default=0.0,
        help="Execution error term (use 0.0 for ideal simulation).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("error_analysis.json"),
        help="Where to write combined error analysis JSON.",
    )
    args = parser.parse_args()

    phase_data = read_json(args.phase_metrics)
    eps_poly = float(phase_data["poly_max_abs_error"])
    eps_phase = float(phase_data["phase_max_abs_error"])
    block_data = None
    block_source = None
    if args.epsilon_block_2 is not None:
        eps_block_2 = float(args.epsilon_block_2)
        block_source = "cli_override"
    elif args.block_metrics.exists():
        block_data = read_json(args.block_metrics)
        eps_block_2 = float(block_data["block_spectral_error"])
        block_source = str(args.block_metrics)
    else:
        eps_block_2 = 0.0
        block_source = "default_zero_no_block_metrics"
    eps_exec = float(args.epsilon_exec)

    # Conservative, guaranteed-style upper-bound metric.
    eps_all_max = eps_poly + eps_phase + eps_block_2 + eps_exec
    # Tighter aggregate proxy if errors are weakly correlated.
    eps_all_rss = math.sqrt(
        eps_poly**2 + eps_phase**2 + eps_block_2**2 + eps_exec**2
    )

    # Optional direct end-to-end metrics from phase synthesis file.
    eps_e2e_max = phase_data.get("total_max_abs_error", phase_data.get("max_abs_error"))
    eps_e2e_rms = phase_data.get("total_rms_error", phase_data.get("rms_error"))
    eps_e2e_max = float(eps_e2e_max) if eps_e2e_max is not None else None
    eps_e2e_rms = float(eps_e2e_rms) if eps_e2e_rms is not None else None

    passed_conservative = eps_all_max <= args.epsilon_target
    passed_rss = eps_all_rss <= args.epsilon_target
    passed_direct = (eps_e2e_max <= args.epsilon_target) if eps_e2e_max is not None else None

    result = {
        "epsilon_target": float(args.epsilon_target),
        "epsilon_poly_max": eps_poly,
        "epsilon_phase_max": eps_phase,
        "epsilon_block_2": eps_block_2,
        "epsilon_exec": eps_exec,
        "epsilon_all_max_sum_bound": eps_all_max,
        "epsilon_all_rss": eps_all_rss,
        "epsilon_e2e_max_direct": eps_e2e_max,
        "epsilon_e2e_rms_direct": eps_e2e_rms,
        "passes_threshold_conservative": passed_conservative,
        "passes_threshold_rss": passed_rss,
        "passes_threshold_direct": passed_direct,
        "target_function": phase_data.get("target_function"),
        "degree": phase_data.get("degree"),
        "effective_degree": phase_data.get("effective_degree"),
        "num_phases": len(phase_data.get("phases", [])),
        "phase_metrics_source": str(args.phase_metrics),
        "block_metrics_source": block_source,
    }

    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("=== Combined error analysis ===")
    print(f"epsilon_target    : {args.epsilon_target:.6e}")
    print(f"epsilon_poly_max  : {eps_poly:.6e}")
    print(f"epsilon_phase_max : {eps_phase:.6e}")
    print(f"epsilon_block_2   : {eps_block_2:.6e}")
    print(f"block_source      : {block_source}")
    print(f"epsilon_all_max(sum-bound): {eps_all_max:.6e}")
    print(f"epsilon_all_rss          : {eps_all_rss:.6e}")
    if eps_e2e_max is not None:
        print(f"epsilon_e2e_max_direct   : {eps_e2e_max:.6e}")
    else:
        print("epsilon_e2e_max_direct   : unavailable")
    if eps_e2e_rms is not None:
        print(f"epsilon_e2e_rms_direct   : {eps_e2e_rms:.6e}")
    else:
        print("epsilon_e2e_rms_direct   : unavailable")
    print(f"passes_conservative: {passed_conservative}")
    print(f"passes_rss         : {passed_rss}")
    print(f"passes_direct      : {passed_direct}")
    print(f"saved_json        : {args.output}")


if __name__ == "__main__":
    main()
