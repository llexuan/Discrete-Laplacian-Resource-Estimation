#!/usr/bin/env python3
"""Estimate physical resources from a liblsqecc scheduled computation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QASM = Path(__file__).resolve().with_name("input_circuit.qasm")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compiler"
DEFAULT_SLICER = (
    PROJECT_ROOT.parent / "liblsqecc" / "build" / "lsqecc_slicer"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate schedule-aware surface-code resources from liblsqecc "
            "slices. Temporary patch JSON is deleted after metrics extraction."
        )
    )
    parser.add_argument("--qasm", type=Path, default=DEFAULT_QASM)
    parser.add_argument(
        "--slicer",
        type=Path,
        default=Path(os.environ.get("LSQECC_SLICER", DEFAULT_SLICER)),
    )
    parser.add_argument(
        "--preset",
        choices=("direct", "viewer", "edpc", "compact"),
        default="direct",
        help=(
            "Historical direct/viewer configurations, or matched edpc/compact "
            "configurations for layout comparisons."
        ),
    )
    parser.add_argument("--code-distance", type=int, default=7)
    parser.add_argument("--cycle-time-ns", type=float, default=1000.0)
    parser.add_argument(
        "--physical-qubits-per-tile-factor",
        type=float,
        default=2.0,
        help="Physical qubits per logical tile divided by d^2.",
    )
    parser.add_argument("--physical-error-rate", type=float, default=1e-3)
    parser.add_argument(
        "--target-failure-probability",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--factory-cycle-rounds",
        type=float,
        default=None,
        help="Optional rounds required per magic-state output per factory.",
    )
    parser.add_argument(
        "--factory-physical-qubits",
        type=float,
        default=None,
        help="Optional physical-qubit footprint of one factory.",
    )
    parser.add_argument(
        "--num-factories",
        type=int,
        default=None,
        help="Factory count (default: number of reserved distillation tiles).",
    )
    parser.add_argument(
        "--disttime",
        type=int,
        default=1,
        help="liblsqecc magic-state replenishment interval in slices.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUT_DIR / "slice_resource_estimate.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=OUTPUT_DIR / "slice_resource_report.txt",
    )
    return parser


def backend_options(preset: str, disttime: int) -> list[str]:
    if preset == "direct":
        return [
            "-L",
            "edpc",
            "--disttime",
            str(disttime),
            "--nostagger",
            "--notwists",
            "--local",
            "-P",
            "wave",
        ]
    if preset == "viewer":
        return ["-L", "compact", "-P", "stream"]
    return [
        "-L",
        preset,
        "-P",
        "wave",
        "--local",
        "--disttime",
        str(disttime),
        "--nostagger",
        "--cnotcorrections",
        "never",
        "-r",
        "graph_search",
        "-g",
        "djikstra",
    ]


def run_checked(command: list[str]) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr or result.stdout or "unknown error"
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}:\n"
            f"{details.strip()}"
        )
    return result.stdout


def instruction_counts(lli_text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in lli_text.splitlines():
        for instruction in line.split(";"):
            instruction = instruction.strip()
            if instruction:
                counts[instruction.split()[0]] += 1
    return counts


def layout_metrics(slices: list[Any]) -> dict[str, Any]:
    if not slices:
        raise ValueError("liblsqecc generated no patch slices.")

    rows = max(len(layout) for layout in slices)
    columns = max(
        (len(row) for layout in slices for row in layout),
        default=0,
    )
    if rows <= 0 or columns <= 0:
        raise ValueError("Generated patch slices have an empty layout.")

    occupied_per_slice: list[int] = []
    patch_type_max: Counter[str] = Counter()
    for layout in slices:
        occupied = 0
        patch_types: Counter[str] = Counter()
        for row in layout:
            for cell in row:
                if isinstance(cell, dict):
                    occupied += 1
                    patch_types[str(cell.get("patch_type", "Unknown"))] += 1
        occupied_per_slice.append(occupied)
        for patch_type, count in patch_types.items():
            patch_type_max[patch_type] = max(
                patch_type_max[patch_type],
                count,
            )

    return {
        "num_slices": len(slices),
        "layout_rows": rows,
        "layout_columns": columns,
        "layout_tiles": rows * columns,
        "max_occupied_tiles": max(occupied_per_slice),
        "mean_occupied_tiles": sum(occupied_per_slice) / len(occupied_per_slice),
        "max_patch_types": dict(sorted(patch_type_max.items())),
        "reserved_distillation_tiles": patch_type_max.get(
            "DistillationQubit",
            0,
        ),
    }


def logical_qubits_from_qasm(qasm_text: str) -> int | None:
    for statement in qasm_text.replace("\n", " ").split(";"):
        statement = statement.strip()
        if statement.startswith("qreg ") and "[" in statement and "]" in statement:
            try:
                return int(statement.rsplit("[", 1)[1].split("]", 1)[0])
            except ValueError:
                return None
    return None


def logical_error_per_round(physical_error_rate: float, distance: int) -> float:
    """Transparent surface-code scaling proxy used only as a warning metric."""
    return 0.1 * (100.0 * physical_error_rate) ** ((distance + 1) / 2.0)


def estimate(args: argparse.Namespace) -> dict[str, Any]:
    if args.code_distance <= 0:
        raise ValueError("--code-distance must be positive.")
    if args.cycle_time_ns <= 0:
        raise ValueError("--cycle-time-ns must be positive.")
    if args.physical_qubits_per_tile_factor <= 0:
        raise ValueError("--physical-qubits-per-tile-factor must be positive.")
    if not 0 < args.physical_error_rate < 1:
        raise ValueError("--physical-error-rate must be between 0 and 1.")
    if not 0 < args.target_failure_probability < 1:
        raise ValueError("--target-failure-probability must be between 0 and 1.")
    if args.disttime <= 0:
        raise ValueError("--disttime must be positive.")
    if args.num_factories is not None and args.num_factories <= 0:
        raise ValueError("--num-factories must be positive.")

    qasm_path = args.qasm.resolve()
    slicer = args.slicer.resolve()
    if not slicer.is_file():
        raise FileNotFoundError(f"lsqecc_slicer not found at {slicer}.")
    qasm_text = qasm_path.read_text(encoding="utf-8")
    if not qasm_text.lstrip().startswith("OPENQASM 2.0;"):
        raise ValueError(f"{qasm_path} is not OpenQASM 2.")

    options = backend_options(args.preset, args.disttime)
    prefix = [
        str(slicer),
        "-I",
        "qasm",
        "-i",
        str(qasm_path),
    ]
    with tempfile.TemporaryDirectory(prefix="lsqecc-resource-") as tmp:
        slices_path = Path(tmp) / "slices.json"
        stats_stdout = run_checked(
            [
                *prefix,
                "-o",
                str(slices_path),
                "-f",
                "stats",
                "--graceful",
                *options,
            ]
        )
        unsliced_lli = run_checked(
            [
                *prefix,
                "--printlli",
                "before",
                "--graceful",
                *options,
            ]
        )
        sliced_lli = run_checked(
            [
                *prefix,
                "--printlli",
                "sliced",
                "--graceful",
                *options,
            ]
        )
        slices = json.loads(slices_path.read_text(encoding="utf-8"))

    if not isinstance(slices, list):
        raise ValueError("Expected liblsqecc slice output to be a JSON array.")

    schedule = layout_metrics(slices)
    unsliced_counts = instruction_counts(unsliced_lli)
    sliced_counts = instruction_counts(sliced_lli)
    sliced_nonempty_lines = sum(
        bool(line.strip()) for line in sliced_lli.splitlines()
    )
    warnings: list[str] = []
    if args.code_distance % 2 == 0:
        warnings.append(
            "Even code distance supplied; rotated surface-code studies "
            "typically use odd distances."
        )

    d = args.code_distance
    core_rounds = schedule["num_slices"] * d
    core_time_ns = core_rounds * args.cycle_time_ns
    layout_physical_qubits = (
        schedule["layout_tiles"]
        * args.physical_qubits_per_tile_factor
        * d**2
    )
    core_volume_ns_qubits = layout_physical_qubits * core_time_ns

    p_logical_round = logical_error_per_round(
        args.physical_error_rate,
        d,
    )
    failure_opportunities = schedule["layout_tiles"] * core_rounds
    estimated_failure_probability = min(
        1.0,
        failure_opportunities * p_logical_round,
    )
    if estimated_failure_probability > args.target_failure_probability:
        warnings.append(
            "Heuristic logical-failure estimate exceeds the requested target; "
            "increase code distance or use a calibrated error model."
        )

    magic_states = unsliced_counts.get("RequestMagicState", 0)
    y_states = unsliced_counts.get("RequestYState", 0)
    factory_model: dict[str, Any] | None = None
    final_rounds = float(core_rounds)
    total_physical_qubits = float(layout_physical_qubits)
    if args.factory_cycle_rounds is not None:
        if args.factory_cycle_rounds <= 0:
            raise ValueError("--factory-cycle-rounds must be positive.")
        default_factories = max(
            1,
            int(schedule["reserved_distillation_tiles"]),
        )
        num_factories = args.num_factories or default_factories
        factory_limited_rounds = (
            magic_states * args.factory_cycle_rounds / num_factories
        )
        final_rounds = max(final_rounds, factory_limited_rounds)
        factory_qubits = (
            0.0
            if args.factory_physical_qubits is None
            else num_factories * args.factory_physical_qubits
        )
        total_physical_qubits += factory_qubits
        factory_model = {
            "num_factories": num_factories,
            "factory_cycle_rounds": args.factory_cycle_rounds,
            "factory_physical_qubits_each": args.factory_physical_qubits,
            "factory_physical_qubits_total": factory_qubits,
            "magic_states_required": magic_states,
            "factory_limited_rounds": factory_limited_rounds,
            "is_runtime_bottleneck": factory_limited_rounds > core_rounds,
        }
        if args.factory_physical_qubits is None:
            warnings.append(
                "Factory throughput is modeled, but no factory physical-qubit "
                "footprint was supplied."
            )
    else:
        warnings.append(
            "No physical magic-state factory cadence was supplied; runtime "
            "uses the scheduler's --disttime assumption only."
        )

    final_time_ns = final_rounds * args.cycle_time_ns
    total_volume_ns_qubits = total_physical_qubits * final_time_ns

    legacy_core_space_qubits = schedule["layout_tiles"] * d**2
    legacy_core_time_ns = schedule["num_slices"] * args.cycle_time_ns * d
    legacy = {
        "description": (
            "Repository-compatible core formulas with fixed distance; known "
            "unit/model limitations are intentionally not hidden."
        ),
        "core_space_qubits": legacy_core_space_qubits,
        "core_timesteps": schedule["num_slices"],
        "core_time_ns": legacy_core_time_ns,
        "core_space_time_volume_ns_qubits": (
            legacy_core_space_qubits * legacy_core_time_ns
        ),
        "warning": (
            "The upstream time_ms field is not reproduced because it sums "
            "nanosecond-labeled quantities without converting units."
        ),
    }

    return {
        "scope": "liblsqecc_schedule_aware_physical_resource_estimate",
        "source_qasm": str(qasm_path),
        "compiler": {
            "backend": "liblsqecc/lsqecc_slicer",
            "preset": args.preset,
            "options": options,
            "disttime_slices": args.disttime,
            "stats_stdout": stats_stdout.strip(),
            "temporary_patch_json_retained": False,
        },
        "logical_circuit": {
            "logical_qubits": logical_qubits_from_qasm(qasm_text),
            "magic_state_requests": magic_states,
            "y_state_requests": y_states,
            "unsliced_instruction_counts": dict(sorted(unsliced_counts.items())),
            "sliced_instruction_counts": dict(sorted(sliced_counts.items())),
        },
        "schedule": {
            **schedule,
            "sliced_lli_nonempty_lines": sliced_nonempty_lines,
        },
        "assumptions": {
            "code_distance": d,
            "cycle_time_ns": args.cycle_time_ns,
            "physical_qubits_per_tile_factor": (
                args.physical_qubits_per_tile_factor
            ),
            "physical_error_rate": args.physical_error_rate,
            "target_failure_probability": args.target_failure_probability,
            "logical_error_proxy": (
                "0.1 * (100 * p_phys)^((d + 1) / 2)"
            ),
        },
        "schedule_aware_estimate": {
            "qec_rounds_core": core_rounds,
            "core_time_ns": core_time_ns,
            "core_time_seconds": core_time_ns * 1e-9,
            "layout_physical_qubits": layout_physical_qubits,
            "core_space_time_volume_ns_qubits": core_volume_ns_qubits,
            "logical_error_per_round_proxy": p_logical_round,
            "failure_opportunities_proxy": failure_opportunities,
            "total_failure_probability_proxy": estimated_failure_probability,
            "factory_model": factory_model,
            "qec_rounds_with_factory_bottleneck": final_rounds,
            "runtime_with_factory_bottleneck_ns": final_time_ns,
            "runtime_with_factory_bottleneck_seconds": final_time_ns * 1e-9,
            "total_physical_qubits_with_factory": total_physical_qubits,
            "total_space_time_volume_ns_qubits": total_volume_ns_qubits,
        },
        "legacy_repository_comparison": legacy,
        "warnings": warnings,
    }


def render_report(result: dict[str, Any]) -> str:
    schedule = result["schedule"]
    logical = result["logical_circuit"]
    assumptions = result["assumptions"]
    estimate_data = result["schedule_aware_estimate"]
    legacy = result["legacy_repository_comparison"]
    lines = [
        "=== liblsqecc slice-based resource estimate ===",
        "",
        f"source QASM             : {result['source_qasm']}",
        f"preset                 : {result['compiler']['preset']}",
        f"compiler options        : {' '.join(result['compiler']['options'])}",
        f"logical qubits         : {logical['logical_qubits']}",
        f"scheduled slices       : {schedule['num_slices']}",
        (
            "layout rows x columns  : "
            f"{schedule['layout_rows']} x {schedule['layout_columns']}"
        ),
        f"layout tiles           : {schedule['layout_tiles']}",
        f"max occupied tiles     : {schedule['max_occupied_tiles']}",
        f"magic-state requests   : {logical['magic_state_requests']}",
        f"Y-state requests       : {logical['y_state_requests']}",
        "",
        "--- physical assumptions ---",
        f"code distance          : {assumptions['code_distance']}",
        f"cycle time             : {assumptions['cycle_time_ns']} ns",
        (
            "physical qubits/tile   : "
            f"{assumptions['physical_qubits_per_tile_factor']} * d^2"
        ),
        f"physical error rate    : {assumptions['physical_error_rate']:.3e}",
        (
            "target failure prob.   : "
            f"{assumptions['target_failure_probability']:.3e}"
        ),
        "",
        "--- schedule-aware estimate ---",
        f"core QEC rounds        : {estimate_data['qec_rounds_core']}",
        f"core runtime           : {estimate_data['core_time_ns']:.3f} ns",
        f"core runtime           : {estimate_data['core_time_seconds']:.6e} s",
        (
            "layout physical qubits : "
            f"{estimate_data['layout_physical_qubits']:.3f}"
        ),
        (
            "core space-time volume : "
            f"{estimate_data['core_space_time_volume_ns_qubits']:.6e} "
            "ns*qubits"
        ),
        (
            "failure proxy          : "
            f"{estimate_data['total_failure_probability_proxy']:.6e}"
        ),
        "",
        "--- legacy repository-compatible core values ---",
        f"core timesteps         : {legacy['core_timesteps']}",
        f"core time              : {legacy['core_time_ns']:.3f} ns",
        f"core space qubits      : {legacy['core_space_qubits']}",
        (
            "core space-time volume : "
            f"{legacy['core_space_time_volume_ns_qubits']:.6e} ns*qubits"
        ),
    ]
    factory = estimate_data["factory_model"]
    if factory is not None:
        lines.extend(
            [
                "",
                "--- factory throughput model ---",
                f"factories              : {factory['num_factories']}",
                (
                    "factory cycle           : "
                    f"{factory['factory_cycle_rounds']} rounds/state"
                ),
                (
                    "factory-limited rounds  : "
                    f"{factory['factory_limited_rounds']:.3f}"
                ),
                (
                    "factory bottleneck      : "
                    f"{factory['is_runtime_bottleneck']}"
                ),
                (
                    "final runtime           : "
                    f"{estimate_data['runtime_with_factory_bottleneck_seconds']:.6e} s"
                ),
            ]
        )
    if result["warnings"]:
        lines.extend(["", "--- warnings ---"])
        lines.extend(f"- {warning}" for warning in result["warnings"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    result = estimate(args)
    output_json = args.output_json.resolve()
    output_report = args.output_report.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    output_report.write_text(render_report(result), encoding="utf-8")
    print(render_report(result), end="")
    print(f"\nJSON report : {output_json}")
    print(f"text report : {output_report}")


if __name__ == "__main__":
    main()
