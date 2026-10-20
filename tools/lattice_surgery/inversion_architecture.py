"""Component-sum architectural projections with one resident layout per method."""

from __future__ import annotations

from collections import Counter

from estimate_sliced_resources import logical_error_per_round


def minimum_distance_proxy(tile_slices, physical_error_rate, target):
    for distance in range(3, 102, 2):
        if (
            tile_slices
            * distance
            * logical_error_per_round(physical_error_rate, distance)
            <= target
        ):
            return distance
    return None


def space_compact(text: str, factor: int = 2) -> str:
    """Insert routing columns between nonfactory columns, preserving factory shape."""
    rows = text.splitlines()
    expanded = ["" for _ in rows]
    for column in range(len(rows[0])):
        for r, row in enumerate(rows):
            expanded[r] += row[column]
        if column + 1 < len(rows[0]) and not any(
            row[column].isdigit() or row[column + 1].isdigit() for row in rows
        ):
            for r in range(len(rows)):
                expanded[r] += "r" * (factor - 1)
    return "\n".join(expanded) + "\n"


def aggregate_components(manifest: dict, records: list[dict], area: int, args) -> dict:
    """Multiply extensive costs; keep the shared physical footprint allocated once."""
    for kind, expected in (
        ("walk", manifest["num_w_applications"]),
        ("projector", manifest["num_projector_rotations"]),
    ):
        if (
            sum(c["occurrences"] for c in manifest["components"] if c["kind"] == kind)
            != expected
        ):
            raise ValueError("Component multiplicities disagree with the full sequence")
    if len(records) != len(manifest["components"]) or any(
        r["status"] != "success" for r in records
    ):
        raise ValueError(
            "All component compilations must succeed; partial schedules are not estimates"
        )
    slices = 0
    counts = Counter()
    contributions = {
        "walk": {"slices": 0, "magic_state_requests": 0},
        "projector": {"slices": 0, "magic_state_requests": 0},
    }
    rows = []
    for component, record in zip(manifest["components"], records):
        if record["resources"]["area"] != area:
            raise ValueError("Components must share one layout area")
        if record["slices"] < 1:
            raise ValueError("Each component must have a completed schedule")
        weight = component["occurrences"]
        weighted = weight * record["slices"]
        slices += weighted
        requests = record["instruction_counts"].get("RequestMagicState", 0)
        for instruction, count in record["instruction_counts"].items():
            counts[instruction] += weight * count
        contributions[component["kind"]]["slices"] += weighted
        contributions[component["kind"]]["magic_state_requests"] += weight * requests
        rows.append(
            {
                "id": component["id"],
                "kind": component["kind"],
                "phi": component.get("phi"),
                "occurrences": weight,
                "one_component_slices": record["slices"],
                "weighted_slices": weighted,
                "one_component_magic_states": requests,
                "logical_t_count": component["t_count"],
                "evidence": record["artifact_directory"],
                "input_sha256": record["input_sha256"],
            }
        )
    d, cycle, factor = (
        args.code_distance,
        args.cycle_time_ns,
        args.physical_qubits_per_tile_factor,
    )
    rounds = slices * d
    core_ns = rounds * cycle
    core_qubits = area * factor * d**2
    opportunities = area * rounds
    failure = min(
        1.0, opportunities * logical_error_per_round(args.physical_error_rate, d)
    )
    final_rounds = rounds
    physical_qubits = core_qubits
    factory = None
    if args.factory_cycle_rounds is not None:
        if args.num_factories is None:
            raise ValueError(
                "Supply tiles are not physical factories: specify --num-factories with --factory-cycle-rounds"
            )
        factory_rounds = (
            counts["RequestMagicState"] * args.factory_cycle_rounds / args.num_factories
        )
        final_rounds = max(rounds, factory_rounds)
        added_qubits = args.num_factories * (args.factory_physical_qubits or 0)
        physical_qubits += added_qubits
        factory = {
            "num_factories": args.num_factories,
            "rounds_per_state": args.factory_cycle_rounds,
            "throughput_lower_bound_rounds": factory_rounds,
            "external_factory_qubits": added_qubits,
            "assumption": "External factory footprint added once; in-grid source/storage tiles remain in layout area. Throughput bound does not model boundary stalls.",
        }
    runtime_ns = final_rounds * cycle
    warnings = [
        "Component-sum projection, not a scheduled full inverse: component boundary orientation, supply carryover and reset/transition costs are not simulated.",
        "One fixed geometry and logical identity map is reused; all logical qubits, including the phase flag, remain resident for every component.",
        "Magic-state sources are abstract. The compiler omits execution-correction details and does not distinguish T/T-dagger in its printed lowering.",
    ]
    if factory is None:
        warnings.append(
            "No physical factory cadence supplied; runtime assumes the compiler's disttime supply model."
        )
    if failure > args.target_failure_probability:
        warnings.append(
            "Full-sequence heuristic failure proxy exceeds the target; this fixed-distance report does not certify the requested reliability."
        )
    if args.code_distance % 2 == 0:
        warnings.append("Even code distance supplied.")
    kappa = manifest.get("algorithm", {}).get("kappa")
    if kappa and kappa < manifest.get("minimum_kappa_for_full_nonzero_spectrum", 0):
        warnings.append(
            "The supplied polynomial kappa does not cover the full nonzero target spectrum; these costs apply to its declared inversion domain."
        )
    logical_t = manifest["logical_gate_counts"].get("t", 0) + manifest[
        "logical_gate_counts"
    ].get("tdg", 0)
    return {
        "scope": "full_inverse_component_sum_surface_code_projection",
        "status": "success",
        "dimensions": manifest["dimensions"],
        "target_n": manifest["target_n"],
        "logical_qubits": manifest["logical_qubits"],
        "num_w_applications": manifest["num_w_applications"],
        "num_projector_rotations": manifest["num_projector_rotations"],
        "unique_projectors": len(rows) - 1,
        "logical_gate_resources": {
            "t_count": logical_t,
            "clifford_gate_count": sum(manifest["logical_gate_counts"].values())
            - logical_t,
        },
        "algorithm": manifest.get("algorithm", {}),
        "schedule_projection": {
            "slices": slices,
            "layout_tiles": area,
            "tile_slices": area * slices,
            "contributions": contributions,
        },
        "physical_projection": {
            "qec_rounds": rounds,
            "core_time_ns": core_ns,
            "core_time_seconds": core_ns * 1e-9,
            "resident_layout_physical_qubits": core_qubits,
            "physical_qubits": physical_qubits,
            "runtime_ns": runtime_ns,
            "runtime_seconds": runtime_ns * 1e-9,
            "space_time_ns_qubits": physical_qubits * runtime_ns,
            "failure_opportunities_proxy": opportunities,
            "failure_probability_proxy": failure,
            "minimum_odd_code_distance_proxy": minimum_distance_proxy(
                area * slices, args.physical_error_rate, args.target_failure_probability
            ),
            "factory_model": factory,
        },
        "assumptions": {
            "code_distance": d,
            "cycle_time_ns": cycle,
            "physical_qubits_per_tile_factor": factor,
            "physical_error_rate": args.physical_error_rate,
            "target_failure_probability": args.target_failure_probability,
            "disttime_slices": args.disttime,
            "staggered": args.staggered,
            "component_boundary_costs_included": False,
            "layout_transitions_required": False,
        },
        "weighted_instruction_counts": dict(sorted(counts.items())),
        "components": rows,
        "warnings": warnings,
    }


def render_report(result: dict) -> str:
    method = result["method"]
    if result["status"] != "success":
        return f"=== Full inverse architecture: {method} ===\nStatus: FAILED\n{result['reason']}\nNo numerical full-sequence estimate is available.\n"
    schedule, physical, assumptions = (
        result["schedule_projection"],
        result["physical_projection"],
        result["assumptions"],
    )
    lines = [
        f"=== Full Laplacian inverse architecture: {method} ===",
        "",
        "Accounting: one W and each unique projector compiled once, multiplied by occurrences.",
        "This is a component-sum projection, not compilation of the complete long circuit.",
        f"Dimensions / n per axis: {result['dimensions']} / {result['target_n']}",
        f"Resident logical qubits: {result['logical_qubits']}",
        f"W applications: {result['num_w_applications']}",
        f"Projector rotations / unique angles: {result['num_projector_rotations']} / {result['unique_projectors']}",
        f"Full-sequence logical Clifford / T count: {result['logical_gate_resources']['clifford_gate_count']:,} / {result['logical_gate_resources']['t_count']:,}",
        f"Layout dimensions: {result['layout_resources']['height']} x {result['layout_resources']['width']}",
        f"Layout tiles (allocated once): {schedule['layout_tiles']:,}",
        f"Layout preparation: {result['layout_preparation']}",
        "",
        "--- projected entire sequence ---",
        f"Walk contribution, slices: {schedule['contributions']['walk']['slices']:,}",
        f"Projector contribution, slices: {schedule['contributions']['projector']['slices']:,}",
        f"Total component-sum slices: {schedule['slices']:,}",
        f"Tile-slices: {schedule['tile_slices']:,}",
        f"QEC rounds: {physical['qec_rounds']:,}",
        f"Resident layout physical qubits: {physical['resident_layout_physical_qubits']:,.0f}",
        f"Total physical qubits (allocated once): {physical['physical_qubits']:,.0f}",
        f"Projected core time: {physical['core_time_seconds']:.9g} s",
        f"Projected time with optional factory bound: {physical['runtime_seconds']:.9g} s",
        f"Space-time cost: {physical['space_time_ns_qubits']:.9e} ns*qubits",
        f"Magic-state requests: {result['weighted_instruction_counts'].get('RequestMagicState', 0):,}",
        f"Heuristic failure probability proxy: {physical['failure_probability_proxy']:.6e}",
        f"Minimum odd distance under this proxy: {physical['minimum_odd_code_distance_proxy']}",
        "",
        "--- assumptions ---",
        f"Code distance: {assumptions['code_distance']}",
        f"QEC cycle time: {assumptions['cycle_time_ns']} ns",
        f"Physical qubits/tile: {assumptions['physical_qubits_per_tile_factor']} * d^2",
        f"Physical error rate: {assumptions['physical_error_rate']:.6e}",
        f"Supply interval / staggering: {assumptions['disttime_slices']} slices / {assumptions['staggered']}",
        f"Pipeline: wave, local decomposition; compiler pin: {result['compiler']['native_commit']}",
        "",
        "--- component measurements ---",
        f"{'Component':<12}{'Occurrences':>14}{'Slices each':>16}{'Weighted slices':>19}{'T requests each':>18}",
    ]
    for row in result["components"]:
        lines.append(
            f"{row['id']:<12}{row['occurrences']:>14,}{row['one_component_slices']:>16,}{row['weighted_slices']:>19,}{row['one_component_magic_states']:>18,}"
        )
    if "optimization" in result:
        optimization = result["optimization"]
        lines += [
            "",
            "--- simulated annealing ---",
            f"Budget / seed: {optimization['budget']} / {optimization['seed']}",
            "Search objective: area * slices of one W followed by the largest-T projector; final totals use every angle's measured cost.",
            f"Representative projector: {optimization['representative_projector']}",
            f"Search time: {optimization['wall_seconds']:.3f} s",
            f"Replay verified: {optimization['replay_verified']}",
        ]
    lines += [
        "",
        "--- model limits and warnings ---",
        *[f"- {warning}" for warning in result["warnings"]],
        "",
        f"Complete evidence: {result['evidence_directory']}",
    ]
    return "\n".join(lines) + "\n"
