"""Component-sum architectural projections with one resident layout per method."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from estimate_sliced_resources import logical_error_per_round
from output_files import resource_paths


def minimum_distance_proxy(tile_slices, physical_error_rate, target, minimum=3):
    """Find the smallest admissible odd distance for one layout's failure budget."""
    first = max(3, minimum)
    first += first % 2 == 0
    for distance in range(first, 102, 2):
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
    unconstrained_minimum = minimum_distance_proxy(
        area * slices, args.physical_error_rate, args.target_failure_probability
    )
    d = args.code_distance
    auto = args.auto_code_distance
    if auto:
        d = minimum_distance_proxy(
            area * slices, args.physical_error_rate,
            args.target_failure_probability, minimum=args.code_distance,
        )
        if d is None:
            raise ValueError(
                "No admissible odd code distance <=101 satisfies this method's failure budget"
            )
    cycle, factor = args.cycle_time_ns, args.physical_qubits_per_tile_factor
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
    if d % 2 == 0:
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
            "minimum_odd_code_distance_proxy": unconstrained_minimum,
            "meets_target_failure_probability": failure <= args.target_failure_probability,
            "factory_model": factory,
        },
        "assumptions": {
            "code_distance": d,
            "distance_selection_mode": "per_method_minimum" if auto else "fixed",
            "requested_minimum_code_distance": args.code_distance if auto else None,
            "distance_selection": (
                "Smallest odd distance for this method satisfying the shared layout-failure budget and requested minimum; physical factory errors are excluded."
                if auto else "User-specified fixed code distance."
            ),
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
        return f"=== Full inverse architecture: {method} ===\nRun: {result.get('run_id', 'unrecorded')}\nStatus: {result['status'].upper()}\n{result['reason']}\nNo numerical full-sequence estimate is available.\n"
    schedule, physical, assumptions = (
        result["schedule_projection"],
        result["physical_projection"],
        result["assumptions"],
    )
    lines = [
        f"=== Full Laplacian inverse architecture: {method} ===",
        f"Run: {result.get('run_id', 'unrecorded')}",
        "Status: SUCCESS",
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
        f"Shared failure budget: <= {assumptions['target_failure_probability']:.6e}",
        f"Meets failure budget: {'Yes' if physical['meets_target_failure_probability'] else 'No'}",
        f"Minimum odd distance under this proxy: {physical['minimum_odd_code_distance_proxy']}",
        "",
        "--- assumptions ---",
        f"Code distance: {assumptions['code_distance']}",
        f"Distance selection: {assumptions['distance_selection']}",
        f"QEC cycle time: {assumptions['cycle_time_ns']} ns",
        f"Physical qubits/tile: {assumptions['physical_qubits_per_tile_factor']} * d^2",
        f"Physical error rate: {assumptions['physical_error_rate']:.6e}",
        f"Supply interval / staggering: {assumptions['disttime_slices']} slices / {assumptions['staggered']}",
        f"Pipeline: wave, local decomposition; compiler pin: {result['compiler']['native_commit']}",
    ]
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
        "",
        "--- appendix: component measurements ---",
        f"{'Component':<12}{'Occurrences':>14}{'Slices each':>16}{'Weighted slices':>19}{'T requests each':>18}",
    ]
    for row in result["components"]:
        lines.append(
            f"{row['id']:<12}{row['occurrences']:>14,}{row['one_component_slices']:>16,}{row['weighted_slices']:>19,}{row['one_component_magic_states']:>18,}"
        )
    return "\n".join(lines) + "\n"


def render_comparison(comparison: dict) -> str:
    """Present the case, shared assumptions, tradeoffs and qualifications together."""
    methods = comparison["methods"]
    names = [name for name in ("annealed", "compact", "edpc") if name in methods]
    successful = [methods[name] for name in names if methods[name]["status"] == "success"]
    lines = [
        "FULL LAPLACIAN INVERSE: ARCHITECTURE COMPARISON",
        f"Run: {comparison.get('run_id', Path(comparison['run_directory']).name)}",
        "Status: " + ("SUCCESS" if len(successful) == len(names) else "INCOMPLETE / FAILED METHODS"),
        "",
        "Component costs are compiled once and multiplied by their occurrences.",
        "Physical qubits are allocated once. This is a full-sequence projection.",
    ]
    if successful:
        case = successful[0]
        lines += [
            "",
            "EXAMPLE",
            f"Dimensions: {case['dimensions']}; n per axis: {case['target_n']} ({2 ** case['target_n']:,} grid points per axis)",
            f"Sequence degree / W applications: {case['num_w_applications']}",
            f"Projector rotations: {case['num_projector_rotations']}; distinct angles: {case['unique_projectors']}",
            f"Resident logical qubits: {case['logical_qubits']}",
            f"Logical Clifford gates: {case['logical_gate_resources']['clifford_gate_count']:,}",
            f"Logical T/T-dagger gates: {case['logical_gate_resources']['t_count']:,}",
            "",
            "SHARED ASSUMPTIONS",
        ]
        for key, label in (
            ("cycle_time_ns", "QEC cycle time (ns)"),
            ("physical_qubits_per_tile_factor", "Physical qubits per tile / d^2"),
            ("physical_error_rate", "Physical error rate"),
            ("target_failure_probability", "Target layout-failure proxy"),
            ("disttime_slices", "Magic-state supply interval (slices)"),
            ("staggered", "Staggered supply"),
        ):
            values = [item["assumptions"][key] for item in successful]
            value = (
                str(values[0])
                if all(v == values[0] for v in values)
                else "DIFFERS; see method reports"
            )
            lines.append(f"{label}: {value}")
        modes = {item["assumptions"].get("distance_selection_mode", "fixed") for item in successful}
        lines.append(
            "Code distance policy: "
            + (
                "each method's minimum admissible odd distance for the shared failure budget"
                if modes == {"per_method_minimum"}
                else "fixed or mixed settings; see individual reports"
            )
        )
        lines.append("Failure proxy: min(1, tiles * slices * d * 0.1 * (100*p_phys)^((d+1)/2)).")
        factory_models = [
            item["physical_projection"]["factory_model"] for item in successful
        ]
        lines.append(
            "Physical factory model: "
            + (
                "not supplied; abstract compiler sources"
                if all(m is None for m in factory_models)
                else "see each method report"
            )
        )

    lines += ["", "RESULTS"]
    table = [["Metric", *[
        "EDPC" if name == "edpc" else name.capitalize() for name in names
    ]]]

    def table_row(label, getter, formatter=lambda value: f"{value:,}"):
        values = []
        for name in names:
            result = methods[name]
            values.append(
                formatter(getter(result))
                if result["status"] == "success"
                else result["status"].upper()
            )
        table.append([label, *values])

    table_row("Layout tiles", lambda r: r["schedule_projection"]["layout_tiles"])
    table_row(
        "Slices for one W",
        lambda r: next(
            c["one_component_slices"] for c in r["components"] if c["kind"] == "walk"
        ),
    )
    table_row(
        "All walk slices",
        lambda r: r["schedule_projection"]["contributions"]["walk"]["slices"],
    )
    table_row(
        "All rotation slices",
        lambda r: r["schedule_projection"]["contributions"]["projector"]["slices"],
    )
    table_row("Total projected slices", lambda r: r["schedule_projection"]["slices"])
    table_row("Code distance", lambda r: r["assumptions"]["code_distance"])
    table_row(
        "Physical qubits",
        lambda r: r["physical_projection"]["physical_qubits"],
        lambda v: f"{v:,.0f}",
    )
    table_row(
        "Projected runtime (s)",
        lambda r: r["physical_projection"]["runtime_seconds"],
        lambda v: f"{v:,.6f}",
    )
    table_row(
        "Space-time (qubit*s)",
        lambda r: r["physical_projection"]["space_time_ns_qubits"] * 1e-9,
        lambda v: f"{v:,.3f}",
    )
    table_row(
        "Failure budget (<=)",
        lambda r: r["assumptions"]["target_failure_probability"],
        lambda v: f"{v:.6e}",
    )
    table_row(
        "Layout-failure proxy",
        lambda r: r["physical_projection"]["failure_probability_proxy"],
        lambda v: f"{v:.6e}",
    )
    table_row(
        "Meets failure budget",
        lambda r: r["physical_projection"]["meets_target_failure_probability"],
        lambda v: "Yes" if v else "No",
    )
    widths = [max(len(row[column]) for row in table) for column in range(len(table[0]))]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines.append(border)
    for index, row in enumerate(table):
        cells = [f" {row[0]:<{widths[0]}} "]
        cells.extend(
            f" {value:^{width}} " if index == 0 else f" {value:>{width}} "
            for value, width in zip(row[1:], widths[1:])
        )
        lines.append("|" + "|".join(cells) + "|")
        if index == 0 or row[0] == "Total projected slices":
            lines.append(border)
    lines.append(border)
    if successful:
        lines += ["", "INTERPRETATION (among successful methods in this run)"]
        for label, key in (
            ("Fewest physical qubits", "physical_qubits"),
            ("Shortest projected runtime", "runtime_seconds"),
            ("Smallest space-time cost", "space_time_ns_qubits"),
        ):
            value = min(r["physical_projection"][key] for r in successful)
            winners = [
                r["method"] for r in successful
                if r["physical_projection"][key] == value
            ]
            lines.append(f"{label}: {', '.join(winners)}.")
        baseline = methods.get("edpc", {})
        if baseline.get("status") == "success":
            lines.append("Relative to EDPC:")
            for name in names:
                r = methods[name]
                if name == "edpc" or r["status"] != "success":
                    continue
                changes = [
                    100 * (
                        r["physical_projection"][key]
                        / baseline["physical_projection"][key] - 1
                    )
                    for key in ("physical_qubits", "runtime_seconds")
                ]
                space = f"{abs(changes[0]):.2f}% {'fewer' if changes[0] < 0 else 'more'} physical qubits"
                runtime = f"{abs(changes[1]):.2f}% {'shorter' if changes[1] < 0 else 'longer'} runtime"
                lines.append(f"  {name}: {space}; {runtime}.")
    lines += ["", "METHOD DETAILS AND LIMITS"]
    warnings = []
    for name in names:
        r = methods[name]
        if r["status"] != "success":
            lines.append(f"{name}: {r['status'].upper()} - {r['reason']}")
            continue
        lines.append(f"{name}: {r['layout_preparation']}.")
        if "optimization" in r:
            opt = r["optimization"]
            lines.append(f"  Annealing budget {opt['budget']}, seed {opt['seed']}; optimizes one W + {opt['representative_projector']}. Full totals measure every distinct angle on the winning layout.")
        for warning in r["warnings"]:
            if warning not in warnings:
                warnings.append(warning)
    lines.extend(f"- {warning}" for warning in warnings)
    lines += [
        "",
        "FILES",
        "Logical baseline: logical/inversion_resource_report.txt",
        "Method reports: annealed_resource_report.txt, compact_resource_report.txt, EDPC_resource_report.txt",
        "Machine-readable results: data/",
        "Inputs and manifest: inputs/",
        f"Complete run evidence: {comparison['run_directory']}",
    ]
    return "\n".join(lines) + "\n"


def publish_results(output: Path, comparison: dict) -> str:
    """Publish one identified run; absent/failed methods cannot reuse old scores."""
    (output / "data").mkdir(parents=True, exist_ok=True)
    comparison["run_id"] = Path(comparison["run_directory"]).name
    for method in ("annealed", "compact", "edpc"):
        result = comparison["methods"].setdefault(
            method,
            {
                "method": method,
                "status": "not_run",
                "reason": "Method was not selected in this run.",
            },
        )
        result["run_id"] = comparison["run_id"]
        paths = resource_paths(output, method)
        paths["json"].write_text(json.dumps(result, indent=2) + "\n")
        paths["report"].write_text(render_report(result))
    text = render_comparison(comparison)
    (output / "data" / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (output / "comparison_report.txt").write_text(text)
    run = Path(comparison["run_directory"])
    if run.is_dir():
        (run / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        (run / "comparison_report.txt").write_text(text)
    root = Path(__file__).resolve().parents[2]
    case_path = output.relative_to(root) if output.is_relative_to(root) else output
    successful = [r for r in comparison["methods"].values() if r["status"] == "success"]
    settings = successful[0]["assumptions"] if successful else {}
    budget = settings.get("target_failure_probability", 0.01)
    mode = settings.get("distance_selection_mode", "per_method_minimum")
    distance = settings.get("requested_minimum_code_distance", 3) if mode == "per_method_minimum" else settings["code_distance"]
    distance_options = (
        f"--auto-code-distance --code-distance {distance}"
        if mode == "per_method_minimum"
        else f"--fixed-code-distance --code-distance {distance}"
    )
    (output / "README.md").write_text(
        "# Full Laplacian inversion resource estimates\n\n"
        "Start with [comparison_report.txt](comparison_report.txt) for the case definition, shared assumptions, results and tradeoffs.\n\n"
        "- [Annealed report](annealed_resource_report.txt)\n"
        "- [Compact report](compact_resource_report.txt)\n"
        "- [EDPC report](EDPC_resource_report.txt)\n"
        "- [Logical baseline](logical/inversion_resource_report.txt)\n\n"
        "`data/` contains JSON results. `logical/` contains the exact logical baseline; "
        "`inputs/` contains phases, the component manifest and generated component QASM. "
        "`runs/<run-id>/` preserves compiler evidence, layouts, LLI and annealing history. "
        "Large component QASM and run evidence remain local and ignored by Git.\n\n"
        "By default, each method independently selects its minimum odd distance for the same failure budget "
        "(search starts at d=3). This report's settings are recorded in the command below.\n\n"
        f"Published run: `{comparison['run_id']}`. Every visible method report is labeled with this run, including failures or unselected methods.\n\n"
        "From the repository root, regenerate this saved baseline with:\n\n```bash\n"
        "PYTHONPATH=src .venv/bin/python tools/lattice_surgery/run_inversion_architecture.py \\\n"
        f"  --logical-metrics {case_path}/logical/inversion_resource_metrics.json \\\n"
        f"  --phases-json {case_path}/inputs/phases.json \\\n"
        f"  --output-dir {case_path} --target-failure-probability {budget:g} \\\n"
        f"  {distance_options}\n```\n\n"
        "These are component-sum projections. Read the reports for component-boundary, factory, spectral-domain and compact-spacing limitations.\n"
    )
    return text
