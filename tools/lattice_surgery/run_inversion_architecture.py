#!/usr/bin/env python3
"""Project the complete inverse's architectural cost from compiled components."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import estimate_sliced_resources as resources
from compiler_options import backend_options, output_prefix
from export_inversion_components import export_components
from inversion_architecture import aggregate_components, render_report, space_compact
from run_layout_comparison import check_installation, NATIVE_PIN
from setup_floorplanner import DEFAULT_CHECKOUT, FLOORPLANNER_PIN

ROOT = Path(__file__).resolve().parents[2]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logical-metrics",
        type=Path,
        default=ROOT / "outputs/metrics/laplacian_inversion_resource_metrics.json",
    )
    parser.add_argument(
        "--phases-json",
        type=Path,
        default=ROOT / "outputs/phases/phases_inversion.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--floorplanner-dir", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("compact", "edpc", "annealed"),
        default=["compact", "edpc", "annealed"],
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=32,
        help="Annealing requests for one W + representative projector (minimum 13).",
    )
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--move-supply", action="store_true")
    parser.add_argument(
        "--supply-count",
        type=int,
        help="Override annealing's abstract M-source count only.",
    )
    parser.add_argument(
        "--compact-spacing",
        type=int,
        choices=(0, 1, 2, 3),
        default=0,
        help="0: try native compact, then routing spacing 2/3 if local Bell-CNOT routes fail. 1: native only.",
    )
    parser.add_argument(
        "--timeout", type=float, default=60, help="Seconds per native subprocess."
    )
    parser.add_argument("--disttime", type=int, default=1)
    parser.add_argument("--staggered", action="store_true")
    parser.add_argument("--code-distance", type=int, default=9)
    parser.add_argument(
        "--auto-code-distance",
        action="store_true",
        help="Select a common odd distance meeting every successful method's layout-failure proxy; --code-distance sets the minimum.",
    )
    parser.add_argument("--cycle-time-ns", type=float, default=1000)
    parser.add_argument("--physical-qubits-per-tile-factor", type=float, default=2)
    parser.add_argument("--physical-error-rate", type=float, default=1e-3)
    parser.add_argument("--target-failure-probability", type=float, default=1e-2)
    parser.add_argument("--factory-cycle-rounds", type=float)
    parser.add_argument("--factory-physical-qubits", type=float)
    parser.add_argument("--num-factories", type=int)
    parser.add_argument(
        "--skip-lli",
        action="store_true",
        help="Keep summaries and unsliced instruction counts; omit sliced LLI files.",
    )
    parser.add_argument("--rebuild-components", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stream_lli(command, path, timeout):
    with path.open("w") as output:
        result = subprocess.run(
            command, stdout=output, stderr=subprocess.PIPE, text=True, timeout=timeout
        )
    if result.returncode != 0 or result.stderr.strip():
        raise RuntimeError(
            f"LLI generation failed: {result.stderr or path.read_text()[:1000]}"
        )


def measure_components(manifest, texts, grid, mode, compiler, folder, args):
    from lattice_floorplan.geometry import remap_qasm
    from lattice_floorplan.layout import Layout

    folder.mkdir(parents=True, exist_ok=True)
    layout_path = folder / "layout.txt"
    layout_path.write_text(grid.text)
    records = []
    for index, component in enumerate(manifest["components"]):
        identifier = component["id"]
        dest = folder / "components" / identifier
        mapped = remap_qasm(texts[identifier], grid)
        record = compiler.run(
            mapped,
            Layout.parse(grid.text),
            dest,
            "qasm",
            retain_slices=False,
            preprocess_layout=mode,
        )
        if record["status"] != "success":
            diagnostics = (dest / "stderr.txt").read_text() + (
                dest / "stdout.txt"
            ).read_text()
            raise RuntimeError(
                f"{identifier}: {record['reason']}: {diagnostics[:1600]}"
            )
        prefix = [
            str(compiler.binary),
            "-I",
            "qasm",
            "-i",
            str(dest / "input.qasm"),
            "--graceful",
        ]
        options = backend_options(mode, args.disttime, args.staggered)
        options += ["--geometry-file", str(layout_path)]
        # Preserve native compact preprocessing for its spaced exported grid;
        # annealing retains its separate EDPC preprocessing contract.
        unsliced = dest / f"{identifier}.lli"
        stream_lli([*prefix, *options, "--printlli", "before"], unsliced, args.timeout)
        instructions = Counter()
        with unsliced.open() as source:
            for line in source:
                instructions.update(resources.instruction_counts(line))
        record["instruction_counts"] = dict(instructions)
        if not args.skip_lli:
            sliced = dest / f"{identifier}_sliced.lli"
            stream_lli(
                [*prefix, *options, "--printlli", "sliced"], sliced, args.timeout
            )
            with sliced.open() as source:
                lines = sum(bool(line.strip()) for line in source)
            if lines != record["slices"]:
                raise ValueError(
                    f"{identifier}: printed LLI disagrees with completed wave frames"
                )
        (dest / "resource_measurement.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        records.append(record)
        if index == 0 or index % 10 == 0 or index + 1 == len(manifest["components"]):
            print(
                f"  {identifier}: {record['slices']:,} slices ({index + 1}/{len(manifest['components'])} component types)",
                flush=True,
            )
    return records


def run_worker(args):
    from lattice_floorplan.api import optimize_layout
    from lattice_floorplan.compiler import Compiler, CompilerConfig
    from lattice_floorplan.geometry import GridState, write_layout_output
    from lattice_floorplan.layout import Layout
    from lattice_floorplan.supply import initialize_supply
    from lattice_floorplan.tempering import TemperingConfig

    output = args.output_dir.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    texts = {}
    for component in manifest["components"]:
        path = Path(component["qasm"])
        if digest(path) != component["qasm_sha256"]:
            raise ValueError(f"Component input changed: {path}")
        texts[component["id"]] = path.read_text()
    adapter, exporter = check_installation(args.floorplanner_dir.resolve())
    config = CompilerConfig(
        local=True,
        distillation_time=args.disttime,
        staggered=args.staggered,
        timeout=args.timeout,
    )
    compiler = Compiler(adapter, config)
    run_root = output / "runs"
    run_root.mkdir(exist_ok=True)
    run = Path(
        tempfile.mkdtemp(
            prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-"), dir=run_root
        )
    )
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    representative = max(
        (c for c in manifest["components"] if c["kind"] == "projector"),
        key=lambda c: c["t_count"],
    )
    # Inputs are normalized with exactly three header/declaration lines.
    pair = texts["W"] + "\n".join(texts[representative["id"]].splitlines()[3:]) + "\n"
    results = {}
    measurements = {}
    for method in dict.fromkeys(args.layouts):
        print(f"\n=== Full inverse components: {method} ===", flush=True)
        started = time.monotonic()
        method_dir = run / output_prefix(method)
        method_dir.mkdir()
        attempts = []
        try:
            initial_text = resources.run_checked(
                [
                    str(exporter),
                    "edpc" if method == "annealed" else method,
                    str(manifest["logical_qubits"]),
                ],
                args.timeout,
            )
            optimization = None
            if method == "annealed":
                initial = GridState.parse(initial_text)
                if args.supply_count:
                    initial = initialize_supply(initial, args.supply_count, args.seed)
                    if initial is None:
                        raise ValueError(
                            "Could not initialize requested M-source count"
                        )
                print(
                    f"  Annealing one W + {representative['id']}: {args.budget} requests, seed {args.seed}",
                    flush=True,
                )
                opt = optimize_layout(
                    pair,
                    initial,
                    adapter_path=adapter,
                    output_directory=method_dir / "optimizer",
                    method="sa",
                    budget=args.budget,
                    seed=args.seed,
                    search_config=TemperingConfig(move_supply=args.move_supply),
                    compiler_config=config,
                )
                grid = GridState.parse(opt["best_grid"], opt["logical_qubit_positions"])
                optimization = {
                    "budget": args.budget,
                    "seed": args.seed,
                    "representative_projector": representative["id"],
                    "move_supply": args.move_supply,
                    "wall_seconds": opt["wall_seconds_with_shared_cache"],
                    "accounting": opt["accounting"],
                    "replay_verified": opt["replay"]["status"] == "success",
                    "best_pair_tile_slices": opt["best_objective"],
                }
                candidates = [
                    (
                        grid,
                        "Annealed shared layout, optimized on one W + representative projector",
                    )
                ]
            elif method == "compact":
                factors = (
                    (1, 2, 3) if args.compact_spacing == 0 else (args.compact_spacing,)
                )
                candidates = [
                    (
                        GridState.parse(space_compact(initial_text, factor)),
                        (
                            "Native compact"
                            if factor == 1
                            else f"Compact with routing spacing {factor}; factory region preserved"
                        ),
                    )
                    for factor in factors
                ]
            else:
                candidates = [(GridState.parse(initial_text), "Native EDPC")]
            last_error = None
            for attempt, (grid, preparation) in enumerate(candidates):
                bundle = method_dir / f"layout_{attempt}"
                write_layout_output(bundle, grid, pair, preparation)
                try:
                    # The common geometry is replayed for every component without
                    # relabeling between calls or dropping idle logical qubits.
                    mode = "compact" if method == "compact" else "edpc"
                    records = measure_components(
                        manifest, texts, grid, mode, compiler, bundle, args
                    )
                    attempts.append({"preparation": preparation, "status": "success"})
                    break
                except (RuntimeError, ValueError, subprocess.SubprocessError) as error:
                    last_error = error
                    attempts.append(
                        {
                            "preparation": preparation,
                            "status": "failed",
                            "reason": str(error),
                        }
                    )
                    if method != "compact" or "at least two tiles" not in str(error):
                        raise
                    print(
                        f"  {preparation} failed on a short local route; trying routing spacing",
                        flush=True,
                    )
            else:
                raise RuntimeError(str(last_error))
            layout_resources = Layout.parse(grid.text).resources()
            result = aggregate_components(
                manifest, records, layout_resources["area"], args
            )
            measurements[method] = records
            result.update(
                method=method,
                layout_resources=layout_resources,
                layout_preparation=preparation,
                layout_attempts=attempts,
                evidence_directory=str(bundle),
                wall_seconds=time.monotonic() - started,
                compiler={
                    "native_commit": NATIVE_PIN,
                    "floorplanner_commit": FLOORPLANNER_PIN,
                    "binary_sha256": compiler.binary_sha256,
                    "config": asdict(config),
                },
                logical_metrics_sha256=manifest["logical_metrics_sha256"],
                phases_sha256=manifest["phases_sha256"],
            )
            if optimization:
                result["optimization"] = optimization
            if attempts[0]["status"] == "failed":
                result["warnings"].append(
                    "Native compact failed local decomposition. The reported compact estimate uses explicitly spaced compact geometry, not the original native grid."
                )
            if args.supply_count and method == "annealed":
                result["warnings"].append(
                    "Annealing uses an overridden M-source count; supply resources differ from native baselines."
                )
        except (
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            subprocess.SubprocessError,
        ) as error:
            result = {
                "method": method,
                "status": "failed",
                "reason": str(error),
                "layout_attempts": attempts,
                "evidence_directory": str(method_dir),
            }
            print(f"  FAILED: {error}", flush=True)
        prefix = output_prefix(method)
        (output / f"{prefix}_resource_estimate.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        (output / f"{prefix}_resource_report.txt").write_text(render_report(result))
        results[method] = result
    if args.auto_code_distance and any(
        r["status"] == "success" for r in results.values()
    ):
        distances = [
            r["physical_projection"]["minimum_odd_code_distance_proxy"]
            for r in results.values()
            if r["status"] == "success"
        ]
        if any(distance is None for distance in distances):
            raise ValueError(
                "No odd code distance <=101 satisfies the requested failure proxy"
            )
        requested = args.code_distance
        selected = max(requested, *distances)
        selected += selected % 2 == 0
        args.code_distance = selected
        for method, records in measurements.items():
            result = results[method]
            if result["status"] != "success":
                continue
            additional = [
                warning
                for warning in result["warnings"]
                if warning.startswith(("Native compact", "Annealing uses"))
            ]
            result.update(
                aggregate_components(
                    manifest,
                    records,
                    result["schedule_projection"]["layout_tiles"],
                    args,
                )
            )
            result["warnings"].extend(additional)
            result["assumptions"].update(
                requested_minimum_code_distance=requested,
                distance_selection="Common odd distance satisfying every successful method's layout-failure proxy; physical factory errors are not included.",
            )
            prefix = output_prefix(method)
            (output / f"{prefix}_resource_estimate.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            (output / f"{prefix}_resource_report.txt").write_text(render_report(result))
        print(
            f"Selected common code distance {selected} under the layout-failure proxy (minimum {requested}).",
            flush=True,
        )
    comparison = {
        "scope": "full_inverse_component_sum_comparison",
        "manifest": str(output / "manifest.json"),
        "run_directory": str(run),
        "methods": results,
    }
    (output / "architecture_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
    lines = [
        "Full inverse: component-sum architectural projection",
        "layout | tiles | projected slices | physical qubits | projected time (s)",
    ]
    for method, result in results.items():
        if result["status"] == "success":
            lines.append(
                f"{method} | {result['schedule_projection']['layout_tiles']} | {result['schedule_projection']['slices']} | {result['physical_projection']['physical_qubits']:g} | {result['physical_projection']['runtime_seconds']:.9g}"
            )
        else:
            lines.append(f"{method} | FAILED: {result['reason']}")
    lines += [
        "",
        "One common resident layout per method. Boundary reset/supply carryover costs are omitted.",
        "Annealing searches one W + representative projector; full totals weight each angle's measured cost.",
    ]
    text = "\n".join(lines) + "\n"
    (output / "architecture_comparison.txt").write_text(text)
    print("\n" + text, flush=True)
    return all(r["status"] == "success" for r in results.values())


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        resources.validate_arguments(args)
        if args.budget < 13:
            raise ValueError("--budget must be at least 13")
        if args.factory_cycle_rounds is not None and args.num_factories is None:
            raise ValueError(
                "Supply cells are not factories; provide --num-factories with --factory-cycle-rounds"
            )
        if args.supply_count is not None and args.supply_count < 1:
            raise ValueError("--supply-count must be positive")
        if args.output_dir is None:
            logical = json.loads(args.logical_metrics.read_text())["logical_resources"]
            args.output_dir = (
                ROOT
                / f"outputs/compiler/inversion_{logical['dimensions']}d_n{logical['target_n']}"
            )
        args.output_dir = args.output_dir.resolve()
        if args.worker:
            raise SystemExit(0 if run_worker(args) else 1)
        python = args.floorplanner_dir.resolve() / ".venv/bin/python"
        if not python.is_file():
            raise ValueError("Run tools/lattice_surgery/setup_floorplanner.py first")
        manifest_path = args.output_dir / "manifest.json"
        reuse = False
        if manifest_path.is_file() and not args.rebuild_components:
            manifest = json.loads(manifest_path.read_text())
            reuse = (
                manifest.get("schema_version") == 1
                and manifest["logical_metrics_sha256"] == digest(args.logical_metrics)
                and manifest["phases_sha256"] == digest(args.phases_json)
                and all(
                    digest(Path(c["qasm"])) == c["qasm_sha256"]
                    for c in manifest["components"]
                )
            )
        if not reuse:
            export_components(args.logical_metrics, args.phases_json, args.output_dir)
        else:
            print(
                "Reusing component exports matched to the logical metrics and phase hashes.",
                flush=True,
            )
        # Import the pinned source explicitly even on hosts that mark editable .pth
        # files hidden; the optimizer still runs in its separate NumPy environment.
        environment = dict(
            os.environ, PYTHONPATH=str(args.floorplanner_dir.resolve() / "src")
        )
        command = [str(python), str(Path(__file__).resolve()), *sys.argv[1:]]
        if "--output-dir" not in sys.argv[1:]:
            command += ["--output-dir", str(args.output_dir)]
        command += ["--worker"]
        raise SystemExit(subprocess.call(command, env=environment))
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
