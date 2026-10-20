#!/usr/bin/env python3
"""Compile one Clifford+T QASM input with compact, EDPC and annealed layouts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import estimate_sliced_resources as resources
import run_lattice_surgery_compiler as compiler
from output_files import prepare_output, resource_paths
from setup_floorplanner import DEFAULT_CHECKOUT, FLOORPLANNER_PIN

NATIVE_PIN = "82a1cbe64717a64a289dd4b0d02e3dc38cae3890"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qasm", type=Path, default=compiler.DEFAULT_QASM)
    parser.add_argument("--floorplanner-dir", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("compact", "edpc", "annealed", "compact_no_clogging"),
        default=["compact", "edpc", "annealed"],
    )
    parser.add_argument("--output-dir", type=Path, default=compiler.COMPILER_OUTPUT_DIR)
    parser.add_argument(
        "--budget",
        type=int,
        default=256,
        help="Annealing requests, including failures/cache hits; minimum 13.",
    )
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument(
        "--supply-count",
        type=int,
        help="Override initial EDPC M count for annealing only; sources are abstract.",
    )
    parser.add_argument(
        "--move-supply",
        action="store_true",
        help="Allow relocating M sources; count stays fixed.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="Seconds per native compiler invocation.",
    )
    parser.add_argument("--disttime", type=int, default=1)
    parser.add_argument("--staggered", action="store_true")
    parser.add_argument("--code-distance", type=int, default=9)
    parser.add_argument("--cycle-time-ns", type=float, default=1000)
    parser.add_argument("--physical-qubits-per-tile-factor", type=float, default=2)
    parser.add_argument("--physical-error-rate", type=float, default=1e-3)
    parser.add_argument("--target-failure-probability", type=float, default=1e-2)
    parser.add_argument("--factory-cycle-rounds", type=float)
    parser.add_argument("--factory-physical-qubits", type=float)
    parser.add_argument("--num-factories", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def normalize_qasm(text: str) -> tuple[str, int]:
    """Reject unsupported semantics and emit one simple instruction per line."""
    source = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    source = re.sub(r"//[^\n]*", "", source)
    if not source.rstrip().endswith(";"):
        raise ValueError("QASM statements must end with semicolons")
    statements = [s.strip() for s in source.split(";") if s.strip()]
    if not statements or statements[0] != "OPENQASM 2.0":
        raise ValueError("Expected OPENQASM 2.0")
    declarations = [
        re.fullmatch(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\]", s) for s in statements
    ]
    registers = [m for m in declarations if m]
    if len(registers) != 1 or int(registers[0][2]) < 1:
        raise ValueError("Provide exactly one nonempty quantum register")
    register, count = registers[0][1], int(registers[0][2])
    gates = []
    declared = False
    for statement in statements[1:]:
        if re.fullmatch(r'include\s+"qelib1.inc"', statement):
            continue
        if re.fullmatch(r"qreg\s+\w+\s*\[\s*\d+\s*\]", statement):
            declared = True
            continue
        if statement.startswith("barrier "):
            for operand in statement[len("barrier ") :].split(","):
                match = re.fullmatch(
                    re.escape(register) + r"(?:\s*\[\s*(\d+)\s*\])?",
                    operand.strip(),
                )
                if not match or (match[1] is not None and int(match[1]) >= count):
                    raise ValueError(f"Invalid barrier operand: {operand}")
            continue
        gate = re.fullmatch(r"(h|x|y|z|s|sdg|t|tdg|cx|cz)\s+(.+)", statement)
        if not gate or not declared:
            raise ValueError(
                f"Unsupported statement: {statement}. Export a synthesized Clifford+T circuit first."
            )
        operands = gate[2].split(",")
        expected = 2 if gate[1] in ("cx", "cz") else 1
        if len(operands) != expected:
            raise ValueError(f"Wrong operand count: {statement}")
        indices = []
        for operand in operands:
            match = re.fullmatch(
                re.escape(register) + r"\s*\[\s*(\d+)\s*\]", operand.strip()
            )
            if not match or int(match[1]) >= count:
                raise ValueError(f"Invalid qubit operand: {operand}")
            indices.append(int(match[1]))
        if len(set(indices)) != len(indices):
            raise ValueError(f"Gate operands must be distinct: {statement}")
        gates.append(gate[1] + " " + ",".join(f"q[{i}]" for i in indices) + ";")
    if not gates:
        raise ValueError("Input contains no gates")
    return (
        "\n".join(
            ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{count}];", *gates]
        )
        + "\n",
        count,
    )


def check_installation(checkout: Path) -> tuple[Path, Path]:
    """Verify source pins and adapter hash, rather than trusting its filename."""
    for directory, pin in (
        (checkout, FLOORPLANNER_PIN),
        (checkout / ".deps/liblsqecc", NATIVE_PIN),
    ):
        actual = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            [
                "git",
                "-C",
                str(directory),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
        ).strip()
        if actual != pin or dirty:
            raise ValueError(f"Expected clean pinned checkout {pin}: {directory}")
    adapter = checkout / ".deps/adapter/liblsqecc_adapter"
    exporter = checkout / ".deps/adapter/export_layout"
    build = json.loads((adapter.parent / "build.json").read_text())
    if build["native_pin"] != NATIVE_PIN:
        raise ValueError("Adapter build metadata has the wrong native revision")
    for binary in (adapter, exporter):
        if (
            build["sha256"].get(str(binary))
            != hashlib.sha256(binary.read_bytes()).hexdigest()
        ):
            raise ValueError(f"Binary differs from adapter build metadata: {binary}")
    return adapter, exporter


def output_paths(output: Path, name: str) -> dict[str, Path]:
    prepare_output(output)
    return resource_paths(output, name)


def compile_result(
    args,
    adapter: Path,
    qasm: Path,
    geometry: Path | None,
    name: str,
    paths: dict[str, Path],
):
    preset = "edpc" if name == "annealed" else name
    compilation = compiler.build_parser().parse_args([])
    compilation.slicer = adapter
    compilation.preset = preset
    compilation.geometry_file = geometry
    compilation.disttime = args.disttime
    compilation.staggered = args.staggered
    compilation.timeout = args.timeout
    compiler._run_fast(compilation, qasm, paths["lli"], paths["sliced"])
    estimation = resources.build_parser().parse_args([])
    for key in (
        "code_distance",
        "cycle_time_ns",
        "physical_qubits_per_tile_factor",
        "physical_error_rate",
        "target_failure_probability",
        "factory_cycle_rounds",
        "factory_physical_qubits",
        "num_factories",
        "disttime",
        "staggered",
        "timeout",
    ):
        setattr(estimation, key, getattr(args, key))
    estimation.slicer, estimation.qasm = adapter, qasm
    estimation.preset, estimation.geometry_file = preset, geometry
    result = resources.estimate(estimation)
    result["source_qasm"] = str(args.qasm.resolve())
    result["compiler"].update(
        preset=name,
        backend="liblsqecc/liblsqecc_adapter",
        native_commit=NATIVE_PIN,
        floorplanner_commit=FLOORPLANNER_PIN,
        compiled_qasm=str(qasm),
        geometry_file=str(geometry) if geometry else None,
    )
    result["schedule"]["tile_slices"] = (
        result["schedule"]["layout_tiles"] * result["schedule"]["num_slices"]
    )
    result["warnings"].append(
        "Abstract supply cells are not physical factory footprints; compiler output omits execution correction details."
    )
    return result


def run_comparison(args) -> bool:
    from lattice_floorplan.api import optimize_layout
    from lattice_floorplan.compiler import CompilerConfig
    from lattice_floorplan.geometry import GridState
    from lattice_floorplan.layout import Layout
    from lattice_floorplan.supply import initialize_supply, supply_count
    from lattice_floorplan.tempering import TemperingConfig

    adapter, exporter = check_installation(args.floorplanner_dir.resolve())
    original = args.qasm.read_bytes()
    qasm, qubits = normalize_qasm(original.decode("utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = output / "runs"
    runs.mkdir(exist_ok=True)
    run = Path(
        tempfile.mkdtemp(
            prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-"), dir=runs
        )
    )
    input_path = run / "input.qasm"
    input_path.write_text(qasm)
    summary = {
        "source_qasm": str(args.qasm.resolve()),
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "normalized_input_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
        "native_commit": NATIVE_PIN,
        "floorplanner_commit": FLOORPLANNER_PIN,
        "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "settings": {
            "pipeline": "wave",
            "local": True,
            "disttime": args.disttime,
            "staggered": args.staggered,
            "router": "graph_search",
            "graph_search": "djikstra",
            "cnotcorrections": "never",
        },
        "run_directory": str(run),
        "run_id": run.name,
        "layouts": {},
    }
    for name in dict.fromkeys(args.layouts):
        started = time.monotonic()
        print(f"\n=== {name}: compiling {qubits} logical qubits ===", flush=True)
        # Work in the run bundle; publish selected files only after successful verification.
        paths = output_paths(run, name)
        try:
            geometry, compiled_qasm, optimization = None, input_path, None
            initial = resources.run_checked(
                [str(exporter), "edpc" if name == "annealed" else name, str(qubits)],
                args.timeout,
            )
            layout_resources = Layout.parse(initial).resources()
            if name == "annealed":
                grid = GridState.parse(initial)
                if args.supply_count is not None:
                    grid = initialize_supply(grid, args.supply_count, args.seed)
                    if grid is None:
                        raise ValueError("Could not construct initial supply placement")
                (run / "initial-layout.txt").write_text(grid.text)
                bundle = run / "annealed_layout"
                config = CompilerConfig(
                    local=True,
                    distillation_time=args.disttime,
                    staggered=args.staggered,
                    timeout=args.timeout,
                )
                print(
                    f"Annealing: {args.budget} requests, seed {args.seed}, {supply_count(grid)} M sources",
                    flush=True,
                )
                optimization = optimize_layout(
                    qasm,
                    grid,
                    adapter_path=adapter,
                    output_directory=bundle,
                    method="sa",
                    budget=args.budget,
                    seed=args.seed,
                    search_config=TemperingConfig(move_supply=args.move_supply),
                    compiler_config=config,
                )
                compiled_qasm = bundle / "mapped-circuit.qasm"
                geometry = bundle / "layout.txt"
            result = compile_result(args, adapter, compiled_qasm, geometry, name, paths)
            if optimization:
                schedule = result["schedule"]
                if (schedule["layout_tiles"], schedule["num_slices"]) != (
                    optimization["best_area"],
                    optimization["best_slices"],
                ):
                    raise ValueError(
                        "Final resource compilation disagrees with annealer winner"
                    )
                stable_bundle = output / "artifacts" / "annealed_layout"
                stable_bundle.mkdir(parents=True, exist_ok=True)
                for filename in (
                    "layout.txt",
                    "layout.svg",
                    "identity.json",
                    "original-circuit.qasm",
                    "mapped-circuit.qasm",
                    "resources.json",
                    "result.json",
                ):
                    shutil.copyfile(bundle / filename, stable_bundle / filename)
                shutil.copyfile(
                    bundle / "replay/result.json", stable_bundle / "replay.json"
                )
                layout_resources = json.loads((bundle / "resources.json").read_text())
                result["optimization"] = {
                    "method": "sa",
                    "seed": args.seed,
                    "budget": args.budget,
                    "move_supply": args.move_supply,
                    "supply_count": supply_count(GridState.parse(geometry.read_text())),
                    "bundle": str(stable_bundle),
                    "full_evidence": str(bundle),
                    "compiler_config": asdict(config),
                    "baseline_slices": optimization["baseline"],
                    "baseline_area": optimization["baseline_area"],
                    "baseline_tile_slices": optimization["baseline_objective"],
                    "accounting": optimization["accounting"],
                    "wall_seconds": time.monotonic() - started,
                    "search_wall_seconds": optimization[
                        "wall_seconds_with_shared_cache"
                    ],
                    "best_tile_slices": optimization["best_objective"],
                    "winner_replay_verified": True,
                }
            result["layout_resources"] = layout_resources
            result["compiler"]["source_sha256"] = summary["source_sha256"]
            result["run_id"] = run.name
            paths["json"].write_text(json.dumps(result, indent=2) + "\n")
            paths["report"].write_text(resources.render_report(result))
            for key, destination in output_paths(output, name).items():
                destination.write_bytes(paths[key].read_bytes())
            summary["layouts"][name] = {
                "status": "success",
                "area_tiles": result["schedule"]["layout_tiles"],
                "slices": result["schedule"]["num_slices"],
                "tile_slices": result["schedule"]["tile_slices"],
                "physical_qubits": result["schedule_aware_estimate"][
                    "layout_physical_qubits"
                ],
                "core_time_ns": result["schedule_aware_estimate"]["core_time_ns"],
                "abstract_m_sources": layout_resources["counts"].get("M", 0),
                "explicit_factory_regions": layout_resources["factory_count"],
                "resource_json": str(output_paths(output, name)["json"]),
                "wall_seconds": time.monotonic() - started,
            }
        except (
            ValueError,
            OSError,
            RuntimeError,
            KeyError,
            subprocess.SubprocessError,
        ) as error:
            summary["layouts"][name] = {
                "status": "failed",
                "reason": str(error),
                "wall_seconds": time.monotonic() - started,
            }
            failed = {
                "method": name,
                "status": "failed",
                "reason": str(error),
                "run_id": run.name,
            }
            destinations = output_paths(output, name)
            destinations["json"].write_text(json.dumps(failed, indent=2) + "\n")
            destinations["report"].write_text(
                f"Mini-circuit architecture: {name}\nRun: {run.name}\n"
                f"Status: FAILED\n{error}\n"
                "No numerical estimate is available for this run.\n"
            )
            print(f"{name} failed: {error}", flush=True)
    (run / "layout_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    prepare_output(output)
    (output / "data" / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "Layout comparison (same input, compiler and scheduling settings)",
        f"Run: {run.name}",
        "",
        "layout | tiles | slices | tile-slices | physical qubits | core time (ns)",
    ]
    for name, row in summary["layouts"].items():
        if row["status"] == "success":
            lines.append(
                f"{name} | {row['area_tiles']} | {row['slices']} | {row['tile_slices']} | {row['physical_qubits']:g} | {row['core_time_ns']:g}"
            )
        else:
            lines.append(f"{name} | FAILED: {row['reason']}")
    lines += [
        "",
        f"Full evidence and optimized layout bundle: {run}",
        "Physical resources use a fixed-distance proxy; M supplies are abstract.",
        "Failed runs have no numeric score. Older stable files, if present, are not results of this run.",
    ]
    report = "\n".join(lines) + "\n"
    (output / "comparison_report.txt").write_text(report)
    (output / "README.md").write_text(
        "# Mini-circuit lattice-surgery results\n\n"
        "Start with [comparison_report.txt](comparison_report.txt). "
        "The annealed, compact and EDPC TXT reports summarize this small QASM circuit.\n\n"
        "`data/` contains JSON estimates; `artifacts/` contains LLI and the winning annealed layout; "
        "`runs/` contains local compiler/search history. Full Laplacian inverse results are in "
        "`../laplacian_inversion/`.\n\n"
        "Regenerate from the repository root:\n\n```bash\n"
        "PYTHONPATH=src .venv/bin/python tools/lattice_surgery/run_layout_comparison.py\n```\n"
    )
    print("\n" + report, flush=True)
    return all(row["status"] == "success" for row in summary["layouts"].values())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        resources.validate_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    if args.budget < 13 or args.timeout <= 0 or args.disttime < 1:
        parser.error("Require --budget >= 13, positive --timeout and --disttime")
    if args.supply_count is not None and args.supply_count < 1:
        parser.error("--supply-count must be positive")
    if not args.worker:
        python = args.floorplanner_dir.resolve() / ".venv/bin/python"
        if not python.is_file():
            parser.error(
                "Run .venv/bin/python tools/lattice_surgery/setup_floorplanner.py first"
            )
        raise SystemExit(
            subprocess.call(
                [str(python), str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
                env=dict(
                    os.environ, PYTHONPATH=str(args.floorplanner_dir.resolve() / "src")
                ),
            )
        )
    try:
        success = run_comparison(args)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Error: {error}\n")
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
