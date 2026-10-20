# Isolated fast lattice-surgery compiler

This directory tests
[`latticesurgery-com/liblsqecc`](https://github.com/latticesurgery-com/liblsqecc)
without adding compiler dependencies to the main `laplacian-qsvt` package. The
older Python
[`lattice-surgery-compiler`](https://github.com/latticesurgery-com/lattice-surgery-compiler)
is retained only as an explicit fallback backend.

## Why it is isolated

The main project currently uses Qiskit 2.x and Python 3.12. The legacy Python
compiler pins Qiskit below 0.35, while the optimized slicer is a separate C++
project. Neither compiler is imported by the Laplacian package.

The integration therefore uses a standalone OpenQASM 2 file:

1. You write a small circuit in `input_circuit.qasm`.
2. `liblsqecc` performs direct Clifford+T compilation and slicing.
3. Generated compiler files remain under `outputs/compiler/`.

Nothing in `src/laplacian_qsvt/` imports `lsqecc`, and the manual circuit is
not generated from the Laplacian resource-estimation code.

## Simulated-annealing layout comparison

This optional path compares compact, EDPC and circuit-specific annealed layouts
on one input. It uses the supported geometry API from
[`surface-code-floorplanner`](https://github.com/akataba/surface-code-floorplanner)
at `7835a4ef645f7c9faccc4a74c678574285a7c82b` and its native adapter at
liblsqecc `82a1cbe64717a64a289dd4b0d02e3dc38cae3890`.
The compiler used by the historical smoke-test setup below is a different pin;
the comparison rebuilds and reruns every selected layout with the new adapter.

### Setup

With the native dependencies listed below installed, run:

```bash
.venv/bin/python tools/lattice_surgery/setup_floorplanner.py
```

This creates the sibling `../surface-code-floorplanner/` checkout, installs its
NumPy-only optimizer in that checkout's `.venv`, builds its pinned native compiler,
and builds `liblsqecc_adapter` plus `export_layout`. It preserves native runtime
library paths when linking the adapter and records the resulting binary hashes.
No Conda or legacy Python compiler is required. Existing checkouts must be clean
and pinned; setup refuses to reset them. Use `--floorplanner-dir /path/to/checkout`
and, if needed, `--prefix /path/to/native/dependencies` or `--jobs 4`.

### Run the same circuit through all three methods

Edit `input_circuit.qasm`, then run from the project root:

```bash
.venv/bin/python tools/lattice_surgery/run_layout_comparison.py
```

This command automatically invokes the isolated optimizer interpreter. Defaults:
256 charged search requests, seed 61, fixed initial EDPC M-source count, wave
scheduling, local compilation, distillation interval 1, no staggering, twist-based
S gates, no CNOT corrections, graph-search/Dijkstra routing, code distance 9 and
1000 ns per QEC cycle. All three layouts use the same adapter binary, normalized
input and physical assumptions. Annealing starts from the exported native EDPC
layout and minimizes rectangular tiles multiplied by completed slices. The
winner can equal the starting layout and need not outperform compact.

To select an input and tune the search:

```bash
.venv/bin/python tools/lattice_surgery/run_layout_comparison.py \
  --qasm /path/to/circuit.qasm \
  --layouts compact edpc annealed \
  --budget 256 --seed 61 --move-supply \
  --disttime 1 --code-distance 9 --cycle-time-ns 1000
```

`--layouts annealed` runs just annealing; `compact_no_clogging` is also available.
`--move-supply` permits relocation while preserving source count.
`--supply-count N` changes that count for the annealed run only: the resulting
experiment varies supply resources as well as geometry. Reports retain each
layout's source/factory-region counts. `--staggered` enables staggering for all
methods; `--timeout` limits each native subprocess in seconds. The search budget
includes baseline/calibration, null proposals, failed candidates and cache hits;
it is not a wall-clock deadline. Native reference calls and winner replay are
additional. There is no inherited benchmark warm start or RL training.

### Write the circuit in Qiskit instead

The optional editable Python input is `input_circuit.py`; its `build_circuit()`
must return a `QuantumCircuit`. Export it in the main project environment:

```bash
.venv/bin/python tools/lattice_surgery/export_qiskit_qasm.py
.venv/bin/python tools/lattice_surgery/run_layout_comparison.py
```

The exporter overwrites `input_circuit.qasm`. Use `--input-python /path/input.py`
and `--output-qasm /path/circuit.qasm` to keep separate inputs. It flattens quantum
registers, decomposes unitary gates, synthesizes numeric Ry/Rz rotations using
the project's pyLIQTR routines, and records the synthesis budget, measured error
bound and dropped global phase in a `.synthesis.json` sidecar. The default total
budget is `--synthesis-tol 1e-6`. This may expand a small rotation-heavy circuit
into many gates; the floorplanner budget counts candidate compilations of that
expanded circuit. It does not automatically export a full Laplacian inverse.

Direct QASM input supports one nonempty register and indexed H, X, Y, Z, S, Sdg,
T, Tdg, CX and CZ instructions. Comments and barriers are removed; statements are
normalized to one per line. Arbitrary rotations, custom gate definitions,
measurements, classical control, resets and multiple QASM registers are rejected
before search. Use the Qiskit exporter for decomposition/synthesis first.

### Outputs, replay and interpretation

Stable files under `outputs/compiler/` include:

- Existing compact and EDPC LLI/resource filenames, refreshed with matched settings.
- `annealed.lli` and `annealed_sliced.lli`.
- `annealed_resource_estimate.json`, `annealed_resource_report.txt`.
- `layout_comparison.json` and `layout_comparison.txt`, including failures.
- `annealed_layout/`: winning grid/SVG, logical identity mapping, original and
  mapped QASM, resources, search history and fresh-replay record.

Every invocation also keeps its complete evidence in a new
`outputs/compiler/annealing_runs/<run-id>/` directory, ignored by Git. Stable
files are published only after that method's final compilation succeeds. A failed
method has no numeric score in the new comparison report and causes a nonzero
exit; older stable files may still exist and must not be interpreted as the new
run. `--output-dir /path/to/new-directory` keeps comparisons entirely separate.

The final resource compilation must match the optimized area/slice count.
Completed slices are counted from patch JSON and checked against both the pinned
wave counter (which includes an extra initial count) and sliced LLI. Temporary
patch JSON is deleted after extraction. Input, geometry and binary hashes plus
both source revisions are recorded.

For portable replay of the saved winner (paths relative to the project root):

```bash
../surface-code-floorplanner/.deps/adapter/liblsqecc_adapter \
  -I qasm -i outputs/compiler/annealed_layout/mapped-circuit.qasm \
  -L edpc --geometry-file outputs/compiler/annealed_layout/layout.txt \
  -P wave --local --disttime 1 --nostagger --cnotcorrections never \
  -r graph_search -g djikstra --printlli sliced --graceful
```

Use the settings saved in the resource report when overriding defaults. Preserve
the grid, `identity.json` and matching QASM together; do not remap mapped QASM a
second time. Stock `-l layout.txt` follows different preprocessing and is not
equivalent to this adapter path.

Physical resources use the existing fixed-distance proxy (tiles × 2d² and slices
× d × cycle time by default). M sources are abstract supplies, not complete
physical distillation factories. The optional factory model is separate. The
pinned compiler omits measurement-dependent correction details and does not
distinguish T/T-dagger in its printed lowering: replay certifies the resource
score, not an end-to-end executable quantum protocol. See the floorplanner's
[correctness limits](https://github.com/akataba/surface-code-floorplanner/blob/7835a4ef645f7c9faccc4a74c678574285a7c82b/docs/correctness.md).

## Full inverse component architecture

`run_inversion_architecture.py` connects the full logical inversion estimate to
the existing native adapter and annealer without materializing the long target
sequence. Run `setup_floorplanner.py` once as described above, then:

```bash
PYTHONPATH=src .venv/bin/python tools/lattice_surgery/run_inversion_architecture.py \
  --logical-metrics outputs/metrics/laplacian_inversion_resource_metrics.json \
  --phases-json outputs/phases/phases_inversion.json \
  --budget 32 --seed 61 --auto-code-distance
```

The default logical/phase paths are those shown. Target dimensions, n, synthesis
tolerance and phase multiplicities come from the logical JSON; generate that JSON
first to change the algorithm. A different environment/decomposition that changes
component gate counts is rejected; rerun the logical estimator first. Component
exports are reused only when logical JSON, phase and QASM hashes match.
`--rebuild-components` forces export. Phase order does not affect this
component-sum model when multiplicities are unchanged.

`export_inversion_components.py` uses the same Clifford+T basis and decomposition
as the logical counters. It exports W and each unique synthesized projector,
preserving inverse gates and synthesis tolerances. Projector predicates map to
the walk's ancillas, the clean flag maps to the additional resident qubit, and
dirty workspace maps to existing system qubits. Every QASM retains the full
resident register. Component counts and weighted totals must match the logical
estimate before architectural compilation.

Each method keeps one common grid/identity mapping for W and every projector.
Native EDPC uses its exported grid. Compact first tries its native grid, then
(only on the known too-short Bell-CNOT route failure) inserts routing columns
with spacing 2/3 while preserving Q/A cells and the contiguous factory region.
Failures are retained. `--compact-spacing 1`, `2` or `3` prescribes the geometry;
`0` means native-then-spacing. A spaced compact result is labeled and is not
represented as a successful native compact baseline.

Annealing uses the pinned SA API with one W followed by the largest-T projector
as the search circuit. It minimizes that template's area × slices, ensuring
phase-flag interactions participate. This is not optimization over all angles or
the full sequence. The verified winning grid/identity mapping is then reused to
measure W and every distinct phase separately. Each final measurement must
complete; a failed phase prevents a numerical estimate for that method.
`--supply-count` changes annealing's abstract M count only and is reported.

For r walk applications and m_j occurrences of phase j:

```text
S_total = r * S_W + sum_j(m_j * S_Rj)
tile-slices = shared layout area * S_total
QEC rounds = S_total * code distance
time = QEC rounds * QEC cycle time
resident physical qubits = shared area * tile factor * code distance^2
```

Qubits are allocated once, including idle flag/system qubits. There are no free
permutations or geometry changes between component types. However, each compiler
invocation starts from default patch orientations and supply availability.
Boundary orientation resets, replenishment carryover and reconciliation costs
are omitted. The result is a component-sum projection, not a scheduled full
circuit or a certified runtime upper bound.

Native evaluation uses summary-only wave counts corrected for the initial extra
count, avoiding large patch-JSON allocations. Unsliced LLI is streamed to disk
and counted for state requests; sliced LLI is streamed and its line count checked
against completed frames. `--skip-lli` omits sliced LLI files; unsliced counts
remain available. No compiler report is generated. `--timeout` limits each native
subprocess; the default annealing `--budget 32` counts score requests, not seconds.

`--auto-code-distance` selects a common odd distance meeting every successful
method's layout-error proxy at `--target-failure-probability` (default 0.01),
with `--code-distance` as a lower bound (default 9). The proxy is
`min(1, area * S_total * d * 0.1 * (100*p_phys)^((d+1)/2))`.
Selection rescales physical estimates without changing measured logical slices.
It excludes magic-state infidelity, decoder behavior and full physical execution
semantics. Without auto selection, distance stays fixed and unmet targets produce
warnings. The full sequence can require a much larger distance than the mini case.

Other physical parameters use the same units as the mini workflow. Supply
intervals are deterministic slices. Optional `--factory-cycle-rounds` requires
explicit `--num-factories`; `--factory-physical-qubits` adds external factory space
once, in addition to abstract in-grid regions. Do not assign the same hardware to
both. This global throughput bound does not reconstruct supply stalls, and the
layout-error proxy excludes factory output errors. Polynomial/synthesis errors
and spectral limitations remain separate in the logical input metadata.

Outputs default to `outputs/compiler/inversion_<D>d_n<n>/` (override with
`--output-dir`), using `EDPC`, `compact` and `annealed` method prefixes:

- `*_resource_report.txt` and `*_resource_estimate.json`: full projections with
  per-W and per-angle measurements, multiplicities and assumptions.
- `architecture_comparison.txt` and `.json`: all methods, including failures.
- `manifest.json`: source/QASM hashes, register map and logical multiplicities.
- `inputs/`, snapshots and `runs/<id>/`: local QASM, grids/identity mappings,
  annealing history/replay and component LLI.

Small reports/manifests are allowed by Git; detailed inputs/runs are ignored.
Component evidence directories retain exact commands, input/geometry hashes,
stdout and stderr. Keep the complete local bundle for replay. The same compiler
semantic limitations documented above still apply.

## Native build dependencies

On macOS, install the compiler toolchain and libraries with:

```bash
xcode-select --install
brew install cmake pkg-config sqlite zstd libpq boost
```

On Debian/Ubuntu, the equivalent official dependencies are:

```bash
sudo apt install -y \
  build-essential \
  cmake \
  pkg-config \
  libsqlite3-dev \
  libzstd-dev \
  libpq-dev \
  libboost-dev
```

Boost is optional, but enables faster path finding when detected.

## One-time setup

Run from the GTRI repository root:

```bash
bash tools/lattice_surgery/setup_compiler.sh
```

The script:

- Clones the compiler as a sibling directory, not inside this repository.
- Checks out the tested compiler commit
  `1d9d9f36c004847e478ad464085927eebc13a0fe`.
- Initializes the required OpenSurgery submodule.
- Creates the separate `lsqecc310` Python 3.10 Conda environment.
- Installs the legacy Qiskit Terra version and a maintained Tweedledum build.
- Installs `lsqecc` only in that environment.
- Clones and builds pinned `liblsqecc` commit
  `fddaecf0d929b0afa0ae72a1adc1df865fab4e18`.
- Produces the optimized `liblsqecc/build/lsqecc_slicer` executable.

The fast compiler is deliberately cloned next to GTRI rather than inside it:

```text
quantum_research/
├── GTRI/
├── lattice-surgery-compiler/
└── liblsqecc/
```

Override locations if needed:

```bash
LSQECC_REPO_DIR=/path/to/lattice-surgery-compiler \
LSQECC_ENV_NAME=my-lsqecc-env \
LIBLSQECC_REPO_DIR=/path/to/liblsqecc \
bash tools/lattice_surgery/setup_compiler.sh
```

The setup script performs the official native build steps:

```bash
git clone --recursive https://github.com/latticesurgery-com/liblsqecc.git
cmake -S liblsqecc -B liblsqecc/build -DCMAKE_BUILD_TYPE=Release
cmake --build liblsqecc/build --target lsqecc_slicer -j 4
```

## Write a circuit

Edit:

`tools/lattice_surgery/input_circuit.qasm`

The tracked file contains only the required OpenQASM 2 header, one logical
register, and commented example gates. Change the register size and add gates
below it. For example:

```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];

h q[0];
cx q[0],q[1];
t q[1];
```

## Run the compiler

The fast direct backend is the default and can run from the main environment:

```bash
.venv/bin/python \
  tools/lattice_surgery/run_lattice_surgery_compiler.py
```

This writes:

- `outputs/compiler/EDPC.lli`
- `outputs/compiler/EDPC_sliced.lli`

By default, the runner uses the upgraded direct-compilation configuration from
the liblsqecc resource-estimation workflow:

- EDPC layout
- Wave scheduling
- Local lattice-surgery instruction decomposition
- Distillation time 1 with unstaggered factories
- Catalytic S-gate implementation disabled through `--notwists`

`EDPC.lli` contains the layout-independent instructions before routing.
`EDPC_sliced.lli` contains the scheduled instructions, with operations on
the same line executing in the same timestep.

## Estimate scheduled physical resources

Run the standalone schedule-aware estimator:

```bash
.venv/bin/python \
  tools/lattice_surgery/estimate_sliced_resources.py \
  --code-distance 7 \
  --cycle-time-ns 1000
```

It reruns the same `liblsqecc` preset into a temporary patch JSON, extracts
the exact schedule and layout dimensions, counts state requests from LLI, and
deletes the temporary JSON. It writes:

- `outputs/compiler/EDPC_resource_estimate.json`
- `outputs/compiler/EDPC_resource_report.txt`

The transparent core model is:

```text
QEC rounds  = scheduled slices * code distance
core time   = QEC rounds * surface-code cycle time
core qubits = layout tiles * tile factor * code distance^2
```

The JSON also includes the repository-compatible legacy core formulas for
comparison. For an optional physical factory-throughput model:

```bash
.venv/bin/python \
  tools/lattice_surgery/estimate_sliced_resources.py \
  --factory-cycle-rounds 15 \
  --factory-physical-qubits 1000 \
  --num-factories 2
```

Without a factory cadence, the report uses the scheduler's `--disttime`
assumption and emits a warning instead of inventing a factory runtime.

For the compact/stream layout closest to the basic viewer example:

```bash
.venv/bin/python tools/lattice_surgery/run_lattice_surgery_compiler.py \
  --preset viewer
```

To use the legacy Python backend explicitly:

```bash
/opt/anaconda3/envs/lsqecc310/bin/python \
  tools/lattice_surgery/run_lattice_surgery_compiler.py \
  --backend python
```

That fallback disables state-vector simulation and the pinned Python
compiler's broken optional resource estimator. It does not disable slice
generation.

## Direct `lsqecc_slicer` usage

The executable built by the setup script is:

```text
../liblsqecc/build/lsqecc_slicer
```

Print layout-independent LLI:

```bash
../liblsqecc/build/lsqecc_slicer \
  -I qasm \
  -i tools/lattice_surgery/input_circuit.qasm \
  -L compact \
  -P stream \
  --graceful \
  --printlli before \
  > outputs/compiler/EDPC.lli
```

Print LLI sliced with the resource-estimation preset:

```bash
../liblsqecc/build/lsqecc_slicer \
  -I qasm \
  -i tools/lattice_surgery/input_circuit.qasm \
  -L edpc \
  --disttime 1 \
  --nostagger \
  --notwists \
  --local \
  -P wave \
  --graceful \
  --printlli sliced \
  > outputs/compiler/EDPC_sliced.lli
```

Important options:

- `-I qasm`: parse OpenQASMmin input.
- `-i`, `-o`: select input and output files.
- `-L compact|compact_no_clogging|edpc`: generate a layout.
- `-P stream|wave|edpc`: select the scheduling pipeline.
- `-r graph_search|graph_search_cached`: select the router.
- `-g djikstra|astar|boost`: select path finding.
- `--numlanes`, `--condensed`, `--explicitfactories`: configure EDPC.
- `--disttime`, `--nostagger`: configure magic-state distillation.
- `--local`, `--notwists`: select direct local surgery decompositions.
- `--printlli`, `--printdag`, `--noslices`: inspect intermediate forms.
- `--graceful`: report slicing failures without an uncontrolled abort.

Run `../liblsqecc/build/lsqecc_slicer --help` for the complete option list.

## Scope

Start with a small hand-written Clifford+T circuit. A fully expanded
target-size QSP circuit can contain millions of gates and should be tested only
after validating progressively larger examples.

`export_rotation_qasm.py` is retained as an optional future utility, but it is
not connected to the default compiler input.
