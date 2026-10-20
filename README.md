# Periodic Laplacian QSVT Resource Estimation

This project estimates logical, fault-tolerant quantum resources for applying
a matrix-inversion polynomial to a periodic discrete Laplacian. It combines:

- QSP/QSVT matrix-inversion polynomial construction with
  [pyqsp](https://github.com/ichuang/pyqsp).
- Scalable Qiskit circuits for 1D, 2D, and 3D periodic-Laplacian block
  encodings.
- A qubitization walk operator whose signal block contains the normalized
  Laplacian.
- Explicit Clifford+T synthesis of arbitrary single-qubit rotations with
  [pyLIQTR](https://github.com/isi-usc-edu/pyLIQTR).
- Logical-qubit, Clifford, T-count, depth, and T-depth reports for the complete
  QSP walk sequence.

The code performs dense matrix checks only at a small verification size. It
computes target-size resource estimates from scalable circuits without
constructing the exponentially large Laplacian matrix.

## Algorithm overview

For `dims = D` spatial dimensions and `n` qubits per dimension, each axis has
`N = 2**n` grid points and the system register has `D*n` qubits.

The normalized periodic Laplacian is

$$
\widetilde L_p^{(D)}
= \frac{1}{D}\sum_{j=0}^{D-1}\widetilde L_{p,j}^{(1)},\qquad
\widetilde L_p^{(1)}
= \frac{1}{4}(S^+ + S^- - 2I),
$$

where `S+` and `S-` are cyclic increment and decrement operators.

The implementation proceeds through four layers:

1. `U_L` block-encodes the normalized Laplacian.
2. The qubitization walk `W` satisfies
   `<Pi|W|Pi> = -L_tilde`, where `|Pi> = |+>|0...0>`.
3. `inversion-phases` constructs an approximation to the scaled inverse
   `1/(2*kappa*x)` on
   `[1/kappa, 1] U [-1, -1/kappa]`.
4. The final estimator composes
   `R(phi_0) W R(phi_1) ... W R(phi_d)` and counts all Clifford+T resources.

For 3D, the selector-state preparation contains an arbitrary `Ry` rotation.
Every QSP projector rotation also contains an arbitrary `Rz(2*phi)`. These
rotations are explicitly synthesized into emitted Clifford+T gate sequences;
they are not estimated only through an asymptotic formula.

## Requirements

- macOS or Linux
- Git
- Python 3.10 through 3.12.2

Python newer than 3.12.2 is currently unsupported by the pinned pyLIQTR
revision.

## Installation

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs the package in editable mode and installs the
dependencies declared in `pyproject.toml`. Editable installation provides the
command-line programs documented below.

Verify the installation:

```bash
inversion-phases --help
laplacian-inversion-resource --help
python -m unittest discover -s tests -v
```

## Complete matrix-inversion workflow

### 1. Generate the inversion polynomial and QSP phases

```bash
inversion-phases \
  --epsilon 0.49 \
  --kappa 2.5 \
  --synthesis-tol 1e-8
```

This creates:

- `outputs/phases/phases_inversion.json`
- `outputs/plots/qsp_inversion_fit.png`

Important parameters:

- `--epsilon` controls the matrix-inversion polynomial approximation.
- `--kappa` defines the valid inversion domain. Modes with
  `|lambda| < 1/kappa` are outside that domain.
- `--synthesis-tol` is a separate implementation-error budget for all
  pyLIQTR-synthesized rotations in the final QSP circuit.

Polynomial error and Clifford+T synthesis error are calculated and reported
separately.

### 2. Estimate the complete 2D QSP circuit

```bash
laplacian-inversion-resource \
  --dims 2 \
  --verify-n 1 \
  --target-n 20
```

`--verify-n` and `--target-n` have different purposes:

- `--verify-n` selects a small matrix size for dense correctness checks.
- `--target-n` selects the qubits per dimension used for logical resources.

For this example, dense verification uses a `2 x 2` grid, while the resource
estimate represents `2**20` grid points per dimension.

### 3. Estimate the complete 3D QSP circuit

Start with a small target:

```bash
laplacian-inversion-resource \
  --dims 3 \
  --verify-n 1 \
  --target-n 2
```

The 3D estimate includes pyLIQTR synthesis for both the QSP projector rotations
and the selector-preparation rotations inside every walk application.

Large 3D target sizes can take a long time because the current Qiskit walk
estimator transpiles large multi-controlled shift circuits.

## Final resource report

The complete inverse-sequence logical estimate is saved in both formats:

- `outputs/metrics/laplacian_inversion_resource_metrics.json`
- `outputs/metrics/laplacian_inversion_resource_report.txt`

The text report shows the entire sequence's totals, one-walk and repeated-walk
costs, projector-rotation contributions, error estimates, spectral coverage and
small-size verification results. These are logical Clifford+T component costs
composed across the sequence; this output does not imply a full target-size QASM
export or physical-layout compilation.

Every `laplacian-inversion-resource` run writes both files. Use
`--output-resource-report` (or `--output-report`) to override the text path.
By default it follows `--output-resource-metrics` in the same directory.

Its main sections are:

- `error_summary`: polynomial, phase-numerical, and synthesis errors.
- `verification`: dense block-encoding and QSP correctness checks.
- `logical_resources`: total logical qubits, gate counts, and depths.
- `walk_contribution`: cost of one `W` and all repeated applications.
- `projector_rotation_contribution`: explicit resources for all
  `R(phi)` operations and unique pyLIQTR rotation syntheses.

The terminal prints the most important values:

- Logical qubits
- Clifford gate count
- T-count, counting both `T` and `T-dagger`
- Total depth
- T-depth
- Number of walk applications and projector rotations
- Polynomial and synthesis error bounds

Depth and T-depth are conservative serial component-sum upper bounds.

The dense verification report is written separately to:

`outputs/metrics/block_metrics_laplacian.json`

## Component-based fault-tolerant architecture for the entire inverse

After generating the full logical inverse estimate, run from this root:

```bash
PYTHONPATH=src .venv/bin/python tools/lattice_surgery/run_inversion_architecture.py \
  --auto-code-distance
```

This reuses the installed floorplanner/adapter setup. It exports the same
Clifford+T W and every distinct projector phase used by the logical estimate,
requires their gate counts to match, measures each component once per layout,
and multiplies its cost by its occurrence count. The saved example has 117 W
applications and 118 rotations at 59 distinct angles. A single arbitrary
rotation is not substituted for all angles.

Reports for each method are written under
`outputs/laplacian_inversion/<D>d_n<n>/`. For the saved 2D example, start with
[comparison_report.txt](outputs/laplacian_inversion/2d_n20/comparison_report.txt).
The same folder has `EDPC_resource_report.txt`, `compact_resource_report.txt`
and `annealed_resource_report.txt`. The comparison describes the case, shared
assumptions, walk/rotation breakdown, qubit/runtime tradeoffs and model limits.

Supporting files have separate locations:

- `logical/`: exact logical JSON baseline and readable logical report.
- `data/`: method resource JSON and `comparison.json`.
- `inputs/`: `manifest.json`, `phases.json` and generated `components/*.qasm`.
- `runs/<run-id>/`: compiler evidence, LLI, layouts and annealing history.

All visible reports identify the same run. Failed or unselected methods have no
numeric score from an earlier run. Small reports and reproducibility inputs are
tracked; large component QASM and run evidence remain local and ignored by Git.
The mini-circuit workflow writes to `outputs/mini_circuit/` separately.

One common geometry and logical-qubit mapping is used for all components within
each method. The full resident register includes the phase flag even during W.
Time, state requests and space-time cost are multiplied; physical qubits are
allocated once. These are component-sum projections: continuous component-boundary
orientation/supply state, resets, factory carryover and transition costs are not
simulated, so projected runtime is not a fully scheduled-circuit upper bound.

Native compact can fail local Bell-CNOT decomposition because of short routes.
The default workflow records that failure and retries with routing spacing 2/3,
without changing the compact factory region. The report explicitly identifies a
spaced compact result. `--compact-spacing 1` requires the native grid and reports
failure if it cannot compile. Annealing optimizes one W followed by the largest-T
projector, then remeasures all distinct projectors on that one winning layout.

Each method independently chooses its minimum odd code distance meeting the same
layout-failure **proxy** budget (`--target-failure-probability 0.01` by default).
Automatic selection is now the default; `--auto-code-distance` remains an
explicit alias. The search starts at 3; `--code-distance` can raise that lower
bound. The table shows the selected distance, budget, achieved failure proxy and
whether the budget is met. Qubits, runtime and space-time costs use each method's
own distance. Use `--fixed-code-distance --code-distance 9` for an explicit fixed
distance comparison. Selection does not certify a complete hardware error budget.
Magic-state supplies remain abstract; physical factories require explicit
throughput and footprint assumptions. See [the architecture workflow](tools/lattice_surgery/README.md#full-inverse-component-architecture)
for controls, replay data and model limits.


### Full 3D inverse at n=20

The full 3D example uses the same phase sequence and synthesis budget as 2D,
with 66 resident logical qubits. Its logical estimate includes all four selector
rotations inside every W. Read
[the 3D architecture comparison](outputs/laplacian_inversion/3d_n20/comparison_report.txt)
and [the case README](outputs/laplacian_inversion/3d_n20/README.md).

To regenerate its logical baseline and architecture reports independently of 2D:

```bash
PYTHONPATH=src .venv/bin/python -m laplacian_qsvt.laplacian.laplacian_inversion_encode \
  --dims 3 --verify-n 1 --target-n 20 --synthesis-tol 1e-8 \
  --phases-json outputs/laplacian_inversion/3d_n20/inputs/phases.json \
  --save-prefix inversion_3d_n20 \
  --output-circuits-dir outputs/laplacian_inversion/3d_n20/inputs/verification \
  --output-block-metrics outputs/laplacian_inversion/3d_n20/logical/verification_metrics.json \
  --output-resource-metrics outputs/laplacian_inversion/3d_n20/logical/inversion_resource_metrics.json \
  --output-resource-report outputs/laplacian_inversion/3d_n20/logical/inversion_resource_report.txt

PYTHONPATH=src .venv/bin/python tools/lattice_surgery/run_inversion_architecture.py \
  --logical-metrics outputs/laplacian_inversion/3d_n20/logical/inversion_resource_metrics.json \
  --phases-json outputs/laplacian_inversion/3d_n20/inputs/phases.json \
  --budget 32 --seed 61 --target-failure-probability 0.01
```

The saved phase sequence still has kappa=2.5. These costs cover the complete
selected QSP sequence on its declared inversion domain; the logical report
records the 3D spectral gap and kappa needed for all nonzero modes.

## Spectral-gap and kappa warning

The periodic Laplacian has a zero mode and cannot be inverted on the complete
space. The inversion polynomial targets only eigenvalues satisfying
`|lambda| >= 1/kappa`.

The program calculates the smallest nonzero eigenvalue magnitude for the
target grid. If the supplied `kappa` is too small to cover the complete
nonzero spectrum, it prints a warning. Resource counts remain valid, but the
polynomial then approximates inversion only on its declared validity domain.

## Component commands

Generate phases for a generic one-qubit target function:

```bash
qsp-phases --degree 12 "cos(x)"
qsp-phases --degree 20 "exp(-x**2)"
```

Estimate only the 1D block encoding:

```bash
laplacian-1d-resource \
  --verify-n 3 \
  --target-n 20 \
  --mcx-workspace 2
```

Estimate one 2D or 3D block-encoding unit:

```bash
laplacian-nd-resource --dims 2 --verify-n 2 --target-n 6
laplacian-nd-resource --dims 3 --verify-n 1 --target-n 2 --prep-tol 1e-8
```

Estimate one qubitization walk:

```bash
laplacian-w-resource --dims 2 --verify-n 2 --target-n 4
laplacian-w-resource --dims 3 --verify-n 1 --target-n 2 --prep-tol 1e-8
```

Each walk command saves both JSON metrics and a readable text resource report:

- `outputs/metrics/laplacian_2d_w_resource_metrics.json` and
  `outputs/metrics/laplacian_2d_w_resource_report.txt` for 2D.
- `outputs/metrics/laplacian_3d_w_resource_metrics.json` and
  `outputs/metrics/laplacian_3d_w_resource_report.txt` for 3D.

The text report includes logical qubits, Clifford/T counts, depths, gate
breakdowns, synthesis details and small-size verification results.
`--output-report` overrides the text-report path; by default it follows
`--output-metrics` in the same directory. Files under `outputs/circuits/` are
circuit drawings. These walk reports count logical resources before physical
layout and QEC mapping.

Run the small `(I + X)/2` reference block-encoding demonstration:

```bash
ix2-block-encoding-demo
```

Run the generic one-qubit error-analysis utility:

```bash
qsp-error-analysis --epsilon-target 1e-6
```

Use `COMMAND --help` to view every available option. Output paths can be
overridden with the corresponding `--output-*` options.

## Output directories

- `outputs/phases/`: QSP phases and polynomial metadata.
- `outputs/metrics/`: correctness checks and resource reports.
- `outputs/circuits/`: ASCII Qiskit circuit drawings.
- `outputs/plots/`: polynomial and QSP-response plots.

Commands write to these repository-relative directories by default, even when
invoked from another working directory.

## Source layout

- `src/laplacian_qsvt/phases/`
  - Generic target-function and matrix-inversion phase generation.
- `src/laplacian_qsvt/synthesis/`
  - Reproducible pyLIQTR-backed `Ry` and `Rz` Clifford+T synthesis.
- `src/laplacian_qsvt/laplacian/`
  - 1D and multidimensional block encodings, walk construction, and full
    inversion resources.
- `src/laplacian_qsvt/demos/`
  - Small reference block-encoding circuits and matrix-evaluation helpers.
- `src/laplacian_qsvt/analysis/`
  - Generic error-combination utility.
- `tests/`
  - Unit and regression tests.

## Optional lattice-surgery compiler smoke test

The harness also supports circuit-specific **simulated-annealing floorplanning**
using the pinned `surface-code-floorplanner` geometry adapter. From this root:

```bash
.venv/bin/python tools/lattice_surgery/setup_floorplanner.py
.venv/bin/python tools/lattice_surgery/run_layout_comparison.py
```

The comparison compiles the same `tools/lattice_surgery/input_circuit.qasm`
under compact, EDPC and annealed layouts using one pinned compiler and matched
settings. It writes separate LLI/resource reports and `outputs/mini_circuit/comparison_report.txt`.
The optimizer and native dependencies stay in a sibling checkout and separate
Python environment. This setup does not require the legacy Conda compiler.
See [the integration guide](tools/lattice_surgery/README.md#simulated-annealing-layout-comparison)
for input/export options, search controls, replay and limitations.

An isolated integration harness is available under
`tools/lattice_surgery/`. It compiles a small, manually written OpenQASM 2
circuit using the optimized C++ `liblsqecc` direct Clifford+T slicer. The
legacy Python compiler remains available only as an explicit fallback.

Neither compiler is a dependency of `laplacian-qsvt`; both remain external to
the Laplacian resource-estimation package.

Set up and run the smoke test with:

```bash
bash tools/lattice_surgery/setup_compiler.sh

.venv/bin/python \
  tools/lattice_surgery/run_lattice_surgery_compiler.py
```

Write the circuit in `tools/lattice_surgery/input_circuit.qasm`. See
`tools/lattice_surgery/README.md` for syntax, environment overrides, generated
files, and current scope limitations. Layout-independent and scheduled lattice
instructions are written to `outputs/mini_circuit/artifacts/EDPC.lli` and
`outputs/mini_circuit/artifacts/EDPC_sliced.lli`.

Estimate schedule-aware timing and physical space with:

```bash
.venv/bin/python tools/lattice_surgery/estimate_sliced_resources.py \
  --code-distance 7 \
  --cycle-time-ns 1000
```

## Testing

Run the complete test suite:

```bash
python -m unittest discover -s tests -v
```

The tests check:

- Explicit `Ry` and `Rz` Clifford+T synthesis accuracy.
- Absence of arbitrary rotations after 3D selector synthesis.
- Block-encoding and walk signal-block correctness.
- QSP projector rotations against dense reference matrices.
- Complete 2D and 3D resource-accounting composition.

The pyLIQTR dependency may emit `gmpy2.local_context` deprecation warnings;
these warnings come from the upstream dependency and do not indicate test
failures.
