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

The complete report is:

`outputs/metrics/laplacian_inversion_resource_metrics.json`

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

An isolated integration harness is available under
`tools/lattice_surgery/`. It compiles a small, manually written OpenQASM 2
circuit using the Python lattice-surgery compiler in a separate Python 3.10
Conda environment.

The compiler is intentionally not a dependency of `laplacian-qsvt`: it
requires a legacy Qiskit version that conflicts with this project's Qiskit
2.x environment.

Set up and run the smoke test with:

```bash
bash tools/lattice_surgery/setup_compiler.sh

/opt/anaconda3/envs/lsqecc310/bin/python \
  tools/lattice_surgery/run_lattice_surgery_compiler.py
```

Write the circuit in `tools/lattice_surgery/input_circuit.qasm`. See
`tools/lattice_surgery/README.md` for syntax, environment overrides, generated
files, and current scope limitations. The complete ordered patch layouts are
written to `outputs/compiler/compiler_slices.json`.

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
