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

Override locations if needed:

```bash
LSQECC_REPO_DIR=/path/to/lattice-surgery-compiler \
LSQECC_ENV_NAME=my-lsqecc-env \
LIBLSQECC_REPO_DIR=/path/to/liblsqecc \
bash tools/lattice_surgery/setup_compiler.sh
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

- `outputs/compiler/compiler_report.txt`
- `outputs/compiler/compiler_slices.json`
- `outputs/compiler/compiler_summary.json`

`compiler_report.txt` combines the legacy Python compiler's Pauli-rotation and
Litinski-transformation views with the fast slicer's command and volume
statistics. Pass `--no-intermediate-report` to skip the legacy section.

By default, the runner uses the upgraded direct-compilation configuration from
the liblsqecc resource-estimation workflow:

- EDPC layout
- Wave scheduling
- Local lattice-surgery instruction decomposition
- Distillation time 1 with unstaggered factories
- Catalytic S-gate implementation disabled through `--notwists`

`compiler_slices.json` is the viewer-compatible complete lattice-surgery
result: an ordered array of 2D patch layouts, one layout per compiler
timestep. It can be uploaded to
[latticesurgery.com](https://latticesurgery.com) for visualization.

For the compact/stream layout closest to the basic viewer example:

```bash
.venv/bin/python tools/lattice_surgery/run_lattice_surgery_compiler.py \
  --preset viewer
```

The exact hosted QASM service can differ by deployed compiler revision and
server settings. `compiler_summary.json` records the local backend commit and
full command for reproducibility.

To use the legacy Python backend explicitly:

```bash
/opt/anaconda3/envs/lsqecc310/bin/python \
  tools/lattice_surgery/run_lattice_surgery_compiler.py \
  --backend python
```

That fallback disables state-vector simulation and the pinned Python
compiler's broken optional resource estimator. It does not disable slice
generation.

## Scope

Start with a small hand-written Clifford+T circuit. A fully expanded
target-size QSP circuit can contain millions of gates and should be tested only
after validating progressively larger examples.

`export_rotation_qasm.py` is retained as an optional future utility, but it is
not connected to the default compiler input.
