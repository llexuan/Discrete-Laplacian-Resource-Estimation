# Isolated lattice-surgery compiler smoke test

This directory tests
[`latticesurgery-com/lattice-surgery-compiler`](https://github.com/latticesurgery-com/lattice-surgery-compiler)
without adding its legacy dependencies to the main `laplacian-qsvt`
environment.

## Why it is isolated

The main project currently uses Qiskit 2.x and Python 3.12. The Python
lattice-surgery compiler pins Qiskit below 0.35 and its CI tested Python
3.7–3.10. Installing both projects into one environment would downgrade and
break the main resource-estimation package.

The integration therefore uses a standalone OpenQASM 2 file:

1. You write a small circuit in `input_circuit.qasm`.
2. A separate `lsqecc310` Conda environment compiles that QASM.
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

Override locations if needed:

```bash
LSQECC_REPO_DIR=/path/to/lattice-surgery-compiler \
LSQECC_ENV_NAME=my-lsqecc-env \
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

Use the isolated environment's interpreter directly:

```bash
/opt/anaconda3/envs/lsqecc310/bin/python \
  tools/lattice_surgery/run_lattice_surgery_compiler.py
```

This writes:

- `outputs/compiler/compiler_report.txt`
- `outputs/compiler/compiler_slices.json`
- `outputs/compiler/compiler_summary.json`

The runner disables state-vector simulation with `SimulatorType.NOOP`; this
tests compilation and lattice-surgery slice generation without simulating the
quantum state. It also disables the pinned compiler's broken optional physical
resource estimator, which receives the wrong internal object type upstream.
This does not disable circuit compilation or slice generation.

`compiler_slices.json` is the complete lattice-surgery result: an ordered
array of 2D patch layouts, one layout per compiler timestep. It can be retained
for visualization, patch scheduling inspection, and later space-time analysis.

## Scope

Start with a small hand-written Clifford+T circuit. Do not use this smoke-test
path for the complete target-size QSP circuit yet:

- The compiler is alpha software.
- Its Python pipeline and dependency versions are old.
- A fully expanded QSP circuit can contain millions of gates.

`export_rotation_qasm.py` is retained as an optional future utility, but it is
not connected to the default compiler input. A separate adapter can be
designed later for small generated Clifford+T circuits or for the newer C++
`liblsqecc` slicer.
