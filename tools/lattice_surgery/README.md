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

The integration therefore uses an OpenQASM 2 file as the boundary:

1. The main `.venv` synthesizes a rotation and exports OpenQASM 2.
2. A separate `lsqecc310` Conda environment compiles that QASM.
3. Generated compiler files remain under `outputs/compiler/`.

Nothing in `src/laplacian_qsvt/` imports `lsqecc`.

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
- Installs `lsqecc` only in that environment.

Override locations if needed:

```bash
LSQECC_REPO_DIR=/path/to/lattice-surgery-compiler \
LSQECC_ENV_NAME=my-lsqecc-env \
bash tools/lattice_surgery/setup_compiler.sh
```

## Export one synthesized rotation

Use the main project environment:

```bash
source .venv/bin/activate

python tools/lattice_surgery/export_rotation_qasm.py \
  --axis rz \
  --theta 0.3 \
  --epsilon 1e-6
```

This writes:

- `outputs/compiler/synthesized_rotation.qasm`
- `outputs/compiler/synthesized_rotation.json`

The metadata records the requested angle, synthesis error, Clifford/T counts,
and dropped global phase. OpenQASM 2 cannot represent global phase, but global
phase is physically irrelevant to lattice-surgery compilation.

## Run the compiler

The main environment does not need to be deactivated when using `conda run`:

```bash
conda run -n lsqecc310 python \
  tools/lattice_surgery/run_lattice_surgery_compiler.py
```

This writes:

- `outputs/compiler/compiler_report.txt`
- `outputs/compiler/compiler_summary.json`

The runner disables state-vector simulation with `SimulatorType.NOOP`; this
tests compilation and lattice-surgery slice generation without simulating the
quantum state.

## Scope

Start with one synthesized rotation. Do not use this smoke-test path for the
complete target-size QSP circuit yet:

- The compiler is alpha software.
- Its Python pipeline and dependency versions are old.
- A fully expanded QSP circuit can contain millions of gates.

Once the single-rotation path is validated, a separate adapter can be designed
for small complete Clifford+T circuits or for the newer C++ `liblsqecc`
slicer.
