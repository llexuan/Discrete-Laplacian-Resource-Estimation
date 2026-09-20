#!/usr/bin/env bash
set -euo pipefail

# Keep the legacy compiler and its old Qiskit version outside this repository.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPILER_DIR="${LSQECC_REPO_DIR:-${PROJECT_ROOT}/../lattice-surgery-compiler}"
ENV_NAME="${LSQECC_ENV_NAME:-lsqecc310}"
COMPILER_COMMIT="${LSQECC_COMMIT:-1d9d9f36c004847e478ad464085927eebc13a0fe}"
TWEEDLEDUM_COMMIT="${TWEEDLEDUM_COMMIT:-3d765c357ea6b9300be8eb77ee3f9d7e4102ed8c}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to create the isolated Python 3.10 environment." >&2
  exit 1
fi

if [[ ! -d "${COMPILER_DIR}/.git" ]]; then
  git clone --recursive --branch dev \
    https://github.com/latticesurgery-com/lattice-surgery-compiler.git \
    "${COMPILER_DIR}"
fi

git -C "${COMPILER_DIR}" checkout "${COMPILER_COMMIT}"
git -C "${COMPILER_DIR}" submodule update --init --recursive

CONDA_BASE="$(conda info --base)"
ENV_PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"

if [[ ! -x "${ENV_PYTHON}" ]]; then
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

"${ENV_PYTHON}" -m pip install --upgrade pip wheel
"${ENV_PYTHON}" -m pip install \
  "setuptools<70" \
  "Cython<3" \
  "numpy<1.24"

# Install Terra without the obsolete Qiskit meta-package (and its unnecessary
# Aer dependency), then install the runtime dependencies explicitly. The
# maintained Tweedledum source fixes the broken 1.1.1 macOS ARM packaging.
"${ENV_PYTHON}" -m pip install --no-deps "qiskit-terra==0.19.2"
"${ENV_PYTHON}" -m pip install \
  "retworkx>=0.10.1" \
  ply \
  psutil \
  scipy \
  sympy \
  dill \
  python-constraint \
  python-dateutil \
  stevedore \
  "symengine>=0.8,<0.9" \
  "pyzx>=0.6.4,<0.7.1" \
  "igraph>=0.9.8,<0.9.10" \
  "tweedledum @ git+https://github.com/boschmitt/tweedledum.git@${TWEEDLEDUM_COMMIT}"

"${ENV_PYTHON}" -m pip install --no-deps -e "${COMPILER_DIR}"
"${ENV_PYTHON}" -c \
  'import qiskit, tweedledum, lsqecc; print("Qiskit:", qiskit.__version__); print("lsqecc ready")'

cat <<EOF

Lattice-surgery compiler setup complete.

Compiler checkout : ${COMPILER_DIR}
Conda environment : ${ENV_NAME}
Compiler Python   : ${ENV_PYTHON}

Write an OpenQASM 2 circuit in:
  ${SCRIPT_DIR}/input_circuit.qasm

Compile it with the isolated environment:
  ${ENV_PYTHON} ${SCRIPT_DIR}/run_lattice_surgery_compiler.py
EOF
