#!/usr/bin/env bash
set -euo pipefail

# Keep the legacy compiler and its old Qiskit version outside this repository.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPILER_DIR="${LSQECC_REPO_DIR:-${PROJECT_ROOT}/../lattice-surgery-compiler}"
ENV_NAME="${LSQECC_ENV_NAME:-lsqecc310}"
COMPILER_COMMIT="${LSQECC_COMMIT:-1d9d9f36c004847e478ad464085927eebc13a0fe}"

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

if ! conda run -n "${ENV_NAME}" python --version >/dev/null 2>&1; then
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -e "${COMPILER_DIR}"

cat <<EOF

Lattice-surgery compiler setup complete.

Compiler checkout : ${COMPILER_DIR}
Conda environment : ${ENV_NAME}

Export a rotation with the main project environment:
  ${PROJECT_ROOT}/.venv/bin/python \
    ${SCRIPT_DIR}/export_rotation_qasm.py --axis rz --theta 0.3 --epsilon 1e-6

Compile it with the isolated environment:
  conda run -n ${ENV_NAME} python \
    ${SCRIPT_DIR}/run_lattice_surgery_compiler.py
EOF
