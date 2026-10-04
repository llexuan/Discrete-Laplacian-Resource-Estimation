# Lattice-surgery compiler outputs

This directory is reserved for the isolated compiler smoke test in
`tools/lattice_surgery/`.

The tracked outputs are:

- `compiler.lli`: layout-independent lattice-surgery instructions.
- `compiler_sliced.lli`: instructions grouped by scheduled timestep.
- `compiler_report.txt`: Pauli/Litinski transformations and exact commands.
- `slice_resource_estimate.json`: schedule-aware machine-readable resources.
- `slice_resource_report.txt`: readable timing and space report.

Each compiler run overwrites these files. The editable QASM input is tracked at
`tools/lattice_surgery/input_circuit.qasm`.

EDPC style compile: 
.venv/bin/python \
  tools/lattice_surgery/run_lattice_surgery_compiler.py

.venv/bin/python \
  tools/lattice_surgery/estimate_sliced_resources.py \
  --code-distance 7 \
  --cycle-time-ns 1000

Compact style compile: 
.venv/bin/python tools/lattice_surgery/run_lattice_surgery_compiler.py \
  --preset viewer \
  --lli outputs/compiler/compact.lli \
  --sliced-lli outputs/compiler/compact_sliced.lli \
  --report outputs/compiler/compact_compiler_report.txt

.venv/bin/python tools/lattice_surgery/estimate_sliced_resources.py \
  --preset viewer \
  --code-distance 9 \
  --cycle-time-ns 1000 \
  --output-json outputs/compiler/compact_resource_estimate.json \
  --output-report outputs/compiler/compact_resource_report.txt