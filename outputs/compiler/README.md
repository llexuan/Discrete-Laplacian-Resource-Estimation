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

.venv/bin/python \
  tools/lattice_surgery/run_lattice_surgery_compiler.py

.venv/bin/python \
  tools/lattice_surgery/estimate_sliced_resources.py \
  --code-distance 7 \
  --cycle-time-ns 1000
