# Lattice-surgery compiler outputs

This directory is reserved for the isolated compiler smoke test in
`tools/lattice_surgery/`.

The report, summary, and complete patch-layout slice file are tracked by Git.
Each compiler run overwrites these files rather than appending to them, so they
contain only the most recent circuit result. The editable QASM input is tracked
separately at `tools/lattice_surgery/input_circuit.qasm`. The default runner
uses the pinned `liblsqecc` direct EDPC/wave backend; its revision and complete
command are recorded in `compiler_summary.json`.
