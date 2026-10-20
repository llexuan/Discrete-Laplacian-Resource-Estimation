# Full Laplacian inversion resource estimates

Start with [comparison_report.txt](comparison_report.txt) for the case definition, shared assumptions, results and tradeoffs.

- [Annealed report](annealed_resource_report.txt)
- [Compact report](compact_resource_report.txt)
- [EDPC report](EDPC_resource_report.txt)
- [Logical baseline](logical/inversion_resource_report.txt)

`data/` contains JSON results. `logical/` contains the exact logical baseline; `inputs/` contains phases, the component manifest and generated component QASM. `runs/<run-id>/` preserves compiler evidence, layouts, LLI and annealing history. Large component QASM and run evidence remain local and ignored by Git.

By default, each method independently selects its minimum odd distance for the same failure budget (search starts at d=3). This report's settings are recorded in the command below.

Published run: `20261020T045955-ltmaue06`. Every visible method report is labeled with this run, including failures or unselected methods.

From the repository root, regenerate this saved baseline with:

```bash
PYTHONPATH=src .venv/bin/python tools/lattice_surgery/run_inversion_architecture.py \
  --logical-metrics outputs/laplacian_inversion/3d_n20/logical/inversion_resource_metrics.json \
  --phases-json outputs/laplacian_inversion/3d_n20/inputs/phases.json \
  --output-dir outputs/laplacian_inversion/3d_n20 --target-failure-probability 0.01 \
  --auto-code-distance --code-distance 3
```

These are component-sum projections. The reports record method configurations, factory assumptions, compiler correction handling and the declared inversion domain.
