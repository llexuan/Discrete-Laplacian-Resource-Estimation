# Generated outputs

For the saved full 2D inverse, start with
[the architecture comparison](laplacian_inversion/2d_n20/comparison_report.txt).

| Folder | Contents |
| --- | --- |
| [laplacian_inversion/2d_n20/](laplacian_inversion/2d_n20/README.md) | Full inverse: annealed, compact, EDPC and comparison reports; logical baseline; supporting data and inputs. |
| [mini_circuit/](mini_circuit/README.md) | Small QASM example: three layout reports, comparison, LLI and optimized layout. |
| `metrics/` | Logical estimator outputs and correctness checks, including single-W 2D/3D reports. Each architecture case keeps its own exact baseline under `logical/`. |
| `phases/` | QSP phase lists and polynomial metadata. |
| `circuits/` | ASCII circuit drawings. |
| `plots/` | QSP approximation plots. |

Readable architecture reports are at the top of each example folder. JSON lives
in `data/`; component inputs and historical runs have their own subfolders.
Large compiler/search evidence stays local and ignored by Git. Each published
inverse comparison and its three method reports identify one run.
