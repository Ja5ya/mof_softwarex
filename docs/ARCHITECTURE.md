# Architecture

This document is a short pointer. Full documentation lives in:

| Document | Contents |
|----------|----------|
| [QUICKSTART_CUSTOM_STUDY.md](QUICKSTART_CUSTOM_STUDY.md) | What MOF does, three layers, custom objectives |
| [DATA_CONTRACT.md](DATA_CONTRACT.md) | Pickle / CSV schemas, config reference, extension levels |
| [../README.md](../README.md) | Install, Slurm reference run, embedded architecture figure |

Architecture figure (repository root): [`mof_architecture.png`](../mof_architecture.png)

- **Teal** — fixed Lagrangian core  
- **Amber** — configurable \(f_P\), \(f_S\), \(f_E\)  
- **Gray** — swappable dataset / ECG use case  

Regenerate with `python scripts/make_figure.py` if you edit the figure script.
