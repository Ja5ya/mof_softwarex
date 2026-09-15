# SoftwareX GitHub readiness checklist

SoftwareX requires a **public GitHub repository** with a documented `README.md`
and a license file. This repository is that software tree.

**Public repo:** [https://github.com/Ja5ya/mof_softwarex](https://github.com/Ja5ya/mof_softwarex)

## Mandatory (SoftwareX / Guide for Authors)

| Requirement | Status |
|-------------|--------|
| Public GitHub repository | Done — [Ja5ya/mof_softwarex](https://github.com/Ja5ya/mof_softwarex) |
| `README.md` | Present |
| License file (`LICENSE` / `LICENSE.txt`) | Present (MIT) |
| Code self-contained (no private sibling repos) | Present (`src/`, `examples/`) |
| Developer documentation | `docs/QUICKSTART_CUSTOM_STUDY.md`, `docs/DATA_CONTRACT.md` |
| Reproducible install | `requirements.txt` (Python 3.10+, TensorFlow 2.15+) |

## Recommended (this MOF package)

| Item | Status |
|------|--------|
| Pluggable objectives (`f_P`, `f_S`, `f_E`) | `src/mof/objectives.py` + YAML `objectives:` / `analysis:` |
| ECG reference use case | `src/train_multi_f_array.py`, `src/run_xai.py` |
| Dataset builders (optional) | `examples/dataset_builders/` |
| Unit tests | `tests/` (data-dependent tests skip without `MOF_PREBUILT_ROOT`) |
| Slurm helpers | `slurm/` |
| Architecture figure | `mof_architecture.png` (also embedded in README) |

## Intentionally excluded from this repo

| Excluded | Why |
|----------|-----|
| `artifacts/`, `slurm_logs/` | Runtime outputs / private run data |
| `configs/env.mimic`, `env.romania`, `env.ecgpsych` | Machine-local paths (ship `*.example` only) |
| `main.tex`, `ref.bib`, paper PDFs | Manuscript — submit separately to SoftwareX |
| Prebuilt pickles / MIMIC WFDB | User-supplied data (PhysioNet / private cohorts) |

## Remaining author actions

1. Keep GitHub URLs in sync in `CITATION.cff` and `README.md` (already set to `Ja5ya/mof_softwarex`).
2. Tag and publish GitHub Release **`v0.2.0`** (SoftwareX S2 metadata).
3. Point manuscript C2 / C7 / S2 at this repo (see SoftwareX `main.tex`).

## Quick validation (no data required)

```bash
cd /path/to/mof_softwarex   # or clone https://github.com/Ja5ya/mof_softwarex.git
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python tests/test_objectives.py -v
PYTHONPATH=src python tests/test_mof_analysis.py -v
PYTHONPATH=src python tests/test_pipeline_config.py -v
```
