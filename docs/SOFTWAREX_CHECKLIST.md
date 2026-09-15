# SoftwareX GitHub readiness checklist

SoftwareX requires a **public GitHub repository** with a documented `README.md`
and a license file. This package is prepared as the uploadable software tree.

## Mandatory (SoftwareX / Guide for Authors)

| Requirement | Status in this package |
|-------------|------------------------|
| Public GitHub repository | **You must create** — replace `YOUR_GITHUB_ORG` in `CITATION.cff` / README |
| `README.md` | Present |
| License file (`LICENSE`) | Present (MIT) |
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
| Architecture figure | `mof_architecture_diagram.png` |

## Intentionally excluded from this upload tree

| Excluded | Why |
|----------|-----|
| `artifacts/`, `slurm_logs/` | Runtime outputs / private run data |
| `configs/env.mimic`, `env.romania`, `env.ecgpsych` | Machine-local absolute paths (use `*.example`) |
| `main.tex`, `ref.bib`, paper PDFs | Manuscript — submit separately to SoftwareX |
| Prebuilt pickles / MIMIC WFDB | User-supplied data (PhysioNet / private cohorts) |

## Before first `git push`

1. Create empty GitHub repo (e.g. `multi_objective_framework`).
2. Replace `YOUR_GITHUB_ORG` in `CITATION.cff` and `README.md`.
3. Copy this folder to a writable location and initialise git:

```bash
cp -a /tmp/multi_objective_framework_softwarex ~/multi_objective_framework
cd ~/multi_objective_framework
git init
git add .
git commit -m "Initial public release of MOF for SoftwareX"
git branch -M main
git remote add origin git@github.com:YOUR_GITHUB_ORG/multi_objective_framework.git
git push -u origin main
```

4. Tag a release matching metadata (`v0.2.0`).
5. Point SoftwareX C2/C7/S2 URLs at that repo.

## Quick validation (no data required)

```bash
cd /path/to/multi_objective_framework
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m unittest tests.test_objectives tests.test_mof_analysis tests.test_pipeline_config -v
```
