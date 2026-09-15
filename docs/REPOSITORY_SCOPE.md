# Repository scope (SoftwareX)

What lives **inside** this repository vs what users supply externally.

## All executable code is in this repo

| Component | Path | Runs when |
|-----------|------|-----------|
| MOF core | `src/mof/`, `src/mof_analysis.py` | bootstrap, objectives, frontier |
| ECG training | `src/train_multi_f_array.py`, `src/models/` | Layer 2 |
| ECG XAI | `src/run_xai.py` | Layer 3 (ECG) |
| Manifest/NPY builder | `src/build_datasets.py` | Romania / custom cohorts |
| MIMIC WFDB builder | `examples/dataset_builders/build_mimic_datasets.py` | MIMIC Layer 1 (optional) |
| Slurm drivers | `slurm/` | HPC submission |
| Tests | `tests/` | CI / local validation |

**No Python modules are imported from** `jo_trial6_data_exploration`,
`multi_objective_framework` (parent), or any other external repository at
runtime. `MOF_MODELS_PARENT` defaults to `src/` inside this repo.

## External inputs (data only, not code)

| Input | Typical source | Config / env |
|-------|----------------|--------------|
| MIMIC WFDB waveforms | PhysioNet MIMIC-IV-ECG | `MOF_WFDB_ROOT` |
| Machine measurements CSV | Same download | `MOF_MEASUREMENTS_CSV` |
| ICD labels CSV | MIMIC-IV-ECG-Ext-ICD | `MOF_LABELS_CSV` |
| Prebuilt pickles | Output of `build_mimic_datasets.py` | `MOF_PREBUILT_ROOT` |
| Romania `.npy` + manifest | Private cohort (not redistributable) | `configs/romania.yaml` |

## Writable outputs (runtime artefacts)

Written to `MOF_DATA_ROOT` / `output_root` (outside repo by default):

```
hyper_sweep/          # training sweep
xai_results/          # explainability CSVs
analysis/             # MOF joined tables
```

## Local development paths

Files like `configs/env.mimic` may contain machine-specific absolute paths
(e.g. your existing `jo_trial6` prebuilt run). For publication, use the
`.example` templates with placeholder paths.

## SoftwareX checklist

- [x] All pipeline source under this repository
- [x] MIMIC dataset builder vendored under `examples/dataset_builders/`
- [x] Custom-study quickstart: `docs/QUICKSTART_CUSTOM_STUDY.md`
- [x] No runtime dependency on external code repositories
- [x] Local env files excluded; only `configs/env.*.example` shipped
- [x] Absolute `/home/...` paths removed from published configs
- [ ] Replace `YOUR_GITHUB_ORG` in `CITATION.cff` / README after creating GitHub repo
