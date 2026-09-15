# Repository scope (SoftwareX)

What lives **inside** this repository vs what users supply externally.

**Public GitHub:** [https://github.com/Ja5ya/mof_softwarex](https://github.com/Ja5ya/mof_softwarex)

## All executable code is in this repo

| Component | Path | Layer |
|-----------|------|-------|
| MOF core | `src/mof/`, `src/mof_analysis.py` | 3 — bootstrap, objectives, frontier |
| ECG training | `src/train_multi_f_array.py`, `src/models/` | 2 |
| ECG XAI | `src/run_xai.py` | 2 |
| Manifest/NPY builder | `src/build_datasets.py` | 1 (optional, `data.mode: build`) |
| MIMIC / ecgpsych builders | `examples/dataset_builders/` | 1 (optional) |
| Slurm drivers | `slurm/` | HPC submission |
| Tests | `tests/` | CI / local validation |

**No Python modules are imported from** external sibling repositories at runtime.
`MOF_MODELS_PARENT` defaults to `src/` inside this repo.

## External inputs (data only, not code)

| Input | Typical source | Config / env |
|-------|----------------|--------------|
| MIMIC WFDB waveforms | PhysioNet MIMIC-IV-ECG | `MOF_WFDB_ROOT` |
| Machine measurements CSV | Same download | `MOF_MEASUREMENTS_CSV` |
| ICD labels CSV | MIMIC-IV-ECG-Ext-ICD | `MOF_LABELS_CSV` |
| Prebuilt pickles | Output of a dataset builder | `MOF_PREBUILT_ROOT` |
| Romania `.npy` + manifest | Private cohort (not redistributable) | `configs/romania.yaml` |
| Psychiatry-ECG beats | Tasci et al. / user download | `configs/ecgpsych.yaml` |

## Writable outputs (runtime artefacts)

Written to `MOF_DATA_ROOT` / `output_root` (outside the repo by default):

```
hyper_sweep/          # training sweep
xai_results/          # explainability CSVs
analysis/             # MOF joined tables
```

## Local development paths

Do **not** commit machine-specific env files. Use the templates:

- `configs/env.mimic.example`
- `configs/env.ecgpsych.example`
- `configs/env.romania.example`
- `configs/env.mimic_build.example`

## SoftwareX checklist

- [x] All pipeline source under this repository
- [x] Dataset builders under `examples/dataset_builders/`
- [x] Custom-study quickstart: `docs/QUICKSTART_CUSTOM_STUDY.md`
- [x] No runtime dependency on external code repositories
- [x] Local env files excluded; only `configs/env.*.example` shipped
- [x] Absolute `/home/...` paths removed from published configs
- [x] Public GitHub: https://github.com/Ja5ya/mof_softwarex
- [ ] GitHub Release tag `v0.2.0` (for SoftwareX S2)
