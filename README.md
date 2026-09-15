# Multi-Objective Framework (MOF)

**Multi-objective model selection under trustworthiness constraints** — with an
ECG classification reference application.

MOF scores every hyperparameter configuration in a sweep on three objectives,
then applies a **fixed Lagrangian feasibility frontier** to find models that
balance them under your constraints. You define the objectives; the frontier
math is fixed.

| Objective | Symbol | Default meaning (ECG reference) |
|-----------|--------|--------------------------------|
| Performance | \(f_P\) | Weighted combination of bootstrap classification metrics |
| Stability | \(f_S\) | Inverse bootstrap variance (higher = more stable) |
| Explainability | \(f_E\) | Weighted combination of XAI trustworthiness components |

**Output:** feasibility frontier plots, reliable shadow prices, and a recommended
model configuration per task — not just "highest AUC wins."

---

## Install

```bash
git clone https://github.com/Ja5ya/mof_softwarex.git
cd multi_objective_framework
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: Python 3.10+, TensorFlow 2.15+ (`requirements.txt`).

---

## Documentation

| Document | When to read it |
|----------|-----------------|
| **[docs/QUICKSTART_CUSTOM_STUDY.md](docs/QUICKSTART_CUSTOM_STUDY.md)** | **Start here** — new dataset, custom \(f_P/f_S/f_E\), full walkthrough |
| [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) | Pickle / CSV schemas, config reference, objective extension levels |
| [docs/REPOSITORY_SCOPE.md](docs/REPOSITORY_SCOPE.md) | What is in-repo vs user-supplied (SoftwareX) |
| [docs/SOFTWAREX_CHECKLIST.md](docs/SOFTWAREX_CHECKLIST.md) | Pre-upload GitHub / SoftwareX checklist |
| [examples/dataset_builders/](examples/dataset_builders/) | MIMIC / ecgpsych dataset builders |

Architecture figure: `mof_architecture_diagram.png` (regenerate with `python scripts/make_figure.py`).

---

## Three layers

| Layer | What | In this repo |
|-------|------|--------------|
| 1. Dataset | User-built pickles or optional builders | `configs/*.yaml` → `data.mode` |
| 2. ECG analysis | Train sweep + explainability *(swappable)* | `train_multi_f_array.py`, `run_xai.py` |
| 3. MOF core | Bootstrap + objectives + frontier *(fixed Lagrangian)* | `mof_analysis.py`, `src/mof/` |

**New study?** → [docs/QUICKSTART_CUSTOM_STUDY.md](docs/QUICKSTART_CUSTOM_STUDY.md)

---

## Pipeline (five stages)

| Stage | Script | Main output |
|-------|--------|-------------|
| 1. Dataset | `build_datasets.py` *(optional)* | Pickles under `datasets/` |
| 2. Train sweep | `train_multi_f_array.py` | `hyper_sweep/.../test_predictions.npz` |
| 3. XAI | `run_xai.py` | `xai_results/.../sweep_fe_<task>.csv` |
| 3b. Merge | `mof_analysis.py merge-fe` | `sweep_fe_summary.csv` |
| 4. Bootstrap | `mof_analysis.py bootstrap` | `bootstrap_results.csv` |
| 5. Frontier | `notebooks/02_feasibility_frontier.ipynb` | `analysis/three_objective_dataset.csv` |

All outputs live under `MOF_DATA_ROOT` / `output_root` in your YAML config.

---

## Reference run — MIMIC on Slurm

```bash
cp configs/env.mimic.example configs/env.mimic   # edit MOF_CONFIG, MOF_DATA_ROOT, MOF_PREBUILT_ROOT
source configs/env.mimic

bash slurm/submit_train.sh    # stages 2 + 4 (training + bootstrap)
# wait for completion
bash slurm/submit_xai.sh      # stages 3 + 3b (XAI + merge-fe)

PYTHONPATH=src python src/mof_analysis.py --config $MOF_CONFIG status
jupyter notebook notebooks/02_feasibility_frontier.ipynb
```

**Run order:** train → bootstrap → XAI → merge-fe → frontier. Do not submit XAI until training finishes.

**Local smoke test** (one task, no Slurm):

```bash
source configs/env.mimic
PYTHONPATH=src python src/train_multi_f_array.py --config $MOF_CONFIG --task-id 0
N_BOOTSTRAP=100 PYTHONPATH=src python src/mof_analysis.py --config $MOF_CONFIG bootstrap
PYTHONPATH=src python src/run_xai.py --config $MOF_CONFIG --code F329 --hyper_sweep_fe
PYTHONPATH=src python src/mof_analysis.py --config $MOF_CONFIG merge-fe
```

Example configs: `configs/mimic.yaml` (6-task test grid), `configs/mimic_f329.yaml` (full single-code sweep), `configs/ecgpsych.yaml` (transfer study), `configs/romania.yaml` (build-mode smoke test).

Sweep size: `n_codes × n_architectures × n_learning_rates × n_dropout_rates` — check with `PYTHONPATH=src python src/pipeline_config.py --config $MOF_CONFIG sweep-count`.

---

## Environment variables

| Variable | Meaning |
|----------|---------|
| `MOF_CONFIG` | Path to study YAML |
| `MOF_DATA_ROOT` | Writable run directory |
| `MOF_PREBUILT_ROOT` | Prebuilt pickle root (`data.mode: prebuilt`) |
| `MOF_AUTO_BOOTSTRAP` | `1` (default): chain bootstrap after `submit_train.sh` |
| `MOF_AUTO_MERGE_FE` | `1` (default): chain merge-fe after `submit_xai.sh` |
| `N_BOOTSTRAP` | Bootstrap resamples (default `1000`; use `100` for tests) |
| `XAI_MODE` | `sweep` (default) or `pareto` |

Full list: see [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md#configuration-reference).

---

## Custom objectives

Edit `objectives:` and `analysis:` in your study YAML, or post-process
`analysis/objective_inputs.csv`. Three extension levels (YAML / pandas / BYO CSVs):
[docs/DATA_CONTRACT.md#objective-extension-levels](docs/DATA_CONTRACT.md#objective-extension-levels).

Walkthrough: [docs/QUICKSTART_CUSTOM_STUDY.md](docs/QUICKSTART_CUSTOM_STUDY.md).

---

## Testing

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

---

## Related work

Extends the NeurIPS multi-objective trustworthiness framework. This package
generalises the pipeline: bring your own dataset → run ECG module → run MOF core.

## License

MIT — see [LICENSE](LICENSE).
