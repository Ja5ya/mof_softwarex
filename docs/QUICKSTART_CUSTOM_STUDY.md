# Quick start: custom dataset and custom objectives

This guide walks through using the Multi-Objective Framework (MOF) with **your own
dataset** and **your own definitions** of \(f_P\), \(f_S\), and \(f_E\). It mirrors
the three-layer architecture figure (`scripts/make_figure.py` →
`figure1_pipeline_architecture.pdf`).

---

## What is MOF?

When you train many model configurations in a hyperparameter sweep, picking the
single best accuracy often hides problems: unstable predictions, poor
explainability, or failure on clinically important metrics.

**MOF** scores every configuration on three scalar objectives and then runs a
**fixed constrained-Lagrangian analysis** to answer:

- Which configurations are *feasible* at your chosen stability and explainability thresholds?
- What is the best achievable performance on that feasible set?
- What is the *shadow price* of tightening each constraint?

| Objective | Symbol | What it measures (default ECG) |
|-----------|--------|-------------------------------|
| **Performance** | \(f_P\) | Bootstrap classification metrics (acc, precision, recall, specificity) |
| **Stability** | \(f_S\) | Bootstrap variance — penalises configs whose metrics jump under resampling |
| **Explainability** | \(f_E\) | XAI trustworthiness components (continuity, compactness, contrastivity) |

**You define** how \(f_P\), \(f_S\), \(f_E\) are computed (YAML, pandas, or custom CSVs).
**MOF fixes** the Lagrangian frontier and shadow-price machinery in `mof_analysis.py`.

### What you get at the end

| Output | Where | Purpose |
|--------|-------|---------|
| `analysis/three_objective_dataset.csv` | `MOF_DATA_ROOT` | Joined \(f_P, f_S, f_E\) per sweep config |
| Feasibility frontier plot | `notebooks/02_feasibility_frontier.ipynb` | Trade-off between stability and explainability thresholds |
| Shadow prices | same notebook | Cost of tightening \(f_S\) or \(f_E\) constraints in \(f_P\) |
| Recommended config | same notebook | Best \(f_P\) on the feasible set for your task |

### Colour key (architecture figure)

| Colour | Meaning |
|--------|---------|
| **Teal** | Fixed MOF core — Lagrangian frontier, shadow prices (`mof_analysis.py`) |
| **Amber** | User-configurable — objective formulas (`objectives:` in YAML) |
| **Gray** | Swappable — dataset builder and ECG training/XAI (optional) |

---

## 0. Install and clone

```bash
git clone <your-repo-url>
cd multi_objective_framework
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Layer 1 — Dataset

### What you need

A run directory with pickle files per [DATA_CONTRACT.md](DATA_CONTRACT.md):

```sh
<dataset_root>/
  shared/recs_normal.pkl
  datasets/<task>_vs_normal/
    recs_psych.pkl
```

Each record: `{"signal": ndarray, "label": 0|1, ...}`.

### Configure

```bash
cp configs/dataset.template.yaml configs/my_study.yaml
```

Edit paths, task codes, and ECG geometry:

```yaml
name: my_study
output_root: /path/to/writable/run

data:
  mode: prebuilt
  prebuilt_root: /path/to/dataset_root

diagnoses:
  codes: [MY_TASK]

ecg:
  n_leads: 12
  signal_len: 5000
```

Set environment variables:

```bash
export MOF_CONFIG=configs/my_study.yaml
export MOF_DATA_ROOT=/path/to/writable/run
export MOF_PREBUILT_ROOT=/path/to/dataset_root
```

__Build mode alternative:__ if you have manifest + `.npy` files instead of pickles,
set `data.mode: build` and run `python src/build_datasets.py --config $MOF_CONFIG`.
See `configs/romania.yaml` for an example.

__Reference builders:__ `examples/dataset_builders/` (MIMIC, ecgpsych).

---

## Layer 2 — Modality analysis (ECG reference)

This layer is **optional**. Skip it entirely if you supply artefact CSVs directly
([Level 3 objectives](#level-3--supply-your-own-artefact-csvs) below).

### Stage 2 — Train hyperparameter sweep

```bash
# Smoke test: one sweep task
PYTHONPATH=src python src/train_multi_f_array.py --config $MOF_CONFIG --task-id 0

# Full sweep on Slurm
bash slurm/submit_train.sh
```

Outputs per config: `hyper_sweep/<task>/<arch>/lr*_dr*/test_predictions.npz`,
`metrics.json`.

### Stage 3 — Explainability (f_E components)

```bash
PYTHONPATH=src python src/run_xai.py --config $MOF_CONFIG --code MY_TASK --hyper_sweep_fe
PYTHONPATH=src python src/mof_analysis.py --config $MOF_CONFIG merge-fe
```

Outputs: `xai_results/hyper_sweep_fe/sweep_fe_<task>.csv` with component columns
(`continuity_valid`, `compactness_valid`, `contrastivity_valid`, …).

---

## Layer 3 — MOF core

### Stage 4 — Bootstrap (f_P, f_S inputs)

```bash
PYTHONPATH=src python src/mof_analysis.py --config $MOF_CONFIG bootstrap
```

Outputs:

| File | Contents |
|------|----------|
| `hyper_sweep/bootstrap_results.csv` | Raw bootstrap mean/std per metric |
| `hyper_sweep/bootstrap_results_with_scores.csv` | + `f_P`, `f_S` from YAML |

Default bootstrap metrics: `acc`, `precision`, `recall`, `specificity`. Custom
metrics require [Level 3](#level-3--supply-your-own-artefact-csvs).

### Stage 5 — Feasibility frontier

```bash
jupyter notebook notebooks/02_feasibility_frontier.ipynb
```

The notebook calls `load_three_objective_dataset()`, which joins bootstrap + XAI,
applies your YAML objectives, and writes `analysis/three_objective_dataset.csv`.

Verify pipeline status:

```bash
PYTHONPATH=src python src/mof_analysis.py --config $MOF_CONFIG status
```

---

## Custom objectives — three extension levels

### Level 1 — Edit YAML (no code)

Add or edit the `objectives:` and `analysis:` blocks in your study YAML
(`configs/dataset.template.yaml` has a full example):

```yaml
objectives:
  f_P:
    metrics: [test_recall_mean, test_specificity_mean]
    weights: [0.7, 0.3]
    normalize: per_task_minmax
  f_S:
    method: inv_std          # inv_std | cv | recip_clipped
  f_E:
    components: [continuity_valid, compactness_valid, contrastivity_valid]
    weights: [0.5, 0.25, 0.25]

analysis:
  task_column: task_id       # optional alias for psych_code
  recall_min: 0.30           # set null to disable
  recall_metric: test_recall_mean
```

__Important:__ when XAI component columns are present, `f_E` is __recomputed__ from
YAML weights via `apply_fe_scores()` — the equal 1/3 value written by `run_xai.py`
is ignored.

Re-open the frontier notebook (or re-run `load_three_objective_dataset()`) after
changing objectives. Re-run bootstrap only if f_P/f_S _input columns_ changed.

| Objective | YAML keys | Input columns (default ECG) |
|-----------|-----------|----------------------------|
| \(f_P\) | `objectives.f_P.metrics`, `.weights`, `.normalize` | `test_*_mean` from bootstrap |
| \(f_S\) | `objectives.f_S.method`, `.metrics` | `test_*_std` from bootstrap |
| \(f_E\) | `objectives.f_E.components`, `.weights` | XAI component columns |

---

### Level 2 — Post-process CSV (any formula)

For non-linear or domain-specific formulas, use the joined table:

```python
import pandas as pd

df = pd.read_csv("analysis/objective_inputs.csv")

df["f_P"] = 0.5 * df["test_recall_mean"] + 0.5 * df["test_specificity_mean"]
df["f_S"] = 1.0 - df["test_recall_std"]
df["f_E"] = df[["continuity_valid", "contrastivity_valid"]].mean(axis=1)

df.to_csv("analysis/my_objectives.csv", index=False)
```

In the frontier notebook, load `my_objectives.csv` and pass it to
`compute_feasibility_frontier()` and `reliable_shadow_price()`. Those functions
only require scalar `f_P`, `f_S`, `f_E` columns grouped by task.

---

### Level 3 — Supply your own artefact CSVs

Skip Layers 1–2. Place files under `MOF_DATA_ROOT`:

```sh
hyper_sweep/bootstrap_results.csv
xai_results/hyper_sweep_fe/sweep_fe_<task>.csv   # or sweep_fe_summary.csv
```

**Required join keys** (one row per hyperparameter configuration):

| Column | Example |
|--------|---------|
| `psych_code` or `task_id` | `MY_TASK` |
| `architecture` | `resnet1d` |
| `learning_rate` | `0.0001` |
| `dropout_rate` | `0.1` |

__Bootstrap CSV__ — include columns referenced by your `objectives.f_P` / `f_S`
config (e.g. `test_recall_mean`, `test_recall_std`, …).

__XAI CSV__ — include columns referenced by `objectives.f_E.components`, or a
precomputed `f_E` column if no components are listed.

Then score and join:

```python
import os
os.environ["MOF_CONFIG"] = "configs/my_study.yaml"
os.environ["MOF_DATA_ROOT"] = "/path/to/run/with/csvs"

from mof_analysis import load_analysis_config, load_three_objective_dataset

cfg = load_analysis_config()
df = load_three_objective_dataset(cfg)
print(df[["f_P", "f_S", "f_E"]].head())
```

Training and XAI scripts are not required at this level.

---

## End-to-end checklist

| Step | Command / file | Layer |
|------|----------------|-------|
| 1. Copy config | `cp configs/dataset.template.yaml configs/my_study.yaml` | — |
| 2. Prepare pickles | external builder or `build_datasets.py` | 1 |
| 3. Train sweep | `train_multi_f_array.py` or Slurm | 2 |
| 4. Bootstrap | `mof_analysis.py bootstrap` | 3 |
| 5. XAI + merge | `run_xai.py` + `mof_analysis.py merge-fe` | 2 |
| 6. Custom objectives | edit YAML (L1) or pandas (L2) or BYO CSV (L3) | 3 |
| 7. Frontier | `notebooks/02_feasibility_frontier.ipynb` | 3 |

---

## Worked examples in this repository

| Study | Config | Notes |
|-------|--------|-------|
| MIMIC F32.9 | `configs/mimic_f329.yaml` | Prebuilt pickles, full sweep |
| Romania smoke test | `configs/romania.yaml` | Build mode from manifest |
| Psychiatry ECG transfer | `configs/ecgpsych.yaml` | Single task SCZ |

---

## Further reading

| Document | Contents |
|----------|----------|
| [DATA_CONTRACT.md](DATA_CONTRACT.md) | Pickle and artefact schemas, config reference |
| [REPOSITORY_SCOPE.md](REPOSITORY_SCOPE.md) | What is in-repo vs user-supplied |
| [README.md](../README.md) | Install, Slurm reference run, environment variables |

Regenerate the architecture figure:

```bash
python scripts/make_figure.py
```
