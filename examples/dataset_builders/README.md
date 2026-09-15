# Dataset builders (Layer 1)

## Psychiatry-ECG (this repo’s primary public path)

Kaggle / Tasci beats → binary **Schizophrenia vs bipolar+depression** (same source; not healthy controls).

```bash
cd /path/to/multi_objective_framework_ecgpsych
python examples/dataset_builders/build_ecgpsych_datasets.py \
  --data-root "/path/to/Psychiatry_ECG" \
  --output-root artifacts/prebuilt \
  --signal-len 384

source configs/env.ecgpsych   # MOF_PREBUILT_ROOT must match --output-root
```

Output layout:

```
<prebuilt_root>/
  shared/recs_normal.pkl                         # bipolar + depression
  datasets/icd10_SCZ_vs_normal/recs_psych.pkl    # schizophrenia
  datasets/icd10_SCZ_vs_normal/recs_normal.pkl   # symlink → shared
  datasets/build_manifest.json
```

## MIMIC (optional reference)

Builds the pickle contract from MIMIC-IV-ECG WFDB + ICD-10 labels.

```bash
cp configs/env.mimic_build.example configs/env.mimic_build
# edit paths, then:
source configs/env.mimic_build

python examples/dataset_builders/build_mimic_datasets.py \
  --output-root "${MOF_PREBUILT_ROOT}" \
  --codes F329,F419,F17200,F17210,F0390,F1010
```
