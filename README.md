# Single-MRI Alzheimer's Multitask Pipeline

This repository implements a public-data reproduction scaffold for the Ma et al. UCSF modeling strategy:

```text
Single T1 MRI scan
-> anatomical representation / tissue segmentation
-> CDR-based dementia-status classification
-> cognitive prediction
```

The current implementation uses OASIS-1 as an ADNI substitute. Each subject contributes one T1-weighted MRI scan, FSL-FAST provides gray matter / white matter / CSF tissue labels, and a 3D multitask network learns one shared anatomical representation with three outputs: segmentation, dementia probability, and MMSE prediction.

Original paper:

- Title: "Predicting categorical and continuous Alzheimer's disease outcomes from a single MRI scan"
- Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC13190282/
- Code: https://github.com/darenma/MultitaskCognition

## Target Pipeline

```text
Input:
  Single T1-weighted MRI volume

Shared representation:
  3D encoder learns anatomical MRI features

Outputs:
  1. Anatomical representation / segmentation
     - FAST-style background / CSF / gray matter / white matter labels

  2. Dementia classification
     - OASIS label: CDR 0 = control, CDR > 0 = impaired/dementia

  3. Cognitive prediction
     - MMSE regression on the 0-30 clinical scale
```

## Dataset Mapping

| Paper dataset | Role in paper | Public-data analogue here |
| --- | --- | --- |
| ADNI | Main AD/CN/MCI training, ADAS-Cog11, longitudinal outcomes | OASIS-1 or OASIS-2 |
| HCP-YA | Segmentation/anatomy pretraining | HCP-YA |
| DLBS | External out-of-sample testing | Optional later external validation |

## Project Layout

```text
data/
  raw/
    hcp_ya/
    oasis/
  processed/
    hcp_ya/
    oasis/
  manifests/
docs/
models/
notebooks/
outputs/
src/
vendor/
```

## Current OASIS-1 Experiment

The main public-data run is configured in `configs/oasis1_200_multitask.yaml`. It uses the maximum balanced OASIS-1 subset available in this setup:

```text
174 subjects total
121 train / 26 validation / 27 test
64 x 64 x 64 downsampled MRI volumes
FSL-FAST tissue segmentation targets
CDR 0 vs CDR > 0 dementia label
MMSE cognitive regression target
```

Current test results:

| Output | Metric | Test result | Interpretation |
| --- | ---: | ---: | --- |
| Anatomical segmentation | Dice | 0.884 | The model learned tissue/anatomy segmentation well. |
| Dementia classification | Balanced accuracy | 0.500 | Classification performance is limited in the current OASIS-1 run. |
| Dementia classification | AUC | 0.758 | Probability ranking shows preliminary signal, but thresholded classification remains limited. |
| Cognitive prediction | MMSE MAE | 2.62 | Baseline cognitive prediction result on the current split. |

These results demonstrate the end-to-end technical pipeline. They should not be interpreted as final clinical performance.

Notebook summary:

- `notebooks/oasis1_200_evaluation.ipynb`

## Quick Start

Create the environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check that the local scaffold imports:

```bash
python -m src.check_setup
```

After downloading data, create manifests:

```bash
python -m src.build_oasis_manifest --clinical-csv data/raw/oasis/clinical.csv --mri-root data/raw/oasis --out data/manifests/oasis_manifest.csv
python -m src.build_hcp_manifest --mri-root data/raw/hcp_ya --out data/manifests/hcp_manifest.csv
```

Generate FSL-FAST segmentations:

```bash
python -m src.run_fast_segmentations --manifest data/manifests/hcp_manifest.csv --out-root data/processed/hcp_ya/fast
python -m src.run_fast_segmentations --manifest data/manifests/oasis_manifest.csv --out-root data/processed/oasis/fast
```

Train segmentation pretraining on HCP-YA:

```bash
python -m src.train_multitask --config configs/hcp_segmentation_pretrain.yaml
```

Train the three-output OASIS pipeline:

```bash
python -m src.train_multitask --config configs/oasis1_200_multitask.yaml
```

Train the same pipeline with age/sex covariates:

```bash
python -m src.train_multitask --config configs/oasis1_200_covariates_multitask.yaml
```

Train with the optional MedicalNet-style backbone:

```bash
git clone https://github.com/Tencent/MedicalNet.git vendor/MedicalNet
python -m src.train_multitask --config configs/oasis1_medicalnet_multitask.yaml
```

Evaluate the three outputs:

```bash
python -m src.evaluate_oasis1 \
  --config configs/oasis1_200_multitask.yaml \
  --checkpoint models/oasis1_200_multitask.ckpt \
  --output-dir outputs/oasis1_200_eval
```

## Current Scope

This is a public-data analogue. It demonstrates:

- T1 MRI input handling
- FSL-FAST tissue segmentation targets
- 3D UNet-style anatomical representation / segmentation
- Dementia classification from a single MRI using CDR-derived labels
- Cognitive prediction from a single MRI using MMSE

It does not claim exact reproduction of ADNI/ADAS-Cog11 results without ADNI access.

See `docs/project_report.md` for a concise project summary, current results, limitations, and next steps.

## UCSF-Like Extensions

The repo now includes implementation hooks for the next UCSF-like upgrades:

| Upgrade | File/config | Status |
| --- | --- | --- |
| Co-registration | `src/coregister_mri.py` | Ready when a template MRI is available |
| MedicalNet backbone | `configs/oasis1_medicalnet_multitask.yaml` | Ready when `vendor/MedicalNet` is cloned |
| Richer CDR labels | `src/prepare_oasis1.py` | Supported for newly generated manifests |
| Age/sex covariates | `configs/oasis1_200_covariates_multitask.yaml` | Ready for next training run |
| Longitudinal/future prediction | `src/build_longitudinal_manifest.py` | Requires OASIS-2, ADNI, or lab follow-up data |
