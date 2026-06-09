# Alzheimer's MRI Multitask Reproduction Scaffold

This project reproduces the modeling strategy of Ma et al. using public substitute datasets. HCP-YA is used for segmentation/anatomical pretraining, and OASIS-1/2 is used as an ADNI substitute for dementia classification and cognitive/severity prediction.

Original paper:

- Title: "Predicting categorical and continuous Alzheimer's disease outcomes from a single MRI scan"
- Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC13190282/
- Code: https://github.com/darenma/MultitaskCognition

## Target Pipeline

```text
Single T1 MRI scan
-> brain segmentation/anatomy representation
-> dementia classification
-> cognitive/severity prediction
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

Fine-tune/evaluate on OASIS:

```bash
python -m src.train_multitask --config configs/oasis_multitask.yaml
```

## Current Scope

This is a public-data analogue. It can demonstrate:

- T1 MRI input handling
- FSL-FAST tissue segmentation targets
- 3D UNet-style segmentation/anatomy representation
- Dementia classification from a single MRI
- Cognitive/severity prediction using OASIS variables such as MMSE or CDR

It does not claim exact reproduction of ADNI/ADAS-Cog11 results without ADNI access.

