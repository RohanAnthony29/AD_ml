# Step-by-Step Procedure: Single MRI to Anatomy, Dementia, and Cognition

## 1. Define Scope

This project reproduces the modeling strategy of Ma et al. using public substitute datasets. The target workflow is:

```text
Single T1 MRI
-> anatomical representation / segmentation
-> dementia classification
-> cognitive prediction
```

HCP-YA can be used for segmentation/anatomical pretraining. OASIS-1/2 is used as an ADNI substitute for dementia classification and MMSE/CDR prediction.

## 2. Get Code

The original repositories are stored under `vendor/`:

```bash
git clone https://github.com/darenma/MultitaskCognition.git vendor/MultitaskCognition
git clone https://github.com/Tencent/MedicalNet.git vendor/MedicalNet
```

## 3. Create Environment

Use Python 3.10:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

External tools:

- FSL / FAST
- dcm2niix, only if the MRI files arrive as DICOM

## 4. Download Datasets

HCP-YA:

- Source: https://www.humanconnectome.org/study/hcp-young-adult
- Use T1-weighted structural MRI.

OASIS-1 or OASIS-2:

- Source: https://www.oasis-brains.org/
- Use T1-weighted MRI and clinical CSV.
- OASIS-1 is best for a fast cross-sectional demo.
- OASIS-2 is better if longitudinal/severity change is needed.

## 5. Organize Data

Place files under:

```text
data/raw/hcp_ya/
data/raw/oasis/
```

## 6. Build Data Manifests

OASIS manifest columns:

```text
subject_id,session_id,mri_path,age,sex,mmse,cdr,label,cdr_binary_label,cdr_three_class_label,cdr_score
```

HCP-YA manifest columns:

```text
subject_id,mri_path,age,sex
```

Label rule for OASIS:

```text
label = 0 if CDR == 0
label = 1 if CDR > 0
```

Richer label columns now supported:

```text
cdr_binary_label       CDR 0 vs CDR > 0
cdr_three_class_label  CDR 0 vs CDR 0.5 vs CDR 1+
cdr_score              raw CDR severity score
mmse                   continuous cognition target
```

## 7. Preprocess MRI

For every T1 MRI:

1. Convert to NIfTI if needed.
2. Reorient/standardize.
3. Co-register to a common template when a template is available.
4. Skull-strip if required.
5. Resize/crop to a fixed shape, for example 128 x 128 x 128.
6. Normalize voxel intensity.

Co-registration command:

```bash
python -m src.coregister_mri \
  --manifest data/manifests/oasis1_200_manifest.csv \
  --template /path/to/MNI152_T1_1mm_brain.nii.gz \
  --output-root data/processed/oasis/coregistered \
  --output-manifest data/manifests/oasis1_200_coreg_manifest.csv \
  --fsl-dir ~/fsl
```

## 8. Generate Segmentation Targets

Run FSL-FAST:

```bash
fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -o output_name input_image.nii.gz
```

This creates gray matter, white matter, and CSF tissue maps. These are the silver-standard segmentation labels.

## 9. HCP-YA Segmentation Pretraining

Train a 3D UNet:

```text
Input: T1 MRI
Target: FSL-FAST segmentation
Output: GM / WM / CSF / background segmentation
```

Save encoder weights for OASIS fine-tuning.

## 10. OASIS Three-Output Multitask Training

Fine-tune on OASIS:

```text
Input:
  T1 MRI

Shared representation:
  3D encoder anatomical features

Output 1:
  tissue segmentation map

Output 2:
  dementia classification

Output 3:
  MMSE or CDR cognitive/severity prediction
```

Loss:

```text
total_loss =
  segmentation_loss
  + classification_loss
  + cognitive_regression_loss
```

Current public OASIS-1 demo command:

```bash
python -m src.train_multitask --config configs/oasis1_200_multitask.yaml
```

MedicalNet-style backbone command:

```bash
git clone https://github.com/Tencent/MedicalNet.git vendor/MedicalNet
python -m src.train_multitask --config configs/oasis1_medicalnet_multitask.yaml
```

The MedicalNet config keeps the same three output tasks, but swaps the encoder to a MedicalNet-style 3D ResNet backbone.

## 10.1. Longitudinal/Future Prediction

OASIS-1 is cross-sectional, so it cannot support future decline prediction by itself. For OASIS-2, ADNI, or lab longitudinal data, build a baseline-to-future manifest:

```bash
python -m src.build_longitudinal_manifest \
  --clinical data/raw/oasis2/clinical.csv \
  --baseline-manifest data/manifests/oasis2_baseline_manifest.csv \
  --output data/manifests/oasis2_longitudinal_manifest.csv \
  --subject-col subject_id \
  --visit-col visit \
  --mmse-col mmse \
  --cdr-col cdr
```

This adds targets such as:

```text
future_mmse
future_mmse_delta
future_cdr
future_cdr_delta
future_decline_label
```

## 11. Evaluation

Anatomical representation / segmentation:

- Dice score

Dementia detection:

- Accuracy
- Balanced accuracy
- AUC
- Sensitivity
- Specificity
- Confusion matrix

Cognitive/severity prediction:

- MAE
- RMSE
- R2
- Pearson/Spearman correlation

Current public OASIS-1 evaluation command:

```bash
python -m src.evaluate_oasis1_demo \
  --config configs/oasis1_200_multitask.yaml \
  --checkpoint models/oasis1_200_multitask.ckpt \
  --output-dir outputs/oasis1_200_eval
```

## 12. Compare to Paper

| Paper task | Public-data analogue |
| --- | --- |
| ADNI | OASIS-1/2 |
| HCP-YA | HCP-YA |
| ADAS-Cog11 | MMSE or CDR |
| CN/MCI/AD | CDR 0 vs CDR > 0 |
| Longitudinal cognition | OASIS-2 follow-up, if used |
