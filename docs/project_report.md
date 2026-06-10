# Project Summary

## Project Goal

This project implements a public-data reproduction scaffold for the UCSF single-MRI multitask modeling strategy:

```text
Single T1 MRI scan
-> anatomical representation / segmentation
-> CDR-based dementia-status classification
-> cognitive prediction
```

The goal is to implement the end-to-end technical workflow and provide a clear baseline that can be extended with larger datasets, additional preprocessing, and GPU-scale training.

## Dataset Used

The current experiment uses OASIS-1 as a public substitute for ADNI.

```text
Subjects: 174
Train: 121
Validation: 26
Test: 27
MRI input: single T1-weighted scan
Dementia label: CDR 0 vs CDR > 0
Cognitive target: MMSE
Segmentation target: FSL-FAST tissue segmentation
```

The label rule is:

```text
CDR 0   -> control / non-demented
CDR > 0 -> impaired / dementia label
```

This label definition is simpler than the UCSF/ADNI setup, which can include richer diagnostic categories and longitudinal cognitive targets.

## Model Design

The model is a 3D multitask UNet-style network.

```text
Input:
  3D T1 MRI volume

Shared encoder:
  learns anatomical MRI representation

Segmentation head:
  predicts background / CSF / gray matter / white matter

Classification head:
  predicts CDR-based dementia-status probability

Cognition head:
  predicts MMSE score
```

The training loss combines the three tasks:

```text
total_loss =
  segmentation_loss
  + dementia_classification_loss
  + MMSE_regression_loss
```

## Current Results

Current OASIS-1 test-set results:

| Task | Metric | Test result |
| --- | ---: | ---: |
| Anatomical segmentation | Dice | 0.884 |
| CDR-based dementia-status classification | Accuracy | 0.519 |
| CDR-based dementia-status classification | Balanced accuracy | 0.500 |
| CDR-based dementia-status classification | AUC | 0.758 |
| Cognitive prediction | MMSE MAE | 2.62 |
| Cognitive prediction | MMSE RMSE | 3.37 |

## Interpretation

The anatomical segmentation task worked well. A test Dice score around 0.88 shows that the model learned tissue-level anatomical structure from the T1 MRI.

The CDR-based dementia-status classification result is limited in the current OASIS-1 experiment. Predicted probabilities stayed near 0.494, so the default 0.5 threshold classified all subjects as control. This produced chance-level balanced accuracy, while the AUC indicates preliminary ranking signal.

The MMSE head provides a baseline regression result, with predictions concentrated near the cohort mean.

## Implemented Workflow

This project demonstrates that the full workflow is operational:

- OASIS MRI and clinical data are converted into a manifest.
- T1 MRI volumes are downsampled and loaded into a 3D model.
- FSL-FAST tissue segmentations are used as anatomical targets.
- A single model produces segmentation, CDR-based dementia-status probability, and MMSE prediction.
- Results are evaluated with segmentation, classification, and regression metrics.
- Outputs are saved as CSV tables and visualization figures.

## Current Limitations

The current version is a public-data reproduction scaffold, not a full UCSF-level replication.

- OASIS-1 is a public substitute, not ADNI.
- The dementia label is simplified to CDR 0 vs CDR > 0.
- MRI volumes were downsampled for CPU training.
- Training used limited CPU resources rather than GPU-scale training.
- The preprocessing pipeline is simplified.
- Age and sex covariate support has been added, but the covariate-enabled model has not yet been trained and evaluated.
- No full hyperparameter search has been run.
- No external validation set has been used yet.

## Next Steps

To move closer to the UCSF study design, the following are needed:

- ADNI or lab dataset access.
- Exact clinical label definitions.
- Preferred preprocessing protocol.
- GPU compute.
- More complete MRI preprocessing.
- Age/sex and other covariate integration.
- Target tasks such as CN vs MCI vs AD, MMSE, CDR, or ADAS-Cog.
- A fixed train/validation/test split.
- External validation data if available.

## Summary

This repository implements a UCSF-style single-MRI multitask pipeline using public OASIS-1 data. The model takes one T1 MRI and produces three outputs: tissue/anatomical segmentation, CDR-based dementia-status classification, and MMSE cognitive prediction. On 174 OASIS-1 subjects, the segmentation task performs well, while the clinical prediction tasks remain limited under the current public-data and training setup. The repository provides a structured baseline for further development with richer labels, longitudinal data, stronger preprocessing, and GPU training.
