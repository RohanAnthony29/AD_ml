# Public-Data Prototype Report

## Project Goal

This project implements a public-data prototype of the UCSF single-MRI multitask modeling strategy:

```text
Single T1 MRI scan
-> anatomical representation / segmentation
-> dementia classification
-> cognitive prediction
```

The goal is not to claim final clinical performance yet. The goal is to show that the full technical pipeline can be built, run, evaluated, and extended when the lab provides stronger data and compute.

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

This is simpler than the UCSF/ADNI setup, which can use richer clinical categories and longitudinal cognitive targets.

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
  predicts dementia probability

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
| Dementia classification | Accuracy | 0.519 |
| Dementia classification | Balanced accuracy | 0.500 |
| Dementia classification | AUC | 0.758 |
| Cognitive prediction | MMSE MAE | 2.62 |
| Cognitive prediction | MMSE RMSE | 3.37 |

## Interpretation

The anatomical segmentation task worked well. A test Dice score around 0.88 shows that the model learned tissue-level anatomical structure from the T1 MRI.

The dementia classifier is not clinically successful yet. The predicted probabilities stayed near 0.494, so the default 0.5 threshold classified every subject as control. This gives chance-level balanced accuracy even though the AUC suggests there may be some weak ranking signal.

The MMSE head gives a reasonable baseline error, but it mostly predicts near the cohort mean. It is not yet a strong individualized cognitive predictor.

## What This Demonstrates

This project demonstrates that the full workflow is operational:

- OASIS MRI and clinical data are converted into a manifest.
- T1 MRI volumes are downsampled and loaded into a 3D model.
- FSL-FAST tissue segmentations are used as anatomical targets.
- A single model produces segmentation, dementia probability, and MMSE prediction.
- Results are evaluated with segmentation, classification, and regression metrics.
- Outputs are saved as CSV tables and visualization figures.

## Current Limitations

The current version is a prototype, not a UCSF-level reproduction.

- OASIS-1 is a public substitute, not ADNI.
- The dementia label is simplified to CDR 0 vs CDR > 0.
- MRI volumes were downsampled for CPU training.
- Training used limited CPU resources rather than GPU-scale training.
- The preprocessing pipeline is simplified.
- Age and sex are not yet fused into the prediction heads.
- No full hyperparameter search has been run.
- No external validation set has been used yet.

## What Is Needed Next

To move closer to the UCSF result, the lab can help with:

- ADNI or lab dataset access.
- The lab's exact clinical label definitions.
- A preferred preprocessing protocol.
- GPU compute.
- More complete MRI preprocessing.
- Age/sex and other covariate integration.
- Clear target tasks such as CN vs MCI vs AD, MMSE, CDR, or ADAS-Cog.
- A fixed train/validation/test split.
- External validation data if available.

## Suggested Professor-Facing Summary

I built a public-data prototype of the UCSF-style single-MRI multitask pipeline. The model takes one T1 MRI and produces three outputs: tissue/anatomical segmentation, dementia classification, and MMSE cognitive prediction. On 174 OASIS-1 subjects, the segmentation/anatomy task works well, but the dementia classification head is not reliable yet. This shows that the engineering pipeline is working, while the clinical prediction performance needs better data, preprocessing, GPU training, and tuning.
