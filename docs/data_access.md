# Data Access Notes

## HCP-YA

Use HCP-YA for segmentation/anatomical pretraining.

- Website: https://www.humanconnectome.org/study/hcp-young-adult
- Download portal: https://db.humanconnectome.org/
- Required: free account and agreement to Open Access Data Use Terms.
- Needed files: T1-weighted structural MRI.

## OASIS-1 / OASIS-2

Use OASIS as the ADNI substitute.

- Website: https://www.oasis-brains.org/
- Needed files: T1-weighted MRI and clinical/demographic tables.
- Useful targets: CDR, MMSE, dementia/nondementia status.

Recommended for the first demo:

1. Start with a tiny sample, for example 5-20 scans.
2. Build `data/manifests/oasis_manifest.csv`.
3. Run FSL-FAST on those scans.
4. Run a smoke-test training pass.

## Why This Is Not Exact ADNI Reproduction

This project can reproduce the modeling strategy:

```text
single T1 MRI -> segmentation/anatomy representation -> dementia classification -> cognition/severity prediction
```

It cannot reproduce the exact ADNI results without ADNI because OASIS does not provide the same labels, splits, ADAS-Cog11 target, and longitudinal outcome structure.

Use this sentence in reports:

> This public-data version validates the pipeline and modeling strategy. Exact reproduction of the paper's reported ADNI/ADAS-Cog11 performance requires ADNI or the lab's labeled dataset.

