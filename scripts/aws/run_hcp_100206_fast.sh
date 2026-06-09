#!/usr/bin/env bash
set -euo pipefail

export FSLDIR="${FSLDIR:-$HOME/fsl}"
source "$FSLDIR/etc/fslconf/fsl.sh"
export PATH="$FSLDIR/bin:$PATH"

PROJECT="$HOME/alzheimers-mri-reproduction"
RAW="$PROJECT/data/raw/hcp_ya/100206/T1w"
OUT="$PROJECT/data/processed/hcp_ya/fast/100206"

mkdir -p "$RAW" "$OUT"

aws s3 cp \
  s3://hcp-openaccess/HCP_1200/100206/T1w/T1w_acpc_dc_restore.nii.gz \
  "$RAW/T1w_acpc_dc_restore.nii.gz"

fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 \
  -o "$OUT/100206" \
  "$RAW/T1w_acpc_dc_restore.nii.gz"

find "$OUT" -maxdepth 1 -type f -print
echo "FAST complete: $OUT"

