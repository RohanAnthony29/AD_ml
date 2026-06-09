#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 /path/to/key.pem ubuntu@EC2_PUBLIC_IP"
  exit 1
fi

KEY="$1"
HOST="$2"

rsync -av \
  -e "ssh -i $KEY" \
  --exclude ".venv" \
  --exclude ".pycache" \
  --exclude "__pycache__" \
  --exclude "data/raw" \
  --exclude "data/processed" \
  --exclude "vendor" \
  ./ "$HOST:~/alzheimers-mri-reproduction/"

