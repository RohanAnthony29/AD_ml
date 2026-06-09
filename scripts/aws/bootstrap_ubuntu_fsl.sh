#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  git \
  python3 \
  python3-pip \
  python3-venv \
  unzip \
  wget

cd "$HOME"

if ! command -v aws >/dev/null 2>&1; then
  TMP_AWS="/tmp/awscliv2.zip"
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "$TMP_AWS"
  rm -rf /tmp/aws
  unzip -q "$TMP_AWS" -d /tmp
  sudo /tmp/aws/install
fi

if [ ! -d "$HOME/alzheimers-mri-reproduction" ]; then
  git clone https://github.com/darenma/MultitaskCognition.git "$HOME/MultitaskCognition-reference"
  mkdir -p "$HOME/alzheimers-mri-reproduction"/{data/raw/hcp_ya,data/processed/hcp_ya/fast,data/manifests,outputs,models}
fi

if ! command -v fast >/dev/null 2>&1; then
  wget -O "$HOME/fslinstaller.py" https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py
  python3 "$HOME/fslinstaller.py" -d "$HOME/fsl"
fi

if ! grep -q "FSLDIR=$HOME/fsl" "$HOME/.bashrc"; then
  {
    echo ''
    echo "export FSLDIR=$HOME/fsl"
    echo 'source "$FSLDIR/etc/fslconf/fsl.sh"'
    echo 'export PATH="$FSLDIR/bin:$PATH"'
  } >> "$HOME/.bashrc"
fi

export FSLDIR="$HOME/fsl"
source "$FSLDIR/etc/fslconf/fsl.sh"
export PATH="$FSLDIR/bin:$PATH"

fast --version || true
echo "Bootstrap complete. Run: source ~/.bashrc"
