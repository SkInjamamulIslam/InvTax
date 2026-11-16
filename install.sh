#!/usr/bin/env bash
# InvTax installer: creates the conda environment, installs packages and downloads the models.
#   bash install.sh            (environment name: invtax)
#   bash install.sh myenv      (custom environment name)
set -e
ENV=${1:-invtax}
cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"; exit 1
fi
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  conda create -y -n "$ENV" python=3.11
fi
conda activate "$ENV"
pip install -r requirements.txt
python download_models.py
python invtax.py --help > /dev/null
echo
echo "InvTax is installed. To classify sequences:"
echo "  conda activate $ENV"
echo "  python invtax.py --fasta your_sequences.fasta --marker COI --out results"
