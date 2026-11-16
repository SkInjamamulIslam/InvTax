# InvTax

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24289250.svg)](https://doi.org/10.5281/zenodo.24289250)

A multi-marker hybrid CNN-Transformer classifier for invertebrate eDNA and eRNA metabarcoding reads from three markers: **COI, 16S and 18S**.

InvTax assigns every sequence a full lineage (kingdom, phylum, class, order, family, genus, species) with a probability score for each rank. It uses a hybrid convolution and attention network on k-mer tokens (k = 4 by default; k = 6 and 8 also provided), trained on a curated reference database of 2.34 million sequences.

## Installation

Requires [conda](https://docs.conda.io/en/latest/miniconda.html) (Linux, macOS or Windows WSL). A GPU is used automatically when available; CPU works too.

```bash
git clone https://github.com/SkInjamamulIslam/InvTax.git
cd InvTax
bash install.sh
```

`install.sh` creates the `invtax` conda environment, installs the packages in `requirements.txt` and downloads the trained models (about 1.1 GB) from Zenodo into `models/`.

To check the installation:

```bash
conda activate invtax
python test_invtax.py
```

## Usage

```bash
conda activate invtax
python invtax.py --fasta my_asvs.fasta --marker COI --out results
```

A folder of FASTA files can be given instead of one file:

```bash
python invtax.py --fasta fasta_folder/ --marker 16S --out results
```

| Option | Description |
|---|---|
| `--fasta` | FASTA file (plain or .gz) or a folder of FASTA files |
| `--marker` | `COI`, `16S` or `18S`. Predictions are restricted to the taxa held in the reference database for that marker. `auto` lets the model decide per sequence |
| `--out` | Output folder (created if needed) |
| `--k` | Model k-mer size: 4 (default), 6 or 8 |
| `--cutoffs` | Minimum probability per rank, kingdom to species (default `0.90,0.80,0.70,0.60,0.50,0.40,0.30`) |
| `--no_cutoffs` | Report the best taxon at every rank with its raw probability |
| `--device` | `auto` (default, GPU if available), `cpu` or `cuda` |
| `--db` | Model folder, if not in `models/` |

## Output

For each FASTA, two files with the same content are written to `--out`:

```
<fasta name>_InvTax_<marker>_k4_<YYYYMMDD>.tsv
<fasta name>_InvTax_<marker>_k4_<YYYYMMDD>.csv
```

One row per sequence:

| Column | Content |
|---|---|
| `seq_id` | Sequence identifier from the FASTA header |
| `marker` | Marker used for the reference restriction |
| `kingdom`, `kingdom_prob` | Kingdom of the assigned phylum and the summed probability of all phyla in that kingdom |
| `phylum` ... `species` | Assigned taxon at each rank |
| `<rank>_prob` | Model probability (0 to 1) for the assigned taxon |

Lineages are decoded top down: each rank is chosen only among the children of the taxon assigned at the rank above, so every row is a consistent lineage. `NA` marks a rank that is not defined for that taxon in the reference taxonomy.

The first rank whose probability falls below its cutoff, and every rank below it, is reported as `unclassified`. Default cutoffs: kingdom 0.90, phylum 0.80, class 0.70, order 0.60, family 0.50, genus 0.40, species 0.30. The probability is always kept, so cutoffs can be changed afterwards (`--cutoffs`), or switched off (`--no_cutoffs`).

## Repository contents

```
invtax.py            classifier (command line)
download_models.py   downloads the trained models from Zenodo
install.sh           environment setup and model download
test_invtax.py       installation test on the bundled example sequences
example/             held-out test sequences of known taxonomy for each marker
scripts/             code used to build the reference database, train, evaluate and benchmark the models
```

## Data and models

Trained models (Hybrid k4, k6 and k8) and the curated COI, 16S and 18S reference database are archived on Zenodo: [https://doi.org/10.5281/zenodo.24289250](https://doi.org/10.5281/zenodo.24289250). `install.sh` downloads the models automatically.

## Citation

Islam SI, Knysh K, Steeves R, Courtenay SC, van den Heuvel MR. InvTax: a multi-marker hybrid CNN-Transformer classifier for invertebrate eDNA and eRNA metabarcoding. In preparation.

## License

MIT License. See `LICENSE`.
