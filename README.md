# InvTax

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24594275.svg)](https://doi.org/10.5281/zenodo.24594275)

A multi-marker hybrid CNN-Transformer classifier for invertebrate eDNA and eRNA metabarcoding reads from three markers: **COI, 16S and 18S**.

InvTax assigns every sequence a full lineage (kingdom, phylum, class, order, family, genus, species) with a probability score for each rank. It uses a hybrid convolution and attention network on k-mer tokens (k = 4).

**Version 2 (recommended)** is trained on the curated reference of 2.34 million COI, 16S and 18S sequences plus 0.73 million primer region fragments recovered from full length NCBI and BOLD barcodes, so that short amplicons (for example MG2, EPTDr2n and Epp regions) are classified as well as full length barcodes. See [CHANGELOG.md](CHANGELOG.md).

| Model | Training data | Use |
|---|---|---|
| v2, hybrid k4 (default, `models/`) | v4 reference + recovered primer region fragments | recommended for all amplicons, especially short ones |
| v1, hybrid k4, k6, k8 (`models_v1/`) | v4 reference (1.55 million training sequences) | models used for the benchmark in the paper |

## Installation

Requires [conda](https://docs.conda.io/en/latest/miniconda.html) (Linux, macOS or Windows WSL). A GPU is used automatically when available; CPU works too.

```bash
git clone https://github.com/SkInjamamulIslam/InvTax.git
cd InvTax
bash install.sh
```

`install.sh` creates the `invtax` conda environment, installs the packages in `requirements.txt` and vsearch, and downloads the v2 model from Zenodo into `models/`.

Other downloads:

```bash
python download_models.py --version v1     # v1 models of the paper benchmark (hybrid k4, k6, k8) -> models_v1/
python download_models.py --refs           # primer region references for invtax_refcheck.py -> refs/
```

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
| `--k` | Model k-mer size: 4 (default); 6 and 8 only with the v1 models (`--db models_v1`) |
| `--cutoffs` | Minimum probability per rank, kingdom to species (default `0.90,0.80,0.70,0.60,0.50,0.40,0.30`) |
| `--no_cutoffs` | Report the best taxon at every rank with its raw probability |
| `--device` | `auto` (default, GPU if available), `cpu` or `cuda` |
| `--db` | Model folder, if not in `models/` (for example `models_v1` for the v1 models) |

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

## Reference check for field data (recommended)

Field sequences often belong to taxa that are missing from every reference. InvTax still names the closest taxon it knows, sometimes with a high probability. `invtax_refcheck.py` aligns each sequence to the reference of its primer region (vsearch) and keeps a rank only when the identity supports it: a strict identity floor per rank (the identity at which InvTax reached 0.90 precision on the test set), or, for genus and species, agreement between InvTax and the nearest reference.

```bash
python download_models.py --refs
python invtax_refcheck.py orient --fasta otus.fasta --ref refs/COI_FWH2_ref_v2.tsv --out otus_fwd.fasta
python invtax.py --fasta otus_fwd.fasta --marker COI --out results
python invtax_refcheck.py check --fasta otus_fwd.fasta --ref refs/COI_FWH2_ref_v2.tsv \
    --invtax results/otus_fwd_InvTax_COI_k4_<date>.csv --marker COI
```

Primer region references provided: `COI_FWH2` (fwhF2/fwhR2n), `COI_MG2` (MG2/LCO1490), `COI_EPTD` (fwhF2/EPTDr2n), `16S_Tak` (Takenaka MtInsects 16S), `16S_EPP` (Epp 16S). The output adds `nearest_ref_identity`, the nearest reference genus and species, and `<rank>_final` columns. For the short 16S Epp region (about 108 bp), species calls are best read only to genus.

## Repository contents

```
invtax.py            classifier (command line)
invtax_refcheck.py   nearest reference identity and rank floors for field data (needs vsearch)
download_models.py   downloads the trained models (v2 default, v1 optional) and the region references from Zenodo
CHANGELOG.md         what changed between versions
install.sh           environment setup and model download
test_invtax.py       installation test on the bundled example sequences
example/             held-out test sequences of known taxonomy for each marker
scripts/             code used to build the reference database, train, evaluate and benchmark the models
scripts/v2/ code used to recover the primer region fragments and build, train and test v2
```

## Data and models

Trained models, the curated COI, 16S and 18S reference database and the primer region references are archived on Zenodo:

| Version | Contents | DOI |
|---|---|---|
| 2.0.0 (current) | v2 model (hybrid k4), primer region references, checksums | [10.5281/zenodo.24594275](https://doi.org/10.5281/zenodo.24594275) |
| 1.0.0 | v1 models (hybrid k4, k6, k8) used in the paper benchmark, v4 reference database | [10.5281/zenodo.24289250](https://doi.org/10.5281/zenodo.24289250) |

`install.sh` downloads the v2 model automatically; `download_models.py` always takes the files from the latest version.

## Citation

Islam SI, Knysh K, Steeves R, Courtenay SC, van den Heuvel MR. InvTax: a multi-marker hybrid CNN-Transformer classifier for invertebrate eDNA and eRNA metabarcoding. In preparation.

## License

MIT License. See `LICENSE`.
