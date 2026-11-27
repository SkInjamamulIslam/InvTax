# Changelog

## 2.0.0 (2026-11-27)

Zenodo: [10.5281/zenodo.24594275](https://doi.org/10.5281/zenodo.24594275)

New default model, **InvTax v2** (hybrid k4), plus a reference check for field data.

**Why.** CRABS in silico PCR keeps a reference sequence only when both primer sites are present. BOLD stores barcodes with the primers trimmed, so almost all BOLD records were lost for amplicons whose primers sit at the ends of the Folmer barcode (MG2/LCO1490: 13,299 of 4.05 million BOLD barcodes kept; fwhF2/EPTDr2n: 1,881). Field OTUs from these regions therefore had no close reference and short reads were classified poorly.

**What changed**

* Primer regions recovered by global alignment from 3.69 million full length NCBI and BOLD sequences for five primer sets (COI fwhF2/fwhR2n, MG2/LCO1490, fwhF2/EPTDr2n; 16S Takenaka MtInsects, Epp).
* Training set: the v4 split unchanged (same train, validation and test rows and label ids) plus 725,988 recovered fragments. Fragments sharing an accession or an identical sequence with any validation, test or held out genus row were removed, as were held out genera and duplicates; at most 10 fragments per taxon per primer set. 37,454 fragments (5% of accessions) were held out as a region test.
* Vocabulary: 264,812 species (v1 246,352); species head 204,799 (v1 163,850).
* Same architecture and training settings as v1; early stopping after epoch 23.

**Results**

* Full length test set (same rows and labels as v1): equal at phylum to family; COI species MCC 0.648 (v1 0.631); 16S species F1 0.587 (v1 0.611).
* Held out primer region fragments, % with the correct genus, v1 to v2: COI MG2 7.0 to 60.1, COI EPTD 21.9 to 73.9, COI FWH2 59.0 to 72.5, 16S Tak 27.4 to 46.3, 16S Epp 9.4 to 22.5.
* Field OTUs from Atlantic Canada (COI MG2): species calls after the reference check rose from 65 to 601 OTUs, and the share of species with BOLD records outside North America only fell from 36.9% to 2.5%.

**New files**

* `invtax_refcheck.py`: strand orientation and nearest reference identity (vsearch) with rank floors and an agreement rule.
* `download_models.py --version v1|v2` and `--refs`.
* `scripts/v2/`: code used for the region recovery, the v5 split, training jobs, packaging and the region test.

v1 models stay available (`python download_models.py --version v1`); they are the models used for the benchmark in the paper.

## 1.0.0 (2026-11-16)

First release: hybrid k4, k6 and k8 models trained on the curated v4 reference (2.34 million COI, 16S and 18S sequences).
