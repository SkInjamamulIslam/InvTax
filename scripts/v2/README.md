# InvTax v2 build scripts

Code used to build InvTax v2. File names inside the scripts carry the dates of the original runs.

1. `block_F1_region_recovery.sh` (local, vsearch): finds the full NCBI + BOLD reference, recovers each primer region by global alignment (`region_recovery.py`), writes the primer region references and the recovered fragments.
2. `block_F2_refcheck_field.sh` (local): strand from the region references, InvTax, identity floors, agreement rule and BOLD collection region (`region_refcheck_bold.py`).
3. `block_R2_train_narval.sh` (Narval, SLURM): `prep_v5.py` builds the v5 split (v4 rows unchanged, fragments appended), then a pilot, full hybrid k4 training (`models/train_hybrid.py` from v1) and evaluation.
4. `block_F3_package_test.sh` (local): `pack_v2.py` packages the model the same way as v1 (checked against the v1 files), `score_frag.py` scores v1 and v2 on the held out region fragments, and the field check is repeated with v2.
