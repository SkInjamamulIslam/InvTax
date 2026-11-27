# Block F3 — InvTax v2 (20261126), run in WSL with the invtax conda env:
#   download v2 from Narval, package it like v1, score v1 vs v2 on the held out region fragments, rerun the field check with v2
set -euo pipefail
B=/mnt/r/invertebrates_database/database_files
INVTAX=$B/InvTax; V1DB=$INVTAX/models
W=$B/invtax_v2_20261126; mkdir -p $W/{dl4,dl5,run5,scripts}
V2=$B/invtax_fieldcheck_v2_20261125
RES=$B/invtax_results_20261125; DIAG=$B/invtax_fieldcheck_20261125; IN=$RES/input
BOLDDIR=/mnt/r/invertebrates_database/crabs_testing
[ -f $V1DB/hybrid_k4/best.pt ] || { echo "v1 models not found in $V1DB"; exit 1; }

cat > $W/scripts/pack_v2_20261126.py <<'PY'
#!/usr/bin/env python3
"""InvTax v2 model package (20261126): label_vocab.json, taxonomy.json, marker_masks, hybrid_k4/{best.pt, species_head_ids.npy}.
The parent table and marker masks are rebuilt from the v5 split with the same rule that reproduces the v1 package
(checked first on the v4 split against the published v1 files).
usage: pack_v2.py V1DB META4 META5 VOCAB5 RUN5 OUTDB"""
import sys, os, json, shutil
import numpy as np

R = ["phylum", "class", "order", "family", "genus", "species"]; MK = ["COI", "16S", "18S"]
V1, META4, META5, VOCAB5, RUN5, OUT = sys.argv[1:7]


def rows(m, mode):
    s = m["set"]
    return {"train": s == 0, "trainval": (s == 0) | (s == 1), "all_but_frag": s <= 3}[mode]


def parents(m, vocab, sel):
    y = m["y"]; par = {}
    for j in range(1, 6):
        c, p = y[sel, j], y[sel, j - 1]; ok = (c >= 0) & (p >= 0)
        pr, cnt = np.unique(np.stack([c[ok], p[ok]], 1), axis=0, return_counts=True)
        o = np.lexsort((-cnt, pr[:, 0])); pr = pr[o]
        first = np.r_[True, pr[1:, 0] != pr[:-1, 0]]
        a = np.full(len(vocab[R[j]]), -1, np.int64); a[pr[first, 0]] = pr[first, 1]
        par[R[j]] = a
    return par


def masks(m, vocab, sid, sel):
    y, mk = m["y"], m["marker"]; out = {}
    for i, mm in enumerate(MK):
        for j, r in enumerate(R):
            v = y[sel & (mk == i), j]; a = np.zeros(len(vocab[r]), bool); a[v[v >= 0]] = True
            out[f"{mm}_{r}"] = a
    return out


# ---------- 1. find the rule that reproduces the v1 package from the v4 split
voc1 = json.load(open(f"{V1}/label_vocab.json")); tax1 = json.load(open(f"{V1}/taxonomy.json"))
mf = sorted(f for f in os.listdir(V1) if f.startswith("marker_masks") and f.endswith(".npz"))
mask1 = np.load(f"{V1}/{mf[-1]}"); sid1 = np.load(f"{V1}/hybrid_k4/species_head_ids.npy")
m4 = np.load(META4)
print("v1 taxonomy.json keys:", list(tax1), "| marker mask file:", mf[-1])
other = [k for k in tax1 if k != "parent"]
sp_len = {len(mask1[f"{x}_species"]) for x in MK}
sp_on_head = sp_len == {len(sid1)}
print(f"v1 species mask length {sp_len} | species vocab {len(voc1['species'])} | species head {len(sid1)}")
best = None
for mode in ("train", "trainval", "all_but_frag"):
    sel = rows(m4, mode)
    par = parents(m4, voc1, sel); msk = masks(m4, voc1, sid1, sel)
    pa = np.mean([np.mean(par[r] == np.asarray(tax1["parent"][r])) for r in R[1:]])
    ma = []
    for k in mask1.files:
        a = msk[k]
        if k.endswith("_species") and sp_on_head:
            a = a[sid1]
        ma.append(np.mean(a == mask1[k]) if a.shape == mask1[k].shape else 0.0)
    ma = float(np.mean(ma))
    print(f"  rows = {mode:<13s} parent agreement {pa*100:7.3f}% | mask agreement {ma*100:7.3f}%")
    if best is None or pa + ma > best[1]:
        best = (mode, pa + ma, pa, ma)
mode, _, pa, ma = best
if pa < 0.99 or ma < 0.99:
    sys.exit(f"could not reproduce the v1 package (best: {mode}, parent {pa:.3f}, masks {ma:.3f}); stopping")
print(f"rule reproduced: rows = {mode}")

# ---------- 2. build the v2 package with that rule
m5 = np.load(META5); voc5 = json.load(open(VOCAB5)); sid5 = np.load(f"{RUN5}/species_head_ids.npy")
for r in R:
    assert voc5[r][:len(voc1[r])] == voc1[r], f"v5 vocabulary does not extend v1 at {r}"
sel = rows(m5, mode)
par = parents(m5, voc5, sel); msk = masks(m5, voc5, sid5, sel)
if sp_on_head:
    for x in MK:
        msk[f"{x}_species"] = msk[f"{x}_species"][sid5]
tax5 = {k: tax1[k] for k in other}
for k in other:
    if isinstance(tax1[k], (list, dict)) and len(tax1[k]) > 50:
        sys.exit(f"taxonomy.json key '{k}' depends on the vocabulary; not handled")
tax5["parent"] = {r: par[r].tolist() for r in R[1:]}
os.makedirs(f"{OUT}/hybrid_k4", exist_ok=True)
json.dump(voc5, open(f"{OUT}/label_vocab.json", "w"))
json.dump(tax5, open(f"{OUT}/taxonomy.json", "w"))
np.savez_compressed(f"{OUT}/marker_masks_v2_20261126.npz", **msk)
for f in ("best.pt", "species_head_ids.npy"):
    shutil.copy(f"{RUN5}/{f}", f"{OUT}/hybrid_k4/{f}")
print(f"v2 package written to {OUT}: species head {len(sid5):,} (v1 {len(sid1):,}), "
      f"genus {len(voc5['genus']):,} (v1 {len(voc1['genus']):,})")
PY
cat > $W/scripts/score_frag_20261126.py <<'PY'
#!/usr/bin/env python3
"""Score InvTax v1 and v2 on the held out primer region fragments (frag_test).
usage: score_frag.py prep FA TRUTH WORK   |   score WORK"""
import sys, os, glob
import pandas as pd

R = ["phylum", "class", "order", "family", "genus", "species"]
BAD = {"", "NA", "nan", "unclassified", "None"}


def read_fa(p):
    h, s = None, []
    for l in open(p):
        l = l.rstrip("\n")
        if l.startswith(">"):
            if h is not None:
                yield h, "".join(s)
            h, s = l[1:], []
        else:
            s.append(l)
    if h is not None:
        yield h, "".join(s)


def prep(fa, truth, work):
    t = pd.read_csv(truth, sep="\t", dtype=str, keep_default_na=False)
    seqs = [s for _, s in read_fa(fa)]
    assert len(seqs) == len(t), "frag_test.fa and frag_test_truth.tsv differ in length"
    t.insert(0, "seq_id", [f"ft{i}" for i in range(len(t))])
    t.to_csv(f"{work}/frag_truth_ids.tsv", sep="\t", index=False)
    for mk in t["marker"].unique():
        os.makedirs(f"{work}/fa_{mk}", exist_ok=True)
        with open(f"{work}/fa_{mk}/frag_test_{mk}.fasta", "w") as f:
            for i in t.index[t["marker"] == mk]:
                f.write(f">{t.at[i, 'seq_id']}\n{seqs[i]}\n")
    print(t.groupby(["marker", "primer_set"]).size().to_string())


def score(work):
    t = pd.read_csv(f"{work}/frag_truth_ids.tsv", sep="\t", dtype=str, keep_default_na=False)
    out = []
    for ver in ("v1", "v2"):
        p = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in glob.glob(f"{work}/pred_{ver}/*.csv")])
        d = t.merge(p[["seq_id"] + R], on="seq_id", how="left", suffixes=("", "_pred"))
        for ps, g in d.groupby("primer_set"):
            for r in R:
                k = ~g[r].isin(BAD)
                if k.sum() == 0:
                    continue
                pr = g.loc[k, f"{r}_pred"].fillna("unclassified")
                cls = ~pr.isin(BAD)
                ok = pr == g.loc[k, r]
                out.append(dict(model=ver, primer_set=ps, rank=r, n=int(k.sum()),
                                classified=round(100 * cls.mean(), 1), correct=round(100 * ok.mean(), 1),
                                precision=round(100 * ok[cls].mean(), 1) if cls.any() else float("nan")))
    s = pd.DataFrame(out)
    s.to_csv(f"{work}/frag_test_scores_20261126.tsv", sep="\t", index=False)
    w = s.pivot_table(index=["primer_set", "rank"], columns="model", values=["correct", "precision", "classified"])
    w = w.reindex([(a, b) for a in sorted(s.primer_set.unique()) for b in R if (a, b) in w.index])
    pd.set_option("display.width", 200)
    print("% of fragments with the correct name (correct), % given a name (classified), % of named that are right (precision)")
    print(w.round(1).to_string())


if __name__ == "__main__":
    {"prep": lambda: prep(*sys.argv[2:5]), "score": lambda: score(sys.argv[2])}[sys.argv[1]]()
PY

echo "== 1. Download v2 from Narval (one password and passcode)"
H=injamam@narval.alliancecan.ca; N=/home/injamam/scratch/inver
CM="-o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm_%r@%h -o ControlPersist=30m"
scp $CM $H:$N/split_np/meta.npz $W/dl4/
scp $CM $H:$N/split_np_v5_20261126/meta.npz $H:$N/split_np_v5_20261126/label_vocab.json $W/dl5/
scp $CM $H:$N/runs_v5_20261126/hybrid_k4_main/best.pt $H:$N/runs_v5_20261126/hybrid_k4_main/species_head_ids.npy $W/run5/
scp $CM $H:$N/split_v5_20261126/frag_test.fa $H:$N/split_v5_20261126/frag_test_truth.tsv $W/
ls -la $W/dl4 $W/dl5 $W/run5

echo "== 2. Package v2 the same way as v1"
python $W/scripts/pack_v2_20261126.py $V1DB $W/dl4/meta.npz $W/dl5/meta.npz $W/dl5/label_vocab.json $W/run5 $W/models_v2_20261126

echo "== 3. Region test: v1 and v2 on the held out fragments"
python $W/scripts/score_frag_20261126.py prep $W/frag_test.fa $W/frag_test_truth.tsv $W
cd $INVTAX
for ver in v1 v2; do
  db=$V1DB; [ $ver = v2 ] && db=$W/models_v2_20261126
  for d in $W/fa_*; do mk=${d##*fa_}
    python invtax.py --fasta $d --marker $mk --k 4 --db $db --out $W/pred_$ver
  done
done
python $W/scripts/score_frag_20261126.py score $W

echo "== 4. Field check with the v2 model (same references, strand and identity rules)"
F=$V2/model_v2_20261126; mkdir -p $F/invtax $F/final
sed -e "s#f\"{V2}/invtax/#f\"{os.environ.get('INVDIR', V2 + '/invtax')}/#g" \
    -e "s#f\"{V2}/final/#f\"{os.environ.get('FINDIR', V2 + '/final')}/#g" $V2/scripts/fc3_20261125.py > $W/scripts/fc3_v2model_20261126.py
grep -q INVDIR $W/scripts/fc3_v2model_20261126.py || { echo "could not redirect the field check outputs"; exit 1; }
python invtax.py --fasta $V2/otus/COI --marker COI --k 4 --db $W/models_v2_20261126 --out $F/invtax
python invtax.py --fasta $V2/otus/16S --marker 16S --k 4 --db $W/models_v2_20261126 --out $F/invtax
INVDIR=$F/invtax FINDIR=$F/final python $W/scripts/fc3_v2model_20261126.py final "$RES" "$DIAG" "$IN" "$V2" "$BOLDDIR" > $F/final/final_log_20261126.txt
python - <<PY
import pandas as pd
a = pd.read_csv("$V2/final/fieldcheck_v2_final_summary_20261125.tsv", sep="\t").set_index("set").T
b = pd.read_csv("$F/final/fieldcheck_v2_final_summary_20261125.tsv", sep="\t").set_index("set").T
rows = [r for r in a.index if any(x in r for x in ("pct_raw", "pct_final", "species_final", "sp_Atlantic", "sp_Canada", "sp_outside"))]
t = pd.concat({"v1": a.loc[rows], "v2": b.loc[rows]}, axis=1).swaplevel(axis=1).sort_index(axis=1, level=0)
pd.set_option("display.width", 250); print(t.to_string())
PY
echo "Outputs: $W (region test), $F/final (field check with v2)"
