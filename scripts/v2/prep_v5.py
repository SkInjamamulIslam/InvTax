#!/usr/bin/env python3
"""InvTax v2 data (20261126): split_v4 + recovered primer region fragments -> split_v5_20261126 and split_np_v5_20261126.
v4 rows keep their order, set, fold and label ids (vocabulary is only appended to), so v1 predictions stay valid.
Fragments are appended: 95% of accessions -> train, 5% -> frag_test (set code 4, not used in training).
Removed from the fragments: accessions or identical sequences of any v4 val/test/novel row, held out genera,
sequences already in v4 train, and duplicates; at most CAP fragments per taxon per primer set."""
import os, sys, json
import numpy as np, pandas as pd

D = "20261126"; SEED = 42; CAP = int(os.environ.get("CAP", 10))
FR = sys.argv[1]
R = ["phylum", "class", "order", "family", "genus", "species"]; UNC = "unclassified"
C = ["accession", "name", "taxid", "kingdom", "phylum", "class", "order", "family", "genus", "species", "sequence", "marker"]
OUT, NP = f"split_v5_{D}", f"split_np_v5_{D}"
os.makedirs(OUT, exist_ok=True); os.makedirs(NP, exist_ok=True)
rng = np.random.default_rng(SEED)
P = lambda *a: print(*a, flush=True)
real = lambda s: (s != "NA") & (s != UNC) & (s != "") & (s != "nan")
hsh = lambda s: pd.util.hash_pandas_object(s.str.upper(), index=False).to_numpy()


def keys(df):  # identical to make_split.py (homonym safe lineage keys)
    k = {"phylum": df["phylum"], "class": df["class"] + "|" + df["phylum"]}
    for r in R[2:]:
        k[r] = df[r] + "|" + df["phylum"] + "|" + df["class"]
    return pd.DataFrame({r: k[r].where(real(df[r]), "") for r in R})


# ---------------- v4
ds = pd.read_csv("split_v4/dataset.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
voc4 = json.load(open("split_v4/label_vocab.json"))
ref = pd.read_csv("reference_v4_final/invert_reference_v4.txt", names=C, sep="\t", header=None, dtype=str,
                  keep_default_na=False, quoting=3, on_bad_lines="skip")
assert len(ref) == len(ds) and (ref["accession"].to_numpy() == ds["accession"].to_numpy()).all(), "reference and split_v4 rows differ"
kref = keys(ref)
held = set(kref.loc[ds["heldout_genus"] == "True", "genus"]) - {""}
P(f"v4 rows {len(ds):,} | sets {ds['set'].value_counts().to_dict()} | held out genera {len(held):,}")
h4 = hsh(ds["sequence"])
nontr = (ds["set"] != "train").to_numpy()
bad_acc = set(ds.loc[nontr, "accession"]); bad_seq = pd.Index(np.unique(h4[nontr])); tr_seq = pd.Index(np.unique(h4[~nontr]))

# ---------------- fragments
fr = pd.read_csv(FR, sep="\t", dtype=str, keep_default_na=False).rename(columns={"id": "accession", "set": "primer_set"})
fr["sequence"] = fr["sequence"].str.upper()
log = [("recovered fragments", len(fr))]
def step(mask, name):
    global fr
    fr = fr[mask].copy(); log.append((name, len(fr)))
L = fr["sequence"].str.len(); step((L >= 20) & (L <= 1000), "length 20 to 1000 bp")
step(real(fr["phylum"]), "phylum known")
step(~fr["species"].str.contains("uncultured bacterium|uncultured archaeon|UNVERIFIED", case=False)
     & ~fr["accession"].str.contains("UNVERIFIED"), "no uncultured or unverified")
kf = keys(fr); step(~kf["genus"].isin(held).to_numpy(), "not a held out genus")
hf = hsh(fr["sequence"])
step(~(fr["accession"].isin(bad_acc).to_numpy() | pd.Index(hf).isin(bad_seq)), "no accession or sequence of v4 val/test/novel")
hf = hsh(fr["sequence"]); step(~pd.Index(hf).isin(tr_seq), "not identical to a v4 train sequence")
kf = keys(fr); hf = hsh(fr["sequence"])
fr["_h"] = hf; fr["_sp"] = kf["species"].to_numpy()
step(~fr.duplicated(["_h", "_sp"]).to_numpy(), "unique sequence per species")

# frag_test: 5% of accessions; drop test rows whose sequence also occurs in fragment train
u = fr["accession"].unique(); te_acc = set(rng.choice(u, max(1, int(0.05 * len(u))), replace=False))
fr["set"] = np.where(fr["accession"].isin(te_acc), "frag_test", "train")
trh = pd.Index(fr.loc[fr["set"] == "train", "_h"].unique())
step(~((fr["set"] == "frag_test") & pd.Index(fr["_h"]).isin(trh)).to_numpy(), "frag_test not identical to fragment train")

# cap per taxon (deepest known key) per primer set, separately for train and frag_test
kf = keys(fr)
deep = kf["species"].where(kf["species"] != "", kf["genus"]).where(lambda s: s != "", kf["family"]) \
    .where(lambda s: s != "", kf["order"]).where(lambda s: s != "", kf["class"]).where(lambda s: s != "", kf["phylum"])
fr["_deep"] = deep.to_numpy()
fr = fr.sample(frac=1, random_state=SEED)
capn = np.where(fr["set"] == "train", CAP, 2)
fr = fr[(fr.groupby(["set", "primer_set", "_deep"]).cumcount().to_numpy() < capn)].copy()
log.append((f"cap {CAP} per taxon per primer set (2 for frag_test)", len(fr)))
P("\nFragment filters:"); [P(f"  {n:<60s} {v:>10,d}") for n, v in log]
P(pd.crosstab(fr["primer_set"], fr["set"], margins=True).to_string())

# ---------------- vocabulary: v4 order kept, new train fragment labels appended
kf = keys(fr); trm = (fr["set"] == "train").to_numpy()
voc5 = {"marker": voc4["marker"]}
P("\nVocabulary (v4 -> v5):")
for r in R:
    old = voc4[r]; seen = set(old)
    new = sorted(set(kf.loc[trm & (kf[r] != "").to_numpy(), r]) - seen)
    voc5[r] = old + new
    P(f"  {r:<8s} {len(old):>8,d} -> {len(voc5[r]):>8,d}")
idx = {r: {k: i for i, k in enumerate(voc5[r])} for r in R}

# v4 rows: recompute ids with v5 vocab (old ids unchanged; test labels unseen in v4 may now be known)
y4 = np.stack([kref[r].map(idx[r]).fillna(-1).astype(np.int32).to_numpy() for r in R], 1)
old = ds[["y_" + r for r in R]].astype(np.int32).to_numpy()
assert (y4[old >= 0] == old[old >= 0]).all(), "v4 label ids changed"
te = (ds["set"] == "test").to_numpy()
for j, r in enumerate(R):
    had = (kref[r] != "").to_numpy() & te
    P(f"  test labels in vocabulary, {r:<8s}: v4 {(old[had, j] >= 0).mean()*100:6.2f}%  v5 {(y4[had, j] >= 0).mean()*100:6.2f}%")
nv = (ds["heldout_genus"] == "True").to_numpy()
assert (y4[nv, 4] < 0).all(), "a held out genus entered the vocabulary"
yf = np.stack([kf[r].map(idx[r]).fillna(-1).astype(np.int32).to_numpy() for r in R], 1)

# ---------------- write split_v5
ds_out = ds.copy()
for j, r in enumerate(R):
    ds_out["y_" + r] = y4[:, j]
ds_out["origin"] = "v4"
mk = {m: i for i, m in enumerate(voc5["marker"])}
f_out = pd.DataFrame({"row_id": np.arange(len(ds), len(ds) + len(fr)), "accession": fr["accession"].to_numpy(),
                      "marker": fr["marker"].to_numpy(), "marker_idx": fr["marker"].map(mk).to_numpy(),
                      "set": fr["set"].to_numpy(), "fold": -1, "heldout_genus": "False",
                      **{"y_" + r: yf[:, j] for j, r in enumerate(R)}, "sequence": fr["sequence"].to_numpy(),
                      "origin": "fragment_" + fr["primer_set"].to_numpy()})
assert f_out["marker_idx"].notna().all(), "unknown marker in fragments"
cols = list(ds.columns) + ["origin"]
allv = pd.concat([ds_out[cols], f_out[cols]], ignore_index=True)
allv.to_csv(f"{OUT}/dataset.tsv.gz", sep="\t", index=False, compression={"method": "gzip", "compresslevel": 1})
json.dump(voc5, open(f"{OUT}/label_vocab.json", "w"))
ft = fr["set"] == "frag_test"
with open(f"{OUT}/frag_test.fa", "w") as f:
    for a, s in zip(fr.loc[ft, "accession"], fr.loc[ft, "sequence"]):
        f.write(f">{a}\n{s}\n")
fr.loc[ft, ["accession", "marker", "primer_set", *R]].to_csv(f"{OUT}/frag_test_truth.tsv", sep="\t", index=False)

# ---------------- encode (same as encode_data.py; frag_test = 4)
seqs = allv["sequence"].str.upper(); Ls = seqs.str.len().to_numpy(np.int64)
lut = np.full(256, 4, np.uint8)
for i, b in enumerate(b"ACGT"):
    lut[b] = i
codes = lut[np.frombuffer("".join(seqs.tolist()).encode("ascii"), dtype=np.uint8)]
off = np.zeros(len(allv) + 1, np.int64); off[1:] = np.cumsum(Ls); assert off[-1] == len(codes)
np.save(f"{NP}/codes.npy", codes); np.save(f"{NP}/offsets.npy", off)
sets = allv["set"].map({"train": 0, "val": 1, "test": 2, "novel_genus": 3, "frag_test": 4}).to_numpy(np.int8)
y = allv[["y_" + r for r in R]].astype(np.int32).to_numpy()
np.savez(f"{NP}/meta.npz", y=y, marker=allv["marker_idx"].astype(np.int8).to_numpy(), set=sets,
         fold=allv["fold"].astype(np.int8).to_numpy(), heldout=(allv["heldout_genus"] == "True").to_numpy(),
         length=Ls.astype(np.int16), row_id=allv["row_id"].astype(np.int64).to_numpy())
json.dump(voc5, open(f"{NP}/label_vocab.json", "w"))
m4 = np.load("split_np/meta.npz")
assert (np.load(f"{NP}/meta.npz")["set"][:len(ds)] == m4["set"]).all(), "v4 set codes changed"
tr = sets == 0; lab = y[tr, 5]; cnt = np.bincount(lab[lab >= 0], minlength=len(voc5["species"]))
P(f"\nrows per set (0 train, 1 val, 2 test, 3 novel, 4 frag_test): {dict(zip(*np.unique(sets, return_counts=True)))}")
P(f"species head size (>= 2 train records): v5 {int((cnt >= 2).sum()):,}")
P(f"written: {OUT}/ and {NP}/  (frag_test.fa + frag_test_truth.tsv for the region test)")
