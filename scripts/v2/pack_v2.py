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
import re
BIN = re.compile(r"^[A-Z][a-z]+ [a-z][a-z-]+$")
for k in other:
    v = tax1[k]
    if not (isinstance(v, (list, dict)) and len(v) > 50):
        continue
    if k != "species_placeholder" or not isinstance(v, list):
        sys.exit(f"taxonomy.json key '{k}' depends on the vocabulary; not handled")
    n1, new_sp = len(voc1["species"]), voc5["species"][len(voc1["species"]):]
    rule = lambda names: [not bool(BIN.match(s.split("|")[0])) for s in names]
    if len(v) == n1:                                   # one flag per species: keep v1 flags, add the new species by rule
        agree = np.mean(np.asarray(v, bool) == np.asarray(rule(voc1["species"])))
        tax5[k] = list(v) + [type(v[0])(x) for x in rule(new_sp)]
    else:                                              # list of species ids: keep v1 ids, add new ids by rule
        agree = np.mean(np.isin(np.flatnonzero(rule(voc1["species"])), np.asarray(v)))
        tax5[k] = list(v) + [n1 + i for i, x in enumerate(rule(new_sp)) if x]
    print(f"species_placeholder: v1 entries kept; {len(new_sp):,} new species flagged by the binomial rule "
          f"(rule agrees with v1 on {agree*100:.1f}%)")
tax5["parent"] = {r: par[r].tolist() for r in R[1:]}
os.makedirs(f"{OUT}/hybrid_k4", exist_ok=True)
json.dump(voc5, open(f"{OUT}/label_vocab.json", "w"))
json.dump(tax5, open(f"{OUT}/taxonomy.json", "w"))
np.savez_compressed(f"{OUT}/marker_masks_v2_20261126.npz", **msk)
for f in ("best.pt", "species_head_ids.npy"):
    shutil.copy(f"{RUN5}/{f}", f"{OUT}/hybrid_k4/{f}")
print(f"v2 package written to {OUT}: species head {len(sid5):,} (v1 {len(sid1):,}), "
      f"genus {len(voc5['genus']):,} (v1 {len(voc1['genus']):,})")
