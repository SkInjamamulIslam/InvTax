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
