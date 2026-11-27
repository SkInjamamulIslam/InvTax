#!/usr/bin/env python3
"""
InvTax reference check: nearest reference identity plus rank floors for InvTax calls (needs vsearch).

Field sequences often come from taxa that are not in any reference. InvTax then still names the closest
taxon it knows, sometimes with high probability. This script aligns every sequence to the reference of its
primer region and keeps a rank only when the identity supports it:

  * strict floor: the identity at which InvTax reached at least 0.90 precision on the test set
      COI  phylum/class/order 75%  family 90%  genus 97%  species 99%
      16S  phylum/class/order 75%  family 90%  genus 99%  species 99%
  * agreement: genus and species are also kept when InvTax and the nearest reference name the same taxon,
      from COI genus 92% / 16S genus 95% / species 97%

Step 1 (optional, recommended): put every sequence on the forward strand of the reference
    python invtax_refcheck.py orient --fasta otus.fasta --ref refs/COI_FWH2_ref_v2.tsv --out otus_fwd.fasta
Step 2: classify
    python invtax.py --fasta otus_fwd.fasta --marker COI --out results
Step 3: check
    python invtax_refcheck.py check --fasta otus_fwd.fasta --ref refs/COI_FWH2_ref_v2.tsv \\
        --invtax results/otus_fwd_InvTax_COI_k4_<date>.csv --marker COI

Output of step 3: <InvTax csv>_refcheck.csv = the InvTax table plus nearest_ref_id, nearest_ref_identity,
nearest_ref_genus, nearest_ref_species, <rank>_final (after the floors) and deepest_final.
References: refs/ from `python download_models.py --refs` (one table per primer region).
"""
import argparse, os, re, shutil, subprocess, sys, tempfile
import numpy as np, pandas as pd

R = ["phylum", "class", "order", "family", "genus", "species"]
FLOOR = {"COI": dict(phylum=75, order=75, family=90, genus=97, species=99, **{"class": 75}),
         "16S": dict(phylum=75, order=75, family=90, genus=99, species=99, **{"class": 75})}
AGREE = {"COI": dict(genus=92, species=97), "16S": dict(genus=95, species=97)}
COMP = str.maketrans("ACGTRYKMBDHVNacgtrykmbdhvn", "TGCAYRMKVHDBNtgcayrmkvhdbn")
BAD = {"", "nan", "NA", "unclassified"}


def read_fa(p):
    h, s = None, []
    for l in open(p, errors="ignore"):
        l = l.rstrip("\r\n")
        if l.startswith(">"):
            if h is not None:
                yield h, "".join(s)
            h, s = l[1:], []
        else:
            s.append(l.strip())
    if h is not None:
        yield h, "".join(s)


def ref_fasta(ref):
    fa = re.sub(r"\.tsv(\.gz)?$", "", ref) + ".fa"
    if not os.path.exists(fa):
        t = pd.read_csv(ref, sep="\t", dtype=str, usecols=["id", "sequence"], keep_default_na=False)
        with open(fa, "w") as f:
            for i, s in zip(t["id"], t["sequence"]):
                f.write(f">{i}\n{s}\n")
    return fa


def search(fasta, ref, threads):
    if not shutil.which("vsearch"):
        sys.exit("vsearch not found: conda install -c bioconda vsearch")
    out = tempfile.NamedTemporaryFile(suffix=".tsv", delete=False).name
    subprocess.run(["vsearch", "--usearch_global", fasta, "--db", ref_fasta(ref), "--id", "0.6", "--strand", "both",
                    "--maxaccepts", "16", "--maxrejects", "64", "--threads", str(threads), "--quiet",
                    "--userfields", "query+target+id+qstrand", "--userout", out], check=True)
    h = pd.read_csv(out, sep="\t", header=None, names=["seq_id", "nearest_ref_id", "nearest_ref_identity", "strand"], dtype=str)
    os.remove(out)
    h["nearest_ref_identity"] = pd.to_numeric(h["nearest_ref_identity"], errors="coerce")
    return h.sort_values("nearest_ref_identity", ascending=False).drop_duplicates("seq_id")


def orient(a):
    h = search(a.fasta, a.ref, a.threads).set_index("seq_id")
    n = 0
    with open(a.out, "w") as o:
        for hd, s in read_fa(a.fasta):
            q = hd.split()[0]
            if q in h.index and h.at[q, "strand"] == "-" and h.at[q, "nearest_ref_identity"] >= 80:
                s = s.translate(COMP)[::-1]; n += 1
            o.write(f">{hd}\n{s}\n")
    print(f"{n} of {sum(1 for _ in read_fa(a.fasta))} sequences reverse complemented -> {a.out}")


def check(a):
    d = pd.read_csv(a.invtax, dtype=str, keep_default_na=False)
    d["seq_id"] = d["seq_id"].str.split().str[0]
    h = search(a.fasta, a.ref, a.threads).drop(columns="strand")
    ref = pd.read_csv(a.ref, sep="\t", dtype=str, usecols=["id", "genus", "species"], keep_default_na=False)
    ref = ref[ref["id"].isin(set(h["nearest_ref_id"]))].drop_duplicates("id")
    h = h.merge(ref.rename(columns={"id": "nearest_ref_id", "genus": "nearest_ref_genus", "species": "nearest_ref_species"}),
                on="nearest_ref_id", how="left")
    d = d.merge(h, on="seq_id", how="left")
    idv = d["nearest_ref_identity"].fillna(0)
    norm = lambda s: s.fillna("").astype(str).str.replace(r"[\s_]+", " ", regex=True).str.strip().str.lower()
    for i, r in enumerate(R):
        lab = d[r].where(~d[r].isin(BAD), "unclassified")
        ok = idv >= FLOOR[a.marker][r]
        if r in AGREE[a.marker]:
            ok |= (idv >= AGREE[a.marker][r]) & (norm(lab) == norm(d[f"nearest_ref_{r}"]))
        prev = (d[f"{R[i-1]}_final"] != "unclassified") if i else True
        d[f"{r}_final"] = np.where(ok & prev & (lab != "unclassified"), lab, "unclassified")
    fin = d[[f"{r}_final" for r in R]].ne("unclassified")
    d["deepest_final"] = [R[int(np.flatnonzero(x)[-1])] if x.any() else "none" for x in fin.to_numpy()]
    out = re.sub(r"\.csv$", "", a.invtax) + "_refcheck.csv"
    d.to_csv(out, index=False)
    raw = d[R].apply(lambda c: ~c.isin(BAD)).mean() * 100
    print(f"{len(d):,} sequences | median nearest reference identity {idv.median():.1f}% | >= 97%: {(idv >= 97).mean()*100:.1f}%")
    print("rank      InvTax   after check")
    for r in R:
        print(f"{r:<9s} {raw[r]:6.1f}%  {fin[f'{r}_final'].mean()*100:6.1f}%")
    print(f"written: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    o = sp.add_parser("orient", help="reverse complement sequences whose best reference hit is on the minus strand")
    o.add_argument("--fasta", required=True); o.add_argument("--ref", required=True); o.add_argument("--out", required=True)
    o.add_argument("--threads", type=int, default=os.cpu_count())
    c = sp.add_parser("check", help="add nearest reference identity and apply the rank floors to an InvTax table")
    c.add_argument("--fasta", required=True); c.add_argument("--ref", required=True); c.add_argument("--invtax", required=True)
    c.add_argument("--marker", choices=["COI", "16S"], required=True)
    c.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    orient(a) if a.cmd == "orient" else check(a)
