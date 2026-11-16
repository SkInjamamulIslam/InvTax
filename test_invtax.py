#!/usr/bin/env python3
"""
InvTax installation test.

Classifies the bundled example sequences (held-out test sequences of known taxonomy,
never used in training) for each marker and reports accuracy per rank.

    python test_invtax.py

Exit code 0 = PASS, 1 = FAIL.
"""
import argparse, datetime, glob, os, subprocess, sys, tempfile
import pandas as pd

R = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
PASS_RULES = {"phylum": 0.90, "class": 0.80}          # minimum accuracy for every marker
EMPTY = {"", "NA", "nan", "unclassified"}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="InvTax installation test")
    ap.add_argument("--db", default=os.environ.get("INVTAX_DB", os.path.join(here, "models")))
    ap.add_argument("--k", type=int, default=4, choices=[4, 6, 8])
    ap.add_argument("--example", default=os.path.join(here, "example"))
    ap.add_argument("--report", default=f"invtax_test_report_{datetime.date.today():%Y%m%d}.tsv")
    a = ap.parse_args()

    truth = pd.read_csv(os.path.join(a.example, "example_truth.tsv"), sep="\t", dtype=str, keep_default_na=False)
    rows, ok = [], True
    with tempfile.TemporaryDirectory() as tmp:
        for m in sorted(truth.marker.unique()):
            fa = os.path.join(a.example, f"example_{m}.fasta")
            out = os.path.join(tmp, m)
            cmd = [sys.executable, os.path.join(here, "invtax.py"), "--fasta", fa, "--marker", m,
                   "--k", str(a.k), "--db", a.db, "--out", out, "--no_cutoffs"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout, r.stderr)
                sys.exit(f"FAIL: invtax.py did not run for {m}")
            p = pd.read_csv(glob.glob(os.path.join(out, "*.tsv"))[0], sep="\t", dtype=str, keep_default_na=False)
            d = p.merge(truth[truth.marker == m], on="seq_id", suffixes=("", "_true"))
            for rk in R:
                if rk + "_true" not in d:
                    continue
                known = ~d[rk + "_true"].isin(EMPTY)
                acc = (d.loc[known, rk] == d.loc[known, rk + "_true"]).mean() if known.any() else float("nan")
                need = PASS_RULES.get(rk)
                good = need is None or acc >= need
                ok &= good
                rows.append({"marker": m, "rank": rk, "n": int(known.sum()), "accuracy": round(acc, 3),
                             "required": need if need else "", "status": "" if need is None else
                             ("pass" if good else "FAIL")})
    rep = pd.DataFrame(rows)
    rep.to_csv(a.report, sep="\t", index=False)
    print(rep.pivot(index="rank", columns="marker", values="accuracy").reindex(R).to_string())
    print(f"\nreport: {a.report}")
    print("InvTax test: PASS" if ok else "InvTax test: FAIL (check the installation and model files)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
