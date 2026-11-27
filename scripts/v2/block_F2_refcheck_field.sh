# Block F2 — Field check v2 part 2 (20261125): strand from references v2, InvTax rerun, consensus rule, BOLD region flag
set -euo pipefail
: "${MK:?set MK}"; : "${RES:?set RES}"; : "${DIAG:?set DIAG}"; : "${INVTAX:?set INVTAX}"
DB=${DB:-/mnt/r/invertebrates_database/database_files}
IN=${IN:-$RES/input}
BOLDDIR=${BOLDDIR:-/mnt/r/invertebrates_database/crabs_testing}
T=$(nproc)
V2=$DB/invtax_fieldcheck_v2_20261125
SETS="16S_EPP 16S_Tak COI_EPTD COI_FWH2 COI_MG2"
cat > $V2/scripts/fc3_20261125.py <<'PY'
#!/usr/bin/env python3
"""InvTax field check v2, part 2 (20261125): strand from references v2, consensus rule, BOLD region flag.
Subcommands: orient | final"""
import sys, os, re, glob, json
import numpy as np, pandas as pd

D = "20261125"
R = ["phylum", "class", "order", "family", "genus", "species"]
SETS = {"16S_EPP": "16S", "16S_Tak": "16S", "COI_EPTD": "COI", "COI_FWH2": "COI", "COI_MG2": "COI"}
# strict floors: Table S7 bins where InvTax precision >= 0.90 (species never reached it -> 99%)
FLOOR = {"COI": dict(phylum=75, order=75, family=90, genus=97, species=99, **{"class": 75}),
         "16S": dict(phylum=75, order=75, family=90, genus=99, species=99, **{"class": 75})}
# consensus floors: genus/species also kept when InvTax and the nearest reference name the same taxon
AGREE = {"COI": dict(genus=92, species=97), "16S": dict(genus=95, species=97)}
ATL = {"nova scotia", "new brunswick", "prince edward island", "newfoundland and labrador", "newfoundland", "quebec"}
COMP = str.maketrans("ACGTRYKMBDHVN", "TGCAYRMKVHDBN")


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


def norm(x):
    return re.sub(r"[\s_]+", " ", str(x)).strip().lower()


def deepest(row, cols):
    d = "none"
    for r, c in zip(R, cols):
        if str(row[c]) in ("", "nan", "unclassified", "NA"):
            break
        d = r
    return d


def orient(otu_fa, hits, out_fa, out_tsv, out_id):
    """hits: query, target, id, qstrand from --strand both against references v2; best identity decides"""
    h = pd.read_csv(hits, sep="\t", header=None, names=["query", "target", "id", "qstrand"], dtype=str)
    h["id"] = pd.to_numeric(h["id"], errors="coerce")
    h = h.sort_values("id", ascending=False).drop_duplicates("query").set_index("query")
    n = 0
    with open(out_fa, "w") as o, open(out_tsv, "w") as t:
        for hd, s in read_fa(otu_fa):
            q = hd.split()[0]
            k = h.at[q, "qstrand"] if q in h.index and h.at[q, "id"] >= 80 else "?"
            if k == "-":
                s = s.upper().translate(COMP)[::-1]
                n += 1
            o.write(f">{hd}\n{s}\n")
            t.write(f"{q}\t{k}\n")
    h.reset_index()[["query", "target", "id"]].to_csv(out_id, sep="\t", header=False, index=False)
    print(f"{os.path.basename(otu_fa)}: {n} OTUs reverse complemented (best hit on minus strand at >= 80%)")


def bold_regions(paths):
    """species -> set of tags from the BOLD data packages (country, Atlantic province)"""
    sp = {}
    for p in paths:
        with open(p, errors="ignore") as f:
            head = f.readline().rstrip("\n").split("\t")
            col = {c.lower(): i for i, c in enumerate(head)}
            i_sp = col.get("species")
            i_co = next((col[c] for c in ("country/ocean", "country", "country_ocean") if c in col), None)
            i_pr = next((col[c] for c in ("province/state", "province", "province_state") if c in col), None)
            if i_sp is None or i_co is None:
                print(f"  skipped {os.path.basename(p)}: no species or country column")
                continue
            for l in f:
                c = l.rstrip("\n").split("\t")
                if len(c) <= max(i_sp, i_co):
                    continue
                s = norm(c[i_sp])
                if not s or s in ("none", "nan"):
                    continue
                co = c[i_co].strip().lower()
                pr = c[i_pr].strip().lower() if i_pr is not None and len(c) > i_pr else ""
                t = sp.setdefault(s, set())
                t.add("any")
                if co == "canada":
                    t.add("canada")
                    if pr in ATL:
                        t.add("atlantic")
                if co in ("canada", "united states", "usa", "united states of america"):
                    t.add("north_america")
    return sp


def region_tag(s, sp):
    t = sp.get(norm(s))
    if t is None:
        return "not_in_BOLD_downloads"
    return ("Atlantic_Canada" if "atlantic" in t else "Canada" if "canada" in t
            else "North_America" if "north_america" in t else "outside_North_America")


def final(RES, DIAG, IN, V2, BOLDDIR):
    bt = sorted(glob.glob(f"{BOLDDIR}/bold_*.tsv"))
    sp_reg = bold_regions(bt)
    print(f"BOLD data packages read: {len(bt)}; species with records: {len(sp_reg):,}")
    out, ex = [], []
    for st, mk in SETS.items():
        stem = os.path.basename(glob.glob(f"{IN}/{st}*.fasta")[0])[:-6]
        d = pd.read_csv(glob.glob(f"{V2}/invtax/{stem}_InvTax_{mk}_k4_*.csv")[0], dtype=str)
        d["seq_id"] = d["seq_id"].str.split().str[0]
        oldf = glob.glob(f"{RES}/{stem}_InvTax_{mk}_k4_*.csv")
        old = pd.read_csv(oldf[0], dtype=str) if oldf else None
        ori = pd.read_csv(f"{V2}/otus/{stem}_orient2.tsv", sep="\t", header=None, dtype=str)
        h = pd.read_csv(f"{V2}/hits/{st}_v2b.tsv", sep="\t", header=None, names=["seq_id", "nearest_ref_id", "nearest_ref_identity"], dtype=str)
        h["nearest_ref_identity"] = pd.to_numeric(h["nearest_ref_identity"], errors="coerce")
        ref = pd.read_csv(f"{V2}/refs/{st}_ref_v2_{D}.tsv", sep="\t", dtype=str, usecols=["id", "genus", "species"])
        ref = ref[ref["id"].isin(set(h["nearest_ref_id"]))].drop_duplicates("id")
        h = h.merge(ref.rename(columns={"id": "nearest_ref_id", "genus": "nearest_ref_genus", "species": "nearest_ref_species"}),
                    on="nearest_ref_id", how="left")
        d = d.merge(h, on="seq_id", how="left")
        d["nearest_ref_identity"] = d["nearest_ref_identity"].fillna(0)
        idv = d["nearest_ref_identity"]
        for r in R:
            lab = d[r].fillna("unclassified")
            ok = (idv >= FLOOR[mk][r])
            if r in AGREE[mk]:
                ok = ok | ((idv >= AGREE[mk][r]) & (lab.map(norm) == d[f"nearest_ref_{r}"].map(norm)))
            prev = d[f"{R[R.index(r)-1]}_final"] != "unclassified" if r != "phylum" else True
            d[f"{r}_final"] = np.where(ok & prev & (lab != "unclassified"), lab, "unclassified")
        d["species_region_BOLD"] = np.where(d["species_final"] != "unclassified",
                                            d["species_final"].map(lambda s: region_tag(s, sp_reg)), "")
        d["deepest_raw"] = d.apply(lambda x: deepest(x, R), axis=1)
        d["deepest_final"] = d.apply(lambda x: deepest(x, [f"{r}_final" for r in R]), axis=1)
        d.to_csv(f"{V2}/final/{stem}_InvTax_{mk}_k4_final_{D}.csv", index=False)
        at = lambda col, r: round(100 * d[col].map(lambda x: x != "none" and R.index(x) >= R.index(r)).mean(), 1)
        o = dict(set=st, OTUs=len(d), reverse_complemented=int((ori.iloc[:, 1] == "-").sum()),
                 id_median=round(idv.median(), 1), pct_id97=round(100 * (idv >= 97).mean(), 1))
        if old is not None:
            o["order_pct_first_run"] = round(100 * old.apply(lambda x: deepest(x, R) in R[2:], axis=1).mean(), 1)
        for nm, col in (("raw", "deepest_raw"), ("final", "deepest_final")):
            for r in ("order", "family", "genus", "species"):
                o[f"{r}_pct_{nm}"] = at(col, r)
        sf = d.loc[d["species_final"] != "unclassified", "species_region_BOLD"]
        o["species_final_OTUs"] = len(sf)
        o["species_final_unique"] = d.loc[d["species_final"] != "unclassified", "species_final"].nunique()
        for k in ("Atlantic_Canada", "Canada", "North_America", "outside_North_America", "not_in_BOLD_downloads"):
            o[f"sp_{k}_pct"] = round(100 * (sf == k).mean(), 1) if len(sf) else np.nan
        out.append(o)
        top = (d[d["species_final"] != "unclassified"].groupby("species_final")
               .agg(OTUs=("seq_id", "size"), median_identity=("nearest_ref_identity", "median"), region=("species_region_BOLD", "first"))
               .sort_values("OTUs", ascending=False).head(8).reset_index())
        top.insert(0, "set", st)
        ex.append(top)
    s = pd.DataFrame(out)
    s.to_csv(f"{V2}/final/fieldcheck_v2_final_summary_{D}.tsv", sep="\t", index=False)
    e = pd.concat(ex)
    e.to_csv(f"{V2}/final/species_final_top_{D}.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(s.T.to_string())
    print("\nMost frequent species kept after the identity and consensus rules:")
    print(e.to_string(index=False))


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    {"orient": lambda: orient(*a), "final": lambda: final(*a)}[c]()
PY
P="python3 $V2/scripts/fc3_20261125.py"

echo "== 1. Strand and nearest reference identity from references v2 (both strands, best hit decides)"
rm -f $V2/otus/COI/*.fasta $V2/otus/16S/*.fasta
for s in $SETS; do
  f=$(ls "$IN"/${s}*.fasta | head -1); stem=$(basename "$f" .fasta); mk=${s%%_*}
  vsearch --usearch_global "$f" --db $V2/refs/${s}_ref_v2_20261125.fa --id 0.6 --strand both --maxaccepts 16 \
          --maxrejects 64 --threads $T --userfields query+target+id+qstrand --userout $V2/hits/${s}_v2both.tsv --quiet
  $P orient "$f" $V2/hits/${s}_v2both.tsv $V2/otus/$mk/$stem.fasta $V2/otus/${stem}_orient2.tsv $V2/hits/${s}_v2b.tsv
done

echo "== 2. InvTax on the corrected OTUs"
( cd "$INVTAX"
  python invtax.py --fasta $V2/otus/COI --marker COI --k 4 --out $V2/invtax
  python invtax.py --fasta $V2/otus/16S --marker 16S --k 4 --out $V2/invtax )

echo "== 3. Identity floors + consensus rule + BOLD region of each species call"
$P final "$RES" "$DIAG" "$IN" "$V2" "$BOLDDIR"
echo "Per OTU tables: $V2/final/*_final_20261125.csv"
