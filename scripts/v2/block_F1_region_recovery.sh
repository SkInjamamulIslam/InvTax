# Block F1 — Field check v2 (20261125): BOLD region recovery, strand fix, InvTax rerun, identity floors
# Uses the same MK, RES, DIAG as the first field check; INVTAX = folder that holds invtax.py
set -euo pipefail
: "${MK:?set MK (Markers_files folder)}"; : "${RES:?set RES (InvTax field results)}"
: "${DIAG:?set DIAG (field check folder)}"; : "${INVTAX:?set INVTAX (folder with invtax.py)}"
DB=${DB:-/mnt/r/invertebrates_database/database_files}
IN=${IN:-$RES/input}
T=$(nproc)
V2=$DB/invtax_fieldcheck_v2_20261125
SETS="16S_EPP 16S_Tak COI_EPTD COI_FWH2 COI_MG2"
mkdir -p $V2/{scripts,full,seeds,pilot,hits,refs,otus/COI,otus/16S,invtax,final}
cat > $V2/scripts/fc2_20261125.py <<'PY'
#!/usr/bin/env python3
"""InvTax field check v2 (20261125): BOLD/NCBI region recovery, references v2, identity floors.
Subcommands: scan | seeds | extract | project | build | final"""
import sys, os, re, gzip, json, glob, random, hashlib
import numpy as np, pandas as pd

D = "20261125"
R = ["phylum", "class", "order", "family", "genus", "species"]
SETS = {"16S_EPP": ("16S", "P227_16S_Epp-degen"), "16S_Tak": ("16S", "MtInsects-16S"),
        "COI_EPTD": ("COI", "EPTDr2n_COI"), "COI_FWH2": ("COI", "fwhF2_COI"),
        "COI_MG2": ("COI", "P223_MG2-LCO1490_COI")}
# Identity floors per rank: lowest identity bin (Table S7, nearest training sequence) from which InvTax
# precision is >= 0.90 in that bin and every higher bin. Species never reached 0.90, so species needs >= 99%.
FLOOR = {"COI": dict(phylum=75, **{"class": 75}, order=75, family=90, genus=97, species=99),
         "16S": dict(phylum=75, **{"class": 75}, order=75, family=90, genus=99, species=99)}
DNA = re.compile(r"^[ACGTURYKMSWBDHVN]+$")
NCBI = re.compile(r"^([A-Z]{1,2}_?\d{5,8}|[A-Z]{4,6}\d{8,10})(\.\d+)?$")
BOLD = re.compile(r"BOLD|^[A-Z0-9]{2,}-[A-Z0-9]+$", re.I)   # BOLD process IDs, e.g. AAD-57892, BIOUG52693-F04
SEED = 42


def src(i):
    i = i.split("|")[0]
    return "NCBI" if NCBI.match(i) else ("BOLD" if BOLD.search(i) else "other")


def opn(p, m="rt"):
    return gzip.open(p, m, errors="ignore") if p.endswith(".gz") else open(p, m, errors="ignore")


def read_crabs(p):
    """CRABS table: id, taxid, (super)kingdom..., phylum, class, order, family, genus, species, sequence"""
    with opn(p) as f:
        for l in f:
            c = l.rstrip("\r\n").split("\t")
            if len(c) < 3:
                continue
            s = c[-1].strip().upper()
            if len(s) < 50 or not DNA.match(s):
                continue
            lin = [x.strip() for x in c[1:-1]][-6:]
            yield c[0].strip(), [""] * (6 - len(lin)) + lin, s


def read_fa(p):
    h, s = None, []
    with opn(p) as f:
        for l in f:
            l = l.rstrip("\r\n")
            if l.startswith(">"):
                if h is not None:
                    yield h, "".join(s)
                h, s = l[1:], []
            else:
                s.append(l.strip())
    if h is not None:
        yield h, "".join(s)


def ref_file(MK, pat):
    c = [p for p in sorted(glob.glob(os.path.join(MK, f"*{pat}*"))) if os.path.isfile(p)]
    if not c:
        sys.exit(f"no reference file matching *{pat}* in {MK}")
    return c[0]


def reservoir(it, k, rng):
    out = []
    for n, x in enumerate(it):
        if n < k:
            out.append(x)
        else:
            j = rng.randint(0, n)
            if j < k:
                out[j] = x
    return out


def deepest(row, cols):
    d = "none"
    for r, c in zip(R, cols):
        v = str(row[c]) if c in row else ""
        if v in ("", "nan", "unclassified", "NA"):
            break
        d = r
    return d


# ------------------------------------------------------------------ scan
def scan(DB, MK, DIAG, V2):
    skip = [os.path.abspath(x) for x in (MK, DIAG, V2)]
    cands = os.environ.get("FULL", "").split()
    for root, dirs, files in ([] if cands else os.walk(DB)):
        ab = os.path.abspath(root)
        if any(ab == s or ab.startswith(s + os.sep) for s in skip):
            dirs[:] = []
            continue
        if ab[len(os.path.abspath(DB)):].count(os.sep) >= 4:
            dirs[:] = []
        for fn in files:
            p = os.path.join(root, fn)
            if not re.search(r"\.(txt|tsv|crabs)(\.gz)?$", fn) or os.path.getsize(p) < float(os.environ.get("FC2_MINSIZE", 50e6)):
                continue
            if any(pat in fn for _, pat in SETS.values()):
                continue
            try:
                ok = next(read_crabs(p), None) is not None
            except Exception:
                ok = False
            if ok:
                cands.append(p)
    if not cands:
        sys.exit(f"no full CRABS reference file (>50 MB) found under {DB}; set FULL to the file(s)")
    rng = random.Random(SEED)
    seen, n_out, per_file = set(), 0, []
    orient, k = [], 0
    with open(f"{V2}/full/full_ref_{D}.fa", "w") as fa:
        for p in cands:
            cnt = dict(file=p, records=0, NCBI=0, BOLD=0, other=0, kept=0)
            for i, lin, s in read_crabs(p):
                cnt["records"] += 1
                so = src(i)
                cnt[so] += 1
                h = hashlib.md5(s.encode()).digest()
                if h in seen:
                    continue
                seen.add(h)
                cnt["kept"] += 1
                n_out += 1
                fa.write(f">{i}\t{so}\t{';'.join(lin)}\n{s}\n")
                if len(s) >= 400:          # reservoir of long sequences for strand orientation
                    if len(orient) < 100000:
                        orient.append((i, s))
                    else:
                        j = rng.randint(0, k)
                        if j < 100000:
                            orient[j] = (i, s)
                    k += 1
            per_file.append(cnt)
    with open(f"{V2}/full/orient_db_{D}.fa", "w") as f:
        for i, s in orient:
            f.write(f">{i}\n{s}\n")
    prim = []
    for st, (mk, pat) in SETS.items():
        p = ref_file(MK, pat)
        c = dict(set=st, file=os.path.basename(p), refs=0, NCBI=0, BOLD=0, other=0)
        L = []
        for i, lin, s in read_crabs(p):
            c["refs"] += 1
            c[src(i)] += 1
            if len(L) < 200000:
                L.append(len(s))
        c["median_len"] = int(np.median(L)) if L else 0
        prim.append(c)
    json.dump(dict(full_files=per_file, full_unique=n_out, primer_sets=prim), open(f"{V2}/scan_{D}.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print("\nFull reference files (before in silico PCR):")
    print(pd.DataFrame(per_file).to_string(index=False))
    print(f"unique full sequences written: {n_out:,}")
    print("\nPer primer reference files (after in silico PCR):")
    print(pd.DataFrame(prim).to_string(index=False))


# ------------------------------------------------------------------ seeds + pilot queries
def seeds(st, otu_fa, MK, V2):
    rng = random.Random(SEED)
    otus = list(read_fa(otu_fa))
    om = int(np.median([len(s) for _, s in otus]))
    ref = [(i, s) for i, _, s in reservoir(read_crabs(ref_file(MK, SETS[st][1])), 30000, rng)]
    rm = int(np.median([len(s) for _, s in ref])) if ref else 0
    mismatch = abs(rm - om) / om > 0.25
    with open(f"{V2}/seeds/{st}.fa", "w") as f:
        for h, s in otus:
            f.write(f">otu_{h.split()[0]}\n{s}\n")
        if not mismatch:
            for i, s in ref:
                f.write(f">ref_{i}\n{s}\n")
    meta = dict(set=st, otu_median=om, ref_median=rm, ref_window_mismatch=bool(mismatch),
                lo=int(0.8 * om), hi=int(1.2 * om))
    json.dump(meta, open(f"{V2}/seeds/{st}.json", "w"))
    pil = reservoir(read_fa(f"{V2}/full/full_ref_{D}.fa"), 20000, rng)
    with open(f"{V2}/pilot/{st}.q.fa", "w") as f:
        for h, s in pil:
            f.write(f">{h}\n{s}\n")
    print(f"{st}: OTU median {om} bp, old reference median {rm} bp"
          + ("  -> old reference is a different window, seeds = field OTUs only" if mismatch else ""))


# ------------------------------------------------------------------ orient OTUs (works on any vsearch version)
def orient(otu_fa, strand_tsv, out_fa, out_tsv):
    st = {}
    if os.path.getsize(strand_tsv):
        for l in open(strand_tsv):
            q, s = l.rstrip("\n").split("\t")[:2]
            st.setdefault(q, s)
    comp = str.maketrans("ACGTRYKMBDHVN", "TGCAYRMKVHDBN")
    n = 0
    with open(out_fa, "w") as o, open(out_tsv, "w") as t:
        for h, s in read_fa(otu_fa):
            q = h.split()[0]
            k = st.get(q, "?")
            if k == "-":
                s = s.upper().translate(comp)[::-1]
                n += 1
            o.write(f">{h}\n{s}\n")
            t.write(f"{q}\t{k}\n")
    print(f"{os.path.basename(otu_fa)}: {n} OTUs flipped to forward, {sum(1 for v in st.values() if v == '+')} already forward")


# ------------------------------------------------------------------ extract windows
def extract(st, hits, fa, out, V2):
    meta = json.load(open(f"{V2}/seeds/{st}.json"))
    best = {}
    with open(hits) as f:
        for l in f:
            q, t, idv, qlo, qhi, tlo, thi, tl = l.rstrip("\n").split("\t")[:8]
            idv, qlo, qhi, tlo, thi, tl = float(idv), int(qlo), int(qhi), int(tlo), int(thi), int(tl)
            L = qhi - qlo + 1
            if idv < 75 or (thi - tlo + 1) / tl < 0.9 or not meta["lo"] <= L <= meta["hi"]:
                continue
            if q not in best or idv > best[q][2]:
                best[q] = (qlo, qhi, idv)
    n = 0
    with open(out, "w") as o:
        for h, s in read_fa(fa):
            c = h.split("\t")
            b = best.get(c[0])
            if b:
                o.write(f"{c[0]}\t{src(c[0])}\t{c[2] if len(c) > 2 else ''}\t{b[2]:.1f}\t{s[b[0]-1:b[1]]}\n")
                n += 1
    print(f"{st}: {n:,} windows recovered")
    return n


def project(st, secs, V2):
    n = sum(1 for _ in open(f"{V2}/pilot/{st}.win.tsv"))
    tot = json.load(open(f"{V2}/scan_{D}.json"))["full_unique"]
    print(f"{st} pilot: {n:,}/20,000 full sequences gave a window ({100*n/20000:.1f}%), "
          f"{secs}s -> full run about {secs*tot/20000/3600:.1f} h, about {int(n*tot/20000):,} windows")
    if n == 0:
        sys.exit(f"{st}: pilot recovered nothing; stopping before the full run")


# ------------------------------------------------------------------ build references v2
def build(MK, V2):
    rows, frag = [], []
    for st, (mk, pat) in SETS.items():
        meta = json.load(open(f"{V2}/seeds/{st}.json"))
        seen, sp_old, sp_new = set(), set(), set()
        c = dict(set=st, ref_v1=0, ref_v1_BOLD=0, kept_v1=0, recovered=0, recovered_BOLD=0, added=0)
        with open(f"{V2}/refs/{st}_ref_v2_{D}.tsv", "w") as t, open(f"{V2}/refs/{st}_ref_v2_{D}.fa", "w") as fa:
            t.write("id\tsource\torigin\t" + "\t".join(R) + "\tsequence\n")
            for i, lin, s in read_crabs(ref_file(MK, pat)):
                c["ref_v1"] += 1
                so = src(i)
                c["ref_v1_BOLD"] += so == "BOLD"
                sp_old.add(lin[5])
                if meta["ref_window_mismatch"]:
                    continue
                h = hashlib.md5((s + "|" + lin[5]).encode()).digest()
                if h in seen:
                    continue
                seen.add(h)
                c["kept_v1"] += 1
                t.write(f"{i}\t{so}\tv1\t" + "\t".join(lin) + f"\t{s}\n")
                fa.write(f">{i}\n{s}\n")
            for l in open(f"{V2}/full/{st}.win.tsv"):
                i, so, tax, idv, s = l.rstrip("\n").split("\t")
                lin = (tax.split(";") + [""] * 6)[:6]
                c["recovered"] += 1
                c["recovered_BOLD"] += so == "BOLD"
                h = hashlib.md5((s + "|" + lin[5]).encode()).digest()
                if h in seen:
                    continue
                seen.add(h)
                c["added"] += 1
                sp_new.add(lin[5])
                t.write(f"{i}\t{so}\trecovered\t" + "\t".join(lin) + f"\t{s}\n")
                fa.write(f">{i}\n{s}\n")
                frag.append((mk, st, i, so, *lin, s))
        c["ref_v2"] = c["kept_v1"] + c["added"]
        c["species_v1"] = len(sp_old - {""})
        c["species_v2"] = len(((sp_old if not meta["ref_window_mismatch"] else set()) | sp_new) - {""})
        rows.append(c)
    pd.DataFrame(frag, columns=["marker", "set", "id", "source", *R, "sequence"]).to_csv(
        f"{V2}/train_fragments_v5_{D}.tsv.gz", sep="\t", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(f"{V2}/refs/ref_v2_summary_{D}.tsv", sep="\t", index=False)
    print(df.to_string(index=False))


# ------------------------------------------------------------------ final: identity floors
def final(RES, DIAG, IN, V2):
    out, ex = [], []
    for st, (mk, pat) in SETS.items():
        stem = os.path.basename(glob.glob(f"{IN}/{st}*.fasta")[0])[:-6]
        new = pd.read_csv(glob.glob(f"{V2}/invtax/{stem}_InvTax_{mk}_k4_*.csv")[0], dtype=str)
        oldf = glob.glob(f"{RES}/{stem}_InvTax_{mk}_k4_*.csv")
        old = pd.read_csv(oldf[0], dtype=str) if oldf else None
        ori = pd.read_csv(f"{V2}/otus/{stem}_orient.tsv", sep="\t", header=None, dtype=str)
        flipped = int((ori.iloc[:, 1] == "-").sum())
        h2 = pd.read_csv(f"{V2}/hits/{st}_v2.tsv", sep="\t", header=None, names=["query", "target", "id"], dtype=str)
        h2["id"] = pd.to_numeric(h2["id"], errors="coerce").fillna(0)
        h2 = h2.sort_values("id", ascending=False).drop_duplicates("query")
        ref = pd.read_csv(f"{V2}/refs/{st}_ref_v2_{D}.tsv", sep="\t", dtype=str, usecols=["id", "species"])
        h2 = h2.merge(ref.drop_duplicates("id").rename(columns={"id": "target", "species": "nearest_ref_species"}), on="target", how="left")
        # v1 identity from the first field check (hits_<stem>.tsv: query, id, alnlen, qstrand)
        h1p = glob.glob(f"{DIAG}/hits_{stem}*.tsv") + glob.glob(f"{DIAG}/hits_{st}*.tsv")
        id1 = pd.Series(dtype=float)
        if h1p:
            h1 = pd.read_csv(h1p[0], sep="\t", header=None, dtype=str)
            if not re.match(r"^[\d.]+$", str(h1.iloc[0, 1])):
                h1 = h1.iloc[1:]
            h1["v"] = pd.to_numeric(h1.iloc[:, 1], errors="coerce")
            id1 = h1.groupby(h1.columns[0])["v"].max()
        new["seq_id"] = new["seq_id"].str.split().str[0]
        d = new.merge(h2.rename(columns={"query": "seq_id", "id": "nearest_ref_identity", "target": "nearest_ref_id"}),
                      on="seq_id", how="left")
        d["nearest_ref_identity"] = d["nearest_ref_identity"].fillna(0)
        fl = FLOOR[mk]
        for r in R:
            ok = (d["nearest_ref_identity"] >= fl[r]) & (d[r].fillna("unclassified") != "unclassified")
            prev = d[f"{R[R.index(r)-1]}_final"] != "unclassified" if r != "phylum" else True
            d[f"{r}_final"] = np.where(ok & prev, d[r], "unclassified")
        d["deepest_raw"] = d.apply(lambda x: deepest(x, R), axis=1)
        d["deepest_final"] = d.apply(lambda x: deepest(x, [f"{r}_final" for r in R]), axis=1)
        d.to_csv(f"{V2}/final/{stem}_InvTax_{mk}_k4_idfloor_{D}.csv", index=False)
        sp = pd.to_numeric(d["species_prob"], errors="coerce")
        raw_sp = d["species"].fillna("unclassified") != "unclassified"
        gone = d[raw_sp & (d["species_final"] == "unclassified")]
        o = dict(set=st, OTUs=len(d), flipped_to_forward=flipped,
                 id_median_v1=round(id1.median(), 1) if len(id1) else np.nan,
                 id_median_v2=round(d["nearest_ref_identity"].median(), 1),
                 pct97_v1=round(100 * (id1 >= 97).mean(), 1) if len(id1) else np.nan,
                 pct97_v2=round(100 * (d["nearest_ref_identity"] >= 97).mean(), 1))
        if old is not None:
            o["order_pct_first_run"] = round(100 * old.apply(lambda x: R.index(deepest(x, R)) >= 2 if deepest(x, R) != "none" else False, axis=1).mean(), 1)
        for nm, col in (("raw", "deepest_raw"), ("final", "deepest_final")):
            for r in ("order", "family", "genus", "species"):
                o[f"{r}_pct_{nm}"] = round(100 * d[col].map(lambda x: x != "none" and R.index(x) >= R.index(r)).mean(), 1)
        o["confident_far_raw"] = int((raw_sp & (sp >= 0.9) & (d["nearest_ref_identity"] < 97)).sum())
        o["species_removed"] = len(gone)
        out.append(o)
        if len(gone):
            g = gone.groupby("species").agg(OTUs=("seq_id", "size"), median_identity=("nearest_ref_identity", "median"),
                                             median_prob=("species_prob", lambda x: pd.to_numeric(x).median()),
                                             kept_as=("deepest_final", lambda x: x.mode().iat[0]))
            g = g.sort_values("OTUs", ascending=False).head(5).reset_index()
            g.insert(0, "set", st)
            ex.append(g)
    s = pd.DataFrame(out)
    s.to_csv(f"{V2}/final/fieldcheck_v2_summary_{D}.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(s.T.to_string())
    if ex:
        e = pd.concat(ex)
        e.to_csv(f"{V2}/final/species_removed_top_{D}.tsv", sep="\t", index=False)
        print("\nLargest species calls removed by the identity floor:")
        print(e.to_string(index=False))


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    {"scan": lambda: scan(*a), "orient": lambda: orient(*a), "seeds": lambda: seeds(*a), "extract": lambda: extract(*a),
     "project": lambda: project(a[0], int(a[1]), a[2]), "build": lambda: build(*a),
     "final": lambda: final(*a)}[c]()
PY
P="python3 $V2/scripts/fc2_20261125.py"
U=query+target+id+qilo+qihi+tilo+tihi+tl

echo "== 1. Find the full reference (NCBI + BOLD, before in silico PCR) and count BOLD per primer set"
if [ -s $V2/scan_20261125.json ] && [ -s $V2/full/orient_db_20261125.fa ]; then echo "already done, skipping"
else $P scan "$DB" "$MK" "$DIAG" "$V2"; fi

echo "== 2. Put every OTU on the forward strand"
for s in $SETS; do
  f=$(ls "$IN"/${s}*.fasta | head -1); stem=$(basename "$f" .fasta); mk=${s%%_*}
  vsearch --usearch_global "$f" --db $V2/full/orient_db_20261125.fa --id 0.6 --strand both --maxaccepts 1 \
          --threads $T --userfields query+qstrand --userout $V2/otus/${stem}_strand.tsv --quiet
  $P orient "$f" $V2/otus/${stem}_strand.tsv $V2/otus/$mk/$stem.fasta $V2/otus/${stem}_orient.tsv
done

echo "== 3. Pilot (20,000 full sequences per primer set), then the full region recovery"
for s in $SETS; do
  f=$(ls $V2/otus/*/${s}*.fasta | head -1)
  $P seeds $s "$f" "$MK" "$V2"
  t0=$(date +%s)
  vsearch --usearch_global $V2/pilot/$s.q.fa --db $V2/seeds/$s.fa --id 0.75 --strand plus \
          --maxaccepts 1 --maxrejects 16 --threads $T --userfields $U --userout $V2/pilot/$s.hits.tsv --quiet
  $P extract $s $V2/pilot/$s.hits.tsv $V2/pilot/$s.q.fa $V2/pilot/$s.win.tsv "$V2"
  $P project $s $(( $(date +%s) - t0 )) "$V2"
done
for s in $SETS; do
  echo "full recovery: $s  (started $(date +%H:%M))"
  vsearch --usearch_global $V2/full/full_ref_20261125.fa --db $V2/seeds/$s.fa --id 0.75 --strand plus \
          --maxaccepts 1 --maxrejects 16 --threads $T --userfields $U --userout $V2/full/$s.hits.tsv --quiet
  $P extract $s $V2/full/$s.hits.tsv $V2/full/full_ref_20261125.fa $V2/full/$s.win.tsv "$V2"
done

echo "== 4. References v2 (old amplicons + recovered windows) and training fragments for InvTax v2"
$P build "$MK" "$V2"

echo "== 5. Nearest reference identity of every OTU against references v2"
for s in $SETS; do
  f=$(ls $V2/otus/*/${s}*.fasta | head -1)
  vsearch --usearch_global "$f" --db $V2/refs/${s}_ref_v2_20261125.fa --id 0.6 --strand plus --maxaccepts 16 \
          --maxrejects 64 --threads $T --userfields query+target+id --userout $V2/hits/${s}_v2.tsv --quiet
done

echo "== 6. InvTax on the oriented OTUs"
( cd "$INVTAX"
  python invtax.py --fasta $V2/otus/COI --marker COI --k 4 --out $V2/invtax
  python invtax.py --fasta $V2/otus/16S --marker 16S --k 4 --out $V2/invtax )

echo "== 7. Identity floors (Table S7) and summary"
$P final "$RES" "$DIAG" "$IN" "$V2"
echo "Outputs: $V2/final  |  references v2: $V2/refs  |  training fragments: $V2/train_fragments_v5_20261125.tsv.gz"
