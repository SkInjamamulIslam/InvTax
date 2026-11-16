#!/usr/bin/env python3
"""
InvTax: deep learning taxonomic classifier for invertebrate COI, 16S and 18S amplicon sequences
(Hybrid convolution + attention model, k-mer tokens).

Usage
    python invtax.py --fasta my_asvs.fasta --marker COI --out results
    python invtax.py --fasta fasta_folder/ --marker 16S --out results      # every FASTA in a folder

Output (in --out): <fasta name>_InvTax_<marker>_k<k>_<YYYYMMDD>.tsv and .csv, one row per sequence:
    seq_id, marker, kingdom, kingdom_prob, phylum, phylum_prob, class, class_prob, order, order_prob,
    family, family_prob, genus, genus_prob, species, species_prob

--marker restricts every rank to the taxa the InvTax reference database holds for that marker.
Lineages are decoded top down (each rank chosen only among children of the rank above), so every row
is a consistent lineage. By default (FISHNET-META style) the first rank whose probability is below its cutoff
(kingdom 0.90, phylum 0.80, class 0.70, order 0.60, family 0.50, genus 0.40, species 0.30) and every rank
below it are reported as 'unclassified'; the probabilities are always kept. Change with --cutoffs, or
use --no_cutoffs for the raw best taxon at every rank.
"""
import argparse, csv, glob, gzip, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

R = ["phylum", "class", "order", "family", "genus", "species"]
MARKERS = ["COI", "16S", "18S"]
NSPEC, PAD, UNK = 2, 0, 1          # token 0 = padding, 1 = unknown k-mer (contains N), k-mers start at 2
MAXTOK = 1024                      # 512 positions after the 2x downsampling convolution

# ---------------------------------------------------------------- phylum -> kingdom
METAZOA = {"Acanthocephala", "Annelida", "Arthropoda", "Brachiopoda", "Bryozoa", "Chaetognatha", "Chordata",
           "Cnidaria", "Ctenophora", "Cycliophora", "Dicyemida", "Echinodermata", "Echiura", "Entoprocta",
           "Gastrotricha", "Gnathostomulida", "Hemichordata", "Kinorhyncha", "Loricifera", "Micrognathozoa",
           "Mollusca", "Myxozoa", "Nematoda", "Nematomorpha", "Nemertea", "Onychophora", "Orthonectida",
           "Pentastomida", "Phoronida", "Placozoa", "Platyhelminthes", "Pogonophora", "Porifera", "Priapulida",
           "Rhombozoa", "Rotifera", "Sipuncula", "Tardigrada", "Xenacoelomorpha"}
PLANTS = {"Anthophyta", "Bryophyta", "Charophyta", "Chlorophyta", "Marchantiophyta", "Prasinodermophyta",
          "Streptophyta", "Tracheophyta"}
NOT_FUNGI = {"Oomycota", "Hyphochytriomycota", "Labyrinthulomycota", "Myxomycota", "Acrasiomycota",
             "Plasmodiophoromycota"}
OLD_BACTERIA = {"Firmicutes", "Tenericutes", "Proteobacteria", "Chloroflexi", "Spirochaetes", "Thermotogae",
                "Deinococcus-Thermus", "Chlamydiae", "Fusobacteria", "Nitrospirae", "Elusimicrobia", "Chlorobi",
                "Aquificae", "Synergistetes", "Dictyoglomi", "Chrysiogenetes", "Thermodesulfobacteria",
                "Gemmatimonadetes", "Fibrobacteres", "Lentisphaerae", "Verrucomicrobia", "Planctomycetes",
                "Acidobacteria", "Actinobacteria", "Cyanobacteria", "Bacteroidetes", "Deferribacteres",
                "Ignavibacteriae", "Armatimonadetes", "Caldiserica", "Coprothermobacterota"}


def kingdom_of(ph):
    p = ph.replace("Candidatus ", "").strip()
    if p in ("", "NA", "unclassified"):
        return "unclassified"
    if p in METAZOA:
        return "Metazoa"
    if p in PLANTS:
        return "Viridiplantae"
    if p in NOT_FUNGI:
        return "Protista"
    if p == "Microsporidia" or p.endswith("mycota"):
        return "Fungi"
    if "archaeota" in p.lower() or p in {"Thermoproteota", "Nitrososphaerota", "Methanobacteriota",
                                         "Halobacteriota", "Thermoplasmatota"}:
        return "Archaea"
    if p in OLD_BACTERIA or p.endswith("ota") or p.endswith("bacteria") or p.endswith("etes"):
        return "Bacteria"
    return "Protista"

# ---------------------------------------------------------------- model (identical to training)
class ConvBlock(nn.Module):
    def __init__(self, d, ks, stride, drop):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, ks, stride, ks // 2)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.stride = stride

    def forward(self, x, valid):
        h = self.norm(x) * valid.unsqueeze(-1).to(x.dtype)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.drop(self.proj(F.gelu(h)))
        if self.stride > 1:
            x = F.avg_pool1d(x.transpose(1, 2), self.stride, self.stride, ceil_mode=True).transpose(1, 2)
            valid = valid[:, ::self.stride]
        return x[:, :h.size(1)] + h, valid[:, :h.size(1)]


class Hybrid(nn.Module):
    def __init__(self, vocab, heads, d, ks, layers, nhead, drop):
        super().__init__()
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD)
        self.convs = nn.ModuleList([ConvBlock(d, ks, 1, drop), ConvBlock(d, ks, 2, drop), ConvBlock(d, ks, 1, drop)])
        self.pos = nn.Embedding(512, d)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, drop, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.rank = nn.ModuleDict({r: nn.Linear(d, n) for r, n in heads.items()})
        self.marker = nn.Linear(d, 3)

    def forward(self, x):
        valid = x != PAD
        h = self.tok(x)
        for c in self.convs:
            h, valid = c(h, valid)
        h = h + self.pos(torch.arange(h.size(1), device=h.device))[None]
        h = self.norm(self.enc(h, src_key_padding_mask=~valid))
        m = valid.unsqueeze(-1).to(h.dtype)
        z = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return {r: f(z) for r, f in self.rank.items()}, self.marker(z)

# ---------------------------------------------------------------- input
def read_fasta(path):
    op = gzip.open if path.endswith(".gz") else open
    ids, seqs, sid, buf = [], [], None, []
    with op(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sid is not None:
                    ids.append(sid); seqs.append("".join(buf))
                sid, buf = line[1:].split()[0], []
            else:
                buf.append(line)
    if sid is not None:
        ids.append(sid); seqs.append("".join(buf))
    return ids, seqs


LUT = np.full(256, 4, np.uint8)
for _i, _b in enumerate(b"ACGT"):
    LUT[_b] = _i
LUT[ord("U")] = 3


def tokenize(seq, k, pow_):
    c = LUT[np.frombuffer(seq.upper().encode("ascii", "replace"), np.uint8)].astype(np.int64)
    if len(c) < k:
        return np.array([UNK], np.int64)
    w = np.lib.stride_tricks.sliding_window_view(c, k)
    ids = w @ pow_ + NSPEC
    ids[(w == 4).any(1)] = UNK
    return ids[:MAXTOK]


# ---------------------------------------------------------------- classifier
class InvTax:
    def __init__(self, db, k=4, device="auto"):
        need = [os.path.join(db, "label_vocab.json"), os.path.join(db, "taxonomy.json"),
                os.path.join(db, f"hybrid_k{k}", "best.pt"), os.path.join(db, f"hybrid_k{k}", "species_head_ids.npy")]
        miss = [f for f in need if not os.path.exists(f)]
        mfiles = sorted(glob.glob(os.path.join(db, "marker_masks*.npz")))
        if miss or not mfiles:
            sys.exit("InvTax model files not found in " + db + "\n  missing: " + ", ".join(miss or ["marker_masks*.npz"]) +
                     "\n  run:  python download_models.py   (or set --db to the model folder)")
        self.k = k
        self.dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else
                                ("cpu" if device == "auto" else device))
        vocab = json.load(open(need[0]))
        tax = json.load(open(need[1]))
        sid = np.load(need[3])
        masks = np.load(mfiles[-1])

        ck = torch.load(need[2], map_location="cpu", weights_only=False)
        ar, heads, sd = ck["args"], ck["heads"], ck["model"]
        nvoc = sd["tok.weight"].shape[0]
        if nvoc != 4 ** k + NSPEC:
            sys.exit(f"token table {nvoc} does not match k={k}")
        self.model = Hybrid(nvoc, heads, ar["d"], ar["conv_kernel"], ar["attn_layers"], ar["heads"], ar["dropout"])
        self.model.load_state_dict({kk.replace("_orig_mod.", ""): v for kk, v in sd.items()})
        self.model.to(self.dev).eval()

        self.names = {r: [s.split("|")[0] for s in vocab[r]] for r in R[:-1]}
        self.names["species"] = [vocab["species"][i].split("|")[0] for i in sid]
        parent = {r: np.asarray(tax["parent"][r], np.int64) for r in R[1:]}
        parent["species"] = parent["species"][sid]
        self.par = {r: torch.as_tensor(parent[r], device=self.dev) for r in R[1:]}
        self.mask = {r: torch.as_tensor(np.stack([masks[f"{m}_{r}"] for m in MARKERS]), device=self.dev) for r in R}
        # phylum and class of every taxon (from the label keys "name|phylum|class"), used to keep
        # taxa without a recorded parent in the reference inside the chosen phylum and class
        ph = self.names["phylum"]
        ph_ix = {p: i for i, p in enumerate(ph)}
        cl_ix = {}
        for i, s in enumerate(vocab["class"]):
            f = s.split("|")
            cl_ix[(f[0], f[1] if len(f) > 1 else "NA")] = i
        keys = {r: vocab[r] for r in R[1:-1]}
        keys["species"] = [vocab["species"][i] for i in sid]
        self.tph, self.tcl = {}, {}
        for r in R[1:]:
            fs = [s.split("|") for s in keys[r]]
            self.tph[r] = torch.as_tensor([ph_ix.get(f[1], -1) if len(f) > 1 else -1 for f in fs], device=self.dev)
            if r != "class":
                self.tcl[r] = torch.as_tensor([cl_ix.get((f[2], f[1]), -1) if len(f) > 2 else -1 for f in fs],
                                              device=self.dev)
        self.kings = sorted({kingdom_of(p) for p in ph})
        self.kid = torch.as_tensor([self.kings.index(kingdom_of(p)) for p in ph], device=self.dev)
        self.pow = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)

    @torch.no_grad()
    def classify(self, seqs, marker, batch=128):
        """Return per-sequence lineage dict, used marker and the model's own marker call."""
        n = len(seqs)
        toks = [tokenize(s, self.k, self.pow) for s in seqs]
        order = np.argsort([len(t) for t in toks], kind="stable")
        RK = ["kingdom"] + R
        name = {r: ["unclassified"] * n for r in RK}
        prob = {r: np.full(n, np.nan) for r in RK}
        used = [None] * n
        own = np.zeros(n, np.int64)
        fixed = None if marker == "auto" else MARKERS.index(marker)
        for b0 in range(0, n, batch):
            idx = order[b0:b0 + batch]
            x = np.zeros((len(idx), max(len(toks[i]) for i in idx)), np.int64)
            for j, i in enumerate(idx):
                x[j, :len(toks[i])] = toks[i]
            out, mk = self.model(torch.from_numpy(x).to(self.dev))
            mpred = mk.argmax(1)
            msel = torch.full_like(mpred, fixed) if fixed is not None else mpred
            own[idx] = mpred.cpu().numpy()
            prev, alive = None, torch.ones(len(idx), dtype=torch.bool, device=self.dev)
            chosen, orphan_win = {}, {}
            for r in R:
                p = F.softmax(out[r].float(), 1)
                allow = self.mask[r][msel]                                # taxa of this marker
                if prev is None:
                    best = torch.where(allow, p, torch.zeros_like(p)).argmax(1)
                else:
                    pr_ = self.par[r][None, :]
                    proper = allow & (pr_ == prev[:, None])               # children of the chosen parent
                    # taxa with no recorded parent: allowed only inside the chosen phylum (and class)
                    tp = self.tph[r][None, :]
                    orphan = allow & (pr_ < 0) & ((tp == chosen["phylum"][:, None]) | (tp < 0))
                    if r in self.tcl and "class" in chosen:
                        tc = self.tcl[r][None, :]
                        orphan &= (tc == chosen["class"][:, None]) | (tc < 0)
                    qp = torch.where(proper, p, torch.zeros_like(p))
                    qo = torch.where(orphan, p, torch.zeros_like(p))
                    ow = qo.max(1).values > qp.max(1).values
                    if r == "class":                                      # never override the phylum
                        ow &= ~proper.any(1)
                    best = torch.where(ow, qo.argmax(1), qp.argmax(1))
                    alive = alive & (proper | orphan).any(1)
                    orphan_win[r] = ow.cpu().numpy()
                chosen[r] = best
                bp = p.gather(1, best[:, None])[:, 0]
                if r == "phylum":                                         # kingdom = summed phylum prob
                    kp = torch.zeros(len(idx), len(self.kings), device=self.dev).index_add_(1, self.kid, p)
                    kb = self.kid[best]
                    kpr = kp.gather(1, kb[:, None])[:, 0].cpu().numpy()
                    kb = kb.cpu().numpy()
                prev = best
                bi, bpn, al = best.cpu().numpy(), bp.cpu().numpy(), alive.cpu().numpy()
                for j, i in enumerate(idx):
                    if al[j]:
                        name[r][i], prob[r][i] = self.names[r][bi[j]], bpn[j]
                        if r == "phylum":
                            name["kingdom"][i], prob["kingdom"][i] = self.kings[kb[j]], kpr[j]
            # a taxon chosen without a recorded parent: the rank above is not defined in the reference
            for ri in range(2, len(R)):
                for j in np.flatnonzero(orphan_win[R[ri]]):
                    i = idx[j]
                    if name[R[ri]][i] != "unclassified" and name[R[ri - 1]][i] != "unclassified":
                        name[R[ri - 1]][i], prob[R[ri - 1]][i] = "NA", np.nan
            for j, i in enumerate(idx):
                used[i] = MARKERS[int(msel[j])]
        return name, prob, used, own


DEFAULT_CUTOFFS = "0.90,0.80,0.70,0.60,0.50,0.40,0.30"   # kingdom, phylum, class, order, family, genus, species


def apply_cutoffs(name, prob, cutoffs):
    """FISHNET-META style: the first rank whose probability is below its cutoff, and every rank
    below it, is reported as 'unclassified' (the probability is kept). Ranks not defined in the
    reference ('NA') are skipped."""
    RK = ["kingdom"] + R
    n = len(name["kingdom"])
    for i in range(n):
        cut = False
        for r, c in zip(RK, cutoffs):
            if name[r][i] == "NA" and not cut:
                continue
            if cut or name[r][i] == "unclassified" or not (prob[r][i] >= c):
                cut = True
                name[r][i] = "unclassified"
    return name


def write_tables(base, ids, name, prob, used):
    RK = ["kingdom"] + R
    header = ["seq_id", "marker"] + [c for r in RK for c in (r, r + "_prob")]
    rows = []
    for i, s in enumerate(ids):
        row = [s, used[i]]
        for r in RK:
            row += [name[r][i], "" if np.isnan(prob[r][i]) else f"{prob[r][i]:.4f}"]
        rows.append(row)
    for ext, dl in ((".tsv", "\t"), (".csv", ",")):
        with open(base + ext, "w", newline="") as fh:
            w = csv.writer(fh, delimiter=dl, lineterminator="\n")
            w.writerow(header); w.writerows(rows)


FA_EXT = (".fasta", ".fa", ".fna", ".fas", ".fasta.gz", ".fa.gz", ".fna.gz", ".fas.gz")

# ---------------------------------------------------------------- command line
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="InvTax: classify invertebrate COI / 16S / 18S amplicon sequences.",
        epilog="example:  python invtax.py --fasta asv.fasta --marker COI --out results")
    ap.add_argument("--fasta", help="input FASTA file (plain or .gz), or a folder of FASTA files")
    ap.add_argument("--marker", choices=MARKERS + ["auto"],
                    help="marker of the reads: COI, 16S or 18S (restricts predictions to that marker's "
                         "reference taxa); auto = model decides per sequence")
    ap.add_argument("--out", help="output folder (created if needed)")
    ap.add_argument("--k", type=int, default=4, choices=[4, 6, 8], help="model k-mer size (default 4)")
    ap.add_argument("--db", default=os.environ.get("INVTAX_DB", os.path.join(here, "models")),
                    help="model folder (default: models/ next to this script, or $INVTAX_DB)")
    ap.add_argument("--cutoffs", default=DEFAULT_CUTOFFS,
                    help="minimum probability for kingdom,phylum,class,order,family,genus,species; "
                         "the first rank below its cutoff and all ranks below it are 'unclassified' "
                         f"(default {DEFAULT_CUTOFFS})")
    ap.add_argument("--no_cutoffs", action="store_true",
                    help="report the best taxon at every rank with its raw probability (no 'unclassified')")
    ap.add_argument("--batch", type=int, default=128, help="sequences per batch (default 128)")
    ap.add_argument("--device", default="auto", help="auto (GPU if available), cpu or cuda")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
    ap.add_argument("--list_kingdoms", action="store_true", help="print the phylum to kingdom map and exit")
    a = ap.parse_args()

    if a.list_kingdoms:
        vocab = json.load(open(os.path.join(a.db, "label_vocab.json")))
        for p in sorted({s.split("|")[0] for s in vocab["phylum"]}, key=lambda x: (kingdom_of(x), x)):
            print(f"{kingdom_of(p)}\t{p}")
        return
    if not (a.fasta and a.marker and a.out):
        ap.error("--fasta, --marker and --out are required")
    if os.path.isdir(a.fasta):
        files = sorted(f for f in glob.glob(os.path.join(a.fasta, "*")) if f.lower().endswith(FA_EXT))
        if not files:
            sys.exit(f"no FASTA files ({', '.join(FA_EXT)}) in {a.fasta}")
    elif os.path.exists(a.fasta):
        files = [a.fasta]
    else:
        sys.exit(f"FASTA not found: {a.fasta}")
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    cutoffs = [float(x) for x in a.cutoffs.split(",")]
    if len(cutoffs) != 7:
        sys.exit("--cutoffs needs 7 values: kingdom,phylum,class,order,family,genus,species")
    clf = InvTax(a.db, a.k, a.device)
    stamp = time.strftime("%Y%m%d")
    print(f"InvTax hybrid k{a.k} | marker {a.marker} | device {clf.dev} | {len(files)} FASTA file(s) | "
          + ("no cutoffs" if a.no_cutoffs else f"cutoffs {a.cutoffs}"), flush=True)
    for fa in files:
        ids, seqs = read_fasta(fa)
        if not ids:
            print(f"skipped (no sequences): {fa}")
            continue
        name, prob, used, own = clf.classify(seqs, a.marker, a.batch)
        if not a.no_cutoffs:
            name = apply_cutoffs(name, prob, cutoffs)
        stem = os.path.basename(fa)
        for e in sorted(FA_EXT, key=len, reverse=True):
            if stem.lower().endswith(e):
                stem = stem[:-len(e)]; break
        base = os.path.join(a.out, f"{stem}_InvTax_{a.marker}_k{a.k}_{stamp}")
        write_tables(base, ids, name, prob, used)

        print(f"\n{os.path.basename(fa)}: {len(ids):,} sequences")
        ntr = sum(len(s) - a.k + 1 > MAXTOK for s in seqs)
        if ntr:
            print(f"  note: {ntr} sequences longer than {MAXTOK + a.k - 1} bp were truncated")
        if a.marker != "auto":
            agree = (own == MARKERS.index(a.marker)).mean()
            print(f"  model's own marker call agrees with --marker {a.marker} for {agree:.1%} of sequences")
        kc = {k: name["kingdom"].count(k) for k in clf.kings + ["unclassified"] if name["kingdom"].count(k)}
        print("  kingdom  " + " | ".join(f"{k} {v:,}" for k, v in kc.items()))
        print("  classified: " + " | ".join(
            f"{r} {sum(x not in ('unclassified', 'NA') for x in name[r]):,}" for r in ["kingdom"] + R))
        print(f"  written: {base}.tsv and .csv")
    print(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
