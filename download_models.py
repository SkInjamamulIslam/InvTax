#!/usr/bin/env python3
"""
Download the InvTax trained models from Zenodo.

    python download_models.py                  # InvTax v2 (recommended)  -> models/
    python download_models.py --version v1     # InvTax v1 (paper benchmark; hybrid k4, k6, k8) -> models_v1/
    python download_models.py --refs           # also the primer region references used by invtax_refcheck.py -> refs/

The files are taken from the latest version of the Zenodo record (doi:10.5281/zenodo.24289250) and checked
against the MD5 sums published with them.
"""
import argparse, hashlib, json, os, shutil, sys, tarfile, tempfile, urllib.request

RECORD = "24289250"                     # any version of the InvTax Zenodo record; the latest version is looked up
FILES = {"v2": "invtax_models_v2.tar.gz", "v1": "invtax_models_v1_20261116.tar.gz"}
REFS = "invtax_region_refs_v2.tar.gz"
MD5S = "MD5SUMS_v2.txt"
API = "https://zenodo.org/api/records/{}/versions/latest"


def progress(n, bs, total):
    if total > 0:
        sys.stdout.write(f"\r  {min(n * bs, total) / 1e6:,.0f} / {total / 1e6:,.0f} MB"); sys.stdout.flush()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def latest_id():
    """id of the latest version of the record (falls back to RECORD if the API cannot be reached)"""
    try:
        with urllib.request.urlopen(API.format(RECORD), timeout=60) as r:
            rec = json.load(r)
        print(f"Zenodo record {rec.get('id')} (version {rec.get('metadata', {}).get('version', '?')})")
        return str(rec.get("id", RECORD))
    except Exception as e:
        print(f"could not look up the latest Zenodo version ({e}); using record {RECORD}")
        return RECORD


def fetch(name, rid, here, sums):
    dst = os.path.join(here, name)
    if not os.path.exists(dst):
        print(f"downloading {name}")
        url = f"https://zenodo.org/records/{rid}/files/{name}?download=1"
        try:
            urllib.request.urlretrieve(url, dst, progress); print()
        except Exception as e:
            if os.path.exists(dst):
                os.remove(dst)
            sys.exit(f"could not download {name} ({e}); get it by hand from https://doi.org/10.5281/zenodo.{RECORD}")
    if name in sums and md5(dst) != sums[name]:
        os.remove(dst); sys.exit(f"checksum mismatch for {name}; the download was removed, please run again")
    return dst


def unpack(tgz, here, target, inner):
    """extract the archive (which holds the folder `inner`) and place that folder at `target`"""
    tmp = tempfile.mkdtemp(dir=here)
    with tarfile.open(tgz) as t:
        try:
            t.extractall(tmp, filter="data")
        except TypeError:                                   # Python < 3.12
            t.extractall(tmp)
    src = os.path.join(tmp, inner)
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.move(src, target); shutil.rmtree(tmp); os.remove(tgz)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", choices=["v2", "v1"], default="v2")
    ap.add_argument("--refs", action="store_true", help="also download the primer region references")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.join(here, "models" if a.version == "v2" else "models_v1")
    need_model = not os.path.exists(os.path.join(dest, "hybrid_k4", "best.pt"))
    need_refs = a.refs and not os.path.isdir(os.path.join(here, "refs"))
    if not need_model and not need_refs:
        print(f"already present: {dest}" + (" and refs/" if a.refs else "")); return
    rid = latest_id()
    p = fetch(MD5S, rid, here, {})
    sums = {l.split()[1].lstrip("*"): l.split()[0] for l in open(p) if len(l.split()) == 2}
    os.remove(p)
    if need_model:
        unpack(fetch(FILES[a.version], rid, here, sums), here, dest, "models")
        print(f"InvTax {a.version} models ready in {dest}")
    if need_refs:
        unpack(fetch(REFS, rid, here, sums), here, os.path.join(here, "refs"), "refs")
        print(f"primer region references ready in {os.path.join(here, 'refs')}")


if __name__ == "__main__":
    main()
