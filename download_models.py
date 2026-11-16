#!/usr/bin/env python3
"""
Download the InvTax trained models from Zenodo into models/ (next to this script).

    python download_models.py
"""
import hashlib, os, sys, tarfile, urllib.request

ZENODO_RECORD = "24289250"                                  # InvTax models, doi:10.5281/zenodo.24289250
MODEL_FILE = "invtax_models_v1_20261116.tar.gz"
MODEL_MD5 = "PUT_MD5_HERE"                                  # <-- md5sum of MODEL_FILE
URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{MODEL_FILE}?download=1"


def progress(n, bs, total):
    if total > 0:
        done = min(n * bs, total)
        sys.stdout.write(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB")
        sys.stdout.flush()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.join(here, "models")
    if os.path.exists(os.path.join(dest, "hybrid_k4", "best.pt")):
        print(f"models already present in {dest}")
        return
    tgz = os.path.join(here, MODEL_FILE)
    if not os.path.exists(tgz):
        print(f"downloading {MODEL_FILE} from Zenodo record {ZENODO_RECORD} (about 1.1 GB)")
        urllib.request.urlretrieve(URL, tgz, progress)
        print()
    if MODEL_MD5 != "PUT_MD5_HERE":
        got = md5(tgz)
        if got != MODEL_MD5:
            os.remove(tgz)
            sys.exit(f"checksum mismatch ({got}); the download was removed, please run again")
    print("extracting")
    with tarfile.open(tgz) as t:
        try:
            t.extractall(here, filter="data")
        except TypeError:                                     # Python < 3.12
            t.extractall(here)
    os.remove(tgz)
    print(f"InvTax models ready in {dest}")


if __name__ == "__main__":
    main()
