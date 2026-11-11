"""Download MIDV-2020 templates + photos (~4.9 GB) from the official FTP and verify MD5.

Source: ftp://smartengines.com/midv-2020 (mirror: http://l3i-share.univ-lr.fr, form-gated).
License: CC BY-SA 2.5; faces from Generated Photos (attribution encouraged).
Citation: K. Bulatov et al., "MIDV-2020: A Comprehensive Benchmark Dataset for Identity Document Analysis",
Computer Optics 46(2), 2022. arXiv:2107.00396.

Usage: python scripts/download_midv2020.py [--parts templates photo]
Extracts into data/raw/midv2020/<part>/{images,annotations}.
"""

import argparse
import hashlib
import tarfile
import urllib.request

from ekyc.io import load_config, path

BASE = "ftp://smartengines.com/midv-2020/dataset/"
MD5 = {
    "templates": "5bf6bd22df4808d4c8df9500e01f4f0a",
    "photo": "e716b2044a0af872cb21c9dc2f51d752",
    "scan_upright": "f6e0aeb6981a89aa2dfc717575fcea28",
    "scan_rotated": "3db227b453121b7e0a8c152e79988edf",
}

ap = argparse.ArgumentParser()
ap.add_argument("--parts", nargs="+", default=["templates", "photo"], choices=list(MD5))
args = ap.parse_args()
root = path(load_config(), "midv")
root.mkdir(parents=True, exist_ok=True)

for part in args.parts:
    tar = root / f"{part}.tar"
    if not tar.exists():
        print("downloading", BASE + tar.name)
        urllib.request.urlretrieve(BASE + tar.name, tar.with_suffix(".part"))
        tar.with_suffix(".part").rename(tar)
    h = hashlib.md5()
    with open(tar, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != MD5[part]:
        raise SystemExit(f"MD5 mismatch for {tar}; delete it and retry")
    dest = root / part
    with tarfile.open(tar) as t:
        t.extractall(dest, filter="data")
    print("extracted", dest)
