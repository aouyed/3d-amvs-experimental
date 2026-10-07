#!/usr/bin/env python3
"""Generate CLIMCAPS L2 granule manifests for July 1 2020 (SNPP + JPSS-1).

Queries the NASA CMR (Common Metadata Repository) for CLIMCAPS L2 (CrIS)
standard (CPS) granules overlapping 2020-07-01 and writes two tab-separated
manifest files consumed by download_l2.py and the collocation scripts:

    data/raw/l2/manifest_snpp.txt
    data/raw/l2/manifest_j1.txt

Each line is:  <granule_id>\t<https_download_url>

Uses only the Python standard library (urllib), so it runs with any Python 3
interpreter and no extra dependencies.
"""
import argparse
import json
import os
import urllib.parse
import urllib.request

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"

# CMR collection short name -> manifest basename.
COLLECTIONS = {
    "SNDRSNIML2CPS": "manifest_snpp.txt",  # Suomi NPP CrIS CLIMCAPS L2 standard (CPS)
    "SNDRJ1IML2CPS": "manifest_j1.txt",    # JPSS-1 (NOAA-20) CrIS CLIMCAPS L2 standard (CPS)
}

DATA_REL = "http://esipfed.org/ns/fedsearch/1.1/data#"
PAGE_SIZE = 2000


def fetch_granules(short_name, temporal):
    """Yield (granule_id, https_url) tuples for all matching granules."""
    page = 1
    while True:
        params = {
            "short_name": short_name,
            "temporal": temporal,
            "page_size": str(PAGE_SIZE),
            "page_num": str(page),
        }
        url = CMR + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=60) as resp:
            feed = json.load(resp)["feed"]

        entries = feed.get("entry", [])
        if not entries:
            break
        for entry in entries:
            gid = entry.get("producer_granule_id")
            href = None
            for link in entry.get("links", []):
                if link.get("inherited"):
                    continue
                if link.get("rel") == DATA_REL:
                    href = link.get("href")
                    break
            if gid and href:
                yield gid, href

        if len(entries) < PAGE_SIZE:
            break
        page += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temporal",
        default="2020-07-01T00:00:00Z,2020-07-02T00:00:00Z",
        help="CMR temporal range (start,end). Default covers all of July 1 2020.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <repo>/data/raw/l2.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "raw", "l2"
    )
    os.makedirs(out_dir, exist_ok=True)

    for short_name, basename in COLLECTIONS.items():
        path = os.path.join(out_dir, basename)
        rows = list(fetch_granules(short_name, args.temporal))
        with open(path, "w") as fh:
            for gid, href in rows:
                fh.write(f"{gid}\t{href}\n")
        print(f"{short_name}: {len(rows)} granules -> {path}", flush=True)


if __name__ == "__main__":
    main()
