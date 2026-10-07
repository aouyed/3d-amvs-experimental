#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resumable, parallel downloader for the July 1 CLIMCAPS L2 granules."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

L2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "l2")
MANIFESTS = [os.path.join(L2, "manifest_snpp.txt"), os.path.join(L2, "manifest_j1.txt")]
MIN_SIZE = 15_000_000   # a valid granule is ~21 MB; treat smaller as partial
WORKERS = 6
COOKIE = os.path.expanduser("~/.urs_cookies")


def tasks():
    out = []
    for m in MANIFESTS:
        if not os.path.exists(m):
            continue
        for line in open(m):
            line = line.strip()
            if not line:
                continue
            gid, url = line.split("\t")
            dest = os.path.join(L2, gid)
            if os.path.exists(dest) and os.path.getsize(dest) >= MIN_SIZE:
                continue  # already complete
            out.append((gid, url, dest))
    return out


def fetch(t):
    gid, url, dest = t
    cmd = ["curl", "-sS", "-n", "-c", COOKIE, "-b", COOKIE, "-L", "-f",
           "-C", "-", "--retry", "4", "--retry-all-errors", "--retry-delay", "2",
           "--max-time", "300", "-o", dest, url]
    r = subprocess.run(cmd, capture_output=True)
    ok = r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) >= MIN_SIZE
    return gid, ok, (r.returncode, r.stderr.decode()[-200:] if r.stderr else "")


def main():
    todo = tasks()
    print(f"{len(todo)} granules to download (of 484 total)")
    done = ok = fail = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, t): t for t in todo}
        for fut in as_completed(futs):
            gid, ok_, err = fut.result()
            done += 1
            if ok_:
                ok += 1
            else:
                fail += 1
                print(f"  FAIL {gid}: {err}", file=sys.stderr)
            if done % 25 == 0:
                print(f"  progress: {done}/{len(todo)} (ok {ok}, fail {fail})")
    print(f"DONE: ok={ok}, fail={fail}, total processed={done}")
    if fail:
        sys.exit(f"{fail} granules failed to download — see errors above.")


if __name__ == "__main__":
    main()
