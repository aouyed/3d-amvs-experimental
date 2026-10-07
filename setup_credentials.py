#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check and interactively set Earthdata (~/.netrc) and CDS (~/.cdsapirc) credentials.

Usage:
    python setup_credentials.py           # prompt for whichever is missing
    python setup_credentials.py --check   # report present/missing only (no prompt)

Both files are written with mode 600 and are gitignored (never commit them).
"""
import argparse
import getpass
import os
import stat
import sys

EARTHDATA_MACHINE = "urs.earthdata.nasa.gov"
CDS_URL = "https://cds.climate.copernicus.eu/api"
NETRC = os.path.expanduser("~/.netrc")
CDSAPIRC = os.path.expanduser("~/.cdsapirc")


def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def earthdata_ok():
    """True if ~/.netrc has a urs.earthdata.nasa.gov machine + login + password."""
    lines = [l.strip() for l in _read(NETRC).splitlines() if l.strip()]
    has_machine = any(l.startswith("machine ") and l.split()[1] == EARTHDATA_MACHINE for l in lines)
    has_login = any(l.startswith("login ") for l in lines)
    has_password = any(l.startswith("password ") for l in lines)
    return has_machine and has_login and has_password


def cds_ok():
    """True if ~/.cdsapirc has a url: and key: line."""
    lines = [_read(CDSAPIRC).strip() for _ in (0,)][0]
    txt = _read(CDSAPIRC)
    return any(l.strip().startswith("url:") for l in txt.splitlines()) and \
           any(l.strip().startswith("key:") for l in txt.splitlines())


def _chmod_600(path):
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def _prompt_secret(prompt):
    """Read a secret without echoing when a TTY is present; else read from stdin."""
    if sys.stdin.isatty():
        return getpass.getpass(prompt)
    line = sys.stdin.readline()
    if not line:
        raise SystemExit("No input available on stdin (non-interactive).")
    return line.rstrip("\n")


def write_earthdata(username, password):
    content = (
        f"machine {EARTHDATA_MACHINE}\n"
        f"    login {username}\n"
        f"    password {password}\n"
    )
    with open(NETRC, "w") as fh:
        fh.write(content)
    _chmod_600(NETRC)


def write_cds(api_key):
    content = f"url: {CDS_URL}\nkey: {api_key}\n"
    with open(CDSAPIRC, "w") as fh:
        fh.write(content)
    _chmod_600(CDSAPIRC)


def prompt_earthdata():
    print(f"Earthdata ({EARTHDATA_MACHINE}) credentials not found.", file=sys.stderr)
    username = input("Earthdata username: ")
    password = _prompt_secret("Earthdata password: ")
    write_earthdata(username, password)
    print(f"wrote {NETRC}")


def prompt_cds():
    print("CDS API credentials (~/.cdsapirc) not found.", file=sys.stderr)
    api_key = _prompt_secret("CDS API key: ")
    write_cds(api_key)
    print(f"wrote {CDSAPIRC}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only, do not prompt")
    args = ap.parse_args()

    if args.check:
        print(f"earthdata: {'present' if earthdata_ok() else 'MISSING'}")
        print(f"cds:       {'present' if cds_ok() else 'MISSING'}")
        return

    if not earthdata_ok():
        prompt_earthdata()
    if not cds_ok():
        prompt_cds()
    print("credentials up to date")


if __name__ == "__main__":
    main()
