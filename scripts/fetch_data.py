#!/usr/bin/env python
"""Fetch and verify the NASA C-MAPSS turbofan dataset.

The dataset is already committed to this repository (13 files, ~11.5 MB
compressed in the git pack), so a fresh clone needs nothing from the network.
This script exists for two reasons:

1. **Verification.** `--verify` re-checks the committed files against
   `data/raw/CHECKSUMS.sha256`, so anyone can confirm the data is the
   unmodified NASA distribution rather than an unexplained blob.
2. **Recovery.** If `data/raw/` is ever emptied, or you want the data in a
   checkout that excludes it, this downloads it from NASA's public mirror and
   verifies every file before writing.

The archive is nested: the outer zip contains `CMAPSSData.zip`, which contains
the text files.

Usage:
    python scripts/fetch_data.py            # download only what is missing, then verify
    python scripts/fetch_data.py --verify   # verify existing files, download nothing
    python scripts/fetch_data.py --force    # re-download and overwrite
"""

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHECKSUM_FILE = RAW_DIR / "CHECKSUMS.sha256"

# NASA Prognostics Center of Excellence data repository (public mirror).
SOURCE_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
INNER_ZIP = "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip"


def load_checksums() -> dict[str, str]:
    """Parse CHECKSUMS.sha256 into {filename: sha256}."""
    if not CHECKSUM_FILE.exists():
        sys.exit(f"Checksum manifest not found: {CHECKSUM_FILE}")

    checksums = {}
    for line in CHECKSUM_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, path = line.split(maxsplit=1)
        checksums[Path(path).name] = digest
    return checksums


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(checksums: dict[str, str]) -> tuple[list[str], list[str]]:
    """Return (missing, corrupted) filenames."""
    missing, corrupted = [], []
    for name, expected in sorted(checksums.items()):
        path = RAW_DIR / name
        if not path.exists():
            missing.append(name)
        elif sha256_of(path) != expected:
            corrupted.append(name)
    return missing, corrupted


def download(checksums: dict[str, str]) -> None:
    """Download the archive and extract every file named in the manifest."""
    print(f"Downloading C-MAPSS from {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL) as response:  # noqa: S310 - fixed HTTPS URL
        payload = response.read()
    print(f"  {len(payload):,} bytes")

    outer = zipfile.ZipFile(io.BytesIO(payload))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(INNER_ZIP)))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in sorted(checksums):
        data = inner.read(name)
        actual = hashlib.sha256(data).hexdigest()
        if actual != checksums[name]:
            sys.exit(
                f"Checksum mismatch for {name}\n"
                f"  expected {checksums[name]}\n"
                f"  got      {actual}\n"
                "Refusing to write. The upstream archive may have changed."
            )
        (RAW_DIR / name).write_bytes(data)
        print(f"  wrote {name} ({len(data):,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--verify", action="store_true", help="verify existing files without downloading"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if files are present"
    )
    args = parser.parse_args()

    checksums = load_checksums()
    missing, corrupted = verify(checksums)

    if args.force:
        download(checksums)
        missing, corrupted = verify(checksums)
    elif missing or corrupted:
        if args.verify:
            for name in missing:
                print(f"MISSING    {name}")
            for name in corrupted:
                print(f"CORRUPTED  {name}")
            return 1
        print(f"{len(missing)} missing, {len(corrupted)} corrupted - fetching.")
        download(checksums)
        missing, corrupted = verify(checksums)

    if missing or corrupted:
        for name in missing:
            print(f"MISSING    {name}")
        for name in corrupted:
            print(f"CORRUPTED  {name}")
        return 1

    print(f"All {len(checksums)} C-MAPSS files present and verified against NASA checksums.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
