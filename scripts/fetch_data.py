"""Download the UCI Diabetes 130-US Hospitals dataset."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"


def main() -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    if (raw_dir / "diabetic_data.csv").exists():
        print("Raw data already exists — skipping download.")
        return

    print(f"Downloading from {URL}...")
    buf = io.BytesIO()
    urlretrieve(URL, buf)
    print("Download complete. Extracting...")

    with zipfile.ZipFile(buf) as zf:
        for name in zf.namelist():
            if name.endswith(".csv"):
                data = zf.read(name)
                out_name = Path(name).name
                (raw_dir / out_name).write_bytes(data)
                print(f"  Extracted {out_name}")

    print(f"Done. Files in {raw_dir}:")
    for f in raw_dir.iterdir():
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()