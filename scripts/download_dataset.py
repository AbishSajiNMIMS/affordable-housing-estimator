"""Download the historical training dataset and verify it.

The CSV is already committed at data/raw/bengaluru_house_prices.csv (it is
published under CC0). Run this script to re-download it from the public
mirror and confirm it is byte-for-byte the expected file:

    python scripts/download_dataset.py

To use Kaggle directly instead, download "Bengaluru House price data"
(https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data),
then place Bengaluru_House_Data.csv at data/raw/bengaluru_house_prices.csv.
The Kaggle copy can differ in line endings, so its checksum may not match;
the columns and rows are the same.
"""
import hashlib
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def main():
    target = config.RAW_DATASET_PATH
    print(f"Downloading {config.TRAINING_MIRROR_URL}")
    resp = requests.get(config.TRAINING_MIRROR_URL, timeout=60)
    resp.raise_for_status()
    digest = hashlib.sha256(resp.content).hexdigest()
    if digest != config.TRAINING_SHA256:
        print(f"Checksum mismatch: got {digest}, expected {config.TRAINING_SHA256}. File not saved.")
        sys.exit(1)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(resp.content)
    print(f"Saved {len(resp.content):,} bytes to {target} (sha256 OK)")


if __name__ == "__main__":
    main()
