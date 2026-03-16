"""
download_liar.py
----------------
Downloads and extracts the LIAR dataset into data/raw/.
Run this once before running liar_preprocessing.py.

Usage:
    python data/download_liar.py
    python data/download_liar.py --output_dir path/to/raw
"""

import argparse
import logging
import zipfile
from pathlib import Path
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LIAR_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
EXPECTED_FILES = ["train.tsv", "valid.tsv", "test.tsv"]


def download_liar(output_dir: str | Path = "data/raw") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "liar_dataset.zip"

    # Check if already downloaded
    if all((output_dir / f).exists() for f in EXPECTED_FILES):
        logger.info("LIAR dataset already present in %s. Skipping download.", output_dir)
        return

    logger.info("Downloading LIAR dataset from %s ...", LIAR_URL)
    urllib.request.urlretrieve(LIAR_URL, zip_path)
    logger.info("Download complete: %s", zip_path)

    logger.info("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    logger.info("Extraction complete.")

    zip_path.unlink()  # Remove zip to save space
    logger.info("Removed zip file.")

    # Verify
    for fname in EXPECTED_FILES:
        fpath = output_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Expected file not found after extraction: {fpath}\n"
                "The dataset structure may have changed. Please download manually."
            )
    logger.info("All expected files verified: %s", EXPECTED_FILES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the LIAR dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Directory to save raw dataset files",
    )
    args = parser.parse_args()
    download_liar(output_dir=args.output_dir)