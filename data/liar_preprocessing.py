"""
liar_preprocessing.py
---------------------
Preprocessing pipeline for the LIAR dataset.
Handles loading, cleaning, label mapping, feature engineering,
train/val/test splits, and export to CSV.
 
Dataset source: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
Expected files: train.tsv, valid.tsv, test.tsv
"""

import os
import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Constants

# Original LIAR columns (no header in TSV files)
LIAR_COLUMNS = [
    "id",               # Statement ID
    "label",            # Fine-grained truthfulness label (6 classes)
    "statement",        # The claim text
    "subject",          # Topics
    "speaker",          # Speaker name
    "speaker_job",      # Speaker's job title
    "state_info",       # State
    "party",            # Speaker's party affiliation 
    "barely_true_counts", 
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context"           # Contextual information (e.g., venue, date)
]

# 6 classes -> binary mappinng: fake vs real
# "pants-fire", "false", "barely-true" -> fake (0)
# "half-true", "mostly-true", "true" -> real (1)
BINARY_LABEL_MAP = {
    "pants-fire": 0,
    "false": 0,
    "barely-true": 0,
    "half-true": 1,
    "mostly-true": 1,
    "true": 1
}

# Ordinal map for regression / ordinal classification
ORDINAL_LABEL_MAP = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}

CREDIT_COLUMNS = [
    "barely_true_counts", 
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts"
]

# I/O
def load_split(filepath: str | Path) -> pd.DataFrame:
    # Load one LIAR .tsv split into a DataFrame.
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(
        filepath,
        sep = "\t",
        header = None,
        names = LIAR_COLUMNS,
        dtype =  str, # Read everything as string first, we'll clean later
        na_values = [""]
    )
    logger.info("Loaded %d rows from %s", len(df), filepath.name)
    return df

def load_all_splits(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    # Load train/val/test splits from a directory.
    # Returns a dict with keys 'train', 'valid', 'test'.
    data_dir = Path(data_dir)
    splits = {}
    for split in ("train", "valid", "test"):
        path = data_dir / f"{split}.tsv"
        splits[split] = load_split(path)
    return splits

# Text cleaning
