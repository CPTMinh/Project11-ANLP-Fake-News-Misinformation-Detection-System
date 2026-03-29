"""
liar_preprocessing.py
---------------------
Preprocessing pipeline for the LIAR dataset.
Handles loading, cleaning, label mapping, feature engineering,
train/val/test splits, and export to CSV.
 
Dataset source: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
Expected files: train.tsv, valid.tsv, test.tsv
"""

import os                       # For environment variables and path handling
import re                       # For regular expressions in text cleaning  
import logging                  # For logging progress and errors
import pandas as pd             # For DataFrame manipulation
import numpy as np              # For numerical operations
from pathlib import Path        # For convenient path handling

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

def clean_text(text: str) -> str:
    # Normalize a raw statement string
    # Strip leading/trailing whitespace
    # Collapse multiple spaces
    # Remove non-ASCII characters
    # Lowercase
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII
    text = text.lower()  # Lowercase
    return text

def clean_categorical(series: pd.Series, fill: str = "unknown") -> pd.Series:
    # Lowercase + strip + fill NaN for categorical string columns
    return (
        series.fillna(fill)
        .str.strip()
        .str.lower()
        .replace("", fill)
    )

# Label handling
def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Add two label columns: 
    #  - label_binary: 0 = fake, 1 = real
    #  - label_ordinal: 0 (pants-fire) ... 5 (true)
    # Rows with missing or invalid labels will be dropped.

    df = df.copy()

    unknown_mask = ~df["label"].isin(BINARY_LABEL_MAP)
    if unknown_mask.any():
        logger.warning(
            "Dropping %d rows with unknown labels: %s",
            unknown_mask.sum(),
            df.loc[unknown_mask, "label"].unique().tolist(),
        )
        df = df[~unknown_mask].reset_index(drop=True)
    
    df["label_binary"] = df["label"].map(BINARY_LABEL_MAP)
    df["label_ordinal"] = df["label"].map(ORDINAL_LABEL_MAP)
    return df

# Missing value handling
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
      - statement  : drop rows where statement is missing (cannot learn without text)
      - credit cols: fill NaN with 0 (speaker has no recorded history)
      - categoricals: fill with 'unknown'
    """
    df = df.copy()
 
    # Critical: drop if no statement
    before = len(df)
    df = df.dropna(subset=["statement"])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with missing statement.", dropped)
 
    # Credit history counts → numeric, NaN → 0
    for col in CREDIT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
 
    # Categorical columns
    for col in ["subject", "speaker", "speaker_job", "state_info", "party", "context"]:
        df[col] = clean_categorical(df[col])
 
    return df

# Feature engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive additional features useful for both classical ML and
    transformer-based approaches.
    """
    df = df.copy()
 
    # --- Text features ---
    df["statement_clean"] = df["statement"].apply(clean_text)
    df["statement_len"] = df["statement_clean"].str.split().str.len()
    df["statement_char_len"] = df["statement_clean"].str.len()
    df["exclamation_count"] = df["statement"].str.count(r"!")
    df["question_count"] = df["statement"].str.count(r"\?")
    df["capitalized_word_ratio"] = df["statement"].apply(_capitalized_ratio)
 
    # --- Speaker credit history ---
    total_statements = df[CREDIT_COLUMNS].sum(axis=1).replace(0, np.nan)
    df["speaker_lie_rate"] = (
        (df["false_counts"] + df["pants_on_fire_counts"]) / total_statements
    ).fillna(0.0)
    df["speaker_total_statements"] = df[CREDIT_COLUMNS].sum(axis=1)
 
    # --- Context: is the statement political? ---
    political_keywords = ["congress", "senate", "president", "republican", "democrat", "white house"]
    df["is_political_context"] = df["context"].apply(
        lambda x: int(any(kw in str(x) for kw in political_keywords))
    )
 
    return df
 
 
def _capitalized_ratio(text: str) -> float:
    """Fraction of words that are fully capitalized (e.g., FAKE, HOAX)."""
    if not isinstance(text, str) or len(text.split()) == 0:
        return 0.0
    words = text.split()
    return sum(1 for w in words if w.isupper() and len(w) > 1) / len(words)
 
 
# Class balance report
 
def report_class_balance(splits: dict[str, pd.DataFrame]) -> None:
    """Log class distribution for each split."""
    for name, df in splits.items():
        counts = df["label"].value_counts()
        binary_counts = df["label_binary"].value_counts()
        logger.info(
            "\n=== %s split (%d rows) ===\n"
            "6-class distribution:\n%s\n"
            "Binary distribution (0=fake, 1=real):\n%s",
            name.upper(), len(df),
            counts.to_string(),
            binary_counts.to_string(),
        )
 
 
# Export
 
def export_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Save processed splits as CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    for name, df in splits.items():
        out_path = output_dir / f"{name}_processed.csv"
        df.to_csv(out_path, index=False)
        logger.info("Saved %s → %s", name, out_path)
 
 
# Main pipeline
 
def run_pipeline(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
) -> dict[str, pd.DataFrame]:
    """
    End-to-end preprocessing pipeline.
 
    Steps:
      1. Load raw .tsv splits
      2. Handle missing values
      3. Map labels (binary + ordinal)
      4. Engineer features
      5. Report class balance
      6. Export to data/processed/
 
    Returns:
      dict with keys 'train', 'valid', 'test' → processed DataFrames
    """
    logger.info("── Step 1: Loading raw splits from %s ──", data_dir)
    splits = load_all_splits(data_dir)
 
    processed = {}
    for name, df in splits.items():
        logger.info("── Processing '%s' split ──", name)
 
        df = handle_missing(df)
        df = map_labels(df)
        df = engineer_features(df)
 
        processed[name] = df
 
    logger.info("── Step 5: Class balance report ──")
    report_class_balance(processed)
 
    logger.info("── Step 6: Exporting to %s ──", output_dir)
    export_splits(processed, output_dir)
 
    logger.info("Pipeline complete.")
    return processed
 
 
# Entry point 
 
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description="LIAR dataset preprocessing pipeline")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Directory containing train.tsv, valid.tsv, test.tsv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed CSV files",
    )
    args = parser.parse_args()
 
    run_pipeline(data_dir=args.data_dir, output_dir=args.output_dir)