"""
roberta_train.py
----------------
Fine-tuning script for RoBERTa on the LIAR binary fake-news dataset.

⚠️  GPU REQUIRED — designed to run in Google Colab (see notebooks/).
    For CPU-only environments use the baseline models in baseline_model.py.

Usage (on Colab or any GPU machine):
    python src/roberta_train.py
    python src/roberta_train.py --config configs/roberta_config.yaml

What this script does:
  1. Load processed CSVs from data/processed/
  2. Tokenise with RobertaTokenizer
  3. Fine-tune roberta-base with HuggingFace Trainer
  4. Early stopping on eval F1
  5. Save best checkpoint to models/roberta_best/
  6. Evaluate on test set and print results + confusion matrix
"""

from __future__ import annotations      # For Python 3.10+ type hinting features

import argparse                         # For CLI argument parsing
import logging                          # For logging progress and errors
import sys                              # For modifying sys.path to import local modules
from pathlib import Path                # For convenient path handling

import numpy as np                      # For numerical operations    
import pandas as pd                     # For DataFrame manipulation    
import torch                            # For model definition and training
import yaml                             # For loading YAML config files
from transformers import (
    EarlyStoppingCallback,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)

# Allow running from project root OR from src/
sys.path.insert(0, str(Path(__file__).parent))  # Ensure we can import from the current directory

from roberta_model import FakeNewsDataset, RobertaClassifier
from evaluate import (
    compute_metrics as compute_eval_metrics,
    error_analysis,
    plot_confusion_matrix,
    print_classification_report,
    print_comparison_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Config

def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Data helpers

def load_split(
    csv_path: Path,
    text_col: str,
    label_col: str,
) -> tuple[list[str], list[int]]:
    """Load a processed CSV and return (texts, labels) lists."""
    df = pd.read_csv(csv_path).dropna(subset=[text_col, label_col])
    texts  = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    logger.info("  Loaded %d examples from %s", len(texts), csv_path.name)
    return texts, labels

# HuggingFace compute_metrics callback

def make_compute_metrics_fn():
    """Return a compute_metrics function compatible with HuggingFace Trainer."""
    from sklearn.metrics import f1_score, accuracy_score

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1":       float(f1_score(labels, preds, average="macro", zero_division=0)),
        }

    return _compute_metrics

# Main training function
def train(config_path: str | Path = "configs/roberta_config.yaml") -> None:
    cfg = load_config(config_path)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device.upper())
    if device == "cpu":
        logger.warning(
            "No GPU detected — training will be VERY slow. "
            "Use Google Colab with GPU runtime."
        )

    # Paths
    data_dir    = Path(cfg["data"]["processed_dir"])
    text_col    = cfg["data"]["text_column"]
    label_col   = cfg["data"]["label_column"]
    model_name  = cfg["model"]["name"]
    max_length  = cfg["model"]["max_length"]
    best_dir    = Path(cfg["output"]["best_model_dir"])
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # Data
    logger.info("Loading data splits...")
    train_texts, train_labels = load_split(data_dir / cfg["data"]["train_file"], text_col, label_col)
    valid_texts, valid_labels = load_split(data_dir / cfg["data"]["valid_file"], text_col, label_col)
    test_texts,  test_labels  = load_split(data_dir / cfg["data"]["test_file"],  text_col, label_col)

    train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer, max_length)
    valid_dataset = FakeNewsDataset(valid_texts, valid_labels, tokenizer, max_length)
    test_dataset  = FakeNewsDataset(test_texts,  test_labels,  tokenizer, max_length)

    # Model
    logger.info("Initialising model: %s", model_name)
    model = RobertaClassifier(
        model_name=model_name,
        num_labels=cfg["model"]["num_labels"],
        dropout=cfg["model"]["dropout"],
    )

    # Training arguments
    t = cfg["training"]
    training_args = TrainingArguments(
        output_dir=cfg["output"]["checkpoint_dir"],
        num_train_epochs=t["num_epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["eval_batch_size"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        fp16=t.get("fp16", False) and (device == "cuda"),  # only on GPU
        seed=t.get("seed", 42),

        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=cfg.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 500),
        logging_dir=cfg["output"]["logging_dir"],
        logging_steps=cfg.get("logging_steps", 100),

        # Best model selection
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        metric_for_best_model=t.get("metric_for_best_model", "eval_f1"),
        greater_is_better=t.get("greater_is_better", True),
        save_total_limit=2,
        report_to="none",  # disable wandb unless explicitly configured
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=make_compute_metrics_fn(),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=t.get("early_stopping_patience", 2)
            )
        ],
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save best model
    logger.info("Saving best model to %s", best_dir)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logger.info("Model and tokenizer saved.")

    # Test evaluation
    logger.info("Evaluating on test set...")
    test_output = trainer.predict(test_dataset)
    y_pred = np.argmax(test_output.predictions, axis=-1)
    y_true = np.array(test_labels)

    print_classification_report(y_true, y_pred, model_name="RoBERTa (fine-tuned)")
    metrics = compute_eval_metrics(y_true, y_pred)
    logger.info("Test metrics: %s", metrics)

    plot_confusion_matrix(
        y_true, y_pred,
        model_name="RoBERTa",
        save_path=results_dir / "confusion_matrix_roberta.png",
        show=False,
    )

    # Error analysis
    test_df = pd.read_csv(data_dir / cfg["data"]["test_file"])
    test_df_clean = test_df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    errors = error_analysis(test_df_clean, y_true, y_pred, text_col="statement")
    errors.to_csv(results_dir / "error_analysis_roberta.csv", index=False)
    logger.info("Error analysis saved to %s", results_dir / "error_analysis_roberta.csv")

    # Final comparison (RoBERTa alone here; merge with baseline results for full table)
    print_comparison_table({"RoBERTa": metrics})

# CLI entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for fake news detection")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/roberta_config.yaml",
        help="Path to RoBERTa YAML config",
    )
    args = parser.parse_args()
    train(config_path=args.config)