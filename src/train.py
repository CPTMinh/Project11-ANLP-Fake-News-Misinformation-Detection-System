"""
train.py
--------
Training loop and evaluation for the RoBERTa + metadata fusion model.

Usage:
    python src/train.py \
        --train_csv data/processed/train_processed.csv \
        --valid_csv data/processed/valid_processed.csv \
        --test_csv  data/processed/test_processed.csv \
        --output_dir models/roberta \
        --epochs 3 \
        --batch_size 16 \
        --lr 2e-5 \
        --dropout 0.1 \
        --max_length 128
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

from model import RoBERTaWithMetadata, LIARDataset
import model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ROBERTA_MODEL = "roberta-base"


# Evaluation

def evaluate_model(
    model: RoBERTaWithMetadata,
    loader: DataLoader,
    device: torch.device,
    split_name: str = "eval",
) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata       = batch["metadata"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, metadata)
            preds = outputs["logits"].argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc      = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_bin   = f1_score(y_true, y_pred, average="binary")
    cm       = confusion_matrix(y_true, y_pred)

    logger.info("── %s ──", split_name.upper())
    logger.info("Accuracy : %.4f", acc)
    logger.info("F1 Macro : %.4f", f1_macro)
    logger.info("F1 Binary: %.4f", f1_bin)
    logger.info("Confusion matrix:\n%s", cm)
    logger.info("\n%s", classification_report(y_true, y_pred, target_names=["Fake", "Real"]))

    return {
        "split": split_name,
        "accuracy":  round(float(acc), 4),
        "f1_macro":  round(float(f1_macro), 4),
        "f1_binary": round(float(f1_bin), 4),
        "confusion_matrix": cm.tolist(),
        "predictions": y_pred.tolist(),
        "labels":      y_true.tolist(),
    }


# Error analysis

def error_analysis(
    df: pd.DataFrame,
    y_true: list,
    y_pred: list,
    output_path: str,
    n: int = 20,
) -> None:
    """Save a CSV of misclassified examples with metadata for manual inspection."""
    df = df.copy()
    df["true"]      = y_true
    df["predicted"] = y_pred
    errors = df[df["true"] != df["predicted"]]

    cols = [
        "statement", "true", "predicted", "label",
        "speaker", "party", "speaker_lie_rate",
        "capitalized_word_ratio", "statement_len",
    ]
    available = [c for c in cols if c in errors.columns]
    errors[available].head(n).to_csv(output_path, index=False)
    logger.info("Error analysis saved → %s (%d misclassified)", output_path, len(errors))


# Training loop

def train(
    train_csv: str,
    valid_csv: str,
    test_csv: str,
    output_dir: str = "models/roberta",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    dropout: float = 0.1,
    max_length: int = 128,
    weight_decay: float = 0.01,
    freeze_base: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load data
    logger.info("Loading CSVs...")
    df_train = pd.read_csv(train_csv)
    df_valid = pd.read_csv(valid_csv)
    df_test  = pd.read_csv(test_csv)

    # Tokeniser & datasets
    logger.info("Loading RoBERTa tokeniser...")
    tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_MODEL)

    train_dataset = LIARDataset(df_train, tokenizer, max_length=max_length)
    cat_columns   = train_dataset.cat_columns           # fit on train, reuse below
    metadata_dim  = train_dataset.metadata_dim

    valid_dataset = LIARDataset(df_valid, tokenizer, cat_columns=cat_columns, max_length=max_length)
    test_dataset  = LIARDataset(df_test,  tokenizer, cat_columns=cat_columns, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    logger.info("Initialising RoBERTa + metadata model (metadata_dim=%d)...", metadata_dim)
    model = RoBERTaWithMetadata(
        roberta_model_name=ROBERTA_MODEL,
        metadata_dim=metadata_dim,
        dropout=dropout,
        freeze_base=freeze_base,
    ).to(device)

    # Resume from checkpoint if specified
    if args.resume_from:
        model.load_state_dict(torch.load(args.resume_from, map_location=torch.device))
        logger.info("Resumed from checkpoint: %s", args.resume_from)

    # Optimiser + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    total_steps   = len(train_loader) * epochs
    warmup_steps  = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Training loop
    best_f1    = 0.0
    best_epoch = 0
    history    = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata       = batch["metadata"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, metadata, labels=labels)
            loss    = outputs["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 100 == 0:
                logger.info(
                    "Epoch %d/%d | Step %d/%d | Loss: %.4f",
                    epoch, epochs, step, len(train_loader), total_loss / step,
                )

        avg_loss = total_loss / len(train_loader)
        logger.info("Epoch %d complete. Avg loss: %.4f", epoch, avg_loss)

        # Validation
        val_metrics = evaluate_model(model, valid_loader, device, split_name=f"valid_epoch{epoch}")
        history.append({"epoch": epoch, "train_loss": avg_loss, **val_metrics})

        # Save best checkpoint
        if val_metrics["f1_macro"] > best_f1:
            best_f1    = val_metrics["f1_macro"]
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info("New best model saved (F1=%.4f)", best_f1)

    # Load best + evaluate on test
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))

    test_metrics = evaluate_model(model, test_loader, device, split_name="test")

    error_analysis(
        df=df_test,
        y_true=test_metrics["labels"],   # intentionally swapped for diff perspective
        y_pred=test_metrics["predictions"],
        output_path=str(output_dir / "error_analysis.csv"),
    )

    # Save training artifacts
    # Save cat_columns so inference pipeline can reconstruct metadata correctly
    import json
    with open(output_dir / "cat_columns.json", "w") as f:
        json.dump(cat_columns, f)

    results = {
        "best_epoch":   best_epoch,
        "best_val_f1":  best_f1,
        "test_metrics": test_metrics,
        "history":      history,
        "config": {
            "model":       ROBERTA_MODEL,
            "epochs":      epochs,
            "batch_size":  batch_size,
            "lr":          lr,
            "dropout":     dropout,
            "max_length":  max_length,
            "metadata_dim": metadata_dim,
        },
    }

    with open(output_dir / "results.json", "w") as f:
        # remove raw predictions list to keep file small
        results_save = results.copy()
        results_save["test_metrics"] = {
            k: v for k, v in test_metrics.items()
            if k not in ("predictions", "labels")
        }
        json.dump(results_save, f, indent=2)

    logger.info("Training complete. Results saved → %s", output_dir)
    return results


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RoBERTa fake news detector")
    parser.add_argument("--train_csv",   default="data/processed/train_processed.csv")
    parser.add_argument("--valid_csv",   default="data/processed/valid_processed.csv")
    parser.add_argument("--test_csv",    default="data/processed/test_processed.csv")
    parser.add_argument("--output_dir",  default="models/roberta")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--max_length",  type=int,   default=128)
    parser.add_argument("--weight_decay",type=float, default=0.01)
    parser.add_argument("--warmup_ratio",  type=float, default=0.1,
                    help="Fraction of total steps used for linear warmup")
    parser.add_argument("--freeze_base", action="store_true",
                        help="Freeze RoBERTa weights, train only the head")
    parser.add_argument("--resume_from", type=str, default=None,
                    help="Path to a best_model.pt checkpoint to resume from")
    args = parser.parse_args()

    train(
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        max_length=args.max_length,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        freeze_base=args.freeze_base,
    )