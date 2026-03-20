"""
tune.py
-------
Optuna-based hyperparameter search for both the TF-IDF baseline
and the RoBERTa + metadata model.

Usage:
    # Tune the baseline
    python src/tune.py --model baseline --n_trials 30

    # Tune RoBERTa (fewer trials — each is expensive)
    python src/tune.py --model roberta --n_trials 10

Results are saved to models/{baseline|roberta}/tuning_results.json
Best parameters are printed at the end.
"""

import argparse                 # for command-line argument parsing
import json                     # for saving results in JSON format
import logging                  # for logging progress and results       
from pathlib import Path        # for handling file paths

import numpy as np              # for numerical operations (not used directly here but often needed in tuning)
import optuna                   # for hyperparameter optimization
import pandas as pd             # for data loading and manipulation
from sklearn.metrics import f1_score    # for evaluating model performance (macro F1 score)

logging.basicConfig(    
    level=logging.INFO,                                         
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)   # reduce Optuna noise


# Shared data loader

def load_data(train_csv: str, valid_csv: str):
    df_train = pd.read_csv(train_csv)
    df_valid = pd.read_csv(valid_csv)
    return df_train, df_valid


# ── Baseline tuning ───────────────────────────────────────────────────────────

def tune_baseline(
    train_csv: str,
    valid_csv: str,
    n_trials: int = 30,
    output_dir: str = "models/baseline",
) -> dict:
    """
    Search space:
        C            : regularisation strength   [1e-3, 10]  (log scale)
        max_features : TF-IDF vocabulary size    [10k, 100k]
        ngram_max    : maximum n-gram size        [1, 3]
    """
    from baseline import FeatureBuilder, LABEL_COL
    from sklearn.linear_model import LogisticRegression

    df_train, df_valid = load_data(train_csv, valid_csv)
    y_train = df_train[LABEL_COL].values
    y_valid = df_valid[LABEL_COL].values

    def objective(trial: optuna.Trial) -> float:
        C            = trial.suggest_float("C",            1e-3, 10.0, log=True)
        max_features = trial.suggest_int("max_features",   10_000, 100_000, step=10_000)
        ngram_max    = trial.suggest_int("ngram_max",      1, 3)

        builder = FeatureBuilder(
            max_features=max_features,
            ngram_range=(1, ngram_max),
        )
        X_train = builder.fit_transform(df_train)
        X_valid = builder.transform(df_valid)

        clf = LogisticRegression(
            C=C, max_iter=3000, solver="saga", random_state=42
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_valid)

        return f1_score(y_valid, y_pred, average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return _save_results(study, output_dir, model_name="baseline")


# RoBERTa tuning

def tune_roberta(
    train_csv: str,
    valid_csv: str,
    n_trials: int = 10,
    output_dir: str = "models/roberta",
    epochs_per_trial: int = 1,       # keep to 1 to save time during search
    max_length: int = 128,
) -> dict:
    """
    Search space:
        lr           : learning rate              [1e-5, 5e-5]  (log scale)
        dropout      : dropout probability        [0.1, 0.4]
        batch_size   : training batch size        {8, 16, 32}
        weight_decay : AdamW weight decay         [0.0, 0.1]
        freeze_base  : freeze RoBERTa weights     {True, False}

    Note: Each trial fine-tunes for epochs_per_trial epoch(s) and
    evaluates on the validation set. Full training uses the best params.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup
    from model import RoBERTaWithMetadata, LIARDataset

    df_train, df_valid = load_data(train_csv, valid_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("RoBERTa tuning on device: %s", device)

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    train_dataset = LIARDataset(df_train, tokenizer, max_length=max_length)
    cat_columns   = train_dataset.cat_columns
    metadata_dim  = train_dataset.metadata_dim
    valid_dataset = LIARDataset(df_valid, tokenizer, cat_columns=cat_columns, max_length=max_length)

    def objective(trial: optuna.Trial) -> float:
        lr           = trial.suggest_float("lr",           1e-5, 5e-5, log=True)
        dropout      = trial.suggest_float("dropout",      0.1,  0.4)
        batch_size   = trial.suggest_categorical("batch_size", [8, 16, 32])
        weight_decay = trial.suggest_float("weight_decay", 0.0,  0.1)
        freeze_base  = trial.suggest_categorical("freeze_base", [True, False])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        model = RoBERTaWithMetadata(
            metadata_dim=metadata_dim,
            dropout=dropout,
            freeze_base=freeze_base,
        ).to(device)

        optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs_per_trial
        scheduler   = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        # One epoch
        model.train()
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata       = batch["metadata"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, metadata, labels=labels)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                outputs = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["metadata"].to(device),
                )
                preds = outputs["logits"].argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].numpy())

        return f1_score(all_labels, all_preds, average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return _save_results(study, output_dir, model_name="roberta")


# Shared result saver

def _save_results(
    study: optuna.Study,
    output_dir: str,
    model_name: str,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best = study.best_trial
    results = {
        "best_params":     best.params,
        "best_f1_macro":   round(best.value, 4),
        "n_trials":        len(study.trials),
        "all_trials": [
            {
                "number":  t.number,
                "params":  t.params,
                "f1_macro": round(t.value, 4) if t.value is not None else None,
            }
            for t in study.trials
        ],
    }

    out_path = output_dir / "tuning_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("── Tuning complete (%s) ──", model_name)
    logger.info("Best F1 (macro): %.4f", best.value)
    logger.info("Best params:     %s", best.params)
    logger.info("Results saved → %s", out_path)

    return results


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--model",       choices=["baseline", "roberta"], required=True)
    parser.add_argument("--train_csv",   default="data/processed/train_processed.csv")
    parser.add_argument("--valid_csv",   default="data/processed/valid_processed.csv")
    parser.add_argument("--n_trials",    type=int, default=20)
    parser.add_argument("--output_dir",  default=None,
                        help="Defaults to models/baseline or models/roberta")
    parser.add_argument("--epochs_per_trial", type=int, default=1,
                        help="RoBERTa only: epochs per Optuna trial")
    args = parser.parse_args()

    output_dir = args.output_dir or f"models/{args.model}"

    if args.model == "baseline":
        tune_baseline(
            train_csv=args.train_csv,
            valid_csv=args.valid_csv,
            n_trials=args.n_trials,
            output_dir=output_dir,
        )
    else:
        tune_roberta(
            train_csv=args.train_csv,
            valid_csv=args.valid_csv,
            n_trials=args.n_trials,
            output_dir=output_dir,
            epochs_per_trial=args.epochs_per_trial,
        )