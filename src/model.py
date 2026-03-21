"""
model.py
--------
RoBERTa fine-tuned model with metadata fusion for fake news detection.

Architecture:
    statement text  →  RoBERTa  →  [CLS] embedding (768-dim)
                                          ↓
    numeric + categorical features  →  MLP projection
                                          ↓
                               concat [CLS + metadata]
                                          ↓
                                 dropout → Linear → logits (2)

The metadata features (speaker credibility, party, context, etc.)
are projected to a small dense vector and concatenated with the
RoBERTa [CLS] token before the final classification head.
This allows the model to jointly reason over language and speaker signals.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig


class MetadataProjection(nn.Module):
    """
    Small MLP that projects raw numeric + one-hot metadata
    into a dense representation compatible with the RoBERTa embedding space.
    """

    def __init__(self, input_dim: int, output_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RoBERTaWithMetadata(nn.Module):
    """
    RoBERTa encoder with a metadata fusion head.

    Args:
        roberta_model_name: HuggingFace model identifier.
                            Recommended: "roberta-base"
        metadata_dim:       Number of input metadata features
                            (numeric + one-hot categoricals).
        meta_proj_dim:      Output size of the metadata projection MLP.
        num_labels:         Number of output classes (2 for binary).
        dropout:            Dropout rate on the classifier head.
        freeze_base:        If True, freeze all RoBERTa weights and only
                            train the classification head. Useful for very
                            limited compute — set to False for full fine-tuning.
    """

    def __init__(
        self,
        roberta_model_name: str = "roberta-base",
        metadata_dim: int = 64,
        meta_proj_dim: int = 64,
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ):
        super().__init__()

        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        hidden_size = self.roberta.config.hidden_size  # 768 for roberta-base

        if freeze_base:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.metadata_proj = MetadataProjection(
            input_dim=metadata_dim,
            output_dim=meta_proj_dim,
            dropout=dropout,
        )

        # Classifier head: [CLS] + projected metadata → logits
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + meta_proj_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            input_ids:      (batch, seq_len) tokenised statement
            attention_mask: (batch, seq_len) attention mask
            metadata:       (batch, metadata_dim) numeric + categorical features
            labels:         (batch,) integer class labels (optional)

        Returns:
            dict with keys:
              "logits"  — (batch, num_labels)
              "loss"    — scalar CrossEntropyLoss (only if labels provided)
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        meta_embedding = self.metadata_proj(metadata)

        fused = torch.cat([cls_embedding, meta_embedding], dim=-1)
        logits = self.classifier(fused)

        result = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result


# ── Dataset ───────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast


# Numeric features passed to the metadata projection
NUMERIC_FEATURES = [
    "statement_len",
    "statement_char_len",
    "exclamation_count",
    "question_count",
    "capitalized_word_ratio",
    "speaker_lie_rate",
    "speaker_total_statements",
    "is_political_context",
]

CATEGORICAL_FEATURES = ["party", "speaker_job", "state_info"]


class LIARDataset(Dataset):
    """
    PyTorch Dataset for the processed LIAR CSV files.

    Tokenises statement text with RoBERTa tokeniser and builds
    metadata feature vectors combining numeric and one-hot categorical features.

    Args:
        df:             Processed DataFrame from liar_preprocessing.py
        tokenizer:      RoBERTa tokeniser instance
        cat_columns:    List of one-hot column names (fit on training set,
                        applied to valid/test). Pass None to fit on this df.
        max_length:     Maximum token sequence length (default 128).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: RobertaTokenizerFast,
        cat_columns: list[str] | None = None,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = df["label_binary"].values.astype(int)
        self.texts = df["statement_clean"].fillna("").tolist()

        # ── Numeric features ──────────────────────────────────────────────────
        self.numeric = df[NUMERIC_FEATURES].fillna(0).values.astype(np.float32)

        # ── Categorical one-hot ───────────────────────────────────────────────
        cat_dummies = pd.get_dummies(
            df[CATEGORICAL_FEATURES].fillna("unknown"), drop_first=False
        )
        if cat_columns is not None:
            cat_dummies = cat_dummies.reindex(columns=cat_columns, fill_value=0)
        self.cat_columns = cat_dummies.columns.tolist()
        self.categorical = cat_dummies.values.astype(np.float32)

        # metadata_dim = numeric cols + one-hot cols
        self.metadata_dim = self.numeric.shape[1] + self.categorical.shape[1]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        metadata = np.concatenate([self.numeric[idx], self.categorical[idx]])

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "metadata":       torch.tensor(metadata, dtype=torch.float32),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }