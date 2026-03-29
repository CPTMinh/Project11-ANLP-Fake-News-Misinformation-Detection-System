"""
roberta_model.py
----------------
RoBERTa model wrapper for binary fake news classification.

Provides:
  - FakeNewsDataset  : PyTorch Dataset wrapping processed LIAR CSV rows
  - RobertaClassifier: roberta-base with a 2-class linear head
  - load_model()     : convenience loader from a checkpoint directory

Designed for GPU training in Google Colab; can also be used for
CPU inference on smaller batches.
"""

from __future__ import annotations          # For Python 3.10+ type hinting features

from pathlib import Path                    # For convenient path handling
from typing import Optional                 # For type annotations

import torch                                # For model definition and inference
from torch import nn                        # For neural network modules
from torch.utils.data import Dataset        # For creating a PyTorch Dataset
from transformers import RobertaModel, RobertaTokenizer # For the RoBERTa encoder and tokenizer

# Dataset

class FakeNewsDataset(Dataset):
    """
    Tokenises statement strings for RoBERTa.

    Parameters
    ----------
    texts      : list/Series of raw or pre-cleaned statement strings
    labels     : iterable of int (0 = fake, 1 = real); pass None for inference
    tokenizer  : pre-loaded RobertaTokenizer
    max_length : max token length (truncate/pad)
    """

    def __init__(
        self,
        texts,
        labels,
        tokenizer: RobertaTokenizer,
        max_length: int = 128,
    ):
        self.texts     = list(texts)
        self.labels    = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = str(self.texts[idx]) if self.texts[idx] is not None else ""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Model

class RobertaClassifier(nn.Module):
    """
    roberta-base encoder + linear classification head.

    Architecture:
        RoBERTa [CLS] hidden state (768-d)
            → Dropout(p)
            → Linear(768, num_labels)

    Parameters
    ----------
    model_name : HuggingFace model ID (default: 'roberta-base')
    num_labels : number of output classes (2 for binary)
    dropout    : dropout probability on the classifier head
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.roberta   = RobertaModel.from_pretrained(model_name)
        self.dropout   = nn.Dropout(p=dropout)
        hidden_size    = self.roberta.config.hidden_size  # 768 for roberta-base
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Returns
        -------
        If labels is provided: (loss, logits)
        Otherwise            : logits
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # [CLS] representation (first token)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # shape: (batch, num_labels)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits

        return logits
    
# Convenience loader

def load_model(
    checkpoint_dir: str | Path,
    model_name: str = "roberta-base",
    num_labels: int = 2,
    dropout: float = 0.1,
    device: str | None = None,
) -> tuple["RobertaClassifier", "RobertaTokenizer"]:
    """
    Load a fine-tuned RobertaClassifier and its tokenizer from a checkpoint.

    The checkpoint directory should contain:
      - pytorch_model.bin  OR  model.safetensors
      - tokenizer files (vocab.json, merges.txt, tokenizer_config.json, etc.)

    Parameters
    ----------
    checkpoint_dir : path to the saved model directory
    device         : 'cuda', 'cpu', or None (auto-detect)

    Returns
    -------
    (model, tokenizer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = Path(checkpoint_dir)
    tokenizer = RobertaTokenizer.from_pretrained(str(checkpoint_dir))

    model = RobertaClassifier(
        model_name=model_name,
        num_labels=num_labels,
        dropout=dropout,
    )

    # Try safetensors first, then .bin
    safetensors_path1 = checkpoint_dir / "pytorch_model.safetensors"
    safetensors_path2 = checkpoint_dir / "model.safetensors"
    bin_path          = checkpoint_dir / "pytorch_model.bin"

    if safetensors_path1.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path1), device=device)
        model.load_state_dict(state_dict, strict=False)
    elif safetensors_path2.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path2), device=device)
        model.load_state_dict(state_dict, strict=False)
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(
            f"No model weights found in {checkpoint_dir}. "
            "Expected pytorch_model.bin, pytorch_model.safetensors, or model.safetensors."
        )

    model = model.to(device)
    model.eval()
    return model, tokenizer