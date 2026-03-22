"""
roberta_inference.py
--------------------
Inference pipeline for the fine-tuned RoBERTa fake news detector.

Can be used:
  1. As a standalone CLI:
       python src/roberta_inference.py --text "Vaccines cause autism."
  2. As an importable module by the REST API or agentic layer:
       from roberta_inference import RobertaInferencePipeline
       pipeline = RobertaInferencePipeline("models/roberta_best")
       result = pipeline.predict("The president signed a new bill today.")

Output format:
    {
        "label": "REAL" | "FAKE",
        "label_id": 1 | 0,
        "confidence": 0.94,      # softmax probability of predicted class
        "probabilities": {       # full softmax distribution
            "FAKE": 0.06,
            "REAL": 0.94
        }
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from roberta_model import RobertaClassifier, load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

LABEL_MAP = {0: "FAKE", 1: "REAL"}


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

class RobertaInferencePipeline:
    """
    Ready-to-use inference wrapper around a fine-tuned RobertaClassifier.

    Parameters
    ----------
    checkpoint_dir : path to the saved model directory
                     (output of roberta_train.py, e.g. models/roberta_best)
    max_length     : token truncation length (must match training config)
    device         : 'cuda', 'cpu', or None (auto-detect)
    batch_size     : number of texts to process per forward pass
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_length: int = 128,
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_length     = max_length
        self.batch_size     = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            "Loading model from %s (device: %s)",
            self.checkpoint_dir, self.device,
        )
        self.model, self.tokenizer = load_model(
            self.checkpoint_dir,
            device=self.device,
        )
        self.model.eval()
        logger.info("Model loaded and ready.")

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        texts: Union[str, list[str]],
    ) -> Union[dict, list[dict]]:
        """
        Run inference on one or more texts.

        Parameters
        ----------
        texts : a single string or a list of strings

        Returns
        -------
        A single result dict (if input was str) or a list of result dicts.
        Each dict contains: label, label_id, confidence, probabilities.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        results = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
            )
            probs = F.softmax(logits, dim=-1).cpu().numpy()

            for prob_row in probs:
                pred_id   = int(prob_row.argmax())
                confidence = float(prob_row[pred_id])
                results.append({
                    "label":     LABEL_MAP[pred_id],
                    "label_id":  pred_id,
                    "confidence": round(confidence, 4),
                    "probabilities": {
                        "FAKE": round(float(prob_row[0]), 4),
                        "REAL": round(float(prob_row[1]), 4),
                    },
                })

        return results[0] if single else results

    # ------------------------------------------------------------------
    # Batch file inference
    # ------------------------------------------------------------------

    def predict_from_csv(
        self,
        csv_path: str | Path,
        text_col: str = "statement_clean",
        output_path: str | Path | None = None,
    ):
        """
        Run inference on a CSV file and optionally save results.

        Parameters
        ----------
        csv_path    : path to input CSV
        text_col    : column name containing text
        output_path : if provided, saves results CSV here

        Returns
        -------
        pd.DataFrame with original columns + prediction columns appended
        """
        import pandas as pd

        df = pd.read_csv(csv_path)
        texts = df[text_col].fillna("").astype(str).tolist()

        logger.info("Running inference on %d examples...", len(texts))
        preds = self.predict(texts)

        df["pred_label"]      = [p["label"] for p in preds]
        df["pred_label_id"]   = [p["label_id"] for p in preds]
        df["pred_confidence"] = [p["confidence"] for p in preds]
        df["prob_fake"]       = [p["probabilities"]["FAKE"] for p in preds]
        df["prob_real"]       = [p["probabilities"]["REAL"] for p in preds]

        if output_path:
            df.to_csv(output_path, index=False)
            logger.info("Predictions saved to %s", output_path)

        return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RoBERTa fake news inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/roberta_best",
        help="Path to the saved model checkpoint directory",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single statement to classify",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to a CSV file for batch inference",
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="statement_clean",
        help="Column name in the CSV containing text",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions CSV (only used with --csv)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max tokenisation length (must match training config)",
    )
    args = parser.parse_args()

    pipeline = RobertaInferencePipeline(
        checkpoint_dir=args.checkpoint,
        max_length=args.max_length,
    )

    if args.text:
        result = pipeline.predict(args.text)
        print("\nInput:", args.text)
        print("Result:", json.dumps(result, indent=2))

    elif args.csv:
        pipeline.predict_from_csv(
            csv_path=args.csv,
            text_col=args.text_col,
            output_path=args.output,
        )

    else:
        parser.error("Provide either --text or --csv")
