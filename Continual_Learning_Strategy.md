# Section 8: Continual Learning & Monitoring Strategy

**Project:** Fake News & Misinformation Detection System  
**Model:** Fine-tuned `roberta-base` on the LIAR dataset (binary classification: FAKE / REAL)

---

## 8.1 Overview

Language on the internet evolves rapidly. New political narratives, emerging events, and evolving rhetorical patterns mean that a static NLP model will degrade in performance over time — a phenomenon known as **model drift** or **data drift**. This document defines the strategy for ensuring the Fake News Detection System remains accurate and reliable throughout its operational life.

---

## 8.2 How New Data Would Be Collected

Continual learning requires a steady stream of labelled or weakly-labelled incoming data. The proposed collection pipeline has three channels:

### Channel 1: Human-in-the-Loop Feedback
- The Streamlit web demo and REST API expose a **feedback mechanism** (e.g., a thumbs-up/down button).
- Users who disagree with a prediction can flag it. These flagged samples are stored in a review queue.
- A human moderator reviews the queue weekly and assigns ground-truth labels to confirmed misclassifications.
- **Volume target:** Even 50–100 curated human-labelled samples per week provide high-quality signal.

### Channel 2: Automated Harvesting from Fact-Checking Sources
- Schedule a nightly ingestion job that scrapes public fact-checking APIs:
  - [PolitiFact API](https://www.politifact.com/)
  - [Snopes RSS feed](https://www.snopes.com/feed/)
  - [ClaimBuster API](https://idir.uta.edu/claimbuster/)
- Each article provides a statement and a verdict (e.g., "Mostly False", "True") that can be mapped to the binary FAKE/REAL schema.
- Deduplication via SHA-256 hashing of the statement text prevents contamination of existing training data.

### Channel 3: Synthetic Augmentation for Edge Cases
- When the error analysis report (produced by `evaluate.py`) identifies a systematic failure category (e.g., short statements, political satire), synthetic examples can be generated using paraphrase models (e.g., `pegasus-paraphrase`) to augment the training set in that category.

---

## 8.3 How Retraining or Fine-Tuning Would Occur

Two retraining strategies are proposed based on the volume and quality of new data:

### Strategy A: Incremental Fine-Tuning (Default)
Applied when new labelled data accumulates to a threshold (proposed: **500+ new samples**).

1. The new data is added to the existing processed training CSV in `data/processed/`.
2. `src/roberta_train.py` is re-run using the same `configs/roberta_config.yaml`.
3. Training starts from the **existing best checkpoint** (`models/roberta_best/`), not from `roberta-base`. This is parameter-efficient and preserves prior knowledge.
4. The retrained model is saved to a **versioned directory**, e.g., `models/roberta_v2/`.

```bash
# Example: Retrain from existing checkpoint
python src/roberta_train.py --config configs/roberta_config.yaml
```

### Strategy B: Full Retraining (Quarterly)
Applied every 3 months regardless of data volume to combat gradual concept drift.

1. Combined dataset (all historical + new data) is re-preprocessed using `src/liar_preprocessing.py`.
2. Training starts from the original `roberta-base` weights for a clean slate.
3. Results are compared against the production model using the held-out test set before deployment.

### Model Versioning Convention

| Version | Description | Directory |
|---|---|---|
| `v1.0` | Initial LIAR-trained model | `models/roberta_best/` |
| `v1.x` | Incremental fine-tune releases | `models/roberta_v1.x/` |
| `v2.0` | Full quarterly retrain | `models/roberta_v2/` |

A `models/model_registry.json` file tracks version metadata (training date, dataset size, test F1).

---

## 8.4 How Performance Degradation Would Be Detected

### 8.4.1 Proposed Monitoring Metrics

The following metrics should be computed on a **rolling 7-day window** of production traffic:

| Metric | Description | Alert Threshold |
|---|---|---|
| **Prediction confidence distribution** | Mean and std of model confidence scores | Mean drops below 0.70 |
| **FAKE/REAL prediction ratio** | Proportion of FAKE vs REAL predictions | Deviates > 15% from the training prior |
| **Flagging rate** | % of predictions flagged as wrong by users | Exceeds 5% of daily traffic |
| **Rolling accuracy** | Accuracy on newly human-labelled samples | Drops > 3% below baseline |
| **Inference latency (P95)** | 95th percentile response time of the API | Exceeds 2 seconds |

### 8.4.2 Statistical Drift Detection

In addition to manual thresholds, two automated drift tests would run weekly:

1. **Data drift (input distribution shift):** Compare the TF-IDF feature distributions of incoming text against the training set distribution using the **KL-divergence** or **Population Stability Index (PSI)**. A PSI > 0.2 triggers a retraining alert.

2. **Concept drift (label shift):** Apply the **Page-Hinkley test** to the rolling error rate. If the error rate shows a statistically significant upward trend over 14 days, an alert is raised.

### 8.4.3 Alerting

- Alerts are sent to the development team via email or Slack webhook.
- A simple monitoring dashboard can be built using **Streamlit** or **Grafana** by reading log files produced by the API.

---

## 8.5 Model Drift Risks and Mitigation Strategies

### Risk 1: Linguistic Style Drift
- **Description:** Fake news creators adapt their writing style over time to evade detection (adversarial evolution). The vocabulary and rhetorical patterns the model learned from LIAR (2015–2017 political statements) may not generalize to 2025+ content.
- **Mitigation:** Combine linguistic model predictions with the Agentic component's Wikipedia fact-check. When model confidence is low (< 0.65), the agent is automatically triggered to provide a secondary verdict based on factual evidence rather than style alone.

### Risk 2: Domain Shift
- **Description:** LIAR is primarily political fact-checks from the US context. The system may underperform on health misinformation, financial fake news, or content from non-English-speaking contexts.
- **Mitigation:** Monitor per-topic error rates. If a specific domain shows > 10% degradation, collect domain-specific data and apply targeted fine-tuning with a higher learning rate on that domain's data.

### Risk 3: Label Schema Evolution
- **Description:** The binary FAKE/REAL scheme may become too coarse. New categories ("satire", "misleading context", "outdated information") may be needed.
- **Mitigation:** Plan a phased migration to a multi-class schema. Begin by adding a third "UNVERIFIED" class (already produced by the Agentic component's Gemini verdict) and building a separate classifier head. Gradually expand labels as labelled data for new categories accumulates.

### Risk 4: Catastrophic Forgetting
- **Description:** Incremental fine-tuning on new data can cause the model to "forget" patterns learned during the initial training.
- **Mitigation:** Use **Elastic Weight Consolidation (EWC)** or **replay-based methods** (mixing a fraction of old training data — typically 20% — into each new fine-tuning run) to prevent performance regression on older data distributions.

---

## 8.6 Summary Table

| Activity | Trigger | Frequency |
|---|---|---|
| New data ingestion | Nightly automated scraping + weekly human review | Daily / Weekly |
| Confidence & flagging rate monitoring | Continuous API logging | Real-time |
| Drift statistical tests (PSI / Page-Hinkley) | Batch job | Weekly |
| Incremental fine-tuning (Strategy A) | 500+ new labelled samples | As needed |
| Full retraining (Strategy B) | Calendar schedule | Quarterly |
| Error analysis report | Post-training or post-evaluation | After each retrain |
| Model registry update | After successful retraining | After each retrain |
