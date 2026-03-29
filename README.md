# 📰 Fake News & Misinformation Detection System

An end-to-end NLP system for detecting fake news and misinformation, built on the [LIAR dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip). The system features a fine-tuned **RoBERTa** model, TF-IDF baseline comparisons, a REST API, an interactive Streamlit web demo, and an **Agentic AI** fact-checking pipeline powered by Wikipedia + Gemini.

---

## 📁 Repository Structure

```
Project11-ANLP-Fake-News-Misinformation-Detection-System/
│
├── src/                        # Core source code
│   ├── liar_preprocessing.py   # Data download, cleaning, and splitting
│   ├── baseline_model.py       # TF-IDF + Logistic Regression / LinearSVC baselines
│   ├── roberta_model.py        # RoBERTa model & dataset class definitions
│   ├── roberta_train.py        # RoBERTa fine-tuning script (GPU required)
│   ├── roberta_inference.py    # Inference pipeline for RoBERTa
│   ├── evaluate.py             # Shared evaluation utilities (metrics, confusion matrix)
│   ├── agent.py                # Agentic AI: RoBERTa + Wikipedia + Gemini LLM pipeline
│   ├── api.py                  # FastAPI REST API for inference
│   └── app.py                  # Streamlit web demo
│
├── data/
│   ├── download_liar.py        # Script to download the LIAR dataset
│   ├── raw/                    # Raw .tsv files from LIAR
│   └── processed/              # Cleaned train/valid/test CSV files
│
├── models/
│   ├── roberta_best/           # Best fine-tuned RoBERTa checkpoint
│   ├── baseline_lr.pkl         # Trained Logistic Regression model
│   ├── baseline_svm.pkl        # Trained LinearSVC model
│   └── baseline_results/       # Baseline evaluation outputs
│
├── configs/
│   ├── roberta_config.yaml     # Hyperparameters for RoBERTa training
│   └── baseline_config.yaml    # Hyperparameters for baseline training
│
├── notebooks/
│   └── roberta_training_colab.ipynb  # Google Colab notebook for GPU training
│
├── tests/
│   ├── test_liar_preprocessing.py
│   └── test_baseline_model.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Setup

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/CPTMinh/Project11-ANLP-Fake-News-Misinformation-Detection-System.git
cd Project11-ANLP-Fake-News-Misinformation-Detection-System
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** For RoBERTa training you need a CUDA-capable GPU or Google Colab. CPU-only machines can still run inference and the baseline models.

### 4. (Optional) Set Up API Key for the Agentic Component

The fact-checking agent uses the Google Gemini API. Get a free key at [Google AI Studio](https://aistudio.google.com/).

```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "your_api_key_here"

# macOS/Linux
export GEMINI_API_KEY="your_api_key_here"
```

---

## 📦 Data Download & Preprocessing

Download the LIAR dataset and generate the processed splits:

```bash
python data/download_liar.py
python src/liar_preprocessing.py
```

This will populate `data/raw/` with the original `.tsv` files and `data/processed/` with `train_processed.csv`, `valid_processed.csv`, and `test_processed.csv`.

---

## 🏋️ Model Training

### Baseline Models (TF-IDF + LR / LinearSVC) — CPU friendly

```bash
python src/baseline_model.py --config configs/baseline_config.yaml
```

Trained models are saved to `models/baseline_lr.pkl` and `models/baseline_svm.pkl`. Evaluation results are saved to `models/baseline_results/`.

### RoBERTa Fine-Tuning — GPU Required

> **Recommended:** Open `notebooks/roberta_training_colab.ipynb` in [Google Colab](https://colab.research.google.com/) with a GPU runtime (T4 GPU) and run from start to end.
> After downloading from cell 8. Download Results, extract the .zip file and put the content in the repo like this: models/roberta_best/...

To train locally on a GPU machine:

```bash
python src/roberta_train.py --config configs/roberta_config.yaml
```

The best checkpoint is saved to `models/roberta_best/`. Training hyperparameters (epochs, learning rate, batch size, etc.) are all controlled via `configs/roberta_config.yaml`.

---

## 🔍 Running Inference

### Python Script (Single Prediction)

```python
from src.roberta_inference import RobertaInferencePipeline

pipeline = RobertaInferencePipeline(checkpoint_dir="models/roberta_best", device="cpu")
result = pipeline.predict("The president signed a new healthcare bill today.")
print(result)
# {'label': 'REAL', 'label_id': 1, 'confidence': 0.94, 'probabilities': {'FAKE': 0.06, 'REAL': 0.94}}
```

### Agentic Fact-Checking (Multi-step Reasoning)

```bash
# Requires GEMINI_API_KEY to be set (see Environment Setup)
python src/agent.py
```

The agent runs a 3-step pipeline:
1. **RoBERTa** — linguistic pattern analysis
2. **Wikipedia Search Tool** — real-world fact retrieval
3. **Gemini LLM** — evidence synthesis and final verdict

---

## 🚀 Deployment

### Option A: REST API (FastAPI)

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

- **Health check:** `GET http://localhost:8000/health`
- **Predict:** `POST http://localhost:8000/predict`
  - Request body: `{"text": "Your news statement here"}`
  - Response: `{"label": "REAL", "confidence": 0.94, "probabilities": {...}, "version": "roberta_best_v1"}`
- **Interactive docs:** `http://localhost:8000/docs`

### Option B: Streamlit Web Demo

```bash
# From the src/ directory
cd src
streamlit run app.py
```

Open your browser at `http://localhost:8501`. The demo supports both direct RoBERTa prediction and the full agentic deep fact-checking mode.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📊 Experiment Tracking & Logging

- All training scripts use Python's `logging` module with timestamped output.
- Training logs are saved to `models/roberta_logs/` by the HuggingFace Trainer.
- To enable **Weights & Biases** tracking, uncomment `wandb>=0.16` in `requirements.txt`, run `wandb login`, and set `report_to="wandb"` in `configs/roberta_config.yaml`.

---

## 📄 Documentation

| Document | Description |
|---|---|
| `Business_Problem_Definition_Doc.pdf` | Business context, stakeholders, success metrics |
| `Data_Description_Document.pdf` | Dataset, preprocessing steps, limitations |
| `Continual_Learning_Strategy.md` | Model monitoring, retraining, and drift strategy |

---

## 🔑 Key Dependencies

| Package | Purpose |
|---|---|
| `transformers>=4.40` | RoBERTa fine-tuning and inference |
| `torch>=2.0` | Deep learning backend |
| `scikit-learn>=1.4` | Baseline models and evaluation metrics |
| `fastapi` | REST API deployment |
| `streamlit` | Web demo interface |
| `google-genai` | Gemini LLM for agentic reasoning |
| `wikipedia` | Search tool for the fact-checking agent |

---

## ⚠️ Notes

- **No secrets in code.** Always use environment variables for API keys (see `GEMINI_API_KEY` above).
- **Large model files** (e.g., `models/roberta_best/`) are excluded from version control via `.gitignore`. Run the training scripts to reproduce them.
- **LIAR dataset** is downloaded via `data/download_liar.py` and is not committed directly to the repository.