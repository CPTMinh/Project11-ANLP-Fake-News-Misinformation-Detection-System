# Section 10: Project Management & Teamwork

**Project:** Fake News & Misinformation Detection System  
**Team Size:** 1 (Solo project)  
**Course:** Applications of Natural Language Processing in Business (ANLP)

---

## 10.1 Project Timeline

The project was executed over approximately **10 weeks** from January 2026 to March 2026.

| Week | Phase | Key Activities |
|---|---|---|
| Week 1–2 | **Planning & Problem Definition** | Define business problem, select dataset (LIAR), write Problem Definition Document, set up Git repository and project structure |
| Week 3 | **Data Management** | Download LIAR dataset, build `liar_preprocessing.py`, perform EDA, write Data Description Document |
| Week 4–5 | **Baseline Modelling** | Implement TF-IDF + Logistic Regression and LinearSVC baselines, hyperparameter search, evaluate and save results |
| Week 5–6 | **RoBERTa Fine-Tuning** | Implement `roberta_model.py` and `roberta_train.py`, run training on Google Colab (GPU), evaluate on test set, error analysis |
| Week 7 | **Deployment** | Build FastAPI REST API (`api.py`), build Streamlit web demo (`app.py`), write `roberta_inference.py` |
| Week 8 | **Agentic AI Component** | Design and implement the 3-step fact-checking agent (`agent.py`): RoBERTa → Wikipedia → Gemini LLM |
| Week 9 | **Documentation & Compliance** | Write Continual Learning Strategy, Data Privacy & Robustness, Ethics, and Project Management documents; complete README |
| Week 10 | **Testing & Finalisation** | Write and run unit tests (`tests/`), review repository for hardcoded secrets, finalize all documents, prepare slides |

---

## 10.2 Task Breakdown (Solo Role Simulation)

Since this is a solo project, all roles were performed by one person. The breakdown below maps tasks to the professional roles they would represent in a real team environment.

| Simulated Role | Responsibilities | Tasks Completed |
|---|---|---|
| **Project Manager** | Planning, timeline, risk tracking | Timeline definition, weekly self-review, scope management |
| **Data Engineer** | Data sourcing, preprocessing pipeline | `data/download_liar.py`, `src/liar_preprocessing.py`, `Data_Description_Document.pdf` |
| **ML Engineer** | Model training, evaluation, optimisation | `src/baseline_model.py`, `src/roberta_train.py`, `src/roberta_model.py`, `src/evaluate.py`, `notebooks/` |
| **Software Engineer** | API design, deployment, code quality | `src/api.py`, `src/app.py`, `src/roberta_inference.py`, `configs/`, `requirements.txt` |
| **AI/Research Engineer** | Agentic AI design and implementation | `src/agent.py` (multi-step reasoning pipeline) |
| **Technical Writer** | Documentation, README, compliance docs | `README.md`, all strategy documents |
| **QA Engineer** | Testing, reproducibility | `tests/test_baseline_model.py`, `tests/test_liar_preprocessing.py` |

---

## 10.3 Tools Used

| Category | Tool |
|---|---|
| Version control | Git / GitHub |
| Development environment | VS Code |
| GPU training | Google Colab (T4 GPU) |
| Experiment configuration | YAML config files |
| Testing | pytest |
| Dependency management | `requirements.txt` + `venv` |

---

## 10.4 Challenges Encountered

| Challenge | How It Was Resolved |
|---|---|
| No local GPU for RoBERTa training | Used Google Colab with a free T4 GPU runtime via dedicated notebook |
| LIAR dataset has 6 fine-grained labels, not binary | Mapped to binary (FAKE = 0 / REAL = 1) in preprocessing; documented trade-offs |
| Agent requires paid API (Gemini) | Used the free-tier Google AI Studio key; abstracted via environment variable |
| Avoiding catastrophic forgetting in continual learning | Documented as a conceptual strategy (EWC + replay) since full implementation is out of scope |

---

## 10.5 How the Project Would Scale in a Real Team Environment

If this project were to transition from a solo academic prototype to a production system within an organisation, the following changes would be made:

### Team Structure
A realistic production team would consist of **5–7 people**:

- 1 × Product Manager (business requirements, user feedback)
- 2 × ML Engineers (model development, evaluation, continual learning)
- 1 × Data Engineer (pipeline, ingestion, storage)
- 1 × Backend Engineer (API, containerisation, CI/CD)
- 1 × Frontend/UX Engineer (dashboard, demo interface)
- 1 × MLOps Engineer (monitoring, drift detection, deployment)

### Process Changes
| Area | Solo Approach | Team Approach |
|---|---|---|
| Version control | Single `main` branch | Feature branches + pull request reviews |
| Configuration | YAML files in repo | Centralised config service (e.g., Hydra, Dynaconf) |
| Training | Manual Colab runs | Automated training jobs (e.g., Vertex AI, AWS SageMaker) |
| Monitoring | Conceptual strategy | Active dashboards (Grafana, Evidently AI) |
| Deployment | Local `uvicorn` server | Containerised (Docker) + Kubernetes orchestration |
| Testing | 2 unit test files | Full CI pipeline (GitHub Actions) with coverage gates |
| Model versioning | Directory naming convention | MLflow or DVC model registry |

### Scalability Considerations
- The FastAPI backend is stateless and can be horizontally scaled behind a load balancer.
- The RoBERTa model weights (~500 MB) should be served from a shared model registry (e.g., HuggingFace Hub private repo or S3) rather than being bundled with the application container.
- For high-throughput scenarios, the inference pipeline would switch from synchronous request-response to an asynchronous queue-based architecture (e.g., Celery + Redis).
