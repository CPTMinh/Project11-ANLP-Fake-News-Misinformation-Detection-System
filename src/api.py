import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .roberta_inference import RobertaInferencePipeline

logger = logging.getLogger(__name__)

# --- Global State for the Model ---
# This ensures the model is loaded only once when the server starts
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the machine learning model
    logger.info("Initializing ML models...")
    try:
        models["roberta"] = RobertaInferencePipeline(
            checkpoint_dir="models/roberta_best",
            # Assuming cpu for basic deployment to save costs, change to 'cuda' if GPU is available
            device="cpu" 
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield
    # Clean up models and release resources
    models.clear()

app = FastAPI(
    title="Fake News Detection API",
    description="A REST API for predicting whether a statement is REAL or FAKE using a fine-tuned RoBERTa model.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Pydantic Models for Input/Output validation ---
class NewsInput(BaseModel):
    text: str = Field(..., title="The text statement to analyze", example="The president signed a new healthcare bill today.")

class PredictionResponse(BaseModel):
    label: str = Field(..., example="REAL")
    label_id: int = Field(..., example=1)
    confidence: float = Field(..., example=0.94)
    probabilities: Dict[str, float] = Field(..., example={"FAKE": 0.06, "REAL": 0.94})
    version: str = Field(..., description="Model version ID")

# --- Endpoints ---
@app.get("/health", tags=["System"])
async def health_check():
    """Check if the API and model are up and running."""
    if "roberta" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "ok", "model": "roberta_best"}

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(news_input: NewsInput):
    """Run fake news detection inference on a single text string."""
    try:
        pipeline = models["roberta"]
        result = pipeline.predict(news_input.text)
        
        # Add model versioning info
        result["version"] = "roberta_best_v1"
        return result
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))