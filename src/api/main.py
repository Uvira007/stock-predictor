"""FastAPI app: predict, retrain, tickers."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import get_settings
from ..model.train import load_model, train_model, finetune_model
from ..model.inference import predict as run_predict
from ..utils.github_push import push_models_to_github


# Global model cache
_model_cache: dict = {}


def _get_model():
    global _model_cache
    if "model" not in _model_cache:
        settings = get_settings()
        models_dir = Path(settings.models_dir)
        if not (models_dir / "model.pt").exists():
            return None, None, None
        model, config, stats = load_model(models_dir=models_dir)
        _model_cache["model"] = model
        _model_cache["config"] = config
        _model_cache["stats"] = stats
    return (
        _model_cache["model"],
        _model_cache["config"],
        _model_cache["stats"],
    )


def _clear_model_cache():
    global _model_cache
    _model_cache.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: optionally load model if exists
    _get_model()
    yield
    # Shutdown
    _clear_model_cache()


app = FastAPI(title="Stock GRU Predictor API", lifespan=lifespan)


# Allow streamlit community cloud and local dev to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8051"],
    allow_origin_regex=r"https?://.*\.streamlit\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Request/Response schemas ---


class PredictRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    n_days: int | None = Field(default=None, description="Number of days to predict (default: model's max)")


class PredictResponse(BaseModel):
    ticker: str
    predictions: List[float]
    last_date: str
    model_predict_days: int
    n_days_returned: int


class RetrainRequest(BaseModel):
    tickers: List[str] | None = Field(default=None, description="Tickers to train on (default: config default)")
    epochs: int = Field(default=50, ge=1, le=200)
    period: str = Field(default="5y", description="Yahoo Finance period")


class RetrainResponse(BaseModel):
    message: str
    job_started: bool


class FineTuneRequest(BaseModel):
    period: str = Field(default="6mo", description="Yahoo Finance period for fine-tuning (e.g. 5y)")
    epochs: int = Field(default=15, ge=1, le=100)
    lr: float = Field(default=1e-4, gt=0, description="Learning rate for fine-tuning")


class FineTuneResponse(BaseModel):
    message: str
    job_started: bool


# --- Endpoints ---


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tickers")
def list_tickers():
    """List tickers supported by the loaded model."""
    model, config, _ = _get_model()
    if config is None:
        return {"tickers": [], "message": "No model loaded. Train a model first."}
    return {"tickers": config["tickers"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict next N days closing price for a ticker (uses latest Yahoo data)."""
    model, config, _ = _get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded. Train a model first (see /retrain).")
    try:
        out = run_predict(
            ticker=req.ticker,
            n_days=req.n_days,
            models_dir=get_settings().models_dir,
        )
        return PredictResponse(**out)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/retrain", response_model=RetrainResponse)
def retrain(req: RetrainRequest, background_tasks: BackgroundTasks):
    """
    Start retraining the model in the background.
    Use cautiously: training can take several minutes.
    """
    def _train():
        _clear_model_cache()
        train_model(
            tickers=req.tickers,
            epochs=req.epochs,
            period=req.period,
        )
        # Reload into cache
        _get_model()
        
        # Push the updated model to GitHub
        try:
            ok, message = push_models_to_github(commit_message="Update Model (retrain)")
            print(f"Push to GitHub after retraining: {message}")
        except Exception as e:
            print(f"Push to GitHub failed: {e}")                   

    background_tasks.add_task(_train)
    return RetrainResponse(
        message="Retrain started in background. This may take several minutes. Model will be reloaded when done.",
        job_started=True,
    )


@app.post("/finetune", response_model=FineTuneResponse)
def finetune(req: FineTuneRequest, background_tasks: BackgroundTasks):
    """
    Fine-tune the loaded model on recent data (e.g. last 6 months)
    Freezes ticker_embed and gru weights; trained only for the fc head.
    Requires an existing model
    """
    settings = get_settings()
    if not(Path(settings.models_dir) / "model.pt").exists():
        raise HTTPException(
            status_code=503,
            detail="No model to fine-tune. train a model first (See /retrain or scripts/train_model.py).",
        )
    
    def _run_finetune():
        _clear_model_cache()
        finetune_model(
            period=req.period,
            epochs=req.epochs,
            lr=req.lr
        )
        _get_model()

        try:
            ok, message = push_models_to_github(commit_message="Update model (finetune)")
            print(f" Pushed to github after finetuning: {message}")
        except Exception as e:
            print(f"Push to github failed: {e}")
    
    background_tasks.add_task(_run_finetune)
    return FineTuneResponse(
        message=f"Fine-tuning started in background (period={req.period}). Model will be reloaded when done.",
        job_started=True
        )


@app.get("/model/status")
def model_status():
    """Whether a trained model is loaded and path."""
    settings = get_settings()
    models_dir = Path(settings.models_dir)
    has_model = (models_dir / "model.pt").exists()
    return {
        "model_loaded": "model" in _model_cache,
        "model_path": str(models_dir / "model.pt") if has_model else None,
        "has_checkpoint": has_model,
    }
