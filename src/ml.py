import pickle
from pathlib import Path
from typing import Any


def load_ml_model(path: str | Path) -> Any:
    """Load a pickled sklearn-style model."""
    p = Path(path)
    with p.open("rb") as f:
        return pickle.load(f)


def score_model(model: Any, x):
    """Return a scalar score for a feature vector."""
    if isinstance(model, dict) and "models" in model:
        model = model["models"].get("route") or list(model["models"].values())[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([x])
        if proba is not None and len(proba[0]) > 1:
            return float(proba[0][1])
        return float(proba[0][0])
    if hasattr(model, "predict"):
        return float(model.predict([x])[0])
    # fallback
    try:
        return float(model(x))
    except Exception:
        return 0.0
