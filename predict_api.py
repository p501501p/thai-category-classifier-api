import json
import hmac
import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model  # type: ignore[reportMissingModuleSource]

from text_utils import thai_tokenize

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
API_KEY = os.environ.get("API_KEY")

vectorizer = joblib.load(ARTIFACT_DIR / "tfidf_vectorizer.joblib")
svd = joblib.load(ARTIFACT_DIR / "svd.joblib")
model = load_model(ARTIFACT_DIR / "ann_model.keras")
with (ARTIFACT_DIR / "metadata.json").open(encoding="utf-8") as metadata_file:
    metadata = json.load(metadata_file)

app = FastAPI(title="Thai Category Classifier")


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=30000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(
    request: PredictionRequest,
    api_key: str | None = Header(default=None, alias="X-API-Key")
) -> dict[str, int | str | float]:
    if not API_KEY:
        raise HTTPException(status_code=503, detail="API key is not configured")
    if api_key is None or not hmac.compare_digest(api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be blank")

    features = vectorizer.transform([text])
    reduced_features = svd.transform(features).astype("float32")
    probabilities = model.predict(reduced_features, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    category_id = int(metadata["index_to_category_id"][class_index])

    return {
        "category_id": category_id,
        "category": metadata["category_names"].get(str(category_id), ""),
        "confidence": float(probabilities[class_index])
    }