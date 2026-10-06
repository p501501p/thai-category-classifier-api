from typing import Any

from fastapi import FastAPI, Header
from pydantic import BaseModel, Field

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
) -> dict[str, Any]:
	from predict_api import predict as classify

	return classify(request, api_key)