from typing import Any
import logging

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Thai Category Classifier")
logger = logging.getLogger(__name__)


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
	try:
		from predict_api import predict as classify

		return classify(request, api_key)
	except HTTPException:
		raise
	except Exception as error:
		logger.exception("Prediction request failed")
		raise HTTPException(
			status_code=500,
			detail=f"Prediction failed ({type(error).__name__})"
		) from error