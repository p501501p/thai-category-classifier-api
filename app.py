from typing import Any
import logging
from pathlib import Path

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
		error_detail = type(error).__name__
		if isinstance(error, OSError):
			filename = Path(error.filename).name if error.filename else "unknown"
			error_detail = f"{error_detail}: {error.strerror} ({filename})"
		raise HTTPException(
			status_code=500,
			detail=f"Prediction failed ({error_detail})"
		) from error