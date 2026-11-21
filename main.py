from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

app = FastAPI(title="NetVision AI Server", version="0.1.0")

class AnalyzeRequest(BaseModel):
    data: Dict[str, Any]

@app.get("/")
def root():
    return {"name": "NetVision AI Server", "status": "starter", "note": "Not a trained model"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    # Placeholder only. No cheat detection or trained model is implemented yet.
    return {"probability": 0.0, "label": "unclassified", "model_ready": False}
