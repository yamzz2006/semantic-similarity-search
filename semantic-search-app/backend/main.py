"""
main.py — FastAPI backend for the Semantic Similarity Search app.

Run with:
    uvicorn backend.main:app --reload

Then open http://127.0.0.1:8000 in your browser.
"""

from pathlib import Path
from io import StringIO

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .search_engine import SemanticSearchEngine

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
SAMPLE_DATA_PATH = BASE_DIR / "data" / "sample_dataset.csv"

app = FastAPI(title="Semantic Similarity Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SemanticSearchEngine()


@app.on_event("startup")
def startup() -> None:
    """Try to reload a previously trained model; otherwise train on the
    bundled sample dataset so the app is usable immediately."""
    if engine.load_from_disk():
        return
    if SAMPLE_DATA_PATH.exists():
        df = pd.read_csv(SAMPLE_DATA_PATH)
        try:
            engine.fit_from_dataframe(df)
        except Exception:
            pass  # app still boots; user can upload their own CSV


# --------------------------------------------------------------------- #
# API schemas
# --------------------------------------------------------------------- #
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)


# --------------------------------------------------------------------- #
# API routes
# --------------------------------------------------------------------- #
@app.get("/api/status")
def status():
    return {
        "ready": engine.is_ready,
        "num_documents": engine.num_documents,
    }


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    try:
        df = pd.read_csv(StringIO(raw.decode("utf-8", errors="ignore")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    try:
        info = engine.fit_from_dataframe(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    return {"status": "trained", **info}


@app.post("/api/search")
def search(req: SearchRequest):
    if not engine.is_ready:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded yet. Upload a CSV first.",
        )
    try:
        return engine.search(req.query, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------- #
# Frontend (static files)
# --------------------------------------------------------------------- #
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
