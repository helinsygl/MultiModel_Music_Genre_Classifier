"""
FastAPI backend for demo_ui.html.

Run:
  source venv/bin/activate
  uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import config
from data_loader import build_manifest
from predict import predict as predict_genre


app = FastAPI(title="Music Genre Classifier API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    # Helpful landing so users don't see {"detail":"Not Found"}.
    return {
        "service": "Music Genre Classifier API",
        "health": "/api/health",
        "predict": "/api/predict",
        "results": "/api/results",
        "stats": "/api/stats",
        "docs": "/docs",
    }


@app.get("/api/results")
def results():
    path = os.path.join(config.RESULTS_DIR, "results_summary.json")
    if not os.path.isfile(path):
        return {"models": [], "path": path}
    with open(path, "r", encoding="utf-8") as f:
        models = json.load(f)
    return {"models": models, "path": path}


@app.get("/api/stats")
def stats():
    df_main = build_manifest(dataset_root=config.MAIN_DATASET_ROOT, max_per_genre=None)
    df_cross = build_manifest(dataset_root=config.CROSS_DATASET_ROOT, max_per_genre=None)
    return {
        "main": {
            "total": int(len(df_main)),
            "per_genre": df_main["genre"].value_counts().to_dict(),
        },
        "cross": {
            "total": int(len(df_cross)),
            "per_genre": df_cross["genre"].value_counts().to_dict(),
        },
        "genres": config.GENRES,
    }


@app.post("/api/predict")
async def predict(
    text: Optional[str] = Form(default=None),
    top_k: int = Form(default=6),
    audio: Optional[UploadFile] = File(default=None),
):
    text = (text or "").strip()

    audio_path = None
    tmp_path = None
    try:
        if audio is not None:
            suffix = os.path.splitext(audio.filename or "")[1] or ".mp3"
            fd, tmp_path = tempfile.mkstemp(prefix="upload_", suffix=suffix)
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(await audio.read())
            audio_path = tmp_path

        # Map UI -> predict.py logic:
        # - audio + text => multimodal
        # - audio only   => audio unimodal
        # - text only    => lyrics unimodal
        label, conf, top = predict_genre(
            audio_path=audio_path,
            lyrics_path=None,
            text=text if text else None,
            top_k=int(top_k),
        )

        if audio_path and text:
            modality = "Multimodal (audio + lyrics)"
        elif audio_path:
            modality = "Audio only"
        else:
            modality = "Lyrics only"

        return {
            "label": label,
            "confidence": float(conf),
            "top_k": [{"genre": g, "prob": float(p)} for g, p in top],
            "modality": modality,
        }
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

