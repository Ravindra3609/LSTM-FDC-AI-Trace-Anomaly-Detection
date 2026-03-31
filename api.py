"""
api.py
======
FastAPI backend for LSTM FDC-AI inference.

Endpoints:
  GET  /            → serves dashboard UI
  GET  /health      → model status
  GET  /metrics     → training metrics + loss curve
  GET  /demo        → random test sequence inference (full trace data)
  POST /predict     → inference on a raw (seq_len × n_sensors) sequence
  GET  /sensor_info → sensor names and metadata
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from lstm_autoencoder import LSTMAutoencoder, load_model
from data_pipeline     import build_dataset, apply_scaler, SENSOR_COLS, SEQ_LEN

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="LSTM FDC-AI API",
    description="Multivariate Time-Series Fault Detection with LSTM Autoencoder + Attention",
    version="1.0.0",
)

# ── Global state ──────────────────────────────────────────────────────────────
model:         Optional[LSTMAutoencoder] = None
meta:          dict  = {}
scaler_params: dict  = {}
test_cache:    dict  = {}


@app.on_event("startup")
async def load_everything():
    global model, meta, scaler_params, test_cache

    try:
        model = load_model(MODELS_DIR / "lstm_ae.pt", device="cpu")
        meta  = json.load(open(MODELS_DIR / "meta.json"))
        scaler_params = json.load(open(MODELS_DIR / "scaler_params.json"))
        print("Model + metadata loaded.")
    except Exception as e:
        print(f"Warning: {e} — run python train.py first.")
        return

    # Cache test sequences for /demo
    try:
        ds = build_dataset(seq_len=meta["seq_len"])
        test_cache = {
            "X_test": ds["X_test"],
            "y_test": ds["y_test"],
        }
        print(f"Test cache ready: {len(ds['X_test'])} sequences.")
    except Exception as e:
        print(f"Could not build test cache: {e}")


# ── Schema ────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    sequence: List[List[float]]   # (seq_len, n_sensors) — raw scaled values


# ── Inference helper ──────────────────────────────────────────────────────────

def run_inference(seq_np: np.ndarray) -> dict:
    """Full inference on one sequence. seq_np: (seq_len, n_sensors)."""
    x = torch.FloatTensor(seq_np).unsqueeze(0)   # (1, T, S)
    model.eval()
    with torch.no_grad():
        out = model.full_output(x)

    recon      = out["reconstruction"][0].cpu().numpy()       # (T, S)
    attn_w     = out["attn_weights"][0].cpu().numpy()         # (T,)
    step_err   = out["step_error"][0].cpu().numpy()           # (T,)
    sensor_err = out["sensor_error"][0].cpu().numpy()         # (S,)
    seq_err    = float(out["seq_error"][0].cpu())

    threshold = meta.get("threshold", 0.01)
    is_fault  = bool(seq_err > threshold)

    # Per-sensor ranking
    sensor_names = meta.get("sensor_names", SENSOR_COLS)
    sensor_rank  = sorted(
        zip(sensor_names, sensor_err.tolist()),
        key=lambda x: x[1], reverse=True
    )

    return {
        "anomaly_score":  round(seq_err, 8),
        "threshold":      round(threshold, 8),
        "is_anomaly":     is_fault,
        "verdict":        "FAULT DETECTED" if is_fault else "NORMAL",
        "fault_pct":      round(seq_err / (threshold + 1e-12) * 100, 1),

        # Trace data for visualization
        "input_trace":    seq_np.tolist(),          # (T, S)
        "recon_trace":    recon.tolist(),           # (T, S)
        "attn_weights":   attn_w.tolist(),          # (T,)
        "step_error":     step_err.tolist(),        # (T,)
        "sensor_error":   sensor_err.tolist(),      # (S,)
        "sensor_names":   sensor_names,
        "sensor_ranking": [{"sensor": s, "error": round(e, 6)}
                           for s, e in sensor_rank],
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    p = BASE_DIR / "index.html"
    return p.read_text() if p.exists() else "<h1>LSTM FDC-AI</h1><p>Visit /docs</p>"


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "seq_len":      meta.get("seq_len", SEQ_LEN),
        "n_sensors":    meta.get("n_sensors", len(SENSOR_COLS)),
    }


@app.get("/metrics")
async def get_metrics():
    if not meta:
        raise HTTPException(503, "Run python train.py first.")
    return {
        "metrics":      meta.get("metrics", {}),
        "threshold":    meta.get("threshold"),
        "loss_history": meta.get("loss_history", []),
    }


@app.get("/sensor_info")
async def sensor_info():
    names = meta.get("sensor_names", SENSOR_COLS)
    descriptions = {
        "s2":  "Total temperature (°R) — fan inlet",
        "s3":  "Total temperature (°R) — HPC outlet",
        "s4":  "Total temperature (°R) — LPT outlet",
        "s7":  "Total pressure (psia) — fan inlet",
        "s8":  "Total pressure (psia) — bypass duct",
        "s9":  "Total pressure (psia) — HPC outlet",
        "s11": "Static pressure (psia) — HPC outlet",
        "s12": "Fuel flow ratio (pps/psi)",
        "s13": "Corrected fan speed (rpm)",
        "s14": "Corrected core speed (rpm)",
        "s15": "Bypass ratio",
        "s17": "Bleed enthalpy",
        "s20": "HPT coolant bleed",
        "s21": "LPT coolant bleed",
    }
    return [{"name": n, "description": descriptions.get(n, n)} for n in names]


@app.post("/predict")
async def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, "Run python train.py first.")
    seq = np.array(req.sequence, dtype=np.float32)
    exp_shape = (meta.get("seq_len", SEQ_LEN), meta.get("n_sensors", len(SENSOR_COLS)))
    if seq.shape != exp_shape:
        raise HTTPException(400, f"Expected shape {exp_shape}, got {seq.shape}")
    return run_inference(seq)


@app.get("/demo")
async def demo(fault: Optional[bool] = None):
    if model is None or not test_cache:
        raise HTTPException(503, "Run python train.py first.")

    X, y = test_cache["X_test"], test_cache["y_test"]

    if fault is True:
        idx_pool = np.where(y == 1)[0].tolist()
    elif fault is False:
        idx_pool = np.where(y == 0)[0].tolist()
    else:
        idx_pool = list(range(len(y)))

    if not idx_pool:
        raise HTTPException(404, "No samples matching filter.")

    idx      = random.choice(idx_pool)
    seq_np   = X[idx]
    result   = run_inference(seq_np)
    result["true_label"]     = int(y[idx])
    result["true_label_str"] = "FAULT" if y[idx] == 1 else "NORMAL"
    result["sample_index"]   = int(idx)
    return result
