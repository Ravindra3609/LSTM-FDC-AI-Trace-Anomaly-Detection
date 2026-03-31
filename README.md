# LSTM FDC-AI — Multivariate Time-Series Fault Detection

**Bidirectional LSTM Autoencoder with Temporal Attention for semiconductor equipment fault detection**
Learns normal equipment trace patterns → flags anomalies → shows *where* and *which sensor* caused the fault

---

## What this project does

| Layer | What | Why |
|---|---|---|
| **Data** | NASA CMAPSS turbofan degradation — 100 engines, 14 sensors, sliding 30-cycle windows | Direct analog to semiconductor fab equipment sensor traces |
| **BiLSTM Encoder** | Reads full 30-step sequence forward AND backward → hidden states at every time step | Captures temporal dependencies — pressure at step 28 is only anomalous given steps 1–27 |
| **Temporal Attention** | Learns a weight per time step — which cycles mattered most | Fault localization: attention concentrates on degradation time steps |
| **LSTM Decoder** | Reconstructs original sequence from latent context vector | Reconstruction error = anomaly score — trained on normal only |
| **Three error signals** | Sequence score (pass/fail) + per-sensor error (which?) + per-timestep error (when?) | Engineers need temporal + spatial fault attribution, not just a binary alarm |
| **FastAPI + Dashboard** | Live trace overlay, attention heatmap, step error bars, sensor ranking | Mirrors the trace anomaly visualization used in production FDC systems |

---

## Quick start — macOS M4 (3 commands)

```bash
cd ~/Desktop/lstm_fdc
chmod +x setup_and_run.sh
bash setup_and_run.sh
```

**Dashboard: http://localhost:8000**
**Swagger API docs: http://localhost:8000/docs**

Training takes **5–8 minutes** on M4 CPU. The script handles everything — venv, packages, data generation, training, and server launch.

---

## Manual setup

```bash
cd ~/Desktop/lstm_fdc

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py                                          # ~5-8 min on M4
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## Project structure

```
lstm_fdc/
│
├── data_pipeline.py        Data layer
│                             - NASA CMAPSS download (falls back to synthetic if offline)
│                             - Removes 7 near-constant sensors → keeps 14 useful
│                             - Min-max normalisation fitted on normal data only
│                             - Sliding window: 30 timesteps × 14 sensors per sequence
│                             - Fault label: last 30 cycles before failure = fault (1)
│
├── lstm_autoencoder.py     Core model
│                             - TemporalAttention: Bahdanau-style additive attention
│                             - LSTMAutoencoder: BiLSTM encoder + attention + LSTM decoder
│                             - train(): trains on normal sequences only (80 epochs)
│                             - compute_threshold(): 95th percentile of normal error
│                             - evaluate(): F1, AUC-ROC, AUC-PR on test set
│                             - save_model() / load_model()
│
├── train.py                One-shot training script
│                             - build_dataset() → train() → threshold → evaluate() → save
│
├── api.py                  FastAPI backend
│                             - Loads model + meta + scaler at startup
│                             - run_inference(): full output including trace data for UI
│                             - GET /demo, POST /predict, GET /metrics, GET /sensor_info
│
├── static/index.html       Dark-theme dashboard
│                             - Canvas-based trace overlay (input vs reconstruction)
│                             - Canvas-based attention heatmap
│                             - Canvas-based step reconstruction error bars
│                             - Sensor anomaly ranking sidebar
│
├── requirements.txt        Python dependencies
├── setup_and_run.sh        macOS one-shot setup + launch
└── models/                 Saved after train.py (auto-created)
    ├── lstm_ae.pt              PyTorch state dict + architecture config
    ├── meta.json               threshold, metrics, sensor_names, loss_history
    └── scaler_params.json      min/max per sensor for normalisation
```

---

## Architecture

```
Input sequence
(batch, 30, 14)
      │
      ▼
┌─────────────────────┐
│  BiLSTM Encoder     │  reads forward + backward
│  2 layers, dim=64   │  output: (batch, 30, 128)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Temporal Attention │  score each of 30 time steps
│  Bahdanau additive  │  weights: (batch, 30)  ← visualised as heatmap
│                     │  context: (batch, 128)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Latent projection  │  context → latent (batch, 32)
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  LSTM Decoder       │  initialised with latent, generates reconstruction
│  2 layers, dim=64   │  output: (batch, 30, 14)
└─────────────────────┘
      │
      ▼
Reconstruction error (MSE)

  seq_error    (batch,)       → anomaly score — compare to threshold
  sensor_error (batch, 14)    → which sensor is most anomalous
  step_error   (batch, 30)    → which time step is most anomalous
  attn_weights (batch, 30)    → where the encoder focused
```

---

## The three dashboard visualisations

### 1. Trace overlay chart
Blue solid = actual sensor input. Orange dashed = model's reconstruction.

- **Normal sequence**: lines closely track each other
- **Fault sequence**: lines diverge, especially toward later time steps

Click any sensor button (s2, s3, s4 ...) to switch between all 14 sensors.

### 2. Attention heatmap
Horizontal colour strip below the trace. One vertical slice per time step.

- **Dark** = low attention (model barely used this step)
- **Bright purple** = high attention (model heavily relied on this step)
- Top-3 highest-attention steps get a highlighted border

In fault sequences, attention concentrates on the degradation region — temporal fault localisation.

### 3. Step reconstruction error bars
Bar height = mean reconstruction error across all 14 sensors at that time step.

- **Red bars** = anomalous sequence
- **Green bars** = normal sequence
- Bars grow taller toward the fault zone (end of sequence)

This is exactly the trace anomaly visualisation used in production FDC systems at semiconductor fabs.

---

## API endpoints

```
GET  /              Serves dashboard UI
GET  /health        Model status, seq_len, n_sensors
GET  /metrics       F1, AUC-ROC, AUC-PR, threshold, loss_history
GET  /sensor_info   Sensor names and physical descriptions
POST /predict       Full inference on one (30 × 14) sequence
GET  /demo          Random test sample (filter: ?fault=true / ?fault=false)
GET  /docs          Swagger UI — interactive API docs
```

### POST /predict — request format

```json
{
  "sequence": [[...14 floats...], [...14 floats...]]
}
```
The sequence must be normalised (min-max scaled using scaler_params.json).
Shape: (30, 14) — 30 time steps × 14 sensors.

### Response format

```json
{
  "anomaly_score":  0.00842,
  "threshold":      0.00310,
  "is_anomaly":     true,
  "verdict":        "FAULT DETECTED",
  "fault_pct":      271.6,
  "input_trace":    [[...], ...],
  "recon_trace":    [[...], ...],
  "attn_weights":   [0.021, 0.019, ..., 0.087],
  "step_error":     [0.001, 0.002, ..., 0.019],
  "sensor_error":   [0.003, 0.012, ..., 0.008],
  "sensor_names":   ["s2", "s3", ...],
  "sensor_ranking": [{"sensor": "s14", "error": 0.031}, ...]
}
```

---

## The 14 sensors

| Sensor | Physical measurement | Fault behaviour |
|---|---|---|
| s2  | Fan inlet total temperature (°R) | Stable baseline |
| s3  | HPC outlet total temperature (°R) | Rises as compressor degrades |
| s4  | LPT outlet total temperature (°R) | Drops as turbine efficiency falls |
| s7  | Fan inlet total pressure (psia) | Operating condition indicator |
| s8  | Bypass duct total pressure (psia) | Increases slightly with degradation |
| s9  | HPC outlet total pressure (psia) | Drops as compressor health declines |
| s11 | HPC outlet static pressure (psia) | Correlated with s9 |
| s12 | Fuel flow ratio (pps/psi) | Increases — engine needs more fuel |
| s13 | Corrected fan speed (rpm) | Decreases with fan degradation |
| s14 | Corrected core speed (rpm) | Most diagnostic — clear downward trend |
| s15 | Bypass ratio | Drops steadily — strong fault signal |
| s17 | Bleed enthalpy | Rises as bleed system compensates |
| s20 | HPT coolant bleed flow | Increases as cooling demand grows |
| s21 | LPT coolant bleed flow | Increases alongside s20 |

---

## Expected training metrics

On synthetic CMAPSS data (generated offline):

| Metric | Typical range |
|---|---|
| F1 Score | 0.86 – 0.92 |
| AUC-ROC | 0.91 – 0.96 |
| AUC-PR | 0.83 – 0.92 |
| Threshold (95th pct normal) | varies by run |

---

## How this differs from FDC-AI (project 2)

| Dimension | FDC-AI (Project 2) | LSTM FDC (This project) |
|---|---|---|
| Input | Flat 56-feature vector (statistical summary) | Raw 30 × 14 sequence (the trace itself) |
| Temporal awareness | None — single time point | Full — LSTM sees 30 steps in order |
| Explainability | SHAP feature attribution | Temporal attention + per-sensor error |
| Visualization | SHAP waterfall bars | Trace overlay + attention heatmap |
| Fault localization | Which feature (averaged over time) | Which time step AND which sensor |
| Labels needed | Yes — XGBoost requires labeled faults | No — Autoencoder trains unsupervised |

Both approaches complement each other in production. This project detects temporal anomalies; the previous one classifies known fault categories with high confidence when labels exist.

---

## Troubleshooting

**Dashboard shows "Visit /docs" instead of UI**
Ensure `api.py` has `BASE_DIR = Path(__file__).parent` and serves `BASE_DIR / "static" / "index.html"`.

**No module named 'pyarrow'**
```bash
pip install pyarrow
```

**Training very slow**
Normal on CPU — 80 epochs takes 5–8 min on M4. Do not interrupt mid-training.

**Models not found on API startup**
Run `python train.py` before `uvicorn`. The `/models` directory must be populated first.

**Port 8000 already in use**
```bash
uvicorn api:app --port 8001
```

**Dataset download fails**
The pipeline automatically generates synthetic CMAPSS-like data offline. Check that `/data/cmapss_fd001.parquet` was created after running `train.py`.

---

## Resume line

> **LSTM FDC-AI Trace Anomaly Detection** — Built BiLSTM Autoencoder with temporal attention for multivariate equipment fault detection on NASA CMAPSS (100 engines, 14 sensors, 30-timestep sequences). Temporal attention mechanism provides time-step-level fault localization visualised as heatmap overlay on sensor traces. Deployed via FastAPI with interactive dashboard showing trace reconstruction overlay, attention heatmap, and per-sensor anomaly ranking — mirrors production FDC trace visualization used in semiconductor manufacturing.

**GitHub repo name:** `lstm-fdc-trace-anomaly`

**Tags:** `fault-detection` `FDC` `LSTM` `autoencoder` `attention` `time-series` `semiconductor` `anomaly-detection` `pytorch` `fastapi` `cmapss` `predictive-maintenance` `trace-visualization`

---

## Portfolio context

This is project 3 of a semiconductor AI portfolio:

| Project | Stack | Focus |
|---|---|---|
| Semiconductor SPC Dashboard | Python, Streamlit, Plotly | Process monitoring — WECO rules, Cp/Cpk |
| FDC-AI Fault Detection Pipeline | PyTorch, XGBoost, SHAP, FastAPI | Fault classification — hybrid ensemble |
| LSTM FDC Trace Anomaly *(this)* | PyTorch (LSTM+Attention), FastAPI | Temporal fault detection — trace visualization |
| WAT PMI AI Judge *(planned)* | XGBoost, SHAP, semi-supervised | Wafer yield prediction |
