"""
data_pipeline.py
================
NASA CMAPSS sequence builder for LSTM Autoencoder.
- Generates synthetic CMAPSS-like data (offline fallback)
- Downloads real CMAPSS FD001 if online
- Builds sliding-window sequences (seq_len × n_sensors)
- Splits into normal-only train and test with fault labels
- Normalises per sensor (min-max to [0, 1])
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
import urllib.request
import json

DATA_DIR   = Path("data")
SEQ_LEN    = 30          # time-steps per sequence
FAULT_ZONE = 30          # last N cycles before failure = fault
STRIDE     = 1           # sliding window stride (1 = every cycle)

# 14 useful sensors from CMAPSS FD001 (others are near-constant)
SENSOR_COLS = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
ALL_COLS    = ["unit","cycle","op1","op2","op3"] + [f"s{i}" for i in range(1,22)]


# ── Synthetic data generator ──────────────────────────────────────────────────

def generate_synthetic(n_units: int = 120, seed: int = 42) -> pd.DataFrame:
    rng  = np.random.default_rng(seed)
    rows = []
    for uid in range(1, n_units + 1):
        max_cyc = int(rng.integers(150, 320))
        for cyc in range(1, max_cyc + 1):
            t   = cyc / max_cyc
            deg = t ** 1.7                           # accelerating degradation
            op  = rng.integers(0, 3)
            row = {"unit": uid, "cycle": cyc,
                   "op1": [0.0,0.42,1.0][op] + rng.normal(0,.01),
                   "op2": [0.0,14.0,25.0][op] + rng.normal(0,.1),
                   "op3": [100.,84.,60.][op] + rng.normal(0,.2)}
            # 14 sensors with realistic degradation signatures
            sigs = {
                "s2":  518.67  + rng.normal(0,.5),
                "s3":  642.68  + 12*deg  + rng.normal(0,.5),
                "s4":  1590.3  - 22*deg  + rng.normal(0,3),
                "s7":  14.62   + rng.normal(0,.01),
                "s8":  21.61   + 2.5*deg + rng.normal(0,.2),
                "s9":  554.36  + rng.normal(0,1),
                "s11": 2388.1  - 55*deg  + rng.normal(0,5),
                "s12": 9065.4  -110*deg  + rng.normal(0,10),
                "s13": 1.3     + .35*deg + rng.normal(0,.01),
                "s14": 47.47   + 6*deg   + rng.normal(0,.3),
                "s15": 521.66  + 4*deg   + rng.normal(0,.5),
                "s17": 2388.1  - 32*deg  + rng.normal(0,5),
                "s20": 8138.6  - 85*deg  + rng.normal(0,10),
                "s21": 8.4195  + .6*deg  + rng.normal(0,.05),
            }
            row.update(sigs)
            rows.append(row)
    df = pd.DataFrame(rows)
    # Add RUL and fault label
    mx = df.groupby("unit")["cycle"].max().rename("max_cyc")
    df = df.join(mx, on="unit")
    df["RUL"]   = df["max_cyc"] - df["cycle"]
    df["fault"] = (df["RUL"] <= FAULT_ZONE).astype(int)
    df.drop(columns=["max_cyc"], inplace=True)
    return df


def load_or_generate(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    cache = data_dir / "cmapss_fd001.parquet"
    data_dir.mkdir(exist_ok=True)
    if cache.exists():
        return pd.read_parquet(cache)

    # Try downloading real CMAPSS
    try:
        url = "https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip"
        import zipfile, io
        with urllib.request.urlopen(url, timeout=12) as r:
            zdata = r.read()
        with zipfile.ZipFile(io.BytesIO(zdata)) as z:
            for name in z.namelist():
                if "FD001" in name:
                    z.extract(name, data_dir)
        train_path = next(data_dir.glob("*train_FD001*"))
        df = pd.read_csv(train_path, sep=r"\s+", header=None,
                         names=ALL_COLS, na_values=["NaN"])
        df["RUL"]   = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
        df["fault"] = (df["RUL"] <= FAULT_ZONE).astype(int)
        df.to_parquet(cache)
        print(f"CMAPSS downloaded — {len(df)} rows")
        return df
    except Exception:
        print("Offline — generating synthetic CMAPSS data...")
        df = generate_synthetic()
        df.to_parquet(cache)
        return df


# ── Normalisation ─────────────────────────────────────────────────────────────

def fit_scaler(df: pd.DataFrame) -> Dict:
    """Min-max scaler params per sensor fitted on normal data."""
    normal = df[df["fault"] == 0]
    params = {}
    for s in SENSOR_COLS:
        lo, hi = normal[s].min(), normal[s].max()
        params[s] = {"min": float(lo), "max": float(hi + 1e-9)}
    return params


def apply_scaler(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    out = df.copy()
    for s in SENSOR_COLS:
        lo, hi = params[s]["min"], params[s]["max"]
        out[s] = (out[s] - lo) / (hi - lo)
        out[s] = out[s].clip(0, 1)
    return out


# ── Sequence builder ──────────────────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    stride:  int = STRIDE,
    normal_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding window over each unit's time series.
    Returns:
      seqs   (N, seq_len, n_sensors)   float32
      labels (N,)                       int   0=normal 1=fault
      units  (N,)                       int   unit id
    """
    seqs, labels, units = [], [], []
    for uid, grp in df.groupby("unit"):
        grp = grp.sort_values("cycle")
        vals  = grp[SENSOR_COLS].values.astype(np.float32)   # (T, S)
        flabs = grp["fault"].values

        for start in range(0, len(vals) - seq_len + 1, stride):
            end   = start + seq_len
            seq   = vals[start:end]
            label = int(flabs[end - 1])         # label = last step's fault
            if normal_only and label == 1:
                continue
            seqs.append(seq)
            labels.append(label)
            units.append(uid)

    return (np.array(seqs,  dtype=np.float32),
            np.array(labels, dtype=np.int64),
            np.array(units,  dtype=np.int64))


# ── Full dataset builder ──────────────────────────────────────────────────────

def build_dataset(seq_len: int = SEQ_LEN) -> Dict:
    df_raw = load_or_generate()

    # Split units: 80% train, 20% test
    units      = df_raw["unit"].unique()
    rng        = np.random.default_rng(0)
    rng.shuffle(units)
    split      = int(len(units) * 0.8)
    train_u    = set(units[:split])
    test_u     = set(units[split:])

    df_train = df_raw[df_raw["unit"].isin(train_u)].copy()
    df_test  = df_raw[df_raw["unit"].isin(test_u)].copy()

    # Fit scaler on train normal
    scaler_params = fit_scaler(df_train)
    df_train_sc   = apply_scaler(df_train, scaler_params)
    df_test_sc    = apply_scaler(df_test,  scaler_params)

    # Build sequences
    X_train, y_train, _ = build_sequences(df_train_sc, seq_len, normal_only=False)
    X_test,  y_test,  _ = build_sequences(df_test_sc,  seq_len, normal_only=False)

    # Normal-only sequences for autoencoder training
    X_normal = X_train[y_train == 0]

    print(f"Sequences — train: {X_train.shape}  test: {X_test.shape}")
    print(f"Normal train: {X_normal.shape}")
    print(f"Fault rate — train: {y_train.mean():.2%}  test: {y_test.mean():.2%}")
    print(f"Sensors: {len(SENSOR_COLS)}, Seq len: {seq_len}")

    # Save scaler for API
    json.dump(scaler_params, open("models/scaler_params.json", "w"), indent=2)

    return {
        "X_train":      X_train,
        "y_train":      y_train,
        "X_normal":     X_normal,
        "X_test":       X_test,
        "y_test":       y_test,
        "scaler_params":scaler_params,
        "seq_len":      seq_len,
        "n_sensors":    len(SENSOR_COLS),
        "sensor_names": SENSOR_COLS,
    }
