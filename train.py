"""
train.py
========
Train the LSTM Autoencoder and save models + metadata.
Run:  python train.py
Time: ~5–8 min on M4 (CPU-only, 80 epochs, 120 units)
"""

import json
import numpy as np
from pathlib import Path

from data_pipeline    import build_dataset, SEQ_LEN, SENSOR_COLS
from lstm_autoencoder import (LSTMAutoencoder, train, compute_threshold,
                               evaluate, save_model)


def main():
    print("=" * 60)
    print("  LSTM FDC — Training")
    print("=" * 60)

    Path("models").mkdir(exist_ok=True)

    # ── 1. Dataset ────────────────────────────────────────────────
    ds = build_dataset(seq_len=SEQ_LEN)

    # ── 2. Build model ────────────────────────────────────────────
    model = LSTMAutoencoder(
        n_sensors  = ds["n_sensors"],
        seq_len    = ds["seq_len"],
        hidden_dim = 64,
        latent_dim = 32,
        n_layers   = 2,
        dropout    = 0.2,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # ── 3. Train on normal sequences only ─────────────────────────
    print(f"\nTraining on {len(ds['X_normal'])} normal sequences...")
    history = train(
        model      = model,
        X_normal   = ds["X_normal"],
        epochs     = 80,
        batch_size = 64,
        lr         = 1e-3,
        device     = "cpu",
        verbose    = True,
    )

    # ── 4. Threshold ──────────────────────────────────────────────
    threshold = compute_threshold(model, ds["X_normal"], pct=95, device="cpu")

    # ── 5. Evaluate ───────────────────────────────────────────────
    metrics = evaluate(model, ds["X_test"], ds["y_test"], threshold)

    # ── 6. Save ───────────────────────────────────────────────────
    save_model(model)

    meta = {
        "threshold":    threshold,
        "metrics":      metrics,
        "sensor_names": ds["sensor_names"],
        "seq_len":      ds["seq_len"],
        "n_sensors":    ds["n_sensors"],
        "loss_history": [round(l, 8) for l in history],
    }
    json.dump(meta, open("models/meta.json", "w"), indent=2)

    print("\n" + "=" * 60)
    print("  Training complete.")
    print(f"  F1:       {metrics['f1']}")
    print(f"  AUC-ROC:  {metrics['auc_roc']}")
    print(f"  AUC-PR:   {metrics['auc_pr']}")
    print("  Run:  uvicorn api:app --port 8000 --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
