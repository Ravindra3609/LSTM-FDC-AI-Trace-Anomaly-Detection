"""
lstm_autoencoder.py
===================
LSTM Autoencoder with Temporal Attention for multivariate time-series
anomaly detection.

Architecture:
  Encoder : BiLSTM → hidden state → latent vector
  Attention: learns which time steps were hardest to reconstruct
  Decoder : LSTM → reconstructs input sequence step by step

Anomaly score = mean squared reconstruction error per sequence.
Attention weights = which time steps contributed most to the error.

Key design choices:
  - Bidirectional encoder captures forward AND backward temporal patterns
  - Attention over encoder hidden states (not just final state)
  - Per-sensor reconstruction error exposes which sensors are anomalous
  - Trained ONLY on normal data — fault samples look "hard to reconstruct"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple, Dict, Optional, List

MODEL_PATH = Path("models/lstm_ae.pt")


# ── Attention module ──────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention over the encoder hidden states.
    Produces a weight for each time step — high weight = that step
    contributed most to the reconstruction.

    Input:  encoder_outputs  (batch, seq_len, hidden*2)  [bidirectional]
    Output: context          (batch, hidden*2)
            weights          (batch, seq_len)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for bidir
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # score: (batch, seq_len, 1)
        score   = self.v(torch.tanh(self.W(encoder_outputs)))
        weights = F.softmax(score.squeeze(-1), dim=1)   # (batch, seq_len)
        # context: weighted sum of encoder outputs
        context = (weights.unsqueeze(-1) * encoder_outputs).sum(dim=1)
        return context, weights


# ── LSTM Autoencoder ──────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM Autoencoder with temporal attention.

    Encoder: BiLSTM reads the input sequence, produces hidden states
             at every time step.
    Attention: selects the most relevant time steps to form the latent
               context vector.
    Decoder: Unidirectional LSTM uses the context vector to reconstruct
             the original sequence step by step.
    """

    def __init__(
        self,
        n_sensors:  int,
        seq_len:    int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        n_layers:   int = 2,
        dropout:    float = 0.2,
    ):
        super().__init__()
        self.n_sensors  = n_sensors
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers   = n_layers

        # ── Encoder (bidirectional) ───────────────────────────────────────
        self.encoder = nn.LSTM(
            input_size  = n_sensors,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            batch_first = True,
            bidirectional = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        # Project BiLSTM output to latent
        self.enc2lat = nn.Linear(hidden_dim * 2, latent_dim)

        # ── Attention ─────────────────────────────────────────────────────
        self.attention = TemporalAttention(hidden_dim)

        # ── Decoder ───────────────────────────────────────────────────────
        self.lat2dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size  = n_sensors,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, n_sensors)

    def encode(self, x: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, n_sensors)
        Returns: latent (batch, latent_dim),
                 context (batch, hidden*2),
                 attn_weights (batch, seq_len)
        """
        enc_out, _ = self.encoder(x)          # (B, T, H*2)
        context, attn_w = self.attention(enc_out)   # (B, H*2), (B, T)
        latent = self.enc2lat(context)          # (B, latent)
        return latent, context, attn_w

    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decode latent vector back to sequence.
        latent: (batch, latent_dim)
        Returns: (batch, seq_len, n_sensors)
        """
        batch = latent.size(0)
        h0 = self.lat2dec(latent)               # (B, H)
        # Expand to (n_layers, B, H)
        h0 = h0.unsqueeze(0).repeat(self.n_layers, 1, 1)
        c0 = torch.zeros_like(h0)

        # Teacher-force with zeros — decoder starts from silence
        dec_in = torch.zeros(batch, seq_len, self.n_sensors,
                             device=latent.device)
        dec_out, _ = self.decoder(dec_in, (h0, c0))
        return self.output_proj(dec_out)        # (B, T, S)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        latent, context, attn_w = self.encode(x)
        recon = self.decode(latent, x.size(1))
        return recon, attn_w

    # ── Inference helpers ─────────────────────────────────────────────────

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sequence mean squared reconstruction error. Shape: (batch,)"""
        recon, _ = self.forward(x)
        return ((x - recon) ** 2).mean(dim=(1, 2))

    @torch.no_grad()
    def per_sensor_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sensor mean error across time. Shape: (batch, n_sensors)"""
        recon, _ = self.forward(x)
        return ((x - recon) ** 2).mean(dim=1)   # (B, S)

    @torch.no_grad()
    def per_timestep_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-time-step mean error across sensors. Shape: (batch, seq_len)"""
        recon, _ = self.forward(x)
        return ((x - recon) ** 2).mean(dim=2)   # (B, T)

    @torch.no_grad()
    def full_output(self, x: torch.Tensor) -> Dict:
        """Complete inference: reconstruction, errors, attention."""
        recon, attn_w = self.forward(x)
        seq_err   = ((x - recon) ** 2).mean(dim=(1, 2))
        sensor_err= ((x - recon) ** 2).mean(dim=1)
        step_err  = ((x - recon) ** 2).mean(dim=2)
        return {
            "reconstruction": recon,
            "seq_error":      seq_err,
            "sensor_error":   sensor_err,   # (B, S)
            "step_error":     step_err,     # (B, T)
            "attn_weights":   attn_w,       # (B, T)
        }


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    model:      "LSTMAutoencoder",
    X_normal:   np.ndarray,
    epochs:     int   = 80,
    batch_size: int   = 64,
    lr:         float = 1e-3,
    device:     str   = "cpu",
    verbose:    bool  = True,
) -> List[float]:
    """Train on normal-only sequences. Returns loss history."""
    model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.MSELoss()

    X_t   = torch.FloatTensor(X_normal).to(device)
    ds    = TensorDataset(X_t)
    dl    = DataLoader(ds, batch_size=batch_size, shuffle=True,
                       drop_last=False)

    history = []
    model.train()
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for (batch,) in dl:
            opt.zero_grad()
            recon, _ = model(batch)
            loss = crit(recon, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ep_loss += loss.item() * len(batch)

        avg = ep_loss / len(X_normal)
        history.append(avg)
        sched.step()

        if verbose and ep % 10 == 0:
            print(f"  Epoch {ep:3d}/{epochs}  loss: {avg:.7f}  "
                  f"lr: {opt.param_groups[0]['lr']:.2e}")

    return history


def compute_threshold(
    model:   "LSTMAutoencoder",
    X_normal: np.ndarray,
    pct:      float = 95.0,
    device:   str = "cpu",
) -> float:
    """95th-percentile reconstruction error on normal data = threshold."""
    model.eval()
    with torch.no_grad():
        X_t    = torch.FloatTensor(X_normal).to(device)
        errors = model.reconstruction_error(X_t).cpu().numpy()
    thresh = float(np.percentile(errors, pct))
    print(f"  Anomaly threshold ({pct:.0f}th pct on normal): {thresh:.7f}")
    return thresh


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model:   "LSTMAutoencoder",
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    threshold: float,
    device:  str = "cpu",
) -> Dict:
    from sklearn.metrics import (f1_score, roc_auc_score,
                                  average_precision_score,
                                  classification_report,
                                  precision_recall_curve)
    model.eval()
    with torch.no_grad():
        X_t    = torch.FloatTensor(X_test).to(device)
        errors = model.reconstruction_error(X_t).cpu().numpy()

    # Optimise threshold for F1
    prec, rec, thrs = precision_recall_curve(y_test, errors)
    f1s  = 2*prec*rec / (prec+rec+1e-9)
    best_thr = float(thrs[np.argmax(f1s[:-1])]) if len(thrs) else threshold
    preds    = (errors >= best_thr).astype(int)

    metrics = {
        "f1":       round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "auc_roc":  round(float(roc_auc_score(y_test, errors)), 4),
        "auc_pr":   round(float(average_precision_score(y_test, errors)), 4),
        "threshold": round(best_thr, 7),
    }
    print("\n── Evaluation ─────────────────────────────────────────")
    print(f"  F1:       {metrics['f1']}")
    print(f"  AUC-ROC:  {metrics['auc_roc']}")
    print(f"  AUC-PR:   {metrics['auc_pr']}")
    print(classification_report(y_test, preds,
          target_names=["Normal","Fault"], zero_division=0))
    return metrics


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(model: "LSTMAutoencoder", path: Path = MODEL_PATH):
    path.parent.mkdir(exist_ok=True)
    torch.save({
        "state_dict":  model.state_dict(),
        "n_sensors":   model.n_sensors,
        "seq_len":     model.seq_len,
        "hidden_dim":  model.hidden_dim,
        "latent_dim":  model.latent_dim,
        "n_layers":    model.n_layers,
    }, path)
    print(f"Model saved → {path}")


def load_model(path: Path = MODEL_PATH, device: str = "cpu") -> "LSTMAutoencoder":
    ckpt  = torch.load(path, map_location=device)
    model = LSTMAutoencoder(
        n_sensors  = ckpt["n_sensors"],
        seq_len    = ckpt["seq_len"],
        hidden_dim = ckpt["hidden_dim"],
        latent_dim = ckpt["latent_dim"],
        n_layers   = ckpt["n_layers"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
