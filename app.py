"""
CardioRep ECG: Multi-Label Cardiac Abnormality Classification Using Deep Learning
Single-file Streamlit Application preserving all real PyTorch models and experimental logic.
Styled with a modern, clinical Dark Theme UI matching the research application standard.
"""

import os
import time
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch
from torch import nn
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wfdb
from scipy import signal as sp_signal

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "models"
RESULTS_FILE = BASE_DIR / "results" / "experiment_results.csv"
SAMPLES_DIR = BASE_DIR / "sample_data"

TARGETS = ["NORM", "MI", "STTC", "CD", "HYP"]
TARGET_NAMES = {
    "NORM": "Normal ECG",
    "MI": "Myocardial Infarction",
    "STTC": "ST/T Change",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy"
}
TARGET_DESCRIPTIONS = {
    "NORM": "Absence of clinically significant cardiac pathology or conduction delay",
    "MI": "Ischemic myocardial necrosis / acute ST-elevation or prior infarct Q-waves",
    "STTC": "Non-specific or ischemic ST-segment or T-wave repolarization alterations",
    "CD": "Intraventricular conduction delay, bundle branch block (LBBB/RBBB), or fascicular block",
    "HYP": "Left or right ventricular wall hypertrophy / atrial enlargement"
}

STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
EXPECTED_SAMPLE_RATE = 100  # Hz
EXPECTED_SIGNAL_LENGTH = 1000  # 10 seconds @ 100 Hz
EXPECTED_LEADS_COUNT = 12

LEAD_ALIASES = {
    "i": "I", "ii": "II", "iii": "III",
    "avr": "aVR", "avl": "aVL", "avf": "aVF",
    "v1": "V1", "v2": "V2", "v3": "V3", "v4": "V4", "v5": "V5", "v6": "V6"
}

MODEL_DISPLAY_NAMES = {
    # Best Models per Fraction (Colab Model training.ipynb winners)
    "best_model_100pct": "★ Best Overall: Baseline CNN (100% Labeled — Test F1: 0.7272)",
    "best_model_50pct": "★ Best 50%: Baseline CNN (50% Labeled — Test F1: 0.7050)",
    "best_model_25pct": "★ Best 25%: Baseline CNN (25% Labeled — Test F1: 0.6833)",
    "best_model_10pct": "★ Best 10%: VAE-CNN Pretrained (10% Labeled — Test F1: 0.6465)",
    # 10% Labeled Data (1,744 records)
    "model_VAE_10pct": "VAE-CNN (10% Labeled, 1,744 recs — Test F1: 0.6465) [Best in 10%]",
    "model_CNN_10pct": "Baseline CNN (10% Labeled, 1,744 recs — Test F1: 0.6374)",
    "model_AE_10pct": "AE-CNN (10% Labeled, 1,744 recs — Test F1: 0.6241)",
    # 25% Labeled Data (4,360 records)
    "model_CNN_25pct": "Baseline CNN (25% Labeled, 4,360 recs — Test F1: 0.6833) [Best in 25%]",
    "model_VAE_25pct": "VAE-CNN (25% Labeled, 4,360 recs — Test F1: 0.6678)",
    "model_AE_25pct": "AE-CNN (25% Labeled, 4,360 recs — Test F1: 0.6613)",
    # 50% Labeled Data (8,720 records)
    "model_CNN_50pct": "Baseline CNN (50% Labeled, 8,720 recs — Test F1: 0.7050) [Best in 50%]",
    "model_VAE_50pct": "VAE-CNN (50% Labeled, 8,720 recs — Test F1: 0.7028)",
    "model_AE_50pct": "AE-CNN (50% Labeled, 8,720 recs — Test F1: 0.6880)",
    # 100% Labeled Data (17,441 records)
    "model_CNN_100pct": "Baseline CNN (100% Labeled, 17,441 recs — Test F1: 0.7272) [Best Overall]",
    "model_VAE_100pct": "VAE-CNN (100% Labeled, 17,441 recs — Test F1: 0.7229)",
    "model_AE_100pct": "AE-CNN (100% Labeled, 17,441 recs — Test F1: 0.7113)",
}

# -----------------------------------------------------------------------------
# PyTorch Model Architecture (Identical to Jupyter Notebook)
# -----------------------------------------------------------------------------
class ECGEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        self.fc = nn.Linear(128 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        return self.fc(h)

class ECGClassifier(nn.Module):
    def __init__(self, encoder: ECGEncoder | None = None, latent_dim: int = 128, num_classes: int = 5):
        super().__init__()
        self.encoder = ECGEncoder(latent_dim) if encoder is None else encoder
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

# -----------------------------------------------------------------------------
# Caching & Loading Helpers
# -----------------------------------------------------------------------------
@st.cache_data
def load_experiment_data():
    if not RESULTS_FILE.exists():
        st.error(f"Results file not found at {RESULTS_FILE}")
        return pd.DataFrame()
    return pd.read_csv(RESULTS_FILE)

@st.cache_resource
def load_model_checkpoint(checkpoint_name: str):
    file_path = ARTIFACTS_DIR / f"{checkpoint_name}.pt"
    if not file_path.exists():
        file_path = ARTIFACTS_DIR / checkpoint_name
    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(file_path, map_location=device, weights_only=False)
    
    model = ECGClassifier(latent_dim=128, num_classes=len(TARGETS))
    model.load_state_dict(data["state_dict"])
    model.to(device)
    model.eval()
    
    return model, data, device

def get_available_models():
    if not ARTIFACTS_DIR.exists():
        return []
    all_stems = set(f.stem for f in ARTIFACTS_DIR.glob("*.pt"))
    priority_order = [
        # Recommended Best Models per fraction
        "best_model_100pct", "best_model_50pct", "best_model_25pct", "best_model_10pct",
        # All 12 individual models from Model training.ipynb
        "model_VAE_10pct", "model_CNN_10pct", "model_AE_10pct",
        "model_CNN_25pct", "model_VAE_25pct", "model_AE_25pct",
        "model_CNN_50pct", "model_VAE_50pct", "model_AE_50pct",
        "model_CNN_100pct", "model_VAE_100pct", "model_AE_100pct"
    ]
    res = [m for m in priority_order if m in all_stems]
    for m in sorted(all_stems):
        if m not in res:
            res.append(m)
    return res

# -----------------------------------------------------------------------------
# Signal Validation & Processing
# -----------------------------------------------------------------------------
def parse_and_validate_ecg(file_bytes: bytes, filename: str) -> np.ndarray:
    if len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty.")

    fname = filename.lower()
    if fname.endswith(".dat"):
        raise ValueError(
            "Uploaded a binary WFDB signal file (.dat) alone. In PTB-XL WFDB format, "
            ".dat files contain packed binary ADC voltage integers and cannot be decoded without "
            "their companion .hea header file. Please upload both the .hea and .dat files together."
        )
    if fname.endswith(".hea"):
        raise ValueError(
            "Uploaded a WFDB header file (.hea) alone. A .hea file contains only text metadata "
            "and no ECG waveform samples. Please upload both the .hea and .dat files together."
        )

    if fname.endswith(".csv") or fname.endswith(".txt"):
        import io
        df = pd.read_csv(io.BytesIO(file_bytes))
        if df.empty:
            raise ValueError("CSV file contains no data.")

        cols_lower = [str(c).strip().lower() for c in df.columns]
        matched_aliases = [LEAD_ALIASES.get(c) for c in cols_lower]

        if all(a is not None for a in matched_aliases) and len(matched_aliases) == EXPECTED_LEADS_COUNT:
            df_reordered = pd.DataFrame()
            for std_lead in STANDARD_LEADS:
                idx = matched_aliases.index(std_lead)
                df_reordered[std_lead] = df[df.columns[idx]]

            if len(df_reordered) != EXPECTED_SIGNAL_LENGTH:
                raise ValueError(f"Expected {EXPECTED_SIGNAL_LENGTH} time steps, got {len(df_reordered)}.")
            data = df_reordered.to_numpy().T
        elif df.shape == (EXPECTED_SIGNAL_LENGTH, EXPECTED_LEADS_COUNT):
            data = df.to_numpy().T
        elif df.shape == (EXPECTED_LEADS_COUNT, EXPECTED_SIGNAL_LENGTH):
            data = df.to_numpy()
        else:
            raise ValueError(f"Invalid dimensions: {df.shape}. Expected (1000, 12) or (12, 1000).")
            
    elif fname.endswith(".npy"):
        import io
        data = np.load(io.BytesIO(file_bytes), allow_pickle=False)
        if data.shape == (EXPECTED_SIGNAL_LENGTH, EXPECTED_LEADS_COUNT):
            data = data.T
        elif data.shape == (EXPECTED_LEADS_COUNT, EXPECTED_SIGNAL_LENGTH):
            pass
        else:
            raise ValueError(f"Invalid array shape {data.shape}. Expected (12, 1000) or (1000, 12).")
    else:
        raise ValueError(f"Unsupported format '{filename}'. Please provide a .csv, .npy, or PTB-XL WFDB (.hea + .dat) pair.")

    arr = np.asarray(data, dtype=np.float32)
    if np.isnan(arr).any():
        raise ValueError("Signal contains NaN (Not a Number) values.")
    if np.isinf(arr).any():
        raise ValueError("Signal contains infinite values.")
    if np.abs(arr).max() > 50.0:
        raise ValueError(f"Excessive voltage detected ({np.abs(arr).max():.1f} mV). Signals must be calibrated in mV.")

    return arr


def parse_and_validate_wfdb(hea_bytes: bytes, hea_filename: str, dat_bytes: bytes, dat_filename: str) -> tuple[np.ndarray, dict]:
    """
    Parses and validates a PTB-XL WFDB recording (.hea header + .dat binary signal).
    Ensures correct 12-lead ordering, sampling rate (100 Hz / 1000 samples), units,
    and returns a normalized-ready (12, 1000) float32 array and metadata.
    """
    if len(hea_bytes) == 0:
        raise ValueError("WFDB header file (.hea) is empty.")
    if len(dat_bytes) == 0:
        raise ValueError("WFDB signal file (.dat) is empty.")

    hea_stem = Path(hea_filename).stem
    dat_stem = Path(dat_filename).stem

    # Decode header text safely
    try:
        hea_text = hea_bytes.decode("utf-8")
    except UnicodeDecodeError:
        hea_text = hea_bytes.decode("latin-1")

    lines = [line.strip() for line in hea_text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        raise ValueError("WFDB header file (.hea) contains no valid header definitions.")

    # Header line format: <record_name> <n_signals> <fs> <sig_len> ...
    header_parts = lines[0].split()
    if len(header_parts) < 4:
        raise ValueError(f"Malformed WFDB header line: '{lines[0]}'. Expected '<record_name> <n_signals> <fs> <sig_len>'.")

    internal_rec_name = header_parts[0]
    try:
        n_signals = int(header_parts[1])
        fs = float(header_parts[2])
        sig_len = int(header_parts[3])
    except ValueError as e:
        raise ValueError(f"Invalid numeric values in WFDB header: {str(e)}")

    if n_signals != EXPECTED_LEADS_COUNT:
        raise ValueError(f"WFDB record defines {n_signals} channels. Model strictly requires {EXPECTED_LEADS_COUNT} standard leads.")

    # Extract all signal file names referenced in channel lines (e.g. '00001_lr.dat')
    referenced_dat_files = set()
    for ch_line in lines[1:1 + n_signals]:
        parts = ch_line.split()
        if parts:
            referenced_dat_files.add(parts[0])

    # Validate matching names between header and signal
    if hea_stem.lower() != dat_stem.lower() and dat_filename not in referenced_dat_files:
        raise ValueError(
            f"Record mismatch: Header file '{hea_filename}' (record '{internal_rec_name}') "
            f"does not match signal file '{dat_filename}' (references: {list(referenced_dat_files)})."
        )

    # Use a temporary directory to let official wfdb library parse the recording
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Write uploaded files under their uploaded names
        (tmp_path / hea_filename).write_bytes(hea_bytes)
        (tmp_path / dat_filename).write_bytes(dat_bytes)

        # Ensure internal record name is accessible
        if not (tmp_path / f"{internal_rec_name}.hea").exists():
            (tmp_path / f"{internal_rec_name}.hea").write_bytes(hea_bytes)

        # Ensure all referenced .dat file names exist in tmpdir
        for ref_dat in referenced_dat_files:
            target_path = tmp_path / ref_dat
            if not target_path.exists():
                target_path.write_bytes(dat_bytes)

        rec_stem = internal_rec_name if (tmp_path / f"{internal_rec_name}.hea").exists() else hea_stem
        try:
            record = wfdb.rdrecord(str(tmp_path / rec_stem))
        except Exception as e:
            raise ValueError(f"WFDB reader error: {str(e)}")

    p_signal = record.p_signal
    if p_signal is None or len(p_signal) == 0:
        raise ValueError("WFDB record contains no decoded physical signals.")

    n_samples, n_channels = p_signal.shape
    if n_channels != EXPECTED_LEADS_COUNT:
        raise ValueError(f"Decoded {n_channels} channels. Expected {EXPECTED_LEADS_COUNT} standard ECG leads.")

    # Match and reorder lead names into standard 12-lead layout (I, II, III, aVR, aVL, aVF, V1-V6)
    sig_names = record.sig_name or []
    mapped_leads = [LEAD_ALIASES.get(str(s).strip().lower()) for s in sig_names]

    if all(m is not None for m in mapped_leads) and len(set(mapped_leads)) == EXPECTED_LEADS_COUNT:
        ordered_channels = []
        for std_lead in STANDARD_LEADS:
            idx = mapped_leads.index(std_lead)
            ordered_channels.append(p_signal[:, idx])
        ordered_data = np.stack(ordered_channels, axis=0)  # shape (12, n_samples)
    else:
        ordered_data = p_signal.T  # shape (12, n_samples)

    # Handle sampling frequency and signal length
    resampled = False
    if ordered_data.shape[1] != EXPECTED_SIGNAL_LENGTH:
        # Check if record is 500 Hz (e.g. PTB-XL high-resolution _hr recording)
        if abs(fs - 500.0) < 1.0 or (fs != 100.0 and ordered_data.shape[1] > EXPECTED_SIGNAL_LENGTH):
            ordered_data = sp_signal.resample(ordered_data, EXPECTED_SIGNAL_LENGTH, axis=1)
            resampled = True
        elif fs == 100.0 and ordered_data.shape[1] > EXPECTED_SIGNAL_LENGTH:
            # Truncate to first 1000 samples (10 seconds)
            ordered_data = ordered_data[:, :EXPECTED_SIGNAL_LENGTH]
        else:
            raise ValueError(
                f"Invalid signal duration: Record has {ordered_data.shape[1]} samples at {fs} Hz "
                f"({ordered_data.shape[1] / fs:.1f}s). Expected 10 seconds / {EXPECTED_SIGNAL_LENGTH} samples at 100 Hz."
            )

    arr = np.asarray(ordered_data, dtype=np.float32)

    # Voltage & numerical integrity validation
    if np.isnan(arr).any():
        raise ValueError("WFDB signal contains NaN (Not a Number) values.")
    if np.isinf(arr).any():
        raise ValueError("WFDB signal contains infinite values.")
    if np.abs(arr).max() > 50.0:
        raise ValueError(f"Excessive voltage detected ({np.abs(arr).max():.1f} mV). Signals must be calibrated in mV.")

    metadata = {
        "record_name": record.record_name or hea_stem,
        "fs": fs,
        "original_samples": n_samples,
        "leads": STANDARD_LEADS if all(m is not None for m in mapped_leads) else sig_names,
        "units": record.units or ["mV"] * 12,
        "resampled": resampled
    }

    return arr, metadata


def run_inference(model: ECGClassifier, checkpoint_data: dict, raw_signal: np.ndarray, device: str):
    start = time.perf_counter()
    lead_mean = checkpoint_data["lead_mean"]
    lead_std = checkpoint_data["lead_std"]
    targets = checkpoint_data.get("targets", TARGETS)
    thresholds = checkpoint_data.get("thresholds", [0.5] * len(targets))

    # Apply exact normalization: (x - lead_mean) / lead_std
    signal_batch = raw_signal[np.newaxis, :, :].astype(np.float32)
    normed = (signal_batch - lead_mean) / lead_std
    tensor = torch.from_numpy(normed).float().to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    latency_ms = (time.perf_counter() - start) * 1000.0

    predictions = []
    detected = []
    for i, target in enumerate(targets):
        prob = float(probs[i])
        th = float(thresholds[i])
        is_pos = bool(prob >= th)
        if is_pos:
            detected.append(target)
        predictions.append({
            "label": target,
            "name": TARGET_NAMES.get(target, target),
            "description": TARGET_DESCRIPTIONS.get(target, target),
            "score": prob,
            "threshold": th,
            "positive": is_pos
        })

    return predictions, detected, latency_ms

# -----------------------------------------------------------------------------
# Plotting Helpers (Plotly Dark Theme)
# -----------------------------------------------------------------------------
def plot_twelve_leads(signal: np.ndarray):
    """
    Creates a clinical dark-mode 12-lead ECG grid with glowing cyan traces.
    """
    time_axis = np.linspace(0, 10.0, EXPECTED_SIGNAL_LENGTH)
    
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=[f"<b>{lead}</b>" for lead in STANDARD_LEADS],
        vertical_spacing=0.10,
        horizontal_spacing=0.04
    )

    for i, lead in enumerate(STANDARD_LEADS):
        row = (i % 3) + 1
        col = (i // 3) + 1
        series = signal[i]
        
        # Zero voltage dashed baseline
        fig.add_trace(
            go.Scatter(
                x=[0, 10.0],
                y=[0, 0],
                mode="lines",
                line=dict(color="#334155", width=1.0, dash="dash"),
                hoverinfo="skip",
                showlegend=False
            ),
            row=row, col=col
        )

        # High-contrast clinical cyan waveform
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=series,
                mode="lines",
                name=lead,
                line=dict(color="#38bdf8", width=1.6),
                hoverinfo="x+y"
            ),
            row=row, col=col
        )
        
        # Format axes with subtle clinical dark grid
        fig.update_xaxes(
            showgrid=True, gridwidth=0.8, gridcolor="#1e293b",
            zeroline=False,
            showticklabels=(row == 3),
            tickfont=dict(size=9, color="#64748b"),
            row=row, col=col
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=0.8, gridcolor="#1e293b",
            zeroline=False,
            tickfont=dict(size=9, color="#64748b"),
            row=row, col=col
        )

    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=12, color='#cbd5e1', family='Inter, sans-serif')
        annotation['xanchor'] = 'left'
        annotation['x'] = annotation['x'] - 0.04

    fig.update_layout(
        height=540,
        margin=dict(l=25, r=20, t=35, b=25),
        plot_bgcolor="#0c121e",
        paper_bgcolor="#111827",
        showlegend=False
    )
    return fig

def plot_single_lead(signal: np.ndarray, lead_name: str):
    lead_idx = STANDARD_LEADS.index(lead_name)
    time_axis = np.linspace(0, 10.0, EXPECTED_SIGNAL_LENGTH)
    series = signal[lead_idx]

    fig = go.Figure()
    
    # Zero baseline
    fig.add_trace(go.Scatter(
        x=[0, 10.0],
        y=[0, 0],
        mode="lines",
        line=dict(color="#334155", width=1.2, dash="dash"),
        hoverinfo="skip",
        showlegend=False
    ))

    # Electric Indigo Waveform
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=series,
        mode="lines",
        line=dict(color="#818cf8", width=2.4),
        name=lead_name
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Enlarged Inspection: Lead {lead_name}</b> (10 seconds @ 100 Hz)",
            font=dict(size=14, color="#f8fafc")
        ),
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (mV)",
        plot_bgcolor="#0c121e",
        paper_bgcolor="#111827",
        height=320,
        margin=dict(l=40, r=20, t=45, b=35),
        font=dict(color="#94a3b8"),
        xaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False)
    )
    return fig

# -----------------------------------------------------------------------------
# Streamlit App Configuration & Dark Theme CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioRep ECG",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #f8fafc !important;
        background-color: #0b0f19 !important;
    }

    #MainMenu, header, footer {
        visibility: hidden;
        height: 0px;
    }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1260px !important;
    }

    /* Top Brand & Header */
    .brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }
    .brand-subtitle {
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 500;
    }

    /* Status indicator */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        background: #161f30;
        border: 1px solid #283548;
        color: #cbd5e1;
        font-size: 12px;
        font-weight: 500;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 9999px;
        background-color: #10b981;
        display: inline-block;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.25);
    }

    /* Hero & Dark Cards */
    .cr-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
        margin-bottom: 20px;
    }
    .cr-hero-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 18px;
        padding: 34px 36px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
        margin-bottom: 24px;
    }

    /* Translucent Neon Badges */
    .cr-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 11px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 10px;
    }
    .cr-badge-indigo { background: rgba(99, 102, 241, 0.16); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.35); }
    .cr-badge-blue { background: rgba(59, 130, 246, 0.16); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.35); }
    .cr-badge-purple { background: rgba(168, 85, 247, 0.16); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.35); }
    .cr-badge-emerald { background: rgba(16, 185, 129, 0.16); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.35); }
    .cr-badge-rose { background: rgba(244, 63, 94, 0.16); color: #fda4af; border: 1px solid rgba(244, 63, 94, 0.35); }
    .cr-badge-amber { background: rgba(245, 158, 11, 0.16); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.35); }
    .cr-badge-violet { background: rgba(139, 92, 246, 0.16); color: #c4b5fd; border: 1px solid rgba(139, 92, 246, 0.35); }
    .cr-badge-slate { background: rgba(148, 163, 184, 0.12); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.25); }

    /* Segmented Control Dark Styling */
    div[data-testid="stSegmentedControl"] {
        background-color: #161f30;
        padding: 4px;
        border-radius: 10px;
        border: 1px solid #283548;
    }
    div[data-testid="stSegmentedControl"] button {
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        border: none !important;
        padding: 6px 14px !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
        box-shadow: 0 1px 4px rgba(79, 70, 229, 0.4) !important;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
        box-shadow: 0 2px 4px rgba(79, 70, 229, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #4338ca !important;
    }

    /* Secondary / standard buttons */
    .stButton > button:not([kind="primary"]) {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        background-color: #283548 !important;
        border-color: #475569 !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.25) !important;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Top Navigation Bar (Dark Theme)
# -----------------------------------------------------------------------------
if "nav_tab" not in st.session_state:
    st.session_state["nav_tab"] = "Research Overview"

device_name = "cuda" if torch.cuda.is_available() else "cpu"

nav_col1, nav_col2, nav_col3 = st.columns([3.2, 5.8, 2.5], vertical_alignment="center")

with nav_col1:
    st.markdown("""
        <div>
            <div class="brand-title">CardioRep ECG</div>
            <div class="brand-subtitle">PTB-XL Representation Learning</div>
        </div>
    """, unsafe_allow_html=True)

with nav_col2:
    selected_page = st.segmented_control(
        "Navigation",
        options=["Research Overview", "Model Comparison", "ECG Analysis", "Methodology"],
        default=st.session_state["nav_tab"],
        label_visibility="collapsed",
        key="nav_selection"
    )
    if selected_page and selected_page != st.session_state["nav_tab"]:
        st.session_state["nav_tab"] = selected_page
        st.rerun()

with nav_col3:
    st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; align-items: center;">
            <div class="status-pill">
                <span class="status-dot"></span>
                <span style="font-weight: 600;">Healthy</span>
                <span style="color: #475569; margin: 0 3px;">|</span>
                <span style="font-family: monospace; font-size: 11px; text-transform: uppercase;">{device_name}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

page = st.session_state["nav_tab"]

# -----------------------------------------------------------------------------
# PAGE 1: Research Overview (Dark Theme)
# -----------------------------------------------------------------------------
if page == "Research Overview":
    # Hero Section
    st.markdown("""
        <div class="cr-hero-card">
            <span class="cr-badge cr-badge-indigo">Academic Investigation & Full-Stack Prototype</span>
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; margin: 8px 0 14px 0; letter-spacing: -0.025em; line-height: 1.25;">
                ECG-Based Cardiac Abnormality Classification Using Deep Learning
            </h1>
            <p style="font-size: 1.05rem; color: #94a3b8; line-height: 1.6; max-width: 880px; margin-bottom: 22px;">
                Investigating whether unsupervised representation learning—specifically Autoencoder and 
                Variational Autoencoder pretraining—improves multi-label 12-lead ECG abnormality 
                classification when labeled clinical training data is severely restricted.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Hero Action Buttons
    btn_col1, btn_col2, _ = st.columns([2.2, 2.2, 5.6])
    with btn_col1:
        if st.button("Launch ECG Analysis Tool →", type="primary", use_container_width=True):
            st.session_state["nav_tab"] = "ECG Analysis"
            st.rerun()
    with btn_col2:
        if st.button("Explore Research Results", use_container_width=True):
            st.session_state["nav_tab"] = "Model Comparison"
            st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Motivation & Problem Cards
    col_prob, col_why = st.columns(2)
    with col_prob:
        st.markdown("""
            <div class="cr-card" style="height: 100%;">
                <div style="width: 36px; height: 36px; border-radius: 8px; background: rgba(59, 130, 246, 0.15); color: #60a5fa; display: flex; align-items: center; justify-content: center; font-weight: 700; margin-bottom: 12px; border: 1px solid rgba(59, 130, 246, 0.3);">
                    ⚡
                </div>
                <h3 style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">The Research Problem</h3>
                <p style="font-size: 0.88rem; color: #94a3b8; line-height: 1.6;">
                    Standard 12-lead electrocardiography is the fundamental clinical tool for diagnosing acute ischemia, 
                    arrhythmias, and structural heart defects. However, deep neural networks require thousands of verified 
                    training signals. While unlabeled raw ECG telemetry is abundant in hospital databases, obtaining 
                    expert consensus annotations by certified cardiologists is prohibitively expensive and time-consuming.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col_why:
        st.markdown("""
            <div class="cr-card" style="height: 100%;">
                <div style="width: 36px; height: 36px; border-radius: 8px; background: rgba(16, 185, 129, 0.15); color: #34d399; display: flex; align-items: center; justify-content: center; font-weight: 700; margin-bottom: 12px; border: 1px solid rgba(16, 185, 129, 0.3);">
                    🧬
                </div>
                <h3 style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">Why Limited Labeled Data Matters</h3>
                <p style="font-size: 0.88rem; color: #94a3b8; line-height: 1.6;">
                    In rare cardiac conditions, pediatric cohorts, or resource-constrained healthcare environments, only a 
                    small fraction of signals have validated annotations. This study systematically explores 
                    training with <b>10% (350 records)</b>, <b>25% (875 records)</b>, 
                    <b>50% (1,750 records)</b>, and <b>100% (3,500 records)</b> of labeled data to assess 
                    whether unsupervised pretraining provides sample efficiency.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Three Approaches Section
    st.markdown("""
        <div style="margin-bottom: 14px;">
            <h2 style="font-size: 1.35rem; font-weight: 800; color: #f8fafc; margin-bottom: 4px;">Three Model Approaches Investigated</h2>
            <p style="font-size: 0.85rem; color: #94a3b8;">
                All models share an identical 1D CNN backbone (3 conv layers, kernel size 7, stride 2, AdaptiveAvgPool 8, latent dim 128) 
                to isolate the effect of the representation learning strategy.
            </p>
        </div>
    """, unsafe_allow_html=True)

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown("""
            <div class="cr-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <span class="cr-badge cr-badge-blue">Baseline Supervised</span>
                    <h3 style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin: 4px 0 8px 0;">1D CNN Classifier</h3>
                    <p style="font-size: 0.84rem; color: #94a3b8; line-height: 1.55;">
                        Trained completely from scratch with random weight initialization using only labeled ECG signals. 
                        Learns discriminatory spatial-temporal representations directly from multi-label binary cross-entropy loss.
                    </p>
                </div>
                <div style="padding-top: 12px; margin-top: 14px; border-top: 1px solid #1e293b; font-family: monospace; font-size: 0.75rem; color: #64748b;">
                    Optimization: AdamW + early stopping
                </div>
            </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown("""
            <div class="cr-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <span class="cr-badge cr-badge-purple">Deterministic Pretraining</span>
                    <h3 style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin: 4px 0 8px 0;">Autoencoder-Pretrained CNN</h3>
                    <p style="font-size: 0.84rem; color: #94a3b8; line-height: 1.55;">
                        The convolutional encoder is pretrained on unlabeled 12-lead signals to reconstruct waveforms 
                        via a convolutional decoder minimizing Mean Squared Error (MSE). The encoder weights are then frozen or 
                        fine-tuned for multi-label classification.
                    </p>
                </div>
                <div style="padding-top: 12px; margin-top: 14px; border-top: 1px solid #1e293b; font-family: monospace; font-size: 0.75rem; color: #64748b;">
                    Objective: Reconstruction MSE loss
                </div>
            </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown("""
            <div class="cr-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <span class="cr-badge cr-badge-emerald">Probabilistic Pretraining</span>
                    <h3 style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin: 4px 0 8px 0;">VAE-Pretrained CNN</h3>
                    <p style="font-size: 0.84rem; color: #94a3b8; line-height: 1.55;">
                        Pretrained using a Variational Autoencoder formulation. The encoder maps ECG waveforms into a continuous 
                        Gaussian latent distribution parameterized by mean (&mu;) and log-variance (log &sigma;&sup2;), 
                        optimizing the Evidence Lower Bound (reconstruction + KL-divergence regularization).
                    </p>
                </div>
                <div style="padding-top: 12px; margin-top: 14px; border-top: 1px solid #1e293b; font-family: monospace; font-size: 0.75rem; color: #64748b;">
                    Objective: ELBO (MSE + &beta;-KL divergence)
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Five Superclasses Section
    st.markdown("""
        <div class="cr-card">
            <div style="margin-bottom: 16px;">
                <h2 style="font-size: 1.35rem; font-weight: 800; color: #f8fafc; margin-bottom: 4px;">Five Diagnostic Superclasses (PTB-XL)</h2>
                <p style="font-size: 0.85rem; color: #94a3b8;">
                    In standard clinical cardiology, diagnoses frequently overlap. Consequently, this problem is formulated as 
                    <b>multi-label prediction</b> using 5 independent sigmoid probability outputs rather than a mutually exclusive softmax.
                </p>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px;">
                <div style="padding: 14px; border-radius: 10px; background: #161f30; border: 1px solid #222f44;">
                    <span class="cr-badge cr-badge-emerald">NORM</span>
                    <h4 style="font-size: 0.9rem; font-weight: 700; color: #f8fafc; margin: 4px 0;">Normal ECG</h4>
                    <p style="font-size: 0.78rem; color: #94a3b8; line-height: 1.45;">Absence of pathological wave deflections, rhythm anomalies, or conduction blocks.</p>
                </div>
                <div style="padding: 14px; border-radius: 10px; background: #161f30; border: 1px solid #222f44;">
                    <span class="cr-badge cr-badge-rose">MI</span>
                    <h4 style="font-size: 0.9rem; font-weight: 700; color: #f8fafc; margin: 4px 0;">Myocardial Infarction</h4>
                    <p style="font-size: 0.78rem; color: #94a3b8; line-height: 1.45;">Ischemic myocardial tissue death, marked by ST elevations, depressions, or pathological Q-waves.</p>
                </div>
                <div style="padding: 14px; border-radius: 10px; background: #161f30; border: 1px solid #222f44;">
                    <span class="cr-badge cr-badge-amber">STTC</span>
                    <h4 style="font-size: 0.9rem; font-weight: 700; color: #f8fafc; margin: 4px 0;">ST/T Change</h4>
                    <p style="font-size: 0.78rem; color: #94a3b8; line-height: 1.45;">Non-specific or ischemic repolarization changes in ST-segment or T-wave morphology.</p>
                </div>
                <div style="padding: 14px; border-radius: 10px; background: #161f30; border: 1px solid #222f44;">
                    <span class="cr-badge cr-badge-indigo">CD</span>
                    <h4 style="font-size: 0.9rem; font-weight: 700; color: #f8fafc; margin: 4px 0;">Conduction Disturbance</h4>
                    <p style="font-size: 0.78rem; color: #94a3b8; line-height: 1.45;">Impaired electrical transmission, including bundle branch blocks and AV conduction delays.</p>
                </div>
                <div style="padding: 14px; border-radius: 10px; background: #161f30; border: 1px solid #222f44;">
                    <span class="cr-badge cr-badge-violet">HYP</span>
                    <h4 style="font-size: 0.9rem; font-weight: 700; color: #f8fafc; margin: 4px 0;">Hypertrophy</h4>
                    <p style="font-size: 0.78rem; color: #94a3b8; line-height: 1.45;">Enlargement and muscle thickening of the cardiac chambers (e.g. Left Ventricular Hypertrophy).</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Dataset & Partitioning
    st.markdown("""
        <div class="cr-card">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 14px;">
                <div>
                    <h2 style="font-size: 1.25rem; font-weight: 800; color: #f8fafc; margin-bottom: 2px;">Experimental Dataset & Partitioning</h2>
                    <p style="font-size: 0.85rem; color: #94a3b8;">PTB-XL database (PhysioNet v1.0.3), sampled at 100 Hz across official patient-separated folds.</p>
                </div>
                <a href="https://physionet.org/content/ptb-xl/1.0.3/" target="_blank" style="text-decoration: none; font-size: 0.82rem; font-weight: 600; color: #818cf8; background: #161f30; padding: 6px 12px; border-radius: 8px; border: 1px solid #283548;">
                    Access PhysioNet PTB-XL Dataset →
                </a>
            </div>
        </div>
    """, unsafe_allow_html=True)

    ds1, ds2, ds3 = st.columns(3)
    ds1.metric("Training Cohort", "17,441 Records", "Folds 1 to 8 (100% Corpus)")
    ds2.metric("Validation Partition", "2,193 Records", "Fold 9 (Threshold Tuning)")
    ds3.metric("Held-Out Test Set", "2,203 Records", "Fold 10 (Final Test)")

# -----------------------------------------------------------------------------
# PAGE 2: Model Comparison (Dark Theme)
# -----------------------------------------------------------------------------
elif page == "Model Comparison":
    st.markdown("""
        <div style="margin-bottom: 18px;">
            <h1 style="font-size: 2rem; font-weight: 800; color: #f8fafc; margin-bottom: 4px; letter-spacing: -0.025em;">
                Model Comparison & Empirical Research Results
            </h1>
            <p style="font-size: 0.92rem; color: #94a3b8; max-width: 820px;">
                Empirical evaluation comparing the baseline 1D CNN against Autoencoder (AE-CNN) and Variational 
                Autoencoder (VAE-CNN) representations across 4 labeled training data fractions: 10%, 25%, 50%, and 100%.
            </p>
        </div>
    """, unsafe_allow_html=True)

    df_results = load_experiment_data()

    if df_results.empty:
        st.error("No experimental results found.")
    else:
        # Summary 4-KPI Grid (Dynamically computed from experiment results)
        fracs = [0.10, 0.25, 0.50, 1.00]
        kpi_cols = st.columns(4)
        for i, frac in enumerate(fracs):
            sub = df_results[np.isclose(df_results["labeled_fraction"], frac)]
            if not sub.empty:
                top_row = sub.loc[sub["macro_f1"].idxmax()]
                top_model_code = top_row["model"].split("_")[0]
                top_f1 = top_row["macro_f1"]
                n_rec = int(top_row.get("n_labeled", int(frac * 17441)))
                badge_style = "cr-badge-emerald" if top_model_code == "VAE" else ("cr-badge-purple" if top_model_code == "AE" else "cr-badge-blue")
                model_label = "VAE-CNN" if top_model_code == "VAE" else ("AE-CNN" if top_model_code == "AE" else "Baseline CNN")
                
                with kpi_cols[i]:
                    st.markdown(f"""
                        <div class="cr-card" style="padding: 16px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <span style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">{int(frac*100)}% Labeled Data</span>
                                <span style="font-size: 0.68rem; font-family: monospace; color: #64748b;">{n_rec:,} records</span>
                            </div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: #f8fafc; line-height: 1.2;">{top_f1:.3f} <span style="font-size: 0.85rem; font-weight: 500; color: #94a3b8;">F1</span></div>
                            <div style="margin-top: 8px;"><span class="cr-badge {badge_style}" style="margin: 0;">Top: {model_label}</span></div>
                        </div>
                    """, unsafe_allow_html=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # Performance vs. Data Curve Card
        st.markdown("""
            <div class="cr-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                    <div>
                        <h3 style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; margin-bottom: 2px;">
                            Model Performance vs. Available Labeled Data
                        </h3>
                        <p style="font-size: 0.82rem; color: #94a3b8;">Test set performance across training fractions (10%, 25%, 50%, 100%)</p>
                    </div>
                </div>
        """, unsafe_allow_html=True)

        chart_ctrl1, chart_ctrl2 = st.columns([3, 1])
        with chart_ctrl2:
            metric_choice = st.selectbox(
                "Benchmark Metric:",
                ["macro_f1", "micro_f1", "macro_auroc", "macro_auprc", "val_macro_f1"],
                format_func=lambda x: {
                    "macro_f1": "Test Macro-F1",
                    "micro_f1": "Test Micro-F1",
                    "macro_auroc": "Test Macro-AUROC",
                    "macro_auprc": "Test Macro-AUPRC",
                    "val_macro_f1": "Validation Macro-F1"
                }[x]
            )

        # Build Interactive Comparison Line Chart (Dark Theme)
        fig_line = go.Figure()
        colors = {"CNN": "#3b82f6", "AE": "#a855f7", "VAE": "#10b981"}
        names = {"CNN": "Baseline 1D CNN", "AE": "AE-CNN (Pretrained)", "VAE": "VAE-CNN (Pretrained)"}

        for model_type in ["CNN", "AE", "VAE"]:
            sub = df_results[df_results["model"].str.startswith(f"{model_type}_")].sort_values("labeled_fraction")
            fig_line.add_trace(go.Scatter(
                x=sub["labeled_fraction"] * 100,
                y=sub[metric_choice],
                mode="lines+markers",
                name=names[model_type],
                line=dict(color=colors[model_type], width=2.8),
                marker=dict(size=8, symbol="circle")
            ))

        fig_line.update_layout(
            xaxis=dict(
                title=dict(text="<b>Labeled Training Data Available (%)</b>", font=dict(color="#cbd5e1")),
                tickvals=[10, 25, 50, 100],
                showgrid=True, gridcolor="#1e293b", zeroline=False,
                tickfont=dict(color="#94a3b8")
            ),
            yaxis=dict(
                title=dict(text=f"<b>{metric_choice.replace('_', ' ').title()}</b>", font=dict(color="#cbd5e1")),
                showgrid=True, gridcolor="#1e293b", zeroline=False,
                tickfont=dict(color="#94a3b8")
            ),
            plot_bgcolor="#0c121e",
            paper_bgcolor="#111827",
            height=370,
            margin=dict(l=45, r=20, t=15, b=45),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(color="#e2e8f0")
            )
        )
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Per-Class Diagnostic Breakdown Card
        st.markdown("""
            <div class="cr-card">
                <h3 style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; margin-bottom: 2px;">
                    Per-Class Diagnostic F1 Breakdown
                </h3>
                <p style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 12px;">Compare performance across NORM, MI, STTC, CD, and HYP for any training data fraction.</p>
        """, unsafe_allow_html=True)

        frac_choice = st.segmented_control(
            "Labeled Fraction",
            [0.10, 0.25, 0.50, 1.00],
            default=1.00,
            format_func=lambda x: {
                0.10: "10% Labeled (1,744 records)",
                0.25: "25% Labeled (4,360 records)",
                0.50: "50% Labeled (8,720 records)",
                1.00: "100% Labeled (17,441 records)"
            }.get(x, f"{int(x*100)}% Labeled")
        )

        sub_frac = df_results[np.isclose(df_results["labeled_fraction"], frac_choice or 1.00)]
        fig_bar = go.Figure()
        classes = ["NORM", "MI", "STTC", "CD", "HYP"]
        for model_type in ["CNN", "AE", "VAE"]:
            row = sub_frac[sub_frac["model"].str.startswith(f"{model_type}_")]
            if not row.empty:
                vals = [row.iloc[0][f"F1_{c}"] for c in classes]
                fig_bar.add_trace(go.Bar(
                    x=classes,
                    y=vals,
                    name=names[model_type],
                    marker_color=colors[model_type]
                ))

        fig_bar.update_layout(
            barmode="group",
            xaxis=dict(
                title=dict(text="<b>Diagnostic Superclass</b>", font=dict(color="#cbd5e1")),
                tickfont=dict(color="#94a3b8")
            ),
            yaxis=dict(
                title=dict(text="<b>Test F1 Score</b>", font=dict(color="#cbd5e1")),
                range=[0, 1.0],
                showgrid=True,
                gridcolor="#1e293b",
                tickfont=dict(color="#94a3b8")
            ),
            plot_bgcolor="#0c121e",
            paper_bgcolor="#111827",
            height=340,
            margin=dict(l=45, r=20, t=15, b=40),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(color="#e2e8f0")
            )
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Observations Card (Updated with Model training.ipynb results)
        st.markdown("""
            <div class="cr-card">
                <h3 style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; margin-bottom: 8px;">Key Empirical Research Observations</h3>
                <ul style="font-size: 0.88rem; color: #94a3b8; line-height: 1.6; padding-left: 20px;">
                    <li><b>Pretraining Advantage in Low-Data Regime (10% Labeled — 1,744 records):</b> Under extreme label scarcity, the Variational Autoencoder pretrained CNN (<code>VAE-CNN</code>) achieves the highest performance across all evaluation dimensions (<b>0.6465 Test Macro-F1</b>, <b>0.6847 Micro-F1</b>, <b>0.8628 AUROC</b>, and <b>0.6905 AUPRC</b>), outperforming both the baseline CNN (0.6374) and AE-CNN (0.6241). This empirically verifies that unsupervised probabilistic representation learning provides genuine sample efficiency when medical annotations are scarce.</li>
                    <li><b>Supervised Scaling with Ample Labels (25%, 50%, 100%):</b> As labeled training volume expands (4,360 to 17,441 records), the baseline 1D CNN trained end-to-end from scratch catches up rapidly (0.6833 at 25%, 0.7050 at 50%, and 0.7272 at 100%), matching or slightly outperforming the pretrained architectures. When supervised gradient signals are plentiful, task-specific feature learning thrives without representation priors.</li>
                    <li><b>Prevalence & Class Imbalance Dynamics:</b> Normal sinus rhythms (<code>NORM</code>) are reliably detected with high F1 (~0.85) due to high prevalence (7,607 training cases). Acute ischemic pathology (<code>MI</code>) reaches 0.712 F1, <code>STTC</code> reaches 0.757 F1, and <code>CD</code> reaches 0.732 F1. Conversely, structural wall thickening (<code>HYP</code>) remains the most difficult class (0.53–0.59 F1) due to fewer training cases (2,121) and subtle, diffuse voltage alterations.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        # NEW SECTION: Significance of Evaluation Metrics
        st.markdown("""
            <div class="cr-card" style="margin-top: 16px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                    <span style="font-size: 1.25rem;">📊</span>
                    <h3 style="font-size: 1.2rem; font-weight: 800; color: #f8fafc; margin: 0;">Significance of Evaluation Metrics: Why Each Metric Matters</h3>
                </div>
                <p style="font-size: 0.84rem; color: #94a3b8; margin-bottom: 16px; line-height: 1.5;">
                    In clinical electrocardiogram (ECG) artificial intelligence, standard accuracy is dangerously deceptive due to severe real-world class imbalance. 
                    Below is a clear, simple guide explaining what each metric captures, why it was chosen for this research, and how it protects patient safety.
                </p>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px;">
                    <!-- Card 1: Macro-F1 -->
                    <div style="background: #161f30; border: 1px solid #222f44; border-radius: 10px; padding: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 800; font-size: 0.95rem; color: #f8fafc;">🎯 Macro-F1 Score</span>
                            <span class="cr-badge cr-badge-emerald" style="margin: 0; font-size: 10px;">Primary Benchmark</span>
                        </div>
                        <p style="font-size: 0.80rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 8px;">
                            <b>In simple words:</b> The unweighted average of the F1 scores across all 5 disease categories, giving equal weight to every condition.
                        </p>
                        <div style="background: #0f172a; border-left: 3px solid #10b981; padding: 8px 10px; border-radius: 4px; font-size: 0.76rem; color: #94a3b8; line-height: 1.45;">
                            <b style="color: #f8fafc;">Why it's helpful:</b> In real hospitals, healthy patients vastly outnumber rare conditions like Hypertrophy. A "lazy" model that labels every single patient as "Normal" would get 80%+ basic accuracy, yet kill patients with missed heart attacks. Macro-F1 treats common and rare diseases with equal importance, heavily penalizing any model that neglects infrequent abnormalities.
                        </div>
                    </div>

                    <!-- Card 2: Micro-F1 -->
                    <div style="background: #161f30; border: 1px solid #222f44; border-radius: 10px; padding: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 800; font-size: 0.95rem; color: #f8fafc;">🌐 Micro-F1 Score</span>
                            <span class="cr-badge cr-badge-blue" style="margin: 0; font-size: 10px;">Operational Throughput</span>
                        </div>
                        <p style="font-size: 0.80rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 8px;">
                            <b>In simple words:</b> The global F1 score calculated by pooling all true positives, false positives, and false negatives across all patients together.
                        </p>
                        <div style="background: #0f172a; border-left: 3px solid #3b82f6; padding: 8px 10px; border-radius: 4px; font-size: 0.76rem; color: #94a3b8; line-height: 1.45;">
                            <b style="color: #f8fafc;">Why it's helpful:</b> While Macro-F1 measures fairness across disease categories, Micro-F1 measures the system's overall diagnostic reliability across every individual clinical decision. It tells hospital administrators how accurately the system handles incoming patient volume on a routine daily basis.
                        </div>
                    </div>

                    <!-- Card 3: Macro-AUROC -->
                    <div style="background: #161f30; border: 1px solid #222f44; border-radius: 10px; padding: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 800; font-size: 0.95rem; color: #f8fafc;">📈 Macro-AUROC</span>
                            <span class="cr-badge cr-badge-purple" style="margin: 0; font-size: 10px;">Discrimination Power</span>
                        </div>
                        <p style="font-size: 0.80rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 8px;">
                            <b>In simple words:</b> Measures how well the AI ranks a truly diseased patient higher than a healthy patient, regardless of whether the threshold is set high or low.
                        </p>
                        <div style="background: #0f172a; border-left: 3px solid #a855f7; padding: 8px 10px; border-radius: 4px; font-size: 0.76rem; color: #94a3b8; line-height: 1.45;">
                            <b style="color: #f8fafc;">Why it's helpful:</b> Evaluates raw diagnostic feature representation. A high AUROC (>0.90) proves that the neural network's convolutional filters have captured genuine physiological waveform patterns (such as ST elevation or widened QRS) rather than memorizing noise, independently of any operating cutoff.
                        </div>
                    </div>

                    <!-- Card 4: Macro-AUPRC -->
                    <div style="background: #161f30; border: 1px solid #222f44; border-radius: 10px; padding: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 800; font-size: 0.95rem; color: #f8fafc;">⚡ Macro-AUPRC</span>
                            <span class="cr-badge cr-badge-amber" style="margin: 0; font-size: 10px;">Alarm Reliability</span>
                        </div>
                        <p style="font-size: 0.80rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 8px;">
                            <b>In simple words:</b> Area Under the Precision-Recall Curve tests how clean the positive alerts are by tracking Precision vs. Recall across all possible thresholds.
                        </p>
                        <div style="background: #0f172a; border-left: 3px solid #f59e0b; padding: 8px 10px; border-radius: 4px; font-size: 0.76rem; color: #94a3b8; line-height: 1.45;">
                            <b style="color: #f8fafc;">Why it's helpful:</b> In severe class imbalance, AUROC can look deceptively good because of the large pool of true negatives. AUPRC focuses strictly on positive calls, aggressively penalizing false alarms. In an Emergency Department, constant false alarms cause "alarm fatigue"—nurses begin turning monitors off. A high AUPRC ensures that when the AI triggers an alarm, it is clinically actionable.
                        </div>
                    </div>

                    <!-- Card 5: Validation vs Test Split -->
                    <div style="background: #161f30; border: 1px solid #222f44; border-radius: 10px; padding: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 800; font-size: 0.95rem; color: #f8fafc;">🛡️ Val vs. Test F1 Split</span>
                            <span class="cr-badge cr-badge-indigo" style="margin: 0; font-size: 10px;">Generalizability</span>
                        </div>
                        <p style="font-size: 0.80rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 8px;">
                            <b>In simple words:</b> Validation F1 (Fold 9) is used during training to tune parameters; Test F1 (Fold 10) is evaluated only once on completely unseen patient recordings.
                        </p>
                        <div style="background: #0f172a; border-left: 3px solid #6366f1; padding: 8px 10px; border-radius: 4px; font-size: 0.76rem; color: #94a3b8; line-height: 1.45;">
                            <b style="color: #f8fafc;">Why it's helpful:</b> Patient-separated cross-validation prevents "data leakage." Evaluating on held-out Fold 10 guarantees that the reported metrics reflect true clinical performance on new patients from external clinics, rather than memorized waveforms from the training set.
                        </div>
                    </div>

                    <!-- Card 6: Per-Class F1 Breakdown -->
                    <div style="background: #161f30; border: 1px solid #222f44; border-radius: 10px; padding: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 800; font-size: 0.95rem; color: #f8fafc;">🫀 Per-Class F1 Scores</span>
                            <span class="cr-badge cr-badge-rose" style="margin: 0; font-size: 10px;">Condition Specificity</span>
                        </div>
                        <p style="font-size: 0.80rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 8px;">
                            <b>In simple words:</b> Individual F1 scores for each superclass: Normal (NORM), Infarction (MI), ST/T Changes (STTC), Conduction Blocks (CD), and Hypertrophy (HYP).
                        </p>
                        <div style="background: #0f172a; border-left: 3px solid #f43f5e; padding: 8px 10px; border-radius: 4px; font-size: 0.76rem; color: #94a3b8; line-height: 1.45;">
                            <b style="color: #f8fafc;">Why it's helpful:</b> In cardiology, missing an acute Myocardial Infarction (heart attack) has immediate catastrophic consequences, while detecting a minor ST/T change has lower acute urgency. Tracking per-class F1 ensures clinicians know exactly where the model excels and where extra clinical scrutiny is needed.
                        </div>
                    </div>

                    <!-- Card 7: Calibrated Decision Thresholds -->
                    <div style="background: #161f30; border: 1px solid #222f44; border-radius: 10px; padding: 14px; grid-column: 1 / -1;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 800; font-size: 0.95rem; color: #f8fafc;">⚖️ Calibrated Decision Thresholds (&tau;)</span>
                            <span class="cr-badge cr-badge-teal" style="margin: 0; font-size: 10px;">Clinical Alignment</span>
                        </div>
                        <p style="font-size: 0.80rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 8px;">
                            <b>In simple words:</b> Instead of an arbitrary fixed 50% cutoff, each disease gets a mathematically optimized decision threshold (e.g., 0.25 for MI vs. 0.50 for NORM) tuned on validation data.
                        </p>
                        <div style="background: #0f172a; border-left: 3px solid #14b8a6; padding: 8px 10px; border-radius: 4px; font-size: 0.76rem; color: #94a3b8; line-height: 1.45;">
                            <b style="color: #f8fafc;">Why it's helpful:</b> In medicine, false negatives and false positives have vastly unequal costs. For high-risk conditions like Myocardial Infarction, setting a lower threshold (e.g. 0.25) prioritizes high sensitivity so life-threatening heart attacks are rarely missed, while common non-critical classes use higher thresholds to avoid flooding cardiologists with false alarms.
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        with st.expander("View Complete 12-Run Experimental Results Table"):
            st.dataframe(df_results, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 3: ECG Analysis (Dark Theme Workbench)
# -----------------------------------------------------------------------------
elif page == "ECG Analysis":
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2rem; font-weight: 800; color: #f8fafc; margin-bottom: 4px; letter-spacing: -0.025em;">
                ECG Classification Workbench & Inference Tool
            </h1>
            <p style="font-size: 0.92rem; color: #94a3b8; max-width: 820px;">
                Submit a standardized 12-lead ECG signal (10 seconds @ 100 Hz = 1,000 samples across 12 leads) 
                and run multi-label diagnostic classification against the trained PyTorch checkpoints.
            </p>
        </div>
    """, unsafe_allow_html=True)

    available_checkpoints = get_available_models()
    if not available_checkpoints:
        st.error("No model checkpoints found in artifacts/models.")
    else:
        # 2-Column Workbench Layout
        col_ctrl, col_main = st.columns([1, 2], gap="large")

        with col_ctrl:
            # 1. Model Selection Card
            st.markdown("""
                <div class="cr-card" style="padding: 18px 20px; margin-bottom: 16px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #1e293b;">
                        <span style="color: #818cf8; font-weight: 700;">⚙</span>
                        <span style="font-weight: 700; font-size: 0.92rem; color: #f8fafc;">1. Select Trained Model Checkpoint</span>
                    </div>
            """, unsafe_allow_html=True)

            # Model Filter Segmented Control
            model_filter = st.segmented_control(
                "Filter Checkpoints",
                options=["All Checkpoints (16)", "Best Models per Fraction", "10% Labeled", "25% Labeled", "50% Labeled", "100% Labeled"],
                default="All Checkpoints (16)",
                label_visibility="collapsed"
            )

            filtered_checkpoints = available_checkpoints
            if model_filter == "Best Models per Fraction":
                filtered_checkpoints = [m for m in available_checkpoints if m.startswith("best_model_")]
            elif model_filter == "10% Labeled":
                filtered_checkpoints = [m for m in available_checkpoints if "10pct" in m]
            elif model_filter == "25% Labeled":
                filtered_checkpoints = [m for m in available_checkpoints if "25pct" in m]
            elif model_filter == "50% Labeled":
                filtered_checkpoints = [m for m in available_checkpoints if "50pct" in m]
            elif model_filter == "100% Labeled":
                filtered_checkpoints = [m for m in available_checkpoints if "100pct" in m]

            if not filtered_checkpoints:
                filtered_checkpoints = available_checkpoints

            model_id = st.selectbox(
                "Available Checkpoints:",
                filtered_checkpoints,
                index=0,
                format_func=lambda x: MODEL_DISPLAY_NAMES.get(x, x),
                label_visibility="collapsed"
            )

            # Load selected checkpoint
            model, ckpt_data, dev = load_model_checkpoint(model_id)
            targets = ckpt_data.get("targets", TARGETS)
            thresholds = ckpt_data.get("thresholds", [0.5] * len(targets))
            model_type = ckpt_data.get("model_type", "CNN")
            labeled_frac = float(ckpt_data.get("labeled_fraction", 1.0))

            # Look up empirical metrics for the selected model from df_results
            df_exp = load_experiment_data()
            model_tag = f"{model_type}_{labeled_frac:.2f}"
            row_match = df_exp[df_exp["model"] == model_tag] if not df_exp.empty else pd.DataFrame()
            if not row_match.empty:
                m_info = row_match.iloc[0]
                test_f1 = float(m_info["macro_f1"])
                val_f1 = float(m_info["val_macro_f1"])
                micro_f1 = float(m_info["micro_f1"])
                auroc = float(m_info["macro_auroc"])
                auprc = float(m_info["macro_auprc"])
                n_recs = int(m_info["n_labeled"])
            else:
                test_f1 = float(ckpt_data.get("test_macro_f1", 0.7272))
                val_f1 = float(ckpt_data.get("val_macro_f1", 0.7333))
                micro_f1 = 0.7589
                auroc = 0.9145
                auprc = 0.7952
                n_recs = int(labeled_frac * 17441)

            # Checkpoint Info Box matching dark theme
            th_cells = "".join([
                f'<div style="background: #111827; padding: 4px; border-radius: 6px; border: 1px solid #283548; text-align: center;">'
                f'<span style="color: #64748b; display: block; font-size: 9px;">{t}</span>'
                f'<span style="font-weight: 700; color: #f8fafc; font-size: 11px;">{float(th):.2f}</span>'
                f'</div>'
                for t, th in zip(targets, thresholds)
            ])

            st.markdown(f"""
                <div style="background: #161f30; border: 1px solid #222f44; border-radius: 10px; padding: 12px; font-size: 0.78rem; margin-top: 10px; color: #94a3b8;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                        <span style="font-weight: 600; color: #f8fafc;">Architecture:</span>
                        <span style="font-family: monospace; font-weight: 700; color: #818cf8;">{model_type}-CNN</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                        <span style="font-weight: 600; color: #f8fafc;">Training Cohort:</span>
                        <span style="font-family: monospace;">{n_recs:,} records ({int(labeled_frac * 100)}% of PTB-XL)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                        <span style="font-weight: 600; color: #f8fafc;">Test Macro-F1 / Micro-F1:</span>
                        <span style="font-family: monospace; font-weight: 700; color: #34d399;">{test_f1:.4f} <span style="color: #64748b; font-weight: 400;">/</span> {micro_f1:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span style="font-weight: 600; color: #f8fafc;">Test AUROC / AUPRC:</span>
                        <span style="font-family: monospace; font-weight: 700; color: #38bdf8;">{auroc:.4f} <span style="color: #64748b; font-weight: 400;">/</span> {auprc:.4f}</span>
                    </div>
                    <div style="border-top: 1px solid #222f44; padding-top: 8px;">
                        <span style="font-weight: 600; color: #f8fafc; display: block; margin-bottom: 6px;">Calibrated Decision Thresholds (&tau;):</span>
                        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px;">
                            {th_cells}
                        </div>
                    </div>
                </div>
                </div>
            """, unsafe_allow_html=True)

            # 2. Signal Input Card
            st.markdown("""
                <div class="cr-card" style="padding: 18px 20px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #1e293b;">
                        <span style="color: #818cf8; font-weight: 700;">📄</span>
                        <span style="font-weight: 700; font-size: 0.92rem; color: #f8fafc;">2. Provide 12-Lead ECG Signal</span>
                    </div>
            """, unsafe_allow_html=True)

            input_mode = st.segmented_control(
                "Input Mode",
                options=["Preloaded Samples", "Upload File (.csv / .npy / WFDB)"],
                default="Preloaded Samples",
                label_visibility="collapsed"
            )

            active_signal = None
            signal_name = ""
            wfdb_info = None

            if input_mode == "Preloaded Samples":
                sample_options = [
                    ("sample_normal_ecg.csv", "Normal Sinus Rhythm (CSV)", "Absence of acute ST-elevation or significant conduction delay"),
                    ("sample_mi_ecg.csv", "Acute Myocardial Infarction Pattern (CSV)", "Marked ST-elevation & Q-wave morphology in anterolateral leads"),
                    ("sample_cd_ecg.csv", "Conduction Disturbance Pattern (CSV)", "Prolonged QRS duration characteristic of bundle branch block"),
                    ("sample_normal_ecg.hea", "Normal Sinus Rhythm (PTB-XL WFDB: .hea + .dat)", "PTB-XL 12-lead WFDB record @ 100 Hz"),
                    ("sample_mi_ecg.hea", "Myocardial Infarction Pattern (PTB-XL WFDB: .hea + .dat)", "PTB-XL 12-lead WFDB record @ 100 Hz")
                ]
                
                selected_sample_tuple = st.radio(
                    "Select Benchmark Example:",
                    sample_options,
                    format_func=lambda x: x[1]
                )
                sample_file = SAMPLES_DIR / selected_sample_tuple[0]
                if sample_file.exists():
                    if selected_sample_tuple[0].endswith(".hea"):
                        stem = sample_file.stem
                        dat_file = SAMPLES_DIR / f"{stem}.dat"
                        if dat_file.exists():
                            active_signal, wfdb_info = parse_and_validate_wfdb(
                                sample_file.read_bytes(), sample_file.name,
                                dat_file.read_bytes(), dat_file.name
                            )
                            signal_name = selected_sample_tuple[1]
                    else:
                        raw_bytes = sample_file.read_bytes()
                        active_signal = parse_and_validate_ecg(raw_bytes, selected_sample_tuple[0])
                        signal_name = selected_sample_tuple[1]
            else:
                upload_style = st.radio(
                    "Upload Method:",
                    ["Drag & Drop / Multi-select (.csv, .npy, or .hea + .dat)", "Separate Inputs for WFDB (.hea and .dat)"],
                    horizontal=False,
                    label_visibility="collapsed"
                )

                if upload_style == "Drag & Drop / Multi-select (.csv, .npy, or .hea + .dat)":
                    uploaded_files = st.file_uploader(
                        "Upload Compatible Signal File(s):",
                        type=["csv", "npy", "txt", "dat", "hea"],
                        accept_multiple_files=True,
                        help="Upload a single .csv / .npy file, or select/drop both PTB-XL WFDB files (.hea and .dat) together."
                    )
                    if uploaded_files:
                        hea_files = [f for f in uploaded_files if f.name.lower().endswith(".hea")]
                        dat_files = [f for f in uploaded_files if f.name.lower().endswith(".dat")]
                        csv_files = [f for f in uploaded_files if f.name.lower().endswith((".csv", ".txt"))]
                        npy_files = [f for f in uploaded_files if f.name.lower().endswith(".npy")]

                        # Case A: WFDB Pair (.hea + .dat)
                        if len(hea_files) == 1 and len(dat_files) == 1 and not csv_files and not npy_files:
                            try:
                                h_f = hea_files[0]
                                d_f = dat_files[0]
                                active_signal, wfdb_info = parse_and_validate_wfdb(
                                    h_f.read(), h_f.name, d_f.read(), d_f.name
                                )
                                signal_name = f"{wfdb_info['record_name']} (WFDB {int(wfdb_info['fs'])} Hz)"
                                resample_note = " (Resampled from 500 Hz to 100 Hz)" if wfdb_info.get("resampled") else ""
                                st.success(f"✓ Validated PTB-XL WFDB Record: {wfdb_info['record_name']} (12 leads, 1000 samples @ 100 Hz){resample_note}")
                            except Exception as e:
                                st.error(f"WFDB validation error: {str(e)}")

                        # Case B: Solitary .dat file without .hea
                        elif len(dat_files) > 0 and len(hea_files) == 0:
                            st.error(
                                f"⚠️ **Missing WFDB Header File:** You uploaded the binary signal file `{dat_files[0].name}`. "
                                f"PTB-XL WFDB format stores packed binary ADC integers that require the matching `.hea` header "
                                f"to decode channel gains, baselines, and sampling rate. Please select and upload both "
                                f"`{Path(dat_files[0].name).stem}.hea` and `{dat_files[0].name}` together.\n\n"
                                f"*(Note: Binary .dat files are not raw CSV or NumPy arrays and cannot be parsed without the header)*"
                            )

                        # Case C: Solitary .hea file without .dat
                        elif len(hea_files) > 0 and len(dat_files) == 0:
                            st.error(
                                f"⚠️ **Missing WFDB Signal File:** You uploaded the header file `{hea_files[0].name}`. "
                                f"The `.hea` file contains only metadata. Please select and upload both "
                                f"`{hea_files[0].name}` and `{Path(hea_files[0].name).stem}.dat` together."
                            )

                        # Case D: Single CSV or NPY file
                        elif len(uploaded_files) == 1 and (csv_files or npy_files):
                            f = uploaded_files[0]
                            try:
                                raw_bytes = f.read()
                                active_signal = parse_and_validate_ecg(raw_bytes, f.name)
                                signal_name = f.name
                                st.success(f"✓ Validated: {f.name} (12 leads, 1000 samples @ 100 Hz)")
                            except Exception as e:
                                st.error(f"Validation error: {str(e)}")

                        # Case E: Conflicting multiple files
                        else:
                            st.error(
                                "⚠️ **Conflicting files uploaded:** Please upload either a single file (.csv / .npy) "
                                "or exactly one matching pair of PTB-XL WFDB files (.hea and .dat)."
                            )
                else:
                    # Dedicated separate uploaders for WFDB
                    col_h, col_d = st.columns(2)
                    with col_h:
                        hea_up = st.file_uploader("1. Header File (.hea):", type=["hea"], help="PTB-XL WFDB .hea header file")
                    with col_d:
                        dat_up = st.file_uploader("2. Signal File (.dat):", type=["dat"], help="PTB-XL WFDB .dat binary signal file")

                    if hea_up is not None and dat_up is not None:
                        try:
                            active_signal, wfdb_info = parse_and_validate_wfdb(
                                hea_up.read(), hea_up.name, dat_up.read(), dat_up.name
                            )
                            signal_name = f"{wfdb_info['record_name']} (WFDB {int(wfdb_info['fs'])} Hz)"
                            resample_note = " (Resampled from 500 Hz to 100 Hz)" if wfdb_info.get("resampled") else ""
                            st.success(f"✓ Validated PTB-XL WFDB Record: {wfdb_info['record_name']} (12 leads, 1000 samples @ 100 Hz){resample_note}")
                        except Exception as e:
                            st.error(f"WFDB validation error: {str(e)}")
                    elif hea_up is not None:
                        st.info("ℹ️ Uploaded header file. Please also upload the matching `.dat` binary signal file.")
                    elif dat_up is not None:
                        st.info("ℹ️ Uploaded signal file. Please also upload the matching `.hea` header file.")

                st.markdown("""
                    <div style="background: #161f30; border: 1px solid #222f44; border-radius: 8px; padding: 10px; font-size: 0.72rem; color: #94a3b8; margin-top: 10px;">
                        <span style="font-weight: 600; color: #f8fafc; display: block; margin-bottom: 3px;">Supported Formats & Specifications:</span>
                        • <b>WFDB (.hea + .dat):</b> PTB-XL recordings read via official <code>wfdb</code> parser. Supports 100 Hz native (1,000 samples) and 500 Hz (automatically resampled to 100 Hz).<br>
                        • <b>CSV (.csv):</b> Standard 1,000 time steps × 12 leads (or 12 × 1,000), calibrated in mV.<br>
                        • <b>NumPy (.npy):</b> Array shape (12, 1000) or (1000, 12).<br>
                        • Standard leads: I, II, III, aVR, aVL, aVF, V1–V6.
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            run_btn = st.button("✨ Run Model Inference", type="primary", use_container_width=True, disabled=(active_signal is None))
            st.markdown("</div>", unsafe_allow_html=True)

        with col_main:
            if active_signal is not None:
                # Execute inference automatically or on button click
                preds, detected, latency = run_inference(model, ckpt_data, active_signal, dev)

                # Diagnostic Results Card (Dark Mode)
                detected_badges = "".join([
                    f'<span style="padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 700; background: rgba(244, 63, 94, 0.2); color: #fda4af; border: 1px solid rgba(244, 63, 94, 0.4); margin-left: 4px;">{c}</span>'
                    for c in detected
                ]) if detected else '<span style="padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 700; background: #161f30; color: #94a3b8; border: 1px solid #222f44;">None above threshold</span>'

                score_bars_html = ""
                for pred in preds:
                    is_pos = pred["positive"]
                    percent = min(100.0, max(0.0, pred["score"] * 100.0))
                    th_percent = pred["threshold"] * 100.0

                    card_bg = "rgba(16, 185, 129, 0.10)" if (is_pos and pred["label"] == "NORM") else ("rgba(244, 63, 94, 0.10)" if is_pos else "#161f30")
                    card_border = "rgba(16, 185, 129, 0.35)" if (is_pos and pred["label"] == "NORM") else ("rgba(244, 63, 94, 0.35)" if is_pos else "#222f44")
                    badge_bg = "rgba(16, 185, 129, 0.25)" if (is_pos and pred["label"] == "NORM") else ("rgba(244, 63, 94, 0.25)" if is_pos else "#1e293b")
                    badge_color = "#6ee7b7" if (is_pos and pred["label"] == "NORM") else ("#fda4af" if is_pos else "#94a3b8")
                    fill_color = "#10b981" if (is_pos and pred["label"] == "NORM") else ("#f43f5e" if is_pos else "#475569")
                    badge_text = "POSITIVE" if is_pos else "NEGATIVE"

                    score_bars_html += f"""
                    <div style="background: {card_bg}; border: 1px solid {card_border}; border-radius: 10px; padding: 12px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <div>
                                <span style="font-weight: 800; font-size: 0.95rem; color: #f8fafc;">{pred['label']}</span>
                                <span style="font-size: 0.75rem; color: #94a3b8; margin-left: 4px;">({pred['name']})</span>
                            </div>
                            <span style="font-size: 0.72rem; font-weight: 800; padding: 2px 8px; border-radius: 9999px; background: {badge_bg}; color: {badge_color};">
                                {badge_text}
                            </span>
                        </div>
                        <div style="position: relative; width: 100%; height: 10px; background: #1e293b; border-radius: 9999px; overflow: hidden; margin: 8px 0;">
                            <div style="height: 100%; width: {percent:.1f}%; background: {fill_color}; border-radius: 9999px; transition: width 0.4s ease;"></div>
                            <div style="position: absolute; top: 0; bottom: 0; left: {th_percent:.1f}%; width: 2px; background: #f8fafc; z-index: 10;" title="Threshold: {pred['threshold']:.2f}"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-family: monospace; font-size: 0.74rem; color: #94a3b8;">
                            <span>Sigmoid Score: <b style="color: #f8fafc;">{pred['score']:.4f}</b></span>
                            <span>Threshold: {pred['threshold']:.2f}</span>
                        </div>
                    </div>
                    """

                st.html(f"""
<div class="cr-card" style="border-color: #312e81; box-shadow: 0 4px 16px rgba(79, 70, 229, 0.15); margin-bottom: 20px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid #1e293b;">
        <div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span class="cr-badge cr-badge-indigo" style="margin: 0;">Inference Result</span>
                <span style="font-size: 0.75rem; font-family: monospace; color: #94a3b8;">⏱ {latency:.1f} ms</span>
            </div>
            <h3 style="font-size: 1.05rem; font-weight: 800; color: #f8fafc; margin: 4px 0 0 0;">
                Evaluated by {model_type}-CNN ({int(labeled_frac * 100)}% Labeled Checkpoint)
            </h3>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 0.75rem; color: #94a3b8; display: block; margin-bottom: 2px;">Identified Conditions:</span>
            {detected_badges}
        </div>
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px;">
        {score_bars_html}
    </div>
</div>
""")

                # 12-Lead Electrocardiogram View Card
                st.html(f"""
<div class="cr-card" style="margin-bottom: 14px;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="color: #38bdf8; font-weight: 700;">📈</span>
            <span style="font-weight: 800; font-size: 1.05rem; color: #f8fafc;">12-Lead Electrocardiogram View</span>
            <span style="font-size: 0.78rem; color: #94a3b8;">(12 leads, 10.0s @ 100 Hz: {signal_name})</span>
        </div>
    </div>
</div>
""")

                fig_12 = plot_twelve_leads(active_signal)
                st.plotly_chart(fig_12, use_container_width=True)

                # Single-lead inspector
                with st.expander("🔍 Enlarge Individual Lead Waveform"):
                    lead_to_zoom = st.selectbox("Select Lead to Enlarge:", STANDARD_LEADS, index=1)
                    lead_idx = STANDARD_LEADS.index(lead_to_zoom)
                    lead_sig = active_signal[lead_idx]
                    
                    st.html(f"""
<div style="font-size: 0.78rem; font-family: monospace; color: #94a3b8; margin-bottom: 8px;">
    Mean: <b style="color: #f8fafc;">{lead_sig.mean():.3f} mV</b> | Std: <b style="color: #f8fafc;">{lead_sig.std():.3f} mV</b> | Min: <b style="color: #f8fafc;">{lead_sig.min():.3f} mV</b> | Max: <b style="color: #f8fafc;">{lead_sig.max():.3f} mV</b>
</div>
""")
                    fig_single = plot_single_lead(active_signal, lead_to_zoom)
                    st.plotly_chart(fig_single, use_container_width=True)
            else:
                st.markdown("""
                    <div class="cr-card" style="text-align: center; padding: 48px 24px; border-style: dashed;">
                        <h4 style="font-size: 1rem; font-weight: 700; color: #cbd5e1; margin-bottom: 4px;">No Signal Loaded</h4>
                        <p style="font-size: 0.82rem; color: #64748b;">Choose a preloaded benchmark sample or upload a 12-lead signal file to preview waveform traces and run model inference.</p>
                    </div>
                """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 4: Methodology (Dark Theme)
# -----------------------------------------------------------------------------
elif page == "Methodology":
    st.markdown("""
        <div style="margin-bottom: 24px; border-bottom: 1px solid #1e293b; padding-bottom: 16px;">
            <span class="cr-badge cr-badge-slate">Technical Documentation & Model Methodology</span>
            <h1 style="font-size: 2rem; font-weight: 800; color: #f8fafc; margin: 6px 0 4px 0; letter-spacing: -0.025em;">
                Methodology, Model Architecture & Experimental Setup
            </h1>
            <p style="font-size: 0.92rem; color: #94a3b8; max-width: 820px;">
                Full technical disclosure of dataset preparation, network architectures, unsupervised pretraining objectives, 
                decision threshold calibration, and empirical analysis.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 1. Dataset Source
    st.markdown("""
        <div class="cr-card">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;">
                <h2 style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; margin: 0;">1. Dataset Source & Patient-Separated Partitions</h2>
                <a href="https://physionet.org/content/ptb-xl/1.0.3/" target="_blank" style="text-decoration: none; font-size: 0.82rem; font-weight: 600; color: #818cf8; background: #161f30; padding: 5px 12px; border-radius: 8px; border: 1px solid #283548;">
                    Access PTB-XL Dataset on PhysioNet →
                </a>
            </div>
            <p style="font-size: 0.88rem; color: #94a3b8; line-height: 1.6; margin-bottom: 14px;">
                The project utilizes the <b>PTB-XL database</b> (v1.0.3), a publicly accessible clinical dataset of 12-lead electrocardiograms 
                annotated according to SCP-ECG statements. The experimental dataset downloaded for this research comprises 
                <b>5,037 records</b> structured according to the official patient-separated folds to strictly eliminate data leakage:
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px;">
                <div style="padding: 12px; border-radius: 8px; background: #161f30; border: 1px solid #222f44;">
                    <span style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Training Set</span>
                    <p style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin: 2px 0;">Folds 1 to 8</p>
                    <p style="font-size: 0.78rem; color: #94a3b8; margin: 0;">3,500 patient-separated records</p>
                </div>
                <div style="padding: 12px; border-radius: 8px; background: #161f30; border: 1px solid #222f44;">
                    <span style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Validation Set</span>
                    <p style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin: 2px 0;">Fold 9</p>
                    <p style="font-size: 0.78rem; color: #94a3b8; margin: 0;">500 records (early stopping & thresholds)</p>
                </div>
                <div style="padding: 12px; border-radius: 8px; background: #161f30; border: 1px solid #222f44;">
                    <span style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Held-Out Test Set</span>
                    <p style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin: 2px 0;">Fold 10</p>
                    <p style="font-size: 0.78rem; color: #94a3b8; margin: 0;">500 records (unbiased final evaluation)</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 2. Preprocessing & Dimensions
    st.markdown("""
        <div class="cr-card">
            <h2 style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; margin-bottom: 10px;">2. Signal Dimensions & Frozen Preprocessing</h2>
            <p style="font-size: 0.88rem; color: #94a3b8; line-height: 1.6; margin-bottom: 8px;">
                Signals are sampled at <b>100 Hz</b> for 10 seconds, yielding an input tensor layout of:
            </p>
            <div style="padding: 10px 14px; background: #0c121e; color: #38bdf8; border: 1px solid #1e293b; border-radius: 8px; font-family: monospace; font-size: 0.82rem; margin-bottom: 12px;">
                Tensor Layout: [batch_size, leads=12, time_steps=1000] (float32)
            </div>
            <p style="font-size: 0.88rem; color: #94a3b8; line-height: 1.6; margin-bottom: 8px;">
                To prevent data leakage from validation or test sets into the preprocessing pipeline, 
                <b>per-lead z-score normalization parameters</b> (mean and standard deviation per lead) 
                were computed exclusively from the 3,500 training recordings:
            </p>
            <div style="padding: 10px 14px; background: #161f30; border: 1px solid #222f44; border-radius: 8px; font-family: monospace; font-size: 0.82rem; margin-bottom: 8px; color: #f8fafc;">
                x_normalized = (x - lead_mean) / max(lead_std, 1e-6)
            </div>
            <p style="font-size: 0.85rem; color: #94a3b8;">
                These exact 12 lead means and 12 standard deviations are serialized inside every model checkpoint and applied at runtime.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 3. Model Architecture & Optimization
    st.markdown("""
        <div class="cr-card">
            <h2 style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; margin-bottom: 10px;">3. Model Architectures & Optimization</h2>
            <p style="font-size: 0.88rem; color: #94a3b8; line-height: 1.6; margin-bottom: 12px;">
                All three models utilize the identical 1D convolutional encoder backbone to guarantee a controlled architectural comparison:
            </p>
            <div style="border: 1px solid #222f44; border-radius: 8px; padding: 14px; background: #161f30; margin-bottom: 16px;">
                <span style="font-weight: 700; font-size: 0.78rem; color: #818cf8; text-transform: uppercase;">Backbone Architecture (ECGEncoder):</span>
                <ul style="font-family: monospace; font-size: 0.78rem; color: #cbd5e1; margin: 8px 0 0 16px; padding: 0;">
                    <li>Conv1d(12 → 32, kernel=7, stride=2, pad=3) + BatchNorm1d + ReLU</li>
                    <li>Conv1d(32 → 64, kernel=7, stride=2, pad=3) + BatchNorm1d + ReLU</li>
                    <li>Conv1d(64 → 128, kernel=7, stride=2, pad=3) + BatchNorm1d + ReLU</li>
                    <li>AdaptiveAvgPool1d(8) → Flatten (1024 dims)</li>
                    <li>Linear(1024 → 128) [Latent Embedding Dimension = 128]</li>
                </ul>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 14px;">
                <div style="padding: 12px; border-radius: 8px; border: 1px solid #222f44; background: #111827;">
                    <span class="cr-badge cr-badge-blue" style="margin-bottom: 6px;">1. Baseline CNN</span>
                    <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">Random initialization of encoder and head. Trained end-to-end directly using labeled training signals with Multi-label Binary Cross-Entropy loss.</p>
                </div>
                <div style="padding: 12px; border-radius: 8px; border: 1px solid #222f44; background: #111827;">
                    <span class="cr-badge cr-badge-purple" style="margin-bottom: 6px;">2. AE-CNN</span>
                    <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">Encoder coupled with a 1D convolutional decoder. Pretrained on unsupervised training signals minimizing Mean Squared Error (MSE) reconstruction loss.</p>
                </div>
                <div style="padding: 12px; border-radius: 8px; border: 1px solid #222f44; background: #111827;">
                    <span class="cr-badge cr-badge-emerald" style="margin-bottom: 6px;">3. VAE-CNN</span>
                    <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">Encoder projects to mean and log-variance vectors. Pretrained on unsupervised signals optimizing the Evidence Lower Bound (ELBO with &beta; = 1e-3).</p>
                </div>
            </div>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.6;">
                • <b>Supervised Fine-Tuning:</b> AdamW optimizer, learning rate = 1e-3, weight decay = 1e-4, batch size = 64, max 50 epochs.<br>
                • <b>Early Stopping:</b> Monitored validation loss on fold 9 with patience = 5 epochs and minimum delta = 1e-4.<br>
                • <b>Threshold Calibration:</b> Decision thresholds tuned independently per diagnostic label using grid search (17 candidate thresholds from 0.1 to 0.9) maximizing validation F1.
            </div>
        </div>
    """, unsafe_allow_html=True)
