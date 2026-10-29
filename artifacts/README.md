# Model Artifacts Directory Guide

This directory contains the trained PyTorch model checkpoints extracted from the experiment bundle (`ptbxl_latest_models_50ep.zip`).

---

## 1. Checkpoint Contents & Identification

The original research experiment evaluated 12 models across 4 training fractions (10%, 25%, 50%, 100%) and 3 model representations (Baseline CNN, Autoencoder-pretrained CNN, and Variational-Autoencoder-pretrained CNN).

At the conclusion of notebook training (Cell 32), the **top-performing model per labeled fraction** (selected according to highest validation Macro-F1 on official patient fold 9) was exported into the archive.

| Checkpoint File | Representation | Labeled Fraction | Records Trained | Validation Macro-F1 | Test Macro-F1 |
|---|---|---|---|---|---|
| `best_model_10pct.pt` | **VAE-CNN** | 10% (0.10) | 350 | 0.6142 | 0.5292 |
| `best_model_25pct.pt` | **Baseline 1D CNN** | 25% (0.25) | 875 | 0.6332 | 0.5896 |
| `best_model_50pct.pt` | **AE-CNN** | 50% (0.50) | 1,750 | 0.6704 | 0.6261 |
| `best_model_100pct.pt` | **AE-CNN** | 100% (1.00) | 3,500 | 0.6985 | 0.6663 |

---

## 2. Checkpoint Data Structure

Each `.pt` file is a serialized dictionary containing the weights, metadata, preprocessing normalization statistics, and decision thresholds:

```python
{
    "state_dict": OrderedDict(...),       # PyTorch state dict for ECGClassifier
    "model_type": str,                   # "CNN", "AE", or "VAE"
    "labeled_fraction": float,           # 0.1, 0.25, 0.5, or 1.0
    "thresholds": np.ndarray,            # [5] float decision thresholds for [NORM, MI, STTC, CD, HYP]
    "targets": list[str],                # ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    "lead_mean": np.ndarray,             # Shape [1, 12, 1] - mean per lead computed on training fold
    "lead_std": np.ndarray               # Shape [1, 12, 1] - standard deviation per lead computed on training fold
}
```

### Frozen Lead Statistics
All models share the exact same lead normalization parameters computed on the 3,500 training recordings:
- **Lead Means (mV):**
  `[-0.000527, -0.000554, -0.000027, 0.000526, -0.000202, -0.000272, 0.001558, -0.001067, -0.001673, -0.001296, -0.001048, -0.001137]`
- **Lead Standard Deviations (mV):**
  `[0.1558, 0.1647, 0.1595, 0.1391, 0.1346, 0.1422, 0.2503, 0.3273, 0.3312, 0.2992, 0.2786, 0.2614]`

### Trained Decision Thresholds
- `best_model_10pct.pt` (VAE): `NORM=0.25, MI=0.20, STTC=0.40, CD=0.25, HYP=0.20`
- `best_model_25pct.pt` (CNN): `NORM=0.55, MI=0.20, STTC=0.30, CD=0.25, HYP=0.30`
- `best_model_50pct.pt` (AE):  `NORM=0.45, MI=0.30, STTC=0.25, CD=0.55, HYP=0.25`
- `best_model_100pct.pt` (AE): `NORM=0.45, MI=0.25, STTC=0.45, CD=0.30, HYP=0.25`

---

## 3. How to Add New Model Checkpoints

The backend dynamically scans `artifacts/models/` for any `.pt` file. To integrate an additional trained model:

1. Save your PyTorch checkpoint as a dictionary with the required schema:
   ```python
   torch.save({
       "state_dict": model.state_dict(),
       "model_type": "CNN",            # or "AE", "VAE", "Transformer", etc.
       "labeled_fraction": 1.0,
       "thresholds": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
       "targets": ["NORM", "MI", "STTC", "CD", "HYP"],
       "lead_mean": lead_mean,         # shape (1, 12, 1)
       "lead_std": lead_std            # shape (1, 12, 1)
   }, "artifacts/models/my_custom_model.pt")
   ```
2. Place the file inside `artifacts/models/`.
3. The backend will automatically discover it on the next `/api/models` call and make it available in the inference UI.
