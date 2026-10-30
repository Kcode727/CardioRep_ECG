# Model Artifacts Directory Guide

This directory contains the trained PyTorch model checkpoints extracted from the experiment bundle (`ptbxl_all_12_models.zip` and `Model training.ipynb`).

---

## 1. Checkpoint Contents & Identification

The complete research suite contains **all 12 models** evaluated across 4 training fractions (10%, 25%, 50%, 100%) and 3 model representations (Baseline CNN, Autoencoder-pretrained CNN, and Variational-Autoencoder-pretrained CNN), plus the 4 best-per-fraction aliases.

### Complete 12-Model Matrix

| Checkpoint File | Representation | Labeled Fraction | Records Trained | Val Macro-F1 | Test Macro-F1 | Test AUROC |
|---|---|---|---|---|---|---|
| `model_VAE_10pct.pt` / `best_model_10pct.pt` | **VAE-CNN** | 10% (0.10) | 1,744 | 0.6665 | **0.6465** | 0.8628 |
| `model_CNN_10pct.pt` | **Baseline CNN** | 10% (0.10) | 1,744 | 0.6626 | 0.6374 | 0.8618 |
| `model_AE_10pct.pt` | **AE-CNN** | 10% (0.10) | 1,744 | 0.6578 | 0.6241 | 0.8519 |
| `model_CNN_25pct.pt` / `best_model_25pct.pt` | **Baseline CNN** | 25% (0.25) | 4,360 | 0.7013 | **0.6833** | 0.8870 |
| `model_VAE_25pct.pt` | **VAE-CNN** | 25% (0.25) | 4,360 | 0.6814 | 0.6678 | 0.8760 |
| `model_AE_25pct.pt` | **AE-CNN** | 25% (0.25) | 4,360 | 0.6833 | 0.6613 | 0.8760 |
| `model_CNN_50pct.pt` / `best_model_50pct.pt` | **Baseline CNN** | 50% (0.50) | 8,720 | 0.7153 | **0.7050** | 0.9025 |
| `model_VAE_50pct.pt` | **VAE-CNN** | 50% (0.50) | 8,720 | 0.7129 | 0.7028 | 0.8990 |
| `model_AE_50pct.pt` | **AE-CNN** | 50% (0.50) | 8,720 | 0.7045 | 0.6880 | 0.8927 |
| `model_CNN_100pct.pt` / `best_model_100pct.pt` | **Baseline CNN** | 100% (1.00) | 17,441 | 0.7333 | **0.7272** | 0.9145 |
| `model_VAE_100pct.pt` | **VAE-CNN** | 100% (1.00) | 17,441 | 0.7296 | 0.7229 | 0.9140 |
| `model_AE_100pct.pt` | **AE-CNN** | 100% (1.00) | 17,441 | 0.7170 | 0.7113 | 0.9080 |

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
