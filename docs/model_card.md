# Model Card: Multi-Label 12-Lead ECG Classification Models

Following the framework proposed by Mitchell et al. (2019), *Model Cards for Model Reporting*.

---

## 1. Model Details

- **Model Name:** CardioRep ECG Classifiers (`best_model_10pct.pt`, `best_model_25pct.pt`, `best_model_50pct.pt`, `best_model_100pct.pt`)
- **Model Type:** 1D Convolutional Neural Network (with optional Autoencoder / Variational Autoencoder pretraining)
- **Framework:** PyTorch 2.x
- **Input Dimension:** 12 leads $\times$ 1,000 samples (10 seconds @ 100 Hz), float32 tensor
- **Output:** 5 independent sigmoid probability scores for multi-label diagnostic superclasses:
  - `NORM`: Normal ECG
  - `MI`: Myocardial Infarction
  - `STTC`: ST/T Change
  - `CD`: Conduction Disturbance
  - `HYP`: Hypertrophy
- **Version:** 1.0.0
- **License:** Open academic research prototype (derived from PhysioNet PTB-XL under Open Data Commons Attribution License)

---

## 2. Intended Use

### 2.1 Primary Intended Use
- Benchmarking unsupervised representation learning methods (AE and VAE) against baseline supervised CNNs under label-scarce regimes (10%, 25%, 50%, and 100% training data).
- Interactive research exploration of 12-lead signal representations and decision threshold dynamics.

### 2.2 Out-of-Scope & Prohibited Uses
- **Clinical Diagnosis:** This model is **NOT** a certified medical diagnostic device. It must **NOT** be used to diagnose patients, assess risk, or guide therapeutic interventions.
- **Emergency Triage:** Must not be used in emergency medical services, intensive care units, or automated alarm systems.
- **Pediatric Telemetry:** The PTB-XL dataset primarily comprises adult populations; neonatal and pediatric ECGs are out of distribution.

---

## 3. Factors, Subpopulations & Bias

- **Demographics:** The PTB-XL database was collected across European clinical centers. Patient age, biological sex, comorbidities, and recording equipment introduce domain shifts.
- **Class Imbalance:** Normal records (`NORM`) comprise ~52% of training data, whereas Hypertrophy (`HYP`) comprises only ~11%, resulting in disparate per-class sensitivity.
- **Artifact Sensitivity:** Baseline wander, muscle tremor (EMG), electrode misplacement, or non-standard lead ordering will compromise prediction accuracy.

---

## 4. Evaluation Metrics & Benchmark Performance

Evaluated on the held-out patient-stratified fold 10 (500 records):

| Model Checkpoint | Backbone Initialization | Labeled Fraction | Val Macro-F1 | Test Macro-F1 | Test Micro-F1 | Test AUROC | Test AUPRC |
|---|---|---|---|---|---|---|---|
| `best_model_10pct.pt` | VAE Pretrained | 10% (350 records) | 0.6142 | 0.5292 | 0.5871 | 0.7674 | 0.5102 |
| `best_model_25pct.pt` | Supervised Baseline | 25% (875 records) | 0.6332 | 0.5896 | 0.6505 | 0.8257 | 0.5794 |
| `best_model_50pct.pt` | Autoencoder Pretrained | 50% (1,750 records) | 0.6704 | 0.6261 | 0.6917 | 0.8430 | 0.6331 |
| `best_model_100pct.pt` | Autoencoder Pretrained | 100% (3,500 records) | 0.6985 | 0.6663 | 0.7219 | 0.8738 | 0.7315 |

---

## 5. Decision Thresholds

Because multi-label classifications are not mutually exclusive, predictions are determined by per-label decision thresholds optimized on validation fold 9 via grid search:

| Label | Description | `10pct` (VAE) | `25pct` (CNN) | `50pct` (AE) | `100pct` (AE) |
|---|---|---|---|---|---|
| `NORM` | Normal ECG | 0.25 | 0.55 | 0.45 | 0.45 |
| `MI` | Myocardial Infarction | 0.20 | 0.20 | 0.30 | 0.25 |
| `STTC` | ST/T Change | 0.40 | 0.30 | 0.25 | 0.45 |
| `CD` | Conduction Disturbance | 0.25 | 0.25 | 0.55 | 0.30 |
| `HYP` | Hypertrophy | 0.20 | 0.30 | 0.25 | 0.25 |

---

## 6. Known Limitations

1. **Subset Training:** Trained on 3,500 records from an incomplete local download of PTB-XL (full corpus: 21,837).
2. **Single-Run Variance:** Results reflect a single random seed; confidence intervals or multiple fold averages were not computed in the original training script.
3. **No Prospective Testing:** Has not undergone clinical validation against prospective patient cohorts or multiple ECG hardware vendors.

---

## 7. Ethical Considerations & Medical Safety

Electrocardiogram classification errors pose severe patient safety hazards:
- **False Negatives:** Overlooking acute myocardial infarction (`MI`) can delay life-saving revascularization.
- **False Positives:** Misidentifying normal variants as conduction disturbance or hypertrophy induces unwarranted anxiety and unnecessary invasive diagnostics.

Therefore, this tool is strictly provided for machine learning research and educational demonstration.
