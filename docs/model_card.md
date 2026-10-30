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

Evaluated on the held-out patient-stratified fold 10 (2,203 records):

| Model Checkpoint | Backbone Initialization | Labeled Fraction | Val Macro-F1 | Test Macro-F1 | Test Micro-F1 | Test AUROC | Test AUPRC |
|---|---|---|---|---|---|---|---|
| `best_model_10pct.pt` | VAE Pretrained | 10% (1,744 records) | 0.6665 | 0.6465 | 0.6847 | 0.8628 | 0.6905 |
| `best_model_25pct.pt` | Supervised Baseline | 25% (4,360 records) | 0.7013 | 0.6833 | 0.7202 | 0.8870 | 0.7401 |
| `best_model_50pct.pt` | Supervised Baseline | 50% (8,720 records) | 0.7153 | 0.7050 | 0.7386 | 0.9025 | 0.7685 |
| `best_model_100pct.pt` | Supervised Baseline | 100% (17,441 records) | 0.7333 | 0.7272 | 0.7589 | 0.9145 | 0.7952 |

---

## 5. Decision Thresholds

Because multi-label classifications are not mutually exclusive, predictions are determined by per-label decision thresholds optimized on validation fold 9 (2,193 records) via grid search:

| Label | Description | `10pct` (VAE) | `25pct` (CNN) | `50pct` (CNN) | `100pct` (CNN) |
|---|---|---|---|---|---|
| `NORM` | Normal ECG | 0.50 | 0.35 | 0.35 | 0.25 |
| `MI` | Myocardial Infarction | 0.30 | 0.30 | 0.45 | 0.50 |
| `STTC` | ST/T Change | 0.25 | 0.35 | 0.40 | 0.40 |
| `CD` | Conduction Disturbance | 0.30 | 0.45 | 0.25 | 0.35 |
| `HYP` | Hypertrophy | 0.25 | 0.20 | 0.35 | 0.20 |

---

## 6. Known Limitations

1. **Full Dataset Benchmark:** Trained on 17,441 records from PTB-XL (total 21,837 records across 10 folds).
2. **Single-Run Variance:** Results reflect a single random seed; confidence intervals or multiple fold averages were not computed in the original training script.
3. **No Prospective Testing:** Has not undergone clinical validation against prospective patient cohorts or multiple ECG hardware vendors.

---

## 7. Ethical Considerations & Medical Safety

Electrocardiogram classification errors pose severe patient safety hazards:
- **False Negatives:** Overlooking acute myocardial infarction (`MI`) can delay life-saving revascularization.
- **False Positives:** Misidentifying normal variants as conduction disturbance or hypertrophy induces unwarranted anxiety and unnecessary invasive diagnostics.

Therefore, this tool is strictly provided for machine learning research and educational demonstration.
