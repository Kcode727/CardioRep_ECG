# Research Methodology: Unsupervised Representation Learning for 12-Lead ECG Classification

**Author / Project:** CardioRep ECG Investigation  
**Status:** Academic Research Prototype  
**Dataset:** PTB-XL v1.0.3 (PhysioNet) — Complete 21,837 Clinical Database Benchmark  
**Primary Question:** *Does unsupervised representation learning (Autoencoders and Variational Autoencoders) improve multi-label cardiac abnormality classification when labeled training data is scarce?*

---

## 1. Scientific Background & Motivation

Standard 12-lead electrocardiograms (ECGs) record the heart's spatial and temporal electrical activity across 12 standard projections:
- **Limb leads:** Lead I, Lead II, Lead III
- **Augmented limb leads:** aVR, aVL, aVF
- **Precordial leads:** V1, V2, V3, V4, V5, V6

In clinical practice, obtaining consensus diagnostic labels from certified cardiologists is labor-intensive and costly. However, modern hospital telemetry systems continuously record millions of raw, unlabeled ECG traces.

This study systematically explores **self-supervised / unsupervised representation learning** as a mechanism to pretrain 1D convolutional encoders on unlabeled waveforms, evaluating whether pretraining improves classification performance in data-constrained regimes where only **10% (1,744 records)**, **25% (4,360 records)**, **50% (8,720 records)**, or **100% (17,441 records)** of labels are available.

---

## 2. Dataset & Partitioning Methodology

### 2.1 The PTB-XL Dataset
The benchmark is based on the complete **PTB-XL database** (v1.0.3), a comprehensive public collection of 10-second, 12-lead ECGs annotated according to SCP-ECG statements.

### 2.2 Dataset Partitions & Patient Separation
To strictly prevent data leakage from morphological memorization, the full dataset of **21,837 clinical records** across 18,885 individual patients is partitioned according to the official stratified 10-fold patient assignment:
- **Training Set (Folds 1–8):** **17,441 patient-separated records** used for unsupervised representation learning (AE and VAE pretraining) and supervised training.
- **Validation Set (Fold 9):** **2,193 records** used for early stopping, validation loss monitoring, and per-class decision threshold calibration ($\tau$).
- **Held-Out Test Set (Fold 10):** **2,203 records** reserved strictly for final, unbiased benchmark evaluation.

---

## 3. Signal Preprocessing & Normalization

### 3.1 Input Dimensions
All recordings are sampled at **100 Hz** for 10 seconds:
- Leads: 12 standard channels in PTB-XL order (`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`).
- Signal Length: 1,000 time steps.
- Tensor Dimension: `[batch_size, 12, 1000]` float32.

### 3.2 Lead Normalization
To prevent data snooping from evaluation sets into preprocessing, per-lead mean ($\mu_{\text{lead}}$) and standard deviation ($\sigma_{\text{lead}}$) were calculated **exclusively on the 17,441 training recordings (Folds 1–8)**:

$$\tilde{x}_{\text{lead}, t} = \frac{x_{\text{lead}, t} - \mu_{\text{lead}}}{\max(\sigma_{\text{lead}}, 10^{-6})}$$

These frozen values were serialized into model checkpoints and are applied to every inference signal.

---

## 4. Model Architectures

All three classifiers share the exact same convolutional feature extractor (`ECGEncoder`) to ensure an unbiased architectural comparison.

### 4.1 Shared Encoder Backbone (`ECGEncoder`)
- **Conv1D Layer 1:** 12 input channels $\rightarrow$ 32 filters, kernel size 7, stride 2, padding 3 + BatchNorm1d + ReLU
- **Conv1D Layer 2:** 32 channels $\rightarrow$ 64 filters, kernel size 7, stride 2, padding 3 + BatchNorm1d + ReLU
- **Conv1D Layer 3:** 64 channels $\rightarrow$ 128 filters, kernel size 7, stride 2, padding 3 + BatchNorm1d + ReLU
- **Adaptive Average Pooling:** `AdaptiveAvgPool1d(8)` $\rightarrow$ flattened to $128 \times 8 = 1024$ features
- **Fully Connected Projection:** Linear layer projecting 1024 $\rightarrow$ 128 latent dimensions

### 4.2 Supervised Classifier Head (`ECGClassifier`)
- Input: 128-dimensional latent vector from `ECGEncoder`
- Fully Connected: Linear(128 $\rightarrow$ 128) + ReLU + Dropout(p = 0.25)
- Output: Linear(128 $\rightarrow$ 5) producing raw logits for the 5 diagnostic superclasses (`NORM`, `MI`, `STTC`, `CD`, `HYP`).

### 4.3 Autoencoder Architecture (`ECGAutoencoder`)
- **Encoder:** `ECGEncoder` projecting 12-lead signal to 128-dim bottleneck.
- **Decoder (`ECGDecoder`):** Linear(128 $\rightarrow$ 1024), reshaped to `[128, 8]`, followed by linear interpolation upsampling (to sizes 125, 250, 500, 1000) and 1D convolutions (kernel size 5) producing reconstructed signal $\hat{x}$ of shape `[12, 1000]`.
- **Pretraining Objective:** Mean Squared Error (MSE) reconstruction loss:
  $$\mathcal{L}_{\text{AE}} = \frac{1}{B} \sum_{i=1}^B \| x_i - \hat{x}_i \|^2$$

### 4.4 Variational Autoencoder Architecture (`ECGVAE`)
- **Encoder:** `ECGEncoder` backbone followed by two parallel linear layers:
  - Mean projection: $\mu = \text{Linear}(128 \rightarrow 128)$
  - Log-variance projection: $\log \sigma^2 = \text{clamp}(\text{Linear}(128 \rightarrow 128), -8, 8)$
- **Reparameterization Trick:** $z = \mu + \epsilon \odot \exp(0.5 \cdot \log \sigma^2)$, where $\epsilon \sim \mathcal{N}(0, I)$.
- **Decoder:** Identical `ECGDecoder` reconstructing signal from sampled latent vector $z$.
- **Pretraining Objective:** Evidence Lower Bound (ELBO):
  $$\mathcal{L}_{\text{VAE}} = \text{MSE}(x, \hat{x}) + \beta \cdot D_{\text{KL}}\left(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, I)\right)$$
  where $\beta = 10^{-3}$.

---

## 5. Supervised Fine-Tuning & Evaluation Protocol

1. **Pretraining Phase:** Autoencoder and VAE encoders were pretrained on the 17,441 training records for 5 epochs without any diagnostic labels.
2. **Supervised Fine-Tuning:**
   - Fractions evaluated: 10% (1,744 records), 25% (4,360 records), 50% (8,720 records), and 100% (17,441 records).
   - The same deterministic index subset was used across all three architectures at each fraction.
   - Loss: Multi-label `BCEWithLogitsLoss`.
   - Optimizer: `AdamW` (learning rate $= 10^{-3}$, weight decay $= 10^{-4}$).
   - Early Stopping: Monitored validation loss on fold 9 with patience $= 5$ epochs and minimum delta $= 10^{-4}$.
3. **Threshold Calibration:**
   - Grid search across 17 candidate thresholds $\in [0.10, 0.90]$ per class to maximize validation F1 on fold 9.
4. **Final Held-Out Reporting:**
   - The frozen model and calibrated thresholds were evaluated once on the 2,203 records of fold 10.

---

## 6. Complete 12-Run Experimental Results Matrix

| Model | Fraction | Records | Val Macro-F1 | Test Macro-F1 | Micro-F1 | AUROC | AUPRC | F1_NORM | F1_MI | F1_STTC | F1_CD | F1_HYP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **VAE** | 10% | 1,744 | **0.6665** | **0.6465** | 0.6847 | 0.8628 | 0.6905 | 0.8179 | 0.5900 | 0.6773 | 0.6148 | 0.5324 |
| CNN | 10% | 1,744 | 0.6626 | 0.6374 | 0.6729 | 0.8618 | 0.6889 | 0.8127 | 0.5811 | 0.6858 | 0.6107 | 0.4968 |
| AE | 10% | 1,744 | 0.6578 | 0.6241 | 0.6601 | 0.8519 | 0.6706 | 0.8073 | 0.5762 | 0.6511 | 0.5939 | 0.4921 |
| **CNN** | 25% | 4,360 | **0.7013** | **0.6833** | 0.7201 | 0.8870 | 0.7401 | 0.8302 | 0.6594 | 0.7049 | 0.6803 | 0.5416 |
| VAE | 25% | 4,360 | 0.6814 | 0.6678 | 0.7056 | 0.8760 | 0.7132 | 0.8323 | 0.6193 | 0.7088 | 0.6208 | 0.5581 |
| AE | 25% | 4,360 | 0.6833 | 0.6613 | 0.7027 | 0.8760 | 0.7182 | 0.8270 | 0.6330 | 0.7026 | 0.6151 | 0.5289 |
| **CNN** | 50% | 8,720 | **0.7153** | **0.7050** | 0.7386 | 0.9025 | 0.7685 | 0.8370 | 0.6825 | 0.7285 | 0.6934 | 0.5835 |
| VAE | 50% | 8,720 | 0.7129 | 0.7028 | 0.7371 | 0.8990 | 0.7621 | 0.8449 | 0.6662 | 0.7443 | 0.6824 | 0.5762 |
| AE | 50% | 8,720 | 0.7045 | 0.6880 | 0.7211 | 0.8927 | 0.7535 | 0.8295 | 0.6573 | 0.7338 | 0.6553 | 0.5643 |
| **CNN** | 100% | 17,441 | **0.7333** | **0.7272** | 0.7589 | 0.9145 | 0.7952 | 0.8492 | 0.7122 | 0.7500 | 0.7324 | 0.5921 |
| VAE | 100% | 17,441 | 0.7296 | 0.7229 | 0.7570 | 0.9140 | 0.7824 | 0.8550 | 0.7102 | 0.7567 | 0.7174 | 0.5753 |
| AE | 100% | 17,441 | 0.7170 | 0.7113 | 0.7491 | 0.9080 | 0.7732 | 0.8523 | 0.6779 | 0.7502 | 0.7027 | 0.5731 |

---

## 7. Key Findings & Discussion

1. **Pretraining Advantage in Low-Data Regime (10% Labeled — 1,744 records):** Under extreme label scarcity, the Variational Autoencoder pretrained CNN (`VAE-CNN`) achieves the highest performance across all evaluation dimensions (**0.6465 Test Macro-F1**, **0.6847 Micro-F1**, **0.8628 AUROC**, and **0.6905 AUPRC**), outperforming both the baseline CNN (0.6374) and AE-CNN (0.6241). This confirms that unsupervised probabilistic representation learning provides genuine sample efficiency when annotations are scarce.
2. **Supervised Scaling with Ample Labels (25%, 50%, 100%):** As labeled training volume expands (4,360 to 17,441 records), the baseline 1D CNN trained end-to-end from scratch catches up and scales best (0.6833 at 25%, 0.7050 at 50%, and 0.7272 at 100%), matching or slightly outperforming the pretrained architectures. When supervised gradient signals are plentiful, task-specific feature learning thrives without representation priors.
3. **Condition Specificity & Class Imbalance Dynamics:** Normal sinus rhythms (`NORM`) are reliably detected with high F1 (~0.85) due to high prevalence (7,607 training cases). Acute ischemic pathology (`MI`) reaches 0.712 F1, `STTC` reaches 0.757 F1, and `CD` reaches 0.732 F1. Conversely, structural wall thickening (`HYP`) remains the most difficult class (0.53–0.59 F1) due to fewer training cases (2,121) and subtle, diffuse voltage alterations.

---

## 8. Clinical & Research Disclaimer

This application and documentation represent an academic research investigation into machine learning representations. It is **not** an FDA-approved or clinically validated diagnostic system. No model outputs should be used to direct clinical therapy.
