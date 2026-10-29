# Research Methodology: Unsupervised Representation Learning for 12-Lead ECG Classification

**Author / Project:** CardioRep ECG Investigation  
**Status:** Academic Research Prototype  
**Dataset:** PTB-XL v1.0.3 Subset (PhysioNet)  
**Primary Question:** *Does unsupervised representation learning (Autoencoders and Variational Autoencoders) improve multi-label cardiac abnormality classification when labeled training data is scarce?*

---

## 1. Scientific Background & Motivation

Standard 12-lead electrocardiograms (ECGs) record the heart's spatial and temporal electrical activity across 12 standard projections:
- **Limb leads:** Lead I, Lead II, Lead III
- **Augmented limb leads:** aVR, aVL, aVF
- **Precordial leads:** V1, V2, V3, V4, V5, V6

In clinical practice, obtaining consensus diagnostic labels from certified cardiologists is labor-intensive and costly. However, modern hospital telemetry systems continuously record millions of raw, unlabeled ECG traces.

This study systematically explores **self-supervised / unsupervised representation learning** as a mechanism to pretrain 1D convolutional encoders on unlabeled waveforms, evaluating whether pretraining improves classification performance in data-constrained regimes where only **10% (350 records)**, **25% (875 records)**, **50% (1,750 records)**, or **100% (3,500 records)** of labels are available.

---

## 2. Dataset & Partitioning Methodology

### 2.1 The PTB-XL Dataset
The benchmark is based on the **PTB-XL database**, a large public collection of 10-second, 12-lead ECGs annotated according to SCP-ECG statements.

### 2.2 Dataset Limitation Disclosure
Due to an incomplete archive download in the experimental environment:
- Total local records available: **5,037 records** (out of 21,837 full dataset).
- Train split: **3,500 records** (sampled from official folds 1–8).
- Validation split: **500 records** (sampled from official fold 9).
- Held-out test split: **500 records** (sampled from official fold 10).

> **Important Disclosure:** The metrics presented herein are preliminary single-run results derived strictly from this 4,500-record subset. They must not be conflated with benchmarks obtained across the complete 21,837-record dataset.

### 2.3 Patient Separation
Records from the same patient must never appear in both training and evaluation splits to avoid optimistic performance bias from morphological memorization. By utilizing PTB-XL's official stratified folds:
- Folds 1–8: Training and unsupervised pretraining.
- Fold 9: Validation loss monitoring and decision threshold tuning.
- Fold 10: Held-out final test evaluation.

---

## 3. Signal Preprocessing & Normalization

### 3.1 Input Dimensions
All recordings are sampled at **100 Hz** for 10 seconds:
- Leads: 12 standard channels in PTB-XL order (`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`).
- Signal Length: 1,000 time steps.
- Tensor Dimension: `[batch_size, 12, 1000]` float32.

### 3.2 Lead Normalization
To prevent data snooping from evaluation sets into preprocessing, per-lead mean ($\mu_{\text{lead}}$) and standard deviation ($\sigma_{\text{lead}}$) were calculated **exclusively on the 3,500 training recordings**:

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
- Output: Linear(128 $\rightarrow$ 5) producing raw logits for the 5 diagnostic superclasses.

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

1. **Pretraining Phase:** Autoencoder and VAE encoders were pretrained on the 3,500 training records for 5 epochs without any diagnostic labels.
2. **Supervised Fine-Tuning:**
   - Fractions evaluated: 10% (350 records), 25% (875 records), 50% (1,750 records), and 100% (3,500 records).
   - The same deterministic index subset was used across all three architectures at each fraction.
   - Loss: Multi-label `BCEWithLogitsLoss`.
   - Optimizer: `AdamW` (learning rate $= 10^{-3}$, weight decay $= 10^{-4}$).
   - Early Stopping: Monitored validation loss on fold 9 with patience $= 5$ epochs and minimum delta $= 10^{-4}$.
3. **Threshold Calibration:**
   - Grid search across 17 candidate thresholds $\in [0.10, 0.90]$ per class to maximize validation F1 on fold 9.
4. **Final Held-Out Reporting:**
   - The frozen model and calibrated thresholds were evaluated once on the 500 records of fold 10.

---

## 6. Experimental Results Matrix

| Fraction | Records | Model | Val Macro-F1 | Test Macro-F1 | Micro-F1 | AUROC | AUPRC | F1_NORM | F1_MI | F1_STTC | F1_CD | F1_HYP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10% | 350 | **VAE** | **0.6142** | **0.5292** | 0.5871 | 0.7674 | 0.5102 | 0.7903 | 0.4188 | 0.6111 | 0.4393 | 0.3867 |
| 10% | 350 | CNN | 0.5806 | 0.5099 | 0.5886 | 0.7721 | 0.4763 | 0.8060 | 0.4337 | 0.5802 | 0.3372 | 0.3924 |
| 10% | 350 | AE | 0.5615 | 0.4871 | 0.5491 | 0.7389 | 0.4495 | 0.7740 | 0.4127 | 0.5276 | 0.3359 | 0.3855 |
| 25% | 875 | **CNN** | **0.6332** | **0.5896** | 0.6505 | 0.8257 | 0.5794 | 0.8273 | 0.4876 | 0.6445 | 0.5576 | 0.4311 |
| 25% | 875 | VAE | 0.6018 | 0.5444 | 0.6169 | 0.7876 | 0.5438 | 0.8202 | 0.4627 | 0.6000 | 0.4505 | 0.3886 |
| 25% | 875 | AE | 0.6002 | 0.5349 | 0.5975 | 0.7831 | 0.5294 | 0.7954 | 0.4533 | 0.5772 | 0.4577 | 0.3908 |
| 50% | 1,750 | **AE** | **0.6704** | **0.6261** | 0.6917 | 0.8430 | 0.6331 | 0.8361 | 0.5192 | 0.7046 | 0.5521 | 0.5185 |
| 50% | 1,750 | CNN | 0.6650 | 0.6236 | 0.6831 | 0.8494 | 0.6498 | 0.8516 | 0.5285 | 0.6935 | 0.6012 | 0.4433 |
| 50% | 1,750 | VAE | 0.6649 | 0.5805 | 0.6440 | 0.8241 | 0.6177 | 0.8096 | 0.5097 | 0.6667 | 0.4800 | 0.4368 |
| 100% | 3,500 | **AE** | **0.6985** | **0.6663** | 0.7219 | 0.8738 | 0.7315 | 0.8508 | 0.5550 | 0.7345 | 0.5978 | 0.5932 |
| 100% | 3,500 | CNN | 0.6902 | 0.6591 | 0.7108 | 0.8743 | 0.7196 | 0.8379 | 0.5581 | 0.7339 | 0.6036 | 0.5620 |
| 100% | 3,500 | VAE | 0.6813 | 0.6582 | 0.7154 | 0.8569 | 0.6881 | 0.8427 | 0.5579 | 0.7398 | 0.5987 | 0.5517 |

---

## 7. Key Findings & Discussion

1. **Pretraining Does Not Universally Dominate:** At 25% labeled data, the randomly initialized CNN baseline clearly surpassed both pretrained variants (0.590 vs. 0.544). Pretraining gave an advantage in the extreme scarce regime (VAE at 10%: 0.529 vs. 0.510) and slight edge at 50% and 100% (AE: 0.626 and 0.666).
2. **Differences Across Disease Classes:** Normal rhythm (NORM) is readily classified with high confidence ($F_1 \approx 0.80 - 0.85$), whereas complex conduction blocks (CD) and ventricular hypertrophy (HYP) exhibit lower F1 ($0.33 - 0.60$) due to smaller sample representation and morphological variability.
3. **Reconstruction vs. Discrimination:** Reconstructing raw signals penalizes baseline wander and electrical noise as heavily as diagnostic ST/T deviations, explaining why naive autoencoding does not consistently produce superior discriminative features.

---

## 8. Clinical & Research Disclaimer

This application and documentation represent an academic research investigation into machine learning representations. It is **not** an FDA-approved or clinically validated diagnostic system. No model outputs should be used to direct clinical therapy.
