# CardioRep ECG: Cardiac Abnormality Classification Using Deep Learning

A full-stack, research-oriented machine learning web application investigating whether **unsupervised representation learning** (Autoencoders and Variational Autoencoders) improves multi-label **12-lead electrocardiogram (ECG)** classification when labeled clinical training data is severely limited.

> **Disclaimer:** This software is an **academic research prototype**, not a medical diagnostic product or clinical decision support system. It has not been approved or cleared by the FDA, CE, or any regulatory body.

---

## 1. Research Overview

The project investigates multi-label classification across five diagnostic superclasses defined by the **PTB-XL** database:
- **NORM:** Normal ECG
- **MI:** Myocardial Infarction
- **STTC:** ST/T Changes
- **CD:** Conduction Disturbance
- **HYP:** Hypertrophy

### Model Architectures Compared
1. **Baseline 1D CNN:** Supervised classifier trained from scratch on labeled ECG signals with random weight initialization.
2. **Autoencoder-Pretrained CNN (AE-CNN):** Encoder pretrained on unlabeled ECG reconstruction (Mean Squared Error) before supervised fine-tuning.
3. **Variational Autoencoder-Pretrained CNN (VAE-CNN):** Encoder pretrained on probabilistic latent representations optimizing the Evidence Lower Bound (ELBO) before supervised fine-tuning.

### Training Data Fractions Evaluated
All models were benchmarked across four labeled training data fractions sampled from the training partition:
- **10%** (350 labeled records)
- **25%** (875 labeled records)
- **50%** (1,750 labeled records)
- **100%** (3,500 labeled records)

### Empirical Research Findings
| Labeled Training Fraction | Baseline CNN | AE-CNN | VAE-CNN | Winning Approach |
|:---:|:---:|:---:|:---:|:---:|
| **10%** | 0.510 | 0.487 | **0.529** | **VAE-CNN** |
| **25%** | **0.590** | 0.535 | 0.544 | **Baseline CNN** |
| **50%** | 0.624 | **0.626** | 0.581 | **AE-CNN** |
| **100%** | 0.659 | **0.666** | 0.658 | **AE-CNN** |

> **Key Research Insight:** Pretraining does **not** uniformly outperform the baseline supervised CNN across all fractions. VAE pretraining demonstrates sample efficiency primarily in extreme data-scarce regimes (10%), while supervised training from scratch recovers strongly at 25%, and AE pretraining provides modest gains at 50% and 100%.

---

## 2. Dataset & Limitations Disclosure

- **Source Dataset:** PTB-XL database (PhysioNet v1.0.3), 100 Hz recordings.
- **Incomplete Download Limitation:** The experiments were conducted on a locally available subset of **5,037 records** (rather than the complete 21,837-record archive).
- **Patient-Separated Folds:** Folds 1–8 (3,500 records) were used for training and unsupervised pretraining; Fold 9 (500 records) for validation early stopping and threshold tuning; Fold 10 (500 records) for final held-out test reporting.
- **Single-Run Results:** The reported metrics are preliminary single-run evaluation results from the available subset and should not be claimed as complete-dataset benchmarks.

---

## 3. Project Structure

```
.
├── app.py                              # Unified Streamlit application (all-in-one UI & PyTorch inference)
├── requirements.txt                    # Python application dependencies
├── Dockerfile                          # Containerized deployment specification
├── .dockerignore                       # Docker build exclusions
├── .gitignore                          # Git tracking exclusions
├── README.md                           # Master project documentation
├── .streamlit/
│   └── config.toml                     # Streamlit native Dark Theme configuration
├── artifacts/
│   ├── README.md                       # Model artifact placement guide
│   └── models/                         # PyTorch model checkpoints (.pt)
│       ├── best_model_10pct.pt         # VAE-CNN (10% labeled data)
│       ├── best_model_25pct.pt         # Baseline CNN (25% labeled data)
│       ├── best_model_50pct.pt         # AE-CNN (50% labeled data)
│       └── best_model_100pct.pt        # AE-CNN (100% labeled data)
├── results/
│   └── experiment_results.csv          # Exact 12-run experimental metrics matrix
├── docs/
│   ├── methodology.md                  # Comprehensive scientific methodology
│   └── model_card.md                   # Standardized AI model card
└── sample_data/
    ├── generate_samples.py             # Synthetic 12-lead ECG generator
    ├── sample_normal_ecg.csv           # Normal sinus rhythm sample (CSV)
    ├── sample_mi_ecg.csv               # Acute MI pattern sample (CSV)
    ├── sample_cd_ecg.csv               # Conduction disturbance sample (CSV)
    └── sample_normal_ecg.npy           # Normal sinus rhythm sample (NumPy)
```

---

## 4. Setup and Execution Instructions

Run the application locally with a single command (configured with a clinical Dark Theme UI):

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

3. The application will immediately open in your browser at `http://localhost:8501`.

---

## 5. Deployment Guide for Streamlit

### Option 1: Streamlit Community Cloud (Free & Fastest)

1. Push this repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Deploy CardioRep ECG application"
   git remote add origin https://github.com/<your-username>/cardiorep-ecg.git
   git push -u origin main
   ```
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **"New app"** and select:
   - **Repository:** `<your-username>/cardiorep-ecg`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy!"** Streamlit will automatically install dependencies from `requirements.txt` and launch your live public URL.

---

### Option 2: Docker Container Deployment (AWS, GCP, Azure, DigitalOcean)

A production-ready `Dockerfile` and `.dockerignore` are included in the repository.

1. Build the Docker image:
   ```bash
   docker build -t cardiorep-ecg .
   ```

2. Run the container locally or in production:
   ```bash
   docker run -d -p 8501:8501 --name cardiorep-ecg cardiorep-ecg
   ```

3. Access the containerized application at `http://localhost:8501`.

---

### Option 3: Hugging Face Spaces (Free Cloud Hosting)

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Select **Streamlit** as the Space SDK.
3. Push the repository files (`app.py`, `requirements.txt`, `.streamlit/`, `artifacts/`, `results/`, `sample_data/`).
4. Hugging Face will automatically build and host the application with public URL and embeddable iframe.

---

### Option 4: Linux Cloud Virtual Machine (Ubuntu EC2 / Droplet + Systemd + Nginx)

1. Connect to your Linux server and clone the code:
   ```bash
   sudo apt update && sudo apt install -y python3-pip python3-venv git
   git clone https://github.com/<your-username>/cardiorep-ecg.git /opt/cardiorep-ecg
   cd /opt/cardiorep-ecg
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create a systemd service (`/etc/systemd/system/streamlit.service`):
   ```ini
   [Unit]
   Description=CardioRep ECG Streamlit Service
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/opt/cardiorep-ecg
   ExecStart=/opt/cardiorep-ecg/venv/bin/streamlit run app.py --server.port=8501 --server.address=127.0.0.1
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

3. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now streamlit
   ```

4. Configure Nginx reverse proxy on port 80/443 with WebSocket support:
   ```nginx
   server {
       listen 80;
       server_name your-ecg-domain.com;

       location / {
           proxy_pass http://127.0.0.1:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_cache_bypass $http_upgrade;
       }
   }
   ```

---

## 6. Accepted Input Signal Specifications

The inference engine strictly requires signals compatible with the PTB-XL training pipeline:
- **Duration:** 10.0 seconds
- **Sampling Frequency:** 100 Hz (1,000 samples per lead)
- **Leads:** 12 leads (`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`)
- **Accepted Formats:**
  1. `.csv`: Table with 1,000 rows and 12 lead columns (with lead name headers).
  2. `.npy`: 2D NumPy array of shape `(12, 1000)` or `(1000, 12)` float32.
- **Validation:** Files with missing columns, improper shape, NaN values, infinite values, or excessive amplitudes are rejected with clear 400 Bad Request explanations.

---

## 7. Artifact Inventory & Missing Files Report

### Discovered & Validated Artifacts
- `artifacts/models/best_model_10pct.pt` (VAE-CNN checkpoint, 10% labeled fraction)
- `artifacts/models/best_model_25pct.pt` (Baseline CNN checkpoint, 25% labeled fraction)
- `artifacts/models/best_model_50pct.pt` (AE-CNN checkpoint, 50% labeled fraction)
- `artifacts/models/best_model_100pct.pt` (AE-CNN checkpoint, 100% labeled fraction)
- `results/experiment_results.csv` (Full 12-run metrics matrix)

### Missing Files / Artifacts Identified
1. **Losing Checkpoints at Each Fraction:** In the training notebook (Cell 32), only the **single best model** per fraction was saved to disk based on validation F1. The non-winning checkpoints (e.g. CNN-10%, AE-10%, VAE-25%, CNN-50%, VAE-50%, CNN-100%, VAE-100%) were not exported to the ZIP archive. Their metrics are fully preserved in `results/experiment_results.csv`, but their binary `.pt` weights are missing from the supplied archive.
2. **Complete PTB-XL Dataset:** As documented, the initial download was interrupted, yielding 5,037 local records instead of the complete 21,837-record archive.
