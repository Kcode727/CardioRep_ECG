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
- **10%** (1,744 labeled records)
- **25%** (4,360 labeled records)
- **50%** (8,720 labeled records)
- **100%** (17,441 labeled records)

### Empirical Research Findings
| Labeled Training Fraction | Baseline CNN | AE-CNN | VAE-CNN | Winning Approach |
|:---:|:---:|:---:|:---:|:---:|
| **10%** | 0.637 | 0.624 | **0.646** | **VAE-CNN** |
| **25%** | **0.683** | 0.661 | 0.668 | **Baseline CNN** |
| **50%** | **0.705** | 0.688 | 0.703 | **Baseline CNN** |
| **100%** | **0.727** | 0.711 | 0.723 | **Baseline CNN** |

> **Key Research Insight:** Pretraining provides its greatest sample efficiency in extreme data scarcity (**10% labels: VAE leads with 0.646 vs. 0.637 baseline CNN and 0.624 AE-CNN**), validating that probabilistic representation learning acts as an effective inductive prior. As labeled training volume expands (25%, 50%, 100%), the supervised baseline 1D CNN scales strongly (reaching 0.727 Macro-F1), matching or slightly exceeding generic reconstruction pretraining.

---

## 2. Dataset & Experimental Partitioning

- **Source Dataset:** PTB-XL database (PhysioNet v1.0.3), 100 Hz recordings (21,837 total clinical records).
- **Patient-Separated Folds:** 
  - Folds 1–8 (17,441 records) for training and unsupervised representation pretraining.
  - Fold 9 (2,193 records) for validation early stopping and per-class decision threshold tuning.
  - Fold 10 (2,203 records) strictly held-out for unbiased final test reporting.

---

## 3. Project Structure

```
.
├── app.py                              # Unified Streamlit application (Dark UI, all 12 models + WFDB inference)
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
│       ├── best_model_10pct.pt         # Top 10%: VAE-CNN (F1: 0.6465)
│       ├── best_model_25pct.pt         # Top 25%: Baseline CNN (F1: 0.6833)
│       ├── best_model_50pct.pt         # Top 50%: Baseline CNN (F1: 0.7050)
│       ├── best_model_100pct.pt        # Top 100%: Baseline CNN (F1: 0.7272)
│       ├── model_CNN_{10,25,50,100}pct.pt   # All Baseline CNN checkpoints
│       ├── model_AE_{10,25,50,100}pct.pt    # All AE-CNN checkpoints
│       └── model_VAE_{10,25,50,100}pct.pt   # All VAE-CNN checkpoints
├── results/
│   ├── experiment_results.csv          # Exact 12-run experimental metrics matrix
│   └── ptbxl_representation_learning_results.csv
├── docs/
│   ├── methodology.md                  # Comprehensive scientific methodology
│   └── model_card.md                   # Standardized AI model card
└── sample_data/
    ├── generate_samples.py             # Synthetic 12-lead ECG generator (CSV, NPY, WFDB)
    ├── sample_normal_ecg.csv           # Normal sinus rhythm sample (CSV)
    ├── sample_normal_ecg.npy           # Normal sinus rhythm sample (NumPy)
    ├── sample_normal_ecg.hea           # Normal sinus rhythm sample (WFDB header)
    ├── sample_normal_ecg.dat           # Normal sinus rhythm sample (WFDB binary signal)
    ├── sample_mi_ecg.csv               # Acute MI pattern sample (CSV)
    ├── sample_mi_ecg.hea               # Acute MI sample (WFDB header)
    ├── sample_mi_ecg.dat               # Acute MI sample (WFDB binary signal)
    ├── sample_cd_ecg.csv               # Conduction disturbance sample (CSV)
    ├── sample_cd_ecg.hea               # Conduction disturbance sample (WFDB header)
    └── sample_cd_ecg.dat               # Conduction disturbance sample (WFDB binary signal)
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
