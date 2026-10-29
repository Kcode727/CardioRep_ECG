"""
Generate synthetic 12-lead ECG signals formatted for the PTB-XL classification model.
Each signal is 10 seconds sampled at 100 Hz (1,000 time steps across 12 leads).
Leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
"""

import numpy as np
import pandas as pd
from pathlib import Path

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 100  # 100 Hz
DURATION = 10  # 10 seconds
N_SAMPLES = FS * DURATION  # 1000 samples

def generate_ecg(heart_rate=72, st_elevation=0.0, qrs_width_factor=1.0, noise_std=0.02, seed=42):
    rng = np.random.default_rng(seed)
    time = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    rr_interval = 60.0 / heart_rate  # seconds per beat
    
    # 3D cardiac dipole projection over time
    # Vector loop components: Vx (frontal left-right), Vy (frontal superior-inferior), Vz (anterior-posterior)
    # We will compute lead signals via vector projection:
    # Lead I = Vx
    # Lead II = 0.5 * Vx + 0.866 * Vy
    # Lead III = Lead II - Lead I = -0.5 * Vx + 0.866 * Vy
    # aVR = - (Lead I + Lead II) / 2
    # aVL = (Lead I - Lead III) / 2
    # aVF = (Lead II + Lead III) / 2
    # V1 to V6: projections onto horizontal plane with R-wave progression
    
    # Generate beat center times
    beat_times = np.arange(0.4, DURATION, rr_interval)
    
    vx = np.zeros(N_SAMPLES)
    vy = np.zeros(N_SAMPLES)
    vz = np.zeros(N_SAMPLES)
    
    for t_beat in beat_times:
        dt = time - t_beat
        
        # P wave (at dt ~ -0.16s)
        p_dt = dt + 0.16
        p_wave = np.exp(-0.5 * (p_dt / 0.035)**2)
        vx += 0.12 * p_wave
        vy += 0.18 * p_wave
        vz += -0.05 * p_wave
        
        # Q wave (at dt ~ -0.03s)
        q_dt = dt + 0.035 * qrs_width_factor
        q_wave = np.exp(-0.5 * (q_dt / (0.012 * qrs_width_factor))**2)
        vx += -0.15 * q_wave
        vy += -0.10 * q_wave
        vz += 0.05 * q_wave
        
        # R wave (at dt ~ 0.0s)
        r_dt = dt
        r_wave = np.exp(-0.5 * (r_dt / (0.025 * qrs_width_factor))**2)
        vx += 0.85 * r_wave
        vy += 1.30 * r_wave
        vz += -0.70 * r_wave
        
        # S wave (at dt ~ +0.04s)
        s_dt = dt - 0.04 * qrs_width_factor
        s_wave = np.exp(-0.5 * (s_dt / (0.018 * qrs_width_factor))**2)
        vx += -0.25 * s_wave
        vy += -0.35 * s_wave
        vz += 0.40 * s_wave
        
        # ST segment elevation / depression
        if st_elevation != 0.0:
            st_dt = dt - 0.12
            st_wave = np.exp(-0.5 * (st_dt / 0.06)**2)
            vy += st_elevation * st_wave
            vz += -st_elevation * 0.8 * st_wave
            
        # T wave (at dt ~ +0.28s)
        t_dt = dt - 0.28
        t_wave = np.exp(-0.5 * (t_dt / 0.07)**2)
        vx += 0.22 * t_wave
        vy += 0.40 * t_wave
        vz += -0.15 * t_wave
        
    # Baseline wander & subtle physiological noise
    wander = 0.05 * np.sin(2 * np.pi * 0.15 * time) + 0.03 * np.cos(2 * np.pi * 0.3 * time)
    
    # Calculate 12 leads from vector dipole:
    leads_data = {}
    leads_data["I"] = vx + wander + rng.normal(0, noise_std, N_SAMPLES)
    leads_data["II"] = 0.5 * vx + 0.866 * vy + wander + rng.normal(0, noise_std, N_SAMPLES)
    leads_data["III"] = leads_data["II"] - leads_data["I"]
    leads_data["aVR"] = -0.5 * (leads_data["I"] + leads_data["II"])
    leads_data["aVL"] = 0.5 * (leads_data["I"] - leads_data["III"])
    leads_data["aVF"] = 0.5 * (leads_data["II"] + leads_data["III"])
    
    # Precordial leads V1-V6 transition
    precordial_coeffs = [
        # (vx_weight, vy_weight, vz_weight)
        (-0.2, 0.1, 1.2),   # V1: mostly anterior (+z), small r, deep S
        (-0.1, 0.2, 1.0),   # V2
        (0.2, 0.5, 0.6),    # V3 (transition)
        (0.6, 0.8, 0.2),    # V4
        (0.9, 0.9, -0.3),   # V5: lateral, tall R
        (0.7, 0.7, -0.5),   # V6
    ]
    
    for i, v_name in enumerate(["V1", "V2", "V3", "V4", "V5", "V6"]):
        cx, cy, cz = precordial_coeffs[i]
        leads_data[v_name] = (cx * vx + cy * vy + cz * vz) + wander + rng.normal(0, noise_std, N_SAMPLES)
        
    df = pd.DataFrame(leads_data)[LEADS]
    return df

def main():
    out_dir = Path("sample_data")
    out_dir.mkdir(exist_ok=True)
    
    # 1. Normal Sinus Rhythm (NORM)
    df_norm = generate_ecg(heart_rate=70, st_elevation=0.0, qrs_width_factor=1.0, seed=42)
    df_norm.to_csv(out_dir / "sample_normal_ecg.csv", index=False)
    np.save(out_dir / "sample_normal_ecg.npy", df_norm.to_numpy().T)  # Shape (12, 1000)
    print("Generated sample_normal_ecg.csv and sample_normal_ecg.npy")
    
    # 2. Myocardial Infarction pattern (Inferior / Anterior ST-elevation & prominent Q)
    df_mi = generate_ecg(heart_rate=84, st_elevation=0.55, qrs_width_factor=1.1, seed=101)
    df_mi.to_csv(out_dir / "sample_mi_ecg.csv", index=False)
    print("Generated sample_mi_ecg.csv")
    
    # 3. Conduction Disturbance pattern (Prolonged wide QRS bundle branch pattern)
    df_cd = generate_ecg(heart_rate=76, st_elevation=0.05, qrs_width_factor=1.8, seed=202)
    df_cd.to_csv(out_dir / "sample_cd_ecg.csv", index=False)
    print("Generated sample_cd_ecg.csv")

if __name__ == "__main__":
    main()
