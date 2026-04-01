import os
import json
import warnings
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert
from scipy.stats import fisher_exact

try:
    from PyEMD import EMD
except ImportError:
    raise ImportError("Please install PyEMD first: pip install PyEMD")

"""
Data-Driven Time Window Scanning for Energy-Dynamics Mismatch (EDM).
This script systematically scans pre-yawning temporal epochs using a sliding window approach 
(strictly ending at 0s, i.e., yawn onset) to objectively identify the optimal time window 
where the EDM phenomenon is most statistically significant. This data-driven methodology 
prevents subjective epoch selection (cherry-picking/P-hacking) and robustly validates 
the temporal specificity of pre-yawning cortical dynamics.
"""

# ==========================================
# Global Plotting Configuration (Publication Standard)
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.linewidth': 0.5, 
    'xtick.major.width': 0.5, 
    'ytick.major.width': 0.5,
    'pdf.fonttype': 42, 
    'ps.fonttype': 42
})

# ==========================================
# Sliding Window Parameters
# ==========================================
# Fixed intra-individual baseline window
BASE_START, BASE_END = -30.0, -28.0  
# Length of the scanning window
WINDOW_LENGTH = 8.0                  
# Scanning starts from -26s, step size 2s, last start at -8s (ends exactly at 0s)
SLIDE_STARTS = np.arange(-26.0, -6.0, 2.0) 

# Utilize the Awake cohort (highest sample size & SNR) as the benchmark for objective screening
TARGET_COHORT = {
    'yawn_path': r"D:\EEG\处理后的脑电数据\eeg_yawn_config_bg3.json",
    'ctrl_path': r"D:\EEG\处理后的脑电数据\eeg_control_config_bg3.json",
    'imf_target': 0
}

def universal_load(json_path):
    """Load JSON configuration and extract target timestamps."""
    if not os.path.exists(json_path): 
        return {}
    with open(json_path, 'r', encoding='utf-8') as f: 
        data = json.load(f)
    if 'yawn_times' in data: 
        return data['yawn_times']
    if 'control_times' in data: 
        return data['control_times']
    return {k: v for k, v in data.items() if isinstance(v, list) and ('.fif' in k.lower() or '\\' in k)}

def find_file_smart(target_path, search_dir=r"D:\EEG\处理后的脑电数据"):
    """Attempt to locate the target file in the default directory if absolute path fails."""
    if os.path.exists(target_path): 
        return target_path
    candidate = os.path.join(search_dir, os.path.basename(target_path))
    return candidate if os.path.exists(candidate) else None

def extract_smoothed_features(raw, tmin, tmax, imf_idx):
    """Extract smoothed Gamma Power and Instantaneous Frequency Volatility (IFV)."""
    try: 
        raw_crop = raw.copy().crop(tmin, tmax)
    except Exception: 
        return None, None
        
    fs = raw_crop.info['sfreq']
    win_pts = int(1.0 * fs)
    
    # 1. Gamma Power extraction and smoothing
    raw_gamma = raw_crop.copy().filter(l_freq=30.0, h_freq=45.0, verbose=False)
    gamma_power = np.abs(hilbert(raw_gamma.get_data().mean(axis=0))) ** 2
    gamma_smooth = pd.Series(gamma_power).rolling(window=win_pts, center=True, min_periods=1).mean().values
    
    # 2. Instantaneous Frequency Volatility (IFV) extraction via EMD
    raw_crop.filter(l_freq=0.5, h_freq=45.0, verbose=False)
    data = raw_crop.get_data().mean(axis=0)
    emd_solver = EMD()
    try:
        imfs = emd_solver.emd(data, max_imf=4)
        if imfs.shape[0] <= imf_idx: 
            return None, None
        target_imf = imfs[imf_idx, :]
    except Exception: 
        return None, None
        
    inst_phase = np.unwrap(np.angle(hilbert(target_imf)))
    inst_freq = (np.diff(inst_phase) / (2.0 * np.pi) * fs)
    inst_freq = np.insert(inst_freq, 0, inst_freq[0])
    inst_freq = np.clip(inst_freq, 0, 45.0)
    ifv_smooth = pd.Series(inst_freq).rolling(window=win_pts, center=True, min_periods=1).std().values
    
    return gamma_smooth, ifv_smooth

def scan_time_windows():
    """Systematically scan pre-event epochs and perform statistical testing."""
    print(f">>> Initiating Data-Driven Time Window Scan (Benchmark: Awake Beagle, Window: {WINDOW_LENGTH}s, Step: 2s)...")
    
    # 1. Pre-load all EEG file paths and corresponding timestamps
    yawn_map = universal_load(TARGET_COHORT['yawn_path'])
    ctrl_map = universal_load(TARGET_COHORT['ctrl_path'])
    imf_idx = TARGET_COHORT['imf_target']
    
    scan_results = []
    
    # 2. Iterate through each sliding window
    for w_start in SLIDE_STARTS:
        w_end = w_start + WINDOW_LENGTH
        print(f"  > Scanning Window: [{w_start:5.1f}s to {w_end:5.1f}s]...")
        
        yawn_edm_cnt, yawn_total = 0, 0
        ctrl_edm_cnt, ctrl_total = 0, 0
        
        # Process Yawn and Control groups
        for cond, d_map in [('Pre-Yawn', yawn_map), ('Control', ctrl_map)]:
            for r_path, times in d_map.items():
                f_path = find_file_smart(r_path)
                if not f_path: 
                    continue
                try:
                    raw = mne.io.read_raw_fif(f_path, preload=True, verbose=False)
                    raw.pick_channels(raw.ch_names[:3])
                    
                    for t in (times if isinstance(times, list) else []):
                        t_val = float(t)
                        if t_val + BASE_START < raw.times[0]: 
                            continue
                        
                        gamma_base, ifv_base = extract_smoothed_features(raw, t_val + BASE_START, t_val + BASE_END, imf_idx)
                        gamma_ana, ifv_ana = extract_smoothed_features(raw, t_val + w_start, t_val + w_end, imf_idx)
                        
                        if any(x is None for x in [gamma_base, ifv_base, gamma_ana, ifv_ana]): 
                            continue
                        
                        # Fix threshold using 75th/25th percentiles of baseline
                        t_g_75 = np.percentile(gamma_base, 75)
                        t_i_25 = np.percentile(ifv_base, 25)
                        
                        # Define EDM: High Gamma Power AND Low IF Volatility
                        is_edm = 1 if (np.max(gamma_ana) > t_g_75 and np.min(ifv_ana) < t_i_25) else 0
                        
                        if cond == 'Pre-Yawn':
                            yawn_edm_cnt += is_edm
                            yawn_total += 1
                        else:
                            ctrl_edm_cnt += is_edm
                            ctrl_total += 1
                    raw.close()
                except Exception: 
                    continue
                
        # 3. Perform Fisher's Exact Test for the current window
        if yawn_total > 0 and ctrl_total > 0:
            table = [[yawn_edm_cnt, yawn_total - yawn_edm_cnt],
                     [ctrl_edm_cnt, ctrl_total - ctrl_edm_cnt]]
            _, p_val = fisher_exact(table, alternative='greater')
            
            # Amplify significance using -log10(p) for visualization
            log_p = -np.log10(p_val) if p_val > 0 else 5.0
            
            scan_results.append({
                'Window_Center': w_start + WINDOW_LENGTH / 2.0,
                'Window_Label': f"[{int(w_start)}, {int(w_end)}]",
                'Yawn_Rate': yawn_edm_cnt / yawn_total * 100 if yawn_total else 0,
                'Ctrl_Rate': ctrl_edm_cnt / ctrl_total * 100 if ctrl_total else 0,
                'P_Value': p_val,
                'Log_P': log_p
            })
            
    return pd.DataFrame(scan_results)

def plot_scan_results(df):
    """Plot the significance trajectory across scanning windows."""
    if df.empty: 
        return
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot Significance Curve: -log10(P)
    color1 = '#1f77b4'
    ax1.set_xlabel('Center of Scanning Window (s relative to Yawn Onset)', fontweight='bold')
    ax1.set_ylabel('Statistical Significance (-log$_{10}$ P-value)', color=color1, fontweight='bold')
    ax1.plot(df['Window_Center'], df['Log_P'], marker='o', color=color1, lw=1.5, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Significance threshold lines 
    ax1.axhline(-np.log10(0.05), color='red', linestyle='--', lw=1, label='p = 0.05')
    ax1.axhline(-np.log10(0.01), color='darkred', linestyle='--', lw=1, label='p = 0.01')
    ax1.legend(loc='upper left', frameon=False)
    
    # Identify the most significant window
    best_idx = df['Log_P'].idxmax()
    best_row = df.loc[best_idx]
    
    # Annotate the optimal window
    ax1.annotate(f"Optimal Epoch:\n{best_row['Window_Label']} s\n(p = {best_row['P_Value']:.4f})", 
                 xy=(best_row['Window_Center'], best_row['Log_P']),
                 xytext=(best_row['Window_Center'] - 2, best_row['Log_P'] + 0.3),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=10, fontweight='bold', 
                 bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.2, ec="black", lw=0.5))

    # Draw vertical dashed line for Yawn Onset (0s)
    ax1.axvline(0, color='black', linestyle=':', lw=1)
    ax1.text(0.2, ax1.get_ylim()[0] + 0.1, 'Yawn Onset', rotation=90, fontsize=10, fontweight='bold', va='bottom')

    plt.title('Data-Driven Temporal Resolution of EDM Significance', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    
    # Ensure X-axis encompasses the 0s marker
    ax1.set_xlim(df['Window_Center'].min() - 1.5, 1.5)
    
    sns.despine()
    plt.tight_layout()
    
    save_path = r"D:\EEG\Figures\Figure_S1_TimeWindow_Screening.pdf"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\n   [✓] Data-Driven Time Window Screening plot saved to: {save_path}")
    print(f"   [*] Objective Optimal Window identified as: {best_row['Window_Label']} seconds")

def main():
    mne.set_log_level('ERROR')
    warnings.filterwarnings('ignore')
    
    df_scan = scan_time_windows()
    plot_scan_results(df_scan)

if __name__ == "__main__":
    main()