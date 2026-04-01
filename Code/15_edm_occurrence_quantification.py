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
    raise ImportError("Please install PyEMD library first: pip install PyEMD")

"""
Quantification of Energy-Dynamics Mismatch (EDM) Occurrence.
This script evaluates the frequency of the EDM phenomenon across three cohorts:
Dog Induction, Awake Beagle, and Human Propofol Anesthesia. 
It establishes an intra-epoch self-baseline to define 75th/25th percentile thresholds 
for Gamma power and Instantaneous Frequency Volatility (IFV), respectively.
Statistical significance is verified using a one-tailed Fisher's exact test.
"""

# ==========================================
# Global Plotting Configuration (Publication Standard)
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif', 
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.linewidth': 0.5,       # Axis linewidth 0.5pt
    'xtick.major.width': 0.5,    # X-axis tick width 0.5pt
    'ytick.major.width': 0.5,    # Y-axis tick width 0.5pt
    'xtick.major.size': 6.0,     # X-axis tick length 6pt
    'ytick.major.size': 6.0,     # Y-axis tick length 6pt
    'pdf.fonttype': 42,          # Ensure editable TrueType fonts in exported PDFs
    'ps.fonttype': 42
})

# ==========================================
# Minimalist Time Window Configuration
# ==========================================
BASE_START, BASE_END = -30.0, -28.0  # Intra-epoch baseline window (for percentile thresholds)
ANA_START, ANA_END = -8.0, 0.0       # Action mutation window (for detecting extrema)

FILES_CONFIG = {
    'Dog Induction (Cluster 2)': {
        'yawn_path': r"D:\EEG\处理后的脑电数据\cluster_2_yawns.json",
        'ctrl_path': r"D:\EEG\处理后的脑电数据\control_induction_times.json",
        'imf_target': 1, 'color': '#9467bd'  # Induction phase, Extract IMF2
    },
    'Awake (Beagle)': {
        'yawn_path': r"D:\EEG\处理后的脑电数据\eeg_yawn_config_bg3.json",
        'ctrl_path': r"D:\EEG\处理后的脑电数据\eeg_control_config_bg3.json",
        'imf_target': 0, 'color': '#2ca02c'  # Awake phase, Extract IMF1
    },
    'Human Propofol': {
        'yawn_path': r"D:\EEG\处理后的脑电数据\human_yawn_config_Propofol.json",
        'ctrl_path': r"D:\EEG\处理后的脑电数据\non-yawn\control_data_b.json",
        'imf_target': 1, 'color': '#d62728'  # Human Propofol, Extract IMF2
    }
}

def universal_load(json_path):
    """Load JSON configuration and extract target timestamps."""
    if not os.path.exists(json_path): 
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f: 
            data = json.load(f)
    except Exception: 
        return {}
        
    if 'yawn_times' in data: 
        return data['yawn_times']
    if 'control_times' in data: 
        return data['control_times']
        
    clean_map = {}
    for k, v in data.items():
        if isinstance(v, list) and ('.fif' in k.lower() or '\\' in k): 
            clean_map[k] = v
    return clean_map

def find_file_smart(target_path, search_dir=r"D:\EEG\处理后的脑电数据"):
    """Attempt to locate the target file in the default directory if absolute path fails."""
    if os.path.exists(target_path): 
        return target_path
    filename = os.path.basename(target_path)
    candidate = os.path.join(search_dir, filename)
    if os.path.exists(candidate): 
        return candidate
    return None

def extract_smoothed_features(raw, tmin, tmax, imf_idx):
    """
    Extract and smooth features with a 1-second window to prevent 
    high-frequency artifacts from interfering with extrema judgments.
    """
    try:
        raw_crop = raw.copy().crop(tmin, tmax)
    except Exception:
        return None, None
        
    fs = raw_crop.info['sfreq']
    win_pts = int(1.0 * fs)
    
    # 1. Gamma Power Envelope
    raw_gamma = raw_crop.copy().filter(l_freq=30.0, h_freq=45.0, verbose=False)
    gamma_power = np.abs(hilbert(raw_gamma.get_data().mean(axis=0))) ** 2
    gamma_smooth = pd.Series(gamma_power).rolling(window=win_pts, center=True, min_periods=1).mean().values
    
    # 2. Instantaneous Frequency Volatility (IFV)
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

# --------------------------
# Main Algorithm: Binarization via Unified Self-Baseline
# --------------------------
def evaluate_unified_baseline_edm():
    """Extract EDM index applying an intra-epoch self-baseline methodology."""
    print(">>> Applying Intra-epoch Self-Baseline to extract EDM index...")
    results = []
    
    for group_name, cfg in FILES_CONFIG.items():
        imf_idx = cfg['imf_target']
        
        for cond, path_key in [('Pre-Yawn', 'yawn_path'), ('Control', 'ctrl_path')]:
            data_map = universal_load(cfg[path_key])
            for r_path, times in data_map.items():
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
                        
                        # 1. Extract Self-Baseline Epoch (-30s to -28s)
                        gamma_base, ifv_base = extract_smoothed_features(raw, t_val + BASE_START, t_val + BASE_END, imf_idx)
                        # 2. Extract Event Test Window (-8s to 0s)
                        gamma_ana, ifv_ana = extract_smoothed_features(raw, t_val + ANA_START, t_val + ANA_END, imf_idx)
                        
                        if any(x is None for x in [gamma_base, ifv_base, gamma_ana, ifv_ana]): 
                            continue
                        
                        # [Core Minimalist Logic]: Calculate Quartile Thresholds
                        thresh_gamma_75 = np.percentile(gamma_base, 75) # Gamma breaks upward above 75%
                        thresh_ifv_25 = np.percentile(ifv_base, 25)     # IFV breaks downward below 25%
                        
                        # Judgment: Is there a breakthrough in the test window?
                        cond_gamma_surge = np.max(gamma_ana) > thresh_gamma_75
                        cond_ifv_drop = np.min(ifv_ana) < thresh_ifv_25
                        
                        # EDM = 1 only if BOTH conditions are met simultaneously
                        is_edm = 1 if (cond_gamma_surge and cond_ifv_drop) else 0
                        
                        results.append({
                            'Group': group_name,
                            'Condition': cond,
                            'Is_EDM': is_edm
                        })
                    raw.close()
                except Exception: 
                    continue
                
    return pd.DataFrame(results)

def plot_results(df, p_values):
    """Plot the occurrence rates with statistical annotations."""
    if df.empty: 
        return
    
    # Calculate occurrence rate percentage
    summary = df.groupby(['Group', 'Condition'])['Is_EDM'].mean().reset_index()
    summary['Is_EDM'] *= 100.0 
    
    fig, ax = plt.subplots(figsize=(8, 6))
    groups = list(FILES_CONFIG.keys())
    
    # Color palette: Control (Gray), Pre-Yawn (Red)
    palette = {'Control': '#7f7f7f', 'Pre-Yawn': '#d62728'}
    
    # Force bar order: Control on the left, Pre-Yawn on the right
    sns.barplot(data=summary, x='Group', y='Is_EDM', hue='Condition', 
                order=groups, hue_order=['Control', 'Pre-Yawn'], 
                palette=palette, edgecolor='black', linewidth=0.5, ax=ax)
    
    # Adjust Y-axis limit to accommodate statistical stars and brackets
    max_y = summary['Is_EDM'].max()
    ax.set_ylim(0, max_y + 20) 
    
    # ==========================================
    # Automatically Draw Significance Brackets & Stars (Linewidth 0.5pt)
    # ==========================================
    for i, group in enumerate(groups):
        p_val = p_values.get(group, 1.0)
        
        if p_val < 0.05:
            # Extract bar heights for the current group
            h_ctrl = summary[(summary['Group'] == group) & (summary['Condition'] == 'Control')]['Is_EDM'].values[0]
            h_pre = summary[(summary['Group'] == group) & (summary['Condition'] == 'Pre-Yawn')]['Is_EDM'].values[0]
            
            # Bracket base height: slightly above the tallest bar
            y_base = max(h_ctrl, h_pre) + 3.0
            line_height = 2.0 
            
            # Bar center X-coordinates (Seaborn default offset is ~0.2)
            x1 = i - 0.2
            x2 = i + 0.2
            
            # Draw bracket (linewidth set to 0.5pt)
            ax.plot([x1, x1, x2, x2], [y_base, y_base + line_height, y_base + line_height, y_base], 
                    lw=0.5, color='black')
            
            # Determine star rating
            stars = '**' if p_val < 0.01 else '*'
            
            # Add star annotation
            ax.text((x1 + x2) * 0.5, y_base + line_height + 0.5, stars, 
                    ha='center', va='bottom', color='black', fontsize=14, fontweight='bold')

    # Chart Decorations
    ax.set_title('EDM Occurrence Rate (Unified Self-Baseline Method)', fontweight='bold', pad=15)
    ax.set_ylabel('EDM Event Frequency (%)', fontweight='bold')
    ax.set_xlabel('')
    plt.xticks(rotation=15)
    
    sns.despine()
    ax.legend(title='Condition', loc='upper right', frameon=False)
    
    plt.tight_layout()
    save_path = r"D:\EEG\Figures\Figure_6H_EDM_Quantification.pdf"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\n   [✓] Publication-ready PDF saved to: {save_path}")

def main():
    mne.set_log_level('ERROR')
    warnings.filterwarnings('ignore')
    
    print("=" * 75)
    print("M1 Ultimate Chart Edition: Intra-epoch Quartile Baseline (75th/25th)")
    print("Baseline: -30 to -28s | Action Window: -8 to 0s")
    print("Criterion: Gamma_Max > Base_75th AND IFV_Min < Base_25th")
    print("=" * 75)
    
    df_res = evaluate_unified_baseline_edm()
    
    if df_res.empty: 
        return
        
    print("\n" + "=" * 75)
    print("Final Statistical Testing Results (One-tailed Fisher's Exact Test)")
    print("=" * 75)
    
    p_values_dict = {}
    
    for group in df_res['Group'].unique():
        df_g = df_res[df_res['Group'] == group]
        df_yawn = df_g[df_g['Condition'] == 'Pre-Yawn']
        df_ctrl = df_g[df_g['Condition'] == 'Control']
        
        if len(df_yawn) == 0 or len(df_ctrl) == 0: 
            continue
        
        yawn_edm_cnt = df_yawn['Is_EDM'].sum()
        yawn_total = len(df_yawn)
        ctrl_edm_cnt = df_ctrl['Is_EDM'].sum()
        ctrl_total = len(df_ctrl)
        
        # Construct Fisher Contingency Table
        table = [[yawn_edm_cnt, yawn_total - yawn_edm_cnt],
                 [ctrl_edm_cnt, ctrl_total - ctrl_edm_cnt]]
        
        # One-tailed test: Probability of exceeding threshold is greater before yawn than in control
        _, p_fisher = fisher_exact(table, alternative='greater')
        
        p_values_dict[group] = p_fisher
        
        yawn_pct = (yawn_edm_cnt / yawn_total) * 100
        ctrl_pct = (ctrl_edm_cnt / ctrl_total) * 100
        
        print(f"\n[Cohort: {group}]")
        print(f"  > EDM Rate: Pre-Yawn {yawn_pct:.1f}% ({yawn_edm_cnt}/{yawn_total}) vs Control {ctrl_pct:.1f}% ({ctrl_edm_cnt}/{ctrl_total})")
        print(f"  > Fisher Exact P-value (one-tailed): {p_fisher:.4e} {'**' if p_fisher<0.01 else '*' if p_fisher<0.05 else '(ns)'}")
    
    plot_results(df_res, p_values_dict)

if __name__ == "__main__":
    main()