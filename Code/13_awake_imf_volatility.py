import os
import json
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.signal import hilbert
import mne
from PyEMD import EMD

"""
Instantaneous Frequency Volatility Analysis (Figure 6E).
This script utilizes Empirical Mode Decomposition (EMD) and Hilbert Transform 
to compute the instantaneous frequency (IF) volatility of the first two Intrinsic 
Mode Functions (IMF1, IMF2) during pre-yawning and matched control epochs.
Results are visualized using publication-quality violin and strip plots.
"""

# ==========================================
# 0. Global Plotting Configuration (Publication Standard)
# ==========================================
warnings.filterwarnings('ignore')

# Typography settings for editable PDFs
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Line widths and tick sizes
mpl.rcParams['axes.linewidth'] = 1.2      
mpl.rcParams['xtick.major.width'] = 1.2   
mpl.rcParams['ytick.major.width'] = 1.2   
mpl.rcParams['lines.linewidth'] = 1.5     
mpl.rcParams['xtick.major.size'] = 5      
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['xtick.direction'] = 'out'   
mpl.rcParams['ytick.direction'] = 'out'

# Font sizing
mpl.rcParams['axes.labelsize'] = 12       
mpl.rcParams['xtick.labelsize'] = 10      
mpl.rcParams['ytick.labelsize'] = 10      
mpl.rcParams['axes.titlesize'] = 12       
mpl.rcParams['legend.fontsize'] = 10      

# ==========================================
# 1. Core Algorithms: Instantaneous Frequency Volatility
# ==========================================
def compute_if_metrics(data, fs):
    """
    Calculate Instantaneous Frequency (IF) Volatility:
    1. Volatility of IMF1 IF (Standard Deviation)
    2. Volatility of IMF2 IF (Standard Deviation)
    """
    emd = EMD()
    emd.emd_num_imfs = 3 
    
    try:
        imfs = emd(data)
    except Exception:
        return np.nan, np.nan

    if imfs.ndim < 2 or imfs.shape[0] < 2:
        return np.nan, np.nan
        
    vol_ifs = []
    for i in range(2):
        imf = imfs[i]
        
        # Apply Analytic Signal via Hilbert Transform
        analytic_signal = hilbert(imf)
        
        # Extract Unwrapped Instantaneous Phase
        phase = np.unwrap(np.angle(analytic_signal))
        
        # Compute Instantaneous Frequency: f = (1/2pi) * d(phase)/dt
        instantaneous_frequency = np.diff(phase) / (2.0 * np.pi) * fs
        
        # Quantify IF volatility via Standard Deviation
        vol_if = np.std(instantaneous_frequency)
        vol_ifs.append(vol_if)
    
    return vol_ifs[0], vol_ifs[1]

# ==========================================
# 2. Data Loading & Signal Processing
# ==========================================
def load_and_process_features(json_path, group_label, time_window_sec=30):
    """Load JSON metadata, extract 30s epochs, and compute EMD metrics."""
    results = []
    print(f"--- Processing {group_label} Group (IF Volatility Analysis) ---")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        return pd.DataFrame()
    
    time_keys = [k for k in config.keys() if 'times' in k]
    if not time_keys: 
        return pd.DataFrame()
    
    file_map = config[time_keys[0]]
    mne.set_log_level('WARNING') 
    
    for file_path, timestamps in file_map.items():
        try:
            # Path tolerance processing
            if not os.path.exists(file_path):
                alt_path = os.path.join(os.path.dirname(json_path), os.path.basename(file_path))
                if os.path.exists(alt_path):
                    file_path = alt_path
                else:
                    print(f"Warning: Data file not found: {file_path}")
                    continue

            raw = mne.io.read_raw_fif(file_path, preload=True, verbose='ERROR')
            
            # Apply strict 45Hz low-pass filter to constrain IMF1 within Beta/Low-Gamma ranges
            raw.filter(l_freq=None, h_freq=45.0, fir_design='firwin', verbose='ERROR')
            
            picks = mne.pick_types(raw.info, eeg=True)
            if len(picks) > 3: 
                picks = picks[:3]
            if len(picks) == 0: 
                continue
            
            if raw.info['sfreq'] > 250: 
                raw.resample(250)
            fs = raw.info['sfreq']
            
            # Ensure timestamps is iterable
            if not isinstance(timestamps, (list, np.ndarray)):
                timestamps = [timestamps]

            for t_end in timestamps:
                t_start = t_end - time_window_sec
                if t_start < 0 or t_end > raw.times[-1]: 
                    continue
                
                raw_crop = raw.copy().crop(tmin=t_start, tmax=t_end)
                data = raw_crop.get_data(picks=picks) * 1e6 
                data_mean = np.mean(data, axis=0)
                
                v_if1, v_if2 = compute_if_metrics(data_mean, fs)
                
                if not np.isnan(v_if1):
                    results.append({
                        'Group': group_label,
                        'IMF1_IF_Vol': v_if1,
                        'IMF2_IF_Vol': v_if2
                    })
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            
    return pd.DataFrame(results)

# ==========================================
# 3. Plotting Functions (Aesthetic Publication Style)
# ==========================================
def plot_aesthetic_publication_figure(ax, data, feature, title, y_label):
    """
    Generate publication-ready figures comprising:
    Violin Plot (Density) + Strip Plot (Individual Data) + Point Plot (Mean/CI).
    """
    palette = {'Non-Yawn': '#B0B0B0', 'Pre-Yawn': '#D62728'}
    order = ['Non-Yawn', 'Pre-Yawn']
    
    # 1. Violin Plot: Distribution density
    sns.violinplot(x='Group', y=feature, data=data, order=order, palette=palette, 
                   ax=ax, inner=None, linewidth=0, saturation=0.85, width=0.7, zorder=1)
    
    # 2. Strip Plot: Individual variations
    sns.stripplot(x='Group', y=feature, data=data, order=order, 
                  color='#333333', alpha=0.6, jitter=0.2, size=4, ax=ax, zorder=5)
    
    # 3. Point Plot: Mean and 95% Confidence Interval
    sns.pointplot(x='Group', y=feature, data=data, order=order,
                  estimator=np.mean, errorbar=('ci', 95),
                  color='black', scale=0.7, capsize=0.1, markers='o', 
                  err_kws={'linewidth': 2}, ax=ax, zorder=10)

    # 4. Statistical Testing & Annotations (Paired T-test)
    group1 = data[data['Group'] == 'Non-Yawn'][feature].values
    group2 = data[data['Group'] == 'Pre-Yawn'][feature].values
    
    if len(group1) > 1 and len(group2) > 1:
        t_stat, p_val = stats.ttest_rel(group1, group2)
        
        # Calculate bracket positions
        y_max = data[feature].max()
        y_range = data[feature].max() - data[feature].min()
        y_line = y_max + y_range * 0.05 
        h = y_range * 0.02 
        
        ax.plot([0, 0, 1, 1], [y_line-h, y_line, y_line, y_line-h], lw=1.2, c='black')
        
        # Determine significance stars
        if p_val < 0.001: star = '***'
        elif p_val < 0.01: star = '**'
        elif p_val < 0.05: star = '*'
        else: star = 'ns'
        
        ax.text(0.5, y_line + h/2, star, ha='center', va='bottom', fontsize=14, fontweight='bold')
    else:
        y_range = 1.0 
        y_line = data[feature].max() if not data.empty else 1.0

    # 5. Aesthetic Enhancements
    ax.set_title(title, fontweight='bold', pad=15) 
    ax.set_ylabel(y_label, fontweight='bold')      
    ax.set_xlabel('') 
    
    sns.despine(ax=ax, trim=False)
    ax.set_ylim(bottom=data[feature].min() - y_range*0.05, top=y_line + y_range*0.15)


# ==========================================
# 4. Main Execution Pipeline
# ==========================================
if __name__ == "__main__":
    path_yawn = r"D:\EEG\处理后的脑电数据\eeg_yawn_config_bg3.json"
    path_control = r"D:\EEG\处理后的脑电数据\eeg_control_config_bg3.json"

    if not os.path.exists(path_yawn) or not os.path.exists(path_control):
        print("Warning: One or both configuration files not found.")
        print("Generating MOCK DATA for aesthetics demonstration...")
        
        np.random.seed(42)
        N_samples = 40
        yawn_imf1 = np.random.normal(loc=5.0, scale=1.5, size=N_samples)
        yawn_imf2 = np.random.normal(loc=3.0, scale=0.8, size=N_samples)
        ctrl_imf1 = np.random.normal(loc=9.0, scale=2.5, size=N_samples)
        ctrl_imf2 = np.random.normal(loc=3.5, scale=1.0, size=N_samples)
        
        df_yawn = pd.DataFrame({'Group': 'Pre-Yawn', 'IMF1_IF_Vol': yawn_imf1, 'IMF2_IF_Vol': yawn_imf2})
        df_control = pd.DataFrame({'Group': 'Non-Yawn', 'IMF1_IF_Vol': ctrl_imf1, 'IMF2_IF_Vol': ctrl_imf2})
    else:
        df_yawn = load_and_process_features(path_yawn, 'Pre-Yawn')
        df_control = load_and_process_features(path_control, 'Non-Yawn')

    if df_yawn.empty or df_control.empty:
        print("Error: No data extracted. Please check your data files and processing logic.")
    else:
        # Align sample sizes for Paired T-test
        if len(df_yawn) != len(df_control):
            print(f"Note: Truncating data for paired test. Yawn N={len(df_yawn)}, Control N={len(df_control)}")
            min_len = min(len(df_yawn), len(df_control))
            df_yawn = df_yawn.iloc[:min_len]
            df_control = df_control.iloc[:min_len]

        df_all = pd.concat([df_yawn, df_control], ignore_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(9, 5), dpi=300)
        plt.subplots_adjust(wspace=0.35)

        # --- Plot 1: IMF1 IF Volatility ---
        plot_aesthetic_publication_figure(
            axes[0], df_all, 'IMF1_IF_Vol', 
            title='High-Freq Stability\n(IMF1 Component)', 
            y_label='IF Volatility (Hz, Std Dev)'
        )

        # --- Plot 2: IMF2 IF Volatility ---
        plot_aesthetic_publication_figure(
            axes[1], df_all, 'IMF2_IF_Vol', 
            title='Mid-Freq Stability\n(IMF2 Component)', 
            y_label='IF Volatility (Hz, Std Dev)'
        )

        # Save the figure
        output_dir = r"D:\EEG\Figures\Publication_Ready" 
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)
            
        save_path = os.path.join(output_dir, 'Fig_IMF_IF_Volatility_Aesthetic.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        
        print(f"Analysis complete. Aesthetic figure saved to: {save_path}")
        plt.show()