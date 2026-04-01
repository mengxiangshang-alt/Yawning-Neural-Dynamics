import os
import json
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.signal import hilbert, butter, filtfilt
import mne

"""
Gamma Band Energy and Volatility Statistical Analysis (Figure 6C).
This script extracts the 30-45Hz Gamma band from pre-yawning and matched control EEG epochs, 
computes the total power energy and its temporal volatility (standard deviation), 
and visualizes the statistical differences using publication-ready violin-strip plots.
"""

# ==========================================
# 0. Global Plotting Configuration (Publication Standard)
# ==========================================
warnings.filterwarnings('ignore')

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Line and tick widths configured for clarity and print standards
mpl.rcParams['axes.linewidth'] = 1.0     
mpl.rcParams['xtick.major.width'] = 1.0   
mpl.rcParams['ytick.major.width'] = 1.0   
mpl.rcParams['lines.linewidth'] = 1.5     
mpl.rcParams['xtick.major.size'] = 5      
mpl.rcParams['ytick.major.size'] = 5

# Font size adaptation
mpl.rcParams['axes.labelsize'] = 12       
mpl.rcParams['xtick.labelsize'] = 10      
mpl.rcParams['ytick.labelsize'] = 10      
mpl.rcParams['axes.titlesize'] = 12       

# Color control
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['text.color'] = 'black'

# ==========================================
# 1. Core Algorithms: Volatility Metrics
# ==========================================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a Butterworth bandpass filter to the data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def compute_advanced_metrics(data, fs):
    """
    Compute specific high-frequency metrics:
    1. Gamma Power Energy: Total energy in the 30-45Hz band.
    2. Gamma Volatility: Standard deviation of the instantaneous power in the 30-45Hz band.
    """
    # --- Metric 1: Gamma Band Power Energy ---
    try:
        # Extract Gamma band (30.0 Hz - 44.9 Hz)
        gamma_data = butter_bandpass_filter(data, 30.0, 44.9, fs, order=4)
        gamma_env = np.abs(hilbert(gamma_data))
        gamma_power = gamma_env ** 2
        gamma_power_energy = np.sum(gamma_power)  
    except Exception:
        gamma_power_energy = np.nan

    # --- Metric 2: Gamma Band Power Volatility ---
    vol_gamma_power = np.std(gamma_power) if not np.isnan(gamma_power_energy) else np.nan
    
    return gamma_power_energy, vol_gamma_power

# ==========================================
# 2. Data Loading & Signal Processing
# ==========================================
def load_and_process_features(json_path, group_label, time_window_sec=30):
    """Load JSON metadata, extract 30s epochs, and compute metrics."""
    results = []
    print(f"--- Processing {group_label} Group (Strict 45Hz Filter) ---")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return pd.DataFrame()
    
    # Compatibility check: automatically find keys containing 'times'
    if isinstance(config, list):
        file_list = config
    else:
        time_keys = [k for k in config.keys() if 'times' in k]
        if not time_keys: 
            file_list = config[list(config.keys())[0]] if config else []
        else:
            file_list = config[time_keys[0]]

    # Convert to iterable format: (file_path, timestamps)
    if isinstance(file_list, dict):
        iterator = file_list.items()
    elif isinstance(file_list, list):
        iterator = []
        for item in file_list:
            if isinstance(item, dict):
                f_k = next((k for k in item.keys() if k in ['file', 'path', 'name', 'file_path']), None)
                t_k = next((k for k in item.keys() if k in ['times', 'time', 'timestamp']), None)
                if f_k and t_k:
                    iterator.append((item[f_k], item[t_k]))
    else:
        return pd.DataFrame()

    mne.set_log_level('ERROR') 
    
    for file_path, timestamps in iterator:
        try:
            # Path tolerance processing
            if not os.path.exists(file_path):
                alt_path = os.path.join(os.path.dirname(json_path), os.path.basename(file_path))
                if os.path.exists(alt_path):
                    file_path = alt_path
            
            if not os.path.exists(file_path):
                continue

            raw = mne.io.read_raw_fif(file_path, preload=True)
            
            # Ensure strict low-pass filtering below 45 Hz
            raw.filter(l_freq=None, h_freq=45.0, fir_design='firwin')
            
            picks = mne.pick_types(raw.info, eeg=True)
            if len(picks) > 3: 
                picks = picks[:3]
            if len(picks) == 0: 
                continue
            
            if raw.info['sfreq'] > 250: 
                raw.resample(250)
            fs = raw.info['sfreq']
            
            if not isinstance(timestamps, (list, np.ndarray)):
                timestamps = [timestamps]

            for t_end in timestamps:
                t_start = t_end - time_window_sec
                if t_start < 0 or t_end > raw.times[-1]: 
                    continue
                
                raw_crop = raw.copy().crop(tmin=t_start, tmax=t_end)
                data = raw_crop.get_data(picks=picks) * 1e6  # Convert to uV
                data_mean = np.mean(data, axis=0)
                
                v_gamma_energy, v_gamma_vol = compute_advanced_metrics(data_mean, fs)
                
                if not np.isnan(v_gamma_energy):
                    results.append({
                        'Group': group_label,
                        'Gamma_Energy': v_gamma_energy,
                        'Gamma_Vol': v_gamma_vol
                    })
                    
        except Exception as e:
            print(f"Skipping {os.path.basename(file_path)}: {e}")
            
    return pd.DataFrame(results)

# ==========================================
# 3. Plotting Functions (Violin + Strip Style)
# ==========================================
def plot_aesthetic_publication_figure(ax, data, feature, title, y_label):
    """
    Generate aesthetic publication-quality charts combining:
    Violin Plot (Density) + Strip Plot (Individuals) + Point Plot (Mean & 95% CI).
    """
    palette = {'Non-Yawn': '#B0B0B0', 'Pre-Yawn': '#D62728'}
    order = ['Non-Yawn', 'Pre-Yawn']
    
    # 1. Violin Plot: Display distribution density
    sns.violinplot(x='Group', y=feature, data=data, order=order, palette=palette, 
                   ax=ax, inner=None, linewidth=0, saturation=0.85, width=0.7, zorder=1)
    
    # 2. Strip Plot: Display individual variations
    sns.stripplot(x='Group', y=feature, data=data, order=order, 
                  color='#333333', alpha=0.6, jitter=0.2, size=4, ax=ax, zorder=5)
    
    # 3. Point Plot: Mean and 95% Confidence Interval
    sns.pointplot(x='Group', y=feature, data=data, order=order,
                  estimator=np.mean, errorbar=('ci', 95),
                  color='black', scale=0.7, capsize=0.1, markers='o', 
                  err_kws={'linewidth': 2}, ax=ax, zorder=10)

    # 4. Statistical Testing (Paired T-test)
    group1 = data[data['Group'] == 'Non-Yawn'][feature].values
    group2 = data[data['Group'] == 'Pre-Yawn'][feature].values
    
    if len(group1) > 1 and len(group2) > 1:
        t_stat, p_val = stats.ttest_rel(group1, group2)
        
        # Draw significance annotation lines
        y_max = data[feature].max()
        y_range = data[feature].max() - data[feature].min()
        y_line = y_max + y_range * 0.05
        h = y_range * 0.02
        
        ax.plot([0, 0, 1, 1], [y_line-h, y_line, y_line, y_line-h], lw=1.2, c='black')
        
        if p_val < 0.001: star = '***'
        elif p_val < 0.01: star = '**'
        elif p_val < 0.05: star = '*'
        else: star = 'ns'
        
        ax.text(0.5, y_line + h/2, star, ha='center', va='bottom', fontsize=14, fontweight='bold')

    # 5. Axis formatting and aesthetic touches
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_xlabel('')
    
    sns.despine(ax=ax, trim=False)
    
    # Adjust Y-axis limit to accommodate significance markers
    ax.set_ylim(bottom=data[feature].min() - y_range * 0.05, top=y_line + y_range * 0.15)


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    path_yawn = r"D:\EEG\处理后的脑电数据\eeg_yawn_config_bg3.json"
    path_control = r"D:\EEG\处理后的脑电数据\eeg_control_config_bg3.json"

    if not os.path.exists(path_yawn):
        print("Warning: Empirical data files not found. Generating Mock Data for Visualization...")
        np.random.seed(42)
        N = 30
        df_yawn = pd.DataFrame({
            'Group': 'Pre-Yawn',
            'Gamma_Energy': np.random.normal(100, 20, N),
            'Gamma_Vol': np.random.normal(15, 5, N)
        })
        df_control = pd.DataFrame({
            'Group': 'Non-Yawn',
            'Gamma_Energy': np.random.normal(150, 25, N),
            'Gamma_Vol': np.random.normal(25, 8, N)
        })
    else:
        df_yawn = load_and_process_features(path_yawn, 'Pre-Yawn')
        df_control = load_and_process_features(path_control, 'Non-Yawn')

    # Pair alignment (Truncate to match paired t-test requirements)
    if len(df_yawn) != len(df_control):
        min_len = min(len(df_yawn), len(df_control))
        df_yawn = df_yawn.iloc[:min_len]
        df_control = df_control.iloc[:min_len]

    df_all = pd.concat([df_yawn, df_control], ignore_index=True)

    if not df_all.empty:
        fig, axes = plt.subplots(1, 2, figsize=(9, 5), dpi=300) 
        plt.subplots_adjust(wspace=0.35) 

        # Subplot 1: Gamma Power Energy
        plot_aesthetic_publication_figure(
            axes[0], df_all, 'Gamma_Energy', 
            title='Gamma Power Energy\n(30-45Hz)', 
            y_label='Gamma Power Energy ($\mu V^2$)'
        )

        # Subplot 2: Gamma Power Volatility
        plot_aesthetic_publication_figure(
            axes[1], df_all, 'Gamma_Vol', 
            title='Volatility of Gamma Power\n(30 - 44.9Hz Band)', 
            y_label='Std of Gamma Power ($\mu V^2$)'
        )

        output_dir = r"D:\EEG\Figures"
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)
            
        save_path = os.path.join(output_dir, 'Fig_Final_Filtered_ViolinStyle.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        
        print(f"Execution successful. Aesthetic figure saved to: {save_path}")
        plt.show()
    else:
        print("Execution failed: No valid data available to plot.")