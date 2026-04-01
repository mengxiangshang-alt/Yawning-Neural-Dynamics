import os
import json
import math
import warnings
import numpy as np
import pandas as pd
import mne
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import scipy.stats as stats
import statsmodels.formula.api as smf
from PyEMD import EMD

"""
Regional EEG Features Extraction and Interaction Analysis (LME Version).
This script extracts Lempel-Ziv Complexity (LZC), Gamma Power, and 
EMD-based Instantaneous Frequency (IF) Volatility across cortical regions.
It employs a Linear Mixed-Effects Model (LME) to evaluate the main effects 
and interactions (Condition x Region), using Dog_ID as a random intercept 
to rigorously control for intra-subject pseudo-replication.
"""

# ==========================================
# 1. Configuration: File Paths and Parameters
# ==========================================
FILES_CONFIG = {
    'Dog Cluster 0': {
        'yawn_path': r"D:\EEG\处理后的脑电数据\cluster_0_yawns.json",
        'ctrl_path': r"D:\EEG\处理后的脑电数据\cluster_0_negative_controls.json",
    },
    'Dog Cluster 1': {
        'yawn_path': r"D:\EEG\处理后的脑电数据\cluster_1_yawns.json",
        'ctrl_path': r"D:\EEG\处理后的脑电数据\cluster_1_negative_controls.json",
    },
}

# External Mapping for Subject IDs (Dog_ID)
MAP_PATH = r"D:\EEG\Figures\哈欠脑电\FILE_TO_DOG_MAP.json"

WINDOW_SEC = 30.0

# 16-Channel Regional Mapping
REGION_MAP = {
    "Frontal": ["Chan 9", "Chan 11", "Chan 12", "Chan 14"],
    "Parietal": ["Chan 3", "Chan 6", "Chan 15", "Chan 16"],
    "Temporal": ["Chan 1", "Chan 8", "Chan 10", "Chan 13"],
    "Occipital": ["Chan 2", "Chan 4", "Chan 5", "Chan 7"]
}

# ==========================================
# 2. Helper Functions & Core Algorithms
# ==========================================
def universal_load(json_path):
    """Load JSON configuration and extract target timestamps."""
    if not os.path.exists(json_path): 
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f: 
            data = json.load(f)
    except: 
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

def find_file_smart(target_path):
    """Attempt to locate the target file in the default directory if absolute path fails."""
    search_dir = r"D:\EEG\处理后的脑电数据"
    if os.path.exists(target_path): 
        return target_path
    filename = os.path.basename(target_path)
    candidate = os.path.join(search_dir, filename)
    return candidate if os.path.exists(candidate) else None

def standardize_channel_names(raw):
    """Standardize channel nomenclature across different recordings."""
    mapping = {}
    for name in raw.ch_names:
        clean_name = name.replace('EEG', '').replace('Ch', '').replace(' ', '').replace('-Ref', '')
        if clean_name.isdigit():
            new_name = f"Chan {clean_name}" 
            mapping[name] = new_name
    if mapping:
        raw.rename_channels(mapping)
    return raw

# --- Core Metric Calculation Logic ---

def calc_lzc(data_array):
    """Calculate Normalized Lempel-Ziv Complexity (Binary Sequence)."""
    threshold = np.median(data_array)
    seq = "".join(['1' if x > threshold else '0' for x in data_array])
    n = len(seq)
    if n == 0: 
        return 0
    c, i, k = 1, 0, 1
    while c + k <= n:
        if c + k > n: 
            break
        w = seq[i : i+k]
        if w in seq[0 : i+k-1]:
            k += 1
        else:
            c += 1
            i += k
            k = 1
    b = n / math.log2(n)
    return c / b

def calc_if_fluctuation(data_array, fs):
    """Calculate Instantaneous Frequency Volatility (Hilbert IF Standard Deviation)."""
    if len(data_array) == 0: 
        return 0
    analytic = hilbert(data_array)
    phase = np.unwrap(np.angle(analytic))
    if len(phase) < 2: 
        return 0
    freq = (np.diff(phase) / (2.0 * np.pi) * fs)
    return np.std(freq)

def calc_gamma_power(data_array):
    """Calculate Gamma Energy (Signal Variance)."""
    return np.var(data_array)

# ==========================================
# 3. Feature Extraction (Including EMD)
# ==========================================
def analyze_eeg_metrics(raw_crop):
    """
    Input: 30s EEG Epoch
    Output: Dictionary of Gamma, LZC, IMF1_IF, IMF2_IF averaged by cortical region.
    """
    results = {}
    fs = raw_crop.info['sfreq']
    
    # 1. Broadband signal (1-45Hz) -> For EMD and LZC
    raw_broad = raw_crop.copy().filter(1.0, 45.0, fir_design='firwin', verbose=False)
    data_broad = raw_broad.get_data() * 1e6  
    
    # 2. Gamma band (30-45Hz) -> For Gamma Power
    raw_gamma = raw_crop.copy().filter(30.0, 45.0, fir_design='firwin', verbose=False)
    data_gamma = raw_gamma.get_data() * 1e6  
    
    ch_metrics = {'Gamma': {}, 'LZC': {}, 'IMF1_IF': {}, 'IMF2_IF': {}}
    emd_engine = EMD()

    for idx, ch_name in enumerate(raw_broad.ch_names):
        ch_metrics['Gamma'][ch_name] = calc_gamma_power(data_gamma[idx])
        ch_metrics['LZC'][ch_name] = calc_lzc(data_broad[idx])
        
        try:
            imfs = emd_engine.emd(data_broad[idx], max_imf=2)
            
            # IMF1
            if imfs.shape[0] >= 1:
                ch_metrics['IMF1_IF'][ch_name] = calc_if_fluctuation(imfs[0], fs)
            else:
                ch_metrics['IMF1_IF'][ch_name] = np.nan
                
            # IMF2
            if imfs.shape[0] >= 2:
                ch_metrics['IMF2_IF'][ch_name] = calc_if_fluctuation(imfs[1], fs)
            else:
                ch_metrics['IMF2_IF'][ch_name] = np.nan
                
        except Exception:
            ch_metrics['IMF1_IF'][ch_name] = np.nan
            ch_metrics['IMF2_IF'][ch_name] = np.nan
        
    # Aggregate and average metrics by cortical region
    for region, target_chs in REGION_MAP.items():
        valid_chs = [ch for ch in target_chs if ch in raw_broad.ch_names]
        
        if valid_chs:
            vals_g = [ch_metrics['Gamma'][ch] for ch in valid_chs]
            results[f"{region}_Gamma"] = np.nanmean(vals_g)
            
            vals_l = [ch_metrics['LZC'][ch] for ch in valid_chs]
            results[f"{region}_LZC"] = np.nanmean(vals_l)
            
            vals_i1 = [ch_metrics['IMF1_IF'][ch] for ch in valid_chs]
            results[f"{region}_IMF1_IF"] = np.nanmean(vals_i1)
            
            vals_i2 = [ch_metrics['IMF2_IF'][ch] for ch in valid_chs]
            results[f"{region}_IMF2_IF"] = np.nanmean(vals_i2)
        else:
            results[f"{region}_Gamma"] = np.nan
            results[f"{region}_LZC"] = np.nan
            results[f"{region}_IMF1_IF"] = np.nan
            results[f"{region}_IMF2_IF"] = np.nan
            
    return results

# ==========================================
# 4. Data Loading and Reconstruction
# ==========================================
def load_and_process_all():
    records = []
    print("--- Starting Data Processing Pipeline ---")
    
    # Load Subject Map for LME Random Effects
    if not os.path.exists(MAP_PATH):
        print(f"Warning: Subject map {MAP_PATH} not found. Using filename as a proxy Dog_ID.")
        file_to_dog_map = {}
    else:
        with open(MAP_PATH, 'r', encoding='utf-8') as f:
            file_to_dog_map = json.load(f)

    for group_name, cfg in FILES_CONFIG.items():
        print(f"-> Processing Group: {group_name}")
        for cond, path_key in [('Pre-Yawn', 'yawn_path'), ('Control', 'ctrl_path')]:
            data_map = universal_load(cfg[path_key])
            count = 0
            
            for r_path, times in data_map.items():
                f_path = find_file_smart(r_path)
                if not f_path: 
                    continue
                
                file_basename = os.path.basename(f_path)
                dog_id = file_to_dog_map.get(file_basename, file_basename)
                
                try:
                    raw = mne.io.read_raw_fif(f_path, preload=True, verbose=False)
                    raw = standardize_channel_names(raw)
                    
                    trial_times = times if isinstance(times, list) else []
                    
                    for t_val in trial_times:
                        t_val = float(t_val)
                        if t_val - WINDOW_SEC < raw.times[0]: 
                            continue
                        
                        raw_crop = raw.copy().crop(t_val - WINDOW_SEC, t_val)
                        res = analyze_eeg_metrics(raw_crop)
                        
                        if any(np.isnan(list(res.values()))): 
                            continue

                        res.update({
                            'Group': group_name, 
                            'Condition': cond,
                            'Dog_ID': dog_id,
                            'Trial_ID': f"{file_basename}_{t_val}"
                        })
                        records.append(res)
                        count += 1
                        
                    raw.close()
                except Exception:
                    continue
            print(f"  [✓] {cond}: {count} valid trials extracted")
            
    return pd.DataFrame(records)

# ==========================================
# 5. Statistical Inference (LME) & Plotting
# ==========================================
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    df_wide = load_and_process_all()
    
    if df_wide.empty:
        print("Data extraction failed. Please check the provided file paths.")
    else:
        save_dir = r"D:\EEG\Figures\补充图"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # ==================================================
        # Global Plotting Style Configuration (Publication Standard)
        # ==================================================
        plt.rcParams.update({
            'pdf.fonttype': 42,         
            'ps.fonttype': 42,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'axes.linewidth': 0.5,     
            'lines.linewidth': 0.5,    
            'xtick.major.width': 0.5,  
            'ytick.major.width': 0.5,  
            'xtick.major.size': 6,     
            'ytick.major.size': 6,     
            'axes.grid': False,        
            'axes.spines.top': False,  
            'axes.spines.right': False,
            'figure.autolayout': True  
        })
        
        palette_colors = {'Pre-Yawn': '#D62728', 'Control': '#808080'}

        clusters = ['Dog Cluster 0', 'Dog Cluster 1']
        indicators = ['Gamma', 'LZC', 'IMF1_IF', 'IMF2_IF']
        
        for cluster in clusters:
            print("\n" + "="*60)
            print(f" Analyzing Data for: {cluster} ")
            print("="*60)
            
            df_cluster = df_wide[df_wide['Group'] == cluster]
            
            if df_cluster.empty:
                print(f"Warning: No data available for {cluster}. Skipping.")
                continue

            for ind in indicators:
                print(f"\n>>> Current Metric: {ind}")
                
                value_vars = [f"{r}_{ind}" for r in REGION_MAP.keys()]
                
                try:
                    df_long = pd.melt(
                        df_cluster, 
                        id_vars=['Trial_ID', 'Condition', 'Dog_ID'], 
                        value_vars=value_vars, 
                        var_name='Region_Raw', 
                        value_name='Metric_Value'
                    )
                except KeyError:
                    print(f"Warning: Missing columns for metric {ind}. Skipping.")
                    continue

                df_long['Region'] = df_long['Region_Raw'].str.replace(f"_{ind}", "")

                # Drop NaNs to ensure LME stability
                df_long = df_long.dropna(subset=['Metric_Value', 'Condition', 'Region', 'Dog_ID'])
                if df_long.empty: 
                    continue

                # --- Linear Mixed-Effects Model (LME) via Likelihood Ratio Test (LRT) ---
                try:
                    # Maximum Likelihood (reml=False) is required for LRT of fixed effects
                    form_full = "Metric_Value ~ C(Condition) * C(Region)"
                    form_red_int = "Metric_Value ~ C(Condition) + C(Region)"
                    form_red_cond = "Metric_Value ~ C(Region)"
                    
                    model_full = smf.mixedlm(form_full, data=df_long, groups=df_long['Dog_ID']).fit(reml=False)
                    model_red_int = smf.mixedlm(form_red_int, data=df_long, groups=df_long['Dog_ID']).fit(reml=False)
                    model_red_cond = smf.mixedlm(form_red_cond, data=df_long, groups=df_long['Dog_ID']).fit(reml=False)
                    
                    # LRT for Interaction Effect (Condition x Region)
                    lr_stat_int = 2 * (model_full.llf - model_red_int.llf)
                    df_int = model_full.df_modelwc - model_red_int.df_modelwc
                    p_interaction = stats.chi2.sf(lr_stat_int, df_int)
                    
                    # LRT for Main Effect of Condition
                    lr_stat_cond = 2 * (model_red_int.llf - model_red_cond.llf)
                    df_cond = model_red_int.df_modelwc - model_red_cond.df_modelwc
                    p_condition = stats.chi2.sf(lr_stat_cond, df_cond)
                    
                    print(f"LME Results - Condition Main Effect P: {p_condition:.4e}")
                    print(f"LME Results - Interaction Effect P:  {p_interaction:.4e}")
                    
                except Exception as e:
                    print(f"LME Computation Error: {e}")
                    p_interaction, p_condition = np.nan, np.nan

                # --- Generate Interaction Plots ---
                fig, ax = plt.subplots(figsize=(6, 4)) 
                
                sns.pointplot(
                    data=df_long, x='Region', y='Metric_Value', hue='Condition', 
                    dodge=True, markers=['o', 's'], capsize=0.1, scale=0.8,
                    palette=palette_colors, 
                    err_kws={'linewidth': 0.5}, 
                    ax=ax
                ) 
                
                plt.setp(ax.lines, linewidth=0.5)
                plt.setp(ax.collections, linewidth=0.5, edgecolor=None)

                cluster_clean = cluster.replace(" ", "_")
                
                # Format the title with LME P-values
                if not np.isnan(p_condition) and not np.isnan(p_interaction):
                    title_str = f'{cluster} - {ind}\nLME Cond p={p_condition:.3f} | Int p={p_interaction:.3f}'
                else:
                    title_str = f'{cluster} - {ind}\nLME Computation Failed'
                    
                ax.set_title(title_str, fontsize=10, pad=15)
                
                ylabels = {
                    'Gamma': 'Gamma Power (μV²)',
                    'LZC': 'Lempel-Ziv Complexity',
                    'IMF1_IF': 'IMF1 IF Fluctuation (Std)',
                    'IMF2_IF': 'IMF2 IF Fluctuation (Std)'
                }
                ax.set_ylabel(ylabels.get(ind, 'Value'), fontsize=9)
                ax.set_xlabel('Cortical Region', fontsize=9)
                
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.legend(frameon=False, fontsize=8, loc='best')

                save_path = os.path.join(save_dir, f"Interaction_LME_{cluster_clean}_{ind}.pdf")
                plt.savefig(save_path, format='pdf', bbox_inches='tight')
                print(f"Chart successfully saved to: {save_path}")
                
                plt.close()