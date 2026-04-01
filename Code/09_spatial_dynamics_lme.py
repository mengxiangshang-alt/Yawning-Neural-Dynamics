import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.multitest as smt
import statsmodels.formula.api as smf
import mne

"""
Spatial Neural Dynamics and Interaction Effect Analysis.
This script evaluates regional EEG spectral power changes using a 
Linear Mixed-Effects Model (LME). The model incorporates subject-level 
variance (Dog_ID) as a random effect to rigorously control for pseudo-replication 
and evaluates the Cluster x Region interaction using a Likelihood Ratio Test (LRT).
"""

# ==========================================
# 1. Configuration & Initialization
# ==========================================
warnings.filterwarnings('ignore')

output_dir = r'D:\EEG\Figures\哈欠脑电\终极优化聚类'
json_files = [
    os.path.join(output_dir, f"cluster_{i}_yawns.json") for i in range(4)
]
result_dir = os.path.join(output_dir, "feature_comparison_PDF_Aggregated_Interaction") 
os.makedirs(result_dir, exist_ok=True)

# Load external mapping dictionary for Dog_ID
MAP_PATH = r"D:\EEG\Figures\哈欠脑电\FILE_TO_DOG_MAP.json"
if not os.path.exists(MAP_PATH):
    print(f"Warning: Mapping file {MAP_PATH} not found. Using filename as a proxy Dog_ID.")
    FILE_TO_DOG_MAP = {}
else:
    with open(MAP_PATH, 'r', encoding='utf-8') as f:
        FILE_TO_DOG_MAP = json.load(f)

# Brain Region Mapping
region_to_channels = {
    "Frontal": ["Chan 9", "Chan 11", "Chan 12", "Chan 14"],
    "Parietal": ["Chan 3", "Chan 6", "Chan 15", "Chan 16"],
    "Temporal": ["Chan 1", "Chan 8", "Chan 10", "Chan 13"],
    "Occipital": ["Chan 2", "Chan 4", "Chan 5", "Chan 7"]
}
channel_to_region = {}
for region, ch_list in region_to_channels.items():
    for ch in ch_list:
        channel_to_region[ch] = region

UMAP_COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44']

# Plot Settings (Publication Standard)
plt.rcParams.update({
    "font.family": "sans-serif", 
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 12, 
    "axes.linewidth": 0.5,
    "lines.linewidth": 0.5, 
    "pdf.fonttype": 42, 
    "ps.fonttype": 42,
    "mathtext.fontset": "custom", 
    "mathtext.rm": "Arial", 
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold"
})

# ==========================================
# 2. Data Loading & Processing
# ==========================================
records = []
print("Scanning JSON files for event metadata...")

for cluster_idx, json_path in enumerate(json_files):
    if not os.path.exists(json_path):
        print(f"Warning: File not found {json_path}")
        continue
    
    print(f"Processing Cluster {cluster_idx}: {os.path.basename(json_path)}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for fif_path, t_yawns in data['yawn_times'].items():
        if not os.path.exists(fif_path): 
            continue

        # Retrieve Dog_ID corresponding to the current file
        file_basename = os.path.basename(fif_path)
        dog_id = FILE_TO_DOG_MAP.get(file_basename, file_basename)
        
        try:
            raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
            sfreq = raw.info['sfreq']
            raw_ch_names = raw.ch_names
        except Exception as e:
            print(f"Error reading {file_basename}: {e}")
            continue

        for i, t_center in enumerate(t_yawns):
            event_id = f"{file_basename}_yawn{i}"
            
            # Define peri-event time windows
            t_start_pre, t_end_pre = t_center - 30, t_center
            t_start_post, t_end_post = t_center, t_center + 30
            
            if t_start_pre < 0 or t_end_post > raw.times[-1]: 
                continue
            
            n_fft = int(sfreq * 2) if int(sfreq * 2) < (30 * sfreq) else int(30 * sfreq)
            
            try:
                psd_pre = raw.compute_psd(method='welch', tmin=t_start_pre, tmax=t_end_pre, 
                                          fmin=1, fmax=40, n_fft=n_fft, verbose=False)
                psd_post = raw.compute_psd(method='welch', tmin=t_start_post, tmax=t_end_post, 
                                           fmin=1, fmax=40, n_fft=n_fft, verbose=False)
                
                data_pre = psd_pre.get_data()
                data_post = psd_post.get_data()
                freqs = psd_pre.freqs
                
                delta_mask = (freqs >= 1) & (freqs <= 4)
                gamma_mask = (freqs >= 30) & (freqs <= 40)
                
                for ch_name, region in channel_to_region.items():
                    if ch_name in raw_ch_names:
                        ch_idx = raw_ch_names.index(ch_name)
                    else: 
                        continue 

                    # Extract Mean Power for target frequency bands
                    val_d_pre = np.mean(data_pre[ch_idx, delta_mask])
                    val_d_post = np.mean(data_post[ch_idx, delta_mask])
                    val_g_pre = np.mean(data_pre[ch_idx, gamma_mask])
                    val_g_post = np.mean(data_post[ch_idx, gamma_mask])
                    
                    # Log-transform to Decibels (dB)
                    db_d_pre = 10 * np.log10(val_d_pre)
                    db_d_post = 10 * np.log10(val_d_post)
                    db_g_pre = 10 * np.log10(val_g_pre)
                    db_g_post = 10 * np.log10(val_g_post)
                    
                    records.append({
                        'event_id': event_id,
                        'Dog_ID': dog_id,
                        'region': region,
                        'cluster': cluster_idx,
                        'delta_change': db_d_post - db_d_pre,
                        'gamma_change': db_g_post - db_g_pre,
                        'global_delta': (db_d_pre + db_d_post) / 2,
                        'global_gamma': (db_g_pre + db_g_post) / 2
                    })
            except Exception: 
                continue

df_long = pd.DataFrame(records)
print(f"Raw Channel Data constructed: {len(df_long)} rows")

# Aggregate data across channels within each region to prevent variance deflation
# Retain Dog_ID as it is required as a random effect grouping variable in LME
df_agg = df_long.groupby(['cluster', 'region', 'event_id', 'Dog_ID'])[
    ['delta_change', 'gamma_change', 'global_delta', 'global_gamma']
].mean().reset_index()
print(f"Aggregated Data constructed: {len(df_agg)} rows (Ready for LME analysis)")

# ==========================================
# 3. Plotting & Statistical Functions
# ==========================================
def plot_regional_beanplot(data, x, y, ylabel, filename, show_legend=False):
    """Plot channel-level data distributions across cortical regions."""
    fig, ax = plt.subplots(figsize=(6, 5)) 
    order = ["Frontal", "Parietal", "Temporal", "Occipital"]

    sns.violinplot(
        data=data, x=x, y=y, hue='cluster', order=order,
        palette=UMAP_COLORS, inner=None, linewidth=0, alpha=0.4, ax=ax, saturation=0.8,
        width=0.7, zorder=1
    )
    sns.stripplot(
        data=data, x=x, y=y, hue='cluster', order=order,
        palette=UMAP_COLORS, dodge=True, size=2.5, alpha=0.5, edgecolor='none', ax=ax, 
        jitter=0.2, zorder=2
    )
    
    if 'change' in y:
        ax.axhline(0, color='#444444', linestyle=':', linewidth=0.5, alpha=0.8, zorder=0)

    for collection in ax.collections: 
        collection.set_rasterized(True)
    sns.despine(ax=ax, offset=5, trim=False)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=8)
    ax.set_xlabel("", fontsize=0)
    ax.tick_params(axis='both', labelsize=12)
    
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:4], [f'Cluster {i}' for i in range(4)], 
                  frameon=False, fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    elif ax.legend_: 
        ax.legend_.remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename), bbox_inches='tight', dpi=600, format='pdf')
    plt.close()

def plot_whole_brain_beanplot(data, y, ylabel, filename):
    """Plot whole-brain averages with Kruskal-Wallis & FDR-corrected MWU tests."""
    fig, ax = plt.subplots(figsize=(4, 6)) 
    order = [0, 1, 2, 3]
    
    sns.violinplot(
        data=data, x='cluster', y=y, order=order,
        palette=UMAP_COLORS, inner=None, linewidth=0, alpha=0.4, ax=ax, saturation=0.8,
        width=0.4, zorder=1
    )
    sns.stripplot(
        data=data, x='cluster', y=y, order=order,
        palette=UMAP_COLORS, size=3, alpha=0.5, edgecolor='none', ax=ax, 
        jitter=0.15, zorder=2
    )
    
    if 'change' in y:
        ax.axhline(0, color='#444444', linestyle=':', linewidth=0.5, alpha=0.8, zorder=0)

    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    groups = [data[data['cluster'] == c][y].values for c in order]
    
    # Global Non-parametric test
    try:
        _, p_global = stats.kruskal(*[g for g in groups if len(g) > 0])
    except ValueError:
        p_global = 1.0

    y_max, y_min = data[y].max(), data[y].min()
    y_range = y_max - y_min
    current_y = y_max + y_range * 0.05 

    # Perform pairwise comparisons if global test is significant
    if p_global < 0.05:
        p_values = []
        valid_pairs = []
        for i, j in pairs:
            if len(groups[i]) > 0 and len(groups[j]) > 0:
                _, p = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                p_values.append(p)
            else:
                p_values.append(1.0)
            valid_pairs.append((i, j))
        
        # False Discovery Rate (FDR) correction
        reject, p_corrected, _, _ = smt.multipletests(p_values, alpha=0.05, method='fdr_bh')
        
        for idx, (i, j) in enumerate(valid_pairs):
            if reject[idx]: 
                p_val = p_corrected[idx]
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                x1, x2 = i, j
                bar_h = y_range * 0.02 
                
                # Draw significance bracket
                ax.plot([x1, x1, x2, x2], [current_y - bar_h, current_y, current_y, current_y - bar_h], 
                        lw=0.8, c='#333333', zorder=100)
                ax.text((x1 + x2) / 2, current_y, sig, ha='center', va='bottom', fontsize=12, zorder=101)
                current_y += y_range * 0.09

    for collection in ax.collections: 
        collection.set_rasterized(True)
    sns.despine(ax=ax, offset=5, trim=False)
    ax.set_ylim(y_min - y_range * 0.1, current_y + y_range * 0.05)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=8)
    ax.set_xlabel("", fontsize=0)
    ax.set_xticks(order)
    ax.set_xticklabels([f"Cluster {i}" for i in order], fontsize=11)
    ax.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename), bbox_inches='tight', dpi=600, format='pdf')
    plt.close()

def plot_interaction_effect(data, y, ylabel, filename):
    """
    Plot interaction effects (Region x Cluster).
    Utilizes Linear Mixed-Effects Model (LME) with Dog_ID as a random intercept 
    to rigorously control for individual baseline differences and pseudo-replication.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    order = ["Frontal", "Parietal", "Temporal", "Occipital"]
    
    # --- 1. Calculate LME on Aggregated Data ---
    try:
        # Maximum Likelihood (reml=False) must be used to compare nested models with different fixed effects
        form_full = f"{y} ~ C(cluster) * C(region)"
        model_full = smf.mixedlm(form_full, data=data, groups=data['Dog_ID']).fit(reml=False)
        
        form_red = f"{y} ~ C(cluster) + C(region)"
        model_red = smf.mixedlm(form_red, data=data, groups=data['Dog_ID']).fit(reml=False)
        
        # Likelihood Ratio Test (LRT) to evaluate the significance of the interaction term
        lr_stat = 2 * (model_full.llf - model_red.llf)
        df_diff = model_full.df_modelwc - model_red.df_modelwc
        p_val = stats.chi2.sf(lr_stat, df_diff)
        
        if p_val < 0.001:
            p_text = r"LME Interaction: $P < 0.001$"
        else:
            p_text = f"LME Interaction: $P = {p_val:.3f}$"
            
    except Exception as e:
        print(f"LME Stats Error for {y}: {e}")
        p_text = "LME Error"

    # --- 2. Plotting ---
    sns.pointplot(
        data=data, x='region', y=y, hue='cluster', order=order,
        palette=UMAP_COLORS, dodge=True, markers='o', scale=0.8,
        errorbar=('ci', 95), capsize=0.1, err_kws={'linewidth': 0.5},
        ax=ax
    )
    
    if 'change' in y:
        ax.axhline(0, color='#444444', linestyle=':', linewidth=0.5, alpha=0.8, zorder=0)

    # --- 3. Add LME P-value Annotation ---
    ax.text(0.03, 0.97, p_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#cccccc'),
            zorder=200)

    sns.despine(ax=ax, trim=False)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=8)
    ax.set_xlabel("Brain Region", fontsize=12, labelpad=8)
    ax.tick_params(axis='both', labelsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [f'Cluster {i}' for i in range(4)], 
              title="Cluster", frameon=False, fontsize=10, 
              loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename), bbox_inches='tight', dpi=600, format='pdf')
    plt.close()
    print(f"Saved Interaction PDF (LME Statistics): {filename}")

# ==========================================
# 4. Execute Pipeline
# ==========================================
if __name__ == "__main__":
    print("\nStarting Regional Plots (High resolution channel data)...")
    plot_regional_beanplot(df_long, 'region', 'delta_change', 'Regional δ Power Change (dB)', 'Regional_Delta_Change.pdf', show_legend=False)
    plot_regional_beanplot(df_long, 'region', 'gamma_change', 'Regional γ Power Change (dB)', 'Regional_Gamma_Change.pdf', show_legend=False)
    plot_regional_beanplot(df_long, 'region', 'global_delta', 'Regional Global δ Power (dB)', 'Regional_Global_Delta.pdf', show_legend=False)
    plot_regional_beanplot(df_long, 'region', 'global_gamma', 'Regional Global γ Power (dB)', 'Regional_Global_Gamma.pdf', show_legend=True)

    print("\nStarting Interaction Plots (Aggregated data with LME Random Effects)...")
    plot_interaction_effect(df_agg, 'delta_change', 'Interaction: δ Power Change (dB)', 'Interaction_Delta_Change.pdf')
    plot_interaction_effect(df_agg, 'gamma_change', 'Interaction: γ Power Change (dB)', 'Interaction_Gamma_Change.pdf')
    plot_interaction_effect(df_agg, 'global_delta', 'Interaction: Global δ Power (dB)', 'Interaction_Global_Delta.pdf')
    plot_interaction_effect(df_agg, 'global_gamma', 'Interaction: Global γ Power (dB)', 'Interaction_Global_Gamma.pdf')

    print("\nStarting Whole Brain Analysis (Averaged by Event & Subject)...")
    df_whole = df_long.groupby(['event_id', 'cluster', 'Dog_ID'])[
        ['delta_change', 'gamma_change', 'global_delta', 'global_gamma']
    ].mean().reset_index()

    plot_whole_brain_beanplot(df_whole, 'delta_change', 'Whole Brain δ Change (dB)', 'WholeBrain_Delta_Change.pdf')
    plot_whole_brain_beanplot(df_whole, 'gamma_change', 'Whole Brain γ Change (dB)', 'WholeBrain_Gamma_Change.pdf')
    plot_whole_brain_beanplot(df_whole, 'global_delta', 'Whole Brain Global δ (dB)', 'WholeBrain_Global_Delta.pdf')
    plot_whole_brain_beanplot(df_whole, 'global_gamma', 'Whole Brain Global γ (dB)', 'WholeBrain_Global_Gamma.pdf')

    print(f"\nAll Analysis Complete. PDF results saved in: {result_dir}")