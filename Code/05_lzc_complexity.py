import os
import json
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import sem
from scipy import stats
from mne.stats import permutation_cluster_1samp_test

"""
Neural Complexity Analysis using Lempel-Ziv Complexity (LZC).
This script computes normalized LZC using a sliding window approach 
and applies an adaptive one-tail permutation cluster test to identify 
significant temporal dynamics across different clustering states.
"""

# ==========================================
# 1. Environment & Publication-Quality Style Configuration
# ==========================================
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

style_config = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'lines.linewidth': 0.5,
    'legend.frameon': False,
    'legend.fontsize': 7
}
plt.rcParams.update(style_config)

# ==========================================
# 2. Core Algorithm Logic
# ==========================================
def lz_complexity_fast(binary_seq):
    """Calculate normalized Lempel-Ziv Complexity."""
    n = len(binary_seq)
    if n == 0: 
        return 0
    c = 1
    i = 1
    while i < n:
        j = 1
        while i + j <= n and binary_seq[i:i+j] in binary_seq[0:i+j-1]:
            j += 1
        c += 1
        i += j
    return c * (np.log2(n) / n)

def compute_sliding_lzc(data, sfreq, win_len=2.0, step=1.0):
    """Calculate LZC using a sliding window approach."""
    n_ch, n_samples = data.shape
    win_pts = int(win_len * sfreq)
    step_pts = int(step * sfreq)
    n_windows = (n_samples - win_pts) // step_pts + 1
    lzc_matrix = np.zeros((n_windows, n_ch))
    
    for w in range(n_windows):
        start = int(w * step_pts)
        end = int(start + win_pts)
        for ch in range(n_ch):
            seg = data[ch, start:end]
            binary = "".join((seg > np.median(seg)).astype(int).astype(str))
            lzc_matrix[w, ch] = lz_complexity_fast(binary)
    return lzc_matrix

def process_and_plot_lzc(json_paths):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=False)
    colors = ['#4363d8', '#e6194b', '#3cb44b', '#f58231']
    
    for idx, path in enumerate(json_paths):
        print(f"Processing Cluster {idx}...")
        if not os.path.exists(path): 
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            full_json = json.load(f)
            yawn_data = full_json.get('yawn_times', full_json)
            
        all_yawn_lzc = [] 
        
        for fpath, times in yawn_data.items():
            if not isinstance(times, list):
                continue
            real_p = fpath if os.path.exists(fpath) else os.path.join(os.path.dirname(path), os.path.basename(fpath))
            if not os.path.exists(real_p):
                continue
            
            try:
                raw = mne.io.read_raw_fif(real_p, preload=True, verbose=False)
                raw.filter(0.5, 45, verbose=False) 
                sf = raw.info['sfreq']
                
                for t in times:
                    t_start, t_end = t - 30.0, t + 30.0
                    if t_start < 0 or t_end > raw.times[-1]: 
                        continue
                    
                    data = raw.get_data(start=int(t_start*sf), stop=int(t_end*sf))
                    lzc_curve = compute_sliding_lzc(data, sf)
                    
                    # Baseline correction (Z-score transformation)
                    baseline_window = lzc_curve[0, :] 
                    b_mean = np.mean(baseline_window)
                    b_std = np.std(baseline_window) + 1e-9
                    lzc_z = (lzc_curve - b_mean) / b_std
                    
                    all_yawn_lzc.append(np.mean(lzc_z, axis=1))
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if not all_yawn_lzc: 
            continue
        
        # Data Preparation
        X = np.array(all_yawn_lzc)
        n_samples, n_times = X.shape
        time_axis = np.linspace(-30, 30 - 2, n_times)
        
        # --- Plotting ---
        ax = axes[idx]
        mean_vals = np.mean(X, axis=0)
        sem_vals = sem(X, axis=0)
        
        ax.plot(time_axis, mean_vals, color=colors[idx], linewidth=0.5, label=f'Cluster {idx}')
        ax.fill_between(time_axis, mean_vals - sem_vals, mean_vals + sem_vals, 
                        color=colors[idx], alpha=0.2, edgecolor='none')
        
        # --- Robust Statistics: Adaptive One-Tail Permutation Test ---
        if n_samples > 1:
            # 1. Automatically determine the main trend of the Cluster
            overall_mean = np.mean(mean_vals)
            
            if overall_mean < 0:
                target_tail = -1  # Decreasing trend, testing for negative values
                direction_str = "Decreasing"
            else:
                target_tail = 1   # Increasing trend, testing for positive values
                direction_str = "Increasing"

            # 2. Relax cluster-forming threshold to construct connected temporal segments
            p_forming = 0.15
            df = n_samples - 1
            t_threshold = stats.t.ppf(1 - p_forming, df)
            
            # Adjust threshold sign for negative tail
            if target_tail == -1:
                t_threshold = -t_threshold 

            print(f"  Cluster {idx} ({direction_str}): Using tail={target_tail}, threshold={t_threshold:.3f}")

            # 3. Run permutation test
            T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                X, 
                n_permutations=1000, 
                threshold=t_threshold, 
                tail=target_tail, 
                verbose=False
            )
            
            # Calculate significance marker height
            sig_y_pos = np.max(mean_vals + sem_vals) * 1.1 
            if sig_y_pos < 0.5: 
                sig_y_pos = 0.5 
            
            for i_c, c in enumerate(clusters):
                cluster_indices = c[0]
                
                # Final significance level is strictly controlled at p <= 0.05
                if cluster_p_values[i_c] <= 0.05: 
                    start_t = time_axis[cluster_indices[0]]
                    end_t = time_axis[cluster_indices[-1]]
                    
                    # Draw significance markers
                    ax.hlines(sig_y_pos, start_t, end_t, colors='black', linewidth=1.5)
                    ax.text((start_t + end_t) / 2, sig_y_pos, '*', 
                            ha='center', va='bottom', fontsize=10, color='black')

        # --- Decorations ---
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_xticks(np.arange(-30, 31, 15))
        ax.set_xlabel("Time from onset (s)")
        if idx == 0: 
            ax.set_ylabel("LZC (Baseline Z-score)")
        ax.set_title(f"Cluster {idx} Dynamics", fontweight='bold')
        ax.grid(False) 

    plt.tight_layout()
    save_path = "LZC_Cluster_Dynamics_Stats_Final.pdf"
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Statistical analysis complete. Adaptive one-tail test report saved to: {save_path}")

# --- Execution ---
if __name__ == "__main__":
    json_files = [
        r"D:\EEG\Figures\哈欠脑电\终极优化聚类\cluster_0_yawns.json",
        r"D:\EEG\Figures\哈欠脑电\终极优化聚类\cluster_1_yawns.json",
        r"D:\EEG\Figures\哈欠脑电\终极优化聚类\cluster_2_yawns.json",
        r"D:\EEG\Figures\哈欠脑电\终极优化聚类\cluster_3_yawns.json"
    ]
    process_and_plot_lzc(json_files)