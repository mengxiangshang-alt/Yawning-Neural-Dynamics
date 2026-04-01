import os
import json
import gc
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import mne
from scipy.signal import welch, hilbert
from scipy.spatial.distance import cdist
from PyEMD import EMD
from sklearn.preprocessing import StandardScaler

"""
Micro-scale Dynamics via EMD and HHT.
This script applies Empirical Mode Decomposition (EMD) and Hilbert-Huang Transform (HHT) 
to analyze the instantaneous frequency (IF) and instantaneous amplitude (IA) volatility 
during peri-yawning states versus matched baseline periods.
"""

# ==========================================
# 0. Environment & Plot Style Configuration
# ==========================================
warnings.filterwarnings("ignore")

# Global font and linewidth configurations
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,             
    'pdf.fonttype': 42,          
    'ps.fonttype': 42,
    'axes.linewidth': 0.5,       
    'lines.linewidth': 0.5,      
    'xtick.major.width': 0.5,    
    'ytick.major.width': 0.5,
    'xtick.major.size': 6,       
    'ytick.major.size': 6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.top': False,    
    'axes.spines.right': False   
})

# ==========================================
# [⭐ Parameter Settings ⭐]
# ==========================================
CLUSTER_ID = 1
CONFIG = {
    'T_PRE': 30.0,           
    'T_GAP': 0,              
    'SEARCH_STEP': 10.0,     
    'BUFFER_ZONE': 90.0,     
}
BASE_DIR = r"D:\EEG\处理后的脑电数据"

# Modify save directory for distinction
BASE_SAVE_DIR = os.path.join(r"D:\EEG\Figures\机理验证_CombinedStyle", f"Cluster_{CLUSTER_ID}_MostStableYawn") 
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# ==========================================
# 1. Core Calculation Functions
# ==========================================
def get_5d_fingerprint(data_1d, sf):
    """Extract 5D Power Spectral Density (PSD) fingerprint for structural matching."""
    try:
        f, psd = welch(data_1d, sf, nperseg=int(sf * 2))
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
        p = [np.sum(psd[(f >= l) & (f < r)]) for l, r in bands]
        return np.log10(np.array(p) + 1e-12)
    except: 
        return np.zeros(5)

def get_imf2_metrics(sig, sf):
    """Extract IMF2, calculate Instantaneous Frequency (IF) and Amplitude (IA), 
    and return aligned time axes."""
    emd = EMD()
    sig_norm = (sig - np.mean(sig)) / np.std(sig)
    imfs = emd.emd(sig_norm, max_imf=3)
    
    if imfs.shape[0] < 2: 
        target_imf = imfs[0]
    else: 
        target_imf = imfs[1] 

    analytic = hilbert(target_imf)
    ia = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2.0 * np.pi) * sf
    
    # Time axis alignment
    t_axis = np.arange(len(inst_freq)) / sf
    
    mask = (inst_freq > 0.5) & (inst_freq < 30)
    
    return target_imf, t_axis[mask], inst_freq[mask], ia[1:][mask]

# ==========================================
# 2. Publication-Quality Plotting Functions
# ==========================================
def plot_scientific_hht_combined(pos_data, neg_data, sf, save_path):
    # --- Calculate Data ---
    p_imf, p_t, p_if, p_ia = get_imf2_metrics(pos_data, sf) 
    n_imf, n_t, n_if, n_ia = get_imf2_metrics(neg_data, sf) 
    
    p_std, n_std = np.std(p_if), np.std(n_if)

    # --- Unify Colorbar Range ---
    global_vmax = np.percentile(np.concatenate([p_ia, n_ia]), 99)
    norm = plt.Normalize(vmin=0, vmax=global_vmax)
    cmap_style = 'jet' 

    # --- Initialize layout ---
    fig = plt.figure(figsize=(12, 2), dpi=300) 
    
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 2], width_ratios=[1, 1, 0.05], 
                          hspace=0.1, wspace=0.15)

    col_yawn = '#D62728' 
    col_ctrl = '#1F77B4' 

    # ================= LEFT COLUMN: PRE-YAWN =================
    ax_y_raw = fig.add_subplot(gs[0, 0])
    t = np.arange(len(pos_data)) / sf
    ax_y_raw.plot(t, p_imf, color=col_yawn, linewidth=0.5)
    
    ax_y_raw.axis('off') 
    ax_y_raw.set_xlim(0, 30)
    ax_y_raw.set_title(f"Pre-Yawn Group (IMF2)\nStable Rhythm ($\sigma$={p_std:.2f})", 
                       loc='center', fontsize=11, fontweight='bold', color='black', pad=5)

    ax_y_spec = fig.add_subplot(gs[1, 0])
    sc_y = ax_y_spec.scatter(p_t, p_if, c=p_ia, cmap=cmap_style, norm=norm, 
                             s=1.5, alpha=0.9, edgecolors='none')
    
    ax_y_spec.set_ylim(0, 20)
    ax_y_spec.set_xlim(0, 30)
    ax_y_spec.set_yticks([0, 10, 20])
    ax_y_spec.set_xticks([0, 15, 30])
    ax_y_spec.set_xticklabels(['-30', '-15', '0'])
    ax_y_spec.set_ylabel('Frequency (Hz)', fontsize=10)
    ax_y_spec.set_xlabel('Time (s)', fontsize=10)
    
    for spine in ax_y_spec.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5) 

    # ================= RIGHT COLUMN: NON-YAWN =================
    ax_c_raw = fig.add_subplot(gs[0, 1])
    ax_c_raw.plot(t, n_imf, color=col_ctrl, linewidth=0.5)
    
    ax_c_raw.axis('off')
    ax_c_raw.set_xlim(0, 30)
    # Title emphasizes Matched control
    ax_c_raw.set_title(f"Matched Non-Yawn (IMF2)\nControl ($\sigma$={n_std:.2f})", 
                       loc='center', fontsize=11, fontweight='bold', color='black', pad=5)

    ax_c_spec = fig.add_subplot(gs[1, 1])
    sc_c = ax_c_spec.scatter(n_t, n_if, c=n_ia, cmap=cmap_style, norm=norm, 
                             s=1.5, alpha=0.9, edgecolors='none')
    
    ax_c_spec.set_ylim(0, 20)
    ax_c_spec.set_xlim(0, 30)
    ax_c_spec.set_yticks([0, 10, 20])
    ax_c_spec.set_xticks([0, 15, 30])
    ax_c_spec.set_xticklabels(['-30', '-15', '0'])
    ax_c_spec.set_xlabel('Time (s)', fontsize=10)
    ax_c_spec.set_yticklabels([]) 
    
    for spine in ax_c_spec.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    # ================= COLORBAR =================
    cax = fig.add_subplot(gs[1, 2])
    cbar = plt.colorbar(sc_y, cax=cax)
    cbar.set_label('Inst. Amplitude (μV)', fontsize=9)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(width=0.5, length=3)

    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"Publication-quality chart saved to: {save_path}")

def plot_if_distributions_like_fig5_v3(yawn_data, non_yawn_data, sf, save_path=None):
    # --- EMD Decomposition ---
    emd = EMD()
    y_norm = (yawn_data - np.mean(yawn_data)) / np.std(yawn_data)
    c_norm = (non_yawn_data - np.mean(non_yawn_data)) / np.std(non_yawn_data)
    
    imfs_yawn = emd.emd(y_norm)[:3]
    imfs_ctrl = emd.emd(c_norm)[:3]

    # Zero-pad to ensure 3 IMFs
    def ensure_3(imfs):
        if len(imfs) < 3:
            pad = np.zeros((3 - len(imfs), imfs.shape[1]))
            return np.vstack([imfs, pad])
        return imfs

    imfs_yawn = ensure_3(imfs_yawn)
    imfs_ctrl = ensure_3(imfs_ctrl)

    # --- Calculate IF ---
    def compute_if(imf):
        analytic = hilbert(imf)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) / (2.0 * np.pi) * sf
        return inst_freq[(inst_freq > 0.1) & (inst_freq < 50)]

    if_yawn_list = [compute_if(imf) for imf in imfs_yawn]
    if_ctrl_list = [compute_if(imf) for imf in imfs_ctrl]

    # --- Color configuration for IMF1/2/3 ---
    imf_colors = ['#E74C3C', '#3498DB', '#2ECC71']  
    labels = ['Pre-Yawn (Most Stable)', 'Matched Non-Yawn']
    data_lists = [if_yawn_list, if_ctrl_list]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300, sharex='all', sharey='all')

    for col in range(2):
        ax = axes[col]
        data_list = data_lists[col]
        label = labels[col]
        freq_ranges = []

        for i in range(3):
            if_data = data_list[i]
            color = imf_colors[i]
            if len(if_data) == 0:
                freq_ranges.append((0, 0))
                continue

            ax.hist(if_data, bins=30, density=True, alpha=0.6, color=color, edgecolor='none', label=f'IMF{i+1}')
            mean_if = np.mean(if_data)
            std_if = np.std(if_data)
            lower, upper = mean_if - std_if, mean_if + std_if
            freq_ranges.append((lower, upper))

            ax.axvline(mean_if, color=color, linestyle='--', linewidth=1.2)
            ax.axvline(lower, color=color, linestyle='-', linewidth=1.0)
            ax.axvline(upper, color=color, linestyle='-', linewidth=1.0)

        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Instantaneous Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_xlim(0, 45)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.legend(loc='upper right', fontsize=8)

        trans = ax.get_xaxis_transform()
        for i, (low, high) in enumerate(freq_ranges):
            if low == 0 and high == 0: 
                continue
            mid = (low + high) / 2
            color = imf_colors[i]
            ax.text(mid, -0.12 - 0.04 * i, f'[{low:.1f}, {high:.1f}]', 
                    transform=trans, color=color, fontsize=7,
                    ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='white', edgecolor='none', alpha=0.7))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Enhanced IF distribution plot saved to: {save_path}")
    plt.show()

# ==========================================
# 3. Main Logic: Two-Step Screening
# ==========================================
if __name__ == "__main__":
    print(f"Mission: 1. Find the most stable yawn  2. Find the nearest control via Euclidean distance")
    
    j_path = os.path.join(BASE_DIR, f"cluster_{CLUSTER_ID}_yawns.json")
    try:
        with open(j_path, 'r', encoding='utf-8') as f: 
            cfg = json.load(f)
        yawn_data = cfg.get('yawn_times', cfg)
    except:
        yawn_data = {}
        print("Failed to read JSON configuration file")

    # --- Step 1: Full scan to find the yawn sample with the lowest IF standard deviation ---
    print("\n[Step 1/2] Searching for the most stable yawn sample...")
    
    lowest_std = float('inf')
    target_yawn_info = None  # Store metadata for the best yawn: {data, fp, fpath, time, sf}
    
    total_files = len(yawn_data)
    for f_idx, (fpath, times) in enumerate(yawn_data.items()):
        real_p = fpath if os.path.exists(fpath) else os.path.join(BASE_DIR, os.path.basename(fpath))
        if not os.path.exists(real_p): 
            continue
        
        try:
            raw = mne.io.read_raw_fif(real_p, preload=True, verbose=False)
            raw.filter(0.5, 45., verbose=False)
            sf = raw.info['sfreq']
            
            for t in times:
                t_s = t - CONFIG['T_GAP'] - CONFIG['T_PRE']
                if t_s < 0: 
                    continue
                
                # Extract yawn data segment
                y_seg = raw.get_data(picks=[0], start=int(t_s*sf), stop=int((t-CONFIG['T_GAP'])*sf))[0]
                
                # Calculate IF volatility
                _, _, y_if, _ = get_imf2_metrics(y_seg, sf)
                current_std = np.std(y_if)
                
                # Record the minimum value
                if current_std < lowest_std:
                    lowest_std = current_std
                    # Record fingerprint for subsequent matching
                    fp = get_5d_fingerprint(y_seg, sf)
                    target_yawn_info = {
                        'data': y_seg.copy(),
                        'fp': fp,
                        'fpath': real_p,
                        'time': t,
                        'sf': sf,
                        'std': current_std
                    }
                    print(f"  [Update] Found lower std: {current_std:.4f} in {os.path.basename(real_p)}")
            
            del raw
            gc.collect()
            
        except Exception as e:
            print(f"Step 1 Error in {os.path.basename(real_p)}: {e}")
            continue

    if target_yawn_info is None:
        print("Step 1 Failed: No valid yawn samples found.")
        exit()

    print(f"\nStep 1 Complete! Most stable yawn found in: {os.path.basename(target_yawn_info['fpath'])} (Std: {target_yawn_info['std']:.4f})")

    # --- Step 2: Find the nearest control sample by Euclidean distance in the matched file ---
    print("\n[Step 2/2] Searching for a fingerprint-matched control sample...")
    
    best_control_data = None
    min_dist = float('inf')
    
    try:
        # Reload target file
        raw = mne.io.read_raw_fif(target_yawn_info['fpath'], preload=True, verbose=False)
        raw.filter(0.5, 45., verbose=False)
        sf = raw.info['sfreq']
        
        # Determine search range (avoid yawn periods)
        current_file_times = []
        for k, v in yawn_data.items():
            if os.path.basename(k) == os.path.basename(target_yawn_info['fpath']):
                current_file_times = v
                break
                
        forbidden = [(t - CONFIG['BUFFER_ZONE'], t + CONFIG['BUFFER_ZONE']) for t in current_file_times]
        search_times = np.arange(0, raw.times[-1] - CONFIG['T_PRE'], CONFIG['SEARCH_STEP'])
        
        # Target fingerprint
        target_fp = target_yawn_info['fp']
        
        # Build feature matrix
        neg_data_pool = []
        for st in search_times:
            if any(not (st + CONFIG['T_PRE'] < s or st > e) for s, e in forbidden): 
                continue
            d = raw.get_data(picks=[0], start=int(st*sf), stop=int((st+CONFIG['T_PRE'])*sf))[0]
            fp = get_5d_fingerprint(d, sf)
            neg_data_pool.append({'data': d, 'fp': fp, 'time': st})
        
        if neg_data_pool:
            fps = np.array([x['fp'] for x in neg_data_pool])
            scaler = StandardScaler()
            scaler.fit(fps) # Fit on candidates
            
            candidates_scaled = scaler.transform(fps)
            target_scaled = scaler.transform([target_fp])
            
            # Calculate distance
            dists = cdist(target_scaled, candidates_scaled, metric='euclidean')[0]
            best_idx = np.argmin(dists)
            
            min_dist = dists[best_idx]
            best_control_data = neg_data_pool[best_idx]['data']
            print(f"Match Found! Min Distance: {min_dist:.4f} at t={neg_data_pool[best_idx]['time']:.1f}s")
        else:
            print("Step 2 Failed: Insufficient control data segments in the file.")

        del raw
        gc.collect()

    except Exception as e:
        print(f"Step 2 Error: {e}")

    # --- Generate final plots ---
    if best_control_data is not None:
        print("\nGenerating final plots...")
        plot_scientific_hht_combined(
            target_yawn_info['data'], 
            best_control_data, 
            target_yawn_info['sf'], 
            os.path.join(BASE_SAVE_DIR, "Figure_HHT_Mechanism_StableMatch.pdf")
        )
        
        plot_if_distributions_like_fig5_v3(
            target_yawn_info['data'],
            best_control_data,
            target_yawn_info['sf'],
            os.path.join(BASE_SAVE_DIR, "Figure_IF_Distribution_StableMatch.pdf")
        )
    else:
        print("Task Failed: Could not generate valid sample pairs.")