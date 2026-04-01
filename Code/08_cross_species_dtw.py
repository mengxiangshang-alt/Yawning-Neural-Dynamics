import os
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import mne
from scipy.signal import detrend, hilbert
from scipy.stats import sem, linregress, mannwhitneyu
from scipy.ndimage import gaussian_filter1d

"""
Cross-Species / Cross-State Comparative Analysis via Dynamic Time Warping (DTW).
This script evaluates the universality of yawning-related neural dynamics by comparing 
EEG spectral power and neural complexity metrics between continuous traces. 
Temporal sequences are aligned using DTW to assess morphological similarities.
"""

# ==========================================
# 0. Global Configuration & Paths 
# ==========================================
# Please modify these paths according to your local directory structure
PATH_YAWN = r"D:\EEG\处理后的脑电数据\cluster_2_yawns.json"
PATH_CONTROL = r"D:\EEG\处理后的脑电数据\control_induction_times.json"
SAVE_DIR = r"D:\EEG\Figures\综合对比分析"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

CONFIG = {
    'T_WINDOW': 30.0,       # Analysis window length (seconds)
    'WIN_SIZE': 2.0,        # Sliding window size (seconds)
    'STEP': 1.0,            # Sliding step (seconds)
    'L_FREQ': 0.5,          # Low frequency cutoff
    'H_FREQ': 45.0,         # High frequency cutoff
    'BASELINE_START': -30,  # Baseline start for Z-scoring
    'BASELINE_END': -28     # Baseline end for Z-scoring
}

# Plotting Color Standards
COLOR_YAWN = '#1f77b4'   # Classic Blue (Pre-Yawn / Experimental)
COLOR_CTRL = '#999999'   # Gray (Matched Control)

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# DTW Dependency Check
try:
    from fastdtw import fastdtw
    HAS_DTW = True
except ImportError:
    HAS_DTW = False
    print("Warning: 'fastdtw' library not detected. Morphological analysis plots cannot be generated. (Run 'pip install fastdtw')")

# ==========================================
# Top-level Plotting Style Settings (Publication Quality)
# Requirement: Editable PDF, 0.5pt lines, 6pt ticks
# ==========================================
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({
    'font.family': 'sans-serif', 
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 0.5,         # Axis spine width 0.5pt
    'lines.linewidth': 0.5,        # Global line width 0.5pt
    'axes.spines.top': False, 
    'axes.spines.right': False,
    'xtick.major.width': 0.5,      # Tick width 0.5pt
    'ytick.major.width': 0.5,
    'xtick.major.size': 6.0,       # Tick length 6pt
    'ytick.major.size': 6.0
})

# ==========================================
# 1. Core Algorithms & Metric Calculation 
# ==========================================
def lz_complexity_fast(binary_seq):
    """Calculate Normalized Lempel-Ziv Complexity (LZC)."""
    n = len(binary_seq)
    if n == 0: 
        return 0
    c, i = 1, 1
    while i < n:
        j = 1
        while i + j <= n and binary_seq[i:i+j] in binary_seq[0:i+j-1]: 
            j += 1
        c += 1
        i += j
    return c * (np.log2(n) / n) if n > 1 else 0

def calculate_metrics_trajectory(raw, window_sec, step_sec):
    """Calculate all core metrics simultaneously: Gamma Energy and LZC."""
    sfreq = raw.info['sfreq']
    n_win = int(window_sec * sfreq)
    n_step = int(step_sec * sfreq)
    
    # Broadband signal for LZC
    data_broad = raw.copy().filter(CONFIG['L_FREQ'], CONFIG['H_FREQ'], verbose=False).get_data()
    
    # Gamma band (30-45Hz) for Energy/Effort mapping
    data_gamma = raw.copy().filter(30, 45, fir_design='firwin', verbose=False).get_data()
    
    n_steps = int((data_broad.shape[1] - n_win) / n_step)
    times = []
    res = {'lzc': [], 'gamma': []}

    for i in range(n_steps):
        start, end = i * n_step, (i * n_step) + n_win
        times.append((start + n_win / 2) / sfreq)
        
        seg_broad = data_broad[:, start:end]
        seg_gamma = data_gamma[:, start:end]

        # Calculate LZC across channels
        ch_lzc = []
        for ch_data in seg_broad:
            binary = "".join((ch_data > np.median(ch_data)).astype(int).astype(str))
            ch_lzc.append(lz_complexity_fast(binary))
        res['lzc'].append(ch_lzc)
        
        # Calculate Gamma Energy (Hilbert Envelope Mean)
        gamma_env = np.abs(hilbert(detrend(seg_gamma, axis=1), axis=1))
        res['gamma'].append(np.mean(gamma_env))

    return np.array(times), res

def process_group_data(json_path):
    """Read raw files, extract epochs, align to events, and apply baseline normalization."""
    if not os.path.exists(json_path): 
        return None
    with open(json_path, 'r', encoding='utf-8') as f: 
        data = json.load(f)
    
    target_dict = data.get('control_times') or data.get('yawn_times', {})
    aligned = {k: [] for k in ['lzc', 'gamma']} 
    target_ts = np.arange(-CONFIG['T_WINDOW'], 0 + CONFIG['STEP'], CONFIG['STEP'])
    
    for fpath, times in target_dict.items():
        if not times or not os.path.exists(fpath): 
            continue
        t0 = times[0]
        try:
            raw = mne.io.read_raw_fif(fpath, preload=True, verbose=False)
            tmin = max(0, t0 - CONFIG['T_WINDOW'] - 2)
            tmax = min(raw.times[-1], t0 + 2)
            
            if tmax <= tmin: 
                continue
            raw.crop(tmin, tmax)
            
            ts_raw, metrics = calculate_metrics_trajectory(raw, CONFIG['WIN_SIZE'], CONFIG['STEP'])
            
            # Temporal alignment to the event onset
            rel_ts = ts_raw + tmin - t0
            mask = (rel_ts >= target_ts.min()) & (rel_ts <= target_ts.max())
            if np.sum(mask) < 5: 
                continue

            # Z-score Normalization using dynamic baseline
            rel_ts_seg = rel_ts[mask]
            base_mask = (rel_ts_seg >= CONFIG['BASELINE_START']) & (rel_ts_seg <= CONFIG['BASELINE_END'])
            if np.sum(base_mask) == 0:
                base_mask = np.ones(len(rel_ts_seg), dtype=bool)

            for key in aligned.keys():
                val_raw = np.array(metrics[key])
                if val_raw.ndim > 1:
                    val = np.mean(val_raw[mask, :], axis=1)
                else:
                    val = val_raw[mask]
                
                b_mean, b_std = np.mean(val[base_mask]), np.std(val)
                norm_val = (val - b_mean) / (b_std if b_std != 0 else 1)
                
                # Interpolate to unified target time axis
                aligned[key].append(np.interp(target_ts, rel_ts_seg, norm_val))
                
        except Exception: 
            pass
    
    for k in aligned: 
        aligned[k] = np.array(aligned[k])
    return target_ts, aligned

# ==========================================
# 2. Statistical Tool Functions 
# ==========================================
def calculate_slope(data_list, time_axis, window=(-30, -5)):
    """Calculate linear trend slope within a specific temporal window."""
    mask = (time_axis >= window[0]) & (time_axis <= window[1])
    x = time_axis[mask]
    slopes = []
    for trace in data_list:
        if np.isnan(trace).any(): 
            continue
        y = trace[mask]
        if len(y) < 2 or np.std(y) == 0: 
            slopes.append(0)
        else:
            slopes.append(linregress(x, y).slope)
    return np.array(slopes)

def get_smooth_derivative(data_matrix, sigma=2.0):
    """Calculate smoothed first derivative (Kinematic Velocity) using Gaussian filtering."""
    if len(data_matrix) == 0: 
        return np.array([])
    smoothed = gaussian_filter1d(data_matrix, sigma=sigma, axis=1)
    velocity = np.gradient(smoothed, axis=1)
    return velocity

# ==========================================
# 3. Advanced Plotting Components 
# ==========================================
def draw_trajectory(ax, t, y_data, c_data, title, ylabel):
    """Draw time trajectory plot with Mean and SEM boundaries."""
    # Control Group (Gray)
    if len(c_data) > 0:
        mu_c = np.nanmean(c_data, axis=0)
        err_c = sem(c_data, axis=0, nan_policy='omit')
        ax.plot(t, mu_c, color=COLOR_CTRL, lw=0.5, label='Control', alpha=0.8)
        ax.fill_between(t, mu_c - err_c, mu_c + err_c, color=COLOR_CTRL, alpha=0.15, lw=0)
    
    # Experimental Group (Blue)
    if len(y_data) > 0:
        mu_y = np.nanmean(y_data, axis=0)
        err_y = sem(y_data, axis=0, nan_policy='omit')
        ax.plot(t, mu_y, color=COLOR_YAWN, lw=0.5, label='Pre-Yawn', alpha=0.9)
        ax.fill_between(t, mu_y - err_y, mu_y + err_y, color=COLOR_YAWN, alpha=0.25, lw=0)
    
    ax.axvline(0, color='k', ls=':', lw=0.5, alpha=0.4)
    ax.set_xlim(-30, 0)
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    if 'A.' in title: 
        ax.legend(loc='upper left', fontsize=8)

def draw_estimation_plot(ax_main, data_ctrl, data_yawn, ylabel, title):
    """Draw Gardner-Altman Estimation Plot (Bootstrap Confidence Intervals & Density)."""
    positions = [0, 1]
    data = [data_ctrl, data_yawn]
    colors = [COLOR_CTRL, COLOR_YAWN]
    
    bp = ax_main.boxplot(
        data, positions=positions, widths=0.4, patch_artist=True, 
        showfliers=False, 
        medianprops={'color': 'k', 'linewidth': 0.5},
        boxprops={'linewidth': 0.5},
        whiskerprops={'linewidth': 0.5},
        capprops={'linewidth': 0.5}
    )
                         
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    
    for i, d in enumerate(data):
        jitter = np.random.normal(i, 0.04, size=len(d))
        ax_main.scatter(jitter, d, color='k', alpha=0.3, s=15, zorder=3, linewidth=0)
        
    # Bootstrap Mean Difference Calculation
    diff = np.mean(data_yawn) - np.mean(data_ctrl)
    ax_diff = ax_main.twinx()
    
    if len(data_ctrl) > 2 and len(data_yawn) > 2:
        n_boot = 5000
        boot_diffs = []
        for _ in range(n_boot):
            res_y = np.random.choice(data_yawn, len(data_yawn), replace=True)
            res_c = np.random.choice(data_ctrl, len(data_ctrl), replace=True)
            boot_diffs.append(np.mean(res_y) - np.mean(res_c))
        
        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
        
        # Output statistical evaluation cleanly
        clean_title = title.replace('\n', ' ')
        print(f"[{clean_title}] Mean Delta: {diff:.4f}, 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

        kde = stats.gaussian_kde(boot_diffs)
        y_grid = np.linspace(min(boot_diffs), max(boot_diffs), 100)
        x_dens = kde(y_grid)
        x_dens = x_dens / max(x_dens) * 0.3 
        
        ax_diff.fill_betweenx(y_grid, 2.2, 2.2 + x_dens, color='k', alpha=0.2, lw=0)
        ax_diff.vlines(2.2, ci_low, ci_high, color='k', lw=0.5)
        
        # Mann-Whitney U Test annotation
        try:
            stat, p_val = mannwhitneyu(data_yawn, data_ctrl, alternative='two-sided')
            sig_char = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            ax_main.text(0.5, max(np.max(data_ctrl), np.max(data_yawn)) * 1.1, sig_char, ha='center')
        except Exception as e:
            print(f"Stats Error: {e}")

    ax_diff.plot([2.2], [diff], 'ko', ms=3)
    
    ax_main.set_xticks([0, 1])
    ax_main.set_xticklabels(['Ctrl', 'Yawn'])
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(title, fontweight='bold', fontsize=10)
    ax_main.set_xlim(-0.5, 2.8)
    
    ax_diff.set_ylabel("Mean Difference (Δ)", rotation=270, labelpad=12)
    ax_diff.spines['top'].set_visible(False)
    ax_diff.spines['left'].set_visible(False)
    ax_diff.spines['bottom'].set_visible(False)
    
    ax_diff.axhline(0, ls=':', c='k', alpha=0.3, lw=0.5)
    if "Slope" in ylabel or "Velocity" in title: 
        ax_main.axhline(0, ls=':', c='k', alpha=0.3, lw=0.5)

def draw_dtw_visual(ax, time_axis, trace_yawn, trace_ctrl_mean, title):
    """Draw DTW Alignment Path Plot to highlight morphological shifts."""
    if not HAS_DTW: 
        return
    
    dist, path = fastdtw(trace_yawn, trace_ctrl_mean)
    offset = np.max(trace_yawn) - np.min(trace_ctrl_mean) + 1.0
    y_shifted = trace_yawn + offset
    
    ax.plot(time_axis, y_shifted, color=COLOR_YAWN, lw=0.5, label='Single Yawn Trace')
    ax.plot(time_axis, trace_ctrl_mean, color=COLOR_CTRL, lw=0.5, ls='--', label='Control Avg (Template)')
    
    from matplotlib.collections import LineCollection
    segments = []
    for i, j in path[::5]: 
        segments.append([(time_axis[i], y_shifted[i]), (time_axis[j], trace_ctrl_mean[j])])
        
    lc = LineCollection(segments, color='gray', alpha=0.2, linewidth=0.5)
    ax.add_collection(lc)
    
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_ylabel("Norm. Amplitude (Shifted)")
    ax.set_xlabel("Time (s)")
    ax.text(time_axis[5], y_shifted[5] - 0.5, f"Dist: {dist:.1f}", color=COLOR_YAWN, fontsize=9)

# ==========================================
# 4. Main Program: Execute Analysis and Plot
# ==========================================
def run_analysis_figure1():
    print("Step 1: Loading and Processing Group Data...")
    yawn_res = process_group_data(PATH_YAWN) 
    ctrl_res = process_group_data(PATH_CONTROL) 
    
    if yawn_res is None or ctrl_res is None:
        print("Error: Could not load essential data files. Please check PATH_YAWN and PATH_CONTROL.")
        return

    t, y_dat = yawn_res
    _, c_dat = ctrl_res
    
    # Filter out NaNs for downstream computation
    def clean(d): 
        return d[~np.isnan(d).any(axis=1)]
    
    y_lzc, c_lzc = clean(y_dat['lzc']), clean(c_dat['lzc'])
    y_gam, c_gam = clean(y_dat['gamma']), clean(c_dat['gamma'])

    # Calculate First Derivative (Dynamic Velocity)
    y_lzc_vel = get_smooth_derivative(y_lzc, sigma=2.0)
    c_lzc_vel = get_smooth_derivative(c_lzc, sigma=2.0)
    y_gam_vel = get_smooth_derivative(y_gam, sigma=2.0)
    c_gam_vel = get_smooth_derivative(c_gam, sigma=2.0)

    print("Step 2: Generating Cross-Species Analysis Plots (8 Subplots)...")
    
    fig1 = plt.figure(figsize=(16, 9))
    gs1 = fig1.add_gridspec(2, 4, width_ratios=[1.2, 1.2, 1.0, 0.8], wspace=0.35, hspace=0.4)
    
    # ================= ROW 1: Neural Complexity (LZC) =================
    ax1 = fig1.add_subplot(gs1[0, 0])
    draw_trajectory(ax1, t, y_lzc, c_lzc, 'A. Complexity Dynamics (LZC)', 'LZC (Z)')
    
    ax2 = fig1.add_subplot(gs1[0, 1])
    draw_trajectory(ax2, t, y_lzc_vel, c_lzc_vel, 'B. LZC Rate of Change', 'd(LZC)/dt')

    ax3 = fig1.add_subplot(gs1[0, 2])
    y_lzc_slope = calculate_slope(y_lzc, t)
    c_lzc_slope = calculate_slope(c_lzc, t)
    
    if len(y_lzc) > 0 and len(c_lzc) > 0:
        tpl_l = np.mean(c_lzc, axis=0)
        idx = np.argmax(y_lzc_slope) if len(y_lzc_slope) > 0 else 0
        draw_dtw_visual(ax3, t, y_lzc[idx], tpl_l, 'C. LZC Pattern Shift\n(DTW Visual)')
        
    ax4 = fig1.add_subplot(gs1[0, 3])
    draw_estimation_plot(ax4, c_lzc_slope, y_lzc_slope, 'Slope (k)', 'D. Trend Analysis\n(Is it rising?)')

    # ================= ROW 2: Gamma Effort Dynamics =================
    ax5 = fig1.add_subplot(gs1[1, 0])
    draw_trajectory(ax5, t, y_gam, c_gam, 'E. Gamma Energy Effort', 'Gamma Amp (Z)')
    
    ax6 = fig1.add_subplot(gs1[1, 1])
    draw_trajectory(ax6, t, y_gam_vel, c_gam_vel, 'F. Gamma Rate of Change', 'd(Gamma)/dt')
    
    ax7 = fig1.add_subplot(gs1[1, 2])
    y_gam_slope = calculate_slope(y_gam, t)
    c_gam_slope = calculate_slope(c_gam, t)

    if len(y_gam) > 0 and len(c_gam) > 0:
        tpl_g = np.mean(c_gam, axis=0)
        idx_g = np.argmax(y_gam_slope) if len(y_gam_slope) > 0 else 0
        draw_dtw_visual(ax7, t, y_gam[idx_g], tpl_g, 'G. Gamma Pattern Shift\n(DTW Visual)')
        
    ax8 = fig1.add_subplot(gs1[1, 3])
    draw_estimation_plot(ax8, c_gam_slope, y_gam_slope, 'Slope (k)', 'H. Effort Trend\n(Struggle Strength)')

    fig1.suptitle("Neural Dynamics of State Transition (Trajectory, Velocity, Morphology & Statistics)", 
                  fontsize=14, fontweight='bold', y=0.98)
    
    save_path = os.path.join(SAVE_DIR, "Figure_1_Trajectory_Velocity_Combined_Final.pdf")
    fig1.savefig(save_path, bbox_inches='tight', dpi=300)
    
    print(f"\nPipeline successfully executed! Output chart saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_analysis_figure1()