import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from pygam import LinearGAM, s
import seaborn as sns

"""
Quantifying neural state dynamics and evaluating state transitions in Beagle dog models.
This script calculates the Euclidean distance between instantaneous neural states 
and the deep anesthesia state center, followed by smoothing the temporal trajectory 
using a Generalized Additive Model (GAM).
"""

# ==========================================
# 1. Plotting Style Configuration (Publication Standard)
# ==========================================
# Font settings: Ensure editable text in exported PDFs
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False 

# Global line width and tick settings
plt.rcParams['lines.linewidth'] = 0.5      # Global line width
plt.rcParams['axes.linewidth'] = 0.5       # Axis border width
plt.rcParams['xtick.major.width'] = 0.5    # X-axis tick width
plt.rcParams['ytick.major.width'] = 0.5    # Y-axis tick width
plt.rcParams['xtick.major.size'] = 6       # X-axis tick length (6pt)
plt.rcParams['ytick.major.size'] = 6       # Y-axis tick length (6pt)

# ==========================================
# 2. Parameter Definitions
# ==========================================
file_path = r'D:\EEG\处理后的脑电数据\after_icaDXS10_0903tri_eeg.fif'
loc_time_point = 90.0    # Time point for Loss of Consciousness (LOC)
window_size = 5.0        # Sliding window size (seconds)
step_size = 2.0          # Step size (seconds)

freq_bands = {
    'Delta': (0.5, 4), 
    'Theta': (4, 8), 
    'Alpha': (8, 13), 
    'Beta': (13, 30), 
    'Gamma': (30, 80)
}

# ==========================================
# 3. Data Processing Functions
# ==========================================
def extract_5d_feature(data_segment, sfreq):
    """
    Extract 5-dimensional spectral power features (Delta to Gamma) 
    from a given EEG data segment.
    """
    n_fft = int(sfreq * window_size)
    psds, freqs = mne.time_frequency.psd_array_welch(
        data_segment, sfreq, fmin=0.5, fmax=80.0, 
        n_fft=n_fft, n_overlap=0, verbose=False
    )
    
    # Average PSD across all channels to represent global state
    avg_psd = np.mean(psds, axis=0)
    total_power = np.sum(avg_psd)
    
    if total_power == 0: 
        return np.zeros(5)
        
    features = []
    for band, (fmin, fmax) in freq_bands.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        features.append(np.sum(avg_psd[idx]) / total_power)
        
    return np.array(features)

# ==========================================
# 4. Main Execution Logic
# ==========================================
if __name__ == '__main__':
    print(f"Loading data from: {file_path}")
    raw = mne.io.read_raw_fif(file_path, preload=True, verbose='error')
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    n_window = int(window_size * sfreq)
    n_step = int(step_size * sfreq)
    
    feature_list = []
    time_points = []
    
    print("Extracting spectral features...")
    for start_idx in range(0, n_samples - n_window, n_step):
        end_idx = start_idx + n_window
        segment = raw.get_data(start=start_idx, stop=end_idx)
        
        feature_list.append(extract_5d_feature(segment, sfreq))
        time_points.append((start_idx + n_window / 2) / sfreq)
        
    feature_matrix = np.array(feature_list)
    time_points = np.array(time_points)
    
    # Calculate Euclidean distance to the LOC state
    loc_idx = np.argmin(np.abs(time_points - loc_time_point))
    F_loc = np.mean(feature_matrix[loc_idx:loc_idx+3], axis=0)
    distances = np.array([euclidean(f, F_loc) for f in feature_matrix])
    adjusted_times = time_points - time_points[loc_idx]

    # ==========================================
    # 5. Generalized Additive Model (GAM) Fitting
    # ==========================================
    print("Fitting GAM...")
    X = adjusted_times.reshape(-1, 1)
    y = distances
    
    # Fit model with grid search for optimal parameters
    gam = LinearGAM(s(0, n_splines=25)).gridsearch(X, y)
    distances_smooth = gam.predict(X)
    conf_int = gam.prediction_intervals(X, width=0.95)

    # ==========================================
    # 6. Generate Publication-Quality Plot
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 8)) 
    
    # Plot raw distance data (linewidth: 0.5)
    ax.plot(adjusted_times, distances, color='gray', linewidth=0.5, 
            alpha=0.4, label='Raw Distance')
    
    # Plot GAM smoothed curve (linewidth: 0.5)
    ax.plot(adjusted_times, distances_smooth, color='black', linewidth=0.5, 
            label='GAM Fit')
    
    # Plot 95% Confidence Interval (linewidth: 0 ensures no extra border on shading)
    ax.fill_between(adjusted_times, conf_int[:, 0], conf_int[:, 1], 
                    color='black', alpha=0.15, label='95% CI', linewidth=0)
    
    # Add decorative reference lines (linewidth: 0.5)
    ax.axvline(0, color='#e74c3c', linestyle='--', linewidth=0.5, label='Unresponsiveness')
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    # Axis labels and title
    ax.set_xlabel('Time relative to Anesthesia Onset (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Euclidean Distance to LOC State', fontsize=12, fontweight='bold')
    ax.set_title('Anesthesia Trajectory (GAM Fit)', fontsize=14, pad=15)
    
    # Dynamically adjust Y-axis limits
    ylim_max = np.max(distances) * 1.1
    ax.set_ylim(-0.05, ylim_max)
    ax.set_xlim(adjusted_times[0], adjusted_times[-1])

    # Detailed tick settings (length: 6pt, width: 0.5pt)
    ax.tick_params(axis='both', which='major', labelsize=10, length=6, 
                   width=0.5, direction='out')
    
    # Ensure spine line widths are exactly 0.5pt
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Remove top and right spines
    sns.despine(ax=ax, trim=False) 
    
    # Add legend
    ax.legend(frameon=False, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save high-resolution PDF for publication
    save_name = 'Anesthesia_Distance_Plot_Publication.pdf'
    plt.savefig(save_name, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot successfully saved as: {save_name}")