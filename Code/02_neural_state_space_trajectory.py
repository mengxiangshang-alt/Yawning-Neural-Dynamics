import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. Basic Configuration (Publication-Quality Plot Standards)
# ==========================================
# Font settings: Ensure editable text in exported PDFs
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Line width and tick settings (Modified to meet journal requirements)
plt.rcParams['lines.linewidth'] = 0.5       # Line width: 0.5pt
plt.rcParams['axes.linewidth'] = 0.5        # Axis border width: 0.5pt
plt.rcParams['xtick.major.width'] = 0.5     # X-axis tick width
plt.rcParams['ytick.major.width'] = 0.5     # Y-axis tick width
plt.rcParams['xtick.major.size'] = 6        # X-axis tick length: 6pt
plt.rcParams['ytick.major.size'] = 6        # Y-axis tick length: 6pt
plt.rcParams['axes.unicode_minus'] = False 

# --- Modify the following parameters based on experimental records ---
FILE_PATH = r'D:\EEG\处理后的脑电数据\after_icaDXS10_0903tri_eeg.fif'
T_LOC = 115.0       # Time of Loss of Consciousness (seconds)
T_ROC = 620.0       # Time of Return of Consciousness (seconds)

# Analytical parameters
WINDOW_SIZE = 5.0   # Window size (seconds)
STEP_SIZE = 2.0     # Step size (seconds)
SMOOTH_WINDOW = 51  # Smoothing window length

# Frequency band definitions
FREQ_BANDS = {
    'Delta': (0.5, 4), 
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30), 
    'Gamma': (30, 80)
}

# ==========================================
# 2. Feature Extraction Function
# ==========================================
def extract_features(raw):
    sfreq = raw.info['sfreq']
    n_window = int(WINDOW_SIZE * sfreq)
    n_step = int(STEP_SIZE * sfreq)
    
    features = []
    times = []
    
    for start in range(0, raw.n_times - n_window, n_step):
        stop = start + n_window
        data = raw.get_data(start=start, stop=stop)
        
        psds, freqs = mne.time_frequency.psd_array_welch(
            data, sfreq, fmin=0.5, fmax=80.0, n_fft=n_window, verbose=False
        )
        
        avg_psd = np.mean(psds, axis=0)
        total_power = np.sum(avg_psd)
        
        if total_power == 0: 
            continue
            
        band_powers = []
        for fmin, fmax in FREQ_BANDS.values():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            band_powers.append(np.sum(avg_psd[idx]) / total_power)
            
        features.append(band_powers)
        times.append((start + n_window / 2) / sfreq)
        
    return np.array(features), np.array(times)

# ==========================================
# 3. Main Execution Logic
# ==========================================
if __name__ == '__main__':
    print(f"Loading data from: {FILE_PATH}")
    raw = mne.io.read_raw_fif(FILE_PATH, preload=True, verbose='error')
    
    # --- A. Feature Calculation ---
    print("Extracting features...")
    X, times = extract_features(raw)
    
    # --- B. Data Standardization ---
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # --- C. Coordinate Center Alignment ---
    deep_mask = (times > T_LOC + 60) & (times < T_ROC - 60)
    if np.sum(deep_mask) > 10:
        deep_center = np.mean(X_std[deep_mask], axis=0)
    else:
        print("Warning: Deep anesthesia period too short, using LOC+ period.")
        loc_idx = np.argmin(np.abs(times - T_LOC))
        deep_center = np.mean(X_std[loc_idx:loc_idx+50], axis=0)
        
    X_std = X_std - deep_center
    
    # --- D. PCA Dimensionality Reduction ---
    print("Running PCA...")
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_std)
    
    # --- E. Trajectory Smoothing ---
    print("Smoothing trajectory...")
    X_smooth = np.zeros_like(X_3d)
    actual_window = min(SMOOTH_WINDOW, len(X_3d) // 2 * 2 + 1)
    
    for i in range(3):
        X_smooth[:, i] = savgol_filter(X_3d[:, i], window_length=actual_window, polyorder=3)

    # ==========================================
    # 4. Plotting (Segmented solid colors & linewidth control)
    # ==========================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get indices for key time points
    loc_idx = np.argmin(np.abs(times - T_LOC))
    roc_idx = np.argmin(np.abs(times - T_ROC))
    
    # --- F. Plot Trajectory by Segments ---
    # 1. Induction Period (Start -> LOC): Blue
    ax.plot(X_smooth[:loc_idx+1, 0], X_smooth[:loc_idx+1, 1], X_smooth[:loc_idx+1, 2], 
            c='blue', label='Induction', linewidth=0.5, alpha=0.9)
    
    # 2. Anesthesia Maintenance Period (LOC -> ROC): Red
    ax.plot(X_smooth[loc_idx:roc_idx+1, 0], X_smooth[loc_idx:roc_idx+1, 1], X_smooth[loc_idx:roc_idx+1, 2], 
            c='red', label='Anesthesia', linewidth=0.5, alpha=0.9)
    
    # 3. Recovery Period (ROC -> End): Green
    ax.plot(X_smooth[roc_idx:, 0], X_smooth[roc_idx:, 1], X_smooth[roc_idx:, 2], 
            c='green', label='Recovery', linewidth=0.5, alpha=0.9)
    
    # --- G. Key Point Markers (Marker edge linewidth set to 0.5) ---
    marker_lw = 0.5
    
    # 1. Start point
    ax.scatter(X_smooth[0,0], X_smooth[0,1], X_smooth[0,2], 
               c='blue', s=80, label='Wake (Start)', edgecolors='white', linewidth=marker_lw, zorder=10)
    
    # 2. LOC
    ax.scatter(X_smooth[loc_idx,0], X_smooth[loc_idx,1], X_smooth[loc_idx,2], 
               c='orange', s=100, label='LOC', edgecolors='white', linewidth=marker_lw, zorder=10)
    
    # 3. Deep anesthesia center (Attractor)
    ax.scatter(0, 0, 0, c='red', s=150, marker='*', label='Deep State Attractor', linewidth=marker_lw, zorder=10)
    
    # 4. ROC
    if roc_idx < len(X_smooth):
        ax.scatter(X_smooth[roc_idx,0], X_smooth[roc_idx,1], X_smooth[roc_idx,2], 
                   c='lime', s=100, label='ROC', edgecolors='white', linewidth=marker_lw, zorder=10)
    
    # 5. End point
    ax.scatter(X_smooth[-1,0], X_smooth[-1,1], X_smooth[-1,2], 
               c='green', s=80, label='End', edgecolors='white', linewidth=marker_lw, zorder=10)

    # --- H. Axes and Decorations ---
    # Remove background pane colors
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Remove grid
    ax.grid(False)
    
    # Set axis labels
    ax.set_xlabel('PC 1', fontweight='bold', labelpad=10)
    ax.set_ylabel('PC 2', fontweight='bold', labelpad=10)
    ax.set_zlabel('PC 3', fontweight='bold', labelpad=10)
    ax.set_title('3D Trajectory of Anesthesia Dynamics', fontsize=14, pad=20)
    
    # Set tick parameters (Ensure length is 6pt, width is 0.5pt)
    ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=0.5)

    # Legend
    ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.1, 1), fontsize=9)
    
    # Adjust viewing angle
    ax.view_init(elev=15, azim=-88)
    
    plt.tight_layout()
    
    # Save output
    save_name = 'Anesthesia_3D_Smooth_Trajectory_Pub.pdf'
    plt.savefig(save_name, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Successfully saved plot as: {save_name}")