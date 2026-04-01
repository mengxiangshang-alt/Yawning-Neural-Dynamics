import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import mne
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

"""
Unsupervised clustering of yawning-related neural dynamics.
This script extracts peri-yawning EEG epochs (-30s to +30s) in Beagle dog models, 
computes 4D spectral features (Delta/Gamma relative changes and global power), 
and identifies distinct macroscopic network trajectories (e.g., transitions 
towards anesthesia or wakefulness) using K-Means and UMAP visualization.
"""

# ==========================================
# 1. Configuration & Publication Plot Settings
# ==========================================
output_dir = r'D:\EEG\Figures\Yawn_Clustering_Optimized'
os.makedirs(output_dir, exist_ok=True)

# Global configuration file path
config_json = r"D:\EEG\处理后的脑电数据\eeg_yawn_config_all.json"
if not os.path.exists(config_json):
    raise FileNotFoundError(f"Configuration file not found: {config_json}")

# Publication-quality plot settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.linewidth": 0.5,
    "axes.labelsize": 13,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none"
})

# ==========================================
# 2. Event Loading & Feature Definition
# ==========================================
print("Step 1: Loading all yawn events from JSON configuration...")

with open(config_json, 'r', encoding='utf-8') as f:
    config = json.load(f)

all_events = []  # Format: (file_path, onset_time)
for file_path, times in config.get("yawn_times", {}).items():
    for t in times:
        all_events.append((file_path, t))

print(f"Loaded {len(all_events)} yawn events in total.")

def compute_4d_features(raw, t_onset):
    """
    Compute 4D spectral features for a single event (averaged across 16 channels):
    1. delta_change = post_delta - pre_delta
    2. gamma_change = post_gamma - pre_gamma
    3. delta_global = whole-window average delta power
    4. gamma_global = whole-window average gamma power
    """
    sfreq = raw.info['sfreq']
    tmin, tmax = -30.0, 30.0
    
    try:
        # Extract [-30, +30] seconds peri-event epoch
        epochs = mne.Epochs(
            raw,
            events=np.array([[int(t_onset * sfreq), 0, 1]]),
            tmin=tmin, tmax=tmax,
            baseline=None, preload=True, verbose=False
        )
        if len(epochs) == 0:
            return None
            
        data = epochs.get_data(copy=False)[0]  # Shape: (n_ch, n_times), n_ch=16
        
        # Morlet Time-Frequency Representation (1-80 Hz)
        freqs = np.logspace(np.log10(1), np.log10(80), 60)
        n_cycles = freqs / 2.0
        power = mne.time_frequency.tfr_array_morlet(
            data[None, :, :],  # Shape: (1, n_ch, n_times)
            sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
            output='power', zero_mean=True
        )[0]  # Shape: (n_ch, n_freqs, n_times)
        
        # Convert to Decibels (dB)
        power_db = 10 * np.log10(power + 1e-12)
        times_array = epochs.times
        
        # Define temporal and spectral masks
        pre_mask = (times_array >= -30) & (times_array <= 0)
        post_mask = (times_array >= 0) & (times_array <= 30)
        delta_mask = (freqs >= 1) & (freqs <= 4)
        gamma_mask = (freqs >= 30) & (freqs <= 80)
        
        # Average across all 16 channels
        power_db_ch_avg = np.mean(power_db, axis=0)  # Shape: (n_freqs, n_times)
        
        # Calculate localized power features
        pre_d = np.mean(power_db_ch_avg[delta_mask, :][:, pre_mask])
        post_d = np.mean(power_db_ch_avg[delta_mask, :][:, post_mask])
        pre_g = np.mean(power_db_ch_avg[gamma_mask, :][:, pre_mask])
        post_g = np.mean(power_db_ch_avg[gamma_mask, :][:, post_mask])
        
        # Calculate global power features
        global_d = np.mean(power_db_ch_avg[delta_mask, :])
        global_g = np.mean(power_db_ch_avg[gamma_mask, :])
        
        return np.array([
            post_d - pre_d,   # delta_change
            post_g - pre_g,   # gamma_change
            global_d,         # delta_global
            global_g          # gamma_global
        ])
    except Exception:
        return None

feature_names = ['delta_change', 'gamma_change', 'delta_global', 'gamma_global']

# ==========================================
# 3. Batch Feature Extraction
# ==========================================
print("\nStep 2: Extracting 4D spectral features...")

X_features = []
event_ids = []  

for file_path, t_onset in all_events:
    if not os.path.exists(file_path):
        print(f"Warning: File missing: {file_path}")
        continue
    
    try:
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        if len(raw.ch_names) != 16:
            continue
        
        features = compute_4d_features(raw, t_onset)
        if features is not None:
            event_id = f"{os.path.basename(file_path)}_{t_onset:.1f}"
            X_features.append(features)
            event_ids.append(event_id)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        continue

X_features = np.array(X_features)
print(f"Extracted {X_features.shape[0]} samples with {X_features.shape[1]} features.")

if X_features.shape[0] == 0:
    raise ValueError("Failed to extract any valid features.")

# ==========================================
# 4. Feature Standardization & K-Means Clustering
# ==========================================
print("\nStep 3: Standardizing features and running K-Means clustering...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
labels_kmeans = kmeans.fit_predict(X_scaled)

# Calculate cluster assignment confidence based on distance
distances = kmeans.transform(X_scaled)
min_distances = np.min(distances, axis=1)
confidence = 1 / (1 + min_distances)

print(f"Clustering complete. Sample distribution across clusters: {np.bincount(labels_kmeans)}")

# ==========================================
# 5. UMAP Visualization
# ==========================================
print("\nStep 4: UMAP Visualization (Direct mapping from 4D space)...")

umap_model = umap.UMAP(
    n_neighbors=8, min_dist=0.3, random_state=42, n_components=2, verbose=False
)
X_umap = umap_model.fit_transform(X_scaled)

colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44']
custom_cmap = ListedColormap(colors)

plt.figure(figsize=(6, 5))
ax = plt.gca()
scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=labels_kmeans, cmap=custom_cmap,
                     s=50, edgecolor='k', linewidth=0.2)

cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1, 2, 3], boundaries=np.arange(-0.5, 5, 1))
cbar.set_label('Cluster', fontsize=12)
ax.set_title("K-Means Clustering (UMAP Projection)", fontsize=13, pad=10)
ax.set_xlabel("UMAP Dimension 1", fontsize=12)
ax.set_ylabel("UMAP Dimension 2", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "clustering_umap.svg"), bbox_inches='tight')
plt.close()

# ==========================================
# 6. Feature Importance Profiling
# ==========================================
print("\nStep 5: Profiling feature importance per cluster...")

feature_means = {}
for i in range(4):
    mask = (labels_kmeans == i)
    if np.sum(mask) > 0:
        feature_means[i] = np.mean(X_scaled[mask], axis=0)
    else:
        feature_means[i] = np.zeros(X_scaled.shape[1])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i in range(4):
    axes[i].barh(feature_names, feature_means[i])
    axes[i].set_title(f'Cluster {i} (n={np.sum(labels_kmeans==i)})', fontsize=12)
    axes[i].set_xlabel('Standardized Feature Value')
    axes[i].tick_params(axis='y', labelsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_comparison.svg"), bbox_inches='tight')
plt.close()

# ==========================================
# 7. Memory-Optimized 16-Channel Average Heatmaps
# ==========================================
print("\nStep 6: Generating 16-channel average TFR heatmaps (Memory Optimized)...")

# Initialize accumulators (Restricted to 1-40Hz to optimize memory usage)
sample_freqs = np.logspace(np.log10(1), np.log10(80), 60)
n_times = int(60 * 250) + 1  # Assuming sfreq=250Hz, [-30,30] -> 15001 points
n_freqs_40 = np.sum(sample_freqs <= 40)

cluster_sum_tfr = {i: np.zeros((16, n_freqs_40, n_times), dtype=np.float32) for i in range(4)}
cluster_valid_count = {i: 0 for i in range(4)}

for (file_path, t_onset), cluster_id in zip(all_events, labels_kmeans):
    if not os.path.exists(file_path):
        continue
    try:
        raw = mne.io.read_raw_fif(file_path, preload=False, verbose=False)
        if len(raw.ch_names) != 16:
            continue
        
        sfreq = raw.info['sfreq']
        start_samp = int((t_onset - 30.0) * sfreq)
        end_samp = int((t_onset + 30.0) * sfreq)
        data, _ = raw[:, start_samp:end_samp]  # Shape: (16, n_times)
        
        # Morlet TFR restricted to 1-40 Hz
        freqs_tfr = np.logspace(np.log10(1), np.log10(40), 40)
        n_cycles = freqs_tfr / 2.0
        power = mne.time_frequency.tfr_array_morlet(
            data[None, :, :], 
            sfreq=sfreq, freqs=freqs_tfr, n_cycles=n_cycles,
            output='power', zero_mean=True
        )[0]
        
        power_db = 10 * np.log10(power + 1e-12)
        cluster_sum_tfr[cluster_id] += power_db
        cluster_valid_count[cluster_id] += 1
        
    except Exception:
        continue

times_plot = np.linspace(-30, 30, n_times)
freqs_plot = np.logspace(np.log10(1), np.log10(40), 40)

for cluster_id in range(4):
    if cluster_valid_count[cluster_id] == 0:
        continue
    
    mean_tfr = cluster_sum_tfr[cluster_id] / cluster_valid_count[cluster_id]
    vmin = np.percentile(mean_tfr, 5)
    vmax = np.percentile(mean_tfr, 95)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    for ch in range(16):
        im = axes[ch].imshow(
            mean_tfr[ch, :, :], aspect='auto', origin='lower',
            extent=[times_plot[0], times_plot[-1], freqs_plot[0], freqs_plot[-1]],
            cmap='turbo', vmin=vmin, vmax=vmax
        )
        axes[ch].set_title(f'Ch {ch+1}', fontsize=10)
        axes[ch].set_xticks([])
        axes[ch].set_yticks([])
        axes[ch].axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.7)
        
    fig.suptitle(f'Cluster {cluster_id} Average TFR (n={cluster_valid_count[cluster_id]})', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cluster_{cluster_id}_16ch_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 8. Pie Chart of Cluster Distribution
# ==========================================
print("\nStep 7: Plotting cluster distribution pie chart...")

cluster_counts = np.bincount(labels_kmeans, minlength=4)
percentages = 100 * cluster_counts / cluster_counts.sum()

plt.figure(figsize=(6, 6))
wedges, texts, autotexts = plt.pie(
    percentages,
    labels=[f'Cluster {i}' for i in range(4)],
    colors=colors, autopct='%1.1f%%', startangle=90,
    textprops={'fontsize': 12, 'color': 'white'},
    wedgeprops={'linewidth': 0.8, 'edgecolor': 'white'}
)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    
plt.title("Cluster Distribution of Yawning Events", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cluster_distribution_pie.svg"), bbox_inches='tight')
plt.close()

# ==========================================
# 9. Result Export & Interpretation
# ==========================================
print("\nStep 8: Exporting results...")

np.save(os.path.join(output_dir, "features.npy"), X_features)
np.save(os.path.join(output_dir, "cluster_labels.npy"), labels_kmeans)
np.save(os.path.join(output_dir, "umap_coords.npy"), X_umap)

event_df = pd.DataFrame({
    'event_id': event_ids,
    'cluster': labels_kmeans,
    'confidence': confidence
})
event_df.to_csv(os.path.join(output_dir, "event_clusters.csv"), index=False)

print(f"\nPipeline successfully completed! Data saved to: {output_dir}")
print("\n--- Feature Interpretation Guide ---")
print("  - delta_change = post_delta - pre_delta")
print("  - gamma_change = post_gamma - pre_gamma")
print("\n--- Anticipated Macro-State Patterns ---")
print("  Cluster 0 -> Trend towards anesthesia: delta_change > 0 (Drowsiness/Inhibition)")
print("  Cluster 1 -> Trend towards wakefulness: delta_change < 0 (Arousal)")
print("  Cluster 2 -> Awake state: High global_gamma, Low global_delta")
print("  Cluster 3 -> Deep anesthesia/sleep: High global_delta, Low global_gamma")