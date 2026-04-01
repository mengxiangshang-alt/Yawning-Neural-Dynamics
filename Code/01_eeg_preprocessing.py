import os
import warnings
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# =====================================================================
# EEG Data Preprocessing Pipeline
# =====================================================================

# Define file paths
bdf_file = r'D:\Original_Data\Data_bdf\DXS08_2102tri.bdf'
file_id = os.path.splitext(os.path.basename(bdf_file))[0]

# Load raw BDF data
raw = mne.io.read_raw_bdf(bdf_file, preload=True)  
print(raw.info)

# Pick EEG channels only and adjust amplitude scale
raw.pick_types(eeg=True)
raw = raw.apply_function(lambda x: x * 0.01)

raw_duration = raw.times[-1] - raw.times[0]
raw.plot(duration=raw_duration, n_channels=32, clipping=None, remove_dc=False)
raw.compute_psd(fmin=0, fmax=150).plot()

# ===============================
# 1. Set Electrode Montage
# ===============================
electrode_positions = {
    'Chan 8': [2.0, -0.5, 0.0],
    'Chan 9': [1.8, -1.8, 0.0],
    'Chan 11': [0.8, -1.0, 0.0],
    'Chan 12': [1.5, -2.5, 0.0],
    'Chan 13': [-1.5, -2.5, 0.0],
    'Chan 14': [-0.8, -1.0, 0.0],
    'Chan 16': [-1.8, -1.8, 0.0],
    'Chan 1': [-2.0, -0.5, 0.0],
    'Chan 2': [-0.8, 1.8, 0.0],
    'Chan 4': [-1.6, 1.0, 0.0],
    'Chan 3': [-0.8, 0.5, 0.0],
    'Chan 6': [0.8, 0.5, 0.0],
    'Chan 5': [1.6, 1.0, 0.0],
    'Chan 7': [0.8, 1.8, 0.0],
    'Chan 15': [-1.0, -3.0, 0.0],
    'Chan 10': [1.0, -3.0, 0.0],
}
montage = mne.channels.make_dig_montage(ch_pos=electrode_positions)
raw.set_montage(montage)

# ===============================
# 2. Mark Bad Channels
# ===============================
bad_channels = ['Chan 1', 'Chan 10']
raw.info['bads'] = bad_channels
raw.plot(duration=raw_duration, n_channels=32, clipping=None, remove_dc=False)

# ===============================
# 3. Band-pass Filtering 
#    (Remove low-frequency drifts & high-frequency noise)
# ===============================
raw = raw.filter(l_freq=0.5, h_freq=80)

# ===============================
# 4. Extreme Amplitude Artifact Removal 
#    (Amplitude thresholding method)
# ===============================
# Operate only on good channels
picks = mne.pick_types(raw.info, eeg=True, exclude='bads')

# Extract data matrix
data = raw.get_data(picks=picks)
means = data.mean(axis=1, keepdims=True)
stds = data.std(axis=1, keepdims=True)

# Set upper and lower thresholds (e.g., ±5 standard deviations)
lower_thresholds = means - 5 * stds
upper_thresholds = means + 5 * stds

# Generate mask to identify artifacts
mask = (data < upper_thresholds) & (data > lower_thresholds)
cleaned_data = np.where(mask, data, np.nan)  # Mark anomalous segments as NaN

# Replace NaN segments with linear interpolation for smoothness
for i in range(cleaned_data.shape[0]):
    ch_data = cleaned_data[i]
    bad_idx = np.isnan(ch_data)
    if np.any(bad_idx):
        good_idx = ~bad_idx
        cleaned_data[i, bad_idx] = np.interp(
            np.flatnonzero(bad_idx),
            np.flatnonzero(good_idx),
            ch_data[good_idx]
        )

# Safely update the raw object with cleaned data
raw._data[picks, :] = cleaned_data

# ===============================
# 5. Re-referencing (Good channels only)
# ===============================
eeg_chs = mne.pick_types(raw.info, eeg=True)
good_ch_names = [raw.ch_names[ch] for ch in eeg_chs if raw.ch_names[ch] not in raw.info['bads']]
raw.set_eeg_reference(ref_channels=good_ch_names, projection=False)

# ===============================
# 6. Downsampling (To accelerate ICA)
# ===============================
raw.resample(250)

# ===============================
# 7. Visual Inspection
# ===============================
raw.plot(duration=raw_duration, n_channels=32, clipping=None, remove_dc=False)

# ===============================
# 8. Independent Component Analysis (ICA)
# ===============================
# Note: Adjust n_components based on dataset specificities
ica = mne.preprocessing.ICA(n_components=14, max_iter="auto", random_state=97)
ica.fit(raw.copy().pick_types(eeg=True, exclude=raw.info['bads']))
ica.plot_sources(raw, start=0, stop=60) 

# Configuration for custom ICA plotting
n_components = ica.n_components_
n_row, n_col = 4, 4
fig = plt.figure(figsize=(6 * n_col, 3 * n_row))

# Obtain ICA time series sources and sampling frequency
ica_sources = ica.get_sources(raw)
sfreq = raw.info['sfreq']

# Extract data for topographic maps
mixing_matrix = ica.mixing_matrix_          
ica_picks = ica.ch_names                        
info_subset = mne.pick_info(raw.info, mne.pick_channels(raw.info['ch_names'], ica_picks))

for i in range(n_components):
    # Calculate power spectrum
    data_source = ica_sources.get_data(picks=[i])[0, :]
    n = len(data_source)
    yf = fft(data_source)
    xf = fftfreq(n, 1 / sfreq)[:n // 2]
    yf_abs = np.abs(yf[:n // 2])

    # Create 1x2 subplots
    gs = fig.add_gridspec(1, 2,
                          left=(i % n_col) / n_col + 0.01,
                          right=(i % n_col + 1) / n_col - 0.01,
                          top=1 - (i // n_col) / n_row - 0.01,
                          bottom=1 - (i // n_col + 1) / n_row + 0.01,
                          wspace=0.06)
    ax_spec = fig.add_subplot(gs[0])
    ax_topo = fig.add_subplot(gs[1])

    # Plot power spectrum
    ax_spec.plot(xf, yf_abs, color='k')
    ax_spec.set_xlim(0, 80)
    ax_spec.set_title(f'IC{i}')
    ax_spec.set_xlabel('Frequency (Hz)')
    ax_spec.set_ylabel('Amplitude')

    # Plot topography 
    topo = mixing_matrix[:, i]
    mne.viz.plot_topomap(topo, info_subset, axes=ax_topo, show=False)
    ax_topo.set_title('')

# Hide unused subplots
for j in range(n_components, n_row * n_col):
    gs = fig.add_gridspec(1, 2,
                          left=(j % n_col) / n_col + 0.01,
                          right=(j % n_col + 1) / n_col - 0.01,
                          top=1 - (j // n_col) / n_row - 0.01,
                          bottom=1 - (j // n_col + 1) / n_row + 0.01)
    for k in range(2):
        ax = fig.add_subplot(gs[k])
        ax.set_visible(False)

plt.tight_layout()
plt.show()

# Specify ICA components to exclude (e.g., physiological artifacts)
exclude_components = [13] 
ica.exclude = exclude_components

# Apply ICA to remove specified artifact components
ica.apply(raw)

# ===============================
# 9. Interpolation and Post-processing
# ===============================
# Interpolate bad channels based on neighboring electrodes
raw.interpolate_bads(reset_bads=False)

# Clear bad channel markers after interpolation
raw.info['bads'] = []          

raw.plot(duration=raw_duration, clipping=None, remove_dc=False)

# Save processed dataset
raw_postica = rf'D:\EEG\处理后的脑电数据\after_ica{file_id}_eeg.fif'
raw.save(raw_postica, overwrite=True)
print(f"Preprocessing completed. Data saved to: {raw_postica}")