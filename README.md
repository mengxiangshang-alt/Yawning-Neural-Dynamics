# Yawning-Neural-Dynamics
Yawning reveals energy-dynamics mismatch and neural inertia across state transitions。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data Availability](https://img.shields.io/badge/Data-Available-success)](#data-availability)

This repository contains the official data sharing plans, computational pipelines, and documentation required to reproduce the findings in the manuscript: *"Yawning reveals energy-dynamics mismatch and neural inertia across state transitions"*. 

Our research investigates the micro-scale neural dynamics of yawning across different arousal states (e.g., propofol anesthesia and wakefulness). Utilizing Beagle dog models and cross-species validation in humans, we combined multi-channel EEG recordings with advanced signal processing to uncover the Energy-Dynamics Mismatch (EDM) and neural inertia phenomena.

---

## 📂 Repository Structure

The repository is organized into three main directories to ensure full reproducibility and logical flow:

### 1. `/Code`
Contains the 15 core Python scripts used for the analytical pipeline, covering everything from raw EEG preprocessing to complex machine learning predictions and Linear Mixed-Effects (LME) statistical modeling.

* **Preprocessing & State Space Mapping**
  * `01_eeg_preprocessing.py`: Automated EEG filtering, artifact rejection, and ICA.
  * `02_neural_state_space_trajectory.py`: 3D PCA trajectory mapping using broad-band EEG spectral features.
  * `03_neural_state_distance_gam.py`: GAM smoothing of neural state distances relative to Loss of Consciousness (LOC).
* **Clustering & Micro-scale Dynamics**
  * `04_kmeans_umap_clustering.py`: Unsupervised K-Means clustering and UMAP visualization of yawning-related dynamics.
  * `05_lzc_complexity.py`: Sliding window Lempel-Ziv Complexity (LZC) and adaptive permutation testing.
  * `06_emd_hht_analysis.py`: Empirical Mode Decomposition (EMD) and Hilbert-Huang Transform (HHT) for instantaneous frequency tracking.
* **Prediction & Validation**
  * `07_random_forest_prediction.py`: Machine learning decoding of pre-yawning micro-states using Random Forest.
  * `08_cross_species_dtw.py`: Dynamic Time Warping (DTW) for cross-species/cross-state morphological alignment.
* **Statistical Modeling (LME & ANOVA)**
  * `09_spatial_dynamics_lme.py`: Regional EEG power dynamics assessed via Linear Mixed-Effects (LME) models.
  * `10_regional_features_interaction_lme.py`: Interaction effect analysis (Region × Condition) using Likelihood Ratio Tests (LRT).
  * `11_figure5_dhlp_statistics.py`: Generation of statistical trajectories and Gardner-Altman estimation plots (Fig 5 D, H, L, P).
  * `12_figure6c_gamma_statistics.py`: Quantification of Gamma band energy and volatility.
  * `13_figure6e_imf_volatility.py`: Instantaneous Frequency Volatility of IMF1/IMF2 components.
* **Robustness & Objective Screening**
  * `14_data_driven_time_window_scan.py`: Data-driven sliding window scan to prevent P-hacking and identify the optimal EDM epoch.
  * `15_edm_occurrence_quantification.py`: Quantification of EDM occurrence using intra-epoch 75th/25th quartile self-baselines.

### 2. `/Data`
Contains the preprocessed, lightweight EEG datasets and extracted feature matrices necessary to run the scripts.
* Includes intermediate `.fif` (MNE format)

### 3. `/Docs`
Contains essential metadata and configuration files.
* Includes `.json` dictionaries mapping specific EEG filenames to exact yawn and control event timestamps (e.g., `eeg_yawn_config_bg3.json`).

---

## ⚙️ System Requirements & Dependencies

The analytical scripts are written in Python. Ensure you have the following key libraries installed:

```bash
pip install numpy pandas scipy scikit-learn statsmodels
pip install mne        # For EEG data processing
pip install PyEMD      # For Empirical Mode Decomposition (EMD)
pip install fastdtw    # For Dynamic Time Warping
pip install matplotlib seaborn # For publication-quality plotting
