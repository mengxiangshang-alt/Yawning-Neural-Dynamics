import os
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import mne
from scipy.signal import welch, hilbert
from scipy.stats import sem
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from PyEMD import EMD

"""
Machine Learning Predictive Analysis of Pre-Yawning Neural Dynamics.
This script evaluates the predictive power of pre-yawning neural activity (e.g., IF volatility)
using a Random Forest classifier. Performance is rigorously evaluated using 
Stratified Group K-Fold cross-validation and a non-parametric permutation test.
"""

# ==========================================
# 0. Environment & Plot Style Configuration
# ==========================================
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE) 

# High-impact journal style configuration (Strict modification)
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42,          # Ensure editable text in PDFs
    'ps.fonttype': 42,
    'axes.linewidth': 0.5,       # Axis linewidth 0.5pt
    'lines.linewidth': 0.5,      # Global linewidth 0.5pt
    'xtick.major.width': 0.5,    # Tick width 0.5pt
    'ytick.major.width': 0.5,
    'xtick.major.size': 6,       # Tick length 6pt
    'ytick.major.size': 6,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# ==========================================
# [⭐ Parameter Settings ⭐]
# ==========================================
CLUSTER_ID = 0 
CONFIG = {
    'T_PRE': 30.0,          # Analysis window length (s)
    'T_GAP': 2.0,           # Prediction lead time (s)
    'WIN_LEN': 4.0,         # HHT calculation short window length (s)
    'WIN_STEP': 2.0,        # HHT sliding step (s)
    'SEARCH_STEP': 10.0,    # Negative sample search step (s)
    'BUFFER_ZONE': 90.0,    # Exclusion zone around yawning events (s)
    'MAX_CHANNELS': 3       # Use first 3 channels to prevent overfitting
}

BASE_DIR = r"D:\EEG\处理后的脑电数据"
BASE_SAVE_DIR = os.path.join(r"D:\EEG\Figures\哈欠预测_独立验证", f"Cluster_{CLUSTER_ID}_Scientific")
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

all_feature_names = [
    'IMF1_IF_Std_Mean', 'IMF1_IF_Std_Std', 
    'IMF2_IF_Std_Mean', 'IMF2_IF_Std_Std', 
    'IMF_Ratio_Mean', 'IMF_Ratio_Std'
]

# ==========================================
# 1. Core Functions for HHT & Physiological Fingerprints
# ==========================================
def get_5d_fingerprint(data_1d, sf):
    """Extract 5D Power Spectral Density (PSD) fingerprint."""
    try:
        f, psd = welch(data_1d, sf, nperseg=int(sf * 2))
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
        p = [np.sum(psd[(f >= l) & (f < r)]) for l, r in bands]
        return np.log10(np.array(p) + 1e-12)
    except: 
        return np.zeros(5)

def safe_hht_metrics_window(data_1d, sf):
    """Safely compute HHT metrics for a short sliding window."""
    data_scaled = data_1d * 1e6 
    if np.std(data_scaled) < 0.01: 
        return np.array([np.nan, np.nan, np.nan])
        
    try:
        emd = EMD()
        imfs = emd.emd(data_scaled, max_imf=4)
        if imfs is None or imfs.shape[0] < 2: 
            return np.array([np.nan, np.nan, np.nan])
            
        res = []
        for i in range(2):
            analytic = hilbert(imfs[i])
            inst_freq = np.diff(np.unwrap(np.angle(analytic))) / (2.0 * np.pi) * sf
            v = inst_freq[(inst_freq > 0.5) & (inst_freq < 100)]
            res.append(np.std(v) if len(v) > 5 else np.nan)
            
        idx_low = 2 if imfs.shape[0] > 2 else -1 
        ratio = np.mean(np.abs(hilbert(imfs[0]))) / (np.mean(np.abs(hilbert(imfs[idx_low]))) + 1e-9)
        res.append(ratio)
        return np.array(res)
    except: 
        return np.array([np.nan, np.nan, np.nan])

def extract_hht_features_30s(data_2d, sf):
    """Extract aggregated HHT features over a 30s epoch."""
    win_pts = int(CONFIG['WIN_LEN'] * sf)
    step_pts = int(CONFIG['WIN_STEP'] * sf)
    n_ch = min(data_2d.shape[0], CONFIG['MAX_CHANNELS'])
    metrics = []
    
    for i in range(0, data_2d.shape[1] - win_pts + 1, step_pts):
        ch_res = [safe_hht_metrics_window(data_2d[ch, i:i+win_pts], sf) for ch in range(n_ch)]
        metrics.append(np.nanmean(ch_res, axis=0))
        
    m = np.array(metrics)
    if np.all(np.isnan(m)): 
        return None
        
    return np.array([
        np.nanmean(m[:, 0]), np.nanstd(m[:, 0]), 
        np.nanmean(m[:, 1]), np.nanstd(m[:, 1]), 
        np.nanmean(m[:, 2]), np.nanstd(m[:, 2])
    ])

# ==========================================
# 2. Main Logic: Data Construction and Pairing
# ==========================================
j_path = os.path.join(BASE_DIR, f"cluster_{CLUSTER_ID}_yawns.json")
with open(j_path, 'r', encoding='utf-8') as f: 
    cfg = json.load(f)
yawn_data = cfg.get('yawn_times', cfg)

file_cache = {}
pos_samples = []
print(f"Processing Cluster {CLUSTER_ID} | Extracting features...")

# Extract positive (yawn) samples and prepare negative sample pools
for f_idx, (fpath, times) in enumerate(yawn_data.items()):
    real_p = fpath if os.path.exists(fpath) else os.path.join(BASE_DIR, os.path.basename(fpath))
    if not os.path.exists(real_p): 
        continue
        
    raw = mne.io.read_raw_fif(real_p, preload=True, verbose=False)
    raw.filter(0.5, 45., verbose=False) 
    sf = raw.info['sfreq']
    
    # Establish pool of valid negative search times
    search_times = np.arange(0, raw.times[-1] - CONFIG['T_PRE'], CONFIG['SEARCH_STEP'])
    forbidden = [(t - CONFIG['BUFFER_ZONE'], t + CONFIG['BUFFER_ZONE']) for t in times]
    valid_neg_times, fingerprints = [], []
    
    for st in search_times:
        if any(not (st + CONFIG['T_PRE'] < s or st > e) for s, e in forbidden): 
            continue
        seg = np.mean(raw.get_data(start=int(st*sf), stop=int((st+CONFIG['T_PRE'])*sf)), axis=0)
        fingerprints.append(get_5d_fingerprint(seg, sf))
        valid_neg_times.append(st)
        
    file_cache[real_p] = {
        'raw': raw, 
        'sf': sf, 
        'neg_times': np.array(valid_neg_times), 
        'neg_fps': np.array(fingerprints)
    }
    
    # Extract positive samples
    for t in times:
        t_s = t - CONFIG['T_GAP'] - CONFIG['T_PRE']
        if t_s < 0: 
            continue
        feat_y = extract_hht_features_30s(
            raw.get_data(start=int(t_s*sf), stop=int((t-CONFIG['T_GAP'])*sf)), sf
        )
        if feat_y is not None:
            pos_samples.append({
                'feat': feat_y, 
                'path': real_p, 
                'f_idx': f_idx, 
                'target_fp': get_5d_fingerprint(
                    raw.get_data(start=int(t_s*sf), stop=int((t-CONFIG['T_GAP'])*sf)).mean(axis=0), sf
                )
            })

# Pair each positive sample with the closest negative sample via Euclidean distance
X_final, y_final, groups_final = [], [], []
scaler_fp = StandardScaler().fit(np.array([p['target_fp'] for p in pos_samples]))

for p in pos_samples:
    cache = file_cache[p['path']]
    if len(cache['neg_fps']) == 0: 
        continue
        
    # Calculate Euclidean distances to find the best matched control
    dists = cdist(
        scaler_fp.transform([p['target_fp']]), 
        scaler_fp.transform(cache['neg_fps']), 
        metric='euclidean'
    )[0]
    best_neg_t = cache['neg_times'][np.argmin(dists)]
    
    # Extract features for the matched negative sample
    feat_n = extract_hht_features_30s(
        cache['raw'].get_data(start=int(best_neg_t*cache['sf']), stop=int((best_neg_t+CONFIG['T_PRE'])*cache['sf'])), 
        cache['sf']
    )
    
    if feat_n is not None:
        # Append positive sample
        X_final.append(p['feat'])
        y_final.append(1)
        groups_final.append(p['f_idx'])
        
        # Append matched negative sample
        X_final.append(feat_n)
        y_final.append(0)
        groups_final.append(p['f_idx'])

# ==========================================
# 3. Statistical Validation & Machine Learning Model
# ==========================================
X_f = SimpleImputer(strategy='mean').fit_transform(X_final)
y_f, g_f = np.array(y_final), np.array(groups_final)

# Cross-validation Strategy
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

mean_fpr = np.linspace(0, 1, 100)
tprs = []
all_y_true, all_y_probs, all_y_pred = [], [], []
importances = np.zeros(6)

print("Training Random Forest Classifier & Generating ROC predictions...")
for tr, te in cv.split(X_f, y_f, groups=g_f):
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_f[tr], y_f[tr])
    probs = rf.predict_proba(X_f[te])[:, 1]
    
    fpr_fold, tpr_fold, _ = roc_curve(y_f[te], probs)
    tprs.append(np.interp(mean_fpr, fpr_fold, tpr_fold))
    tprs[-1][0] = 0.0
    
    all_y_true.extend(y_f[te])
    all_y_probs.extend(probs)
    all_y_pred.extend(rf.predict(X_f[te]))
    importances += rf.feature_importances_ / cv.get_n_splits()

print("Executing Permutation Test to evaluate significance...")
n_perm = 5000
perm_aucs = []

for _ in range(n_perm):
    y_perm = np.random.permutation(y_f)
    perm_fold_aucs = []
    for tr, te in cv.split(X_f, y_perm, groups=g_f):
        rf_p = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
        rf_p.fit(X_f[tr], y_perm[tr])
        p_p = rf_p.predict_proba(X_f[te])[:, 1]
        perm_fold_aucs.append(auc(*roc_curve(y_perm[te], p_p)[:2]))
    perm_aucs.append(np.mean(perm_fold_aucs))

actual_auc = auc(mean_fpr, np.mean(tprs, axis=0))
p_val = (np.sum(np.array(perm_aucs) >= actual_auc) + 1) / (n_perm + 1)

# ==========================================
# 4. High-Impact Publication Visualization
# ==========================================
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), dpi=300) 

# Minimalist color palette
main_color = '#2F4F4F'   # Dark Slate Gray
acc_color = '#B22222'    # Firebrick for accents
shadow_color = '#808080' # Grey for shadows

# --- Subplot 1: ROC Curve with SEM Shadow ---
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
std_error = sem(tprs, axis=0)

axes[0].plot(mean_fpr, mean_tpr, color=main_color, lw=0.5, label=f'Mean ROC (AUC = {actual_auc:.2f})')
axes[0].fill_between(mean_fpr, mean_tpr - std_error, mean_tpr + std_error, 
                     color=shadow_color, alpha=0.2, linewidth=0)
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, lw=0.5)
axes[0].set_title('Decoding Performance')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc='lower right', frameon=False)

# --- Subplot 2: Permutation Test Histogram ---
sns.histplot(perm_aucs, kde=False, ax=axes[1], color='lightgrey', alpha=0.6, 
             edgecolor='black', linewidth=0.5)
axes[1].axvline(actual_auc, color=acc_color, linestyle='--', lw=0.5, label=f'Actual (p={p_val:.3f})')
axes[1].set_title('Permutation Test')
axes[1].set_xlabel('AUC Value')
axes[1].set_ylabel('Frequency')
axes[1].legend(frameon=False)

# --- Subplot 3: Confusion Matrix ---
cm = confusion_matrix(all_y_true, all_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', ax=axes[2], cbar=False,
            xticklabels=['Non-Yawn', 'Yawn'], yticklabels=['Non-Yawn', 'Yawn'],
            linewidths=0.5, linecolor='black')
axes[2].set_title(f'Confusion Matrix\n(Acc: {accuracy_score(all_y_true, all_y_pred):.2%})')
axes[2].set_xlabel('Predicted Label')
axes[2].set_ylabel('True Label')

# --- Subplot 4: Feature Importance ---
idx_sort = np.argsort(importances)
axes[3].barh(np.array(all_feature_names)[idx_sort], importances[idx_sort], 
             color=main_color, alpha=0.6, edgecolor='black', linewidth=0.5)
axes[3].set_title('Feature Contribution')
axes[3].set_xlabel('Gini Importance')

plt.tight_layout()

# Save as high-quality PDF
pdf_path = os.path.join(BASE_SAVE_DIR, f"Cluster_{CLUSTER_ID}_Scientific_Report_Fine.pdf")
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.show()

print(f"Analysis successfully completed. Publication-quality figure saved to: {pdf_path}")
print(f"Statistical Significance (Permutation P-value): {p_val:.4f}")