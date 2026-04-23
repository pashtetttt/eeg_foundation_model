"""
Feature visualization and analysis for EEG age-group classification.

Generates:
1. Feature correlation matrix (heatmap)
2. Mutual information between features and target
3. SHAP values for feature importance
4. Feature clustering dendrogram

Requirements:
  pip install seaborn shap scipy

Usage:
  python visualize_features.py --eyes closed          # use closed-eyes data
  python visualize_features.py --eyes open            # use open-eyes data
  python visualize_features.py --eyes closed --max 100  # limit samples
  python visualize_features.py --eyes closed --shap   # include SHAP analysis (slower)
  python visualize_features.py --eyes closed --all    # generate all visualizations
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Suppress MNE and other warnings before importing mne
warnings.filterwarnings("ignore", message=".*does not conform to MNE naming conventions.*")
warnings.filterwarnings("ignore", message=".*Physical range is not defined.*")
warnings.filterwarnings("ignore", message=".*Channels contain different.*filters.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from eeg_features import extract_all_features, get_feature_names, feature_description

# Check for optional dependencies
try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False
    print("⚠️  seaborn not installed. Install with: pip install seaborn")

try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    print("⚠️  scipy not installed. Install with: pip install scipy")

# Same config as train_rf
BASE = Path(".")
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
VIS_DIR = RESULTS_DIR / "visualizations"

GROUPS = {
    "дошкольники (124)": "preschooler",
    "младшие школьники (177)": "primary",
    "подростки (160)": "teenager",
    "юность(273)": "adolescence",
}

RANDOM_STATE = 42
CLOSED_EYES_SUBSTRINGS = ("_zg", "_ZG", "_зг", "_ЗГ")
OPEN_EYES_SUBSTRINGS = ("_og", "_OG", "_ог", "_ОГ")


def find_edf_files(data_dir: Path, group_folder: str, max_per_group: int | None, eyes_condition: str = "closed") -> list[Path]:
    folder = data_dir / group_folder
    if not folder.exists():
        return []
    paths = sorted(folder.rglob("*.edf"))
    if eyes_condition == "closed":
        paths = [p for p in paths if any(s in p.name for s in CLOSED_EYES_SUBSTRINGS)]
    else:
        paths = [p for p in paths if any(s in p.name for s in OPEN_EYES_SUBSTRINGS)]
    if max_per_group is not None:
        paths = paths[:max_per_group]
    return paths


def load_features_and_labels(data_dir: Path, max_per_group: int | None, eyes_condition: str = "closed"):
    """Load features and labels from EDF files."""
    import mne
    
    X_list, y_list = [], []
    class_names = list(GROUPS.values())
    canonical_ch_names = None
    
    for group_folder, label_name in GROUPS.items():
        paths = find_edf_files(data_dir, group_folder, max_per_group, eyes_condition)
        if not paths:
            continue
        label_idx = class_names.index(label_name)
        for p in paths:
            try:
                raw = mne.io.read_raw_edf(p, preload=True, verbose=False)
                data_check = raw.get_data()
                if not np.all(np.isfinite(data_check)):
                    continue
                out = extract_all_features(raw, canonical_ch_names, eyes_condition=eyes_condition)
                if canonical_ch_names is None:
                    feat, canonical_ch_names = out
                else:
                    feat = out
                X_list.append(feat)
                y_list.append(label_idx)
            except Exception:
                pass
    
    if not X_list:
        raise FileNotFoundError(f"No .edf files found under {data_dir}.")
    return np.array(X_list), np.array(y_list), class_names


def plot_correlation_matrix(X: np.ndarray, feature_names: list[str], timestamp: str, eyes_condition: str):
    """Plot and save feature correlation matrix heatmap."""
    if not _HAS_SEABORN:
        print("  ⚠️  Skipping correlation matrix: seaborn not installed")
        return
    
    print("  Computing correlation matrix...")
    
    # Compute correlation matrix (use subset of features if too many)
    max_features = 50
    if X.shape[1] > max_features:
        print(f"    Too many features ({X.shape[1]}), selecting top {max_features} by variance...")
        variances = np.var(X, axis=0)
        top_idx = np.argsort(variances)[::-1][:max_features]
        X_subset = X[:, top_idx]
        names_subset = [feature_names[i] for i in top_idx]
    else:
        X_subset = X
        names_subset = feature_names
    
    corr_matrix = np.corrcoef(X_subset.T)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        xticklabels=names_subset,
        yticklabels=names_subset,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Correlation"}
    )
    ax.set_title(f"Feature Correlation Matrix (top {len(names_subset)} features by variance)\nEyes: {eyes_condition}", fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    
    plot_path = VIS_DIR / f"correlation_matrix_{eyes_condition}_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Correlation matrix saved: {plot_path}")
    
    # Save correlation matrix as numpy array
    np.save(VIS_DIR / f"correlation_matrix_{eyes_condition}_{timestamp}.npy", corr_matrix)


def plot_mutual_information(X: np.ndarray, y: np.ndarray, feature_names: list[str], timestamp: str, eyes_condition: str, top_k: int = 30):
    """Plot and save mutual information between features and target."""
    print("  Computing mutual information...")
    
    # Compute mutual information
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE, n_neighbors=3)
    
    # Rank features
    order = np.argsort(mi_scores)[::-1]
    top_order = order[:top_k]
    top_names = [feature_names[i] for i in top_order]
    top_scores = mi_scores[top_order]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
    y_pos = np.arange(len(top_names))
    bars = ax.barh(y_pos, top_scores, align="center", color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mutual Information", fontsize=10)
    ax.set_title(f"Top {top_k} Features by Mutual Information\nEyes: {eyes_condition}", fontsize=12)
    
    # Add value labels
    for bar, score in zip(bars, top_scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{score:.4f}", va="center", fontsize=7)
    
    plt.tight_layout()
    
    plot_path = VIS_DIR / f"mutual_information_{eyes_condition}_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Mutual information plot saved: {plot_path}")
    
    # Save scores
    np.save(VIS_DIR / f"mi_scores_{eyes_condition}_{timestamp}.npy", mi_scores)
    
    return mi_scores


def plot_shap_values(X: np.ndarray, y: np.ndarray, feature_names: list[str], timestamp: str, eyes_condition: str, top_k: int = 30):
    """Plot and save SHAP values for feature importance."""
    print("  Computing SHAP values (this may take a while)...")
    
    try:
        import shap
    except ImportError:
        print("  ⚠️  shap package not installed. Install with: pip install shap")
        return
    
    # Train a model for SHAP analysis
    print("    Training Random Forest for SHAP...")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Use a subset for SHAP computation (faster)
    max_samples = 100
    if X.shape[0] > max_samples:
        print(f"    Using {max_samples} samples for SHAP computation...")
        rng = np.random.default_rng(RANDOM_STATE)
        X_subset = X[rng.choice(X.shape[0], max_samples, replace=False)]
    else:
        X_subset = X
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_subset)
    
    # Handle different SHAP output formats:
    # - Old versions: list of arrays (one per class) with shape (samples, features)
    # - New versions: single array with shape (samples, features, classes)
    if isinstance(shap_values, list):
        # Multi-class list format: average across classes
        shap_values_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        shap_values_summary = np.mean(shap_values_mean, axis=0)
    elif shap_values.ndim == 3:
        # New 3D array format (samples, features, classes)
        # Average absolute SHAP values across samples and classes
        shap_values_summary = np.mean(np.mean(np.abs(shap_values), axis=0), axis=1)
    else:
        # 2D array (samples, features) - binary classification or regression
        shap_values_summary = np.mean(np.abs(shap_values), axis=0)
    
    # Ensure 1D array
    shap_values_summary = np.asarray(shap_values_summary).flatten()
    
    # Rank features
    order = np.argsort(shap_values_summary)[::-1]
    top_order = order[:top_k]
    top_names = [feature_names[i] for i in top_order]
    top_scores = shap_values_summary[top_order]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
    y_pos = np.arange(len(top_names))
    bars = ax.barh(y_pos, top_scores, align="center", color="coral")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(f"Top {top_k} Features by SHAP Value\nEyes: {eyes_condition}", fontsize=12)
    
    # Add value labels
    for bar, score in zip(bars, top_scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{score:.4f}", va="center", fontsize=7)
    
    plt.tight_layout()
    
    plot_path = VIS_DIR / f"shap_values_{eyes_condition}_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP values plot saved: {plot_path}")
    
    # Also save SHAP summary plot (beeswarm-style)
    fig, ax = plt.subplots(figsize=(12, 8))
    if isinstance(shap_values, list):
        shap_for_plot = np.mean([sv for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        shap_for_plot = np.mean(shap_values, axis=2)  # Average across classes
    else:
        shap_for_plot = shap_values
    
    # Simplified summary plot
    shap_values_abs_mean = np.mean(np.abs(shap_for_plot), axis=0).flatten()
    top_20_idx = np.argsort(shap_values_abs_mean)[::-1][:20]
    
    # Create scatter plot
    colors = shap_for_plot[:, top_20_idx]
    for i, feat_idx in enumerate(top_20_idx):
        ax.scatter(colors[:, i], np.full(colors.shape[0], i), 
                   s=20, alpha=0.5, cmap="RdBu", c=colors[:, i])
    
    ax.set_yticks(range(len(top_20_idx)))
    ax.set_yticklabels([feature_names[i] for i in top_20_idx], fontsize=7)
    ax.set_xlabel("SHAP value", fontsize=10)
    ax.set_title(f"SHAP Summary Plot (top 20 features)\nEyes: {eyes_condition}", fontsize=12)
    ax.invert_yaxis()
    
    plt.tight_layout()
    shap_plot_path = VIS_DIR / f"shap_summary_{eyes_condition}_{timestamp}.png"
    plt.savefig(shap_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP summary plot saved: {shap_plot_path}")
    
    return shap_values_summary


def plot_feature_clustering(X: np.ndarray, feature_names: list[str], timestamp: str, eyes_condition: str, max_features: int = 50):
    """Plot and save feature clustering dendrogram."""
    if not _HAS_SCIPY:
        print("  ⚠️  Skipping feature clustering: scipy not installed")
        return
    if not _HAS_SEABORN:
        print("  ⚠️  Skipping correlation clustermap: seaborn not installed")
    
    print("  Computing feature clustering...")
    
    # Select top features by variance
    variances = np.var(X, axis=0)
    top_idx = np.argsort(variances)[::-1][:max_features]
    X_subset = X[:, top_idx]
    names_subset = [feature_names[i] for i in top_idx]
    
    # Compute distance matrix between features (based on correlation)
    corr_matrix = np.corrcoef(X_subset.T)
    # Ensure symmetry (floating point errors can cause asymmetry)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    # Convert correlation to distance: distance = 1 - |correlation|
    distance_matrix = 1 - np.abs(corr_matrix)
    # Ensure symmetry
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # Hierarchical clustering
    # Convert to condensed form for linkage
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method="average")
    
    # Plot dendrogram
    fig, ax = plt.subplots(figsize=(14, 10))
    dendrogram(
        linkage_matrix,
        labels=names_subset,
        orientation="left",
        leaf_font_size=7,
        ax=ax,
    )
    ax.set_title(f"Feature Clustering Dendrogram (top {max_features} features by variance)\nEyes: {eyes_condition}", fontsize=12)
    ax.set_xlabel("Distance", fontsize=10)
    plt.tight_layout()
    
    plot_path = VIS_DIR / f"feature_clustering_{eyes_condition}_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature clustering saved: {plot_path}")
    
    # Also plot correlation matrix with clustering
    if _HAS_SEABORN:
        g = sns.clustermap(
            corr_matrix,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            figsize=(12, 10),
            cbar_kws={"label": "Correlation"},
            xticklabels=names_subset,
            yticklabels=names_subset,
        )
        g.fig.suptitle(f"Feature Correlation with Clustering\nEyes: {eyes_condition}", fontsize=12, y=1.02)
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)
        
        clustermap_path = VIS_DIR / f"correlation_clustermap_{eyes_condition}_{timestamp}.png"
        plt.savefig(clustermap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Correlation clustermap saved: {clustermap_path}")


def main():
    parser = argparse.ArgumentParser(description="Feature visualization and analysis.")
    parser.add_argument("--max", type=int, default=None, help="Max samples per class")
    parser.add_argument("--eyes", type=str, default="closed", choices=["closed", "open"],
                        help="Eyes condition: 'closed' (default) or 'open'")
    parser.add_argument("--correlation", action="store_true", help="Generate correlation matrix")
    parser.add_argument("--mi", action="store_true", help="Generate mutual information plot")
    parser.add_argument("--shap", action="store_true", help="Generate SHAP values plot")
    parser.add_argument("--clustering", action="store_true", help="Generate feature clustering")
    parser.add_argument("--all", action="store_true", help="Generate all visualizations")
    args = parser.parse_args()
    
    # Determine which plots to generate
    generate_all = args.all or not (args.correlation or args.mi or args.shap or args.clustering)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Feature Visualization")
    print(f"Eyes condition: {args.eyes}")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)
    
    print("\nLoading data...")
    X, y, class_names = load_features_and_labels(DATA_DIR, max_per_group=args.max, eyes_condition=args.eyes)
    feature_names = get_feature_names(eyes_condition=args.eyes)
    
    print(f"Total samples: {X.shape[0]}, features: {X.shape[1]}")
    print(feature_description(eyes_condition=args.eyes))
    
    # Scale features for better visualization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\nGenerating visualizations...")
    
    if generate_all or args.correlation:
        plot_correlation_matrix(X_scaled, feature_names, timestamp, args.eyes)
    
    if generate_all or args.mi:
        plot_mutual_information(X_scaled, y, feature_names, timestamp, args.eyes)
    
    if generate_all or args.shap:
        plot_shap_values(X_scaled, y, feature_names, timestamp, args.eyes)
    
    if generate_all or args.clustering:
        plot_feature_clustering(X_scaled, feature_names, timestamp, args.eyes)
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {VIS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
