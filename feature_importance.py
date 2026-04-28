"""
Feature importance and feature selection for EEG age-group classification.

Uses the same data and features as train_rf.py. Fits a Random Forest to get
feature_importances_, ranks features, and optionally selects top-K or threshold-based
features and reports CV weighted F1. Results and plot saved to results/.

Run from thesis folder: python feature_importance.py [--max N]

Usage:
  python feature_importance.py           # use all closed-eyes files
  python feature_importance.py --max 100 # limit for faster run
  python feature_importance.py --top 50 # only evaluate selection with top 50 (default: try 30, 50, 80)
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import mne
import numpy as np

warnings.filterwarnings("ignore", message=".*does not conform to MNE naming conventions.*")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eeg_features import extract_all_features, get_feature_names, feature_description, get_feature_indices_by_category

# Same config as train_rf
BASE = Path(".")
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
GROUPS = {
    "preschooler": "preschooler",
    "primary": "primary",
    "teenager": "teenager",
    "adolescence": "adolescence",
}
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
CLOSED_EYES_SUBSTRINGS = ("_zg", "_ZG", "_зг", "_ЗГ")
OPEN_EYES_SUBSTRINGS = ("_og", "_OG", "_ог", "_ОГ")


def find_edf_files(data_dir: Path, group_folder: str, max_per_group: int | None, eyes_condition: str = "closed") -> list[Path]:
    folder = data_dir / group_folder
    if not folder.exists():
        return []
    paths = sorted(folder.rglob("*.edf"))
    # Filter for eyes condition
    if eyes_condition == "closed":
        paths = [p for p in paths if any(s in p.name for s in CLOSED_EYES_SUBSTRINGS)]
    else:
        paths = [p for p in paths if any(s in p.name for s in OPEN_EYES_SUBSTRINGS)]
    if max_per_group is not None:
        paths = paths[:max_per_group]
    return paths


def load_features_and_labels(data_dir: Path, max_per_group: int | None, eyes_condition: str = "closed"):
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


def main():
    parser = argparse.ArgumentParser(description="Feature importance and selection (RF-based).")
    parser.add_argument("--max", type=int, default=None, help="Max samples per class")
    parser.add_argument("--top", type=int, default=None, help="Only run selection with this many top features (default: try 30, 50, 80)")
    parser.add_argument("--plot-top", type=int, default=50, help="Number of top features to show in bar plot")
    parser.add_argument("--eyes", type=str, default="closed", choices=["closed", "open"],
                        help="Eyes condition: 'closed' (default) or 'open'. Closed-eyes adds 3 alpha-rhythm topography features.")
    parser.add_argument("--features", type=str, default="all", 
                        help="Feature subset to use: 'all' (default), 'alpha', 'band_power', 'spectral', etc.")
    args = parser.parse_args()

    print("Loading data (same as train_rf)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, y, class_names = load_features_and_labels(DATA_DIR, args.max, eyes_condition=args.eyes)
    
    # Select feature subset
    feature_indices = get_feature_indices_by_category(eyes_condition=args.eyes)
    if args.features not in feature_indices:
        valid = list(feature_indices.keys())
        raise ValueError(f"Unknown feature subset '{args.features}'. Valid options: {valid}")
    
    selected_idx = feature_indices[args.features]
    X = X[:, selected_idx]
    
    # Get feature names for selected subset
    all_names = get_feature_names(eyes_condition=args.eyes)
    feature_names = [all_names[i] for i in selected_idx]
    
    print(f"Total samples: {X.shape[0]}, features: {X.shape[1]} (selected: {args.features})")
    print(feature_description(eyes_condition=args.eyes))

    # Balanced subset and split (same as train_rf)
    rng = np.random.default_rng(RANDOM_STATE)
    n_per_class = np.array([(y == i).sum() for i in range(len(class_names))])
    n_balance = int(n_per_class.min())
    balanced_idx = []
    for i in range(len(class_names)):
        idx = np.where(y == i)[0]
        balanced_idx.append(rng.choice(idx, size=n_balance, replace=False))
    balanced_idx = np.concatenate(balanced_idx)
    rng.shuffle(balanced_idx)
    X_bal, y_bal = X[balanced_idx], y[balanced_idx]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, _ = next(sss.split(X_bal, y_bal))
    X_train, y_train = X_bal[train_idx], y_bal[train_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Fit RF on full (scaled) train set to get importances
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, class_weight="balanced", random_state=RANDOM_STATE
    )
    rf.fit(X_train_scaled, y_train)
    importances = rf.feature_importances_
    names = feature_names  # Use already filtered feature names
    assert len(names) == len(importances), f"{len(names)} vs {len(importances)}"

    # Rank by importance
    order = np.argsort(importances)[::-1]
    ranked_names = [names[i] for i in order]
    ranked_imp = importances[order]

    # Save ranked table
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_txt = RESULTS_DIR / f"feature_importance_{timestamp}.txt"
    lines = [
        "Feature importance (Random Forest, balanced train set)",
        f"Total features: {len(names)}",
        "",
        "Rank\tImportance\tFeature",
        "-" * 60,
    ]
    for r, (name, imp) in enumerate(zip(ranked_names, ranked_imp), 1):
        lines.append(f"{r}\t{imp:.6f}\t{name}")
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"Ranked list saved: {out_txt}")

    # Plot top K
    K = min(args.plot_top, len(ranked_names))
    fig, ax = plt.subplots(figsize=(10, max(6, K * 0.15)))
    ax.barh(range(K), ranked_imp[:K], align="center")
    ax.set_yticks(range(K))
    ax.set_yticklabels(ranked_names[:K], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {K} features (Random Forest)")
    plt.tight_layout()
    plot_path = RESULTS_DIR / f"feature_importance_plot_{timestamp}.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot_path}")

    # Feature selection: CV weighted F1 with top-K features
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pipeline_full = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=20, class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    if args.top is not None:
        top_k_list = [args.top]
    else:
        # Try different K values based on available features
        n_feats = len(feature_names)
        top_k_list = [k for k in [30, 50, 80, 120, 200] if k < n_feats]
        top_k_list.append(n_feats)  # Always try all features

    selection_lines = [
        "",
        "Feature selection (5-fold CV weighted F1)",
        "Top K features (by importance) used for training.",
        "",
        "K\tWeighted F1",
        "-" * 30,
    ]
    best_k, best_f1 = None, -1.0
    for k in top_k_list:
        if k > len(order):
            k = len(order)
        sel_idx = order[:k]
        X_sel = X[:, sel_idx]  # unscaled; pipeline will scale inside each CV fold
        y_cv = cross_val_predict(pipeline_full, X_sel, y, cv=cv)
        f1 = f1_score(y, y_cv, average="weighted", zero_division=0)
        selection_lines.append(f"{k}\t{f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_k = k
    selection_lines.append("")
    selection_lines.append(f"Best: K={best_k}, weighted F1={best_f1:.4f}")

    # Append selection results to the same file
    with open(out_txt, "a", encoding="utf-8") as f:
        f.write("\n".join(selection_lines))

    # Save list of best-K feature names for use in training
    best_names_path = RESULTS_DIR / f"selected_features_top{best_k}_{timestamp}.txt"
    best_names_path.write_text(
        "\n".join(ranked_names[:best_k]),
        encoding="utf-8",
    )
    print(f"Best K={best_k} (weighted F1={best_f1:.4f}). Selected feature names: {best_names_path}")

    print("\nDone. Use the ranked list and selected_features_*.txt to reduce features in training if needed.")


if __name__ == "__main__":
    main()
