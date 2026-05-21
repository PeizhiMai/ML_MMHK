#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/Gcalpath/rho_re_g"
DEFAULT_FEATURES = BASE / "ml_physics_features_eta0p1_nonnegative_20260507/physics_features.npz"
DEFAULT_GAP_CSV = BASE / "gap_analysis_eta0p1_nonnegative_20260507/nonnegative_v_gap_classification_maxk_rho0.csv"
DEFAULT_LUTTINGER = (
    BASE / "ml_luttinger_path_reG0_eta0p1_nonnegative_gaplt0p01_20260507/luttinger_path_features.npz"
)
DEFAULT_BASELINE = BASE / "ml_baseline_eta0p1_20260504/ml_ready_dataset.npz"
DEFAULT_OUTPUT = BASE / "ml_unsupervised_kernel_eta0p1_nonnegative_20260507"

RANDOM_STATE = 7
GAP_OPEN_THRESHOLD = 0.10
STRONG_GAP_THRESHOLD = 0.01
DEEP_GAP_THRESHOLD = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kernel-PCA, spectral-clustering, One-Class-SVM, and bootstrap-stability "
            "refinement for eta=0.1 V>=0 Green-function data."
        )
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--gap-csv", type=Path, default=DEFAULT_GAP_CSV)
    parser.add_argument("--luttinger-features", type=Path, default=DEFAULT_LUTTINGER)
    parser.add_argument("--baseline-dataset", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--analysis-v-min", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--n-bootstrap", type=int, default=40)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.80)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def uv_key(uv: np.ndarray | tuple[float, float] | list[float]) -> tuple[float, float]:
    return (round(float(uv[0]), 3), round(float(uv[1]), 3))


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_ready(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_features(path: Path, v_min: float) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    uv = np.asarray(z["uv"], dtype=float)
    features = np.asarray(z["features"], dtype=float)
    names = [str(x) for x in z["feature_names"]]
    mask = uv[:, 1] >= v_min - 1.0e-12
    uv = uv[mask]
    features = np.nan_to_num(features[mask], nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(uv[:, 1] < v_min - 1.0e-12):
        raise ValueError("Feature file contains samples below requested V minimum after filtering.")
    return {"uv": uv, "features": features, "feature_names": names}


def load_gap(path: Path, v_min: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["V"] >= v_min - 1.0e-12].copy()
    df["U_key"] = df["U"].round(3)
    df["V_key"] = df["V"].round(3)
    df["gap_open"] = df["maxk_rho0_ratio"] < GAP_OPEN_THRESHOLD
    df["strong_gap"] = df["maxk_rho0_ratio"] < STRONG_GAP_THRESHOLD
    df["deep_gap"] = df["maxk_rho0_ratio"] < DEEP_GAP_THRESHOLD
    return df


def annotations_for_uv(uv: np.ndarray, gap: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame({"U": uv[:, 0], "V": uv[:, 1]})
    base["U_key"] = base["U"].round(3)
    base["V_key"] = base["V"].round(3)
    merged = base.merge(
        gap[
            [
                "U_key",
                "V_key",
                "maxk_rho0_ratio",
                "gap_open",
                "strong_gap",
                "deep_gap",
            ]
        ],
        on=["U_key", "V_key"],
        how="left",
        validate="one_to_one",
    )
    if merged["maxk_rho0_ratio"].isna().any():
        missing = merged[merged["maxk_rho0_ratio"].isna()][["U", "V"]]
        raise ValueError(f"Gap table missing rows for {missing.to_dict(orient='records')[:5]}")
    return merged.drop(columns=["U_key", "V_key"])


def feature_block_indices(names: list[str], block: str) -> np.ndarray:
    if block == "rho":
        idx = [i for i, name in enumerate(names) if name.startswith("rho_")]
    elif block == "reG":
        idx = [i for i, name in enumerate(names) if name.startswith("reG_")]
    elif block == "combined":
        idx = list(range(len(names)))
    else:
        raise ValueError(block)
    return np.asarray(idx, dtype=int)


def median_rbf_gamma(x_scaled: np.ndarray) -> float:
    dist = pairwise_distances(x_scaled, metric="sqeuclidean")
    tri = dist[np.triu_indices_from(dist, k=1)]
    tri = tri[np.isfinite(tri) & (tri > 0)]
    if tri.size == 0:
        return 1.0
    return float(1.0 / (2.0 * np.median(tri)))


def safe_cluster_metrics(x_scaled: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    unique = np.unique(labels)
    if unique.size < 2 or unique.size >= labels.size:
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    return {
        "silhouette": float(silhouette_score(x_scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(x_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(x_scaled, labels)),
    }


def fit_kernel_pca(x_scaled: np.ndarray, gamma: float, random_state: int) -> np.ndarray:
    n_components = min(6, x_scaled.shape[0] - 1, x_scaled.shape[1])
    if n_components < 2:
        raise ValueError("Kernel PCA needs at least two components.")
    model = KernelPCA(
        n_components=n_components,
        kernel="rbf",
        gamma=gamma,
        eigen_solver="arpack",
        random_state=random_state,
        remove_zero_eig=True,
    )
    coords = model.fit_transform(x_scaled)
    return np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)


def bootstrap_label_stability(
    x_scaled: np.ndarray,
    base_labels: np.ndarray,
    method: str,
    k: int,
    gamma: float,
    random_state: int,
    n_bootstrap: int,
    fraction: float,
) -> tuple[float | None, float | None]:
    if x_scaled.shape[0] < 8:
        return None, None
    rng = np.random.default_rng(random_state + 1009)
    n_sub = max(k + 2, int(round(fraction * x_scaled.shape[0])))
    n_sub = min(n_sub, x_scaled.shape[0] - 1)
    scores: list[float] = []
    for boot_idx in range(n_bootstrap):
        subset = np.sort(rng.choice(x_scaled.shape[0], size=n_sub, replace=False))
        try:
            if method == "spectral":
                labels = SpectralClustering(
                    n_clusters=k,
                    affinity="rbf",
                    gamma=gamma,
                    assign_labels="kmeans",
                    random_state=random_state + boot_idx,
                    n_init=20,
                ).fit_predict(x_scaled[subset])
            elif method == "kernel_pca_kmeans":
                coords = fit_kernel_pca(x_scaled[subset], gamma, random_state + boot_idx)
                labels = KMeans(n_clusters=k, n_init=40, random_state=random_state + boot_idx).fit_predict(
                    coords[:, : min(4, coords.shape[1])]
                )
            else:
                raise ValueError(method)
            if np.unique(labels).size >= 2:
                scores.append(float(adjusted_rand_score(base_labels[subset], labels)))
        except Exception:
            continue
    if not scores:
        return None, None
    return float(np.mean(scores)), float(np.std(scores))


def cluster_quality_score(row: dict[str, Any]) -> float:
    sil = row.get("silhouette")
    boot = row.get("bootstrap_ari_mean")
    sil_value = float(sil) if sil is not None and np.isfinite(sil) else -1.0
    boot_value = float(boot) if boot is not None and np.isfinite(boot) else -1.0
    return 0.55 * sil_value + 0.45 * boot_value


def build_raw_pca_representations(
    baseline_path: Path,
    uv_target: np.ndarray,
    v_min: float,
    random_state: int,
    n_components: int = 10,
    windows: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> dict[str, np.ndarray]:
    z = np.load(baseline_path, allow_pickle=True)
    uv_all = np.asarray(z["uv"], dtype=float)
    omega = np.asarray(z["omega"], dtype=float)
    rho = np.asarray(z["rho_G"], dtype=float)
    re_g = np.asarray(z["reG"], dtype=float)
    keep = uv_all[:, 1] >= v_min - 1.0e-12
    uv_all = uv_all[keep]
    rho = rho[keep]
    re_g = re_g[keep]
    key_to_index = {uv_key(row): idx for idx, row in enumerate(uv_all)}
    order = np.asarray([key_to_index[uv_key(row)] for row in uv_target], dtype=int)
    rho = rho[order]
    re_g = re_g[order]

    reps: dict[str, np.ndarray] = {}
    for window in windows:
        mask = np.abs(omega) <= window
        suffix = f"w{str(window).replace('.', 'p')}"
        blocks = {
            f"raw_rho_low_{suffix}_pca": rho[:, :, mask],
            f"raw_reG_low_{suffix}_pca": re_g[:, :, mask],
        }
        blocks[f"raw_combined_low_{suffix}_pca"] = np.concatenate(
            [
                blocks[f"raw_rho_low_{suffix}_pca"].reshape(uv_target.shape[0], -1),
                blocks[f"raw_reG_low_{suffix}_pca"].reshape(uv_target.shape[0], -1),
            ],
            axis=1,
        )
        for name, block in blocks.items():
            flat = block.reshape(uv_target.shape[0], -1)
            flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
            flat_scaled = StandardScaler().fit_transform(flat)
            n_comp = min(n_components, flat_scaled.shape[0] - 1, flat_scaled.shape[1])
            pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=random_state)
            reps[name] = pca.fit_transform(flat_scaled)
    return reps


def load_luttinger_table(path: Path) -> dict[tuple[float, float], np.ndarray]:
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=True)
    uv = np.asarray(z["uv"], dtype=float)
    features = np.asarray(z["features"], dtype=float)
    return {uv_key(row): np.asarray(features[idx], dtype=float) for idx, row in enumerate(uv)}


def luttinger_representation_for_uv(
    table: dict[tuple[float, float], np.ndarray],
    uv_target: np.ndarray,
) -> np.ndarray | None:
    if not table:
        return None
    rows: list[np.ndarray] = []
    for row in uv_target:
        values = table.get(uv_key(row))
        if values is None:
            return None
        rows.append(values)
    return np.nan_to_num(np.vstack(rows), nan=0.0, posinf=0.0, neginf=0.0)


def make_representations(
    features: dict[str, Any],
    raw_reps: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    names = features["feature_names"]
    x = features["features"]
    reps = {
        "rho": x[:, feature_block_indices(names, "rho")],
        "reG": x[:, feature_block_indices(names, "reG")],
        "combined": x,
    }
    reps.update(raw_reps)
    return {name: np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) for name, value in reps.items()}


def run_cluster_refinement(
    subset_name: str,
    uv: np.ndarray,
    annotations: pd.DataFrame,
    representation_name: str,
    x: np.ndarray,
    random_state: int,
    n_bootstrap: int,
    bootstrap_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    x_scaled = StandardScaler().fit_transform(x)
    gamma0 = median_rbf_gamma(x_scaled)
    gamma_factors = (0.5, 1.0, 2.0)
    k_values = range(2, min(5, x_scaled.shape[0] - 1))
    metrics_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    embeddings: dict[str, np.ndarray] = {}

    for gamma_factor in gamma_factors:
        gamma = gamma0 * gamma_factor
        try:
            kpca_coords = fit_kernel_pca(x_scaled, gamma, random_state)
            emb_key = f"{subset_name}__{representation_name}__rbf_gamma_factor_{gamma_factor:g}"
            embeddings[emb_key] = kpca_coords
            for k in k_values:
                labels = KMeans(n_clusters=k, n_init=60, random_state=random_state).fit_predict(
                    kpca_coords[:, : min(4, kpca_coords.shape[1])]
                )
                ari_mean, ari_std = bootstrap_label_stability(
                    x_scaled,
                    labels,
                    "kernel_pca_kmeans",
                    k,
                    gamma,
                    random_state,
                    n_bootstrap,
                    bootstrap_fraction,
                )
                row = {
                    "subset": subset_name,
                    "representation": representation_name,
                    "method": "kernel_pca_kmeans",
                    "kernel": "rbf",
                    "gamma_factor": gamma_factor,
                    "gamma": gamma,
                    "k": k,
                    "n_samples": int(x_scaled.shape[0]),
                    **safe_cluster_metrics(x_scaled, labels),
                    "bootstrap_ari_mean": ari_mean,
                    "bootstrap_ari_std": ari_std,
                }
                row["quality_score"] = cluster_quality_score(row)
                metrics_rows.append(row)
                label_rows.extend(labels_to_rows(subset_name, representation_name, row, uv, annotations, labels))
        except Exception as exc:
            metrics_rows.append(
                {
                    "subset": subset_name,
                    "representation": representation_name,
                    "method": "kernel_pca_kmeans",
                    "kernel": "rbf",
                    "gamma_factor": gamma_factor,
                    "gamma": gamma,
                    "k": "",
                    "n_samples": int(x_scaled.shape[0]),
                    "error": str(exc),
                    "quality_score": -1.0,
                }
            )

        for k in k_values:
            try:
                labels = SpectralClustering(
                    n_clusters=k,
                    affinity="rbf",
                    gamma=gamma,
                    assign_labels="kmeans",
                    random_state=random_state,
                    n_init=30,
                ).fit_predict(x_scaled)
                ari_mean, ari_std = bootstrap_label_stability(
                    x_scaled,
                    labels,
                    "spectral",
                    k,
                    gamma,
                    random_state,
                    n_bootstrap,
                    bootstrap_fraction,
                )
                row = {
                    "subset": subset_name,
                    "representation": representation_name,
                    "method": "spectral_clustering",
                    "kernel": "rbf",
                    "gamma_factor": gamma_factor,
                    "gamma": gamma,
                    "k": k,
                    "n_samples": int(x_scaled.shape[0]),
                    **safe_cluster_metrics(x_scaled, labels),
                    "bootstrap_ari_mean": ari_mean,
                    "bootstrap_ari_std": ari_std,
                }
                row["quality_score"] = cluster_quality_score(row)
                metrics_rows.append(row)
                label_rows.extend(labels_to_rows(subset_name, representation_name, row, uv, annotations, labels))
            except Exception as exc:
                metrics_rows.append(
                    {
                        "subset": subset_name,
                        "representation": representation_name,
                        "method": "spectral_clustering",
                        "kernel": "rbf",
                        "gamma_factor": gamma_factor,
                        "gamma": gamma,
                        "k": k,
                        "n_samples": int(x_scaled.shape[0]),
                        "error": str(exc),
                        "quality_score": -1.0,
                    }
                )

    return metrics_rows, label_rows, embeddings


def labels_to_rows(
    subset_name: str,
    representation_name: str,
    metric_row: dict[str, Any],
    uv: np.ndarray,
    annotations: pd.DataFrame,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    label_name = (
        f"{metric_row['method']}_k{metric_row['k']}_"
        f"gamma{str(metric_row['gamma_factor']).replace('.', 'p')}"
    )
    rows: list[dict[str, Any]] = []
    for idx, (target, label) in enumerate(zip(uv, labels)):
        ann = annotations.iloc[idx]
        rows.append(
            {
                "subset": subset_name,
                "representation": representation_name,
                "label_name": label_name,
                "method": metric_row["method"],
                "gamma_factor": metric_row["gamma_factor"],
                "k": metric_row["k"],
                "sample_index": idx,
                "U": float(target[0]),
                "V": float(target[1]),
                "cluster": int(label),
                "maxk_rho0_ratio": float(ann["maxk_rho0_ratio"]),
                "gap_open": bool(ann["gap_open"]),
                "strong_gap": bool(ann["strong_gap"]),
                "deep_gap": bool(ann["deep_gap"]),
            }
        )
    return rows


def run_oneclass_svm(
    subset_name: str,
    uv: np.ndarray,
    annotations: pd.DataFrame,
    representation_name: str,
    x: np.ndarray,
    random_state: int,
) -> list[dict[str, Any]]:
    del random_state
    x_scaled = StandardScaler().fit_transform(x)
    gamma0 = median_rbf_gamma(x_scaled)
    rows: list[dict[str, Any]] = []
    for gamma_factor in (0.5, 1.0, 2.0):
        gamma = gamma0 * gamma_factor
        for nu in (0.05, 0.10, 0.15, 0.20):
            model = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
            pred = model.fit_predict(x_scaled)
            score = model.decision_function(x_scaled).ravel()
            for idx, (target, label, value) in enumerate(zip(uv, pred, score)):
                ann = annotations.iloc[idx]
                rows.append(
                    {
                        "subset": subset_name,
                        "representation": representation_name,
                        "gamma_factor": gamma_factor,
                        "gamma": gamma,
                        "nu": nu,
                        "sample_index": idx,
                        "U": float(target[0]),
                        "V": float(target[1]),
                        "oneclass_label": int(label),
                        "oneclass_score": float(value),
                        "maxk_rho0_ratio": float(ann["maxk_rho0_ratio"]),
                        "gap_open": bool(ann["gap_open"]),
                        "strong_gap": bool(ann["strong_gap"]),
                        "deep_gap": bool(ann["deep_gap"]),
                    }
                )
    return rows


def summarize_best_clusters(
    metrics: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    good = [row for row in metrics if row.get("quality_score") is not None and row.get("error", "") == ""]
    best_by_subset_rep_method: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in good:
        key = (str(row["subset"]), str(row["representation"]), str(row["method"]))
        old = best_by_subset_rep_method.get(key)
        if old is None or float(row["quality_score"]) > float(old["quality_score"]):
            best_by_subset_rep_method[key] = row
    label_df = pd.DataFrame(labels)
    summaries: list[dict[str, Any]] = []
    for row in best_by_subset_rep_method.values():
        label_name = (
            f"{row['method']}_k{row['k']}_"
            f"gamma{str(row['gamma_factor']).replace('.', 'p')}"
        )
        selected = label_df[
            (label_df["subset"] == row["subset"])
            & (label_df["representation"] == row["representation"])
            & (label_df["label_name"] == label_name)
        ]
        for cluster, group in selected.groupby("cluster"):
            summaries.append(
                {
                    "subset": row["subset"],
                    "representation": row["representation"],
                    "method": row["method"],
                    "label_name": label_name,
                    "k": row["k"],
                    "cluster": int(cluster),
                    "count": int(len(group)),
                    "U_min": float(group["U"].min()),
                    "U_median": float(group["U"].median()),
                    "U_max": float(group["U"].max()),
                    "V_min": float(group["V"].min()),
                    "V_median": float(group["V"].median()),
                    "V_max": float(group["V"].max()),
                    "gap_ratio_median": float(group["maxk_rho0_ratio"].median()),
                    "deep_gap_fraction": float(group["deep_gap"].mean()),
                    "quality_score": row["quality_score"],
                    "silhouette": row.get("silhouette"),
                    "bootstrap_ari_mean": row.get("bootstrap_ari_mean"),
                }
            )
    return summaries


def select_best_metric(
    metrics: list[dict[str, Any]],
    subset: str,
    representation: str,
    method: str,
) -> dict[str, Any] | None:
    rows = [
        row
        for row in metrics
        if row.get("subset") == subset
        and row.get("representation") == representation
        and row.get("method") == method
        and row.get("error", "") == ""
        and row.get("silhouette") is not None
    ]
    if not rows:
        return None
    return max(rows, key=cluster_quality_score)


def plot_cluster_uv(
    out: Path,
    title: str,
    labels: pd.DataFrame,
    subset: str,
    representation: str,
    metric: dict[str, Any],
) -> None:
    label_name = (
        f"{metric['method']}_k{metric['k']}_"
        f"gamma{str(metric['gamma_factor']).replace('.', 'p')}"
    )
    selected = labels[
        (labels["subset"] == subset)
        & (labels["representation"] == representation)
        & (labels["label_name"] == label_name)
    ]
    if selected.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    palette = ["#2b83ba", "#d73027", "#1a9850", "#7b3294", "#fdae61", "#4d4d4d"]
    markers = ["o", "s", "^", "D", "P", "X"]
    for idx, (cluster, group) in enumerate(selected.groupby("cluster")):
        ax.scatter(
            group["U"],
            group["V"],
            s=58,
            color=palette[idx % len(palette)],
            marker=markers[idx % len(markers)],
            edgecolor="white",
            linewidth=0.65,
            alpha=0.95,
            label=f"cluster {int(cluster)}: n={len(group)}",
        )
    ax.set_xlabel("U")
    ax.set_ylabel("V")
    ax.set_title(title)
    ax.grid(True, color="#d9dee5", linewidth=0.7, alpha=0.85)
    ax.legend(frameon=True, fontsize=8, loc="upper left")
    ax.text(
        0.985,
        0.025,
        f"sil={metric.get('silhouette'):.3f}, boot={metric.get('bootstrap_ari_mean'):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_oneclass_uv(out: Path, title: str, scores: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    sc = ax.scatter(
        scores["U"],
        scores["V"],
        c=scores["oneclass_score"],
        cmap="viridis",
        s=58,
        edgecolor="white",
        linewidth=0.65,
    )
    outliers = scores[scores["oneclass_label"] < 0]
    if not outliers.empty:
        ax.scatter(
            outliers["U"],
            outliers["V"],
            facecolor="none",
            edgecolor="#d73027",
            linewidth=1.2,
            s=110,
            label="One-Class SVM outlier",
        )
        ax.legend(frameon=True, fontsize=8, loc="upper left")
    ax.set_xlabel("U")
    ax.set_ylabel("V")
    ax.set_title(title)
    ax.grid(True, color="#d9dee5", linewidth=0.7, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="decision score")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def midpoint_candidates(
    rows: pd.DataFrame,
    label_col: str,
    source: str,
    tier: int,
    max_du: float,
    max_dv: float,
    weight: float,
    existing: set[tuple[float, float]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    values = rows.reset_index(drop=True)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            a = values.iloc[i]
            b = values.iloc[j]
            if bool(a[label_col]) == bool(b[label_col]):
                continue
            du = abs(float(a["U"]) - float(b["U"]))
            dv = abs(float(a["V"]) - float(b["V"]))
            if du > max_du or dv > max_dv:
                continue
            u = round(0.5 * (float(a["U"]) + float(b["U"])), 3)
            v = round(0.5 * (float(a["V"]) + float(b["V"])), 3)
            if v < 0 or (u, v) in existing:
                continue
            dist = math.sqrt((du / max(max_du, 1.0e-9)) ** 2 + (dv / max(max_dv, 1.0e-9)) ** 2)
            ratio_mid = math.sqrt(float(a["maxk_rho0_ratio"]) * float(b["maxk_rho0_ratio"]))
            candidates.append(
                {
                    "U": u,
                    "V": v,
                    "tier": tier,
                    "source": source,
                    "score": weight / max(dist, 1.0e-6),
                    "nearest_ratio_geomean": ratio_mid,
                    "neighbor_a": f"({float(a['U']):.3f},{float(a['V']):.3f})",
                    "neighbor_b": f"({float(b['U']):.3f},{float(b['V']):.3f})",
                    "reason": source,
                }
            )
    return candidates


def cluster_boundary_candidates(
    label_rows: pd.DataFrame,
    subset: str,
    representation: str,
    metric: dict[str, Any],
    existing: set[tuple[float, float]],
) -> list[dict[str, Any]]:
    label_name = (
        f"{metric['method']}_k{metric['k']}_"
        f"gamma{str(metric['gamma_factor']).replace('.', 'p')}"
    )
    selected = label_rows[
        (label_rows["subset"] == subset)
        & (label_rows["representation"] == representation)
        & (label_rows["label_name"] == label_name)
    ].copy()
    candidates: list[dict[str, Any]] = []
    values = selected.reset_index(drop=True)
    for i in range(len(values)):
        best_j = None
        best_dist = None
        for j in range(len(values)):
            if int(values.iloc[i]["cluster"]) == int(values.iloc[j]["cluster"]):
                continue
            du = float(values.iloc[i]["U"]) - float(values.iloc[j]["U"])
            dv = float(values.iloc[i]["V"]) - float(values.iloc[j]["V"])
            dist = math.sqrt((du / 2.0) ** 2 + (dv / 0.75) ** 2)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is None or best_dist is None or best_dist > 1.4:
            continue
        a = values.iloc[i]
        b = values.iloc[best_j]
        u = round(0.5 * (float(a["U"]) + float(b["U"])), 3)
        v = round(0.5 * (float(a["V"]) + float(b["V"])), 3)
        if v < 0 or (u, v) in existing:
            continue
        candidates.append(
            {
                "U": u,
                "V": v,
                "tier": 1,
                "source": "strong_gap_cluster_boundary",
                "score": 4.0 / max(best_dist, 1.0e-6),
                "nearest_ratio_geomean": math.sqrt(
                    float(a["maxk_rho0_ratio"]) * float(b["maxk_rho0_ratio"])
                ),
                "neighbor_a": f"({float(a['U']):.3f},{float(a['V']):.3f})",
                "neighbor_b": f"({float(b['U']):.3f},{float(b['V']):.3f})",
                "reason": "nearest midpoint between strong-gap spectral clusters",
            }
        )
    return candidates


def oneclass_candidates(scores: pd.DataFrame, existing: set[tuple[float, float]]) -> list[dict[str, Any]]:
    selected = scores[
        (scores["gamma_factor"] == 1.0)
        & (np.isclose(scores["nu"], 0.10))
        & (scores["representation"].isin(["raw_rho_low_w2p0_pca", "raw_reG_low_w1p0_pca"]))
    ].copy()
    selected = selected.sort_values("oneclass_score").head(20)
    offsets = [(0.25, 0.0), (-0.25, 0.0), (0.0, 0.125), (0.0, -0.125)]
    candidates: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        for du, dv in offsets:
            u = round(float(row["U"]) + du, 3)
            v = round(float(row["V"]) + dv, 3)
            if u < 0.5 or u > 12.0 or v < 0.0 or v > 4.0 or (u, v) in existing:
                continue
            candidates.append(
                {
                    "U": u,
                    "V": v,
                    "tier": 2,
                    "source": "oneclass_svm_boundary",
                    "score": float(-row["oneclass_score"]) + 1.0,
                    "nearest_ratio_geomean": float(row["maxk_rho0_ratio"]),
                    "neighbor_a": f"({float(row['U']):.3f},{float(row['V']):.3f})",
                    "neighbor_b": "",
                    "reason": f"local refinement around low-density/boundary sample in {row['subset']}",
                }
            )
    return candidates


def consolidate_candidates(candidates: list[dict[str, Any]], max_rows: int = 36) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], dict[str, Any]] = {}
    for row in candidates:
        key = (float(row["U"]), float(row["V"]))
        if key not in grouped:
            grouped[key] = dict(row)
            grouped[key]["support_count"] = 1
        else:
            old = grouped[key]
            old["score"] = float(old["score"]) + float(row["score"])
            old["tier"] = min(int(old["tier"]), int(row["tier"]))
            old["source"] = f"{old['source']};{row['source']}"
            old["reason"] = f"{old['reason']}; {row['reason']}"
            old["support_count"] = int(old["support_count"]) + 1
    rows = list(grouped.values())
    rows.sort(key=lambda r: (int(r["tier"]), -float(r["score"]), float(r["U"]), float(r["V"])))
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows[:max_rows], start=1):
        row = dict(row)
        row["rank"] = idx
        row["U"] = f"{float(row['U']):.3f}"
        row["V"] = f"{float(row['V']):.3f}"
        row["score"] = f"{float(row['score']):.6g}"
        row["nearest_ratio_geomean"] = f"{float(row['nearest_ratio_geomean']):.6g}"
        out.append(row)
    return out


def make_recommendations(
    annotations: pd.DataFrame,
    labels: pd.DataFrame,
    metrics: list[dict[str, Any]],
    oneclass: pd.DataFrame,
    existing_uv: np.ndarray,
) -> list[dict[str, Any]]:
    existing = {uv_key(row) for row in existing_uv}
    candidates: list[dict[str, Any]] = []
    candidates.extend(
        midpoint_candidates(
            annotations,
            "gap_open",
            "gap_open_threshold_R0_0p10",
            tier=1,
            max_du=1.0,
            max_dv=0.75,
            weight=3.0,
            existing=existing,
        )
    )
    gap_rows = annotations[annotations["gap_open"]].copy()
    candidates.extend(
        midpoint_candidates(
            gap_rows,
            "strong_gap",
            "strong_gap_threshold_R0_0p01",
            tier=1,
            max_du=1.0,
            max_dv=0.75,
            weight=3.5,
            existing=existing,
        )
    )
    strong_rows = annotations[annotations["strong_gap"]].copy()
    candidates.extend(
        midpoint_candidates(
            strong_rows,
            "deep_gap",
            "deep_gap_threshold_R0_0p001",
            tier=2,
            max_du=1.0,
            max_dv=0.75,
            weight=2.5,
            existing=existing,
        )
    )
    best_strong = select_best_metric(
        metrics,
        "nonnegative_v_strong_gap",
        "raw_reG_low_w1p0_pca",
        "spectral_clustering",
    )
    if best_strong is not None:
        candidates.extend(
            cluster_boundary_candidates(
                labels,
                "nonnegative_v_strong_gap",
                "raw_reG_low_w1p0_pca",
                best_strong,
                existing,
            )
        )
    candidates.extend(oneclass_candidates(oneclass, existing))
    return consolidate_candidates(candidates)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    if not args.no_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_features(args.features, args.analysis_v_min)
    gap = load_gap(args.gap_csv, args.analysis_v_min)
    annotations = annotations_for_uv(loaded["uv"], gap)
    if np.any(loaded["uv"][:, 1] < args.analysis_v_min - 1.0e-12):
        raise ValueError("This refinement pass must not include V below the requested minimum.")

    raw_reps = build_raw_pca_representations(
        args.baseline_dataset,
        loaded["uv"],
        args.analysis_v_min,
        args.random_state,
    )
    luttinger_table = load_luttinger_table(args.luttinger_features)
    reps_all = make_representations(loaded, raw_reps)

    subset_masks = {
        "nonnegative_v": np.ones(loaded["uv"].shape[0], dtype=bool),
        "nonnegative_v_gap_open": annotations["gap_open"].to_numpy(dtype=bool),
        "nonnegative_v_strong_gap": annotations["strong_gap"].to_numpy(dtype=bool),
    }
    representation_plan = {
        "nonnegative_v": ["combined", "raw_rho_low_w2p0_pca", "raw_reG_low_w1p0_pca"],
        "nonnegative_v_gap_open": [
            "combined",
            "raw_rho_low_w2p0_pca",
            "raw_reG_low_w1p0_pca",
            "raw_combined_low_w1p0_pca",
        ],
        "nonnegative_v_strong_gap": [
            "combined",
            "raw_reG_low_w1p0_pca",
            "raw_reG_low_w0p5_pca",
            "raw_rho_low_w2p0_pca",
            "luttinger_geometry",
        ],
    }

    metrics_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    oneclass_rows: list[dict[str, Any]] = []
    embeddings: dict[str, np.ndarray] = {}

    for subset_name, mask in subset_masks.items():
        uv = loaded["uv"][mask]
        ann = annotations[mask].reset_index(drop=True)
        for representation_name in representation_plan[subset_name]:
            if representation_name == "luttinger_geometry":
                x = luttinger_representation_for_uv(luttinger_table, uv)
                if x is None:
                    continue
            elif representation_name in reps_all:
                x = reps_all[representation_name][mask]
            else:
                continue
            if x.shape[0] < 8 or x.shape[1] < 2:
                continue
            metrics, labels, emb = run_cluster_refinement(
                subset_name,
                uv,
                ann,
                representation_name,
                x,
                args.random_state,
                args.n_bootstrap,
                args.bootstrap_fraction,
            )
            metrics_rows.extend(metrics)
            label_rows.extend(labels)
            embeddings.update(emb)
            if subset_name in {"nonnegative_v_gap_open", "nonnegative_v_strong_gap"}:
                oneclass_rows.extend(
                    run_oneclass_svm(subset_name, uv, ann, representation_name, x, args.random_state)
                )

    labels_df = pd.DataFrame(label_rows)
    oneclass_df = pd.DataFrame(oneclass_rows)
    cluster_summary_rows = summarize_best_clusters(metrics_rows, label_rows)
    recommendation_rows = make_recommendations(
        annotations.copy(),
        labels_df,
        metrics_rows,
        oneclass_df,
        loaded["uv"],
    )

    save_csv(args.output_dir / "kernel_spectral_metrics.csv", metrics_rows)
    save_csv(args.output_dir / "cluster_labels_long.csv", label_rows)
    save_csv(args.output_dir / "cluster_physics_summary.csv", cluster_summary_rows)
    save_csv(args.output_dir / "oneclass_svm_scores.csv", oneclass_rows)
    save_csv(args.output_dir / "additional_data_recommendations.csv", recommendation_rows)
    if embeddings:
        np.savez_compressed(args.output_dir / "kernel_pca_embeddings.npz", **embeddings)

    if not args.no_plots and not labels_df.empty:
        for subset_name, rep_name, method, filename, title in [
            (
                "nonnegative_v_gap_open",
                "raw_rho_low_w2p0_pca",
                "spectral_clustering",
                "gap_open_raw_rho_w2p0_spectral_uv.png",
                "Gap-open spectral clustering: raw rho_G |omega| <= 2",
            ),
            (
                "nonnegative_v_strong_gap",
                "raw_reG_low_w1p0_pca",
                "spectral_clustering",
                "strong_gap_raw_reG_w1p0_spectral_uv.png",
                "Strong-gap spectral clustering: raw ReG |omega| <= 1",
            ),
            (
                "nonnegative_v_strong_gap",
                "luttinger_geometry",
                "kernel_pca_kmeans",
                "strong_gap_luttinger_kernel_pca_uv.png",
                "Strong-gap kernel PCA clustering: Luttinger geometry",
            ),
        ]:
            metric = select_best_metric(metrics_rows, subset_name, rep_name, method)
            if metric is not None:
                plot_cluster_uv(plots_dir / filename, title, labels_df, subset_name, rep_name, metric)

    if not args.no_plots and not oneclass_df.empty:
        for subset_name, rep_name, filename, title in [
            (
                "nonnegative_v_gap_open",
                "raw_rho_low_w2p0_pca",
                "gap_open_raw_rho_w2p0_oneclass_svm.png",
                "Gap-open One-Class SVM boundary: raw rho_G",
            ),
            (
                "nonnegative_v_strong_gap",
                "raw_reG_low_w1p0_pca",
                "strong_gap_raw_reG_w1p0_oneclass_svm.png",
                "Strong-gap One-Class SVM boundary: raw ReG",
            ),
        ]:
            selected = oneclass_df[
                (oneclass_df["subset"] == subset_name)
                & (oneclass_df["representation"] == rep_name)
                & (oneclass_df["gamma_factor"] == 1.0)
                & (np.isclose(oneclass_df["nu"], 0.10))
            ]
            if not selected.empty:
                plot_oneclass_uv(plots_dir / filename, title, selected)

    best_metrics = sorted(
        [row for row in metrics_rows if row.get("error", "") == "" and row.get("silhouette") is not None],
        key=cluster_quality_score,
        reverse=True,
    )[:20]
    summary = {
        "inputs": {
            "features": str(args.features),
            "gap_csv": str(args.gap_csv),
            "luttinger_features": str(args.luttinger_features),
            "baseline_dataset": str(args.baseline_dataset),
        },
        "scope": {
            "analysis_v_min": args.analysis_v_min,
            "n_nonnegative_v": int(loaded["uv"].shape[0]),
            "n_gap_open": int(annotations["gap_open"].sum()),
            "n_strong_gap": int(annotations["strong_gap"].sum()),
            "n_deep_gap": int(annotations["deep_gap"].sum()),
        },
        "methods": {
            "kernel_pca": "RBF KernelPCA followed by KMeans",
            "spectral_clustering": "RBF spectral clustering",
            "oneclass_svm": "RBF One-Class SVM for boundary/anomaly detection",
            "bootstrap_stability": (
                f"{args.n_bootstrap} subsamples at fraction {args.bootstrap_fraction}, "
                "reported as ARI versus full-sample labels on the overlap"
            ),
        },
        "best_metrics": best_metrics,
        "recommendation_file": str(args.output_dir / "additional_data_recommendations.csv"),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(summary), handle, indent=2, sort_keys=True)

    print(f"Wrote kernel/SVM unsupervised refinement to {args.output_dir}")
    print(
        f"Scope: V >= {args.analysis_v_min:g}, n={loaded['uv'].shape[0]}, "
        f"gap-open={int(annotations['gap_open'].sum())}, "
        f"strong-gap={int(annotations['strong_gap'].sum())}"
    )
    if best_metrics:
        top = best_metrics[0]
        print(
            "Best metric: "
            f"{top['subset']} / {top['representation']} / {top['method']} "
            f"k={top['k']} gamma_factor={top['gamma_factor']} "
            f"silhouette={top['silhouette']:.3f} "
            f"bootstrap_ARI={top['bootstrap_ari_mean']:.3f}"
        )
    if recommendation_rows:
        print("Top recommended new points:")
        for row in recommendation_rows[:10]:
            print(f"  ({row['U']}, {row['V']}) tier {row['tier']}  {row['source']}")


if __name__ == "__main__":
    main()
