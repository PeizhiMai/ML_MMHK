#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/Gcalpath/rho_re_g"
DEFAULT_FULL_FEATURES = BASE / "ml_physics_features_eta0p1_full_20260504/physics_features.npz"
DEFAULT_POSITIVE_FEATURES = BASE / "ml_physics_features_eta0p1_positive_20260504/physics_features.npz"
DEFAULT_GAP_CSV = BASE / "gap_analysis_eta0p1_20260504/positive_v_gap_classification_maxk_rho0.csv"
DEFAULT_LUTTINGER_FEATURES = (
    BASE / "ml_luttinger_path_reG0_eta0p1_positive_gaplt0p01_20260504/luttinger_path_features.npz"
)
DEFAULT_BASELINE_DATASET = BASE / "ml_baseline_eta0p1_20260504/ml_ready_dataset.npz"
DEFAULT_OUTPUT_DIR = BASE / "ml_unsupervised_eta0p1_20260505"
GAP_OPEN_THRESHOLD = 0.10
STRONG_GAP_THRESHOLD = 0.01
DEEP_GAP_THRESHOLD = 0.001
RANDOM_STATE = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unsupervised regime analysis for eta=0.1 rho_G/reG physics features."
    )
    parser.add_argument("--full-features", type=Path, default=DEFAULT_FULL_FEATURES)
    parser.add_argument("--positive-features", type=Path, default=DEFAULT_POSITIVE_FEATURES)
    parser.add_argument("--gap-csv", type=Path, default=DEFAULT_GAP_CSV)
    parser.add_argument("--luttinger-features", type=Path, default=DEFAULT_LUTTINGER_FEATURES)
    parser.add_argument("--baseline-dataset", type=Path, default=DEFAULT_BASELINE_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--analysis-v-min",
        type=float,
        default=0.0,
        help="Minimum V value defining the main analyzed subset. Use 0 for V >= 0, 0.001 for V > 0.",
    )
    parser.add_argument(
        "--subset-prefix",
        type=str,
        default=None,
        help="Optional name prefix for the main analyzed subset, e.g. nonnegative_v.",
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def uv_key(uv: np.ndarray | tuple[float, float] | list[float]) -> tuple[float, float]:
    return (round(float(uv[0]), 3), round(float(uv[1]), 3))


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


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
        return finite_float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def load_physics_features(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    uv = np.asarray(z["uv"], dtype=float)
    features = np.asarray(z["features"], dtype=float)
    names = [str(x) for x in z["feature_names"]]
    if features.shape[1] != len(names):
        raise ValueError(f"{path}: feature-name count does not match feature matrix.")
    if np.any(~np.isfinite(features)):
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return {"uv": uv, "features": features, "feature_names": names}


def load_gap_table(path: Path) -> dict[tuple[float, float], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    table: dict[tuple[float, float], dict[str, Any]] = {}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"U", "V", "maxk_rho0_ratio"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            key = uv_key((float(row["U"]), float(row["V"])))
            ratio = float(row["maxk_rho0_ratio"])
            table[key] = {
                "maxk_rho0_ratio": ratio,
                "gap_open": ratio < GAP_OPEN_THRESHOLD,
                "strong_gap": ratio < STRONG_GAP_THRESHOLD,
            }
    return table


def load_luttinger_features(path: Path) -> dict[tuple[float, float], dict[str, float]]:
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=True)
    uv = np.asarray(z["uv"], dtype=float)
    features = np.asarray(z["features"], dtype=float)
    names = [str(x) for x in z["feature_names"]]
    table: dict[tuple[float, float], dict[str, float]] = {}
    for target, row in zip(uv, features):
        table[uv_key(target)] = {name: float(value) for name, value in zip(names, row)}
    return table


def load_luttinger_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return (
        np.asarray(z["uv"], dtype=float),
        np.asarray(z["features"], dtype=float),
        [str(x) for x in z["feature_names"]],
    )


def load_raw_low_energy_representations(
    path: Path,
    n_components: int = 10,
    omega_windows: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
    random_state: int = RANDOM_STATE,
) -> dict[tuple[float, float], dict[str, np.ndarray]]:
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=True)
    uv = np.asarray(z["uv"], dtype=float)
    omega = np.asarray(z["omega"], dtype=float)
    rho = np.asarray(z["rho_G"], dtype=float)
    re_g = np.asarray(z["reG"], dtype=float)

    table: dict[tuple[float, float], dict[str, np.ndarray]] = {uv_key(target): {} for target in uv}
    multiscale_parts: dict[tuple[float, float], list[np.ndarray]] = {uv_key(target): [] for target in uv}

    for omega_window in omega_windows:
        mask = np.abs(omega) <= omega_window
        if not np.any(mask):
            raise ValueError(f"No omega points in |omega| <= {omega_window}")
        suffix = f"w{str(omega_window).replace('.', 'p')}"
        raw_blocks = {
            f"raw_rho_low_{suffix}_pca": rho[:, :, mask],
            f"raw_reG_low_{suffix}_pca": re_g[:, :, mask],
        }
        raw_blocks[f"raw_combined_low_{suffix}_pca"] = np.concatenate(
            [
                raw_blocks[f"raw_rho_low_{suffix}_pca"].reshape(uv.shape[0], -1),
                raw_blocks[f"raw_reG_low_{suffix}_pca"].reshape(uv.shape[0], -1),
            ],
            axis=1,
        )

        for name, block in raw_blocks.items():
            flat = block.reshape(uv.shape[0], -1)
            flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
            flat_scaled = StandardScaler().fit_transform(flat)
            n_comp = min(n_components, flat_scaled.shape[0] - 1, flat_scaled.shape[1])
            if n_comp < 2:
                continue
            pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=random_state)
            coords = pca.fit_transform(flat_scaled)
            explained = pca.explained_variance_ratio_
            for target, row in zip(uv, coords):
                key = uv_key(target)
                table[key][name] = row.astype(float)
                if name.startswith("raw_combined_low_"):
                    multiscale_parts[key].append(row.astype(float))
            for target in uv:
                table[uv_key(target)][f"{name}__explained_variance_ratio"] = explained.astype(float)

    for key, parts in multiscale_parts.items():
        if parts:
            table[key]["raw_combined_multiscale_low_pca"] = np.concatenate(parts)
    return table


def annotate_samples(
    uv: np.ndarray,
    gap_table: dict[tuple[float, float], dict[str, Any]],
    luttinger_table: dict[tuple[float, float], dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, target in enumerate(uv):
        key = uv_key(target)
        gap_info = gap_table.get(key, {})
        ratio = finite_float(gap_info.get("maxk_rho0_ratio"))
        gap_open = bool(gap_info.get("gap_open", False)) if ratio is not None else None
        strong_gap = bool(gap_info.get("strong_gap", False)) if ratio is not None else None
        lut = luttinger_table.get(key, {})
        rows.append(
            {
                "sample_index": idx,
                "U": float(target[0]),
                "V": float(target[1]),
                "maxk_rho0_ratio": ratio,
                "gap_open_ratio_lt_0p10": gap_open,
                "strong_gap_ratio_lt_0p01": strong_gap,
                "deep_gap_ratio_lt_0p001": (ratio < DEEP_GAP_THRESHOLD)
                if ratio is not None
                else None,
                "luttinger_crossing_count": finite_float(lut.get("crossing_count")),
                "luttinger_weighted_zero_center": finite_float(lut.get("weighted_zero_center")),
            }
        )
    return rows


def feature_indices(names: list[str], representation: str) -> np.ndarray:
    if representation == "rho":
        idx = [i for i, name in enumerate(names) if name.startswith("rho_")]
    elif representation == "reG":
        idx = [i for i, name in enumerate(names) if name.startswith("reG_")]
    elif representation == "combined":
        idx = list(range(len(names)))
    else:
        raise ValueError(f"Unknown representation {representation!r}")
    if not idx:
        raise ValueError(f"No features found for representation {representation!r}")
    return np.asarray(idx, dtype=int)


def subset_mask(name: str, annotations: list[dict[str, Any]], v_min: float = 0.0) -> np.ndarray:
    base_mask = np.asarray([row["V"] >= v_min for row in annotations], dtype=bool)
    if name == "full":
        return np.ones(len(annotations), dtype=bool)
    if name in {"positive_v", "nonnegative_v"}:
        return base_mask
    if name.endswith("_gap_open"):
        return np.asarray(
            [keep and bool(row["gap_open_ratio_lt_0p10"]) for row, keep in zip(annotations, base_mask)],
            dtype=bool,
        )
    if name.endswith("_strong_gap"):
        return np.asarray(
            [keep and bool(row["strong_gap_ratio_lt_0p01"]) for row, keep in zip(annotations, base_mask)],
            dtype=bool,
        )
    if name.endswith("_deep_gap"):
        return np.asarray(
            [keep and bool(row["deep_gap_ratio_lt_0p001"]) for row, keep in zip(annotations, base_mask)],
            dtype=bool,
        )
    raise ValueError(f"Unknown subset {name!r}")


def metric_scores(x: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    unique = sorted(set(int(v) for v in labels if int(v) >= 0))
    if len(unique) < 2 or len(unique) >= x.shape[0]:
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    return {
        "silhouette": float(silhouette_score(x, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
        "davies_bouldin": float(davies_bouldin_score(x, labels)),
    }


def cluster_count(labels: np.ndarray) -> int:
    return len(set(int(v) for v in labels if int(v) >= 0))


def kmeans_stability(x: np.ndarray, k: int, random_state: int) -> float | None:
    if x.shape[0] <= k:
        return None
    seeds = [random_state + offset for offset in range(8)]
    labels = [
        KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(x)
        for seed in seeds
    ]
    scores = [
        adjusted_rand_score(labels[0], labels[idx])
        for idx in range(1, len(labels))
    ]
    return float(np.mean(scores))


def top_cluster_features(
    x_scaled: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    cluster_id: int,
    n_top: int = 5,
) -> str:
    in_cluster = labels == cluster_id
    if not np.any(in_cluster):
        return ""
    mean_z = x_scaled[in_cluster].mean(axis=0)
    order = np.argsort(np.abs(mean_z))[::-1][:n_top]
    return "; ".join(f"{feature_names[idx]}={mean_z[idx]:+.2f}z" for idx in order)


def cluster_summary_rows(
    subset_name: str,
    representation: str,
    method: str,
    k_value: int | str,
    labels: np.ndarray,
    x_scaled: np.ndarray,
    feature_names: list[str],
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels_unique = sorted(set(int(v) for v in labels))
    for label in labels_unique:
        mask = labels == label
        cluster_annotations = [row for row, keep in zip(annotations, mask) if keep]
        ratios = [
            float(row["maxk_rho0_ratio"])
            for row in cluster_annotations
            if row["maxk_rho0_ratio"] is not None
        ]
        gap_known = [
            bool(row["gap_open_ratio_lt_0p10"])
            for row in cluster_annotations
            if row["gap_open_ratio_lt_0p10"] is not None
        ]
        strong_gap_known = [
            bool(row["strong_gap_ratio_lt_0p01"])
            for row in cluster_annotations
            if row["strong_gap_ratio_lt_0p01"] is not None
        ]
        deep_gap_known = [
            bool(row["deep_gap_ratio_lt_0p001"])
            for row in cluster_annotations
            if row["deep_gap_ratio_lt_0p001"] is not None
        ]
        crossings = [
            float(row["luttinger_crossing_count"])
            for row in cluster_annotations
            if row["luttinger_crossing_count"] is not None
        ]
        centers = [
            float(row["luttinger_weighted_zero_center"])
            for row in cluster_annotations
            if row["luttinger_weighted_zero_center"] is not None
        ]
        u_values = np.asarray([row["U"] for row in cluster_annotations], dtype=float)
        v_values = np.asarray([row["V"] for row in cluster_annotations], dtype=float)
        row = {
            "subset": subset_name,
            "representation": representation,
            "method": method,
            "k": k_value,
            "cluster": label,
            "count": int(mask.sum()),
            "U_min": float(np.min(u_values)),
            "U_median": float(np.median(u_values)),
            "U_max": float(np.max(u_values)),
            "V_min": float(np.min(v_values)),
            "V_median": float(np.median(v_values)),
            "V_max": float(np.max(v_values)),
            "gap_ratio_median": float(np.median(ratios)) if ratios else None,
            "gap_open_fraction": float(np.mean(gap_known)) if gap_known else None,
            "strong_gap_fraction": float(np.mean(strong_gap_known)) if strong_gap_known else None,
            "deep_gap_fraction": float(np.mean(deep_gap_known)) if deep_gap_known else None,
            "luttinger_crossing_count_median": float(np.median(crossings)) if crossings else None,
            "luttinger_weighted_zero_center_median": float(np.median(centers)) if centers else None,
            "top_feature_z_means": top_cluster_features(
                x_scaled, labels, feature_names, label
            )
            if label >= 0
            else "",
        }
        rows.append(row)
    return rows


def run_embeddings(x_scaled: np.ndarray, random_state: int) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    n_components = min(3, x_scaled.shape[0], x_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_coords = pca.fit_transform(x_scaled)
    if n_components < 3:
        pca_coords = np.pad(pca_coords, ((0, 0), (0, 3 - n_components)), constant_values=np.nan)
    embeddings["pca"] = pca_coords
    embeddings["pca_explained_variance_ratio"] = np.asarray(
        list(pca.explained_variance_ratio_) + [np.nan] * (3 - n_components),
        dtype=float,
    )

    if x_scaled.shape[0] >= 8:
        perplexity = min(30, max(5, (x_scaled.shape[0] - 1) // 4))
        try:
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=random_state,
            )
            embeddings["tsne"] = tsne.fit_transform(x_scaled)
            embeddings["tsne_perplexity"] = np.asarray([perplexity], dtype=float)
        except TypeError:
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                random_state=random_state,
            )
            embeddings["tsne"] = tsne.fit_transform(x_scaled)
            embeddings["tsne_perplexity"] = np.asarray([perplexity], dtype=float)

    if x_scaled.shape[0] >= 10:
        try:
            spectral = SpectralEmbedding(
                n_components=2,
                n_neighbors=min(10, x_scaled.shape[0] - 1),
                random_state=random_state,
            )
            embeddings["spectral"] = spectral.fit_transform(x_scaled)
        except Exception:
            pass
    return embeddings


def run_clustering(
    x_scaled: np.ndarray,
    random_state: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    metrics: list[dict[str, Any]] = []
    labels_by_name: dict[str, np.ndarray] = {}
    max_k = min(6, x_scaled.shape[0] - 1)
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=50, random_state=random_state)
        labels = kmeans.fit_predict(x_scaled)
        name = f"kmeans_k{k}"
        labels_by_name[name] = labels
        scores = metric_scores(x_scaled, labels)
        metrics.append(
            {
                "method": "kmeans",
                "k": k,
                "label_name": name,
                "n_clusters": cluster_count(labels),
                "inertia": float(kmeans.inertia_),
                "stability_adjusted_rand": kmeans_stability(x_scaled, k, random_state),
                **scores,
            }
        )

        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
        labels = gmm.fit_predict(x_scaled)
        name = f"gmm_k{k}"
        labels_by_name[name] = labels
        scores = metric_scores(x_scaled, labels)
        metrics.append(
            {
                "method": "gmm",
                "k": k,
                "label_name": name,
                "n_clusters": cluster_count(labels),
                "aic": float(gmm.aic(x_scaled)),
                "bic": float(gmm.bic(x_scaled)),
                **scores,
            }
        )

        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(x_scaled)
        name = f"agglomerative_k{k}"
        labels_by_name[name] = labels
        scores = metric_scores(x_scaled, labels)
        metrics.append(
            {
                "method": "agglomerative",
                "k": k,
                "label_name": name,
                "n_clusters": cluster_count(labels),
                **scores,
            }
        )

    if x_scaled.shape[0] >= 8:
        for eps in [0.8, 1.2, 1.6, 2.0]:
            dbscan = DBSCAN(eps=eps, min_samples=max(4, min(8, x_scaled.shape[0] // 10)))
            labels = dbscan.fit_predict(x_scaled)
            name = f"dbscan_eps{eps:g}"
            labels_by_name[name] = labels
            scores = metric_scores(x_scaled, labels)
            metrics.append(
                {
                    "method": "dbscan",
                    "k": f"eps={eps:g}",
                    "label_name": name,
                    "n_clusters": cluster_count(labels),
                    "noise_fraction": float(np.mean(labels < 0)),
                    **scores,
                }
            )
    return metrics, labels_by_name


def best_label(metrics: list[dict[str, Any]], method: str) -> str | None:
    candidates = [
        row
        for row in metrics
        if row["method"] == method and row.get("silhouette") is not None and row.get("n_clusters", 0) >= 2
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda row: (float(row["silhouette"]), -int(row["n_clusters"])))
    return str(candidates[-1]["label_name"])


def consensus_from_best_labels(
    labels_by_name: dict[str, np.ndarray],
    best: dict[str, str | None],
) -> tuple[np.ndarray | None, dict[str, float | None]]:
    selected = [
        labels_by_name[name]
        for method, name in best.items()
        if method in {"kmeans", "gmm", "agglomerative"} and name is not None
    ]
    if len(selected) < 2:
        return None, {"mean_pairwise_ari": None, "mean_sample_agreement": None}

    n_samples = selected[0].shape[0]
    agreement = np.zeros(n_samples, dtype=float)
    for idx in range(n_samples):
        votes: dict[int, int] = {}
        for labels in selected:
            label = int(labels[idx])
            votes[label] = votes.get(label, 0) + 1
        agreement[idx] = max(votes.values()) / len(selected)

    pairwise_ari = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            pairwise_ari.append(adjusted_rand_score(selected[i], selected[j]))

    return agreement, {
        "mean_pairwise_ari": float(np.mean(pairwise_ari)) if pairwise_ari else None,
        "mean_sample_agreement": float(np.mean(agreement)),
    }


def gapped_recommendation(metrics: list[dict[str, Any]], consensus: dict[str, float | None]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in metrics
        if row["method"] == "kmeans"
        and row.get("silhouette") is not None
        and row.get("stability_adjusted_rand") is not None
        and int(row.get("n_clusters", 0)) >= 2
    ]
    if not candidates:
        return None
    consensus_ari = consensus.get("mean_pairwise_ari") or 0.0
    best: dict[str, Any] | None = None
    best_score = -np.inf
    for row in candidates:
        k = int(row["k"])
        small_k_penalty = 1.0 - 0.035 * max(0, k - 2)
        score = (
            float(row["silhouette"])
            * float(row["stability_adjusted_rand"])
            * (0.65 + 0.35 * float(consensus_ari))
            * small_k_penalty
        )
        if score > best_score:
            best_score = score
            best = {
                "label_name": row["label_name"],
                "k": k,
                "method": "kmeans",
                "gapped_score": float(score),
                "silhouette": float(row["silhouette"]),
                "stability_adjusted_rand": float(row["stability_adjusted_rand"]),
                "consensus_pairwise_ari": consensus.get("mean_pairwise_ari"),
                "consensus_sample_agreement": consensus.get("mean_sample_agreement"),
            }
    return best


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_uv_clusters(
    path: Path,
    annotations: list[dict[str, Any]],
    labels: np.ndarray,
    title: str,
) -> None:
    u = np.asarray([row["U"] for row in annotations], dtype=float)
    v = np.asarray([row["V"] for row in annotations], dtype=float)
    ratio = np.asarray(
        [
            np.nan if row["maxk_rho0_ratio"] is None else float(row["maxk_rho0_ratio"])
            for row in annotations
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    scatter = ax.scatter(u, v, c=labels, cmap="tab10", s=52, edgecolor="black", linewidth=0.35)
    for idx, value in enumerate(ratio):
        if np.isfinite(value) and value < STRONG_GAP_THRESHOLD:
            ax.scatter(u[idx], v[idx], s=95, facecolors="none", edgecolors="black", linewidth=1.0)
    ax.set_xlabel("U")
    ax.set_ylabel("V")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_embedding(
    path: Path,
    coords: np.ndarray,
    color: np.ndarray,
    title: str,
    color_label: str,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=color, cmap=cmap, s=52, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label=color_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_gap_by_cluster(
    path: Path,
    annotations: list[dict[str, Any]],
    labels: np.ndarray,
    title: str,
) -> None:
    ratios = []
    label_values = []
    for row, label in zip(annotations, labels):
        if row["maxk_rho0_ratio"] is None:
            continue
        ratios.append(float(row["maxk_rho0_ratio"]))
        label_values.append(int(label))
    if not ratios:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.scatter(label_values, ratios, s=48, edgecolor="black", linewidth=0.35)
    ax.axhline(GAP_OPEN_THRESHOLD, color="tab:orange", linestyle="--", label="gap open < 0.10")
    ax.axhline(STRONG_GAP_THRESHOLD, color="tab:red", linestyle="--", label="strong gap < 0.01")
    ax.set_yscale("log")
    ax.set_xlabel("cluster")
    ax.set_ylabel("max-k rho(0) ratio")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_dataset(
    subset_name: str,
    uv: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    annotations: list[dict[str, Any]],
    output_dir: Path,
    random_state: int,
    make_plots: bool,
    extra_representations: dict[str, tuple[np.ndarray, list[str]]] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    if make_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "subset": subset_name,
        "n_samples": int(features.shape[0]),
        "representations": {},
    }
    all_metric_rows: list[dict[str, Any]] = []
    all_cluster_summary_rows: list[dict[str, Any]] = []
    long_label_rows: list[dict[str, Any]] = []
    primary_assignments: list[dict[str, Any]] = []
    pca_npz: dict[str, np.ndarray] = {}
    tsne_npz: dict[str, np.ndarray] = {}

    representation_inputs: list[tuple[str, np.ndarray, list[str]]] = []
    for representation in ["rho", "reG", "combined"]:
        idx = feature_indices(feature_names, representation)
        representation_inputs.append(
            (representation, features[:, idx], [feature_names[i] for i in idx])
        )
    if extra_representations:
        representation_inputs.extend(
            (name, matrix, names)
            for name, (matrix, names) in extra_representations.items()
            if matrix.shape[0] == features.shape[0] and matrix.shape[1] >= 2
        )

    for representation, x, names in representation_inputs:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        embeddings = run_embeddings(x_scaled, random_state)
        metrics, labels_by_name = run_clustering(x_scaled, random_state)
        for row in metrics:
            row_with_context = {
                "subset": subset_name,
                "representation": representation,
                **row,
            }
            all_metric_rows.append(row_with_context)

        best = {
            "kmeans": best_label(metrics, "kmeans"),
            "gmm": best_label(metrics, "gmm"),
            "agglomerative": best_label(metrics, "agglomerative"),
            "dbscan": best_label(metrics, "dbscan"),
        }
        consensus_agreement, consensus_metrics = consensus_from_best_labels(labels_by_name, best)
        recommendation = gapped_recommendation(metrics, consensus_metrics) if "gap" in subset_name else None
        summary["representations"][representation] = {
            "n_features": int(x.shape[1]),
            "pca_explained_variance_ratio": [
                finite_float(v) for v in embeddings["pca_explained_variance_ratio"]
            ],
            "tsne_perplexity": finite_float(embeddings.get("tsne_perplexity", [np.nan])[0])
            if "tsne_perplexity" in embeddings
            else None,
            "best_labels_by_silhouette": best,
            "consensus": consensus_metrics,
            "gapped_recommendation": recommendation,
            "metrics": metrics,
        }

        pca_npz[f"{subset_name}__{representation}"] = embeddings["pca"]
        pca_npz[f"{subset_name}__{representation}__explained_variance_ratio"] = embeddings[
            "pca_explained_variance_ratio"
        ]
        if "tsne" in embeddings:
            tsne_npz[f"{subset_name}__{representation}"] = embeddings["tsne"]

        for label_name, labels in labels_by_name.items():
            method = label_name.split("_k")[0].split("_eps")[0]
            k_value: int | str
            if "_k" in label_name:
                k_value = int(label_name.rsplit("_k", 1)[1])
            else:
                k_value = label_name.rsplit("_eps", 1)[1]
            all_cluster_summary_rows.extend(
                cluster_summary_rows(
                    subset_name,
                    representation,
                    method,
                    k_value,
                    labels,
                    x_scaled,
                    names,
                    annotations,
                )
            )
            for row, label in zip(annotations, labels):
                long_label_rows.append(
                    {
                        "subset": subset_name,
                        "representation": representation,
                        "label_name": label_name,
                        "sample_index": row["sample_index"],
                        "U": row["U"],
                        "V": row["V"],
                        "cluster": int(label),
                    }
                )

        if representation == "combined":
            primary = {method: labels_by_name[name] for method, name in best.items() if name}
            pca_coords = embeddings["pca"]
            tsne_coords = embeddings.get("tsne")
            for row_idx, row in enumerate(annotations):
                out = dict(row)
                out["subset"] = subset_name
                out["pca1"] = float(pca_coords[row_idx, 0])
                out["pca2"] = float(pca_coords[row_idx, 1])
                out["pca3"] = float(pca_coords[row_idx, 2])
                if tsne_coords is not None:
                    out["tsne1"] = float(tsne_coords[row_idx, 0])
                    out["tsne2"] = float(tsne_coords[row_idx, 1])
                for method, labels in primary.items():
                    out[f"{method}_best_cluster"] = int(labels[row_idx])
                if consensus_agreement is not None:
                    out["consensus_best_method_agreement"] = float(consensus_agreement[row_idx])
                primary_assignments.append(out)

        plot_representations = {
            "combined",
            "raw_combined_low_w1p0_pca",
            "raw_combined_low_w0p25_pca",
            "raw_combined_low_w0p5_pca",
            "raw_combined_low_w2p0_pca",
            "raw_combined_multiscale_low_pca",
            "raw_rho_low_w2p0_pca",
            "raw_reG_low_w0p5_pca",
            "luttinger_geometry",
        }
        if make_plots and best["kmeans"] is not None and representation in plot_representations:
            labels = labels_by_name[best["kmeans"]]
            plot_uv_clusters(
                plot_dir / f"{subset_name}_{representation}_best_kmeans_uv.png",
                annotations,
                labels,
                f"{subset_name} {representation} best KMeans on U,V",
            )
            plot_embedding(
                plot_dir / f"{subset_name}_{representation}_pca_clusters.png",
                embeddings["pca"][:, :2],
                labels,
                f"{subset_name} {representation} PCA colored by cluster",
                "cluster",
                cmap="tab10",
            )
            if "tsne" in embeddings:
                plot_embedding(
                    plot_dir / f"{subset_name}_{representation}_tsne_clusters.png",
                    embeddings["tsne"],
                    labels,
                    f"{subset_name} {representation} t-SNE colored by cluster",
                    "cluster",
                    cmap="tab10",
                )
            ratios = np.asarray(
                [
                    np.nan if row["maxk_rho0_ratio"] is None else float(row["maxk_rho0_ratio"])
                    for row in annotations
                ],
                dtype=float,
            )
            finite = np.isfinite(ratios)
            if np.any(finite):
                plot_embedding(
                    plot_dir / f"{subset_name}_{representation}_pca_gap_ratio.png",
                    embeddings["pca"][finite, :2],
                    np.log10(ratios[finite]),
                    f"{subset_name} {representation} PCA colored by log10 gap ratio",
                    "log10 ratio",
                )
                plot_gap_by_cluster(
                    plot_dir / f"{subset_name}_{representation}_gap_ratio_by_cluster.png",
                    annotations,
                    labels,
                    f"{subset_name} gap ratio by {representation} best KMeans cluster",
                )

    save_csv(output_dir / f"{subset_name}_metrics.csv", all_metric_rows)
    save_csv(output_dir / f"{subset_name}_cluster_physics_summary.csv", all_cluster_summary_rows)
    save_csv(output_dir / f"{subset_name}_cluster_labels_long.csv", long_label_rows)
    save_csv(output_dir / f"{subset_name}_primary_assignments.csv", primary_assignments)
    np.savez_compressed(output_dir / f"{subset_name}_embedding_pca.npz", **pca_npz)
    if tsne_npz:
        np.savez_compressed(output_dir / f"{subset_name}_embedding_tsne.npz", **tsne_npz)
    return {
        "summary": summary,
        "metric_rows": all_metric_rows,
        "cluster_summary_rows": all_cluster_summary_rows,
        "primary_assignments": primary_assignments,
        "pca_npz": pca_npz,
        "tsne_npz": tsne_npz,
    }


def copy_selected_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    return [{key: row.get(key, "") for key in keys} for row in rows]


def gapped_recommendation_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset, subset_info in summary.get("subsets", {}).items():
        if "gap" not in subset:
            continue
        for representation, rep_info in subset_info.get("representations", {}).items():
            rec = rep_info.get("gapped_recommendation")
            if not rec:
                continue
            rows.append(
                {
                    "subset": subset,
                    "n_samples": subset_info.get("n_samples"),
                    "representation": representation,
                    **rec,
                }
            )
    rows.sort(key=lambda row: float(row.get("gapped_score") or -np.inf), reverse=True)
    return rows


def raw_extra_representations_for_uv(
    uv: np.ndarray,
    raw_table: dict[tuple[float, float], dict[str, np.ndarray]],
) -> dict[str, tuple[np.ndarray, list[str]]]:
    if not raw_table:
        return {}
    names = [
        "raw_rho_low_w0p25_pca",
        "raw_reG_low_w0p25_pca",
        "raw_combined_low_w0p25_pca",
        "raw_rho_low_w0p5_pca",
        "raw_reG_low_w0p5_pca",
        "raw_combined_low_w0p5_pca",
        "raw_rho_low_w1p0_pca",
        "raw_reG_low_w1p0_pca",
        "raw_combined_low_w1p0_pca",
        "raw_rho_low_w2p0_pca",
        "raw_reG_low_w2p0_pca",
        "raw_combined_low_w2p0_pca",
        "raw_combined_multiscale_low_pca",
    ]
    extras: dict[str, tuple[np.ndarray, list[str]]] = {}
    keys = [uv_key(target) for target in uv]
    for name in names:
        if any(name not in raw_table.get(key, {}) for key in keys):
            continue
        matrix = np.vstack([raw_table[key][name] for key in keys])
        feature_names = [f"{name}_{idx + 1}" for idx in range(matrix.shape[1])]
        extras[name] = (matrix, feature_names)
    return extras


def luttinger_extra_representation_for_uv(
    uv: np.ndarray,
    luttinger_matrix: tuple[np.ndarray, np.ndarray, list[str]] | None,
) -> dict[str, tuple[np.ndarray, list[str]]]:
    if luttinger_matrix is None:
        return {}
    lut_uv, lut_features, lut_names = luttinger_matrix
    table = {uv_key(target): row for target, row in zip(lut_uv, lut_features)}
    keys = [uv_key(target) for target in uv]
    if any(key not in table for key in keys):
        return {}
    matrix = np.vstack([table[key] for key in keys])
    return {"luttinger_geometry": (matrix, lut_names)}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_prefix = args.subset_prefix or (
        "nonnegative_v" if args.analysis_v_min <= 0.0 else "positive_v"
    )

    full = load_physics_features(args.full_features)
    positive = load_physics_features(args.positive_features)
    gap_table = load_gap_table(args.gap_csv)
    luttinger_table = load_luttinger_features(args.luttinger_features)
    luttinger_matrix = load_luttinger_matrix(args.luttinger_features)
    raw_low_energy_table = load_raw_low_energy_representations(
        args.baseline_dataset,
        random_state=args.random_state,
    )

    if tuple(full["feature_names"]) != tuple(positive["feature_names"]):
        raise ValueError("Full and positive feature files use different feature names.")

    full_annotations = annotate_samples(full["uv"], gap_table, luttinger_table)
    positive_annotations = annotate_samples(positive["uv"], gap_table, luttinger_table)

    positive_keys = {uv_key(row) for row in positive["uv"] if row[1] >= args.analysis_v_min}
    gap_keys = set(gap_table)
    missing_gap = positive_keys.difference(gap_keys)
    if missing_gap:
        raise ValueError(f"Gap table is missing analyzed V samples: {sorted(missing_gap)[:5]}")

    datasets = {
        "full": (full["uv"], full["features"], full_annotations),
    }
    base_mask = subset_mask(subset_prefix, positive_annotations, args.analysis_v_min)
    datasets[subset_prefix] = (
        positive["uv"][base_mask],
        positive["features"][base_mask],
        [row for row, keep in zip(positive_annotations, base_mask) if keep],
    )

    for subset_name in [
        f"{subset_prefix}_gap_open",
        f"{subset_prefix}_strong_gap",
        f"{subset_prefix}_deep_gap",
    ]:
        mask = subset_mask(subset_name, positive_annotations, args.analysis_v_min)
        datasets[subset_name] = (
            positive["uv"][mask],
            positive["features"][mask],
            [row for row, keep in zip(positive_annotations, mask) if keep],
        )

    all_summary: dict[str, Any] = {
        "inputs": {
            "full_features": str(args.full_features),
            "positive_features": str(args.positive_features),
            "gap_csv": str(args.gap_csv),
            "baseline_dataset": str(args.baseline_dataset)
            if args.baseline_dataset.exists()
            else None,
            "luttinger_features": str(args.luttinger_features)
            if args.luttinger_features.exists()
            else None,
        },
        "thresholds": {
            "gap_open_ratio_lt": GAP_OPEN_THRESHOLD,
            "strong_gap_ratio_lt": STRONG_GAP_THRESHOLD,
            "deep_gap_ratio_lt": DEEP_GAP_THRESHOLD,
        },
        "random_state": int(args.random_state),
        "analysis_v_min": float(args.analysis_v_min),
        "subset_prefix": subset_prefix,
        "subsets": {},
        "interpretation": {},
    }
    combined_primary_assignments: list[dict[str, Any]] = []
    combined_cluster_summaries: list[dict[str, Any]] = []
    combined_metrics: list[dict[str, Any]] = []
    pca_npz_all: dict[str, np.ndarray] = {}
    tsne_npz_all: dict[str, np.ndarray] = {}

    for subset_name, (uv, features, annotations) in datasets.items():
        if features.shape[0] < 4:
            continue
        result = analyze_dataset(
            subset_name,
            uv,
            features,
            full["feature_names"],
            annotations,
            output_dir,
            args.random_state,
            make_plots=not args.no_plots,
            extra_representations={
                **raw_extra_representations_for_uv(uv, raw_low_energy_table),
                **(
                    luttinger_extra_representation_for_uv(uv, luttinger_matrix)
                    if subset_name in {f"{subset_prefix}_strong_gap", f"{subset_prefix}_deep_gap"}
                    else {}
                ),
            },
        )
        all_summary["subsets"][subset_name] = result["summary"]
        combined_primary_assignments.extend(result["primary_assignments"])
        combined_cluster_summaries.extend(result["cluster_summary_rows"])
        combined_metrics.extend(result["metric_rows"])
        pca_npz_all.update(result["pca_npz"])
        tsne_npz_all.update(result["tsne_npz"])

    save_csv(output_dir / "cluster_assignments.csv", combined_primary_assignments)
    save_csv(output_dir / "cluster_physics_summary.csv", combined_cluster_summaries)
    save_csv(output_dir / "cluster_metrics.csv", combined_metrics)
    save_csv(output_dir / "gapped_recommendations.csv", gapped_recommendation_rows(all_summary))
    np.savez_compressed(output_dir / "embedding_pca.npz", **pca_npz_all)
    if tsne_npz_all:
        np.savez_compressed(output_dir / "embedding_tsne.npz", **tsne_npz_all)

    positive_summary = all_summary["subsets"].get(subset_prefix, {})
    combined_info = positive_summary.get("representations", {}).get("combined", {})
    best_kmeans_name = combined_info.get("best_labels_by_silhouette", {}).get("kmeans")
    best_kmeans_metrics = None
    if best_kmeans_name:
        for row in combined_info.get("metrics", []):
            if row.get("label_name") == best_kmeans_name:
                best_kmeans_metrics = row
                break
    all_summary["interpretation"] = {
        "primary_subset": subset_prefix,
        "primary_representation": "combined",
        "primary_best_kmeans": best_kmeans_name,
        "primary_best_kmeans_metrics": best_kmeans_metrics,
        "note": (
            "Clusters are fit only on standardized rho_G/reG physics features. "
            "U, V, gap ratios, and Luttinger features are used only after fitting for interpretation."
        ),
    }

    with (output_dir / "summary.json").open("w") as handle:
        json.dump(json_ready(all_summary), handle, indent=2, sort_keys=True)

    print(f"Wrote unsupervised analysis to {output_dir}")
    if best_kmeans_name and best_kmeans_metrics:
        print(
            f"{subset_prefix} combined best KMeans: "
            f"{best_kmeans_name}, silhouette={best_kmeans_metrics.get('silhouette'):.3f}, "
            f"stability ARI={best_kmeans_metrics.get('stability_adjusted_rand'):.3f}"
        )


if __name__ == "__main__":
    main()
