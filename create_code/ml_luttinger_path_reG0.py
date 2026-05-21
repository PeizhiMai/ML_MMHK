#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/Gcalpath/rho_re_g/ml_baseline_expanded_w1201/ml_ready_dataset.npz"
DEFAULT_OUTPUT_DIR = ROOT / "data/Gcalpath/rho_re_g/ml_luttinger_path_reG0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn from reG(k, omega=0) zero crossings along Gamma-X-M-Gamma."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--v-min", type=float, default=None)
    parser.add_argument("--v-max", type=float, default=None)
    parser.add_argument(
        "--maxk-rho0-ratio-max",
        type=float,
        default=None,
        help="Keep only samples with max_k rho_G(k,0) / max_omega <rho_G>_path below this value.",
    )
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--max-crossings", type=int, default=6)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def path_centers(n_path: int) -> np.ndarray:
    centers = np.zeros(n_path, dtype=float)
    for i in range(n_path):
        if i < 80:
            centers[i] = i + 1.0
        else:
            centers[i] = centers[79] + 1.0 + (i - 80) * np.sqrt(2.0)
    return centers


def segment_name(center: float, gamma_x_end: float, x_m_end: float) -> str:
    if center <= gamma_x_end:
        return "Gamma-X"
    if center <= x_m_end:
        return "X-M"
    return "M-Gamma"


def crossing_positions(re0: np.ndarray, x: np.ndarray, max_crossings: int) -> list[float]:
    values = np.asarray(re0, dtype=float)
    signs = np.sign(values)
    signs[signs == 0.0] = 1.0
    changes = np.where(signs[1:] != signs[:-1])[0]
    positions: list[float] = []
    for idx in changes[:max_crossings]:
        denom = values[idx + 1] - values[idx]
        frac = 0.5 if abs(denom) < 1.0e-14 else -values[idx] / denom
        frac = float(np.clip(frac, 0.0, 1.0))
        positions.append(float(x[idx] + frac * (x[idx + 1] - x[idx])))
    return positions


def extract_luttinger_features(
    dataset: Path,
    v_min: float | None,
    v_max: float | None,
    maxk_rho0_ratio_max: float | None,
    max_crossings: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, object]], np.ndarray]:
    z = np.load(dataset, allow_pickle=True)
    uv = z["uv"]
    re_g = z["reG"]
    rho_g = z["rho_G"]
    omega = z["omega"]
    k_path = z["k_path"]

    mask = np.ones(uv.shape[0], dtype=bool)
    if v_min is not None:
        mask &= uv[:, 1] >= v_min
    if v_max is not None:
        mask &= uv[:, 1] <= v_max
    if maxk_rho0_ratio_max is not None:
        omega_zero_idx_all = int(np.argmin(np.abs(omega)))
        path_average = np.clip(rho_g, 0.0, None).mean(axis=1)
        denominator = path_average.max(axis=1)
        numerator = np.clip(rho_g[:, :, omega_zero_idx_all], 0.0, None).max(axis=1)
        ratio = numerator / np.where(denominator > 1.0e-12, denominator, 1.0)
        mask &= ratio < maxk_rho0_ratio_max
    uv = uv[mask]
    re_g = re_g[mask]
    rho_g = rho_g[mask]
    if uv.size == 0:
        raise ValueError("No samples remain after applying V filters.")

    omega_zero_idx = int(np.argmin(np.abs(omega)))
    if abs(float(omega[omega_zero_idx])) > 1.0e-12:
        raise ValueError(f"No exact omega=0 grid point found; nearest is {omega[omega_zero_idx]}.")

    x = path_centers(k_path.shape[0])
    gamma_x_end = x[40]
    x_m_end = x[80]

    names = [
        "crossing_count",
        "reG0_mean",
        "reG0_std",
        "reG0_min",
        "reG0_max",
        "abs_reG0_mean",
        "abs_reG0_min",
        "abs_reG0_max",
        "rho0_mean",
        "rho0_min",
        "rho0_max",
        "weighted_zero_center",
    ]
    for idx in range(max_crossings):
        names.extend(
            [
                f"crossing_{idx + 1}_x",
                f"crossing_{idx + 1}_segment_code",
                f"crossing_{idx + 1}_rho0_interp",
            ]
        )

    features: list[list[float]] = []
    crossing_rows: list[dict[str, object]] = []
    segment_code = {"Gamma-X": 0.0, "X-M": 1.0, "M-Gamma": 2.0, "none": -1.0}

    for sample_idx, (target, re_sample, rho_sample) in enumerate(zip(uv, re_g, rho_g)):
        re0 = re_sample[:, omega_zero_idx]
        rho0 = rho_sample[:, omega_zero_idx]
        abs_re0 = np.abs(re0)
        weights = 1.0 / (abs_re0 + 1.0e-8)
        positions = crossing_positions(re0, x, max_crossings)
        row = [
            float(len(positions)),
            float(np.mean(re0)),
            float(np.std(re0)),
            float(np.min(re0)),
            float(np.max(re0)),
            float(np.mean(abs_re0)),
            float(np.min(abs_re0)),
            float(np.max(abs_re0)),
            float(np.mean(rho0)),
            float(np.min(rho0)),
            float(np.max(rho0)),
            float(np.sum(x * weights) / np.sum(weights)),
        ]

        for crossing_idx, position in enumerate(positions):
            seg = segment_name(position, gamma_x_end, x_m_end)
            rho_interp = float(np.interp(position, x, rho0))
            row.extend([position, segment_code[seg], rho_interp])
            crossing_rows.append(
                {
                    "sample_index": int(sample_idx),
                    "U": float(target[0]),
                    "V": float(target[1]),
                    "crossing_index": int(crossing_idx + 1),
                    "x_path": position,
                    "segment": seg,
                    "rho_G_k_omega0_interp": rho_interp,
                }
            )

        for _ in range(max_crossings - len(positions)):
            row.extend([-1.0, segment_code["none"], 0.0])
        features.append(row)

    return uv, np.asarray(features, dtype=float), names, crossing_rows, x


def deterministic_split(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_samples)
    test_mask = indices % 5 == 0
    return indices[~test_mask], indices[test_mask]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    return {
        "rmse_U": float(rmse[0]),
        "rmse_V": float(rmse[1]),
        "mae_U": float(mae[0]),
        "mae_V": float(mae[1]),
        "r2_U": float(r2[0]),
        "r2_V": float(r2[1]),
    }


def fit_predict(x: np.ndarray, y: np.ndarray, random_state: int) -> tuple[dict[str, float], dict[str, float], np.ndarray, np.ndarray]:
    train_idx, test_idx = deterministic_split(x.shape[0])
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 25)))
    model.fit(x[train_idx], y[train_idx])
    holdout_pred = model.predict(x[test_idx])

    cv_pred = np.zeros_like(y, dtype=float)
    splitter = KFold(n_splits=5, shuffle=True, random_state=random_state)
    for fold_train, fold_test in splitter.split(x):
        fold_model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 25)))
        fold_model.fit(x[fold_train], y[fold_train])
        cv_pred[fold_test] = fold_model.predict(x[fold_test])

    return (
        regression_metrics(y[test_idx], holdout_pred),
        regression_metrics(y, cv_pred),
        holdout_pred,
        cv_pred,
    )


def write_crossings(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["sample_index", "U", "V", "crossing_index", "x_path", "segment", "rho_G_k_omega0_interp"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_crossings(uv: np.ndarray, crossing_rows: list[dict[str, object]], output: Path) -> None:
    if not crossing_rows:
        return
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for crossing_idx, marker in [(1, "o"), (2, "s"), (3, "^"), (4, "D"), (5, "v"), (6, "P")]:
        rows = [row for row in crossing_rows if row["crossing_index"] == crossing_idx]
        if not rows:
            continue
        sc = ax.scatter(
            [row["U"] for row in rows],
            [row["V"] for row in rows],
            c=[row["x_path"] for row in rows],
            cmap="viridis",
            marker=marker,
            s=42,
            edgecolor="black",
            linewidth=0.25,
            label=f"crossing {crossing_idx}",
        )
    ax.set_xlabel("U")
    ax.set_ylabel("V")
    ax.set_title("reG(k,0)=0 crossing position along path")
    fig.colorbar(sc, ax=ax, label="path coordinate")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    uv, features, feature_names, crossing_rows, path_x = extract_luttinger_features(
        args.dataset, args.v_min, args.v_max, args.maxk_rho0_ratio_max, args.max_crossings
    )
    holdout_metrics, cv_metrics, holdout_pred, cv_pred = fit_predict(features, uv, args.random_state)
    train_idx, test_idx = deterministic_split(uv.shape[0])

    np.savez(
        args.output_dir / "luttinger_path_features.npz",
        uv=uv,
        features=features,
        feature_names=np.asarray(feature_names),
        path_x=path_x,
        target_columns=np.asarray(["U", "V"]),
    )
    np.savez(
        args.output_dir / "luttinger_path_ridge_predictions.npz",
        train_idx=train_idx,
        test_idx=test_idx,
        holdout_true=uv[test_idx],
        holdout_pred=holdout_pred,
        cv_true=uv,
        cv_pred=cv_pred,
    )
    write_crossings(args.output_dir / "luttinger_crossings.csv", crossing_rows)
    if not args.no_plots:
        plot_crossings(uv, crossing_rows, args.output_dir / "luttinger_crossings.png")

    crossing_counts, crossing_count_freq = np.unique(features[:, 0].astype(int), return_counts=True)
    summary = {
        "dataset": str(args.dataset),
        "v_min": args.v_min,
        "v_max": args.v_max,
        "maxk_rho0_ratio_max": args.maxk_rho0_ratio_max,
        "n_samples": int(uv.shape[0]),
        "n_features": int(features.shape[1]),
        "max_crossings": int(args.max_crossings),
        "crossing_count_distribution": {
            str(int(count)): int(freq) for count, freq in zip(crossing_counts, crossing_count_freq)
        },
        "feature_file": str(args.output_dir / "luttinger_path_features.npz"),
        "crossing_file": str(args.output_dir / "luttinger_crossings.csv"),
        "ridge": {
            "holdout": holdout_metrics,
            "five_fold_cv": cv_metrics,
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
