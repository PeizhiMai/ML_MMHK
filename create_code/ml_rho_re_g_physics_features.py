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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/Gcalpath/rho_re_g/ml_baseline_v2/ml_ready_dataset.npz"
DEFAULT_OUTPUT_DIR = ROOT / "data/Gcalpath/rho_re_g/ml_physics_features"
EPS = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract physics-inspired rho_G/reG features and evaluate simple regressors."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--v-min", type=float, default=None, help="Keep only samples with V >= this value.")
    parser.add_argument("--v-max", type=float, default=None, help="Keep only samples with V <= this value.")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def trapz(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


def add_feature(names: list[str], values: list[float], name: str, value: float) -> None:
    names.append(name)
    values.append(float(value))


def summarize_vector(
    names: list[str],
    values: list[float],
    prefix: str,
    vector: np.ndarray,
    include_minmax: bool = False,
) -> None:
    add_feature(names, values, f"{prefix}_mean", np.mean(vector))
    add_feature(names, values, f"{prefix}_std", np.std(vector))
    if include_minmax:
        add_feature(names, values, f"{prefix}_min", np.min(vector))
        add_feature(names, values, f"{prefix}_max", np.max(vector))


def gap_from_path_average(path_average: np.ndarray, omega: np.ndarray, threshold_fraction: float) -> float:
    threshold = threshold_fraction * float(np.max(path_average))
    if threshold <= 0.0:
        return float("nan")
    center = int(np.argmin(np.abs(omega)))
    if path_average[center] >= threshold:
        return 0.0

    positive = np.where((omega >= 0.0) & (path_average >= threshold))[0]
    negative = np.where((omega <= 0.0) & (path_average >= threshold))[0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    return float(omega[positive[0]] - omega[negative[-1]])


def rho_features(rho: np.ndarray, omega: np.ndarray) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []
    rho_pos = np.clip(rho, 0.0, None)
    path_average = rho_pos.mean(axis=0)
    total_by_k = trapz(rho_pos, omega, axis=1)
    total_safe = np.where(total_by_k > EPS, total_by_k, EPS)

    summarize_vector(names, values, "rho_total_weight", total_by_k, include_minmax=True)

    zero_idx = int(np.argmin(np.abs(omega)))
    zero_values = rho_pos[:, zero_idx]
    summarize_vector(names, values, "rho_zero_weight", zero_values, include_minmax=True)

    for window in [0.2, 0.5, 1.0, 2.0]:
        mask = np.abs(omega) <= window
        low_by_k = trapz(rho_pos[:, mask], omega[mask], axis=1)
        summarize_vector(names, values, f"rho_low_weight_abs_omega_le_{window:g}", low_by_k)
        summarize_vector(names, values, f"rho_low_fraction_abs_omega_le_{window:g}", low_by_k / total_safe)

    positive_mask = omega >= 0.0
    negative_mask = omega <= 0.0
    positive_weight = trapz(rho_pos[:, positive_mask], omega[positive_mask], axis=1)
    negative_weight = trapz(rho_pos[:, negative_mask], omega[negative_mask], axis=1)
    summarize_vector(names, values, "rho_positive_weight", positive_weight)
    summarize_vector(names, values, "rho_negative_weight", negative_weight)
    summarize_vector(
        names,
        values,
        "rho_particle_hole_asymmetry",
        (positive_weight - negative_weight) / total_safe,
    )

    centroids = trapz(rho_pos * omega[None, :], omega, axis=1) / total_safe
    widths = np.sqrt(
        trapz(rho_pos * (omega[None, :] - centroids[:, None]) ** 2, omega, axis=1) / total_safe
    )
    summarize_vector(names, values, "rho_centroid", centroids, include_minmax=True)
    summarize_vector(names, values, "rho_width", widths, include_minmax=True)

    peak_idx = np.argmax(rho_pos, axis=1)
    peak_omega = omega[peak_idx]
    peak_value = rho_pos[np.arange(rho_pos.shape[0]), peak_idx]
    summarize_vector(names, values, "rho_peak_omega_by_k", peak_omega, include_minmax=True)
    summarize_vector(names, values, "rho_peak_value_by_k", peak_value, include_minmax=True)

    for threshold_fraction in [0.01, 0.05, 0.10]:
        add_feature(
            names,
            values,
            f"rho_pathavg_gap_threshold_{threshold_fraction:g}",
            gap_from_path_average(path_average, omega, threshold_fraction),
        )

    path_peak_idx = int(np.argmax(path_average))
    add_feature(names, values, "rho_pathavg_peak_omega", omega[path_peak_idx])
    add_feature(names, values, "rho_pathavg_peak_value", path_average[path_peak_idx])

    if np.any(positive_mask):
        pos_idx = np.where(positive_mask)[0][int(np.argmax(path_average[positive_mask]))]
        add_feature(names, values, "rho_pathavg_positive_peak_omega", omega[pos_idx])
        add_feature(names, values, "rho_pathavg_positive_peak_value", path_average[pos_idx])
    if np.any(negative_mask):
        neg_idx = np.where(negative_mask)[0][int(np.argmax(path_average[negative_mask]))]
        add_feature(names, values, "rho_pathavg_negative_peak_omega", omega[neg_idx])
        add_feature(names, values, "rho_pathavg_negative_peak_value", path_average[neg_idx])

    return names, values


def re_g_features(re_g: np.ndarray, omega: np.ndarray) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []
    abs_re = np.abs(re_g)
    path_average = re_g.mean(axis=0)
    zero_idx = int(np.argmin(np.abs(omega)))

    summarize_vector(names, values, "reG_zero_value", re_g[:, zero_idx], include_minmax=True)
    summarize_vector(names, values, "reG_zero_abs_value", abs_re[:, zero_idx], include_minmax=True)

    if 0 < zero_idx < omega.size - 1:
        slope0 = (re_g[:, zero_idx + 1] - re_g[:, zero_idx - 1]) / (
            omega[zero_idx + 1] - omega[zero_idx - 1]
        )
        summarize_vector(names, values, "reG_zero_slope", slope0, include_minmax=True)

    rms_by_k = np.sqrt(np.mean(re_g**2, axis=1))
    signed_integral = trapz(re_g, omega, axis=1)
    abs_integral = trapz(abs_re, omega, axis=1)
    summarize_vector(names, values, "reG_rms", rms_by_k, include_minmax=True)
    summarize_vector(names, values, "reG_signed_integral", signed_integral)
    summarize_vector(names, values, "reG_abs_integral", abs_integral, include_minmax=True)

    for window in [0.2, 0.5, 1.0, 2.0]:
        mask = np.abs(omega) <= window
        low_abs = trapz(abs_re[:, mask], omega[mask], axis=1)
        summarize_vector(names, values, f"reG_low_abs_integral_abs_omega_le_{window:g}", low_abs)

    max_idx = np.argmax(path_average)
    min_idx = np.argmin(path_average)
    add_feature(names, values, "reG_pathavg_max_omega", omega[max_idx])
    add_feature(names, values, "reG_pathavg_max_value", path_average[max_idx])
    add_feature(names, values, "reG_pathavg_min_omega", omega[min_idx])
    add_feature(names, values, "reG_pathavg_min_value", path_average[min_idx])

    signs = np.signbit(re_g)
    sign_changes = np.count_nonzero(signs[:, 1:] != signs[:, :-1], axis=1)
    summarize_vector(names, values, "reG_sign_change_count", sign_changes, include_minmax=True)

    return names, values


def extract_features(
    dataset: Path,
    v_min: float | None = None,
    v_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    z = np.load(dataset, allow_pickle=True)
    uv = z["uv"]
    rho = z["rho_G"]
    re_g = z["reG"]
    omega = z["omega"]
    mask = np.ones(uv.shape[0], dtype=bool)
    if v_min is not None:
        mask &= uv[:, 1] >= v_min
    if v_max is not None:
        mask &= uv[:, 1] <= v_max
    uv = uv[mask]
    rho = rho[mask]
    re_g = re_g[mask]
    if uv.size == 0:
        raise ValueError("No samples remain after applying V filters.")

    all_rows: list[list[float]] = []
    feature_names: list[str] | None = None
    for idx in range(uv.shape[0]):
        rho_names, rho_values = rho_features(rho[idx], omega)
        re_names, re_values = re_g_features(re_g[idx], omega)
        names = rho_names + re_names
        row = rho_values + re_values
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise RuntimeError("Feature names changed across samples.")
        all_rows.append(row)

    features = np.asarray(all_rows, dtype=float)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return uv, features, list(feature_names or [])


def feature_indices(feature_names: list[str], block: str) -> np.ndarray:
    if block == "rho":
        return np.asarray([idx for idx, name in enumerate(feature_names) if name.startswith("rho_")])
    if block == "reG":
        return np.asarray([idx for idx, name in enumerate(feature_names) if name.startswith("reG_")])
    if block == "combined":
        return np.arange(len(feature_names))
    raise ValueError(f"Unknown feature block {block!r}")


def deterministic_split(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_samples)
    test_mask = indices % 5 == 0
    return indices[~test_mask], indices[test_mask]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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


def make_model(name: str, random_state: int):
    if name == "ridge":
        alphas = np.logspace(-4, 4, 25)
        return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=3,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model {name!r}")


def cross_validated_predictions(
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    random_state: int,
) -> np.ndarray:
    predictions = np.zeros_like(y, dtype=float)
    splitter = KFold(n_splits=5, shuffle=True, random_state=random_state)
    for train_idx, test_idx in splitter.split(x):
        model = make_model(model_name, random_state)
        model.fit(x[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(x[test_idx])
    return predictions


def standardized_ridge_importance(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    top_n: int = 12,
) -> dict[str, list[dict[str, float | str]]]:
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 25)))
    model.fit(x_train, y_train)
    ridge = model.named_steps["ridgecv"]
    coefficients = np.asarray(ridge.coef_)
    result: dict[str, list[dict[str, float | str]]] = {}
    for target_idx, target in enumerate(["U", "V"]):
        order = np.argsort(np.abs(coefficients[target_idx]))[::-1][:top_n]
        result[target] = [
            {"feature": feature_names[idx], "coefficient": float(coefficients[target_idx, idx])}
            for idx in order
        ]
    return result


def write_feature_csv(path: Path, uv: np.ndarray, features: np.ndarray, feature_names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["U", "V", *feature_names])
        for target, row in zip(uv, features):
            writer.writerow([float(target[0]), float(target[1]), *[float(x) for x in row]])


def plot_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, idx, label in zip(axes, [0, 1], ["U", "V"]):
        ax.scatter(y_true[:, idx], y_pred[:, idx], s=42, edgecolor="black", linewidth=0.25)
        lo = min(float(y_true[:, idx].min()), float(y_pred[:, idx].min()))
        hi = max(float(y_true[:, idx].max()), float(y_pred[:, idx].max()))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_xlabel(f"true {label}")
        ax.set_ylabel(f"predicted {label}")
        ax.set_title(label)
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    uv, features, feature_names = extract_features(args.dataset, args.v_min, args.v_max)
    train_idx, test_idx = deterministic_split(uv.shape[0])

    np.savez(
        args.output_dir / "physics_features.npz",
        uv=uv,
        features=features,
        feature_names=np.asarray(feature_names),
        target_columns=np.asarray(["U", "V"]),
    )
    write_feature_csv(args.output_dir / "physics_features.csv", uv, features, feature_names)

    summary: dict[str, object] = {
        "dataset": str(args.dataset),
        "v_min": args.v_min,
        "v_max": args.v_max,
        "n_samples": int(uv.shape[0]),
        "n_features": int(features.shape[1]),
        "n_rho_features": int(len(feature_indices(feature_names, "rho"))),
        "n_reG_features": int(len(feature_indices(feature_names, "reG"))),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "feature_file": str(args.output_dir / "physics_features.npz"),
        "feature_csv": str(args.output_dir / "physics_features.csv"),
        "blocks": {},
    }

    for block in ["rho", "reG", "combined"]:
        block_idx = feature_indices(feature_names, block)
        block_names = [feature_names[idx] for idx in block_idx]
        x = features[:, block_idx]
        block_summary: dict[str, object] = {
            "n_features": int(x.shape[1]),
            "models": {},
        }
        for model_name in ["ridge", "random_forest"]:
            model = make_model(model_name, args.random_state)
            model.fit(x[train_idx], uv[train_idx])
            holdout_pred = model.predict(x[test_idx])
            cv_pred = cross_validated_predictions(x, uv, model_name, args.random_state)
            block_summary["models"][model_name] = {
                "holdout": metrics(uv[test_idx], holdout_pred),
                "five_fold_cv": metrics(uv, cv_pred),
            }
            np.savez(
                args.output_dir / f"{block}_{model_name}_predictions.npz",
                train_idx=train_idx,
                test_idx=test_idx,
                holdout_true=uv[test_idx],
                holdout_pred=holdout_pred,
                cv_true=uv,
                cv_pred=cv_pred,
            )
            if not args.no_plots:
                plot_prediction_scatter(
                    uv[test_idx],
                    holdout_pred,
                    f"{block} {model_name} physics features",
                    args.output_dir / f"{block}_{model_name}_holdout_predictions.png",
                )

        block_summary["ridge_top_features"] = standardized_ridge_importance(
            x[train_idx], uv[train_idx], block_names
        )
        summary["blocks"][block] = block_summary

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
