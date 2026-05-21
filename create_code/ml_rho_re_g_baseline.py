#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/Gcalpath/rho_re_g/rho_re_g_all_complete.npz"
DEFAULT_OUTPUT_DIR = ROOT / "data/Gcalpath/rho_re_g/ml_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run first-pass PCA and ridge-regression baselines on rho_G/reG data."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG diagnostics.")
    return parser.parse_args()


def load_samples(dataset: Path) -> dict[str, np.ndarray]:
    z = np.load(dataset, allow_pickle=True)
    metadata = z["metadata"]
    columns = [str(item) for item in z["metadata_columns"]]
    col = {name: idx for idx, name in enumerate(columns)}

    uv = metadata[:, [col["U"], col["V"]]]
    unique_uv, first_idx = np.unique(uv, axis=0, return_index=True)
    order = np.argsort(first_idx)
    uv = unique_uv[order]

    n_samples = uv.shape[0]
    n_path = metadata.shape[0] // n_samples
    n_omega = z["omega"].shape[0]
    expected_rows = n_samples * n_path
    if metadata.shape[0] != expected_rows:
        raise ValueError("Metadata rows do not form a rectangular sample x k-path array.")

    metadata_blocks = metadata.reshape(n_samples, n_path, metadata.shape[1])
    reshaped_uv = metadata_blocks[:, 0, :][:, [col["U"], col["V"]]]
    if not np.allclose(uv, reshaped_uv):
        raise ValueError("Rows are not ordered as contiguous 121-point blocks for each (U,V).")

    rho_g = z["rho_G"].reshape(n_samples, n_path, n_omega)
    re_g = z["reG"].reshape(n_samples, n_path, n_omega)
    k_path = metadata_blocks[0, :, :][:, [col["kx"], col["ky"]]]

    return {
        "uv": uv,
        "rho_G": rho_g,
        "reG": re_g,
        "omega": z["omega"],
        "k_path": k_path,
        "eta": np.asarray(z["eta"]),
        "steps_max": np.asarray(z["steps_max"]),
        "q": np.asarray(z["q"]),
    }


def save_ml_dataset(samples: dict[str, np.ndarray], output: Path) -> None:
    """Save a compact sample-level tensor package for downstream ML runs."""
    np.savez(
        output,
        uv=samples["uv"],
        rho_G=samples["rho_G"],
        reG=samples["reG"],
        omega=samples["omega"],
        k_path=samples["k_path"],
        eta=samples["eta"],
        steps_max=samples["steps_max"],
        q=samples["q"],
        sample_axis=np.asarray(["sample", "k_path", "omega"]),
        target_columns=np.asarray(["U", "V"]),
    )


def flatten_representation(samples: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name == "rho_G":
        return samples["rho_G"].reshape(samples["rho_G"].shape[0], -1)
    if name == "reG":
        return samples["reG"].reshape(samples["reG"].shape[0], -1)
    if name == "rho_G_plus_reG":
        return np.concatenate(
            [
                flatten_representation(samples, "rho_G"),
                flatten_representation(samples, "reG"),
            ],
            axis=1,
        )
    raise ValueError(f"Unknown representation {name!r}")


def standardize_train_apply(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1.0e-12] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def pca_fit(x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    x_centered = x - mean
    gram = x_centered @ x_centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]

    keep = min(n_components, np.count_nonzero(eigenvalues > 1.0e-12))
    singular_values = np.sqrt(eigenvalues[:keep])
    components = (x_centered.T @ eigenvectors[:, :keep]).T
    components /= singular_values[:, None]

    variances = eigenvalues / max(x.shape[0] - 1, 1)
    explained = variances / variances.sum()
    return mean, components, explained[:keep]


def pca_transform(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (x - mean) @ components.T


def pca_fit_transform(x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, components, explained = pca_fit(x, n_components)
    scores = pca_transform(x, mean, components)
    return scores, components, explained


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_aug = np.c_[np.ones(x_train.shape[0]), x_train]
    test_aug = np.c_[np.ones(x_test.shape[0]), x_test]
    penalty = np.eye(x_aug.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y_train)
    return test_aug @ weights


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    rmse = np.sqrt(np.mean(residual**2, axis=0))
    mae = np.mean(np.abs(residual), axis=0)
    ss_res = np.sum(residual**2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / ss_tot
    return {
        "rmse_U": float(rmse[0]),
        "rmse_V": float(rmse[1]),
        "mae_U": float(mae[0]),
        "mae_V": float(mae[1]),
        "r2_U": float(r2[0]),
        "r2_V": float(r2[1]),
    }


def make_train_test_split(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Deterministic holdout spread across the ordered parameter grid.
    indices = np.arange(uv.shape[0])
    test_mask = indices % 5 == 0
    return indices[~test_mask], indices[test_mask]


def plot_pca(scores: np.ndarray, uv: np.ndarray, explained: np.ndarray, title: str, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, color_values, label in zip(axes, [uv[:, 0], uv[:, 1]], ["U", "V"]):
        sc = ax.scatter(scores[:, 0], scores[:, 1], c=color_values, cmap="viridis", s=42, edgecolor="black", linewidth=0.25)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"colored by {label}")
        fig.colorbar(sc, ax=ax, label=label)
    fig.suptitle(f"{title}: PC1 {explained[0]:.1%}, PC2 {explained[1]:.1%}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, idx, label in zip(axes, [0, 1], ["U", "V"]):
        ax.scatter(y_true[:, idx], y_pred[:, idx], s=42, edgecolor="black", linewidth=0.25)
        lo = min(y_true[:, idx].min(), y_pred[:, idx].min())
        hi = max(y_true[:, idx].max(), y_pred[:, idx].max())
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
    samples = load_samples(args.dataset)
    print(f"Loaded {args.dataset}", flush=True)
    save_ml_dataset(samples, args.output_dir / "ml_ready_dataset.npz")
    print(f"Saved {args.output_dir / 'ml_ready_dataset.npz'}", flush=True)
    uv = samples["uv"]
    train_idx, test_idx = make_train_test_split(uv)

    summary = {
        "dataset": str(args.dataset),
        "n_samples": int(uv.shape[0]),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "n_k_path": int(samples["k_path"].shape[0]),
        "n_omega": int(samples["omega"].shape[0]),
        "omega_min": float(samples["omega"].min()),
        "omega_max": float(samples["omega"].max()),
        "eta": float(samples["eta"]),
        "steps_max": int(samples["steps_max"]),
        "q": [int(x) for x in samples["q"]],
        "ml_ready_dataset": str(args.output_dir / "ml_ready_dataset.npz"),
        "representations": {},
    }

    for name in ["rho_G", "reG", "rho_G_plus_reG"]:
        print(f"Processing {name}", flush=True)
        x = flatten_representation(samples, name)
        x_all, _, _, _ = standardize_train_apply(x, x)

        scores, _, explained = pca_fit_transform(x_all, args.n_components)
        print(f"  PCA complete for {name}", flush=True)

        x_train, x_test, _, _ = standardize_train_apply(x[train_idx], x[test_idx])
        train_pca_mean, train_components, train_explained = pca_fit(
            x_train, min(args.n_components, x_train.shape[0])
        )
        train_scores = pca_transform(x_train, train_pca_mean, train_components)
        test_scores = pca_transform(x_test, train_pca_mean, train_components)
        y_pred = ridge_predict(train_scores, uv[train_idx], test_scores, args.ridge_alpha)
        metrics = regression_metrics(uv[test_idx], y_pred)
        print(f"  Ridge complete for {name}", flush=True)

        np.savez(
            args.output_dir / f"{name}_pca_scores.npz",
            uv=uv,
            scores=scores,
            explained_variance_ratio=explained,
        )
        np.savez(
            args.output_dir / f"{name}_ridge_predictions.npz",
            train_idx=train_idx,
            test_idx=test_idx,
            y_true=uv[test_idx],
            y_pred=y_pred,
            train_explained_variance_ratio=train_explained,
        )
        if not args.no_plots:
            plot_pca(scores, uv, explained, name, args.output_dir / f"{name}_pca.png")
            plot_predictions(
                uv[test_idx],
                y_pred,
                f"{name} ridge on PCA scores",
                args.output_dir / f"{name}_ridge_predictions.png",
            )
            print(f"  Plots saved for {name}", flush=True)

        summary["representations"][name] = {
            "pca_explained_variance_ratio": [float(x) for x in explained],
            "pca_explained_variance_ratio_cumulative": [float(x) for x in np.cumsum(explained)],
            "regression_train_pca_explained_variance_ratio": [float(x) for x in train_explained],
            "regression_train_pca_explained_variance_ratio_cumulative": [
                float(x) for x in np.cumsum(train_explained)
            ],
            "ridge_on_pca": metrics,
        }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
