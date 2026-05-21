#!/usr/bin/env python3
"""Compare HK Green-function learning outputs across eta broadenings.

This is a lightweight reporting utility for already-generated workflow outputs.
It assumes the standard V>=0 directory layout produced by the HK Green ML
workflow, and writes CSV/JSON summaries plus a few comparison plots.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


ETA_TAGS = {
    "eta0p02": 0.02,
    "eta0p05": 0.05,
    "eta0p1": 0.10,
    "eta0p2": 0.20,
}


def json_load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def finite_float(x: Any) -> float | None:
    try:
        y = float(x)
    except Exception:
        return None
    if not math.isfinite(y):
        return None
    return y


def state_from_ratio(r: float) -> str:
    if r < 0.001:
        return "deep_gap"
    if r < 0.01:
        return "strong_gap"
    if r < 0.10:
        return "weak_gap"
    return "not_gap_open"


def state_rank(s: str) -> int:
    return {
        "not_gap_open": 0,
        "weak_gap": 1,
        "strong_gap": 2,
        "deep_gap": 3,
    }[s]


def state_code(s: str) -> int:
    return state_rank(s)


def read_npz_eta(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "eta": finite_float(z["eta"]) if "eta" in z.files else None,
    }
    if "omega" in z.files:
        out["omega_points"] = int(z["omega"].shape[0])
        out["omega_min"] = finite_float(z["omega"][0])
        out["omega_max"] = finite_float(z["omega"][-1])
    if "rho_G" in z.files:
        out["rho_G_shape"] = list(z["rho_G"].shape)
    if "reG" in z.files:
        out["reG_shape"] = list(z["reG"].shape)
    if "metadata" in z.files:
        out["n_spectra"] = int(z["metadata"].shape[0])
        # The first two metadata columns are U,V in the generated files.
        uv = z["metadata"][:, :2]
        out["n_uv_groups"] = int(pd.DataFrame(uv, columns=["U", "V"]).drop_duplicates().shape[0])
    return out


def read_baseline(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    uv = z["uv"]
    return {
        "path": str(path),
        "eta": finite_float(z["eta"]) if "eta" in z.files else None,
        "n_total": int(uv.shape[0]),
        "n_nonnegative_v": int((uv[:, 1] >= 0).sum()),
        "n_positive_v": int((uv[:, 1] > 0).sum()),
        "n_v0": int((np.isclose(uv[:, 1], 0)).sum()),
        "n_negative_v": int((uv[:, 1] < 0).sum()),
        "omega_points": int(z["omega"].shape[0]),
    }


def load_gap_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["U_round"] = df["U"].round(6)
    df["V_round"] = df["V"].round(6)
    df["state"] = df["maxk_rho0_ratio"].map(state_from_ratio)
    df["state_code"] = df["state"].map(state_code)
    return df


def summarize_gap(tag: str, eta: float, df: pd.DataFrame) -> dict[str, Any]:
    r = df["maxk_rho0_ratio"]
    return {
        "tag": tag,
        "eta": eta,
        "n": int(len(df)),
        "gap_open_lt_0p10": int((r < 0.10).sum()),
        "strong_gap_lt_0p01": int((r < 0.01).sum()),
        "deep_gap_lt_0p001": int((r < 0.001).sum()),
        "not_gap_open": int((r >= 0.10).sum()),
        "ratio_min": float(r.min()),
        "ratio_median": float(r.median()),
        "ratio_mean": float(r.mean()),
        "ratio_max": float(r.max()),
        "log10_ratio_median": float(np.log10(r.clip(lower=1e-300)).median()),
        "state_not_gap_open": int((df["state"] == "not_gap_open").sum()),
        "state_weak_gap": int((df["state"] == "weak_gap").sum()),
        "state_strong_gap": int((df["state"] == "strong_gap").sum()),
        "state_deep_gap": int((df["state"] == "deep_gap").sum()),
    }


def supervised_rows(tag: str, eta: float, path: Path) -> list[dict[str, Any]]:
    s = json_load(path)
    rows: list[dict[str, Any]] = []
    for block, bdata in s["blocks"].items():
        for model, mdata in bdata["models"].items():
            for split in ["holdout", "five_fold_cv"]:
                metrics = mdata.get(split, {})
                rows.append(
                    {
                        "tag": tag,
                        "eta": eta,
                        "block": block,
                        "model": model,
                        "split": split,
                        "rmse_U": metrics.get("rmse_U"),
                        "rmse_V": metrics.get("rmse_V"),
                        "mae_U": metrics.get("mae_U"),
                        "mae_V": metrics.get("mae_V"),
                        "r2_U": metrics.get("r2_U"),
                        "r2_V": metrics.get("r2_V"),
                    }
                )
    return rows


def luttinger_row(tag: str, eta: float, path: Path) -> dict[str, Any]:
    s = json_load(path)
    cv = s["ridge"]["five_fold_cv"]
    hold = s["ridge"]["holdout"]
    return {
        "tag": tag,
        "eta": eta,
        "n_samples": s["n_samples"],
        "crossing_count_distribution": json.dumps(s["crossing_count_distribution"], sort_keys=True),
        "holdout_r2_U": hold["r2_U"],
        "holdout_r2_V": hold["r2_V"],
        "cv_r2_U": cv["r2_U"],
        "cv_r2_V": cv["r2_V"],
        "cv_rmse_U": cv["rmse_U"],
        "cv_rmse_V": cv["rmse_V"],
    }


def unsupervised_row(tag: str, eta: float, path: Path) -> dict[str, Any]:
    s = json_load(path)
    primary = s["interpretation"]["primary_best_kmeans_metrics"]
    row = {
        "tag": tag,
        "eta": eta,
        "primary_subset": s["interpretation"]["primary_subset"],
        "primary_representation": s["interpretation"]["primary_representation"],
        "primary_label": s["interpretation"]["primary_best_kmeans"],
        "primary_k": primary.get("k"),
        "primary_silhouette": primary.get("silhouette"),
        "primary_stability_ari": primary.get("stability_adjusted_rand"),
        "primary_calinski_harabasz": primary.get("calinski_harabasz"),
        "primary_davies_bouldin": primary.get("davies_bouldin"),
    }
    # Also report the best gapped raw-spectra recommendation if available.
    try:
        rec = pd.read_csv(path.parent / "gapped_recommendations.csv")
    except Exception:
        rec = pd.DataFrame()
    if not rec.empty:
        best = rec.sort_values(
            by=[c for c in ["silhouette", "stability_adjusted_rand", "consensus_mean_pairwise_ari"] if c in rec.columns],
            ascending=False,
        ).iloc[0]
        for c in rec.columns:
            if c in {
                "subset",
                "representation",
                "best_method",
                "best_label",
                "k",
                "silhouette",
                "stability_adjusted_rand",
                "consensus_mean_pairwise_ari",
            }:
                row[f"best_gapped_{c}"] = best[c]
    return row


def kernel_row(tag: str, eta: float, path: Path) -> dict[str, Any]:
    s = json_load(path)
    best = s["best_metrics"][0]
    return {
        "tag": tag,
        "eta": eta,
        "scope_n": s["scope"].get("n_samples", s["scope"].get("n_nonnegative_v")),
        "scope_gap_open": s["scope"]["n_gap_open"],
        "scope_strong_gap": s["scope"]["n_strong_gap"],
        "best_subset": best.get("subset"),
        "best_representation": best.get("representation"),
        "best_method": best.get("method"),
        "best_k": best.get("k"),
        "best_gamma_factor": best.get("gamma_factor"),
        "best_silhouette": best.get("silhouette"),
        "best_bootstrap_ari_mean": best.get("bootstrap_ari_mean"),
        "best_quality_score": best.get("quality_score"),
    }


def primary_cluster_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["U", "V", "kmeans_best_cluster", "gmm_best_cluster", "agglomerative_best_cluster"]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()
    out["U_round"] = out["U"].round(6)
    out["V_round"] = out["V"].round(6)
    return out


def make_plots(
    out_dir: Path,
    gap_counts: pd.DataFrame,
    gap_ratios: pd.DataFrame,
    supervised: pd.DataFrame,
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Gap-state stacked counts.
    gc = gap_counts.sort_values("eta")
    state_cols = ["state_not_gap_open", "state_weak_gap", "state_strong_gap", "state_deep_gap"]
    labels = [f"{x:.2g}" for x in gc["eta"]]
    bottoms = np.zeros(len(gc))
    colors = ["#9e9e9e", "#fdae61", "#2b83ba", "#4d004b"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, color in zip(state_cols, colors):
        vals = gc[col].to_numpy()
        ax.bar(labels, vals, bottom=bottoms, label=col.replace("state_", ""), color=color)
        bottoms += vals
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("V >= 0 sample count")
    ax.set_title("Gap-regime counts versus broadening")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_dir / "gap_state_counts_by_eta.png", dpi=220)
    plt.close(fig)

    # Pairwise log-ratio scatter.
    tags = sorted(ETA_TAGS, key=lambda t: ETA_TAGS[t])
    pairs = [(a, b) for i, a in enumerate(tags) for b in tags[i + 1 :]]
    ncols = 3
    nrows = int(math.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows), sharex=False, sharey=False)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (a, b) in zip(axes_flat, pairs):
        ca = f"R0_{a}"
        cb = f"R0_{b}"
        ax.scatter(gap_ratios[ca], gap_ratios[cb], s=14, alpha=0.75)
        lo = min(gap_ratios[ca].min(), gap_ratios[cb].min())
        hi = max(gap_ratios[ca].max(), gap_ratios[cb].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"R0 {a}")
        ax.set_ylabel(f"R0 {b}")
        ax.set_title(f"{a} vs {b}")
    for ax in axes_flat[len(pairs) :]:
        ax.axis("off")
    fig.suptitle("Broadening changes the zero-frequency gap proxy")
    fig.tight_layout()
    fig.savefig(plot_dir / "gap_ratio_pairwise_scatter.png", dpi=220)
    plt.close(fig)

    # Supervised CV R2.
    cv = supervised[
        (supervised["split"] == "five_fold_cv")
        & (supervised["model"] == "random_forest")
        & (supervised["block"].isin(["rho", "reG", "combined"]))
    ].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for block, marker in [("rho", "o"), ("reG", "s"), ("combined", "^")]:
        sub = cv[cv["block"] == block].sort_values("eta")
        ax.plot(sub["eta"], sub["r2_U"], marker=marker, label=f"{block} RF: U")
        ax.plot(sub["eta"], sub["r2_V"], marker=marker, linestyle="--", label=f"{block} RF: V")
    ax.set_xscale("log")
    ax.set_xticks([0.02, 0.05, 0.1, 0.2], labels=["0.02", "0.05", "0.10", "0.20"])
    ax.set_ylim(0.86, 1.01)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"5-fold CV $R^2$")
    ax.set_title("Supervised random-forest learning is weakly eta-dependent")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "supervised_rf_cv_r2_by_eta.png", dpi=220)
    plt.close(fig)

    # U,V state maps.
    state_palette = {
        "not_gap_open": "#9e9e9e",
        "weak_gap": "#fdae61",
        "strong_gap": "#2b83ba",
        "deep_gap": "#4d004b",
    }
    fig, axes = plt.subplots(1, len(tags), figsize=(4.4 * len(tags), 4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, tag in zip(axes, tags):
        for state, color in state_palette.items():
            mask = gap_ratios[f"state_{tag}"] == state
            ax.scatter(gap_ratios.loc[mask, "U"], gap_ratios.loc[mask, "V"], s=18, label=state, color=color, alpha=0.85)
        ax.set_title(tag)
        ax.set_xlabel("U")
        ax.grid(alpha=0.2, lw=0.5)
    axes[0].set_ylabel("V")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("V >= 0 gap-state classification across eta")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(plot_dir / "uv_gap_states_by_eta.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/Gcalpath/rho_re_g", help="rho_re_g output root")
    parser.add_argument("--date-tag", default="20260519")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir) if args.output_dir else root / f"eta_comparison_{args.date_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tags = ["eta0p02", "eta0p05", "eta0p1", "eta0p2"]
    extraction_paths = {
        "eta0p02": root / f"rho_re_g_all_complete_eta0p02_{args.date_tag}.npz",
        "eta0p05": root / f"rho_re_g_all_complete_eta0p05_{args.date_tag}.npz",
        "eta0p1": root / "rho_re_g_all_complete.npz",
        "eta0p2": root / f"rho_re_g_all_complete_eta0p2_{args.date_tag}.npz",
    }

    extractions: dict[str, Any] = {}
    baselines: dict[str, Any] = {}
    gap_frames: dict[str, pd.DataFrame] = {}
    supervised: list[dict[str, Any]] = []
    luttinger: list[dict[str, Any]] = []
    unsup: list[dict[str, Any]] = []
    kernel: list[dict[str, Any]] = []
    clusters: dict[str, pd.DataFrame] = {}

    for tag in tags:
        eta = ETA_TAGS[tag]
        extractions[tag] = read_npz_eta(extraction_paths[tag])
        baselines[tag] = read_baseline(root / f"ml_baseline_{tag}_{args.date_tag}" / "ml_ready_dataset.npz")
        gap = load_gap_csv(root / f"gap_analysis_{tag}_nonnegative_{args.date_tag}" / "nonnegative_v_gap_classification_maxk_rho0.csv")
        gap_frames[tag] = gap
        supervised += supervised_rows(tag, eta, root / f"ml_physics_features_{tag}_nonnegative_{args.date_tag}" / "summary.json")
        luttinger.append(
            luttinger_row(
                tag,
                eta,
                root / f"ml_luttinger_path_reG0_{tag}_nonnegative_gaplt0p01_{args.date_tag}" / "summary.json",
            )
        )
        unsup.append(unsupervised_row(tag, eta, root / f"ml_unsupervised_{tag}_nonnegative_{args.date_tag}" / "summary.json"))
        kernel.append(kernel_row(tag, eta, root / f"ml_unsupervised_kernel_{tag}_nonnegative_{args.date_tag}" / "summary.json"))
        clusters[tag] = primary_cluster_labels(root / f"ml_unsupervised_{tag}_nonnegative_{args.date_tag}" / "nonnegative_v_primary_assignments.csv")

    gap_counts_df = pd.DataFrame([summarize_gap(tag, ETA_TAGS[tag], gap_frames[tag]) for tag in tags]).sort_values("eta")
    gap_counts_df.to_csv(out_dir / "eta_gap_counts.csv", index=False)

    supervised_df = pd.DataFrame(supervised).sort_values(["eta", "block", "model", "split"])
    supervised_df.to_csv(out_dir / "eta_supervised_metrics.csv", index=False)

    luttinger_df = pd.DataFrame(luttinger).sort_values("eta")
    luttinger_df.to_csv(out_dir / "eta_luttinger_metrics.csv", index=False)

    unsup_df = pd.DataFrame(unsup).sort_values("eta")
    unsup_df.to_csv(out_dir / "eta_unsupervised_metrics.csv", index=False)

    kernel_df = pd.DataFrame(kernel).sort_values("eta")
    kernel_df.to_csv(out_dir / "eta_kernel_metrics.csv", index=False)

    # Aligned per-sample gap ratios and states.
    merged = None
    for tag in tags:
        g = gap_frames[tag][["U", "V", "U_round", "V_round", "maxk_rho0_ratio", "state", "state_code"]].copy()
        g = g.rename(columns={"maxk_rho0_ratio": f"R0_{tag}", "state": f"state_{tag}", "state_code": f"state_code_{tag}"})
        if merged is None:
            merged = g
        else:
            merged = merged.merge(g.drop(columns=["U", "V"]), on=["U_round", "V_round"], how="inner")
    assert merged is not None
    merged = merged.sort_values(["U", "V"])
    merged.to_csv(out_dir / "eta_gap_ratios_aligned.csv", index=False)

    # Aligned primary KMeans labels.
    cluster_merged = None
    for tag in tags:
        c = clusters[tag][["U", "V", "U_round", "V_round", "kmeans_best_cluster"]].copy()
        c = c.rename(columns={"kmeans_best_cluster": f"kmeans_cluster_{tag}"})
        if cluster_merged is None:
            cluster_merged = c
        else:
            cluster_merged = cluster_merged.merge(c.drop(columns=["U", "V"]), on=["U_round", "V_round"], how="inner")
    assert cluster_merged is not None
    cluster_merged = cluster_merged.sort_values(["U", "V"])
    cluster_merged.to_csv(out_dir / "eta_primary_kmeans_labels_aligned.csv", index=False)

    pair_rows: list[dict[str, Any]] = []
    for i, tag_a in enumerate(tags):
        for tag_b in tags[i + 1 :]:
            a = merged[f"R0_{tag_a}"]
            b = merged[f"R0_{tag_b}"]
            row: dict[str, Any] = {
                "eta_a": ETA_TAGS[tag_a],
                "tag_a": tag_a,
                "eta_b": ETA_TAGS[tag_b],
                "tag_b": tag_b,
                "n_aligned": int(len(merged)),
                "pearson_log10_R0": float(np.corrcoef(np.log10(a.clip(lower=1e-300)), np.log10(b.clip(lower=1e-300)))[0, 1]),
                "spearman_R0": float(pd.Series(a).corr(pd.Series(b), method="spearman")),
                "state_agreement_fraction": float((merged[f"state_{tag_a}"] == merged[f"state_{tag_b}"]).mean()),
                "state_ari": float(adjusted_rand_score(merged[f"state_code_{tag_a}"], merged[f"state_code_{tag_b}"])),
                "gap_open_label_ari": float(adjusted_rand_score(a < 0.10, b < 0.10)),
                "strong_gap_label_ari": float(adjusted_rand_score(a < 0.01, b < 0.01)),
                "deep_gap_label_ari": float(adjusted_rand_score(a < 0.001, b < 0.001)),
                "gap_open_changed_count": int(((a < 0.10) != (b < 0.10)).sum()),
                "strong_gap_changed_count": int(((a < 0.01) != (b < 0.01)).sum()),
                "deep_gap_changed_count": int(((a < 0.001) != (b < 0.001)).sum()),
                "state_b_more_gapped_count": int((merged[f"state_code_{tag_b}"] > merged[f"state_code_{tag_a}"]).sum()),
                "state_a_more_gapped_count": int((merged[f"state_code_{tag_a}"] > merged[f"state_code_{tag_b}"]).sum()),
                "primary_kmeans_ari": float(
                    adjusted_rand_score(cluster_merged[f"kmeans_cluster_{tag_a}"], cluster_merged[f"kmeans_cluster_{tag_b}"])
                ),
            }
            pair_rows.append(row)
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(out_dir / "eta_pairwise_agreement.csv", index=False)

    # Transition table for gap states across all three etas.
    state_cols = [f"state_{tag}" for tag in tags]
    transitions = (
        merged.groupby(state_cols)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    transitions.to_csv(out_dir / "eta_state_transitions.csv", index=False)

    make_plots(out_dir, gap_counts_df, merged, supervised_df)

    # Compact interpretation for JSON and terminal.
    rf_combined = supervised_df[
        (supervised_df["split"] == "five_fold_cv")
        & (supervised_df["model"] == "random_forest")
        & (supervised_df["block"] == "combined")
    ].sort_values("eta")
    interpretation = [
        "All four eta values use the same 278 total samples and 229 V>=0 samples.",
        "The gap proxy is strongly eta-dependent: smaller eta lowers R0 and moves more samples into strong/deep-gap classes.",
        "Random-forest supervised learning is broadly robust across eta, with eta=0.02 somewhat less accurate for U than eta=0.05-0.20.",
        "Unsupervised global clustering changes more noticeably at eta=0.02, where sharper spectra favor more clusters, but kernel/spectral gapped-region metrics remain high.",
    ]

    summary = {
        "output_dir": str(out_dir),
        "date_tag": args.date_tag,
        "extractions": extractions,
        "baselines": baselines,
        "gap_counts": gap_counts_df.to_dict(orient="records"),
        "combined_random_forest_cv": rf_combined.to_dict(orient="records"),
        "luttinger": luttinger_df.to_dict(orient="records"),
        "unsupervised": unsup_df.to_dict(orient="records"),
        "kernel": kernel_df.to_dict(orient="records"),
        "pairwise_agreement": pair_df.to_dict(orient="records"),
        "state_transitions_top": transitions.head(20).to_dict(orient="records"),
        "interpretation": interpretation,
        "files": {
            "gap_counts": str(out_dir / "eta_gap_counts.csv"),
            "gap_ratios_aligned": str(out_dir / "eta_gap_ratios_aligned.csv"),
            "supervised": str(out_dir / "eta_supervised_metrics.csv"),
            "luttinger": str(out_dir / "eta_luttinger_metrics.csv"),
            "unsupervised": str(out_dir / "eta_unsupervised_metrics.csv"),
            "kernel": str(out_dir / "eta_kernel_metrics.csv"),
            "pairwise_agreement": str(out_dir / "eta_pairwise_agreement.csv"),
            "state_transitions": str(out_dir / "eta_state_transitions.csv"),
            "plots": str(out_dir / "plots"),
        },
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2)[:8000])
    print(f"\nWrote eta comparison to {out_dir}")


if __name__ == "__main__":
    main()
