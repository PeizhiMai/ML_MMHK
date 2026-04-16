#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FILE_RE = re.compile(
    r"L(?P<Lx>\d+)x(?P<Ly>\d+)Nup(?P<Nup>\d+)Ndn(?P<Ndn>\d+)"
    r"U(?P<U>-?\d+(?:\.\d+)?)V(?P<V>-?\d+(?:\.\d+)?)"
    r"tp(?P<tp>-?\d+(?:\.\d+)?)f(?P<f>[^Q]+)Q(?P<Q>[^_]+)_(?P<kind>G[hp])\.dat"
)

PANEL_XLIMS = {
    (8.0, -4.0): (-25.0, -5.0),
    (8.0, -2.0): (-20.0, 0.0),
    (8.0, 0.0): (-10.0, 10.0),
    (8.0, 2.0): (-5.0, 20.0),
    (8.0, 4.0): (0.0, 30.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot spectral functions for all available (U,V) sets versus Lanczos iteration count."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing *_Gh.dat and *_Gp.dat files.",
    )
    parser.add_argument(
        "--omega-max",
        type=float,
        default=20.0,
        help="Maximum absolute frequency shown in the plot.",
    )
    parser.add_argument(
        "--omega-points",
        type=int,
        default=601,
        help="Number of omega grid points.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.05,
        help="Lorentzian broadening eta.",
    )
    parser.add_argument(
        "--iter-start",
        type=int,
        default=1000,
        help="Starting Lanczos length.",
    )
    parser.add_argument(
        "--iter-stop",
        type=int,
        default=2000,
        help="Maximum Lanczos length.",
    )
    parser.add_argument(
        "--iter-step",
        type=int,
        default=200,
        help="Lanczos length spacing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path. Defaults to <data-dir>/spectral_function_vs_iteration.png",
    )
    return parser.parse_args()


def parse_green_file(filename: Path) -> dict:
    contents = filename.read_text().splitlines()
    data = {
        "Lx": None,
        "Ly": None,
        "tp": None,
        "U": None,
        "V": None,
        "E0": None,
        "Gap": None,
        "Q": [],
        "nrm2": [],
        "a": [],
        "b": [],
    }

    for line in contents:
        line_data = line.split()
        if not line_data:
            continue
        key = line_data[0]
        if key in {"L", "Lx"}:
            data["Lx"] = int(line_data[1])
        elif key == "Ly":
            data["Ly"] = int(line_data[1])
        elif key == "tp":
            data["tp"] = float(line_data[1])
        elif key == "U":
            data["U"] = float(line_data[1])
        elif key == "V":
            data["V"] = float(line_data[1])
        elif key == "E0":
            data["E0"] = float(line_data[1])
        elif key == "Gap":
            data["Gap"] = float(line_data[1])
        elif key == "Q":
            data["Q"].append(tuple(int(x) for x in line_data[1].split(",")))
        elif key == "nrm2":
            data["nrm2"].append(float(line_data[1]))
        elif key == "a":
            data["a"].append(np.array([float(x) for x in line_data[1:]], dtype=float))
        elif key == "b":
            data["b"].append(np.array([float(x) for x in line_data[1:]], dtype=float))

    return data


def spectral_component(
    omega: np.ndarray,
    eta: float,
    coeff_a: np.ndarray,
    coeff_b: np.ndarray,
    length: int,
    nrm2: float,
    e0: float,
    u: float,
    kind: str,
) -> np.ndarray:
    omega_complex = omega.astype(complex) + 1j * eta
    if kind == "Gh":
        shifted = omega_complex + u / 2 - e0
        a_eff = shifted[:, None] + coeff_a[:length][None, :]
    else:
        shifted = omega_complex + u / 2 + e0
        a_eff = shifted[:, None] - coeff_a[:length][None, :]

    res = np.zeros_like(omega_complex)
    for j in range(length - 1, 0, -1):
        res = -(coeff_b[j] * coeff_b[j]) / (a_eff[:, j] + res)
    denominator = a_eff[:, 0] + res
    return -2.0 * (nrm2 * nrm2 / denominator).imag


def extract_q_component(data: dict, q_target: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float, float]:
    for q_idx, momentum in enumerate(data["Q"]):
        if momentum != q_target:
            continue
        return data["a"][q_idx], data["b"][q_idx], data["nrm2"][q_idx], data["E0"]
    raise ValueError(f"Q={q_target} not found in file.")


def discover_pairs(data_dir: Path) -> list[dict]:
    grouped: dict[tuple[float, float], dict] = {}
    for path in sorted(data_dir.glob("*.dat")):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        key = (float(match.group("U")), float(match.group("V")))
        entry = grouped.setdefault(
            key,
            {"U": key[0], "V": key[1], "Gh": None, "Gp": None},
        )
        entry[match.group("kind")] = path
    pairs = [entry for entry in grouped.values() if entry["Gh"] is not None and entry["Gp"] is not None]
    pairs.sort(key=lambda item: (item["U"], item["V"]))
    return pairs


def build_iteration_list(requested: list[int], max_available: int) -> list[int]:
    usable = [value for value in requested if value <= max_available]
    if requested and requested[-1] > max_available and max_available not in usable:
        usable.append(max_available)
    return usable


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    requested_iterations = list(range(args.iter_start, args.iter_stop + 1, args.iter_step))
    q_target = (0, 0)

    pairs = discover_pairs(data_dir)
    if not pairs:
        raise FileNotFoundError(f"No matching Gh/Gp file pairs found in {data_dir}")

    cmap = plt.get_cmap("viridis")

    for pair in pairs:
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.8))
        gh_data = parse_green_file(pair["Gh"])
        gp_data = parse_green_file(pair["Gp"])
        gh_a, gh_b, gh_nrm2, gh_e0 = extract_q_component(gh_data, q_target)
        gp_a, gp_b, gp_nrm2, gp_e0 = extract_q_component(gp_data, q_target)
        omega_min, omega_max = PANEL_XLIMS.get((pair["U"], pair["V"]), (-args.omega_max, args.omega_max))
        omega = np.linspace(omega_min, omega_max, args.omega_points)

        max_available = min(len(gh_a), len(gh_b), len(gp_a), len(gp_b))
        iteration_list = build_iteration_list(requested_iterations, max_available)
        colors = cmap(np.linspace(0.08, 0.92, len(iteration_list)))
        panel_min = np.inf
        panel_max = -np.inf

        for color, steps in zip(colors, iteration_list):
            gh_spec = spectral_component(omega, args.eta, gh_a, gh_b, steps, gh_nrm2, gh_e0, gh_data["U"], "Gh")
            gp_spec = spectral_component(omega, args.eta, gp_a, gp_b, steps, gp_nrm2, gp_e0, gp_data["U"], "Gp")
            total_spec = gh_spec + gp_spec
            panel_min = min(panel_min, float(np.min(total_spec)))
            panel_max = max(panel_max, float(np.max(total_spec)))
            ax.plot(omega, total_spec, lw=1.5, color=color, label=f"{steps}")

        ax.set_title(f"U={pair['U']:g}, V={pair['V']:g}")
        ax.grid(alpha=0.2, lw=0.5)
        ax.set_xlim(float(omega[0]), float(omega[-1]))
        if np.isfinite(panel_min) and np.isfinite(panel_max):
            if np.isclose(panel_min, panel_max):
                pad = max(1.0e-6, 0.05 * max(abs(panel_min), 1.0))
            else:
                pad = 0.08 * (panel_max - panel_min)
            ax.set_ylim(panel_min - pad, panel_max + pad)
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"$A(\omega)$")
        if max_available < args.iter_stop:
            ax.text(
                0.98,
                0.97,
                f"max={max_available}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
        ax.legend(title="Iterations", frameon=False)
        fig.tight_layout()

        if args.output:
            stem = args.output.resolve().with_suffix("")
            output = stem.parent / f"{stem.name}_U{pair['U']:g}_V{pair['V']:g}.png"
        else:
            output = data_dir / f"spectral_function_vs_iteration_U{pair['U']:g}_V{pair['V']:g}.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure to {output}")


if __name__ == "__main__":
    main()
