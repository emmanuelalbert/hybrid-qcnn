"""
Generates manuscript-ready figures summarizing the full study:

  Fig 1a: Per-seed test accuracy distribution, VQC vs matched classical baseline,
          for both placements (strip + box plot, failed seeds highlighted).
  Fig 1b: Barren-plateau-style failure rate, by placement x init strategy
          (default vs small_angle), as a grouped bar chart.
  Fig 2:  Representative training curves -- one healthy seed vs one failed seed --
          showing the loss/gradient-norm signature of the failure mode.

Usage:
    python make_manuscript_figures.py --seeds 0 1 2 3 4 5 6 7 8 9 --results_dir ../results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

MODEL_LABELS = {
    "output_vqc": "VQC (output)",
    "classical": "Classical\n(output-matched)",
    "middle_vqc": "VQC (middle)",
    "classical_middle": "Classical\n(middle-matched)",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})


def load_runs(results_dir: Path, model_name: str, seeds: list, suffix: str = "") -> dict:
    runs = {}
    for seed in seeds:
        path = results_dir / f"{model_name}{suffix}_seed{seed}.json"
        if path.exists():
            with open(path) as f:
                runs[seed] = json.load(f)
    return runs


def fig1a_accuracy_distribution(results_dir: Path, seeds: list, out_path: Path):
    """Strip + box plot of per-seed test accuracy, one panel per placement."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    placements = [
        ("output_vqc", "classical", "Output placement", axes[0]),
        ("middle_vqc", "classical_middle", "Middle placement", axes[1]),
    ]

    for vqc_name, classical_name, title, ax in placements:
        vqc_runs = load_runs(results_dir, vqc_name, seeds)
        classical_runs = load_runs(results_dir, classical_name, seeds)

        vqc_accs = np.array([r["test_accuracy"] for r in vqc_runs.values()])
        classical_accs = np.array([r["test_accuracy"] for r in classical_runs.values()])
        vqc_failed = vqc_accs < 0.5

        positions = [1, 2]
        box_data = [vqc_accs, classical_accs]
        bp = ax.boxplot(box_data, positions=positions, widths=0.45, showfliers=False,
                         patch_artist=True, medianprops={"color": "black", "linewidth": 1.5})
        bp["boxes"][0].set_facecolor("#a8c6f0")
        bp["boxes"][1].set_facecolor("#f5c396")

        # jittered strip points, healthy vs failed colored differently
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.08, 0.08, size=len(vqc_accs))
        healthy_mask = ~vqc_failed
        ax.scatter(np.full(healthy_mask.sum(), 1) + jitter[healthy_mask], vqc_accs[healthy_mask],
                   color="tab:blue", edgecolors="black", zorder=3, s=45, label="VQC (healthy)")
        if vqc_failed.any():
            ax.scatter(np.full(vqc_failed.sum(), 1) + jitter[vqc_failed], vqc_accs[vqc_failed],
                       color="tab:red", edgecolors="black", zorder=3, s=60, marker="X",
                       label="VQC (failed)")

        jitter2 = rng.uniform(-0.08, 0.08, size=len(classical_accs))
        ax.scatter(np.full(len(classical_accs), 2) + jitter2, classical_accs,
                   color="tab:orange", edgecolors="black", zorder=3, s=45, label="Classical")

        n_failed = int(vqc_failed.sum())
        ax.set_xticks(positions)
        ax.set_xticklabels(["VQC", "Classical\n(param-matched)"])
        ax.set_title(f"{title}\n(VQC failure rate: {n_failed}/{len(vqc_accs)})")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Test accuracy")
        ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("Fig. 1a — Per-seed test accuracy: VQC vs. parameter-matched classical baseline\n"
                  "(n=10 seeds per model; params matched exactly per placement)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


def fig1b_failure_rates(results_dir: Path, seeds: list, out_path: Path):
    """Grouped bar chart: failure rate by placement x init strategy."""
    configs = [
        ("output_vqc", "", "Output,\ndefault init"),
        ("output_vqc", "_small_angle", "Output,\nsmall_angle init"),
        ("middle_vqc", "", "Middle,\ndefault init"),
        ("middle_vqc", "_small_angle", "Middle,\nsmall_angle init"),
    ]

    labels, rates, counts = [], [], []
    for model_name, suffix, label in configs:
        runs = load_runs(results_dir, model_name, seeds, suffix=suffix)
        if not runs:
            continue
        accs = np.array([r["test_accuracy"] for r in runs.values()])
        n_failed = int((accs < 0.5).sum())
        labels.append(label)
        rates.append(100 * n_failed / len(accs))
        counts.append((n_failed, len(accs)))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["tab:blue", "tab:cyan", "tab:orange", "#ffb86b"]
    bars = ax.bar(range(len(labels)), rates, color=colors[:len(labels)], edgecolor="black")
    for bar, (n_failed, n_total) in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{n_failed}/{n_total}", ha="center", fontsize=10)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Barren-plateau-style failure rate (%)")
    ax.set_ylim(0, max(rates) + 12 if rates else 10)
    ax.set_title("Fig. 1b — Failure rate by VQC placement and weight initialization strategy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


def fig2_training_signature(results_dir: Path, out_path: Path,
                             model_name: str = "output_vqc", failed_seed: int = 1, healthy_seed: int = 0):
    """Overlay a failed vs healthy training curve to show the barren-plateau signature."""
    failed = load_runs(results_dir, model_name, [failed_seed]).get(failed_seed)
    healthy = load_runs(results_dir, model_name, [healthy_seed]).get(healthy_seed)
    if not failed or not healthy or "history" not in failed or not failed["history"]:
        print(f"Skipping Fig 2: missing history for {model_name} seed{failed_seed}/seed{healthy_seed}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ep_f = [h["epoch"] for h in failed["history"]]
    ep_h = [h["epoch"] for h in healthy["history"]]

    axes[0].plot(ep_f, [h["train_loss"] for h in failed["history"]], "-o", color="tab:red",
                 label=f"Failed (seed {failed_seed})", markersize=4)
    axes[0].plot(ep_h, [h["train_loss"] for h in healthy["history"]], "-o", color="tab:green",
                 label=f"Healthy (seed {healthy_seed})", markersize=4)
    axes[0].axhline(np.log(10), color="gray", linestyle="--", linewidth=1, label="ln(10) [random guess]")
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep_f, [h["grad_norm_mean"] for h in failed["history"]], "-o", color="tab:red",
                 label=f"Failed (seed {failed_seed})", markersize=4)
    axes[1].plot(ep_h, [h["grad_norm_mean"] for h in healthy["history"]], "-o", color="tab:green",
                 label=f"Healthy (seed {healthy_seed})", markersize=4)
    axes[1].set_title("Mean gradient norm")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Fig. 2 — Barren-plateau failure signature ({MODEL_LABELS.get(model_name, model_name)})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--results_dir", type=str, default="../results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    fig1a_accuracy_distribution(results_dir, args.seeds, results_dir / "fig1a_accuracy_distribution.png")
    fig1b_failure_rates(results_dir, args.seeds, results_dir / "fig1b_failure_rates.png")
    fig2_training_signature(results_dir, results_dir / "fig2_barren_plateau_signature.png",
                             model_name="output_vqc", failed_seed=1, healthy_seed=0)


if __name__ == "__main__":
    main()