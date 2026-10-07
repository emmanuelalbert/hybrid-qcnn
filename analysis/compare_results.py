"""
Phase 0 comparison: load results/{model}_seed{seed}.json for all 4 models and
produce a side-by-side table plus two plots (accuracy vs. params, training time).

Usage:
    python compare_results.py --seed 0 --results_dir ../results
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

MODEL_LABELS = {
    "output_vqc": "CNN + VQC (output)",
    "middle_vqc": "CNN + VQC (middle)",
    "classical": "Classical baseline (output-matched)",
    "classical_middle": "Classical baseline (middle-matched)",
}

MODEL_ORDER = ["output_vqc", "classical", "middle_vqc", "classical_middle"]


def load_results(results_dir: Path, seed: int) -> dict:
    results = {}
    missing = []
    for model_name in MODEL_ORDER:
        path = results_dir / f"{model_name}_seed{seed}.json"
        if not path.exists():
            missing.append(str(path))
            continue
        with open(path) as f:
            results[model_name] = json.load(f)
    if missing:
        print("WARNING: missing result files:")
        for m in missing:
            print(f"  - {m}")
    return results


def print_table(results: dict):
    header = f"{'Model':<38}{'Params':>10}{'Test Acc':>10}{'Test F1':>10}{'Train Time (s)':>16}"
    print(header)
    print("-" * len(header))
    for model_name in MODEL_ORDER:
        if model_name not in results:
            continue
        r = results[model_name]
        label = MODEL_LABELS[model_name]
        print(f"{label:<38}{r['n_params']:>10}{r['test_accuracy']:>10.4f}"
              f"{r['test_f1']:>10.4f}{r['total_train_time_sec']:>16.1f}")

    # Direct paired comparisons -- the sharpest signal in the study
    print()
    if "output_vqc" in results and "classical" in results:
        d_acc = results["output_vqc"]["test_accuracy"] - results["classical"]["test_accuracy"]
        print(f"Output placement: VQC vs. matched classical -> accuracy delta = {d_acc:+.4f} "
              f"({'VQC ahead' if d_acc > 0 else 'classical ahead' if d_acc < 0 else 'tie'})")
    if "middle_vqc" in results and "classical_middle" in results:
        d_acc = results["middle_vqc"]["test_accuracy"] - results["classical_middle"]["test_accuracy"]
        print(f"Middle placement: VQC vs. matched classical -> accuracy delta = {d_acc:+.4f} "
              f"({'VQC ahead' if d_acc > 0 else 'classical ahead' if d_acc < 0 else 'tie'})")


def plot_accuracy_vs_params(results: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name in MODEL_ORDER:
        if model_name not in results:
            continue
        r = results[model_name]
        # marker distinguishes VQC vs classical; color distinguishes placement (output vs middle)
        marker = "o" if "vqc" in model_name else "s"
        color = "tab:blue" if "middle" not in model_name else "tab:orange"
        ax.scatter(r["n_params"], r["test_accuracy"], s=120, marker=marker, color=color,
                   edgecolors="black", zorder=3)
        ax.annotate(MODEL_LABELS[model_name], (r["n_params"], r["test_accuracy"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs. parameter count (Phase 0, single seed)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


def plot_training_time(results: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    names = [MODEL_LABELS[m] for m in MODEL_ORDER if m in results]
    times = [results[m]["total_train_time_sec"] for m in MODEL_ORDER if m in results]
    # color distinguishes VQC vs classical here, since that's the comparison this plot is about
    colors = ["tab:blue" if "vqc" in m else "tab:orange" for m in MODEL_ORDER if m in results]
    ax.bar(range(len(names)), times, color=colors, edgecolor="black")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Total training time (s)")
    ax.set_title("Training time by model (Phase 0, single seed)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="../results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir, args.seed)

    if not results:
        print("No results found. Run train.py for each model first.")
        return

    print_table(results)
    plot_accuracy_vs_params(results, results_dir / f"accuracy_vs_params_seed{args.seed}.png")
    plot_training_time(results, results_dir / f"training_time_seed{args.seed}.png")


if __name__ == "__main__":
    main()