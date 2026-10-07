"""
Plot per-epoch training curves (train loss, val accuracy, val F1, grad norm) for all 4 models,
to check whether models were still improving at the final epoch or had plateaued/converged.

Usage:
    python plot_training_curves.py --seed 0 --results_dir ../results
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
MODEL_COLORS = {
    "output_vqc": "tab:blue",
    "classical": "tab:blue",
    "middle_vqc": "tab:orange",
    "classical_middle": "tab:orange",
}
MODEL_STYLES = {
    "output_vqc": "-o",
    "classical": "--s",
    "middle_vqc": "-o",
    "classical_middle": "--s",
}


def load_results(results_dir: Path, seed: int) -> dict:
    results = {}
    for model_name in MODEL_ORDER:
        path = results_dir / f"{model_name}_seed{seed}.json"
        if path.exists():
            with open(path) as f:
                results[model_name] = json.load(f)
    return results


def plot_curves(results: dict, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_loss, ax_acc, ax_f1, ax_grad = axes.flatten()

    for model_name in MODEL_ORDER:
        if model_name not in results:
            continue
        history = results[model_name]["history"]
        epochs = [h["epoch"] for h in history]
        color = MODEL_COLORS[model_name]
        style = MODEL_STYLES[model_name]
        label = MODEL_LABELS[model_name]

        ax_loss.plot(epochs, [h["train_loss"] for h in history], style, color=color, label=label, markersize=4)
        ax_acc.plot(epochs, [h["val_accuracy"] for h in history], style, color=color, label=label, markersize=4)
        ax_f1.plot(epochs, [h["val_f1"] for h in history], style, color=color, label=label, markersize=4)
        ax_grad.plot(epochs, [h["grad_norm_mean"] for h in history], style, color=color, label=label, markersize=4)

    ax_loss.set_title("Train loss per epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(fontsize=8)

    ax_acc.set_title("Validation accuracy per epoch")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Val accuracy")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(fontsize=8)

    ax_f1.set_title("Validation F1 per epoch")
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("Val F1")
    ax_f1.grid(alpha=0.3)
    ax_f1.legend(fontsize=8)

    ax_grad.set_title("Mean gradient norm per epoch")
    ax_grad.set_xlabel("Epoch")
    ax_grad.set_ylabel("Grad norm (mean across batches)")
    ax_grad.grid(alpha=0.3)
    ax_grad.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    # Quick convergence check printed to console
    print("\nConvergence check (val accuracy, first epoch -> last epoch):")
    for model_name in MODEL_ORDER:
        if model_name not in results:
            continue
        history = results[model_name]["history"]
        first_acc = history[0]["val_accuracy"]
        last_acc = history[-1]["val_accuracy"]
        second_last_acc = history[-2]["val_accuracy"] if len(history) > 1 else first_acc
        still_improving = last_acc > second_last_acc
        print(f"  {MODEL_LABELS[model_name]:<38} {first_acc:.4f} -> {last_acc:.4f}  "
              f"({'still improving at final epoch' if still_improving else 'flattened/plateaued'})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="../results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir, args.seed)

    if not results:
        print("No results found.")
        return

    plot_curves(results, results_dir / f"training_curves_seed{args.seed}.png")


if __name__ == "__main__":
    main()