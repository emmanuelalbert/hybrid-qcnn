"""
Diagnose why a specific seed failed: compares its per-epoch train_loss, val_accuracy, and
grad_norm against a healthy seed for the same model, to distinguish between:
  (a) stuck/near-zero gradients from the start (barren-plateau-like initialization)
  (b) diverging/oscillating loss (LR or instability issue)
  (c) genuinely still improving but cut off too early by patience

Usage:
    python diagnose_failed_seed.py --model output_vqc --failed_seed 1 --healthy_seed 0
    python diagnose_failed_seed.py --model middle_vqc --failed_seed 3 --healthy_seed 0
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load(results_dir: Path, model_name: str, seed: int) -> dict:
    path = results_dir / f"{model_name}_seed{seed}.json"
    with open(path) as f:
        return json.load(f)


def diagnose(model_name: str, failed_seed: int, healthy_seed: int, results_dir: Path):
    failed = load(results_dir, model_name, failed_seed)
    healthy = load(results_dir, model_name, healthy_seed)

    print(f"=== {model_name}: seed {failed_seed} (failed, acc={failed['test_accuracy']:.4f}) "
          f"vs seed {healthy_seed} (healthy, acc={healthy['test_accuracy']:.4f}) ===\n")

    print(f"{'Epoch':<7}{'FAILED train_loss':<20}{'FAILED val_acc':<17}{'FAILED grad_norm':<18}"
          f"{'HEALTHY train_loss':<20}{'HEALTHY val_acc':<17}{'HEALTHY grad_norm':<18}")
    max_len = max(len(failed["history"]), len(healthy["history"]))
    for i in range(max_len):
        f_row = failed["history"][i] if i < len(failed["history"]) else None
        h_row = healthy["history"][i] if i < len(healthy["history"]) else None
        f_str = (f"{f_row['train_loss']:<20.4f}{f_row['val_accuracy']:<17.4f}{f_row['grad_norm_mean']:<18.4f}"
                  if f_row else f"{'--':<20}{'--':<17}{'--':<18}")
        h_str = (f"{h_row['train_loss']:<20.4f}{h_row['val_accuracy']:<17.4f}{h_row['grad_norm_mean']:<18.4f}"
                  if h_row else f"{'--':<20}{'--':<17}{'--':<18}")
        print(f"{i+1:<7}{f_str}{h_str}")

    # Simple automatic read
    failed_grad_norms = [h["grad_norm_mean"] for h in failed["history"]]
    healthy_grad_norms = [h["grad_norm_mean"] for h in healthy["history"]]
    failed_first_grad = failed_grad_norms[0]
    failed_last_grad = failed_grad_norms[-1]
    failed_loss_trend = failed["history"][-1]["train_loss"] - failed["history"][0]["train_loss"]

    print("\n--- Diagnosis ---")
    if failed_first_grad < 0.05:
        print(f"Gradient norm started near-zero ({failed_first_grad:.4f}) -> consistent with a "
              f"barren-plateau-like / unlucky initialization: the circuit started in a flat "
              f"region and never got a useful gradient signal.")
    elif failed_loss_trend > -0.1:
        print(f"Train loss barely moved over the whole run ({failed['history'][0]['train_loss']:.4f} "
              f"-> {failed['history'][-1]['train_loss']:.4f}) despite non-trivial gradients "
              f"(mean grad norm ~{sum(failed_grad_norms)/len(failed_grad_norms):.4f}) -> "
              f"suggests the optimizer is stuck (e.g. oscillating around a poor local point) "
              f"rather than lacking gradient signal entirely.")
    else:
        print(f"Loss was decreasing (delta={failed_loss_trend:.4f}) when training stopped at "
              f"epoch {failed['epochs_trained']} -> possible that patience={failed.get('patience')} "
              f"cut this run off before it could recover. Consider rerunning this seed alone with "
              f"higher patience/epochs to check.")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs_f = [h["epoch"] for h in failed["history"]]
    epochs_h = [h["epoch"] for h in healthy["history"]]

    axes[0].plot(epochs_f, [h["train_loss"] for h in failed["history"]], "-o", color="tab:red", label=f"seed{failed_seed} (failed)")
    axes[0].plot(epochs_h, [h["train_loss"] for h in healthy["history"]], "-o", color="tab:green", label=f"seed{healthy_seed} (healthy)")
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_f, [h["val_accuracy"] for h in failed["history"]], "-o", color="tab:red", label=f"seed{failed_seed} (failed)")
    axes[1].plot(epochs_h, [h["val_accuracy"] for h in healthy["history"]], "-o", color="tab:green", label=f"seed{healthy_seed} (healthy)")
    axes[1].set_title("Val accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs_f, [h["grad_norm_mean"] for h in failed["history"]], "-o", color="tab:red", label=f"seed{failed_seed} (failed)")
    axes[2].plot(epochs_h, [h["grad_norm_mean"] for h in healthy["history"]], "-o", color="tab:green", label=f"seed{healthy_seed} (healthy)")
    axes[2].set_title("Mean gradient norm")
    axes[2].set_xlabel("Epoch")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"{model_name}: failed seed {failed_seed} vs healthy seed {healthy_seed}")
    fig.tight_layout()
    out_path = results_dir / f"diagnose_{model_name}_seed{failed_seed}_vs_seed{healthy_seed}.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved diagnostic plot: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--failed_seed", type=int, required=True)
    parser.add_argument("--healthy_seed", type=int, required=True)
    parser.add_argument("--results_dir", type=str, default="../results")
    args = parser.parse_args()

    diagnose(args.model, args.failed_seed, args.healthy_seed, Path(args.results_dir))


if __name__ == "__main__":
    main()