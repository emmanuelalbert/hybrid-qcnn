"""
Single-run training loop: one model, one seed, on MNIST.

Usage:
    python train.py --model output_vqc --epochs 30 --patience 5
    python train.py --model output_vqc --epochs 30 --patience 5 --init_method small_angle

Early stopping: --epochs is a MAX epoch budget. Training stops if val_loss hasn't
improved for --patience consecutive epochs; best-val-loss weights are restored before
final test evaluation.

--init_method / --init_std only affect VQC models (output_vqc, middle_vqc); they're
ignored (with a warning) for the classical baselines, which have no such parameter.
"""

import argparse
import copy
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

sys.path.append(str(Path(__file__).resolve().parent.parent / "models"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "data"))

from datasets import get_mnist_loaders
from model_output_vqc import CNNVQCOutput
from model_middle_vqc import CNNVQCMiddle
from model_classical import CNNClassicalBaseline
from model_classical_middle import CNNClassicalBaselineMiddle


MODEL_REGISTRY = {
    "output_vqc": CNNVQCOutput,
    "middle_vqc": CNNVQCMiddle,
    "classical": CNNClassicalBaseline,
    "classical_middle": CNNClassicalBaselineMiddle,
}
VQC_MODELS = {"output_vqc", "middle_vqc"}  # models that accept init_method/init_std


def set_seed(seed: int):
    torch.manual_seed(seed)


def compute_grad_stats(model: nn.Module) -> dict:
    norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    if not norms:
        return {"grad_norm_mean": 0.0, "grad_norm_std": 0.0}
    norms_t = torch.tensor(norms)
    return {
        "grad_norm_mean": norms_t.mean().item(),
        "grad_norm_std": norms_t.std().item() if len(norms) > 1 else 0.0,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return {"loss": avg_loss, "accuracy": acc, "f1": f1}


def train_one_run(model_name: str, seed: int = 0, epochs: int = 30, lr: float = 1e-2,
                   train_size: int = 5000, val_size: int = 1000, test_size: int = 1000,
                   batch_size: int = 64, device: str = "cpu", patience: int = 5,
                   min_delta: float = 1e-4, init_method: str = "default",
                   init_std: float = 0.1) -> dict:
    set_seed(seed)
    device = torch.device(device)

    train_loader, val_loader, test_loader = get_mnist_loaders(
        train_size=train_size, val_size=val_size, test_size=test_size,
        batch_size=batch_size, seed=seed,
    )

    model_cls = MODEL_REGISTRY[model_name]
    if model_name in VQC_MODELS:
        model = model_cls(init_method=init_method, init_std=init_std).to(device)
    else:
        if init_method != "default":
            print(f"NOTE: --init_method={init_method} ignored for '{model_name}' "
                  f"(classical models have no such parameter)")
        model = model_cls().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_params = model.n_params()
    print(f"[{model_name}] seed={seed}, trainable params={n_params}, "
          f"max_epochs={epochs}, patience={patience}, init_method={init_method}")

    history = []
    total_start = time.time()

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_since_improvement = 0
    stopped_early = False

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        epoch_grad_stats = []
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            epoch_grad_stats.append(compute_grad_stats(model))
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        epoch_time = time.time() - epoch_start
        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, device)

        grad_means = [g["grad_norm_mean"] for g in epoch_grad_stats]
        grad_stds = [g["grad_norm_std"] for g in epoch_grad_stats]
        agg_grad_mean = sum(grad_means) / len(grad_means)
        agg_grad_std = sum(grad_stds) / len(grad_stds)

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "epoch_time_sec": epoch_time,
            "grad_norm_mean": agg_grad_mean,
            "grad_norm_std": agg_grad_std,
        }
        history.append(epoch_log)

        if val_metrics["loss"] < best_val_loss - min_delta:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
            improvement_marker = " *best*"
        else:
            epochs_since_improvement += 1
            improvement_marker = ""

        print(f"  epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
              f"grad_norm={agg_grad_mean:.4f}+-{agg_grad_std:.4f} time={epoch_time:.1f}s"
              f"{improvement_marker}")

        if epochs_since_improvement >= patience:
            print(f"  Early stopping at epoch {epoch+1} "
                  f"(no val_loss improvement for {patience} epochs; best was epoch {best_epoch})")
            stopped_early = True
            break

    model.load_state_dict(best_state)

    total_time = time.time() - total_start
    test_metrics = evaluate(model, test_loader, device)
    print(f"[{model_name}] TEST (best epoch={best_epoch}): "
          f"acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f}")

    result = {
        "model_name": model_name,
        "seed": seed,
        "n_params": n_params,
        "max_epochs": epochs,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "patience": patience,
        "init_method": init_method,
        "total_train_time_sec": total_time,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "history": history,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30, help="max epoch budget")
    parser.add_argument("--patience", type=int, default=5,
                         help="stop if val_loss doesn't improve for this many epochs")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="../results")
    parser.add_argument("--init_method", type=str, default="default",
                         choices=["default", "small_angle"],
                         help="VQC weight init strategy (ignored for classical models)")
    parser.add_argument("--init_std", type=float, default=0.1,
                         help="std for small_angle init")
    args = parser.parse_args()

    result = train_one_run(
        model_name=args.model, seed=args.seed, epochs=args.epochs,
        lr=args.lr, device=args.device, patience=args.patience,
        init_method=args.init_method, init_std=args.init_std,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # init_method is baked into the filename when non-default, so small_angle runs
    # don't collide with / overwrite your existing default-init results.
    suffix = "" if args.init_method == "default" else f"_{args.init_method}"
    out_path = out_dir / f"{args.model}{suffix}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()