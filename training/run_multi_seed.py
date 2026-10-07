"""
Multi-seed runner: repeats train_one_run across several seeds, for one model or all 4,
optionally sweeping the VQC init_method.

Usage:
    python run_multi_seed.py --model output_vqc --seeds 0 1 2 3 4
    python run_multi_seed.py --all --seeds 0 1 2 3 4
    python run_multi_seed.py --model output_vqc --seeds 0 1 2 3 4 5 6 7 8 9 --init_method small_angle
    python run_multi_seed.py --model middle_vqc --seeds 0 1 2 3 4 5 6 7 8 9 --init_method small_angle

Result files are saved as results/{model}_seed{seed}.json for default init, or
results/{model}_small_angle_seed{seed}.json for small_angle init, so the two init
strategies never overwrite each other and can be aggregated/compared separately.

Skips a run if its result file already exists, unless --overwrite is passed.
"""

import argparse
import sys
import json
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from train import train_one_run, MODEL_REGISTRY, VQC_MODELS


def run_multi_seed(model_names, seeds, epochs, patience, lr, device, out_dir,
                    init_method="default", init_std=0.1, overwrite=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(model_names) * len(seeds)
    run_idx = 0
    overall_start = time.time()

    for model_name in model_names:
        effective_init = init_method if model_name in VQC_MODELS else "default"
        if init_method != "default" and model_name not in VQC_MODELS:
            print(f"NOTE: --init_method={init_method} ignored for '{model_name}' "
                  f"(classical models have no such parameter)")

        suffix = "" if effective_init == "default" else f"_{effective_init}"

        for seed in seeds:
            run_idx += 1
            out_path = out_dir / f"{model_name}{suffix}_seed{seed}.json"

            if out_path.exists() and not overwrite:
                print(f"[{run_idx}/{total_runs}] SKIP {model_name} seed={seed} init={effective_init} "
                      f"(already exists at {out_path}, use --overwrite to redo)")
                continue

            print(f"\n[{run_idx}/{total_runs}] RUNNING {model_name} seed={seed} init={effective_init} ...")
            result = train_one_run(
                model_name=model_name, seed=seed, epochs=epochs,
                patience=patience, lr=lr, device=device,
                init_method=effective_init, init_std=init_std,
            )
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[{run_idx}/{total_runs}] Saved {out_path}")

    total_time = time.time() - overall_start
    print(f"\nAll runs complete. Total wall-clock time: {total_time/60:.1f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()),
                         help="single model to run across seeds; omit if using --all")
    parser.add_argument("--all", action="store_true", help="run all 4 registered models")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n_seeds", type=int, default=5,
                         help="if --seeds not given, use seeds 0..n_seeds-1")
    parser.add_argument("--epochs", type=int, default=30, help="max epoch budget per run")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="../results")
    parser.add_argument("--init_method", type=str, default="default",
                         choices=["default", "small_angle"],
                         help="VQC weight init strategy (ignored for classical models)")
    parser.add_argument("--init_std", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("must pass either --model <name> or --all")

    model_names = list(MODEL_REGISTRY.keys()) if args.all else [args.model]
    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))

    print(f"Models: {model_names}")
    print(f"Seeds: {seeds}")
    print(f"Init method: {args.init_method}")
    print(f"Total runs: {len(model_names) * len(seeds)}")

    run_multi_seed(
        model_names=model_names, seeds=seeds, epochs=args.epochs,
        patience=args.patience, lr=args.lr, device=args.device,
        out_dir=args.out_dir, init_method=args.init_method,
        init_std=args.init_std, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()