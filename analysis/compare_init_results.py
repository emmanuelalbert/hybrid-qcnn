"""
Compares "default" vs "small_angle" VQC weight init using REAL multi-seed training
results (not the static gradient probe, which couldn't detect the failure mode --
see compare_init_strategies.py's docstring/history for why).

Usage:
    python compare_init_results.py --model output_vqc --seeds 0 1 2 3 4 5 6 7 8 9
    python compare_init_results.py --model middle_vqc --seeds 0 1 2 3 4 5 6 7 8 9

Expects both sets of result files to already exist:
    results/{model}_seed{N}.json              (default init)
    results/{model}_small_angle_seed{N}.json  (small_angle init)
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load_runs(results_dir: Path, model_name: str, seeds: list, suffix: str = "") -> dict:
    runs = {}
    missing = []
    for seed in seeds:
        path = results_dir / f"{model_name}{suffix}_seed{seed}.json"
        if path.exists():
            with open(path) as f:
                runs[seed] = json.load(f)
        else:
            missing.append(str(path))
    if missing:
        print(f"WARNING: missing files for suffix='{suffix}':")
        for m in missing:
            print(f"  - {m}")
    return runs


def summarize_runs(label: str, runs: dict):
    if not runs:
        print(f"{label}: no data")
        return None

    accs = np.array([r["test_accuracy"] for r in runs.values()])
    n_failed = int((accs < 0.5).sum())
    failed_seeds = [s for s, r in runs.items() if r["test_accuracy"] < 0.5]
    healthy_accs = accs[accs >= 0.5]

    print(f"\n{label} (n={len(runs)} seeds):")
    per_seed = ", ".join(f"seed{s}={r['test_accuracy']:.4f}" for s, r in runs.items())
    print(f"  Per-seed: {per_seed}")
    print(f"  Mean +- std:  {accs.mean():.4f} +- {accs.std():.4f}   Median: {np.median(accs):.4f}")
    print(f"  Failure rate: {n_failed}/{len(runs)} ({100*n_failed/len(runs):.0f}%)  "
          f"failed seeds: {failed_seeds if failed_seeds else 'none'}")
    if 0 < len(healthy_accs) < len(accs):
        print(f"  Excluding failures: {healthy_accs.mean():.4f} +- {healthy_accs.std():.4f} (n={len(healthy_accs)})")

    return {"accs": accs, "n_failed": n_failed, "n_total": len(runs), "seeds": list(runs.keys())}


def compare(model_name: str, seeds: list, results_dir: Path):
    print(f"=== Init strategy comparison: {model_name} ===")

    default_runs = load_runs(results_dir, model_name, seeds, suffix="")
    small_angle_runs = load_runs(results_dir, model_name, seeds, suffix="_small_angle")

    default_summary = summarize_runs("DEFAULT init (Uniform[0, 2*pi))", default_runs)
    small_angle_summary = summarize_runs("SMALL_ANGLE init (Normal(0, std))", small_angle_runs)

    if default_summary is None or small_angle_summary is None:
        print("\nCan't compare -- one or both init strategies missing results.")
        return

    # Failure rate comparison (the primary question: does small_angle reduce failures?)
    d_rate = default_summary["n_failed"] / default_summary["n_total"]
    s_rate = small_angle_summary["n_failed"] / small_angle_summary["n_total"]
    print(f"\n--- Failure rate: default={100*d_rate:.0f}% vs small_angle={100*s_rate:.0f}% ---")
    if s_rate < d_rate:
        print(f"  small_angle reduced the failure rate ({default_summary['n_failed']} -> "
              f"{small_angle_summary['n_failed']} failed seeds out of {default_summary['n_total']}).")
    elif s_rate > d_rate:
        print(f"  small_angle INCREASED the failure rate ({default_summary['n_failed']} -> "
              f"{small_angle_summary['n_failed']} failed seeds out of {default_summary['n_total']}).")
    else:
        print(f"  No difference in failure rate ({default_summary['n_failed']} failed seeds each).")

    # Note on statistical power: with ~10 seeds, a failure-rate difference of 1-2 seeds
    # is within noise. A formal test (Fisher's exact, since these are small counts) is
    # included for reference, but treat the p-value as indicative, not conclusive, unless
    # you've run enough seeds for the counts to be reasonably large.
    contingency = [
        [default_summary["n_failed"], default_summary["n_total"] - default_summary["n_failed"]],
        [small_angle_summary["n_failed"], small_angle_summary["n_total"] - small_angle_summary["n_failed"]],
    ]
    odds_ratio, p_value = stats.fisher_exact(contingency)
    print(f"  Fisher's exact test on failure counts: p={p_value:.4f} "
          f"({'not conclusive with this sample size' if p_value >= 0.05 else 'significant'}) "
          f"-- treat cautiously with only ~10 seeds per arm")

    # Accuracy comparison restricted to common seeds (paired)
    common_seeds = sorted(set(default_summary["seeds"]) & set(small_angle_summary["seeds"]))
    if len(common_seeds) >= 2:
        default_runs_by_seed = {s: r["test_accuracy"] for s, r in default_runs.items()}
        small_angle_runs_by_seed = {s: r["test_accuracy"] for s, r in small_angle_runs.items()}
        d_vals = np.array([default_runs_by_seed[s] for s in common_seeds])
        s_vals = np.array([small_angle_runs_by_seed[s] for s in common_seeds])

        print(f"\n--- Paired accuracy comparison (n={len(common_seeds)} common seeds) ---")
        print(f"  Default:     {d_vals.mean():.4f} +- {d_vals.std():.4f}")
        print(f"  Small_angle: {s_vals.mean():.4f} +- {s_vals.std():.4f}")
        try:
            w_stat, w_p = stats.wilcoxon(s_vals, d_vals)
            print(f"  Wilcoxon (small_angle vs default): W={w_stat:.3f}, p={w_p:.4f}")
        except ValueError as e:
            print(f"  Wilcoxon unavailable: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["output_vqc", "middle_vqc"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--results_dir", type=str, default="../results")
    args = parser.parse_args()

    compare(args.model, args.seeds, Path(args.results_dir))


if __name__ == "__main__":
    main()