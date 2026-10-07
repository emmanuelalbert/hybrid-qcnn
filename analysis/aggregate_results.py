"""
Aggregate multi-seed results: mean +/- std per model, plus paired significance tests
between each VQC placement and its parameter-matched classical twin.

Usage:
    python aggregate_results.py --seeds 0 1 2 3 4 --results_dir ../results
    python aggregate_results.py --n_seeds 5 --results_dir ../results

A paired t-test is used (not an unpaired/independent t-test) because each seed produces
one VQC result and one classical result trained on the IDENTICAL data split (same seed ->
same train/val/test partition). Pairing controls for per-seed difficulty/luck in the data
split itself, which is a more sensitive test than comparing the two groups' distributions
independently.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

MODEL_LABELS = {
    "output_vqc": "CNN + VQC (output)",
    "middle_vqc": "CNN + VQC (middle)",
    "classical": "Classical baseline (output-matched)",
    "classical_middle": "Classical baseline (middle-matched)",
}
MODEL_ORDER = ["output_vqc", "classical", "middle_vqc", "classical_middle"]

# (vqc_model, matched_classical_model, human label for the placement)
PAIRS = [
    ("output_vqc", "classical", "Output placement"),
    ("middle_vqc", "classical_middle", "Middle placement"),
]


def load_all_results(results_dir: Path, seeds: list) -> dict:
    """Returns {model_name: {seed: result_dict}}, skipping any missing files."""
    results = {m: {} for m in MODEL_ORDER}
    missing = []
    for model_name in MODEL_ORDER:
        for seed in seeds:
            path = results_dir / f"{model_name}_seed{seed}.json"
            if path.exists():
                with open(path) as f:
                    results[model_name][seed] = json.load(f)
            else:
                missing.append(str(path))
    if missing:
        print("WARNING: missing result files (these seeds/models will be excluded):")
        for m in missing:
            print(f"  - {m}")
    return results


def summarize(results: dict, seeds: list):
    """Print mean+-std AND median accuracy/F1/params/time per model.

    Median is reported alongside mean because a handful of barren-plateau-style failed
    seeds (near-random accuracy) can dominate the mean and inflate std to the point of
    being uninformative. Median better reflects "typical" performance when training
    succeeds, while the failure rate itself (see below) is reported separately.
    """
    print(f"{'Model':<38}{'N':>4}{'Acc (mean+-std)':>22}{'Acc (median)':>14}{'F1 (mean+-std)':>22}{'Time (s, mean)':>16}")
    print("-" * 116)

    summary = {}
    for model_name in MODEL_ORDER:
        runs = results[model_name]
        if not runs:
            continue
        accs = np.array([r["test_accuracy"] for r in runs.values()])
        f1s = np.array([r["test_f1"] for r in runs.values()])
        times = np.array([r["total_train_time_sec"] for r in runs.values()])

        summary[model_name] = {"accs": accs, "f1s": f1s, "times": times, "seeds": list(runs.keys())}

        label = MODEL_LABELS[model_name]
        print(f"{label:<38}{len(runs):>4}"
              f"{f'{accs.mean():.4f} +- {accs.std():.4f}':>22}"
              f"{f'{np.median(accs):.4f}':>14}"
              f"{f'{f1s.mean():.4f} +- {f1s.std():.4f}':>22}"
              f"{f'{times.mean():.1f}':>16}")

        # Flag high-variance models, show per-seed breakdown, and report failure rate --
        # a large std can hide catastrophic failures on individual seeds that a
        # mean +/- std summary alone would mask.
        if accs.std() > 0.05:
            per_seed = ", ".join(f"seed{s}={r['test_accuracy']:.4f}" for s, r in runs.items())
            print(f"    ^ HIGH VARIANCE (std={accs.std():.4f}) -- per-seed: {per_seed}")
            n_failed = 0
            for s, r in runs.items():
                if r["test_accuracy"] < 0.5:
                    n_failed += 1
                    print(f"    ^ seed{s} looks like a FAILED run (acc={r['test_accuracy']:.4f}, "
                          f"stopped_early={r.get('stopped_early')}, "
                          f"epochs_trained={r.get('epochs_trained')}, "
                          f"best_epoch={r.get('best_epoch')})")
            print(f"    ^ Failure rate: {n_failed}/{len(runs)} ({100*n_failed/len(runs):.0f}%) seeds "
                  f"< 0.5 accuracy -- see 'excluding failures' stats below")

            healthy_accs = accs[accs >= 0.5]
            if 0 < len(healthy_accs) < len(accs):
                print(f"    ^ Excluding failed seeds: acc = {healthy_accs.mean():.4f} +- "
                      f"{healthy_accs.std():.4f} (n={len(healthy_accs)})")

    return summary


def paired_significance_tests(summary: dict):
    print("\nPaired comparisons (VQC vs. matched classical baseline, same seeds):")
    for vqc_name, classical_name, label in PAIRS:
        if vqc_name not in summary or classical_name not in summary:
            print(f"  {label}: SKIPPED (missing data for one or both models)")
            continue

        vqc_seeds = set(summary[vqc_name]["seeds"])
        classical_seeds = set(summary[classical_name]["seeds"])
        common_seeds = sorted(vqc_seeds & classical_seeds)

        if len(common_seeds) < 2:
            print(f"  {label}: SKIPPED (need >=2 common seeds for a paired test, "
                  f"have {len(common_seeds)})")
            continue

        # re-pull accuracy arrays restricted to the common seeds, in matching order
        vqc_runs = {s: a for s, a in zip(summary[vqc_name]["seeds"], summary[vqc_name]["accs"])}
        classical_runs = {s: a for s, a in zip(summary[classical_name]["seeds"], summary[classical_name]["accs"])}
        vqc_vals = np.array([vqc_runs[s] for s in common_seeds])
        classical_vals = np.array([classical_runs[s] for s in common_seeds])

        deltas = vqc_vals - classical_vals
        mean_delta = deltas.mean()
        std_delta = deltas.std(ddof=1) if len(deltas) > 1 else 0.0

        t_stat, p_value = stats.ttest_rel(vqc_vals, classical_vals)

        # Wilcoxon signed-rank test as a robustness check: unlike the paired t-test, it
        # doesn't assume the deltas are roughly normally distributed, so it's less thrown
        # off by an outlier failure (e.g. one seed at -0.85 delta from a barren-plateau run).
        try:
            w_stat, w_p_value = stats.wilcoxon(vqc_vals, classical_vals)
            wilcoxon_str = f"Wilcoxon: W={w_stat:.3f}, p={w_p_value:.4f}"
        except ValueError as e:
            # Wilcoxon fails if all deltas are zero, or with very small n -- not fatal
            wilcoxon_str = f"Wilcoxon: unavailable ({e})"

        sig_marker = "SIGNIFICANT (p<0.05)" if p_value < 0.05 else "not significant (p>=0.05)"
        direction = "VQC ahead" if mean_delta > 0 else "classical ahead" if mean_delta < 0 else "tied"

        print(f"\n  {label} (n={len(common_seeds)} seeds: {common_seeds}):")
        print(f"    VQC accuracy:       {vqc_vals.mean():.4f} +- {vqc_vals.std():.4f}  (median {np.median(vqc_vals):.4f})")
        print(f"    Classical accuracy: {classical_vals.mean():.4f} +- {classical_vals.std():.4f}  (median {np.median(classical_vals):.4f})")
        print(f"    Mean delta (VQC - classical): {mean_delta:+.4f} +- {std_delta:.4f}  ({direction})")
        print(f"    Paired t-test:  t={t_stat:.3f}, p={p_value:.4f}  -> {sig_marker}")
        print(f"    {wilcoxon_str}  (robust to outlier failures -- prefer this if any seed failed)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n_seeds", type=int, default=5,
                         help="if --seeds not given, use seeds 0..n_seeds-1")
    parser.add_argument("--results_dir", type=str, default="../results")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))
    results_dir = Path(args.results_dir)

    results = load_all_results(results_dir, seeds)
    summary = summarize(results, seeds)

    if not summary:
        print("No results found for the given seeds.")
        return

    paired_significance_tests(summary)


if __name__ == "__main__":
    main()