import csv

import numpy as np
from scipy import stats

from experiments import RESULTS_DIR, NAMED_STRATEGIES, result_path


STATS_DIR = RESULTS_DIR / "stats"
class Sample:
    def __init__(self, measure, p_noise, memory):
        self.measure = measure # COOPERATION, ENTROPY, or a strategy: 'TFT', 'PAVLOV', 'ALLD', 'OTHER' (only with memory 1)
        self.p_noise = p_noise
        self.memory = memory

    def label(self):
        return f"{self.measure} (p_noise={self.p_noise}, memory={self.memory})"


def load_values(sample, last_generations):
    path = result_path(sample.memory, sample.p_noise)
    if not path.exists():
        raise FileNotFoundError(f"No results for {sample.label()} ({path}). Run 'python run.py' first!")

    data = np.load(path)
    if sample.measure == "COOPERATION":
        end_of_run = data["cooperation"][:, -last_generations:]  # (seeds, generations)
    elif sample.measure == "ENTROPY":
        end_of_run = data["mean_bit_entropy"][:, -last_generations:]  # (seeds, generations)
    else:
        if sample.memory != 1:
            raise ValueError(f"Memory {sample.memory} strategies are not identified: use COOPERATION or ENTROPY.")
        labels = [str(label) for label in data["labels"]]  # column names of named_frequencies
        if sample.measure not in labels:
            raise ValueError(
                f"Unknown measure '{sample.measure}' for memory {sample.memory}: "
                f"use COOPERATION, ENTROPY or one of {labels}"
            )
        column = labels.index(sample.measure)
        end_of_run = data["named_frequencies"][:, -last_generations:, column]  # (seeds, generations)

    return data["seeds"], end_of_run.mean(axis=1)


def rank_sum_test(sample_a, sample_b, last_generations): # two-sided Wilcoxon rank-sum test (independent samples)
    if sample_a.measure != sample_b.measure:
        raise ValueError("sample_a and sample_b must have the same measure!")
    _, values_a = load_values(sample_a, last_generations)
    _, values_b = load_values(sample_b, last_generations)

    all_values = np.concatenate([values_a, values_b])
    if np.all(all_values == all_values[0]):
        # Every seed has the same value in both samples: there is no difference to test
        statistic, p_value = len(values_a) * len(values_b) / 2, 1.0
    else:
        statistic, p_value = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")

    return {
        "test": "Wilcoxon rank-sum",
        "sample_a": sample_a.label(),
        "sample_b": sample_b.label(),
        "median_a": float(np.median(values_a)),
        "median_b": float(np.median(values_b)),
        "statistic": float(statistic),
        "p_value": float(p_value),
    }


def vargha_delaney_a12(sample_a, sample_b, last_generations): # prob of that seed of A gives a larger value than a seed of B.
    # 0.5 = no difference, 1 = A always larger, 0 = B always larger.
    # It compares every seed of A with every seed of B (independent samples)
    if sample_a.measure != sample_b.measure:
        raise ValueError("sample_a and sample_b must have the same measure!")
    _, values_a = load_values(sample_a, last_generations)
    _, values_b = load_values(sample_b, last_generations)

    # differences[i, j] = A[i] - B[j], every value of A against every value of B
    differences = values_a[:, np.newaxis] - values_b[np.newaxis, :]
    wins = np.sum(differences > 0)
    ties = np.sum(differences == 0)
    a12 = (wins + 0.5 * ties) / differences.size

    return {
        "test": "Vargha-Delaney A12",
        "sample_a": sample_a.label(),
        "sample_b": sample_b.label(),
        "a12": float(a12)
    }

def percentile_interval(bootstrap_values, confidence):
    alpha = 1 - confidence
    low = np.percentile(bootstrap_values, 100 * alpha / 2)
    high = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
    return float(low), float(high)


def bootstrap_median_ci(values, n_bootstrap=10000, confidence=0.95, seed=0):
    rng = np.random.default_rng(seed)
    medians = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        resample = rng.choice(values, size=len(values), replace=True)
        medians[b] = np.median(resample)
    return percentile_interval(medians, confidence)


def bootstrap_median_difference(sample_a, sample_b, last_generations, n_bootstrap=10000, confidence=0.95, seed=0):
    _, values_a = load_values(sample_a, last_generations)
    _, values_b = load_values(sample_b, last_generations)

    rng = np.random.default_rng(seed)
    differences = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        resample_a = rng.choice(values_a, size=len(values_a), replace=True)
        resample_b = rng.choice(values_b, size=len(values_b), replace=True)
        differences[b] = np.median(resample_a) - np.median(resample_b)
    ci_low, ci_high = percentile_interval(differences, confidence)

    return {
        "test": "Bootstrap difference of medians",
        "sample_a": sample_a.label(),
        "sample_b": sample_b.label(),
        "median_difference": float(np.median(values_a) - np.median(values_b)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence": confidence,
    }


def compare(sample_a, sample_b, last_generations):
    _, values_a = load_values(sample_a, last_generations)
    _, values_b = load_values(sample_b, last_generations)
    rank_sum = rank_sum_test(sample_a, sample_b, last_generations)
    effect = vargha_delaney_a12(sample_a, sample_b, last_generations)
    difference = bootstrap_median_difference(sample_a, sample_b, last_generations)
    return {
        "sample_a": sample_a.label(),
        "sample_b": sample_b.label(),
        "median_a": rank_sum["median_a"],
        "median_a_ci": bootstrap_median_ci(values_a, confidence = 0.95, seed = 0),
        "median_b": rank_sum["median_b"],
        "median_b_ci": bootstrap_median_ci(values_b, confidence = 0.95, seed = 0),
        "median_difference": difference["median_difference"],
        "median_difference_ci": (difference["ci_low"], difference["ci_high"]),
        "p_rank_sum": rank_sum["p_value"],
        "a12": effect["a12"],
    }


def measures_of(memory):
    labels = list(NAMED_STRATEGIES)
    labels.append("OTHER")
    if memory == 1:
        return ["COOPERATION", "ENTROPY"] + labels
    return ["COOPERATION", "ENTROPY"] 


def write_csv(path, rows):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def summary_rows(memories, noise_levels):
    rows = []
    for memory in memories:
        for p_noise in noise_levels:
            for measure in measures_of(memory):
                _, values = load_values(Sample(measure, p_noise, memory), 100)
                ci_low, ci_high = bootstrap_median_ci(values, confidence = 0.95, seed = 0)
                rows.append({
                    "memory": memory,
                    "p_noise": p_noise,
                    "measure": measure,
                    "median": float(np.median(values)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_seeds": len(values),
                })
    return rows


def comparison_pairs(memories, noise_levels, control_noise):
    pairs = []
    for memory in memories:
        for p_noise in noise_levels:
            if p_noise == control_noise:
                continue
            for measure in measures_of(memory):
                pairs.append((Sample(measure, p_noise, memory), Sample(measure, control_noise, memory)))

    for memory in memories:
        for index in range(1, len(noise_levels)):
            previous_noise = noise_levels[index - 1]
            p_noise = noise_levels[index]
            if previous_noise == control_noise:
                continue  # already compared against the control
            for measure in measures_of(memory):
                pairs.append((Sample(measure, p_noise, memory), Sample(measure, previous_noise, memory)))

    if 1 in memories and 2 in memories:
        for p_noise in noise_levels:
            for measure in measures_of(2):  # COOPERATION and ENTROPY
                pairs.append((Sample(measure, p_noise, 2), Sample(measure, p_noise, 1)))
    return pairs


def comparison_rows(pairs):
    rows = []
    for sample_a, sample_b in pairs:
        result = compare(sample_a, sample_b, last_generations = 100)
        rows.append({
            "sample_a": result["sample_a"],
            "sample_b": result["sample_b"],
            "median_a": result["median_a"],
            "median_b": result["median_b"],
            "median_difference": result["median_difference"],
            "diff_ci_low": result["median_difference_ci"][0],
            "diff_ci_high": result["median_difference_ci"][1],
            "p_rank_sum": result["p_rank_sum"],
            "a12": result["a12"],
        })
    return rows


def write_statistics(memories, noise_levels, control_noise):
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(STATS_DIR / "summary.csv", summary_rows(memories, noise_levels))

    rows = comparison_rows(comparison_pairs(memories, noise_levels, control_noise))
    write_csv(STATS_DIR / "comparisons.csv", rows)

    # Short summary on screen: only the cooperation rate
    print("\nCooperation rate, difference of medians (A - B) [95% CI], A12, p (rank-sum):")
    for row in rows:
        if row["sample_a"].startswith("COOPERATION"):
            print(
                f"  {row['sample_a']} vs {row['sample_b']}: "
                f"{row['median_difference']:+.3f} [{row['diff_ci_low']:+.3f}, {row['diff_ci_high']:+.3f}], "
                f"A12 = {row['a12']:.2f}, p = {row['p_rank_sum']:.2g}"
            )
