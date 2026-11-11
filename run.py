import shutil
import time
from multiprocessing import Pool

import numpy as np

from analysis.plots import plot_all
from analysis.statistical_report import write_statistics
from experiments import (
    RESULTS_DIR,
    ExperimentConfig,
    MatchResults,
    build_matrices,
    result_path,
    run_evolution,
)
from strategies import NAMED_STRATEGIES

# Variable parameters of the study 
MEMORIES = [1, 2] 
NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.05] # p=0 for control; not too high noise
SEEDS = list(range(30))


def make_tasks(config, p_noise):
    extra_data = {}
    tasks = [] # store each seed result into tuples: (results, config, seed)
    payoff_matrices = []
    cooperation_matrices = []
    for seed in SEEDS:
        results = MatchResults(config, p_noise, np.random.default_rng(seed))
        
        if config.memory == 1: # relatively light, we can compute them 
            payoff_matrix, cooperation_matrix = build_matrices(results)
            payoff_matrices.append(payoff_matrix)
            cooperation_matrices.append(cooperation_matrix)
        tasks.append((results, config, seed))

    if config.memory == 1:
        extra_data["payoff_matrix"] = np.array(payoff_matrices)  # (seeds, 32, 32)
        extra_data["cooperation_matrix"] = np.array(cooperation_matrices)  # (seeds, 32, 32)
    return tasks, extra_data


def main() -> None:
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)

    n_runs = len(MEMORIES) * len(NOISE_LEVELS) * len(SEEDS)
    print(f"{len(MEMORIES)} memories x {len(NOISE_LEVELS)} noise levels x {len(SEEDS)} seeds = {n_runs} runs")
    start = time.time()

    # Pool() starts one worker process per CPU core
    with Pool() as pool:
        for memory in MEMORIES:
            config = ExperimentConfig()
            config.memory = memory
            labels = list(NAMED_STRATEGIES) + ["OTHER"]  # named strategies, memory 1 only

            for number, p_noise in enumerate(NOISE_LEVELS, start=1):
                tasks, extra_data = make_tasks(config, p_noise)

                # Each seed is an independent run, so they can run at the same time on different cores.
                # starmap calls run_evolution(*task) for every task and keeps the order of SEEDS.
                histories = pool.starmap(run_evolution, tasks)

                # Put the seeds together: one row per seed
                data = {
                    "seeds": np.array(SEEDS),
                    "mean_payoff": np.array([h["mean_payoff"] for h in histories]),  # (seeds, generations)
                    "cooperation": np.array([h["cooperation"] for h in histories]),  # (seeds, generations)
                    
                    # Diversity: (seeds, generations, positions) and its average over positions (seeds, generations)
                    "bit_entropy": np.array([h["bit_entropy"] for h in histories]),
                    "mean_bit_entropy": np.array([h["mean_bit_entropy"] for h in histories]),
                }
                if memory == 1:
                    data["labels"] = np.array(labels)  # column names of named_frequencies
                    # (seeds, generations, named strategies + OTHER)
                    data["named_frequencies"] = np.array([h["named_frequencies"] for h in histories])
                    # (seeds, generations, 32): frequency of every memory-1 strategy
                    data["frequencies"] = np.array([h["frequencies"] for h in histories])
                data.update(extra_data)

                path = result_path(memory, p_noise)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **data)  # every key of the dict becomes an array in the file

                top = ""
                if memory == 1:
                    final_frequencies = data["named_frequencies"][:, -1].mean(axis=0)  # last generation, mean over seeds
                    top = ", ".join(f"{name} {freq:.0%}" for name, freq in zip(labels, final_frequencies)) + " | "
                elapsed = time.time() - start
                print(
                    f"[memory {memory} | {number}/{len(NOISE_LEVELS)}] p_noise={p_noise:<5} | "
                    f"final payoff {data['mean_payoff'][:, -1].mean():.2f} | "
                    f"final C {data['cooperation'][:, -1].mean():.0%} | {top}{elapsed:.0f}s"
                )

            # Plots (saved in plots/) 
            plot_all(memory, NOISE_LEVELS)

    # (saved in results/stats/) 
    # NOISE_LEVELS[0] = 0.0 is the control
    write_statistics(MEMORIES, NOISE_LEVELS, control_noise=NOISE_LEVELS[0])


if __name__ == "__main__":
    main()
