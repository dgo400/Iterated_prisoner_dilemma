# Coevolution of strategies in the Iterated Prisoner's Dilemma under noise

Authors: Diego Mallo Bozzolla, David Garcia Ortiz

## Introduction

This project studies how cooperation emerges and stays stable in an evolving population of strategies playing the Iterated Prisoner's Dilemma (IPD). Strategies are evolved with a genetic algorithm (DEAP), and every strategy plays against every other strategy in a round robin.

The main question is how the level of **noise** affects cooperation. With probability p_noise each intended action is flipped (C becomes D or D becomes C). We compare:

- `p_noise = 0` (control, no noise)
- `p_noise = 0.005`, `0.01`, `0.02`, `0.05`

We also compare **memory 1** strategies (the next move depends on the last round) with **memory 2** strategies (the next move depends on the last two rounds).

Payoffs are the classical ones: T = 5 (temptation), R = 3 (reward), P = 1 (punishment), S = 0 (sucker). The mean payoff per round is therefore between 0 (always sucker) and 3 (full cooperation); 1 is full mutual defection.

### Strategy encoding

A memory-1 strategy is a bit string of 5 bits: (first move, CC, CD, DC, DD). Each bit after the first is the action to play after the previous round outcome, written from the player's own point of view (own action first). This gives 2^5 = 32 possible strategies. The named ones are:

| Name   | Bits (first, CC, CD, DC, DD) | Behaviour                                          |
|--------|------------------------------|----------------------------------------------------|
| ALLC   | 0 0 0 0 0                    | Always cooperate                                   |
| ALLD   | 1 1 1 1 1                    | Always defect                                      |
| TFT    | 0 0 1 0 1                    | Start with C, then copy the opponent's last move   |
| PAVLOV | 0 0 1 1 0                    | Win-stay, lose-shift                               |
| GRIM   | 0 0 1 1 1                    | Cooperate until the first defection, then defect   |

A memory-2 strategy has 2 + 16 = 18 bits (the first two moves plus one bit per possible outcome of the last two rounds). Memory-2 strategies do not have names.

## Set up the environment

Requirements: Python 3.10 or newer.

From the project folder, in a terminal (Windows PowerShell shown; on Linux/macOS use `source .venv/bin/activate`):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

- `requirements.txt` installs numpy, deap, matplotlib and scipy.
- `pip install -e .` makes the modules in `src/` importable (`game`, `strategies`, `experiments`, `analysis`). Without it, `python run.py` fails with `ModuleNotFoundError`.

## Module structure

```
run.py                          # Entry point: runs all the experiments, plots and statistics
src/
  strategies.py                 # Bit-string strategies
  game.py                       # Match engine
  experiments.py                # Genetic algorithm (DEAP), fitness, cooperation, diversity result paths
  analysis/
    plots.py                    # Figures (cooperation, payoff, diversity, strategy frequencies, single seeds)
    statistical_report.py       # Summary table and statistical tests (CSV files)
```

- `run.py` sets the study parameters (`MEMORIES`, `NOISE_LEVELS`, `SEEDS`) and runs the loops over memory, noise and seed.
- `experiments.py` holds the fixed parameters in the class `ExperimentConfig`, and the evolution in `run_evolution()`.
- `game.py` plays many matches at once with NumPy (vectorised over matches). `play_match()` is a simple one-match version, kept for reference.

### Genetic algorithm

Default parameters (`ExperimentConfig` in `src/experiments.py`):

| Parameter                         | Value |
|-----------------------------------|-------|
| Population size                   | 100   |
| Generations                       | 500   |
| Crossover probability (uniform)   | 0.7   |
| Mutation: bit-flip probability    | 0.01 per bit (every child is mutated) |
| Tournament size                   | 3     |
| Elites (kept each generation)     | 2     |
| Rounds per match                  | 150   |
| Repeated matches per pair (noise) | 20 (1 when `p_noise = 0`) |

Each generation:

1. Every pair of strategies in the population plays (results are cached, so a pair is played only once per seed).
2. The fitness of an individual is its mean payoff per round against the other individuals of the population (the game against itself is removed).
3. The 2 best individuals are copied to the next generation (elitism).
4. The rest of the population is chosen by tournament selection, recombined with uniform crossover and mutated with bit flips.

The evolution is coevolutionary: everybody is evaluated again in each generation, against the current population.

## Run the experiment

### How to run it

From the project folder (with the environment active):

```powershell
python run.py
```

### How to change parameters or run only part of the study

There is no command-line option. Edit the constants at the top of `run.py`:

```python
MEMORIES = [1, 2]                                 # which memories to run
NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.05]     # first value is the control (0.0)
SEEDS = list(range(30))                           # random seeds, one run per seed
```

- To run only memory 1: set `MEMORIES = [1]`.
- To run only some noise levels: change `NOISE_LEVELS`. Keep `0.0` as the first value, because the statistics compare every noise level against the control.

**Warning:** `main()` in `run.py` deletes the whole `results/` folder at the start of every run. Copy the results you want to keep before running again. The `plots/` folder is not deleted, so old figures are overwritten only if the same file name is produced.

### How results are saved

Raw results are saved per memory and noise level, one compressed file each:

```
results/
  memory_1/
    noise_0.0.npz
    noise_0.005.npz
    noise_0.01.npz
    noise_0.02.npz
    noise_0.05.npz
  memory_2/
    ...
  stats/
    summary.csv
    comparisons.csv
plots/
  cooperation_memory_1.png, payoff_memory_1.png, diversity_memory_1.png,
  strategies_memory_1.png, seeds_memory_1.png, (and the same for memory 2, without strategies)
```

Read a result file with NumPy:

```python
import numpy as np
data = np.load("results/memory_1/noise_0.01.npz")
print(data.files)   # list of the arrays inside
```

### What the results include

Arrays in each `.npz` file. Dimensions: seeds (30) x generations (501, from generation 0 to 500).

| Key                 | Shape                       | Meaning                                                        |
|---------------------|-----------------------------|----------------------------------------------------------------|
| `seeds`             | (seeds,)                    | Seed of each run                                               |
| `mean_payoff`       | (seeds, generations)        | Mean payoff per round of the population                        |
| `cooperation`       | (seeds, generations)        | Fraction of C moves actually played (noise included)           |
| `bit_entropy`      | (seeds, generations, bits)  | Entropy of each bit position (0 = all equal, 1 = half and half) |
| `mean_bit_entropy`  | (seeds, generations)        | `bit_entropy` averaged over the bit positions (diversity)      |
| `labels`            | (named + 1,)                | Names of the columns of `named_frequencies` (memory 1)         |
| `named_frequencies` | (seeds, generations, 6)     | Frequency of ALLC, ALLD, TFT, PAVLOV, GRIM and OTHER (memory 1) |
| `frequencies`       | (seeds, generations, 32)    | Frequency of each of the 32 memory-1 strategies (memory 1)     |
| `payoff_matrix`     | (seeds, 32, 32)             | Payoff per round of strategy i against j, averaged over matches (memory 1) |
| `cooperation_matrix`| (seeds, 32, 32)             | Cooperation rate of strategy i against j (memory 1)            |

The matrices are computed only for memory 1 (they are cheap there). For memory 2 the file has only the first five keys.

`summary.csv` contains, for each memory, noise level and measure (COOPERATION, ENTROPY, and the strategy frequencies for memory 1): the median over the seeds of the mean over the last 100 generations, and its 95% bootstrap confidence interval.

`comparisons.csv` contains, for each pair of conditions:

- median of each sample and difference of medians with its 95% bootstrap CI,
- p-value of the Wilcoxon rank-sum test (Mann-Whitney U, independent samples),
- Vargha-Delaney A12 effect size.

The pairs compared are: each noise level against the control (`p_noise = 0`), each noise level against the previous one, and memory 2 against memory 1 for the same noise level.

The statistics use the last 100 generations of each run, so they describe the end state of the evolution, not the start.
