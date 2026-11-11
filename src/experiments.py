import random
from collections import Counter
from pathlib import Path

import numpy as np
from deap import algorithms, base, creator, tools

from game import play_many_matches, PayoffMatrix 
from strategies import (
    NAMED_STRATEGIES,
    from_int,
    strategy_length,
    strategy_name,
    to_int,
)

RESULTS_DIR = Path("results") # save results
def result_path(memory, p_noise):
    return RESULTS_DIR / f"memory_{memory}" / f"noise_{p_noise}.npz"


class ExperimentConfig: # store the fixed parameters of each experiment
    def __init__(self):
        self.memory = 1
        self.population_size = 100
        self.n_generations = 500
        self.crossover_prob = 0.7 # pc, uniform crossover
        self.mutation_prob_bit = 0.01 # pm, bit-flip probability per bit
        self.tournament_size = 3 # number of randomly choosen individuals from the population
        self.n_elites = 2
        self.n_rounds = 150
        self.n_repeats = 20 # matches averaged per pair when there is noise



class MatchResults:
    def __init__(self, config, p_noise, rng):
        self.config = config
        self.length = strategy_length(config.memory)
        self.p_noise = p_noise
        self.n_rounds = config.n_rounds
        self.n_repeats = config.n_repeats
        if p_noise == 0:
            self.n_repeats = 1  # without noise every match is identical
        self.rng = rng
        self.payoff = {}  # (i, j) -> average payoff per round of strategy i against strategy j
        self.cooperation = {}  # (i, j) -> fraction of rounds in which i actually played C against j


    def play_pairs(self, pairs): # play all these pairs (i, j), each pair play n_repeats matches (results averaged over them)
        if len(pairs) == 0:
            return

        # One row of bits per pair, then each row repeated n_repeats times (one row per match)
        bits_i = np.array([from_int(i, self.length) for i, j in pairs]) 
        bits_j = np.array([from_int(j, self.length) for i, j in pairs])
        players = np.repeat(bits_i, self.n_repeats, axis=0)
        opponents = np.repeat(bits_j, self.n_repeats, axis=0)

        scores_i, scores_j, cooperations_i, cooperations_j = play_many_matches(players, opponents, n_rounds=self.n_rounds, p_noise=self.p_noise, rng=self.rng, payoffs=PayoffMatrix())

        # The rows of the same pair are consecutive: reshape to (pairs, repeats) and add them up
        n_pairs = len(pairs)
        n_total_rounds = self.n_repeats * self.n_rounds
        payoff_i = scores_i.reshape(n_pairs, self.n_repeats).sum(axis=1) / n_total_rounds
        payoff_j = scores_j.reshape(n_pairs, self.n_repeats).sum(axis=1) / n_total_rounds
        cooperation_i = cooperations_i.reshape(n_pairs, self.n_repeats).sum(axis=1) / n_total_rounds
        cooperation_j = cooperations_j.reshape(n_pairs, self.n_repeats).sum(axis=1) / n_total_rounds

        for k, (i, j) in enumerate(pairs):
            self.payoff[(i, j)] = float(payoff_i[k])
            self.payoff[(j, i)] = float(payoff_j[k])
            self.cooperation[(i, j)] = float(cooperation_i[k])
            self.cooperation[(j, i)] = float(cooperation_j[k])

    def play_missing_pairs(self, numbers): # iterate over all posible configurations (numbers) to ensure all strategies matched 
        missing = []
        for position, i in enumerate(numbers):
            for j in numbers[position:]: 
                if (i, j) not in self.payoff:
                    missing.append((i, j))
        self.play_pairs(missing)

    def get_payoff(self, i, j): 
        if (i, j) not in self.payoff:
            self.play_pairs([(i, j)])
        return self.payoff[(i, j)]

    def get_cooperation(self, i, j):
        if (i, j) not in self.cooperation:
            self.play_pairs([(i, j)])
        return self.cooperation[(i, j)]


def build_matrices(results):
    memory = results.config.memory
    n_bits = strategy_length(memory)
    n_strategies = 2 ** n_bits 

    results.play_missing_pairs(list(range(n_strategies)))

    payoff_matrix = np.zeros((n_strategies, n_strategies))
    cooperation_matrix = np.zeros((n_strategies, n_strategies))
    for i in range(n_strategies):
        for j in range(n_strategies):
            payoff_matrix[i, j] = results.payoff[(i, j)]
            cooperation_matrix[i, j] = results.cooperation[(i, j)]
    return payoff_matrix, cooperation_matrix


def print_classical_table(payoff_matrix):
    names = list(NAMED_STRATEGIES)
    print("        " + "".join(f"{name:>8}" for name in names))
    for row_name in names:
        i = to_int(NAMED_STRATEGIES[row_name])
        row = "".join(f"{payoff_matrix[i, to_int(NAMED_STRATEGIES[c])]:8.2f}" for c in names)
        print(f"{row_name:>8}{row}")


if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def make_toolbox(config):
    n_bits = strategy_length(config.memory)
    toolbox = base.Toolbox()
    toolbox.register("attr_bit", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bit, n=n_bits)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=config.mutation_prob_bit)
    toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
    return toolbox


def evaluate_population(population, results):
    numbers = [to_int(individual) for individual in population] # turn into integer form to manage strategies
    counts = Counter(numbers)
    n_opponents = len(population) - 1

    # Play now, all at once, the pairs of strategies that have never met before
    results.play_missing_pairs(list(counts))

    # All the copies of a strategy have the same fitness, so it is computed once per strategy
    fitness_of = {}
    for i in counts:
        total = 0.0
        for j in counts:
            total += counts[j] * results.get_payoff(i, j)
        total -= results.get_payoff(i, i)  # remove the game against itself
        fitness_of[i] = total / n_opponents

    for individual, number in zip(population, numbers):
        individual.fitness.values = (fitness_of[number],)
    return counts


def population_cooperation(counts, results):
    population_size = sum(counts.values())
    n_opponents = population_size - 1

    total = 0.0
    for i in counts:
        # Cooperation rate of one individual of strategy i against the rest of the population
        cooperations = 0.0
        for j in counts:
            cooperations += counts[j] * results.get_cooperation(i, j)
        cooperations -= results.get_cooperation(i, i)  # remove the game against itself
        total += counts[i] * cooperations / n_opponents
    return total / population_size


def named_frequencies(counts):
    population_size = sum(counts.values())
    labels = list(NAMED_STRATEGIES)
    labels.append("OTHER")
    
    frequencies = np.zeros(len(labels))
    for number, count in counts.items():
        name = strategy_name(from_int(number, strategy_length(1)))
        frequencies[labels.index(name)] += count / population_size
    return frequencies


def bit_entropy(population) -> np.ndarray:
    bits = np.array(population)  # (individuals, positions)
    fraction_of_d = bits.mean(axis=0)  # p for each position

    entropy = np.zeros(len(fraction_of_d))
    for position, p in enumerate(fraction_of_d):
        if 0 < p < 1:  # if p is 0 or 1 everybody has the same bit: entropy 0
            entropy[position] = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return entropy


def run_evolution(results, config, seed):
    random.seed(seed)  
    toolbox = make_toolbox(config)
    population = toolbox.population(n=config.population_size)
    n_strategies = 2 ** strategy_length(results.config.memory)
    
    n_points = config.n_generations + 1
    mean_payoff = np.zeros(n_points)
    cooperation = np.zeros(n_points)
    entropy_per_bit = np.zeros((n_points, strategy_length(config.memory)))
    named = np.zeros((n_points, len(list(NAMED_STRATEGIES)) + 1))  # only filled with memory 1
    all_frequencies = np.zeros((n_points, n_strategies))  # only filled with memory 1

    for generation in range(n_points): # Coevolution: everyone is re-evaluated every generation (even elites)
        counts = evaluate_population(population, results)
        mean_payoff[generation] = np.mean([ind.fitness.values[0] for ind in population])
        cooperation[generation] = population_cooperation(counts, results)
        entropy_per_bit[generation] = bit_entropy(population)
        
        if config.memory == 1: # named strategies for memory 1
            named[generation] = named_frequencies(counts)
            for number, count in counts.items():
                all_frequencies[generation, number] = count / config.population_size

        if generation == config.n_generations:
            break

        elites = [toolbox.clone(ind) for ind in tools.selBest(population, config.n_elites)]
        parents = toolbox.select(population, config.population_size - config.n_elites)
        
        # mutpb=1.0: every child goes through mutFlipBit, which flips each bit with prob pm
        offspring = algorithms.varAnd(parents, toolbox, cxpb=config.crossover_prob, mutpb=1.0)
        population = elites + offspring

    history = {
        "mean_payoff": mean_payoff,
        "cooperation": cooperation,
        "bit_entropy": entropy_per_bit,  # (generations, positions)
        "mean_bit_entropy": entropy_per_bit.mean(axis=1),  # (generations,) average over positions
    }
    if config.memory == 1:
        history["named_frequencies"] = named
        history["frequencies"] = all_frequencies
    return history
