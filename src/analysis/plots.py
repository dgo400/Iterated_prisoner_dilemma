import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments import result_path

PLOTS_DIR = Path("plots")

# Colors (validated categorical palette, always in this fixed order).
# A noise level always gets the same color, and a strategy always gets the same color in every figure.
SERIES_COLORS = ["tab:blue", "tab:orange", "tab:green", "gold", "tab:pink", "tab:red", "tab:purple", "tab:cyan"]
STRATEGY_COLORS = {
    "ALLC": "tab:blue",
    "ALLD": "tab:orange",
    "TFT": "tab:green",
    "PAVLOV": "gold",
    "GRIM": "tab:pink",
    "OTHER": "tab:gray",  
}
TEXT_COLOR = "dimgray"
GRID_COLOR = "lightgray"
REFERENCE_COLOR = "tab:gray"

def load_results(memory, p_noise):
    path = result_path(memory, p_noise)
    return np.load(path)


def style_axis(axis):
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(colors=TEXT_COLOR)


def save_figure(figure, output_file):
    PLOTS_DIR.mkdir(exist_ok=True)
    figure.savefig(PLOTS_DIR / output_file, dpi=150)
    plt.close(figure)
    print(f"Saved {PLOTS_DIR / output_file}")


def make_panels(n_panels):
    n_columns = 3
    n_rows = math.ceil(n_panels / n_columns)
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(11, 3.2 * n_rows), sharex=True, sharey=True)
    return figure, list(np.array(axes).flatten())


def put_legend(figure, axes, n_panels, handles, labels):
    if len(axes) > n_panels:
        legend_axis = axes[n_panels]
        legend_axis.axis("off")
        legend_axis.legend(handles, labels, loc="center", frameon=False)
        for empty_axis in axes[n_panels + 1:]:
            empty_axis.axis("off")
    else:
        figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)


def plot_over_generations(memory, noise_levels, key, y_label, y_limits, title, output_file, reference_lines):
    figure, axis = plt.subplots(figsize=(7, 4))

    for index, p_noise in enumerate(noise_levels):
        values = load_results(memory, p_noise)[key]  # (seeds, generations)
        generations = np.arange(values.shape[1])
        median = np.median(values, axis=0)
        lower = np.percentile(values, 25, axis=0)
        upper = np.percentile(values, 75, axis=0)

        color = SERIES_COLORS[index]
        axis.plot(generations, median, color=color, linewidth=1.5, label=f"p_noise = {p_noise}")
        axis.fill_between(generations, lower, upper, color=color, alpha=0.2, linewidth=0)

    if reference_lines is not None:
        for y_value, text in reference_lines.items():
            axis.axhline(y_value, color=REFERENCE_COLOR, linewidth=1, linestyle="--")
            axis.text(generations[-1], y_value, f" {text}", color=TEXT_COLOR, va="bottom", ha="right", fontsize=8)

    axis.set_xlabel("Generation")
    axis.set_ylabel(y_label)
    axis.set_ylim(*y_limits)
    axis.set_title(title)
    style_axis(axis)
    axis.legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, output_file)


def plot_cooperation(memory, noise_levels):
    plot_over_generations(
        memory, noise_levels, key="cooperation",
        y_label="Cooperation rate (fraction of C moves)", y_limits=(0, 1),
        title=f"Cooperation over generations (Memory {memory})",
        output_file=f"cooperation_memory_{memory}.png",
        reference_lines = None
    )


def plot_mean_payoff(memory, noise_levels):
    plot_over_generations(
        memory, noise_levels, key="mean_payoff",
        y_label="Mean payoff per round", y_limits=(0, 3.3),
        title=f"Mean payoff over generations (Memory {memory})",
        output_file=f"payoff_memory_{memory}.png",
        reference_lines={3.0: "full cooperation (R)", 1.0: "full defection (P)"},
    )


def plot_diversity(memory, noise_levels):
    plot_over_generations(
        memory, noise_levels, key="mean_bit_entropy",
        y_label="Mean bit entropy", y_limits=(0, 1),
        title=f"Diversity over generations (Memory {memory})",
        output_file=f"diversity_memory_{memory}.png",
        reference_lines=None
    )


def plot_strategy_frequencies(memory, noise_levels):
    n_panels = len(noise_levels)
    figure, axes = make_panels(n_panels)

    for axis, p_noise in zip(axes, noise_levels):
        data = load_results(memory, p_noise)
        labels = [str(label) for label in data["labels"]]
        frequencies = data["named_frequencies"]  # (seeds, generations, labels)
        generations = np.arange(frequencies.shape[1])

        for column, label in enumerate(labels):
            mean = frequencies[:, :, column].mean(axis=0)
            axis.plot(generations, mean, color=STRATEGY_COLORS[label], linewidth=1.5, label=label)

        axis.set_title(f"p_noise = {p_noise}")
        axis.set_ylim(0, 1)
        style_axis(axis)

    for axis in axes[:n_panels]:
        axis.set_xlabel("Generation")
        axis.set_ylabel("Frequency (mean over seeds)")

    handles, labels = axes[0].get_legend_handles_labels()
    put_legend(figure, axes, n_panels, handles, labels)
    figure.suptitle(f"Strategy frequencies over generations (Memory {memory})")
    figure.tight_layout()
    save_figure(figure, f"strategies_memory_{memory}.png")


def plot_single_seeds(memory, noise_levels, seeds) -> None:
    n_panels = len(noise_levels)
    figure, axes = make_panels(n_panels)

    for axis, p_noise in zip(axes, noise_levels):
        data = load_results(memory, p_noise)
        seed_list = list(data["seeds"])
        for index, seed in enumerate(seeds):
            row = seed_list.index(seed)  # row of this seed in the results file
            cooperation = data["cooperation"][row]
            axis.plot(cooperation, color=SERIES_COLORS[index], linewidth=1, label=f"seed {seed}")

        axis.set_title(f"p_noise = {p_noise}")
        axis.set_ylim(0, 1)
        style_axis(axis)

    for axis in axes[:n_panels]:
        axis.set_xlabel("Generation")
        axis.set_ylabel("Cooperation rate")

    handles, labels = axes[0].get_legend_handles_labels()
    put_legend(figure, axes, n_panels, handles, labels)
    figure.suptitle(f"Cooperation in single runs (Memory {memory})")
    figure.tight_layout()
    save_figure(figure, f"seeds_memory_{memory}.png")


def plot_all(memory, noise_levels):
    plot_cooperation(memory, noise_levels)
    plot_mean_payoff(memory, noise_levels)
    plot_diversity(memory, noise_levels)
    if memory == 1:  # memory 2 strategies are not identified
        plot_strategy_frequencies(memory, noise_levels)
    plot_single_seeds(memory, noise_levels, seeds = [0, 1, 2, 3, 4]) # five first seeds by default
