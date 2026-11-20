"""Charts for the notebook and the project report."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN_LABELS = {
    "cpu-1": "CPU × 1",
    "cpu-ddp-2": "CPU × 2 (DDP)",
    "cuda-1": "CUDA × 1",
}


def read_results(directory: Path) -> dict:
    return {label: json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
            for name, label in RUN_LABELS.items()}


def validation_figure(results: dict):
    metric = next(iter(results.values()))["selection_metric"]
    comparison_label, run = max(results.items(), key=lambda entry: entry[1]["best"][metric])
    figure, (curves, bars) = plt.subplots(
        1, 2, figsize=(11.2, 4.1), layout="constrained", width_ratios=(1.4, 1)
    )
    colors = dict(zip(results, ("#13604F", "#8B4F81", "#42788F")))
    curves.axhline(run["popularity"][metric], color="#78828B", linestyle="--",
                   linewidth=1.5, label="Популярность")
    for label, result in results.items():
        history = result["history"]
        curves.plot([row["epoch"] for row in history], [row[metric] for row in history],
                    label=label, color=colors[label], linewidth=1.8)
    best_epoch = run["best"]["epoch"]
    highlight = colors[comparison_label]
    curves.scatter(best_epoch, run["best"][metric], color=highlight, zorder=3, s=24)
    curves.set(title="Качество по эпохам", xlabel="Эпоха", ylabel=metric.upper())
    curves.legend(loc="lower right", fontsize=8, frameon=False)

    metrics = list(run["popularity"])
    positions = np.arange(len(metrics))
    width = 0.34
    for shift, label, values, color in (
        (-width / 2, "Популярность", run["popularity"], "#78828B"),
        (width / 2, "ENMF-U", run["best"], highlight),
    ):
        columns = bars.bar(positions + shift, [values[key] for key in metrics],
                           width, label=label, color=color, zorder=3)
        bars.bar_label(columns, fmt="%.3f", padding=3, fontsize=8)
    highest_score = max(max(run["best"][key], run["popularity"][key]) for key in metrics)
    bars.set(
        title=f"Сравнение на {best_epoch}-й эпохе ({comparison_label})",
        xticks=positions,
        xticklabels=[key.upper() for key in metrics],
        ylim=(0, 1.18 * highest_score),
    )
    bars.legend(loc="upper left", fontsize=8, frameon=False, ncols=2)
    for axis in (curves, bars):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
    return figure
