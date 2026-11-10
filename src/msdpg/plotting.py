"""Matplotlib plots for ms-DPG explanations."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

def plot_spf(explanation, ax=None):
    """Plot the SPF matrix."""
    ax = ax or plt.subplots()[1]
    image = ax.imshow(explanation.spf, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(explanation.spf.columns)), explanation.spf.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(explanation.spf.index)), explanation.spf.index)
    ax.set_xlabel("target label"); ax.set_ylabel("source label")
    ax.figure.colorbar(image, ax=ax, label="SPF")
    return ax

def plot_lifts(pair, ax=None):
    """Plot source and target lifts of shared predicates on logarithmic axes."""
    ax = ax or plt.subplots()[1]
    data = pair.shared
    colors = {"co-enriching": "#2a9d8f", "separating": "#e76f51", "co-depleting": "#457b9d"}
    for kind, group in data.groupby("kind"):
        ax.scatter(group.lift_A, group.lift_B, label=kind, color=colors[kind])
    ax.axhline(1, color="black", lw=.7); ax.axvline(1, color="black", lw=.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"lift: {pair.source}"); ax.set_ylabel(f"lift: {pair.target}")
    ax.legend()
    return ax
