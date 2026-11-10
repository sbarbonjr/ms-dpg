"""Regenerate SPF heatmaps from the frozen atlas JSON."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def heatmap(ax, record: dict, show_labels: bool) -> None:
    matrix = np.asarray(record["spf"], dtype=float)
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_title(record["dataset"])
    if show_labels:
        labels = record["labels"]
        ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=6)
        ax.set_yticks(range(len(labels)), labels, fontsize=6)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    return image

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if args.dataset:
        record = data[args.dataset]
        fig, ax = plt.subplots(figsize=(6, 5))
        image = heatmap(ax, record, True)
        fig.colorbar(image, ax=ax, label="SPF")
        ax.set_xlabel("target label"); ax.set_ylabel("source label")
    else:
        names = list(data)
        cols = 3; rows = int(np.ceil(len(names) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(9, 3 * rows))
        image = None
        for ax, name in zip(axes.flat, names):
            image = heatmap(ax, data[name], False)
        for ax in axes.flat[len(names):]: ax.axis("off")
        fig.colorbar(image, ax=axes.ravel().tolist(), label="SPF", shrink=.75)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")

if __name__ == "__main__":
    main()
