"""Regenerate the native-versus-BR summary table from frozen per-pair results."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    grouped = frame.groupby("dataset", sort=False).agg(
        spf_excess=("excess", "median"),
        native_f1=("native_macro_f1", "median"),
        br_f1=("br_macro_f1", "median"),
    )
    latex_row_end = r" \\"
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        f"Dataset & SPF excess & Native F1 & BR F1{latex_row_end}",
        "\\midrule",
    ]
    for dataset, row in grouped.iterrows():
        lines.append(
            f"{dataset} & {row.spf_excess:.3f} & {row.native_f1:.3f} & {row.br_f1:.3f}{latex_row_end}"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
