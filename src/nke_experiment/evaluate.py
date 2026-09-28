"""Held-out Choice evaluation; never infer general intelligence from synthetic data."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from .checkpoint import load_checkpoint
from .train import read_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained head checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "calibration", "test"), default="test")
    args = parser.parse_args()
    rows = read_rows(args.data / f"{args.split}.jsonl")
    if not rows:
        raise ValueError("evaluation split is empty")
    model = load_checkpoint(args.checkpoint)
    results = defaultdict(lambda: {"n": 0, "correct": 0, "brier_sum": 0.0, "log_loss_sum": 0.0})
    for row in rows:
        prediction = model.decide(row)
        probabilities = prediction["probabilities"]
        target = row["label"] or "none"
        metrics = results[row["family"]]
        metrics["n"] += 1
        metrics["correct"] += prediction["choice"] == target
        metrics["brier_sum"] += sum((p - float(name == target)) ** 2 for name, p in probabilities.items())
        metrics["log_loss_sum"] -= math.log(max(probabilities[target], 1e-12))
    report = {family: {"n": v["n"], "accuracy": v["correct"] / v["n"],
                       "brier": v["brier_sum"] / v["n"], "log_loss": v["log_loss_sum"] / v["n"]}
              for family, v in sorted(results.items())}
    print(json.dumps({"split": args.split, "synthetic_oracle_labels": True, "by_family": report}, indent=2))


if __name__ == "__main__":
    main()
