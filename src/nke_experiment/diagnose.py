"""Reproducible per-family majority baselines and optional checkpoint errors."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def report(train: list[dict], held_out: list[dict], model=None, max_errors: int = 3) -> dict:
    if not train or not held_out:
        raise ValueError("both training and held-out splits must contain rows")
    families = sorted({row["family"] for row in held_out})
    train_counts = {family: Counter(row["label"] or "none" for row in train if row["family"] == family)
                    for family in families}
    if any(not counts for counts in train_counts.values()):
        raise ValueError("training split lacks a held-out family")
    majority = {family: sorted(counts, key=lambda label: (-counts[label], label))[0]
                for family, counts in train_counts.items()}
    output = {}
    for family in families:
        rows = [row for row in held_out if row["family"] == family]
        labels = [row["label"] or "none" for row in rows]
        baseline = majority[family]
        entry = {"n": len(rows), "train_majority_label": baseline,
                 "label_counts": dict(sorted(Counter(labels).items())),
                 "majority_accuracy": sum(label == baseline for label in labels) / len(rows)}
        if model is not None:
            correct = 0
            by_step = defaultdict(lambda: [0, 0])
            errors = []
            for row, target in zip(rows, labels):
                predicted = model.decide(row)["choice"]
                hit = predicted == target
                correct += hit
                by_step[row["step"]][0] += hit
                by_step[row["step"]][1] += 1
                if not hit and len(errors) < max_errors:
                    errors.append({"scenario_id": row["scenario_id"], "step": row["step"],
                                   "expected": target, "predicted": predicted})
            entry.update({"model_accuracy": correct / len(rows),
                          "accuracy_delta_vs_majority": correct / len(rows) - entry["majority_accuracy"],
                          "model_accuracy_by_step": {str(step): hits / n for step, (hits, n) in sorted(by_step.items())},
                          "error_examples": errors})
        output[family] = entry
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare held-out results to train-split majority baselines")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "calibration", "test"), default="test")
    parser.add_argument("--checkpoint", type=Path, help="optional trained head checkpoint")
    args = parser.parse_args()
    model = None
    if args.checkpoint:
        from .checkpoint import load_checkpoint
        model = load_checkpoint(args.checkpoint)
    result = report(read_rows(args.data / "train.jsonl"), read_rows(args.data / f"{args.split}.jsonl"), model)
    print(json.dumps({"split": args.split, "synthetic_oracle_labels": True, "by_family": result}, indent=2))


if __name__ == "__main__":
    main()
