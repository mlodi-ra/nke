"""Train Variant A heads on deterministic synthetic data, saving actual weights."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
from pathlib import Path

import torch
from safetensors.torch import save_file

from .model import SnapshotDecisionModel
from .sampling import balanced_rows


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evaluate(model: SnapshotDecisionModel, rows: list[dict]) -> dict[str, float]:
    model.eval()
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for row in rows:
            target = row["label"] or "none"
            guess = model.decide(row)["choice"]
            counts[row["family"]][0] += int(guess == target)
            counts[row["family"]][1] += 1
    report = {family: correct / n for family, (correct, n) in sorted(counts.items())}
    report["overall"] = sum(correct for correct, _ in counts.values()) / len(rows) if rows else float("nan")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a real Variant A neural checkpoint")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--backbone", default="answerdotai/ModernBERT-base")
    args = parser.parse_args()
    torch.manual_seed(17)
    rows = balanced_rows(read_rows(args.data / "train.jsonl"), args.limit)
    validation = balanced_rows(read_rows(args.data / "validation.jsonl"), max(15, args.limit // 5))
    if not rows or not validation or args.epochs < 1:
        raise ValueError("training and validation rows and positive epochs required")
    if {row["family"] for row in rows} != {row["family"] for row in validation}:
        raise ValueError("training and validation must cover the same families; increase --limit or dataset size")
    model = SnapshotDecisionModel(args.backbone)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    best = -1.0
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        random.Random(17 + epoch).shuffle(rows)
        model.train()
        total = 0.0
        for row in rows:
            choices = list(row["candidates"]) + ["none"]
            target = torch.tensor([choices.index(row["label"] or "none")], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(row).unsqueeze(0), target)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        accuracies = evaluate(model, validation)
        print(json.dumps({"epoch": epoch + 1, "train_loss": total / len(rows), "validation_accuracy": accuracies}))
        if accuracies["overall"] > best:
            best = accuracies["overall"]
            # Save trainable heads only; backbone and revision remain an external dependency.
            weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
                       if not k.startswith("encoder.")}
            save_file(weights, str(args.output / "nke_heads.safetensors"))
            (args.output / "config.json").write_text(json.dumps({"backbone": args.backbone,
                 "seed": 17, "train_rows_used": len(rows), "validation_rows_used": len(validation),
                 "epochs_completed": epoch + 1, "validation_accuracy": accuracies,
                 "sampling": "balanced complete scenarios across families; deterministic scenario IDs",
                 "limitations": "Synthetic Choice-only model; not calibrated; no field accuracy claim"}, indent=2) + "\n")
    print("Saved trained head weights and configuration to", args.output)


if __name__ == "__main__":
    main()
