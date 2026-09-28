"""Generate a separate final set after architecture and tuning are fixed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import FAMILIES, trajectory


def write_holdout(path: Path, start_id: int = 10000, scenarios: int = 200) -> None:
    if start_id < 0 or scenarios < 1:
        raise ValueError("nonnegative start and positive scenario count required")
    path.mkdir(parents=True, exist_ok=True)
    with (path / "test.jsonl").open("w", encoding="utf-8") as output:
        for family in FAMILIES:
            for scenario_id in range(start_id, start_id + scenarios):
                for row in trajectory(family, scenario_id):
                    row["split"] = "test"
                    output.write(json.dumps(row, sort_keys=True) + "\n")
    (path / "manifest.json").write_text(json.dumps({"generator": "nke-v3",
         "purpose": "fresh final evaluation; generate only after model selection",
         "start_id": start_id, "end_id_exclusive": start_id + scenarios,
         "scenarios_per_family": scenarios, "rows": scenarios * len(FAMILIES) * 5,
         "warning": "Synthetic oracle labels are not field accuracy"}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a separate synthetic final holdout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-id", type=int, default=10000)
    parser.add_argument("--scenarios", type=int, default=200)
    args = parser.parse_args()
    write_holdout(args.output, args.start_id, args.scenarios)


if __name__ == "__main__":
    main()
