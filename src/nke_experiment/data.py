"""Synthetic trajectories with exact reference labels; not evidence of field accuracy."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .state import Event, StateStore

FAMILIES = ("resource", "access", "priority")
SPLITS = ("train", "validation", "calibration", "test")


def assigned_split(family: str, scenario_id: int) -> str:
    digest = hashlib.sha256(f"nke-v2:{family}:{scenario_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 80:
        return "validation"
    if bucket < 90:
        return "calibration"
    return "test"


def _resource(rng: random.Random) -> tuple[list[tuple[str, dict]], list[dict], str, list[str]]:
    skill = rng.choice(["network", "database", "identity"])
    records = [("job", {"kind": "job", "skill": skill})]
    costs = rng.sample(range(1, 9), 4)
    for i in range(4):
        records.append((f"worker:{i}", {"kind": "worker", "skill": rng.choice([skill, "other"]),
                                        "capacity": rng.randrange(2), "cost": costs[i]}))
    updates = []
    for step in range(4):
        worker = f"worker:{rng.randrange(4)}"
        changes = {"capacity": rng.randrange(2)} if step % 2 else {"skill": rng.choice([skill, "other"])}
        updates.append({"entity_id": worker, "changes": changes})
    candidates = [f"worker:{i}" for i in range(4)]
    rng.shuffle(candidates)
    return records, updates, "Which worker can handle the job at the lowest cost?", candidates


def _access(rng: random.Random) -> tuple[list[tuple[str, dict]], list[dict], str, list[str]]:
    role = rng.choice(["analyst", "operator", "auditor"])
    records = [("person", {"kind": "person", "role": role}), ("policy", {"kind": "policy", "allowed_roles": ["operator"]})]
    updates = [{"entity_id": "person", "changes": {"role": rng.choice(["analyst", "operator", "auditor"])}} for _ in range(4)]
    return records, updates, "Should this person have access?", ["allow", "deny"]


def _priority(rng: random.Random) -> tuple[list[tuple[str, dict]], list[dict], str, list[str]]:
    records = [("incident", {"kind": "incident", "severity": rng.randrange(1, 6), "affected_services": rng.randrange(1, 4)})]
    updates = [{"entity_id": "incident", "changes": {"severity": rng.randrange(1, 6), "affected_services": rng.randrange(1, 4)}} for _ in range(4)]
    return records, updates, "What is the incident priority?", ["low", "medium", "high"]


def oracle(family: str, records: dict[str, dict[str, Any]], candidates: list[str]) -> str | None:
    if family == "resource":
        job = records["job"]
        eligible = [c for c in candidates if c in records and records[c]["skill"] == job["skill"] and records[c]["capacity"] > 0]
        return min(eligible, key=lambda c: (records[c]["cost"], c)) if eligible else None
    if family == "access":
        return "allow" if records["person"]["role"] in records["policy"]["allowed_roles"] else "deny"
    if family == "priority":
        incident = records["incident"]
        value = incident["severity"] + incident["affected_services"]
        return "high" if value >= 7 else "medium" if value >= 4 else "low"
    raise ValueError(family)


def trajectory(family: str, scenario_id: int) -> list[dict[str, Any]]:
    if family not in FAMILIES:
        raise ValueError(family)
    rng = random.Random(f"nke-v2:{family}:{scenario_id}")
    records, updates, question, candidates = {"resource": _resource, "access": _access, "priority": _priority}[family](rng)
    state = StateStore()
    events: list[Event] = []
    rows = []
    for entity_id, record in records:
        event = Event(state.version + 1, "upsert", entity_id, record)
        state.apply(event)
        events.append(event)
    for step in range(5):
        snapshot = state.snapshot()
        rows.append({"family": family, "scenario_id": scenario_id, "step": step,
                     "split": assigned_split(family, scenario_id), "state": snapshot,
                     "events": [asdict(event) for event in events],
                     "question": question, "candidates": candidates,
                     "label": oracle(family, state.records, candidates)})
        if step < len(updates):
            update = updates[step]
            entity_id = update["entity_id"]
            new_record = {**state.records[entity_id], **update["changes"]}
            event = Event(state.version + 1, "upsert", entity_id, new_record)
            state.apply(event)
            events.append(event)
    return rows


def write_dataset(path: Path, scenarios_per_family: int) -> dict[str, int]:
    if scenarios_per_family < 1:
        raise ValueError("scenarios_per_family must be positive")
    path.mkdir(parents=True, exist_ok=True)
    counts = {split: 0 for split in SPLITS}
    handles = {split: (path / f"{split}.jsonl").open("w", encoding="utf-8") for split in SPLITS}
    try:
        for family in FAMILIES:
            for scenario_id in range(scenarios_per_family):
                for row in trajectory(family, scenario_id):
                    handles[row["split"]].write(json.dumps(row, sort_keys=True) + "\n")
                    counts[row["split"]] += 1
    finally:
        for handle in handles.values():
            handle.close()
    (path / "manifest.json").write_text(json.dumps({"generator": "nke-v2", "scenarios_per_family": scenarios_per_family,
                                                    "families": FAMILIES, "rows": counts,
                                                    "warning": "Synthetic oracle labels are not field accuracy"}, indent=2) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic NKE research trajectories")
    parser.add_argument("--output", type=Path, default=Path("data/generated"))
    parser.add_argument("--scenarios", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(write_dataset(args.output, args.scenarios), sort_keys=True))


if __name__ == "__main__":
    main()
