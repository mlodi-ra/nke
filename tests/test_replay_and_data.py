from pathlib import Path

import pytest

from nke_experiment.data import FAMILIES, assigned_split, oracle, trajectory, write_dataset
from nke_experiment.diagnose import report
from nke_experiment.sampling import balanced_rows
from nke_experiment.state import Event, StateError, StateStore


def test_replay_rejects_missing_versions_and_unknown_deletions():
    state = StateStore()
    state.apply(Event(1, "upsert", "a", {"kind": "worker", "capacity": 1}))
    with pytest.raises(StateError, match="expected version 2"):
        state.apply(Event(3, "delete", "a"))
    assert state.version == 1 and "a" in state.records
    with pytest.raises(StateError, match="unknown"):
        state.apply(Event(2, "delete", "missing"))
    assert state.version == 1
    state.apply(Event(2, "upsert", "a", {"kind": "worker", "capacity": 0}))
    assert StateStore.replay([Event(1, "upsert", "a", {"kind": "worker", "capacity": 1}),
                              Event(2, "upsert", "a", {"kind": "worker", "capacity": 0})]).snapshot() == state.snapshot()


def test_state_snapshot_cannot_mutate_store():
    state = StateStore()
    state.apply(Event(1, "upsert", "a", {"kind": "person", "role": "operator"}))
    state.snapshot()["records"]["a"]["role"] = "auditor"
    assert state.records["a"]["role"] == "operator"


def test_trajectories_have_oracle_labels_and_scenario_split():
    for family in FAMILIES:
        for scenario_id in range(20):
            rows = trajectory(family, scenario_id)
            assert len(rows) == 5
            assert len({row["split"] for row in rows}) == 1
            assert rows[0]["split"] == assigned_split(family, scenario_id)
            for row in rows:
                assert row["label"] == oracle(family, row["state"]["records"], row["candidates"])
                assert row["label"] in row["candidates"] or row["label"] is None
                assert StateStore.replay([Event(**event) for event in row["events"]]).snapshot() == row["state"]
            assert rows == trajectory(family, scenario_id)


def test_dataset_outputs_disjoint_scenarios(tmp_path: Path):
    counts = write_dataset(tmp_path, 40)
    assert sum(counts.values()) == 3 * 40 * 5
    seen = set()
    for split in ("train", "validation", "calibration", "test"):
        for line in (tmp_path / f"{split}.jsonl").read_text().splitlines():
            import json
            row = json.loads(line)
            assert row["split"] == split
            key = row["family"], row["scenario_id"]
            if row["step"] == 0:
                assert key not in seen
                seen.add(key)
    assert len(seen) == 120


def test_training_selection_balances_families_and_keeps_scenarios_intact():
    rows = [row for family in FAMILIES for scenario in range(20) for row in trajectory(family, scenario)]
    selected = balanced_rows(rows, 150)
    assert len(selected) == 150
    assert {family: sum(row["family"] == family for row in selected) for family in FAMILIES} == {
        family: 50 for family in FAMILIES}
    assert all(sum(row["family"] == family and row["scenario_id"] == scenario for row in selected) == 5
               for family in FAMILIES for scenario in range(10))


def test_baseline_uses_training_labels_only():
    train = [{"family": "access", "label": "deny"}] * 3 + [{"family": "access", "label": "allow"}]
    held_out = [{"family": "access", "label": "allow"}] * 4
    result = report(train, held_out)
    assert result["access"]["train_majority_label"] == "deny"
    assert result["access"]["majority_accuracy"] == 0.0
