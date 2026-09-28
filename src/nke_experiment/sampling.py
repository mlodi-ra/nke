"""Keep all steps of a scenario in one balanced training selection."""

from __future__ import annotations

from collections import defaultdict


def balanced_rows(rows: list[dict], limit: int) -> list[dict]:
    if limit < 1:
        raise ValueError("row limit must be positive")
    groups: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["family"]][row["scenario_id"]].append(row)
    scenarios = {family: [groups[family][sid] for sid in sorted(groups[family])]
                 for family in sorted(groups)}
    selected: list[dict] = []
    while any(scenarios.values()):
        progressed = False
        for family in sorted(scenarios):
            if not scenarios[family]:
                continue
            group = scenarios[family][0]
            if len(selected) + len(group) > limit:
                continue
            selected.extend(scenarios[family].pop(0))
            progressed = True
        if not progressed:
            break
    return selected
