"""Authoritative replay state. Learned representations must derive from this state."""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any


class StateError(ValueError):
    pass


@dataclass(frozen=True)
class Event:
    version: int
    op: str
    entity_id: str
    record: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.version < 1 or not self.entity_id:
            raise StateError("event requires positive version and entity_id")
        if self.op not in {"upsert", "delete"}:
            raise StateError("op must be upsert or delete")
        if self.op == "upsert" and (not isinstance(self.record, dict) or not self.record):
            raise StateError("upsert requires nonempty record")
        if self.op == "delete" and self.record is not None:
            raise StateError("delete cannot carry a record")


@dataclass
class StateStore:
    version: int = 0
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def apply(self, event: Event) -> None:
        if event.version != self.version + 1:
            raise StateError(f"expected version {self.version + 1}, got {event.version}")
        if event.op == "delete":
            if event.entity_id not in self.records:
                raise StateError(f"cannot delete unknown entity {event.entity_id}")
            del self.records[event.entity_id]
        else:
            assert event.record is not None
            if "kind" not in event.record:
                raise StateError("record requires kind")
            self.records[event.entity_id] = deepcopy(event.record)
        self.version = event.version

    def snapshot(self) -> dict[str, Any]:
        return {"version": self.version, "records": {k: deepcopy(v) for k, v in sorted(self.records.items())}}

    @classmethod
    def replay(cls, events: list[Event]) -> StateStore:
        store = cls()
        for event in events:
            store.apply(event)
        return store
