"""Explicit answerability semantics shared by training and inference."""

from __future__ import annotations


def candidate_names(row: dict) -> list[str]:
    candidates = list(row["candidates"])
    if row.get("allow_none", True):  # Legacy v2 data always included 'none'.
        if "none" in candidates:
            raise ValueError("none must not be repeated in candidates")
        candidates.append("none")
    return candidates
