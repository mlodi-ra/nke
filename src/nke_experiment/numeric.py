"""Schema-neutral numeric fields for the experimental typed input channel."""

from __future__ import annotations

import hashlib
import math
from typing import Any


def numeric_fields(record: dict[str, Any], buckets: int = 1024) -> list[tuple[int, float]]:
    fields = []
    for key, value in sorted(record.items()):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite numeric field: {key}")
        index = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % buckets
        fields.append((index, number))
    return fields
