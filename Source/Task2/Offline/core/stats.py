"""Thống kê nhỏ dùng standard library để preflight không kéo NumPy."""

from __future__ import annotations

import math
from collections.abc import Iterable


def percentiles(values: Iterable[int], points: tuple[int, ...]) -> dict[str, int]:
    """Tính percentile tuyến tính tương thích cách audit cũ dùng NumPy."""

    ordered = sorted(values)
    if not ordered:
        return {}
    last = len(ordered) - 1
    result: dict[str, int] = {}
    for point in points:
        position = last * point / 100
        lower = math.floor(position)
        upper = math.ceil(position)
        value = float(ordered[lower])
        if upper != lower:
            value += (ordered[upper] - ordered[lower]) * (position - lower)
        result[f"p{point}"] = int(value)
    return result
