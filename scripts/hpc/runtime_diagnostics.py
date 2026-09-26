"""Host-side numeric summaries for runtime diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _tree_numeric_report(jax: Any, value: Any) -> dict[str, Any]:
    squared_norm = 0.0
    maximum = 0.0
    element_count = 0
    finite = True
    nonzero_leaf_count = 0
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        array = np.asarray(leaf, dtype=np.float64)
        finite = finite and bool(np.all(np.isfinite(array)))
        squared_norm += float(np.sum(array * array, dtype=np.float64))
        maximum = max(maximum, float(np.max(np.abs(array))) if array.size else 0.0)
        element_count += int(array.size)
        nonzero_leaf_count += int(bool(np.any(array != 0.0)))
    return {
        "all_finite": finite,
        "element_count": element_count,
        "leaf_count": len(leaves),
        "nonzero_leaf_count": nonzero_leaf_count,
        "l2_norm": float(np.sqrt(squared_norm)),
        "maximum_absolute_value": maximum,
    }
