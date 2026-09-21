"""Host-only attribution of the unchanged daily state anchor; no new loss."""
from __future__ import annotations

import numpy as np

from research.daily_coarse_graining.constrained_daily_replay import (
    CARBON_INVENTORY_FIELDS,
    THERMAL_FIELDS,
    WATER_INVENTORY_FIELDS,
)


def state_coordinate_owners(contract):
    """Mirror the declared group packing, retaining canonical flat coordinates."""
    leaves = {
        (leaf.component, leaf.path[0]): leaf
        for leaf in contract.state_leaves
        if len(leaf.path) == 1 and leaf.start is not None and leaf.stop is not None
    }

    def field(component, name, full=False):
        leaf = leaves[component, name]
        shape = leaf.full_shape if full and leaf.selected_pft_indices else leaf.shape
        return [(leaf.key, tuple(int(v) for v in index)) for index in np.ndindex(tuple(shape))]

    hydrol = "hydrol_previous_step_state"
    stomate = "slowproc_stomate_previous_step_state"
    other = [None] * int(contract.continuous_state_width)
    for leaf in contract.state_leaves:
        if leaf.start is not None and leaf.stop is not None:
            other[int(leaf.start):int(leaf.stop)] = [
                (leaf.key, tuple(int(v) for v in index)) for index in np.ndindex(tuple(leaf.shape))
            ]
    water = []
    for name in WATER_INVENTORY_FIELDS:
        entries = field(hydrol, name)
        water.extend(entries)
        if name == "qsintveg":
            water.extend(entries)
    return {
        "canopy": field(hydrol, "qsintveg"),
        "soil": field(hydrol, "mc"),
        "snow": sum((field(hydrol, name) for name in ("snow", "snow_nobio", "snowliq")), []),
        "water": water,
        "carbon": sum((field(stomate, name) for name in CARBON_INVENTORY_FIELDS), []),
        "thermal": sum((field(component, name, full=True) for component, name, _ in THERMAL_FIELDS
                        if (component, name) in leaves), []),
        "other": other,
    }


def summarize_state_anchor(payload, scales, owners, *, top_count=12):
    """Resolve MSE and signed bias contributions, including cross-field bias.

    Bias allocations may be negative: e_i * sum(e) / N**2 sums to the
    squared mean error. Squaring each field's bias separately would be wrong.
    """
    groups = {}
    for name in payload["predicted"]._fields:
        pred = np.asarray(getattr(payload["predicted"], name), dtype=np.float64).reshape(-1)
        expected = np.asarray(getattr(payload["expected"], name), dtype=np.float64).reshape(-1)
        scale = np.asarray(getattr(scales.state, name), dtype=np.float64).reshape(-1)
        mask = (np.asarray(getattr(payload["predicted_defined"], name), dtype=bool).reshape(-1)
                & np.asarray(getattr(payload["expected_defined"], name), dtype=bool).reshape(-1)
                & np.asarray(getattr(scales.state_loss_eligible, name), dtype=bool).reshape(-1))
        mask &= np.isfinite(pred) & np.isfinite(expected) & np.isfinite(scale) & (scale > 0.0)
        if len(owners[name]) != pred.size:
            raise ValueError(f"state diagnostic owner width drift: {name}")
        error = (np.where(mask, pred, 0.0) - np.where(mask, expected, 0.0)) / np.where(mask, scale, 1.0)
        count = int(np.count_nonzero(mask))
        divisor = max(count, 1)
        mse_parts = error**2 / divisor
        bias_parts = error * np.sum(error) / divisor**2
        fields = {}
        for index in np.flatnonzero(mask):
            owner = owners[name][index]
            if owner is None:
                raise ValueError(f"missing state diagnostic owner: {name}[{index}]")
            entry = fields.setdefault(owner[0], dict(count=0, mse_contribution=0.0,
                                                     bias_squared_contribution=0.0))
            entry["count"] += 1
            entry["mse_contribution"] += float(mse_parts[index])
            entry["bias_squared_contribution"] += float(bias_parts[index])
        for entry in fields.values():
            entry["loss_contribution"] = 0.5 * (entry["mse_contribution"]
                                               + entry["bias_squared_contribution"])
        order = sorted(np.flatnonzero(mask), key=lambda i: -abs(float(error[i])))[:top_count]
        coordinates = [dict(group_index=int(i), field=owners[name][i][0],
                            coordinate=owners[name][i][1], predicted=float(pred[i]),
                            expected=float(expected[i]), scale=float(scale[i]),
                            normalized_error=float(error[i])) for i in order]
        mse, bias = float(np.sum(mse_parts)), float(np.sum(bias_parts))
        groups[name] = dict(count=count, mse=mse, bias_squared=bias, loss=0.5 * (mse + bias),
                            fields=sorted((dict(field=k, **v) for k, v in fields.items()),
                                          key=lambda row: -abs(row["loss_contribution"])),
                            largest_coordinates=coordinates)
    active = [group["loss"] for group in groups.values() if group["count"]]
    return dict(total=float(np.mean(active)) if active else 0.0, groups=groups)
