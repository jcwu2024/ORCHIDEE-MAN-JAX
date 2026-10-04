"""Conservative independent single-landpoint batching of complete day blocks.

No process equations live here. The transition uses the source-backed scalar
driver chain (see orchestration._paper_later_day_block_transition_factory).
The outer lane axis does not change the scalar driver's inner npts=1 layout.
This research adapter has no persistent executable cache or production CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, NamedTuple

import jax
import numpy as np
import yaml

from . import orchestration as driver
from .fast_state import fast_state_from_previous_packet


class BlockArguments(NamedTuple):
    values_by_component: Any
    forcing_days: Any
    science_day_numbers: Any
    stomate_parameters: Any
    hydrol_table_arrays: Any
    landpoint_payload: Any
    stomate_restart_template: Any
    stomate_season_values: Any
    diffuco_parameter_values: Any


@dataclass(frozen=True)
class PreparedLane:
    landpoint_id: str
    context: Any
    initial_state: Any
    tables: Any
    dispatch: Any
    arguments: BlockArguments
    static_signature: Any


# These four vectors are explicitly substituted throughout the scalar block.
# Source: orchestration._compiled_stomate_parameter_values and the overrides
# in paper_1961_driver_later_day_runtime_result; parameters.py:812-852.
_STOMATE_DYNAMIC = frozenset({"alloc_min", "residence_time", "vcmax25", "maint_resp_slope"})
_DYNAMIC_RUN_KEYS = frozenset(
    {"ALLOC_MIN", "RESIDENCE_TIME", "VCMAX25", "MAINT_RESP_SLOPE_A", "MAINT_RESP_SLOPE_B", "MAINT_RESP_SLOPE_C"}
)
_DOMAIN_RUN_KEYS = frozenset({"LIMIT_WEST", "LIMIT_EAST", "LIMIT_NORTH", "LIMIT_SOUTH"})
_METADATA = frozenset({"provenance", "source_paths"})


def _items(value):
    if is_dataclass(value):
        return {f.name: getattr(value, f.name) for f in fields(value)}
    if hasattr(value, "_asdict"):
        return value._asdict()
    if isinstance(value, dict):
        return value
    return None


def _freeze(value):
    """Exact, dtype-aware static value identity; unknown objects are rejected."""
    if value is None:
        return ("none",)
    members = _items(value)
    if members is not None:
        return tuple((k, _freeze(v)) for k, v in sorted(members.items()) if k not in _METADATA)
    if isinstance(value, Path):
        return ("path", str(value.resolve()))
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_freeze(v) for v in value))
    array = np.asarray(value)
    if array.dtype.kind not in "biufcUS":
        raise ValueError(f"unsupported static value {type(value).__name__}")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError("nonfinite static scientific value")
    return ("array", array.shape, array.dtype.str, sha256(array.tobytes()).hexdigest())


def static_signature(context, initial_state, tables, dispatch):
    """Audit scientific closures, not config filenames or PFT/shape alone.

    All context fields default to exact comparison. Only audited dynamic or
    host-reader fields are excluded below. Unknown run.def changes remain
    incompatible, even if they might only affect unused branches. First-step
    domain/forcing/restart selection is host work; the numeric payload and
    STOMATE templates are supplied independently for every lane. The daily
    compiled path consumes no other first-step restart component.
    """
    omitted = {
        "config_path",
        "run_def_path",
        "reference_run_dir",
        "run_def_values",
        "domain_override",
        "run_scalars",
        "stomate_static",
        "diffuco_pft14_static_kwargs",
        "first_step_bundle",
        "first_step_restart_state",
        "first_step_stomate_boundary",
        "provenance",
    }
    values = {f.name: getattr(context, f.name) for f in fields(context) if f.name not in omitted}
    values["config_content"] = sha256(context.config_path.read_bytes()).hexdigest()
    # Existing point-varying ARJV enters the ninth dynamic argument's arjv;
    # diffuco.py:1989 and paper_binding.py:PAPER_POINT_BINDING_OWNERS. Do not
    # exclude arbitrary similarly named parameters for other PFT identities.
    mangrove_id = driver._paper_mangrove_fortran_pft_id(context.run_scalars)
    dynamic_diffuco_run_keys = {f"ARJV__{mangrove_id:05d}"}
    values["run_def"] = {
        k: v
        for k, v in context.run_def_values.items()
        if k not in _DOMAIN_RUN_KEYS
        and k not in dynamic_diffuco_run_keys
        and k.split("__", 1)[0] not in _DYNAMIC_RUN_KEYS
    }
    scalars = _items(context.run_scalars)
    values["run_scalars"] = {k: v for k, v in scalars.items() if k != "raw_run_def"}
    values["stomate_parameters"] = {
        k: v for k, v in _items(context.stomate_static.parameters).items() if k not in _STOMATE_DYNAMIC
    }
    values["stomate_kwargs"] = {k: v for k, v in context.stomate_static.kwargs.items() if k not in _STOMATE_DYNAMIC}
    values["diffuco_kwargs"] = {
        k: v
        for k, v in context.diffuco_pft14_static_kwargs.items()
        if k not in {"trans_co2_pft_params", "rstruct_const"}
    }
    # ok_laidev is static dispatch; all other trans_co2 values are dynamic.
    values["diffuco_trans_ok_laidev"] = context.diffuco_pft14_static_kwargs["trans_co2_pft_params"].get("ok_laidev")
    spec = fast_state_from_previous_packet(initial_state).spec
    values["state_layout"] = (spec.components, spec.field_names_by_component)
    values["mineral_bounds"] = (tables.mineral.imin, tables.mineral.imax)
    values["dispatch"] = dispatch
    values["npts"] = context.first_step_bundle.domain.nbindex
    return _freeze(values)


def prepare_lane(landpoint_id, context, initial_state, *, year=1961, day_numbers=range(2, 9)):
    """Prepare all nine scalar arguments from this lane's own inputs."""
    days = np.asarray(tuple(day_numbers), dtype=np.int32)
    steps = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    if steps != 48 or context.first_step_bundle.domain.nbindex != 1:
        raise ValueError("batch requires 48 half-hours and inner npts=1")
    if len(days) != 7 or not np.array_equal(days, np.arange(days[0], days[0] + 7)):
        raise ValueError("prototype requires seven consecutive science days")
    if initial_state.tstep != int(days[0] - 1) * steps - 1:
        raise ValueError("initial state is not aligned with first science day")
    declared = yaml.safe_load(context.config_path.read_text(encoding="utf-8"))
    if declared.get("paper_mosaic", {}).get("execution_unit") != "independent_single_landpoint":
        raise ValueError("batch requires declared independent single-landpoint contract")
    # RIVER_ROUTING may be TRUE in the paper run. With nbp_glo=1 the source
    # follows its explicit no-routing branch; do not change that input flag.
    # Fortran: sechiba.f90::sechiba_main 1227-1240; stomate/entry.py's source-backed
    # stomate_no_routing_entry_source carries the same nbp_glo condition.
    if context.hydrol_soil_peat_hydro or context.topmodel_new:
        raise ValueError("unsupported compiled/independent-landpoint branch")
    inputs = driver._paper_1961_later_day_transition_inputs(
        context=context,
        current_state=initial_state,
        year=year,
        start=int(days[0] - 1) * steps,
        steps_per_stomate=steps,
        fixed_format_trace_dir=None,
        static_trace_fields=None,
        prebuild_day_payloads=True,
    )
    if (
        inputs.hydrol_runtime_static_tables is None
        or inputs.diffuco_day_static_cache is None
        or inputs.hydrol_static_template is None
    ):
        raise ValueError("compiled day admission prerequisites missing")
    tables = inputs.hydrol_runtime_static_tables
    if tables.peat is not None:
        raise ValueError("peat tables unsupported by complete-day block")
    forcing = [
        driver._paper_compiled_forcing_day(
            context,
            year=year,
            start_tstep=int(d - 1) * steps,
            steps_per_stomate=steps,
        )
        for d in days
    ]
    arguments = BlockArguments(
        fast_state_from_previous_packet(initial_state).values_by_component,
        jax.tree_util.tree_map(lambda *v: np.stack(v), *forcing),
        days,
        driver._compiled_stomate_parameter_values(context),
        driver._compiled_hydrol_table_arrays(tables),
        driver._compiled_landpoint_payload(inputs.compiled_base_payload_template),
        context.first_step_restart_state.stomate,
        {
            k: v
            for k, v in context.first_step_restart_state.stomate_readstart.season_state._asdict().items()
            if k != "provenance"
        },
        driver._compiled_diffuco_parameter_values(context),
    )
    dispatch = driver._paper_daily_carbon_static_dispatch(context, initial_state)
    return PreparedLane(
        landpoint_id,
        context,
        initial_state,
        tables,
        dispatch,
        arguments,
        static_signature(context, initial_state, tables, dispatch),
    )


def _schema(tree):
    # None belongs to the PyTree structure, never to a broadcast leaf.
    paths, structure = jax.tree_util.tree_flatten_with_path(tree)
    result = []
    for path, value in paths:
        a = np.asarray(value)
        if a.dtype.kind not in "biufc":
            raise ValueError(f"nonnumeric dynamic input at {jax.tree_util.keystr(path)}")
        result.append((jax.tree_util.keystr(path), a.shape, a.dtype.str))
    return structure, tuple(result)


def audit_compatible_lanes(lanes):
    if not lanes:
        raise ValueError("empty batch")
    first = lanes[0]
    schema = _schema(first.arguments)
    for lane in lanes[1:]:
        if lane.static_signature != first.static_signature:
            paths = _static_difference_paths(first.static_signature, lane.static_signature)
            raise ValueError(f"incompatible scientific closure: {lane.landpoint_id}: {paths[:20]}")
        if _schema(lane.arguments) != schema:
            raise ValueError(f"schema/dtype/None mismatch: {lane.landpoint_id}")
        if not np.array_equal(lane.arguments.science_day_numbers, first.arguments.science_day_numbers):
            raise ValueError(f"science day mismatch: {lane.landpoint_id}")
    return {
        "passed": True,
        "lanes": [v.landpoint_id for v in lanes],
        "dynamic_leaf_count": len(schema[1]),
        "static_digest": sha256(repr(first.static_signature).encode()).hexdigest(),
        "shared_science_days": first.arguments.science_day_numbers.tolist(),
        "inner_npts": 1,
        "block_size": 7,
        "dynamic_consumer_audit": {
            "four_stomate_vectors": "orchestration._compiled_stomate_parameter_values; stomate/parameters.py:812-852",
            "selected_pft_ARJV": "sechiba/diffuco.py:1989 -> trans_co2.arjv -> orchestration._compiled_diffuco_parameter_values",
            "domain_limits": "host forcing/domain selection; all lane numeric payloads and forcings independently stacked",
            "unknown_run_def_differences": "rejected",
            "RIVER_ROUTING": "unchanged static flag; nbp_glo=1 explicit no-routing source branch sechiba.f90:1227-1240",
        },
        "lane_metadata": [
            {
                "landpoint_id": lane.landpoint_id,
                "state_provenance": dict(
                    zip(
                        fast_state_from_previous_packet(lane.initial_state).spec.components,
                        fast_state_from_previous_packet(lane.initial_state).spec.provenance_by_component,
                        strict=True,
                    )
                ),
                "season_provenance": lane.context.first_step_restart_state.stomate_readstart.season_state.provenance,
            }
            for lane in lanes
            if lane.context is not None
        ],
    }


def _static_difference_paths(left, right, path="static"):
    if left == right:
        return []
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        mapping = all(isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str) for v in left + right)
        if mapping and tuple(v[0] for v in left) == tuple(v[0] for v in right):
            return [
                p
                for (key, a), (_, b) in zip(left, right, strict=True)
                for p in _static_difference_paths(a, b, path + "." + key)
            ]
        return [
            p
            for i, (a, b) in enumerate(zip(left, right, strict=True))
            for p in _static_difference_paths(a, b, path + f"[{i}]")
        ]
    return [path]


def stack_lanes(lanes):
    audit_compatible_lanes(lanes)
    result = []
    for i in range(9):
        result.append(
            lanes[0].arguments[i]
            if i == 2
            else jax.tree_util.tree_map(lambda *v: np.stack(v), *(x.arguments[i] for x in lanes))
        )
    return BlockArguments(*result)


def make_scalar_transition(lane, **capture_kwargs):
    return driver._paper_later_day_block_transition_factory(
        lane.context.config_path,
        context=lane.context,
        initial_state=lane.initial_state,
        prebound_hydrol_runtime_static_tables=lane.tables,
        daily_carbon_dispatch=lane.dispatch,
        **capture_kwargs,
    )[0]


def make_batch_transition(lanes, **capture_kwargs):
    """Transform the pure function before AOT; all numeric lane inputs vary."""
    audit_compatible_lanes(lanes)
    return jax.vmap(make_scalar_transition(lanes[0], **capture_kwargs), in_axes=(0, 0, None, 0, 0, 0, 0, 0, 0))
