"""Contracts for a bounded same-precision T7 initial-state JVP diagnostic."""
from __future__ import annotations

import numpy as np

STATE_WIDTH = 3854
SAMPLE_INDEX = 3
STATE_COORDINATES = (434, 2382, 2621, 200)
RTOL = 1e-5
ATOL = 1e-6
GPU_SECONDS_CAP = 3600
ROUTE_SECONDS_CAP = 1500
LOWER_SECONDS_CAP = 180
BACKEND_SECONDS_CAP = 900
LOAD_SECONDS_CAP = 300
JVP_CYCLES = 4


def direction_matrix(*, dtype=np.float64, batch_size=4, state_width=STATE_WIDTH,
                     sample_index=SAMPLE_INDEX, coordinates=STATE_COORDINATES):
    """One sample-3 unit direction per row, including the declared control."""
    if batch_size != 4 or state_width != STATE_WIDTH or sample_index != SAMPLE_INDEX:
        raise ValueError("fixed B4/sample3/full-state scope required")
    if tuple(coordinates) != STATE_COORDINATES:
        raise ValueError("the three discrepancy coordinates and control are fixed")
    if np.dtype(dtype) != np.dtype("float64"):
        raise ValueError("FP64 initial-state direction required")
    result = np.zeros((len(coordinates), batch_size, state_width), dtype=dtype)
    for row, coordinate in enumerate(coordinates):
        result[row, sample_index, coordinate] = 1
    return result


def validate_state_input_metadata(metadata, directions):
    """Validate the audited prepared-state identity without JAX or a model trace."""
    if (metadata.get("schema_version") != "saved_T7_exact_state_input_metadata_v1"
            or metadata.get("status") != "passed" or metadata.get("sample") != SAMPLE_INDEX
            or metadata.get("artifact_id") != 299 or metadata.get("window_index") != 0
            or metadata.get("day_index") != 0):
        raise ValueError("sample-3/case-299 exact prepared-state metadata required")
    identity = metadata.get("identities", {})
    if (identity.get("initial_state_dtype") != "float64"
            or len(identity.get("initial_state_sha256", "")) != 64
            or identity.get("candidate_config_matches_saved_identity") is not True):
        raise ValueError("FP64 saved sample-3 initial-state identity required")
    rows = {int(row.get("coordinate", -1)): row for row in metadata.get("rows", ())
            if row.get("sample") == SAMPLE_INDEX and row.get("artifact_id") == 299}
    if set(rows) != set(STATE_COORDINATES):
        raise ValueError("all discrepancy coordinates and control require audited input metadata")
    for row in rows.values():
        if row.get("canonical_defined_mask") is not True or row.get("continuous_tail_mask") is not False:
            raise ValueError("declared direction coordinate must be source-defined and outside the tail mask")
        if not np.isfinite(float(row["raw_initial_value"])):
            raise ValueError("prepared state coordinate must be finite")
    matrix = np.asarray(directions)
    if matrix.shape != (len(STATE_COORDINATES), 4, STATE_WIDTH) or matrix.dtype != np.dtype("float64"):
        raise ValueError("FP64 direction matrix/full-state B4 ABI differs")
    for index, coordinate in enumerate(STATE_COORDINATES):
        expected = np.zeros((4, STATE_WIDTH), dtype=np.float64)
        expected[SAMPLE_INDEX, coordinate] = 1.0
        if not np.array_equal(matrix[index], expected):
            raise ValueError("unit direction is not isolated to its declared sample/coordinate")
    return {"sample": SAMPLE_INDEX, "artifact_id": 299, "window_index": 0,
            "initial_state_dtype": "float64", "initial_state_sha256": identity["initial_state_sha256"],
            "state_width": STATE_WIDTH, "coordinates": list(STATE_COORDINATES),
            "source_defined_count": len(rows), "tail_masked_count": 0}


def compare_projections(b1_scaled, b4, *, rtol=RTOL, atol=ATOL):
    """Compare route JVPs and VJP projections with full failure records."""
    b1_scaled = np.asarray(b1_scaled)
    b4 = np.asarray(b4)
    if b1_scaled.shape != (len(STATE_COORDINATES),) or b4.shape != b1_scaled.shape:
        raise ValueError("four scalar projections per route required")
    if b1_scaled.dtype.kind != "f" or b4.dtype.kind != "f":
        raise ValueError("floating projections required")
    if not np.all(np.isfinite(b1_scaled)) or not np.all(np.isfinite(b4)):
        raise ValueError("finite projections required")
    b1_scaled, b4 = b1_scaled.astype(np.float64), b4.astype(np.float64)
    allowance = atol + rtol * np.abs(b4)
    difference = np.abs(b1_scaled - b4)
    return {
        "coordinates": list(STATE_COORDINATES),
        "B1_div4": b1_scaled.tolist(),
        "B4_mean": b4.tolist(),
        "absolute_difference": difference.tolist(),
        "allowance": allowance.tolist(),
        "failed_coordinates": [STATE_COORDINATES[i] for i in np.flatnonzero(difference > allowance)],
        "rtol": rtol,
        "atol": atol,
        "passed": bool(np.all(difference <= allowance)),
    }


def compare_scalar(actual, reference, *, coordinate, rtol=RTOL, atol=ATOL):
    actual, reference = float(actual), float(reference)
    allowance = atol + rtol * abs(reference)
    difference = abs(actual - reference)
    return {"coordinate": int(coordinate), "actual": actual, "reference": reference,
            "absolute_difference": difference, "allowance": allowance,
            "passed": bool(np.isfinite(actual) and np.isfinite(reference) and difference <= allowance),
            "rtol": rtol, "atol": atol}


def project_b1_state_vjp(state_gradient, *, sample_index=SAMPLE_INDEX,
                         coordinates=STATE_COORDINATES):
    """Project the selected B1 VJP sample, normalized to the B4 mean loss."""
    gradient = np.asarray(state_gradient)
    if gradient.shape != (STATE_WIDTH,) or gradient.dtype != np.dtype("float64"):
        raise ValueError("B1 full-state floating VJP required")
    if sample_index != SAMPLE_INDEX or tuple(coordinates) != STATE_COORDINATES:
        raise ValueError("fixed projection scope required")
    return np.asarray([gradient[index] / 4 for index in coordinates], dtype=gradient.dtype)


def project_b4_state_vjp(state_gradient, *, sample_index=SAMPLE_INDEX,
                         coordinates=STATE_COORDINATES):
    """Project sample 3 from the saved mean-objective B4 VJP."""
    gradient = np.asarray(state_gradient)
    if gradient.shape != (4, STATE_WIDTH) or gradient.dtype != np.dtype("float64"):
        raise ValueError("B4 full-state floating VJP required")
    if sample_index != SAMPLE_INDEX or tuple(coordinates) != STATE_COORDINATES:
        raise ValueError("fixed projection scope required")
    return np.asarray([gradient[sample_index, index] for index in coordinates], dtype=gradient.dtype)


def paid_pair_envelope_report(jvp_by_route, analysis):
    """Place JVPs against the saved paid B1/B4 repeat ranges, without gating."""
    records = analysis.get("coordinates")
    if not isinstance(records, list) or analysis.get("rtol") != RTOL or analysis.get("atol") != ATOL:
        raise ValueError("saved paid-pair analysis precision/coordinate records differ")
    selected = {int(row["coordinate"]): row for row in records
                if int(row.get("sample", -1)) == SAMPLE_INDEX
                and int(row.get("coordinate", -1)) in STATE_COORDINATES[:3]}
    if set(selected) != set(STATE_COORDINATES[:3]):
        raise ValueError("paid-pair envelopes must cover all three declared discrepancy coordinates")
    report = {}
    for route in ("B1", "B4"):
        values = np.asarray(jvp_by_route[route])
        if values.shape != (len(STATE_COORDINATES),) or not np.all(np.isfinite(values)):
            raise ValueError("four finite JVP values required per route")
        route_name = "reference" if route == "B1" else "actual"
        rows = {}
        for index, coordinate in enumerate(STATE_COORDINATES[:3]):
            envelope = selected[coordinate].get("same_session_paid_pairs_only", {}).get(route_name)
            bounds = (envelope.get("minimum"), envelope.get("maximum")) if isinstance(envelope, dict) else None
            if (not bounds or any(not isinstance(value, (int, float)) or not np.isfinite(value) for value in bounds)
                    or bounds[0] > bounds[1]):
                raise ValueError("complete saved paid-pair range required for " + route + ":" + str(coordinate))
            rows[str(coordinate)] = {"value": float(values[index]), "minimum": float(bounds[0]),
                "maximum": float(bounds[1]), "inside_observed_range": bool(bounds[0] <= values[index] <= bounds[1])}
        rows[str(STATE_COORDINATES[3])] = {"value": float(values[3]), "envelope_available": False,
            "reason": "predeclared control coordinate has no saved paid-pair envelope"}
        report[route] = rows
    return {"source_schema": analysis.get("schema_version"), "is_acceptance_gate": False, "routes": report}


def validate_occupation_plan(*, total_seconds, lower_seconds, backend_seconds,
                             route_seconds, load_seconds, calls, cycles=JVP_CYCLES,
                             compile_cap_per_route=1, route_mode="both"):
    """Reject plans exceeding any predeclared occupation or graph cap."""
    values = (total_seconds, lower_seconds, backend_seconds, route_seconds, load_seconds)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("positive integer occupation caps required")
    expected_calls = 16 if route_mode == "B4-only" else 32
    backend_cap = 1500 if route_mode == "B4-only" else (1800 if route_mode == "B1-completion" else BACKEND_SECONDS_CAP)
    if (route_mode not in ("both", "B4-only", "B1-completion") or total_seconds > GPU_SECONDS_CAP
            or lower_seconds > LOWER_SECONDS_CAP or backend_seconds > backend_cap
            or route_seconds > (1800 if route_mode == "B4-only" else (2400 if route_mode == "B1-completion" else ROUTE_SECONDS_CAP))
            or load_seconds > LOAD_SECONDS_CAP or calls != expected_calls
            or compile_cap_per_route != 1):
        raise ValueError("JVP occupation, call, cycle, or compile cap exceeded")
    if route_mode == "B1-completion" and backend_seconds > 1800:
        raise ValueError("B1 completion backend cap exceeded")
    route_count = 1 if route_mode in ("B4-only", "B1-completion") else 2
    load_count = 3 if route_mode == "B1-completion" else 2
    if route_count * route_seconds + load_count * load_seconds > total_seconds:
        raise ValueError("route and saved-load caps exceed total occupation")
    return {"total_seconds": total_seconds, "lower_seconds_per_route": lower_seconds,
            "backend_seconds_per_route": backend_seconds, "route_seconds_per_route": route_seconds,
            "load_seconds_per_saved_executable": load_seconds, "JVP_calls": calls,
            "cycles_per_route": cycles, "new_JVP_graph_compiles_per_route": compile_cap_per_route,
            "route_mode": route_mode,
            "saved_executable_load_count": load_count,
            "new_JVP_graph_compiles": {"B1": int(route_mode in ("both", "B1-completion")),
                                        "B4": int(route_mode in ("both", "B4-only"))}}
