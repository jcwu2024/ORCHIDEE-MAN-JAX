"""Finite cross-landpoint binding audit for the paper PFT14 mosaic.

This module validates configuration and asset routing only. It does not infer
scientific process behavior from annual outputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from jax_orchidee.driver.init import parse_run_def_bool
from jax_orchidee.sechiba.diffuco import pft14_trans_co2_parameter_inputs_from_run_def
from jax_orchidee.stomate.parameters import load_paper_case_stomate_parameter_bundle

from .reference_layout import PaperLandpointReference, inventory_paper_references
from .run_def_materialization import (
    YEAR_DYNAMIC_KEYS,
    materialize_case_run_def_values,
    read_run_def_values,
)


PAPER_POINT_VARYING_RUN_DEF_KEYS = (
    "LIMIT_WEST",
    "LIMIT_EAST",
    "LIMIT_SOUTH",
    "LIMIT_NORTH",
    "VCMAX25__00014",
    "ARJV__00014",
    "MAINT_RESP_SLOPE_C__00014",
    "ALLOC_MIN__00014",
    "RESIDENCE_TIME__00014",
)

PAPER_POINT_BINDING_OWNERS = {
    "LIMIT_WEST": "driver.orchestration._domain_override_from_run_def_values -> forcing domain",
    "LIMIT_EAST": "driver.orchestration._domain_override_from_run_def_values -> forcing domain",
    "LIMIT_SOUTH": "driver.orchestration._domain_override_from_run_def_values -> forcing domain",
    "LIMIT_NORTH": "driver.orchestration._domain_override_from_run_def_values -> forcing domain",
    "VCMAX25__00014": "stomate.parameters.PaperCaseStomateParameterBundle.vcmax25[13]",
    "ARJV__00014": "sechiba.diffuco.pft14_trans_co2_parameter_inputs_from_run_def['arjv']",
    "MAINT_RESP_SLOPE_C__00014": "stomate.parameters.PaperCaseStomateParameterBundle.maint_resp_slope[13,0]",
    "ALLOC_MIN__00014": "stomate.parameters.PaperCaseStomateParameterBundle.alloc_min[13]",
    "RESIDENCE_TIME__00014": "stomate.parameters.PaperCaseStomateParameterBundle.residence_time[13]",
}


@dataclass(frozen=True)
class PaperLandpointBindingAudit:
    landpoint_id: str
    passed: bool
    checks: Mapping[str, bool]
    values: Mapping[str, float]
    consumer_values: Mapping[str, float]
    package_signature_sha256: str
    errors: tuple[str, ...]

    def to_json_row(self) -> dict[str, object]:
        return {
            "landpoint_id": self.landpoint_id,
            "passed": self.passed,
            "checks": dict(self.checks),
            "values": dict(self.values),
            "consumer_values": dict(self.consumer_values),
            "package_signature_sha256": self.package_signature_sha256,
            "errors": list(self.errors),
        }


def expected_paper_domain_limits(landpoint_id: str) -> dict[str, float]:
    """Decode the one-degree lower-left index used by the paper job mosaic.

    Fortran protocol provenance: ``fortran_run_scripts/paper_250919/Job0_bio``
    materializes a two-degree forcing zoom. The paper IDs are one-based
    longitude/latitude indices whose lower-left coordinate is
    ``(-180 + i - 1, -90 + j - 1)``.
    """

    parts = landpoint_id.split("-")
    if len(parts) != 2:
        raise ValueError(f"invalid paper landpoint ID: {landpoint_id}")
    i_grid, j_grid = (int(float(part)) for part in parts)
    west = -180.0 + i_grid - 1
    south = -90.0 + j_grid - 1
    return {
        "LIMIT_WEST": west,
        "LIMIT_EAST": west + 2.0,
        "LIMIT_SOUTH": south,
        "LIMIT_NORTH": south + 2.0,
    }


def varying_case_run_def_keys(references: Iterable[PaperLandpointReference]) -> tuple[str, ...]:
    """Return all archived case keys whose values vary across landpoints."""

    values: dict[str, set[str]] = {}
    for reference in references:
        if reference.run_def is None:
            continue
        for key, value in read_run_def_values(reference.run_def).items():
            values.setdefault(key, set()).add(value.strip())
    return tuple(sorted(key for key, distinct in values.items() if len(distinct) > 1))


def _same_float(left: str | float, right: str | float) -> bool:
    return bool(np.float64(left) == np.float64(right))


def _same_run_def_value(left: str, right: str) -> bool:
    try:
        return _same_float(left, right)
    except ValueError:
        try:
            return parse_run_def_bool(left) == parse_run_def_bool(right)
        except ValueError:
            return left.strip() == right.strip()


def _asset_binding_checks(reference: PaperLandpointReference) -> dict[str, bool]:
    output = reference.output_dir.resolve() if reference.output_dir is not None else None

    def bound(path: Path | None) -> bool:
        return path is not None and path.exists() and output is not None and path.resolve().parent == output

    return {
        "output_package": output is not None and output.exists(),
        "driver_start_bound": bound(reference.driver_start),
        "sechiba_start_bound": bound(reference.sechiba_start),
        "stomate_start_bound": bound(reference.stomate_start),
        "driver_restart_bound": bound(reference.driver_restart),
        "sechiba_restart_bound": bound(reference.sechiba_restart),
        "stomate_restart_bound": bound(reference.stomate_restart),
        "run_def_bound": bound(reference.run_def),
        "used_run_def_bound": (
            reference.used_run_def is not None
            and reference.used_run_def.exists()
            and output is not None
            and reference.used_run_def.resolve().parent.parent == output
        ),
        "fifty_annual_histories": len(reference.histories) == 50,
        "annual_years_1961_2010": reference.years_available == tuple(range(1961, 2011)),
    }


def audit_paper_landpoint_binding(
    root: str | Path,
    reference: PaperLandpointReference,
) -> PaperLandpointBindingAudit:
    """Audit one package from archived Fortran inputs through JAX consumers."""

    root = Path(root)
    errors: list[str] = []
    checks = _asset_binding_checks(reference)
    if reference.run_def is None or reference.used_run_def is None:
        errors.append("missing run.def or z1/used_run.def")
        return PaperLandpointBindingAudit(reference.landpoint_id, False, checks, {}, {}, "", tuple(errors))

    case = read_run_def_values(reference.run_def)
    used = read_run_def_values(reference.used_run_def)
    materialized = materialize_case_run_def_values(
        base_used_run_def=reference.used_run_def,
        case_run_def=reference.run_def,
        base_is_fortran_used_truth=True,
    )
    # z1/used_run.def is the final Fortran getin truth. Case run.def entries
    # may add differently-cased submission-script keys, but must not mutate an
    # already materialized static key. Year-dependent restart/forcing values
    # and the local vegetation-file path are intentionally rebound elsewhere.
    preserved_keys = set(used) - set(YEAR_DYNAMIC_KEYS) - {"VEGETATION_FILE"}
    mutated_used_keys = sorted(
        key
        for key in preserved_keys
        if key not in materialized or not _same_run_def_value(used[key], materialized[key])
    )
    checks["fortran_used_static_values_preserved"] = not mutated_used_keys
    if mutated_used_keys:
        errors.append(f"materialization mutated Fortran used_run.def keys: {mutated_used_keys}")
    values: dict[str, float] = {}
    for key in PAPER_POINT_VARYING_RUN_DEF_KEYS:
        present = key in case and key in used and key in materialized
        checks[f"{key}:present"] = present
        if not present:
            errors.append(f"{key} missing from case, used, or materialized run.def")
            continue
        values[key] = float(case[key])
        checks[f"{key}:case_matches_fortran_used"] = _same_float(case[key], used[key])
        checks[f"{key}:materialized_matches_fortran_used"] = _same_float(materialized[key], used[key])

    expected_limits = expected_paper_domain_limits(reference.landpoint_id)
    for key, expected in expected_limits.items():
        checks[f"{key}:matches_landpoint_id"] = key in values and _same_float(values[key], expected)

    consumer_values: dict[str, float] = {}
    try:
        stomate = load_paper_case_stomate_parameter_bundle(
            root / "configs" / "orchidee_man_250919.yaml",
            used_run_def=reference.used_run_def,
        )
        consumer_values.update(
            {
                "VCMAX25__00014": float(stomate.vcmax25[13]),
                "MAINT_RESP_SLOPE_C__00014": float(stomate.maint_resp_slope[13, 0]),
                "ALLOC_MIN__00014": float(stomate.alloc_min[13]),
                "RESIDENCE_TIME__00014": float(stomate.residence_time[13]),
            }
        )
        diffuco = pft14_trans_co2_parameter_inputs_from_run_def(used)
        consumer_values["ARJV__00014"] = float(diffuco["arjv"])
    except Exception as exc:  # The audit must retain the failing package row.
        errors.append(f"consumer assembly failed: {type(exc).__name__}: {exc}")

    for key, actual in consumer_values.items():
        checks[f"{key}:consumer_matches_fortran_used"] = key in values and _same_float(actual, used[key])

    signature_payload = {
        "landpoint_id": reference.landpoint_id,
        "iteration_id": reference.iteration_id,
        "param_set": reference.param_set,
        "varying_values": {key: used.get(key) for key in PAPER_POINT_VARYING_RUN_DEF_KEYS},
        "asset_names": {
            name: None if path is None else path.name
            for name, path in (
                ("driver_start", reference.driver_start),
                ("sechiba_start", reference.sechiba_start),
                ("stomate_start", reference.stomate_start),
                ("driver_restart", reference.driver_restart),
                ("sechiba_restart", reference.sechiba_restart),
                ("stomate_restart", reference.stomate_restart),
            )
        },
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    failed_checks = [name for name, passed in checks.items() if not passed]
    errors.extend(f"failed check: {name}" for name in failed_checks)
    return PaperLandpointBindingAudit(
        landpoint_id=reference.landpoint_id,
        passed=not errors,
        checks=checks,
        values=values,
        consumer_values=consumer_values,
        package_signature_sha256=signature,
        errors=tuple(errors),
    )


def audit_all_paper_landpoint_bindings(root: str | Path) -> tuple[PaperLandpointBindingAudit, ...]:
    """Audit the finite set of locally archived paper packages."""

    references = inventory_paper_references(root)
    actual_varying = set(varying_case_run_def_keys(references))
    expected_varying = set(PAPER_POINT_VARYING_RUN_DEF_KEYS)
    if actual_varying != expected_varying:
        missing = sorted(expected_varying - actual_varying)
        unexpected = sorted(actual_varying - expected_varying)
        raise ValueError(f"paper varying-key contract drift: missing={missing}, unexpected={unexpected}")
    return tuple(audit_paper_landpoint_binding(root, reference) for reference in references)
