"""Source-backed carbon ownership and label sufficiency for daily surrogates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CarbonFieldOwnership:
    field: str
    classification: str
    counts_in_carbon_inventory: bool
    successor_role: str
    provenance: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class OkLeakDriverSeries:
    field: str
    role: str
    provenance: tuple[str, ...]


OK_LEAK_CARBON_FIELD_OWNERSHIP = (
    CarbonFieldOwnership(
        "litter_above",
        "independent_stock",
        True,
        "exact_scan_carry",
        (
            "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90::"
            "littercalc_leak lines 1710, 2324, 2577-2664",
        ),
        "Aboveground structural and metabolic litter carbon.",
    ),
    CarbonFieldOwnership(
        "litter_below",
        "independent_stock",
        True,
        "exact_scan_carry",
        (
            "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90::"
            "littercalc_leak lines 1713, 2325, 2735-2776",
        ),
        "Vertically resolved structural and metabolic litter carbon.",
    ),
    CarbonFieldOwnership(
        "lignin_struc_above",
        "ratio_state",
        False,
        "exact_scan_writeback",
        (
            "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90::"
            "littercalc_leak lines 2353-2375",
        ),
        "Lignin fraction of structural litter, not an additional carbon pool.",
    ),
    CarbonFieldOwnership(
        "lignin_struc_below",
        "ratio_state",
        False,
        "exact_scan_writeback",
        (
            "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90::"
            "littercalc_leak lines 2377-2388",
        ),
        "Layer-wise lignin fraction, not an additional carbon pool.",
    ),
    CarbonFieldOwnership(
        "litterpart",
        "partition_state",
        False,
        "exact_scan_writeback",
        (
            "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90::"
            "littercalc_leak lines 2393-2405",
        ),
        "PFT contribution fractions for aboveground litter.",
    ),
    CarbonFieldOwnership(
        "dead_leaves",
        "overlapping_tracer",
        False,
        "exact_scan_writeback",
        (
            "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90::"
            "littercalc_leak lines 2138-2243, 2577-2725",
        ),
        "Leaf-litter tracer used for dead-leaf cover; it overlaps litter stock.",
    ),
    *(
        CarbonFieldOwnership(
            field,
            "overlapping_partition",
            False,
            "exact_scan_writeback",
            (
                "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90::"
                "littercalc_leak lines 2245-2317, 2625-2710",
            ),
            "Fuel-size partition of aboveground litter; do not count it twice.",
        )
        for field in ("fuel_1hr", "fuel_10hr", "fuel_100hr", "fuel_1000hr")
    ),
    CarbonFieldOwnership(
        "carbon_32l",
        "independent_stock",
        True,
        "exact_scan_carry",
        (
            "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::"
            "soilcarbon_leak lines 1348-1448, 1605-1785",
        ),
        "Three particulate soil-carbon pools across 32 layers.",
    ),
    CarbonFieldOwnership(
        "DOC",
        "independent_stock",
        True,
        "exact_scan_carry",
        (
            "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::"
            "soilcarbon_leak lines 1528-2303",
        ),
        "Free and adsorbed dissolved-organic-carbon pools.",
    ),
    CarbonFieldOwnership(
        "interception_storage",
        "independent_stock",
        True,
        "exact_scan_carry",
        (
            "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::"
            "soilcarbon_leak lines 1495-1525, 1240-1284, 2309-2349",
        ),
        "Canopy DOC storage; unweighted in the source mass-balance inventory.",
    ),
    CarbonFieldOwnership(
        "deepC_peat",
        "derived_diagnostic",
        False,
        "exact_scan_derived_writeback",
        (
            "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::"
            "soilcarbon_leak lines 1375-1437",
        ),
        "Reset from carbon_32l before decomposition and never an independent stock owner.",
    ),
)


OK_LEAK_DRIVER_SERIES = (
    OkLeakDriverSeries(
        "soil_mc",
        "litter controls, soil water, and 32-layer moisture",
        (
            "jax_orchidee/driver/orchestration.py::_paper_compiled_ok_leak_fold",
        ),
    ),
    OkLeakDriverSeries(
        "wat_flux",
        "vertical DOC water transport",
        (
            "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::"
            "soilcarbon_leak lines 1840-1968",
        ),
    ),
    OkLeakDriverSeries(
        "runoff_per_soil",
        "DOC runoff export",
        (
            "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::"
            "soilcarbon_leak lines 2052-2303",
        ),
    ),
    OkLeakDriverSeries(
        "drainage_per_soil",
        "DOC drainage export",
        (
            "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::"
            "soilcarbon_leak lines 2052-2303",
        ),
    ),
    OkLeakDriverSeries(
        "runoff2peat",
        "DOC transfer to peat",
        (
            "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::"
            "soilcarbon_leak lines 2052-2303",
        ),
    ),
    OkLeakDriverSeries(
        "canopy2ground",
        "canopy DOC drip",
        (
            "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90::"
            "soilcarbon_leak lines 1495-1525",
        ),
    ),
    OkLeakDriverSeries(
        "precip2ground",
        "TF-DOC ground deposition driver",
        (
            "jax_orchidee/stomate/soilcarbon_kernels.py::"
            "soilcarbon_leak_tf_doc_inputs",
        ),
    ),
    OkLeakDriverSeries(
        "precip2canopy",
        "TF-DOC canopy deposition driver",
        (
            "jax_orchidee/stomate/soilcarbon_kernels.py::"
            "soilcarbon_leak_tf_doc_inputs",
        ),
    ),
    OkLeakDriverSeries(
        "temp_sol",
        "aboveground litter temperature control",
        (
            "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90::"
            "littercalc_leak lines 1950-1968",
        ),
    ),
    OkLeakDriverSeries(
        "tdeep",
        "soil-carbon temperature and diffusion control",
        (
            "jax_orchidee/driver/orchestration.py::_paper_compiled_ok_leak_fold",
        ),
    ),
    OkLeakDriverSeries(
        "hsdeep",
        "soil-carbon moisture control",
        (
            "jax_orchidee/driver/orchestration.py::_paper_compiled_ok_leak_fold",
        ),
    ),
    OkLeakDriverSeries(
        "shumdiag_peat",
        "PERMA_PEAT moisture extension",
        (
            "jax_orchidee/driver/orchestration.py::_paper_compiled_ok_leak_fold",
        ),
    ),
    OkLeakDriverSeries(
        "resp_maint_part_radia",
        "flooded-root respiration contribution",
        (
            "fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3415-3458",
        ),
    ),
)


AGGREGATE_TRANSFER_LABELS = (
    "litter_input",
    "litter_respiration",
    "litter_to_doc",
    "poc_respiration",
    "poc_to_doc",
    "doc_to_poc",
    "doc_respiration",
    "doc_external_input",
    "doc_export",
)


def _validate_ownership_table() -> None:
    fields = tuple(item.field for item in OK_LEAK_CARBON_FIELD_OWNERSHIP)
    if len(fields) != len(set(fields)):
        raise RuntimeError("duplicate OK_LEAK carbon ownership field")
    expected = {
        "litter_above",
        "litter_below",
        "lignin_struc_above",
        "lignin_struc_below",
        "litterpart",
        "dead_leaves",
        "fuel_1hr",
        "fuel_10hr",
        "fuel_100hr",
        "fuel_1000hr",
        "carbon_32l",
        "DOC",
        "interception_storage",
        "deepC_peat",
    }
    if set(fields) != expected:
        raise RuntimeError("OK_LEAK carbon ownership table is incomplete")


_validate_ownership_table()


def extract_ok_leak_driver_series(record: Any) -> dict[str, np.ndarray]:
    """Extract the 48 source-order driver arrays needed by the exact scan."""

    transition = record.half_hour_transition
    stacks = transition.compiled_entry_stacks
    if stacks is None:
        raise ValueError("OK_LEAK driver capture requires compiled entry stacks")
    missing = tuple(
        item.field
        for item in OK_LEAK_DRIVER_SERIES
        if item.field != "resp_maint_part_radia" and item.field not in stacks
    )
    if missing:
        raise ValueError(f"compiled entry stacks are missing OK_LEAK drivers: {missing}")
    maintenance = getattr(record, "maintenance_resp_parts", None)
    if maintenance is None:
        raise ValueError("OK_LEAK driver capture requires maintenance_resp_parts")
    arrays = {
        item.field: np.asarray(
            maintenance if item.field == "resp_maint_part_radia" else stacks[item.field]
        )
        for item in OK_LEAK_DRIVER_SERIES
    }
    step_counts = {int(value.shape[0]) for value in arrays.values()}
    if step_counts != {48}:
        raise ValueError(f"OK_LEAK driver series must contain 48 steps, got {sorted(step_counts)}")
    return arrays


def ok_leak_driver_capture_metadata(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Return a deterministic schema and storage estimate for one captured day."""

    expected = tuple(item.field for item in OK_LEAK_DRIVER_SERIES)
    if tuple(arrays) != expected:
        raise ValueError("OK_LEAK driver arrays must follow the source-order schema")
    fields = []
    total_bytes = 0
    for item in OK_LEAK_DRIVER_SERIES:
        value = np.asarray(arrays[item.field])
        total_bytes += int(value.nbytes)
        fields.append(
            {
                "field": item.field,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": int(value.nbytes),
                "role": item.role,
                "provenance": list(item.provenance),
            }
        )
    return {
        "schema_version": "ok_leak_driver_capture_v1",
        "step_count": 48,
        "fields": fields,
        "uncompressed_bytes_per_day": total_bytes,
    }


def audit_carbon_budget_contract(
    contract_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit whether a Markov contract can train a conservative successor."""

    leaves = tuple(contract_metadata["fast_day_target_leaves"])
    ok_leaves = {
        str(item["path"][0]): item
        for item in leaves
        if str(item["family"]) == "ok_leak"
    }
    expected_fields = tuple(item.field for item in OK_LEAK_CARBON_FIELD_OWNERSHIP)
    missing_fields = tuple(field for field in expected_fields if field not in ok_leaves)
    extra_fields = tuple(sorted(set(ok_leaves) - set(expected_fields)))
    widths = {
        field: int(item["stop"]) - int(item["start"])
        for field, item in ok_leaves.items()
    }

    diagnostics = {
        str(item["name"])
        for item in contract_metadata.get("diagnostic_leaves", ())
    }
    target_keys = {
        f"{str(item['family'])}.{str(item['path'][0])}"
        for item in leaves
    }
    exact_scan_labels = {
        item.field: (
            f"ok_leak_driver.{item.field}" in target_keys
            or f"ok_leak_driver.{item.field}" in diagnostics
        )
        for item in OK_LEAK_DRIVER_SERIES
    }
    aggregate_labels = {
        name: (
            f"carbon_transfer.{name}" in target_keys
            or f"carbon_transfer.{name}" in diagnostics
        )
        for name in AGGREGATE_TRANSFER_LABELS
    }

    ownership = []
    for item in OK_LEAK_CARBON_FIELD_OWNERSHIP:
        ownership.append(
            {
                **asdict(item),
                "target_width": widths.get(item.field, 0),
                "present": item.field in ok_leaves,
            }
        )
    inventory_width = sum(
        widths.get(item.field, 0)
        for item in OK_LEAK_CARBON_FIELD_OWNERSHIP
        if item.counts_in_carbon_inventory
    )
    derived_width = sum(
        widths.get(item.field, 0)
        for item in OK_LEAK_CARBON_FIELD_OWNERSHIP
        if item.classification == "derived_diagnostic"
    )
    overlap_width = sum(widths.values()) - inventory_width - derived_width

    exact_scan_ready = bool(exact_scan_labels) and all(exact_scan_labels.values())
    aggregate_ready = bool(aggregate_labels) and all(aggregate_labels.values())
    return {
        "schema_version": "carbon_budget_contract_audit_v1",
        "contract_schema_version": str(contract_metadata["schema_version"]),
        "fast_day_target_width": int(contract_metadata["fast_day_target_width"]),
        "ok_leak_target_width": sum(widths.values()),
        "independent_inventory_target_width": inventory_width,
        "derived_diagnostic_target_width": derived_width,
        "overlapping_or_ratio_target_width": overlap_width,
        "missing_ok_leak_fields": list(missing_fields),
        "extra_ok_leak_fields": list(extra_fields),
        "ownership": ownership,
        "exact_scan_driver_labels": exact_scan_labels,
        "aggregate_transfer_labels": aggregate_labels,
        "capabilities": {
            "endpoint_stock_supervision": not missing_fields,
            "exact_scan_driver_supervision": exact_scan_ready,
            "aggregate_transfer_supervision": aggregate_ready,
        },
        "decision": (
            "exact_scan_successor_ready"
            if exact_scan_ready
            else "capture_bounded_ok_leak_driver_auxiliary_dataset"
        ),
        "forbidden": (
            "independent_deepC_peat_prediction",
            "independent_unconstrained_stock_residual",
            "paid_training_before_conservative_micro_gate",
        ),
    }
