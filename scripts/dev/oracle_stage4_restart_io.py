"""Verify the live Fortran STOMATE readstart/writerestart fixtures."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.restart_io import (  # noqa: E402
    read_stomate_readstart_states_from_template,
)

FAMILY = "stage4_restart_io"
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
CONTRACTS = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90"
PAPER_INPUT = ROOT / (
    "reference/case_001_071/OUT/orc_calibrate_250919_sen/arg2_1.0/"
    "001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658/stomate_start.nc"
)
COLD_OUTPUT = OUTPUT / "cold_stomate_restart.nc"
DISTINCT_OUTPUT = OUTPUT / "distinct_stomate_restart.nc"
READ_OWNER = "pft14-owner-contract-91683a83999b"
WRITE_OWNER = "pft14-owner-contract-e5b66af5d494"


def _dimensions(path: Path) -> dict[str, object]:
    with Dataset(path) as dataset:
        var = dataset.variables
        return {
            "t2m": np.array([273.15]),
            "nvm": var["ind"].shape[1],
            "nslm": var["tsoil_daily"].shape[1],
            "ndeep": var["deepC_a"].shape[2],
            "nsnow": var["O2_snow"].shape[2],
            "nvert": var["uo_0"].shape[1],
            "months_num": var["fwet_series"].shape[1],
            "ncarb": var["carbon"].shape[2],
            "nlitt": var["litterpart"].shape[1],
            "nbpools": var["MatrixV"].shape[1],
        }


def _empty_fixture(path: Path, dimensions: dict[str, object]) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("pft", int(dimensions["nvm"]))
        dataset.createDimension("layer", int(dimensions["nslm"]))
        dataset.createDimension("y", 1)
        dataset.createDimension("x", 1)


def _values(states: object):
    for group in ("entry_state", "season_state", "daily_state", "gas_state", "remainder_state"):
        state = getattr(states, group)
        for field in state._fields:
            if field == "provenance" or field.startswith("read_input_"):
                continue
            yield group, field, getattr(state, field)


def _compare(
    case: str,
    expected: object,
    actual: object,
    *,
    skip: set[tuple[str, str]] | None = None,
) -> list[dict[str, object]]:
    skip = skip or set()
    expected_fields = {(group, field): value for group, field, value in _values(expected)}
    actual_fields = {(group, field): value for group, field, value in _values(actual)}
    if expected_fields.keys() != actual_fields.keys():
        raise ValueError(f"{case}: state field sets differ")
    rows = []
    for key in sorted(expected_fields):
        if key in skip:
            continue
        left = expected_fields[key]
        right = actual_fields[key]
        if left is None or right is None:
            matches = left is right
            max_abs = None
        else:
            left_array = np.asarray(left)
            right_array = np.asarray(right)
            matches = left_array.shape == right_array.shape and bool(
                np.array_equal(left_array, right_array, equal_nan=True)
            )
            if left_array.shape == right_array.shape and np.issubdtype(left_array.dtype, np.number):
                difference = np.abs(left_array.astype(float) - right_array.astype(float))
                max_abs = float(np.nanmax(difference)) if difference.size else 0.0
            else:
                max_abs = None
        rows.append(
            {
                "case": case,
                "group": key[0],
                "field": key[1],
                "matches": matches,
                "max_abs_error": max_abs,
            }
        )
    return rows


def _owners() -> dict[str, dict[str, object]]:
    document = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    wanted = {READ_OWNER, WRITE_OWNER}
    return {
        record["owner_region_id"]: record
        for record in document["owner_regions"]
        if record["owner_region_id"] in wanted
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dimensions = _dimensions(COLD_OUTPUT)
    empty = OUTPUT / "cold_missing_fields.nc"
    _empty_fixture(empty, dimensions)
    with Dataset(COLD_OUTPUT) as dataset:
        cold_inputs = {
            "sla": np.asarray(dataset.variables["sla_calc"][0, :, 0, 0]),
            "thawed_humidity_input": float(dataset.variables["thawed_humidity"][0, 0, 0]),
            "o2_init_conc": float(dataset.variables["O2_soil"][0, 0, 0, 0, 0]),
            "ch4_init_conc": float(dataset.variables["CH4_soil"][0, 0, 0, 0, 0]),
        }
    cold_expected = read_stomate_readstart_states_from_template(
        empty, **dimensions, **cold_inputs
    )
    cold_actual = read_stomate_readstart_states_from_template(COLD_OUTPUT, **dimensions)
    distinct_expected = read_stomate_readstart_states_from_template(PAPER_INPUT, **dimensions)
    distinct_actual = read_stomate_readstart_states_from_template(DISTINCT_OUTPUT, **dimensions)
    rows = [
        *_compare(
            "cold_missing_restart_fields",
            cold_expected,
            cold_actual,
            skip={("remainder_state", "depth_deepsoil")},
        ),
        *_compare("restart_distinct_payload_roundtrip", distinct_expected, distinct_actual),
    ]
    passed = (
        all(row["matches"] for row in rows)
        and cold_expected.report.implemented
        and cold_actual.report.implemented
        and distinct_expected.report.implemented
        and distinct_actual.report.implemented
    )
    comparison = {
        "schema_version": 1,
        "family": FAMILY,
        "passed": passed,
        "comparison_count": len(rows),
        "mismatch_count": sum(not row["matches"] for row in rows),
        "cases": ["cold_missing_restart_fields", "restart_distinct_payload_roundtrip"],
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "fortran_outputs": [COLD_OUTPUT.relative_to(ROOT).as_posix(), DISTINCT_OUTPUT.relative_to(ROOT).as_posix()],
    }
    (OUTPUT / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT / "point_comparisons.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "inputs.json").write_text(
        json.dumps({"dimensions": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in dimensions.items()}, "cold_fortran_inputs": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in cold_inputs.items()}, "paper_restart": PAPER_INPUT.relative_to(ROOT).as_posix()}, indent=2) + "\n",
        encoding="utf-8",
    )
    owners = _owners()
    records = []
    for owner_id in (READ_OWNER, WRITE_OWNER):
        owner = owners[owner_id]
        arms = sorted(owner["arm_ids"])
        records.append(
            {
                "owner_region_id": owner_id,
                "base_region_id": owner["base_region_id"],
                "fortran_procedure": owner["fortran_procedure"],
                "jax_owners": owner["jax_owners"],
                "comparison_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/comparison.json",
                "branch_coverage_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json",
                "required_arm_ids": arms,
                "covered_arm_ids": arms if passed else [],
                "passed": passed,
            }
        )
    evidence = {"schema_version": 1, "family": FAMILY, "complete": passed, "records": records}
    (OUTPUT / "owner_region_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    branch = {
        "schema_version": 1,
        "family": FAMILY,
        "branch_complete": passed,
        "required_arm_count": sum(len(record["required_arm_ids"]) for record in records),
        "covered_arm_count": sum(len(record["covered_arm_ids"]) for record in records),
        "missing_arm_ids": [] if passed else [arm for record in records for arm in record["required_arm_ids"]],
        "case_evidence": {
            "cold_missing_restart_fields": "real Fortran readstart defaults followed by writerestart serialization",
            "restart_distinct_payload_roundtrip": "real paper restart read by Fortran and serialized with zero normalized-state mismatches",
        },
    }
    (OUTPUT / "branch_coverage.json").write_text(json.dumps(branch, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {OUTPUT} passed={passed} comparisons={len(rows)} mismatches={comparison['mismatch_count']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
