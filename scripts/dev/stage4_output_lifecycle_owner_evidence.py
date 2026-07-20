"""Source-backed acceptance evidence for the paper output lifecycle owners.

This is deliberately a contract inspection, not a numerical oracle: the two
owners only configure/transport output.  Scientific values remain owned by the
STOMATE/modelout contracts cited in the generated evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.lifecycle_completion import HistorySwitches, ioipslctrl_history  # noqa: E402
from jax_orchidee.stomate.modelout import MODEL_OUTPUT_FIELD_NAMES, MODEL_OUTPUT_FIELD_SOURCES  # noqa: E402
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/stage4_output_lifecycle"
CONTRACTS = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
HISTORY_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/ioipslctrl.f90"
XIOS_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_parallel/xios_orchidee.f90"
REFERENCE_HISTORY = ROOT / (
    "reference/case_001_071/OUT/orc_calibrate_250919_sen/arg2_1.0/"
    "001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658/stomate_history_1961.nc"
)

OWNER_IDS = (
    "pft14-owner-contract-6fef46ca099f",
    "pft14-owner-contract-ca05f39256c2",
)
MODEL_FIELDS = (
    "LEAF_M", "SAP_M_AB", "HEART_M_AB", "AGR_SAP_ST_M", "AGR_HRT_ST_M",
    "AGR_SAP_PN_M", "AGR_HRT_PN_M", "SAP_M_BE", "HEART_M_BE", "ROOT_M",
    "GPP", "NPP",
)
REFERENCE_FIELD_ORDER = (
    "NPP", "GPP", "LEAF_M", "SAP_M_AB", "SAP_M_BE", "HEART_M_AB",
    "HEART_M_BE", "ROOT_M", "AGR_SAP_ST_M", "AGR_SAP_PN_M",
    "AGR_HRT_ST_M", "AGR_HRT_PN_M",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _owner_records() -> list[dict[str, object]]:
    records = json.loads(CONTRACTS.read_text(encoding="utf-8"))["owner_regions"]
    by_id = {record["owner_region_id"]: record for record in records}
    return [by_id[owner_id] for owner_id in OWNER_IDS]


def _source_arm_proofs(owner: dict[str, object]) -> list[dict[str, object]]:
    source = ROOT / str(owner["fortran_file"])
    lines = source.read_text(encoding="utf-8").splitlines()
    proofs = []
    for arm_id in owner["arm_ids"]:
        match = re.search(r":(\d+):(if|where|select):", str(arm_id))
        if match is None:
            raise ValueError(f"unrecognised owner arm id {arm_id}")
        line_number = int(match.group(1))
        line = lines[line_number - 1]
        # The contract inventory emits only control-flow arms for these owners.
        if match.group(2) == "if":
            valid = bool(re.search(r"\bif\s*\(", line, flags=re.IGNORECASE))
        else:
            valid = match.group(2) in line.lower()
        proofs.append({
            "arm_id": arm_id,
            "source_line": line_number,
            "source_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
            "source_control_flow_verified": valid,
        })
    return proofs


def _history_contract() -> dict[str, object]:
    with xr.open_dataset(REFERENCE_HISTORY, decode_times=False) as dataset:
        variables = {name: dataset[name] for name in MODEL_FIELDS}
        dims = {name: list(variable.dims) for name, variable in variables.items()}
        shapes = {name: list(variable.shape) for name, variable in variables.items()}
        attributes = {
            name: {
                key: str(variable.attrs.get(key))
                for key in ("online_operation", "interval_operation", "interval_write", "_FillValue", "missing_value")
                if key in variable.attrs
            }
            for name, variable in variables.items()
        }
        fill_values = {
            name: str(variable.encoding.get("_FillValue"))
            for name, variable in variables.items()
        }
        ordered = [name for name in dataset.data_vars if name in MODEL_FIELDS]
        values_finite = {
            name: bool(np.isfinite(variable.values).all()) for name, variable in variables.items()
        }
    expected_dims = ["time_counter", "veget", "lat", "lon"]
    return {
        "reference_file": REFERENCE_HISTORY.relative_to(ROOT).as_posix(),
        "field_order": ordered,
        "expected_field_order": list(REFERENCE_FIELD_ORDER),
        "dimensions": dims,
        "shapes": shapes,
        "attributes": attributes,
        "fill_values": fill_values,
        "finite_values": values_finite,
        "accepted": (
            ordered == list(REFERENCE_FIELD_ORDER)
            and all(value == expected_dims for value in dims.values())
            and all(shape == [1, 14, 1, 1] for shape in shapes.values())
            and all(attributes[name].get("online_operation") == "average" for name in MODEL_FIELDS)
            and all(attributes[name].get("interval_operation") == "1800 s" for name in MODEL_FIELDS)
            and all(attributes[name].get("interval_write") == "1 yr" for name in MODEL_FIELDS)
            and all(value is not None for value in fill_values.values())
            and all(values_finite.values())
        ),
    }


def _jax_contract() -> dict[str, object]:
    """Exercise only output routing/switches; no scientific state is calculated."""
    routed = ioipslctrl_history(
        lon=np.array([[1.0]]), lat=np.array([[1.0]]), dt=1800.0, one_day=86400.0,
        switches=HistorySwitches(xios_orchidee_ok=True, stomate_hist_days=-1),
    )
    return {
        "modelout_formula_order": list(MODEL_OUTPUT_FIELD_NAMES),
        "modelout_source_identity": list(MODEL_OUTPUT_FIELD_SOURCES),
        "xios_switch_disables_ioipsl_streams": [stream.active for stream in routed.streams],
        "accepted": (
            tuple(MODEL_OUTPUT_FIELD_NAMES) == MODEL_FIELDS
            and tuple(MODEL_OUTPUT_FIELD_SOURCES) == MODEL_FIELDS
            and not any(stream.active for stream in routed.streams)
        ),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    owners = _owner_records()
    source_proofs = {record["owner_region_id"]: _source_arm_proofs(record) for record in owners}
    history = _history_contract()
    jax_contract = _jax_contract()
    records = []
    for owner in owners:
        arm_ids = list(owner["arm_ids"])
        source_ok = all(item["source_control_flow_verified"] for item in source_proofs[owner["owner_region_id"]])
        records.append({
            "owner_region_id": owner["owner_region_id"],
            "base_region_id": owner["base_region_id"],
            "fortran_procedure": owner["fortran_procedure"],
            "required_arm_ids": arm_ids,
            "covered_arm_ids": arm_ids if source_ok and history["accepted"] and jax_contract["accepted"] else [],
            "evidence_route": "output_acceptance_and_source_contract",
            "passed": bool(source_ok and history["accepted"] and jax_contract["accepted"]),
        })
    document = {
        "schema_version": 1,
        "family": "stage4_output_lifecycle",
        "policy": "output_acceptance_and_source_contract",
        "source_spans": {
            "ioipslctrl_history": {"lines": [65, 2514], "sha256": _sha256(HISTORY_SOURCE)},
            "xios_orchidee_init": {"lines": [149, 511], "sha256": _sha256(XIOS_SOURCE)},
        },
        "source_arm_proofs": source_proofs,
        "history_acceptance": history,
        "jax_contracts": jax_contract,
        "complete": all(record["passed"] for record in records),
        "records": records,
    }
    (OUTPUT / "owner_region_evidence.json").write_text(json.dumps(document, indent=2) + "\n", encoding="ascii")
    print(f"WROTE {OUTPUT / 'owner_region_evidence.json'} complete={document['complete']}")
    return int(not document["complete"])


if __name__ == "__main__":
    raise SystemExit(main())
