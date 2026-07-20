"""Build PFT14 owner evidence for ``stomate_data::data`` from server assets.

The branch profile establishes that the linked original procedure executed.  The
schema-3 capture supplies the actual PFT14 inputs used by that invocation, and
the strict comparator supplies the output equality check.  This builder derives
the owner-region arm set from those three evidence sources; it does not turn an
unexercised alternate configuration into a coverage claim.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "outputs/reference_mode/micro_oracles/stage4_stomate_data"
CAPTURE = FAMILY / "fortran_server_pft14_schema3.log"
PROFILE = FAMILY / "fortran_server_original_branch_coverage.txt"
COMPARISON = FAMILY / "comparison.json"
CONTRACT = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_data.f90"
OUTPUT = FAMILY / "owner_region_evidence.json"
OWNER_ID = "pft14-owner-contract-95639b7ddbb3"


def _comparator_module():
    path = ROOT / "scripts/dev/compare_stage4_stomate_data_capture.py"
    spec = importlib.util.spec_from_file_location("stage4_stomate_data_comparator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _owner() -> dict:
    document = json.loads(CONTRACT.read_text(encoding="utf-8"))
    matches = [
        item for item in document["owner_regions"]
        if item["owner_region_id"] == OWNER_ID
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one contract owner {OWNER_ID}")
    return matches[0]


def _profile_counts() -> tuple[int, int]:
    for line in PROFILE.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields[:2] == ["function", "-"] and "stomate_data_mp_data_" in fields:
            total, covered = int(fields[-3]), int(fields[-2])
            if not covered or covered > total:
                break
            return covered, total
    raise ValueError("original stomate_data profile has no executed procedure record")


def _profile_line_counts() -> dict[int, int]:
    counts: dict[int, int] = {}
    for raw in PROFILE.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if len(fields) == 4 and all(value.lstrip("-").isdigit() for value in fields):
            line_number, count = int(fields[1]), int(fields[3])
            counts[line_number] = max(counts.get(line_number, 0), count)
    return counts


def _source_line(line_number: int) -> str:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    return lines[line_number - 1].strip()


def _require_source(branch_id: str, expected: str) -> None:
    line_number = int(branch_id.split(":")[1])
    actual = _source_line(line_number)
    if actual != expected:
        raise ValueError(f"{branch_id}: expected {expected!r}, found {actual!r}")


def _predicate_arms(config: dict) -> dict[str, str]:
    undef, minimum = config["undef"], config["min_stomate"]
    return {
        "261": "true" if config["is_tree_pft14"] else "false",
        "275": "true" if config["pheno_type_pft14"] != 1 else "false",
        "303": "true" if config["ok_laidev_pft14"] else "false",
        "317": "true" if config["natural_pft14"] or config["is_grassland_manag_pft14"] else "false",
        "346": "true" if not config["ok_dgvm"] and config["use_age_class"] else "false",
        "367": "true" if config["is_tree_pft14"] else "false",
        "385": "true" if config["is_tree_pft14"] else "false",
        "408": "true" if abs(config["tmin_crit_pft14"] - undef) > minimum else "false",
        "422": "true" if abs(config["tcm_crit_pft14"] - undef) > minimum else "false",
        "453": "true" if config["pheno_model_pft14"] in {"moigdd", "humgdd"} and any(value == undef for value in config["pheno_gdd_crit_pft14"]) else "false",
        "504": "true" if config["senescence_type_pft14"] in {"cold", "mixed"} and any(value == undef for value in config["senescence_temp_pft14"]) else "false",
        "519": "true" if config["senescence_type_pft14"] in {"dry", "mixed"} and config["senescence_hum_pft14"] == undef else "false",
        "535": "true" if config["senescence_type_pft14"] in {"dry", "mixed"} and config["nosenescence_hum_pft14"] == undef else "false",
        "584": "true" if config["is_tree_pft14"] else "false",
    }


SOURCE_CONDITIONS = {
    "261": "IF ( is_tree(j) ) THEN", "275": "IF ( pheno_type(j) .NE. 1 ) THEN",
    "303": "IF (ok_LAIdev(j)) THEN", "317": "IF ( natural(j) .OR. is_grassland_manag(j) ) THEN",
    "346": "IF (.NOT. ok_dgvm .AND. use_age_class) THEN", "367": "IF ( is_tree(j) ) THEN",
    "385": "IF ( is_tree(j) ) THEN", "408": "IF ( ABS( tmin_crit(j) - undef ) .GT. min_stomate ) THEN",
    "422": "IF ( ABS ( tcm_crit(j) - undef ) .GT. min_stomate ) THEN",
    "453": "IF ( ( ( pheno_model(j) .EQ. 'moigdd' ) .OR. &",
    "504": "IF ( ( ( senescence_type(j) .EQ. 'cold' ) .OR. &",
    "519": "IF ( ( ( senescence_type(j) .EQ. 'dry' ) .OR. &",
    "535": "IF ( ( ( senescence_type(j) .EQ. 'dry' ) .OR. &",
    "584": "IF ( is_tree(j) ) THEN",
}

# Branch-body lines from the pinned source. A zero count is evidence that the
# fatal guard did not fire; a positive count is evidence that its continuation
# or selected assignment executed in the original compiled-NVM=14 run.
PROFILE_ARM_LINES = {
    "261:if:true": (269, True), "275:if:false": (279, True),
    "303:if:false": (317, True), "317:if:true": (318, True),
    "346:if:false": (347, False), "367:if:true": (369, True),
    "385:if:true": (391, True), "408:if:true": (409, True),
    "408:if:false": (411, True), "422:if:true": (423, True),
    "422:if:false": (425, True), "453:if:false": (456, False),
    "504:if:false": (507, False), "519:if:false": (522, False),
    "535:if:false": (538, False), "584:if:true": (585, True),
}


def build() -> dict:
    comparator = _comparator_module()
    config, result_data = comparator.parse_capture(CAPTURE)
    comparison = comparator.compare(CAPTURE)
    if not comparison["passed"]:
        raise ValueError("strict schema-3 JAX comparison failed")
    owner = _owner()
    arms = _predicate_arms(config)
    required_ids = set(owner["arm_ids"])
    required_lines = {arm_id.split(":")[1] for arm_id in required_ids}
    if required_lines != set(arms):
        raise ValueError(
            "owner contract source lines do not match the captured predicate set; "
            f"missing={sorted(required_lines - set(arms))} "
            f"unexpected={sorted(set(arms) - required_lines)}"
        )
    for line, condition in SOURCE_CONDITIONS.items():
        _require_source(f"fortran_source/ORCHIDEE/src_stomate/stomate_data.f90:{line}:if", condition)
    covered_blocks, total_blocks = _profile_counts()
    line_counts = _profile_line_counts()
    profile_arm_counts = {}
    for arm_id in required_ids:
        _, line, branch, arm = arm_id.rsplit(":", 3)
        evidence_line, must_execute = PROFILE_ARM_LINES[f"{line}:{branch}:{arm}"]
        count = line_counts.get(evidence_line)
        if count is None or (count > 0) != must_execute:
            raise ValueError(f"{arm_id}: unexpected profile count at source line {evidence_line}: {count}")
        profile_arm_counts[arm_id] = {"source_line": evidence_line, "execution_count": count}
    return {
        "schema_version": 1,
        "complete": True,
        "records": [{
            "owner_region_id": OWNER_ID,
            "base_region_id": owner["base_region_id"],
            "fortran_procedure": "data",
            "required_arm_ids": sorted(required_ids),
            "covered_arm_ids": sorted(required_ids),
            "passed": True,
            "evidence": {
                "original_fortran_capture": CAPTURE.relative_to(ROOT).as_posix(),
                "original_fortran_capture_sha256": hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
                "original_fortran_profile": PROFILE.relative_to(ROOT).as_posix(),
                "original_fortran_profile_sha256": hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
                "original_procedure_blocks": {"covered": covered_blocks, "total": total_blocks},
                "profile_arm_counts": profile_arm_counts,
                "strict_comparison": COMPARISON.relative_to(ROOT).as_posix(),
                "strict_comparison_sha256": hashlib.sha256(COMPARISON.read_bytes()).hexdigest(),
                "source": SOURCE.relative_to(ROOT).as_posix(),
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "arm_derivation": "schema-3 PFT14 source configuration evaluated against exact Fortran IF predicates",
                "pft14_predicates": arms,
                "sentinel_conversion_outputs": {
                    "tmin_crit_pft14": result_data["tmin_crit_pft14"],
                    "tcm_crit_pft14": result_data["tcm_crit_pft14"],
                },
            },
        }],
        "note": (
            "The original profile covers the listed owner-contract arms across the active "
            "compiled-NVM=14 PFT vector. pft14_predicates records the narrower PFT14-index "
            "dispositions from schema-3; alternate PFT14 configurations are not claimed covered."
        ),
    }


def main() -> None:
    OUTPUT.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {OUTPUT}")


if __name__ == "__main__":
    main()
