"""Extract compact canonical frozen binding layout from metadata only."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
source = ROOT / "outputs/research/global_throughput/round4-inputs-remote-canonical/remote-preparation-audit.json"
raw = source.read_bytes()
audit = json.loads(raw)
records = audit["new_point_records"]
assert len(records) == 60
layout = records[0]["positional_state_representation_migration"]["bound_layout"]
assert all(record["positional_state_representation_migration"]["bound_layout"] == layout for record in records)
assert audit["prepared_pickle_sha256"] == "be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7"
assert layout[0][0] == "diffuco_previous_step_state" and layout[1][0][13] == "roughheight_pft"
assert layout[0][5] == "sechiba_finalize_state" and layout[1][5][18] == "roughheight_pft"
proofs = []
for record in records:
    migration = record["positional_state_representation_migration"]
    bijection = migration["named_field_bijection"]
    for component in bijection:
        ci, si = component["bound_component_index"], component["source_component_index"]
        assert layout[0][ci] == component["component"] == migration["source_layout"][0][si]
        for field in component["fields"]:
            assert (
                layout[1][ci][field["bound_field_index"]]
                == field["field"]
                == migration["source_layout"][1][si][field["source_field_index"]]
            )
    proofs.append(hashlib.sha256(json.dumps(bijection, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
screen = audit["actual_full64_reader_screen"]
assert screen["passed"] is True and screen["dynamic_leaf_count"] == 536
assert screen["static_digest"] == "b1b4d5b9819ff8845ecb7b0cc68bf641da1c5740effc1c13185f9bde872fe329"
failed_report = (
    ROOT
    / "outputs/research/global_throughput/gpu-lifecycle-execution-20261011/terminal-evidence/run/phased_resident_control/capacity/B1024/report.json"
)
report = json.loads(failed_report.read_text())
assert report["input_compatibility_audit"]["static_digest"] == screen["static_digest"]
document = {
    "state_layout": layout,
    "provenance": {
        "source_audit_path": str(source.resolve()),
        "source_audit_sha256": hashlib.sha256(raw).hexdigest(),
        "source_audit_bytes": len(raw),
        "source": "new_point_records[*].positional_state_representation_migration.bound_layout",
        "all60_bound_layouts_exactly_equal": True,
        "point_count": 60,
        "canonical_prepared_pickle_sha256": audit["prepared_pickle_sha256"],
        "source_input_audit_sha256": audit["source_audit_sha256"],
        "actual_full64_reader_screen": {
            key: screen[key]
            for key in (
                "passed",
                "dynamic_leaf_count",
                "block_size",
                "inner_npts",
                "unique_real_points",
                "static_digest",
            )
        },
        "failed_report_static_digest_exactly_matches": True,
        "failed_report_sha256": hashlib.sha256(failed_report.read_bytes()).hexdigest(),
        "all60_named_field_bijections_source_bound_indices_verified": True,
        "named_field_bijection_record_sha256": proofs,
        "authority": "remote canonical frozen binding layout including the canonical four anchors; local preparation after_layout is not used",
        "no_prepared_pickle_loaded": True,
        "no_model_or_JAX_imported": True,
    },
}
destination = Path(__file__).with_name("state-layout.json")
destination.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
print(
    json.dumps(
        {
            "status": "passed_metadata_layout_extraction",
            "components": layout[0],
            "field_counts": [len(names) for names in layout[1]],
            "source_audit_sha256": document["provenance"]["source_audit_sha256"],
            "output_bytes": destination.stat().st_size,
        }
    )
)
