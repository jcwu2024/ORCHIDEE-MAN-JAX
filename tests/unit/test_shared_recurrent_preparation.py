import json
from types import SimpleNamespace

import pytest

from scripts.hpc.prepare_shared_recurrent_admission import (
    ROOT,
    load_protocol_and_calibration,
    selected_window_metadata,
)
from scripts.hpc.shared_transformer_runtime import select_artifact_ids


def evidence(tmp_path):
    protocol_path = ROOT / "manifests/coarse_graining/shared_recurrent_admission_v1.json"
    protocol = json.loads(protocol_path.read_text())
    daily = json.loads((protocol_path.parent / protocol["daily_protocol"]).read_text())
    records = [{"update": i, "runtime_year": 1961 + i % 3} for i in range(20)]
    ids = select_artifact_ids(records, daily["calibration"]["artifact_count"])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": records, "identity": {"source": "accepted_v6"}}))
    report = {"status": "passed", "protocol": daily,
              "calibration": {"split": "train/train", "artifact_ids": ids},
              "data": {"prepared_manifest_identity": {"source": "accepted_v6"}}}
    saved = tmp_path / "calibration.json"
    saved.write_text(json.dumps(report))
    return protocol_path, saved, manifest, report, ids


def test_reuses_declared_train_calibration_and_selects_two_artifacts(tmp_path):
    path, saved, manifest, _, ids = evidence(tmp_path)
    protocol, _daily, _saved, selected = load_protocol_and_calibration(path, saved, manifest)
    assert selected == ids[:2]
    assert protocol["joint_daily_coefficients"] == [1/3, 1/9, 1/9]
    assert protocol["joint_recurrent_flux_stock_coefficients"] == [1/9, 1/9]


@pytest.mark.parametrize("change", ["split", "identity", "protocol", "selection", "status"])
def test_rejects_inapplicable_saved_calibration(tmp_path, change):
    path, saved, manifest, report, _ids = evidence(tmp_path)
    if change == "split":
        report["calibration"]["split"] = "validation/validation"
    elif change == "identity":
        report["data"]["prepared_manifest_identity"] = {"source": "another"}
    elif change == "protocol":
        report["protocol"]["architecture"]["thermal_transform"] = "direct_tanh"
    elif change == "selection":
        report["calibration"]["artifact_ids"] = list(reversed(report["calibration"]["artifact_ids"]))
    else:
        report["status"] = "running"
    saved.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_protocol_and_calibration(path, saved, manifest)


def test_window_selection_reports_actual_first_window_identity():
    artifact = SimpleNamespace(schedule_entry={"update": 819, "components": {
        "seven_day": [{"landpoint_id": "site", "year": 1961, "start_rows": [86, 100]}]}})
    actual = selected_window_metadata(artifact, "seven_day", 7)
    assert actual == dict(artifact_id=819, component="seven_day", horizon=7, window_index=0,
                          landpoint_id="site", year=1961, start_row=86)
