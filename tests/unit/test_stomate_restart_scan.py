from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.dev.scan_stomate_io_restart_fields import DEFAULT_SOURCE, scan_stomate_io_restart_fields  # noqa: E402


def test_scan_stomate_io_restart_fields_resolves_var_name_and_literal_calls():
    scan = scan_stomate_io_restart_fields(DEFAULT_SOURCE)
    records = scan["records"]
    pairs = {(record["line"], record["call"], record["name"]) for record in records}

    assert scan["summary"]["records"] == 340
    assert scan["summary"]["unique_call_field_pairs"] == 333
    assert scan["summary"]["unique_fields"] == 167
    assert (2355, "restput_p", "delta_fsave") in pairs
    assert (2358, "restput_p", "litter_above") in pairs
    assert (2414, "restput_p", "turnover_daily") in pairs
    assert (2624, "restput_p", "turnover_longterm") in pairs
