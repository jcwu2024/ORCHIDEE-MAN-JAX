"""Original zero-route values through real writers and one HYDROL consumer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.dev import audit_phase06_root_joins as joins

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("compiled", (False, True))
def test_original_zero_route_reaches_actual_daily_conversion(compiled):
    reference = json.loads(joins.FIXTURE.read_text(encoding="utf-8"))
    for name, actual in joins.routing_fields(compiled=compiled).items():
        np.testing.assert_array_equal(actual, reference["expected"][name], err_msg=name)


def test_actual_single_point_hydrol_receives_zero_reinfiltration_and_irrigation():
    from scripts.dev.oracle_hydrol_component_owner import run_owner_case

    result = run_owner_case(1)
    assert result.ok
    for field in ("reinfiltration_soil", "irrigation_soil"):
        np.testing.assert_array_equal(result.module_inputs[field], np.zeros((1, 6)))
    for field in ("returnflow", "reinfiltration", "irrigation"):
        np.testing.assert_array_equal(result.routing_zero[field], np.zeros((1, 1)))
    # Independently preserved original HYDROL case1 numerical return proves
    # that the real wrapper/component executed; no physics are replaced.
    np.testing.assert_allclose(result.diagnostics.drysoil_frac, [.10504178806323072], rtol=1e-12, atol=1e-14)


@pytest.mark.external_data
def test_original_routing_span_and_saved_output_identities():
    reference = json.loads(joins.FIXTURE.read_text(encoding="utf-8"))
    path = ROOT / reference["source"]
    if not path.is_file():
        pytest.skip("original routing source identity not executed: external source absent")
    lines = path.read_bytes().splitlines(keepends=True)
    for span in reference["source_spans"]:
        assert hashlib.sha256(b"".join(lines[span["start"] - 1:span["end"]])).hexdigest() == span["sha256"]
