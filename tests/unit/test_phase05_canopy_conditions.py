"""Finite PB01 phase05 original-reference conditions; no model trajectory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from scripts.dev import oracle_phase05_canopy_conditions as canopy

REFERENCE = json.loads((Path(__file__).parents[1]/"fixtures/phase05_canopy_reference.json").read_text())
FIELDS = [(g,k) for g,fields in REFERENCE["expected"].items() for k in fields]


@pytest.fixture(scope="module", params=(False,True), ids=("eager","jit"))
def actual(request):
    assert jax.default_backend()=="cpu"
    assert jax.config.jax_enable_x64
    return canopy.actual_values(use_jit=request.param)


@pytest.mark.parametrize("group,field", FIELDS)
def test_phase05_original_fortran_field(actual, group, field):
    np.testing.assert_allclose(actual[group][field], REFERENCE["expected"][group][field],
                               rtol=REFERENCE["rtol"], atol=REFERENCE["atol"])


def test_phase05_declared_input_identity():
    inputs = dict(roots={k:canopy.make_control_inputs(k) for k in canopy.CAPS},
                  albedo=canopy.make_albedo_inputs(), low_lai=canopy.make_low_lai_inputs(),
                  producer=canopy.producer_inputs(canopy.make_low_lai_inputs()), albedo_switches=canopy.ALBEDOS)
    normalized = json.loads(json.dumps(inputs, default=lambda value:value.tolist()))
    assert normalized==REFERENCE["inputs"]


@pytest.mark.external_data
def test_phase05_original_source_identity():
    checks = []
    for group, identity in REFERENCE["reference_identity"].items():
        spans = identity["spans"]
        if "source" in spans:
            checks.append((canopy.ROOT/spans["source"], spans))
        else:
            for name, span in spans.items():
                filename = "slowproc.f90" if name.startswith("slowproc_") else "condveg.f90"
                checks.append((canopy.ROOT/"fortran_source/ORCHIDEE/src_sechiba"/filename,span))
    missing = sorted({str(path) for path,_ in checks if not path.is_file()})
    if missing:
        pytest.skip("Original-source identity NOT EXECUTED; missing: "+", ".join(missing))
    for path,span in checks:
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==span["source_sha256"]
        block = b"".join(raw.splitlines(keepends=True)[span["start_line"]-1:span["end_line"]])
        assert hashlib.sha256(block).hexdigest()==span.get("span_sha256",span.get("sha256"))
