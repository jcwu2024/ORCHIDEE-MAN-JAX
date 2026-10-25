"""Source-order regression for the defined (possibly negative) Fortran limiter."""
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from jax_orchidee.stomate.soilcarbon_kernels import (
    IACT, ICARBON, IFREE, NPOOL, soilcarbon_leak_doc_diffusion,
)

REFERENCE = json.loads((Path(__file__).parent / "fixtures/soilcarbon_diffusion_source_order.json").read_text())


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "narrow_jit"])
@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda x: x["id"])
def test_diffusion_matches_independent_source_order_fixture(case, compiled):
    # Thickness sqrt(D) makes the scalar interface coefficient D/h**2 exactly 1.
    # Source peak arithmetic: f0=3*2/(2+2)=1.5; then f1=3*2/(1.5+2)=12/7.
    dif_doc = REFERENCE["d_doc"] * REFERENCE["dt_days"]
    thickness = np.sqrt(dif_doc)
    doc = np.zeros((1, 2, 3, 2, NPOOL, 1), dtype=np.float64)
    doc[0, 1, :, IFREE, IACT, ICARBON] = case["stocks"]
    doc[0, 1, :, 1, IACT, ICARBON] = 77.0
    temperature = np.broadcast_to(np.asarray(case["temperatures"])[None, :, None], (1, 3, 2))
    function = jax.jit(soilcarbon_leak_doc_diffusion) if compiled else soilcarbon_leak_doc_diffusion
    result = function(doc, temperature, np.arange(4) * thickness, np.asarray([dif_doc]))
    np.testing.assert_allclose(np.asarray(result.doc_flux_diff)[0, 1, :, IACT, ICARBON], case["flux"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(result.doc)[0, 1, :, IFREE, IACT, ICARBON], case["updated"], rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(np.asarray(result.doc)[0, 1, :, 1, IACT, ICARBON], 77.0)
