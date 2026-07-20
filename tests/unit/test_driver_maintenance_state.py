from __future__ import annotations

import numpy as np

from jax_orchidee.driver.orchestration import _write_maintenance_lai_to_previous_fields
from jax_orchidee.stomate.carbon_kernels import ICARBON, ILEAF, NPARTS


def test_maintenance_lai_writeback_updates_stomate_and_next_diffuco_state():
    npts, nvm = 1, 3
    previous_lai = np.asarray([[7.0, 8.0, 9.0]], dtype=np.float64)
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, :, ILEAF, ICARBON] = [4.0, 5.0, 6.0]
    sla_calc = np.asarray([[0.5, 0.4, 0.3]], dtype=np.float64)
    prior = {
        "lai": previous_lai,
        "biomass": biomass,
        "sla_calc": sla_calc,
    }
    fields = {
        "slowproc_stomate_previous_step_state": {"lai": previous_lai},
        "diffuco_previous_step_state": {"lai": previous_lai},
    }

    _write_maintenance_lai_to_previous_fields(fields, prior, (False, True, False))

    expected = np.asarray([[0.0, 8.0, 1.8]], dtype=np.float64)
    np.testing.assert_allclose(np.asarray(fields["slowproc_stomate_previous_step_state"]["lai"]), expected)
    np.testing.assert_allclose(np.asarray(fields["diffuco_previous_step_state"]["lai"]), expected)
