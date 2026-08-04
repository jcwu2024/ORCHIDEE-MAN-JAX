from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jax_orchidee.driver.orchestration import (
    DriverDailyFluxLabelsV1,
    DriverEnergyFluxStepV1,
    DriverSechibaDailyFluxV1,
    DriverWaterTransferStepV1,
)
from jax_orchidee.stomate.integration import OkLeakTransferStepV1
from research.daily_coarse_graining.daily_flux_capture import (
    CAPTURE_LABEL_PATHS,
    daily_flux_capture_arrays,
    validate_daily_flux_capture,
    write_daily_flux_capture,
)

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "manifests/coarse_graining/daily_flux_label_inventory_v1.json"


def _filled(named_tuple_type, value: float):
    return named_tuple_type(**{name: np.asarray([value]) for name in named_tuple_type._fields})


def _labels() -> DriverDailyFluxLabelsV1:
    return DriverDailyFluxLabelsV1(
        sechiba=DriverSechibaDailyFluxV1(
            water=_filled(DriverWaterTransferStepV1, 1.0),
            energy=_filled(DriverEnergyFluxStepV1, 2.0),
        ),
        ok_leak=_filled(OkLeakTransferStepV1, 3.0),
    )


def test_daily_flux_capture_covers_every_frozen_non_identifiable_label():
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    required = {entry["id"] for entry in inventory["labels"] if entry["availability"] == "missing_non_identifiable"}
    assert set(CAPTURE_LABEL_PATHS) == required
    report = validate_daily_flux_capture(_labels(), inventory_path=INVENTORY)
    assert report["ok"]
    assert report["required_capture_label_count"] == 29


def test_daily_flux_capture_serialization_boundary_has_no_step_axis():
    arrays = daily_flux_capture_arrays(_labels())
    assert arrays
    assert all(array.shape == (1,) for array in arrays.values())
    assert all(not (array.ndim and array.shape[0] == 48) for array in arrays.values())


def test_daily_flux_capture_serializes_defined_mask_for_inactive_nan(tmp_path):
    labels = _labels()
    labels.sechiba.energy.netrad_pft[0] = np.nan
    report = write_daily_flux_capture(
        labels,
        output_dir=tmp_path,
        inventory_path=INVENTORY,
    )
    assert report["ok"]
    assert report["nan_paths_requiring_defined_mask"] == ["energy.netrad_pft"]
    with np.load(tmp_path / "daily_flux_labels.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["energy.netrad_pft__defined"], [False])
