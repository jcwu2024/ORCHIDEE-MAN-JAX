from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from jax_orchidee.driver.init import read_run_scalars
from jax_orchidee.driver.orchestration import (
    _effective_thermosoil_wetdiaglong,
    _paper_mangrove_fortran_pft_id,
    _paper_mangrove_pft_index,
    _select_run_def_pft_vector,
    _validate_stomate_main_supported_switches,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
CATALOG = ROOT / "configs" / "pft_catalogs" / "orchidee_man_paper_250919.json"


def test_validate_stomate_main_switches_rejects_ok_leak_false_full_driver():
    with pytest.raises(NotImplementedError, match="OK_LEAK=n"):
        _validate_stomate_main_supported_switches(
            {
                "STOMATE_OK_STOMATE": "y",
                "OK_LEAK": "n",
                "OK_PC": "y",
            }
        )


def test_validate_stomate_main_switches_allows_paper_ok_leak_path():
    _validate_stomate_main_supported_switches(
        {
            "STOMATE_OK_STOMATE": "y",
            "OK_LEAK": "y",
            "OK_PC": "n",
        }
    )


@pytest.mark.parametrize(
    ("values", "expected"),
    (
        ({"OK_WETDIAGLONG": "TRUE"}, True),
        ({"OK_WETDIAGLONG": "FALSE", "OK_FREEZE": "TRUE", "OK_PC": "TRUE"}, True),
        ({"OK_WETDIAGLONG": "FALSE", "OK_FREEZE": "TRUE", "OK_FREEZE_THERMIX": "FALSE", "OK_PC": "TRUE"}, False),
        ({"OK_WETDIAGLONG": "FALSE", "OK_LEAK": "TRUE"}, True),
        ({"OK_WETDIAGLONG": "FALSE", "OK_FREEZE": "TRUE", "OK_PC": "FALSE", "OK_LEAK": "FALSE"}, False),
    ),
)
def test_effective_thermosoil_wetdiaglong_matches_fortran_initialization(values, expected):
    assert _effective_thermosoil_wetdiaglong(values) is expected


def test_generic_orchestration_selects_mangrove_by_stable_id_in_compact_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    catalog_payload["layouts"]["paper_compact"] = ["bare_soil", "mangrove_pft14"]
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog_payload), encoding="utf-8")

    config_payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config_payload["pft_catalog"] = {"path": str(catalog_path), "layout_id": "paper_compact"}
    config_payload["structural_overrides"]["NVM"] = 2
    config_path = tmp_path / "compact.yaml"
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")
    monkeypatch.setenv("ORCHIDEE_REPO_ROOT", str(ROOT))

    run_scalars = read_run_scalars(config_path)
    selected = _select_run_def_pft_vector(
        {
            "TEST_VALUE__00001": "10.0",
            "TEST_VALUE__00014": "140.0",
        },
        "TEST_VALUE",
        run_scalars,
        nvm=2,
    )

    assert run_scalars.pft_ids == ("bare_soil", "mangrove_pft14")
    assert _paper_mangrove_pft_index(run_scalars) == 1
    assert _paper_mangrove_fortran_pft_id(run_scalars) == 14
    np.testing.assert_array_equal(selected, [10.0, 140.0])

    with pytest.raises(ValueError, match="model-state PFT axis"):
        _select_run_def_pft_vector(
            {"TEST_VALUE__00001": "10.0", "TEST_VALUE__00014": "140.0"},
            "TEST_VALUE",
            run_scalars,
            nvm=14,
        )
