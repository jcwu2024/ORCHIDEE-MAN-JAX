from __future__ import annotations

import pytest

from jax_orchidee.driver.orchestration import (
    _effective_thermosoil_wetdiaglong,
    _validate_stomate_main_supported_switches,
)


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
