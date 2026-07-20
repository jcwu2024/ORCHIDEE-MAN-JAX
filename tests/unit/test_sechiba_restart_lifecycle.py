from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.restart_io import (  # noqa: E402
    read_sechiba_restart_state,
    sechiba_finalize_source_state_from_restart,
)
from jax_orchidee.sechiba.restart_lifecycle import (  # noqa: E402
    SECHIBA_FINALIZE_SOURCE_FIELDS,
    transition_sechiba_finalize_source_state,
)


def _source_state():
    root = ROOT / "reference" / "OUT" / "orc_calibrate_250919_sen" / "arg2_1.0" / "001.0-071.0"
    files = sorted(root.rglob("sechiba_start.nc"))
    if not files:
        pytest.skip("paper sechiba_start.nc is unavailable")
    return sechiba_finalize_source_state_from_restart(read_sechiba_restart_state(files[0]))


def test_transition_requires_explicit_owner_for_all_93_fields() -> None:
    previous = _source_state()
    with pytest.raises(ValueError, match="ownership mismatch"):
        transition_sechiba_finalize_source_state(
            previous,
            updates={"rstruct": previous["rstruct"]},
            carried_fields=(),
            update_provenance={"rstruct": ("diffuco.f90::diffuco_main",)},
            carry_provenance={},
        )


def test_transition_records_updates_and_source_backed_carries_without_defaults() -> None:
    previous = _source_state()
    updated = {"rstruct"}
    carried = tuple(sorted(SECHIBA_FINALIZE_SOURCE_FIELDS - updated))
    transition = transition_sechiba_finalize_source_state(
        previous,
        updates={"rstruct": previous["rstruct"]},
        carried_fields=carried,
        update_provenance={"rstruct": ("diffuco.f90::diffuco_main lines 629-710",)},
        carry_provenance={name: ("audited source carry",) for name in carried},
    )

    assert len(transition.fields) == 93
    assert transition.updated_fields == ("rstruct",)
    assert len(transition.carried_fields) == 92
