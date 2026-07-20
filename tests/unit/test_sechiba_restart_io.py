from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.restart_io import (  # noqa: E402
    DEFAULT_SECHIBA_RESTART_SCHEMA,
    SechibaRestartState,
    SECHIBA_RESTART_COMPONENT_FIELDS,
    SLOWPROC_FINALIZE_ALIASES,
    read_sechiba_restart_state,
    sechiba_restart_state_from_finalize_packets,
    write_sechiba_restart_state,
)
from jax_orchidee.sechiba.restart_export import (  # noqa: E402
    paper_sechiba_restart_state_from_finalize_state,
)
from jax_orchidee.sechiba.enerbil import enerbil_finalize_restart_packet  # noqa: E402
from jax_orchidee.sechiba.hydrol_thermosoil_completion import (  # noqa: E402
    explicitsnow_finalize_restart_packet,
)
from jax_orchidee.stomate.restart_io import StomateRestartPhysicalState  # noqa: E402
from scripts.dev.extract_stomate_restart_netcdf_schema import extract_schema  # noqa: E402


def _template() -> Path:
    root = (
        ROOT
        / "reference"
        / "OUT"
        / "orc_calibrate_250919_sen"
        / "arg2_1.0"
        / "001.0-071.0"
    )
    paths = sorted(root.rglob("sechiba_start.nc"))
    if not paths:
        pytest.skip("paper sechiba_start.nc is unavailable")
    return paths[0]


def _physical(path: Path) -> StomateRestartPhysicalState:
    with Dataset(path) as dataset:
        return StomateRestartPhysicalState(
            **{
                name: np.asarray(dataset.variables[name][:])
                for name in ("nav_lon", "nav_lat", "nav_lev", "time", "time_steps")
            }
        )


def test_sechiba_schema_matches_fortran_paper_file() -> None:
    actual = json.loads(DEFAULT_SECHIBA_RESTART_SCHEMA.read_text(encoding="utf-8"))
    expected = extract_schema(
        _template(),
        scope="ORCHIDEE-MAN PFT14 independent single-landpoint SECHIBA restart",
        fortran_file="fortran_source/ORCHIDEE/src_sechiba/sechiba.f90",
        subroutine="sechiba_finalize and component finalizers",
        lines="1800-1923",
    )
    assert actual == expected
    assert len(actual["dimensions"]) == 16
    assert len(actual["variables"]) == 98


def test_sechiba_standalone_restart_roundtrips_all_93_scientific_variables(
    tmp_path: Path,
) -> None:
    template = _template()
    original = read_sechiba_restart_state(template)
    changed = SechibaRestartState(
        fields={
            name: np.asarray(value) + (index + 1) / 1024.0
            for index, (name, value) in enumerate(original.fields.items())
        },
        provenance_by_field=original.provenance_by_field,
    )
    output = tmp_path / "sechiba_restart.nc"

    report = write_sechiba_restart_state(
        output,
        state=changed,
        physical_state=_physical(template),
    )
    restored = read_sechiba_restart_state(output)

    assert len(changed.fields) == len(restored.fields) == 93
    for name in changed.fields:
        np.testing.assert_array_equal(restored.fields[name], changed.fields[name])
    assert len(report.written_fields) == 93
    assert report.unsupported_fields == ()
    assert restored.fields["frac_age"].shape == (1, 14, 4)
    assert restored.fields["moistc"].shape == (1, 11, 6)
    assert restored.fields["us"].shape == (1, 14, 6, 11)
    assert restored.fields["ptn"].shape == (1, 32, 14)


def test_sechiba_writer_requires_exact_file_field_denominator(tmp_path: Path) -> None:
    original = read_sechiba_restart_state(_template())
    fields = dict(original.fields)
    fields.pop("qsurf")
    with pytest.raises(ValueError, match="coverage mismatch.*qsurf"):
        write_sechiba_restart_state(
            tmp_path / "missing.nc",
            state=SechibaRestartState(fields, original.provenance_by_field),
            physical_state=_physical(_template()),
        )


def test_seven_finalize_owners_assemble_exact_93_field_denominator() -> None:
    original = read_sechiba_restart_state(_template())
    packets = {
        component: {
            name: original.fields[name]
            for name in fields
        }
        for component, fields in SECHIBA_RESTART_COMPONENT_FIELDS.items()
    }
    reverse_aliases = {output: source for source, output in SLOWPROC_FINALIZE_ALIASES.items()}
    packets["slowproc"] = {
        reverse_aliases.get(name, name): value
        for name, value in packets["slowproc"].items()
    }

    assembled = sechiba_restart_state_from_finalize_packets(**packets)

    assert set(assembled.fields) == set(original.fields)
    assert len(assembled.fields) == 93
    for name, value in original.fields.items():
        np.testing.assert_array_equal(assembled.fields[name], value)


def test_enerbil_and_explicit_snow_finalize_packets_cover_source_fields() -> None:
    state = read_sechiba_restart_state(_template()).fields
    enerbil = enerbil_finalize_restart_packet(
        evapot=state["evapot"],
        evapot_corr=state["evapot_corr"],
        temp_sol=state["temp_sol"],
        temp_sol_pft=state["temp_sol_pft"],
        tsol_rad=state["tsolrad"],
        qsurf=state["qsurf"],
        fluxsens=state["fluxsens"],
        fluxlat=state["fluxlat"],
        vevapp=state["evapora"],
        temp_sol_pot=state["tempsolpot"],
        q_sol_pot=state["qsolpot"],
    )
    snow = explicitsnow_finalize_restart_packet(
        **{name: state[name] for name in SECHIBA_RESTART_COMPONENT_FIELDS["explicit_snow"]}
    )
    assert set(enerbil) == SECHIBA_RESTART_COMPONENT_FIELDS["enerbil"]
    assert set(snow) == SECHIBA_RESTART_COMPONENT_FIELDS["explicit_snow"]


def test_paper_finalize_state_runs_all_component_owners_without_carry() -> None:
    original = read_sechiba_restart_state(_template())
    source = dict(original.fields)
    aliases = {
        "cdrag_pft": "q_cdrag_pft",
        "tsolrad": "tsol_rad",
        "evapora": "vevapp",
        "tempsolpot": "temp_sol_pot",
        "qsolpot": "q_sol_pot",
        "moistc": "mc",
        "moistcl": "mcl",
        "soilalbedo_bg": "soilalb_bg",
        "refSOC": "refsoc",
        "shum_ngrnd_prmlng": "shum_ngrnd_permalong",
        **{output: input_name for input_name, output in SLOWPROC_FINALIZE_ALIASES.items()},
    }
    for output, input_name in aliases.items():
        source[input_name] = source.pop(output)

    assembled = paper_sechiba_restart_state_from_finalize_state(source, kjit=17520)

    assert set(assembled.fields) == set(original.fields)
    for name, value in original.fields.items():
        np.testing.assert_array_equal(assembled.fields[name], value)
