from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.restart_export import (  # noqa: E402
    stomate_restart_states_from_day_end_packet,
)
from jax_orchidee.stomate.reference import (  # noqa: E402
    read_stomate_daily_accumulator_state,
    read_stomate_restart_entry_state,
    read_stomate_restart_season_state,
)
from jax_orchidee.stomate.restart_io import write_stomate_restart_states_from_template  # noqa: E402


def _template() -> Path:
    case_root = (
        ROOT
        / "reference"
        / "OUT"
        / "orc_calibrate_250919_sen"
        / "arg2_1.0"
        / "001.0-071.0"
    )
    paths = sorted(case_root.rglob("stomate_start.nc"))
    if not paths:
        pytest.skip("no real stomate_start.nc under reference/OUT")
    return paths[0]


def test_day_end_packet_merges_and_roundtrips_all_three_restart_contracts(tmp_path: Path):
    template = _template()
    entry = read_stomate_restart_entry_state(template)
    season = read_stomate_restart_season_state(template)
    daily = read_stomate_daily_accumulator_state(template)
    changed_biomass = np.asarray(entry.biomass) + 0.25
    changed_t2m_month = np.asarray(season.t2m_month) + 1.5
    changed_daily = daily._replace(
        gpp_daily=np.asarray(daily.gpp_daily) + 2.0,
        precip_daily=np.asarray(daily.precip_daily) + 3.0,
    )
    packet = {
        "biomass": changed_biomass,
        "t2m_month": changed_t2m_month,
        "gpp_daily": changed_daily.gpp_daily,
        "daily_accumulators": changed_daily._asdict(),
        "lai": np.ones_like(entry.gpp_daily),
    }

    merged_entry, merged_season, merged_daily, report = stomate_restart_states_from_day_end_packet(
        packet,
        base_entry=entry,
        base_season=season,
        base_daily=daily,
    )
    output = tmp_path / "day_end_restart.nc"
    write_stomate_restart_states_from_template(
        template,
        output,
        entry_state=merged_entry,
        season_state=merged_season,
        daily_state=merged_daily,
    )

    np.testing.assert_array_equal(read_stomate_restart_entry_state(output).biomass, changed_biomass)
    np.testing.assert_array_equal(read_stomate_restart_season_state(output).t2m_month, changed_t2m_month)
    np.testing.assert_array_equal(read_stomate_daily_accumulator_state(output).precip_daily, changed_daily.precip_daily)
    assert "biomass" in report.entry_updates
    assert "t2m_month" in report.season_updates
    assert "precip_daily" in report.daily_updates
    assert report.packet_only_fields == ("lai",)


def test_day_end_packet_requires_explicit_consistent_daily_reset_state():
    template = _template()
    entry = read_stomate_restart_entry_state(template)
    season = read_stomate_restart_season_state(template)
    daily = read_stomate_daily_accumulator_state(template)

    with pytest.raises(ValueError, match="must contain explicit daily_accumulators"):
        stomate_restart_states_from_day_end_packet(
            {}, base_entry=entry, base_season=season, base_daily=daily
        )

    with pytest.raises(ValueError, match="gpp_daily differ"):
        stomate_restart_states_from_day_end_packet(
            {
                "gpp_daily": np.asarray(daily.gpp_daily) + 1.0,
                "daily_accumulators": daily._asdict(),
            },
            base_entry=entry,
            base_season=season,
            base_daily=daily,
        )
