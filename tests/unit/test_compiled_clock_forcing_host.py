"""Actual host forcing factory with synthetic weather and bound chronology."""
from types import SimpleNamespace as NS

import numpy as np

from jax_orchidee.driver import orchestration as orch
from jax_orchidee.driver.source_clock import SourceClockBinding, SourceClockRuntime, SourceClockSetup
from jax_orchidee.sechiba.ioipsl_calendar import IOIPSLCalendar


def test_actual_compiled_forcing_day_uses_source_frame_independent_of_weather_year(monkeypatch):
    date0 = IOIPSLCalendar("noleap").ymds2ju(2000, 12, 31, 0.)
    lock = IOIPSLCalendar.from_scalar_startdate("noleap", date0)
    binding = SourceClockBinding("synthetic-host-forcing", "noleap", date0, 1800.,
                                lock.startdate_day, lock.startdate_second, 0)
    runtime = SourceClockRuntime(binding, np.int64(16), np.int64(0), np.int64(17))
    land = np.ones((3, 1), dtype=np.float64)
    forcing = NS(**{name: land for name in (
        "zlev", "zlevuv", "u", "v", "qair", "temp_air", "pb", "precip_rain",
        "precip_snow", "lwdown", "swdown")})
    context = NS(config_path="synthetic-forcing.yml", dt_sechiba=1800.,
        forcing_split=1, forcing_nb_spread=1,
        source_clock_setup=SourceClockSetup(binding, runtime, None, None, None),
        first_step_bundle=NS(domain=NS(nbindex=3), static_trace_fields=NS(
            salinity=np.asarray([1., 2., 3.]), tide_height=np.asarray([4., 5., 6.]))))
    calls = []
    monkeypatch.setattr(orch, "read_annual_co2", lambda path, year: 400.)
    def weather(path, *, model_tstep, year, **kwargs):
        calls.append((year, model_tstep))
        return forcing
    monkeypatch.setattr(orch, "read_forcing_model_step_cached", weather)
    packed = orch._paper_compiled_forcing_day(context, year=2099, start_tstep=0,
        steps_per_stomate=3, source_clock_runtime=runtime)
    assert calls == [(2099, 0), (2099, 1), (2099, 2)]
    np.testing.assert_array_equal(packed.source_clock.local_tstep, [0, 1, 2])
    np.testing.assert_array_equal(packed.source_clock.istp, [17, 18, 19])
    np.testing.assert_array_equal(packed.source_clock.year, [2000, 2000, 2000])
    np.testing.assert_array_equal(packed.source_clock.second, [30600., 32400., 34200.])
    for i, istp in enumerate((17, 18, 19)):
        expected = binding.clock_at_istp(istp)
        assert packed.source_clock.julian_diff[i] == expected.julian_diff
    assert packed.temp_air.shape == (3, 3)
    np.testing.assert_array_equal(packed.precip_rain, np.ones((3, 3)) * 1800.)
