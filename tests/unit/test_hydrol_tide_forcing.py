"""Bounded source text/stream-position checks; no complete model execution."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from jax_orchidee.driver.source_clock import SourceClockBinding, SourceClockRuntime
from jax_orchidee.driver.tides import (
    PreparedTideForcing,
    TideForcingError,
    bind_tide_forcing_run,
    parse_fortran_f10_3_record,
    prepare_tide_forcing,
    tide_forcing_for_local_tstep,
    tide_forcing_for_source_istp,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/orchidee_man_250919.yaml"
REFERENCE = json.loads((ROOT / "tests/fixtures/hydrol_tide_boundary.json").read_text())


@pytest.fixture
def compact_config(tmp_path):
    """Versioned source records exercise actual text IO without ignored data."""
    fixture = json.loads((ROOT / "tests/fixtures/hydrol_driver_boundary.json").read_text())
    mapping = {}
    for name, records in (("positive_file", fixture["positive_records"]),
                          ("differential_file", fixture["differential_records"])):
        path = tmp_path / ("WTM_forcing_pos.txt" if name == "positive_file" else "WTM_forcing_diff.txt")
        path.write_text("\n".join(records) + "\n", encoding="ascii")
        mapping[name] = str(path)
    config = tmp_path / "case.yaml"
    config.write_text(yaml.safe_dump({"drivers": {"water_table_text": mapping}}), encoding="utf8")
    return config


def _setup(run_id="native-1961", first=1):
    binding = SourceClockBinding(run_id, "noleap", 2437300.0, 1800.0, 2437300, 0.0, first - 1)
    return SimpleNamespace(binding=binding, time_control=SimpleNamespace(istp=first))


def _small_prepared():
    return PreparedTideForcing(np.asarray([308.33, 329.0, -2.0, 0.0]),
                               np.asarray([44.83, 20.67, -12.0, 2.0]), "positive", "diff")


@pytest.mark.parametrize("row", REFERENCE["rows"], ids=lambda row: repr(row["records"]))
def test_f10_3_matches_original_fortran_read_bitwise(row):
    for record, bits in zip(row["records"], row["float64_bits"], strict=True):
        actual = parse_fortran_f10_3_record(record)
        assert format(int(actual.view(np.uint64)), "016X") == bits


def test_prepared_actual_assets_preserve_independent_first_differential():
    if not all((ROOT / "data/forcing_VN" / name).is_file() for name in ("WTM_forcing_pos.txt", "WTM_forcing_diff.txt")):
        pytest.skip("optional local source-asset identity gate: bulk paper tide files are not installed")
    values = {"WTD_FILENAME": "/legacy/forcing_VN/WTM_forcing_pos.txt",
              "wtd_filename2": "/legacy/forcing_VN/WTM_forcing_diff.txt"}
    prepared = prepare_tide_forcing(CONFIG, run_def_values=values)
    assert prepared.positive.size == 17520
    # Source formatted input retains the extra terminal blank record as zero.
    assert prepared.differential.size == 17521
    assert prepared.differential[-1] == 0.0
    assert prepared is prepare_tide_forcing(CONFIG, run_def_values=values)
    run = bind_tide_forcing_run(prepared, _setup())
    assert tide_forcing_for_source_istp(run, 1) == (308.33, 44.83)
    assert tide_forcing_for_source_istp(run, 2) == (329.0, 20.67)
    # No invented initial predecessor, conversion, clipping or recomputed diff.
    assert tide_forcing_for_source_istp(run, 1).dwtp_tide != 0.0
    assert prepared.differential[2] != prepared.positive[2] - prepared.positive[1]


def test_versioned_prefix_prepare_is_mandatory_and_preserves_formatted_records(compact_config):
    values = {"wtd_filename": "/paper/WTM_forcing_pos.txt", "wtd_filename2": "/paper/WTM_forcing_diff.txt"}
    prepared = prepare_tide_forcing(compact_config, run_def_values=values)
    assert prepared is prepare_tide_forcing(compact_config, run_def_values=values)
    assert prepared.positive.size == 4 and prepared.differential.size == 5
    assert prepared.differential[-1] == 0.0
    run = bind_tide_forcing_run(prepared, _setup())
    assert tide_forcing_for_source_istp(run, 1) == (308.33, 44.83)
    assert tide_forcing_for_source_istp(run, 2) == (329.0, 20.67)
    with pytest.raises(TideForcingError, match="outside available"):
        tide_forcing_for_source_istp(run, 5)


def test_prepared_arrays_cannot_be_mutated_or_write_enabled():
    prepared = _small_prepared()
    with pytest.raises(ValueError):
        prepared.positive[0] = 0
    with pytest.raises(ValueError):
        prepared.positive.setflags(write=True)


def test_historical_used_log_omission_uses_explicit_originating_run_def():
    from jax_orchidee.driver.init import parse_run_def
    directory = ROOT / "outputs/server_1961_trace_full_20260623/run"
    required = [directory / "used_run.def", directory / "run.def",
                ROOT / "data/forcing_VN/WTM_forcing_pos.txt", ROOT / "data/forcing_VN/WTM_forcing_diff.txt"]
    if not all(path.is_file() for path in required):
        pytest.skip("optional historical source-identity gate: original run log/source or bulk tide files are unavailable")
    effective = parse_run_def(directory / "used_run.def")
    assert effective["wtd_filename"] == effective["wtd_filename2"] == ""
    with pytest.raises(TideForcingError, match="missing declared"):
        prepare_tide_forcing(CONFIG, run_def_values=effective)
    prepared = prepare_tide_forcing(CONFIG, run_def_values=effective,
                                    source_run_def_path=directory / "run.def")
    assert (prepared.positive[0], prepared.differential[0]) == (308.33, 44.83)


@pytest.mark.parametrize("source_text", ["", "wtd_filename=wrong.txt\nwtd_filename2=WTM_forcing_diff.txt\n",
    "wtd_filename=WTM_forcing_pos.txt\nWTD_FILENAME=wrong.txt\nwtd_filename2=WTM_forcing_diff.txt\n"])
def test_originating_run_def_must_supply_unambiguous_matching_declarations(tmp_path, source_text):
    source = tmp_path / "run.def"
    source.write_text(source_text)
    with pytest.raises(TideForcingError):
        prepare_tide_forcing(CONFIG, run_def_values={"wtd_filename": "", "wtd_filename2": ""},
                            source_run_def_path=source)


def test_missing_originating_source_and_nonempty_effective_mismatch_are_rejected(tmp_path):
    with pytest.raises(TideForcingError, match="unavailable"):
        prepare_tide_forcing(CONFIG, run_def_values={}, source_run_def_path=tmp_path / "missing.def")
    source = tmp_path / "run.def"
    source.write_text("wtd_filename=WTM_forcing_pos.txt\nwtd_filename2=WTM_forcing_diff.txt\n")
    with pytest.raises(TideForcingError, match="disagrees"):
        prepare_tide_forcing(CONFIG, run_def_values={"wtd_filename": "wrong.txt"}, source_run_def_path=source)


def test_stream_index_follows_source_frame_across_local_label_rebase():
    setup = _setup(first=73)  # e.g. nonzero TIME_SKIP: OPEN still starts at row0.
    run = bind_tide_forcing_run(_small_prepared(), setup)
    runtime = SourceClockRuntime(setup.binding, np.int64(74), np.int64(0), np.int64(75))
    assert tide_forcing_for_local_tstep(run, runtime, 0) == (-2.0, -12.0)
    assert tide_forcing_for_local_tstep(run, runtime, 1) == (0.0, 2.0)
    with pytest.raises(TideForcingError, match="outside available"):
        tide_forcing_for_local_tstep(run, runtime, 2)


def test_continuous_year_boundary_does_not_reopen_but_new_native_restart_does():
    prepared = _small_prepared()
    setup = _setup()
    run = bind_tide_forcing_run(prepared, setup)
    continuous = SimpleNamespace(binding=setup.binding, time_control=SimpleNamespace(istp=3))
    assert bind_tide_forcing_run(prepared, continuous, previous=run) is run
    assert tide_forcing_for_source_istp(run, 3) == (-2.0, -12.0)
    restarted = bind_tide_forcing_run(prepared, _setup("native-1962", first=17521), previous=run)
    assert tide_forcing_for_source_istp(restarted, 17521) == (308.33, 44.83)
    # A stream is finite; no automatic modulo to hide exhausted input.
    with pytest.raises(TideForcingError, match="outside available"):
        tide_forcing_for_source_istp(run, 17521)


def test_same_run_changed_streams_are_rejected():
    prepared = _small_prepared()
    setup = _setup()
    run = bind_tide_forcing_run(prepared, setup)
    changed = PreparedTideForcing(prepared.positive + 1, prepared.differential, "positive", "diff")
    with pytest.raises(TideForcingError, match="changed within"):
        bind_tide_forcing_run(changed, setup, previous=run)


def test_local_frame_cannot_be_taken_from_another_native_run():
    run = bind_tide_forcing_run(_small_prepared(), _setup())
    other = _setup("another-run")
    runtime = SourceClockRuntime(other.binding, np.int64(0), np.int64(0), np.int64(1))
    with pytest.raises(TideForcingError, match="binding"):
        tide_forcing_for_local_tstep(run, runtime, 0)


@pytest.mark.parametrize("istp", [-1, 1.5, None, np.nan, [1]])
def test_invalid_source_positions_raise_the_producer_error(istp):
    run = bind_tide_forcing_run(_small_prepared(), _setup())
    with pytest.raises(TideForcingError):
        tide_forcing_for_source_istp(run, istp)


@pytest.mark.parametrize("declarations", [
    {}, {"wtd_filename": "wrong.txt", "wtd_filename2": "WTM_forcing_diff.txt"},
    {"wtd_filename": "WTM_forcing_pos.txt", "WTD_FILENAME": "different.txt", "wtd_filename2": "WTM_forcing_diff.txt"},
])
def test_explicit_local_mapping_does_not_ignore_mismatched_declared_asset(declarations):
    with pytest.raises(TideForcingError):
        prepare_tide_forcing(CONFIG, run_def_values=declarations)


@pytest.mark.parametrize("record", ["bad", "**********", "NaN", "Inf", "1e999"])
def test_invalid_or_nonfinite_input_is_rejected(record):
    with pytest.raises(TideForcingError):
        parse_fortran_f10_3_record(record)


def test_prefetched_pair_contains_only_numeric_dynamic_leaves():
    import jax
    import jax.numpy as jnp
    run = bind_tide_forcing_run(_small_prepared(), _setup())
    with jax.enable_x64():
        values = jax.tree_util.tree_map(jnp.asarray, tide_forcing_for_source_istp(run, 1))
        assert [leaf.shape for leaf in jax.tree_util.tree_leaves(values)] == [(), ()]
        # Both leaves remain dynamic arguments, rather than host reads in jit.
        compiled = jax.jit(lambda forcing: jnp.stack((forcing.wtp_tide, forcing.dwtp_tide)))
        np.testing.assert_array_equal(compiled(values), np.asarray([308.33, 44.83]))
        next_values = jax.tree_util.tree_map(jnp.asarray, tide_forcing_for_source_istp(run, 2))
        np.testing.assert_array_equal(compiled(next_values), np.asarray([329.0, 20.67]))
