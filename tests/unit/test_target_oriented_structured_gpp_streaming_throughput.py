from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    ErrorSufficientStatistics,
)
from scripts.hpc.run_target_oriented_structured_gpp_streaming_throughput import (
    BATCH_SIZES,
    MAXIMUM_BATCH_SIZE,
    DirectGppBatch,
    _direct_statistics_and_gradient,
    _round_robin,
)


def test_structured_gpp_streaming_round_robins_point_year_rows():
    values = (("a0", "a1"), ("b0", "b1"), ("c0", "c1"))
    assert _round_robin(values) == ("a0", "b0", "c0", "a1", "b1", "c1")


def test_structured_gpp_streaming_batch_sizes_are_static_and_divisible():
    assert BATCH_SIZES[-1] == MAXIMUM_BATCH_SIZE == 208
    assert all(value % 4 == 0 for value in BATCH_SIZES)
    assert np.all(np.diff(BATCH_SIZES) > 0)


def test_structured_gpp_streaming_uses_daily_operator_year_batch_axis():
    class Inputs(NamedTuple):
        year: object

    batch = DirectGppBatch(inputs=Inputs(year=jnp.arange(4)), process_targets={})
    statistics = ErrorSufficientStatistics(
        sum_squared=jnp.asarray(1.0),
        sum_signed=jnp.asarray(0.5),
        count=jnp.asarray(2.0),
    )
    specialist = {"weight": jnp.ones((2,))}
    total, gradient, timing = _direct_statistics_and_gradient(
        jax,
        jnp,
        specialist,
        batch,
        {
            "one_forward": lambda _specialist, _batch: statistics,
            "one_gradient": lambda _specialist, _batch, _total: (
                jnp.asarray(0.0),
                {"weight": jnp.ones((2,))},
            ),
        },
    )
    assert float(total.count) == 2.0
    assert np.array_equal(np.asarray(gradient["weight"]), np.ones((2,)))
    assert timing["total_seconds"] >= 0.0


def test_structured_gpp_streaming_launcher_is_free_node_zero_update():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "scripts/hpc/run_target_oriented_structured_gpp_streaming_throughput.sh"
    ).read_text(encoding="utf-8")
    assert "sbatch" not in launcher
    assert "target-oriented-structured-gpp-data-coverage-db9763b.json" in launcher
    assert "--selected-checkpoint" in launcher
    assert "--coverage-report" in launcher
