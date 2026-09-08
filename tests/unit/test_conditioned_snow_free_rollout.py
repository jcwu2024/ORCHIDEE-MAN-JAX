"""Exercise the complete screen/report loop with production HardCounts types."""
import json
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.types import NativeForcingInput
from scripts.hpc import audit_conditioned_snow_free_rollout as audit
from scripts.hpc.run_target_oriented_full_step_preflight import HardCounts


def test_screen_completes_with_namedtuple_hard_counts(monkeypatch):
    keys = [key for key, _, _ in audit.SNOW_ENDPOINTS]
    leaves = [NS(key=k, start=i, stop=i + 1) for i, k in enumerate(keys)]
    state = jnp.array([reference for _, reference, _ in audit.SNOW_ENDPOINTS])
    values = jnp.tile(state, (7, 1))
    micro = NS(inputs=NS(continuous_state=values[None], forcing=NativeForcingInput(
        jnp.zeros((1, 7, 2, 1)), jnp.zeros((1, 7, 2)), jnp.ones((1, 7, 2)),
        jnp.zeros((1, 7, 2), dtype=bool), jnp.ones((1, 7, 2), dtype=bool))))
    masked = NS(value=jnp.ones(7), defined=jnp.ones(7, dtype=bool))
    observations = NS(fluxes={"gpp": masked}, stocks={"agb": masked})
    trace = NS(result=NS(next_state=NS(other=values)))
    hard = HardCounts(0, 0, 0, 7, 0, 0, 0)
    bootstrap = {"context": {"runtime": {"jax": jax, "contract": NS(state_leaves=leaves)}},
                 "head": NS(forcing_layout=None), "fixed_parameters": {}, "flux_basis": ()}
    monkeypatch.setattr(audit, "CASES", ((322, 7, 5),))
    monkeypatch.setattr(audit.window, "_slice", lambda *args: micro)
    monkeypatch.setattr(audit.window, "replace_conditioned_coupled_gpp_trainable", lambda *args: {})
    monkeypatch.setattr(audit, "with_boundary", lambda b, enabled: dict(b, steps={"seven_forward": lambda *a: (1., hard)}))
    monkeypatch.setattr(audit.integration, "make_compact_trace_evaluator", lambda b: lambda *a: (trace, observations))
    monkeypatch.setattr(audit, "make_trace_statistics", lambda *a: lambda *b: (1., observations))
    monkeypatch.setattr(audit, "science_summary", lambda *a: {"total": 1.})
    monkeypatch.setattr(audit, "warm_snow_free_regime", lambda *a: np.ones(7, dtype=bool))
    # Real reports contain plain JSON values; JAX synchronization has no work in this fixture.
    monkeypatch.setattr(jax, "block_until_ready", lambda value: value)
    emitted = []
    report = audit.experiment(bootstrap, {}, {322: NS(batch=NS(seven_day=None))},
                             lambda stage, row: emitted.append(stage))
    json.dumps(report, allow_nan=False)
    assert report["all_repair_forward_gates_pass"]
    assert len(report["cases"]) == 2
    assert all(row["hard_failure_count"] == 0 for row in report["cases"])
    assert report["cases"][0]["hard"]["declared_dynamic_state_mask_mismatch"] == 7
    assert emitted.count("case_completed") == 2
