"""Bounded CPU input-only audit. Never constructs a head or a rollout.

Run with ORCJAX python -B. Real artifact299 evidence is limited to the four
saved scalar metadata rows; shape probes and all pool/normalizer fixtures are
explicitly synthetic. FP64 NumPy formulas are independent derivative oracles,
not alternative production precision. Finite differences evaluate those smooth
real formulas, not the discontinuous machine rounding of a float32 program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from research.daily_coarse_graining.daily_markov_contract import (
    StateLeafSpec, reconstruct_state_fields_compiled,
)
from research.daily_coarse_graining.daily_operator.input_assembly import _defined
from research.daily_coarse_graining.daily_operator.input_conditioning import standardize_continuous
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (
    _layer_norm, _masked_summary, _stable_head_l2_normalize,
)

RTOL, ATOL = 1e-5, 1e-6
DIAGNOSTICS = ROOT / "outputs/diagnostics/shared-latent-reader-20261004"


def compare(actual, expected):
    actual, expected = (np.asarray(x, np.float64) for x in (actual, expected))
    if actual.shape != expected.shape:
        raise ValueError("comparison shape drift")
    finite = np.isfinite(actual) & np.isfinite(expected)
    error = np.abs(actual - expected)
    allowance = ATOL + RTOL * np.abs(expected)
    passed = finite & (error <= allowance)
    return {"passed": bool(np.all(passed)), "checked": int(actual.size),
            "failed": int(np.count_nonzero(~passed)),
            "max_absolute_error": float(np.max(error[finite], initial=0)),
            "max_error_to_allowance": float(np.max((error / allowance)[finite], initial=0)),
            "nonfinite": int(np.count_nonzero(~finite))}


def summary_oracle(values, direction, pool, defined):
    """Population summary analytic JVP; equal-split extrema at exact ties.

    At ties this is the declared AD selection, not an ordinary real derivative.
    Zero variance follows the production primitive's selected zero tangent.
    """
    values, direction = (np.asarray(x, np.float64) for x in (values, direction))
    primal, tangent = [], []
    for selected in np.asarray(pool, bool) & np.asarray(defined, bool)[None, :]:
        if not np.any(selected):
            primal.append([0., 0., 0., 0.])
            tangent.append([0., 0., 0., 0.])
            continue
        x, dx = values[selected], direction[selected]
        mean = np.mean(x)
        sd = np.linalg.norm(x - mean) / np.sqrt(x.size)
        primal.append([mean, sd, np.min(x), np.max(x)])
        tangent.append([np.mean(dx), np.dot(x - mean, dx) / (x.size * sd) if sd else 0.,
                        np.mean(dx[x == np.min(x)]), np.mean(dx[x == np.max(x)])])
    return np.asarray(primal), np.asarray(tangent)


def layer_norm_oracle(values, direction):
    x, dx = (np.asarray(z, np.float64) for z in (values, direction))
    c = x - np.mean(x, axis=-1, keepdims=True)
    dc = dx - np.mean(dx, axis=-1, keepdims=True)
    r = np.sqrt(np.mean(c * c, axis=-1, keepdims=True) + 1e-6)
    return c / r, dc / r - c * np.mean(c * dc, axis=-1, keepdims=True) / r**3


def stable_norm_oracle(values, direction, epsilon=1e-6):
    """Algebraic real formula x/sqrt(sum(x^2)+eps^2 max(abs(x))^2)."""
    x, dx = (np.asarray(z, np.float64) for z in (values, direction))
    m = np.max(np.abs(x), axis=-1, keepdims=True)
    ties = np.abs(x) == m
    dm = np.sum(np.where(ties, np.sign(x) * dx, 0.), axis=-1, keepdims=True)
    dm /= np.sum(ties, axis=-1, keepdims=True)
    r = np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + epsilon**2 * m**2)
    safe = np.where(r > epsilon, r, 1.)
    dr = (np.sum(x * dx, axis=-1, keepdims=True) + epsilon**2 * m * dm) / safe
    y = np.where(r > epsilon, x / safe, 0.)
    dy = np.where(r > epsilon, dx / safe - x * dr / safe**2, 0.)
    return y, dy


def derivative_check(name, function, values, direction, oracle=None, fd=False):
    """Compare scalar-vmap routes and independent analytic directional oracle."""
    x, dx = jnp.asarray(values), jnp.asarray(direction)
    primal, tangent = jax.jvp(function, (x,), (dx,))
    # Each of the four samples is distinct; only sample3 has a nonzero direction.
    samples = jnp.stack([x * .7, x * 1.1, x * 1.3, x])
    directions = jnp.zeros_like(samples).at[3].set(dx)
    batched_primal, batched_tangent = jax.jvp(jax.vmap(function), (samples,), (directions,))
    cotangent = jnp.asarray(np.linspace(.25, 1.25, np.asarray(primal).size).reshape(primal.shape),
                            dtype=primal.dtype)
    _, pullback = jax.vjp(function, x)
    input_cotangent = pullback(cotangent)[0]
    lhs = np.sum(np.asarray(tangent, np.float64) * np.asarray(cotangent, np.float64))
    rhs = np.sum(np.asarray(dx, np.float64) * np.asarray(input_cotangent, np.float64))
    record = {"name": name, "fixture": "synthetic", "input_shape": list(x.shape),
              "input_dtype": str(x.dtype), "output_shape": list(primal.shape),
              "serial_vs_vmap_primal": compare(primal, batched_primal[3]),
              "serial_vs_vmap_jvp": compare(tangent, batched_tangent[3]),
              "other_sample_zero_tangent": bool(np.all(np.asarray(batched_tangent[:3]) == 0)),
              "jvp_vjp_pairing": compare(lhs, rhs)}
    if oracle is not None:
        reference, derivative = oracle(np.asarray(x), np.asarray(dx))
        record["FP64_oracle_primal"] = compare(primal, reference)
        record["FP64_oracle_jvp"] = compare(tangent, derivative)
        if fd:
            # Chosen relative displacements are safely resolvable in FP64. Do
            # not infer failure from a tiny FP32 finite difference.
            steps = [1e-3, 1e-4, 1e-5]
            record["FP64_oracle_finite_difference"] = []
            for step in steps:
                x64, dx64 = np.asarray(x, np.float64), np.asarray(dx, np.float64)
                plus = oracle(x64 + step * dx64, dx64)[0]
                minus = oracle(x64 - step * dx64, dx64)[0]
                record["FP64_oracle_finite_difference"].append({"step": step,
                    **compare((plus - minus) / (2 * step), derivative)})
    return record


def saved_normalization_rows(metadata):
    result = []
    for row in metadata["rows"]:
        value, mean, scale = (row[k] for k in ("raw_initial_value", "continuous_mean", "continuous_scale"))
        function = lambda x: standardize_continuous(x, mean, scale,
            row["continuous_tail_threshold"], coordinate_mask=row["continuous_tail_mask"])
        primal, tangent = jax.jvp(function, (jnp.asarray(value, jnp.float64),), (jnp.asarray(1., jnp.float64),))
        # Saved four rows all use the legacy branch. AD uses cast scale, not
        # the quantization derivative of casting the primal.
        if row["continuous_tail_mask"]:
            raise ValueError("saved real-coordinate audit only admits legacy rows")
        reference = 1. / float(np.float32(scale))
        ulp = abs(float(np.spacing(np.float32(primal))))
        result.append({"coordinate": row["coordinate"], "fixture": "saved_artifact299_scalar_metadata",
            "raw_value": value, "mean": mean, "scale": scale,
            "source_summary_defined": row["source_summary_defined_after_inactive_pft_mask"],
            "normalized_primal": float(primal), "raw_coordinate_jvp": float(tangent),
            "historical_primal": compare(primal, row["historical_FP32_z"]),
            "cast_scale_analytic_jvp": compare(tangent, reference),
            "FP64_unrounded_reciprocal_scale": 1. / scale,
            "normalized_FP32_one_ulp": ulp, "one_ulp_raw_equivalent": ulp * scale,
            "interpretation": "small scale amplifies raw state derivatives; does not establish batch-error cause"})
    return result


def reconstruction_check():
    # Contract is deliberately synthetic, using real total width and a real
    # PFT-axis placement pattern. No serialized prepared-data identity implied.
    leaf = StateLeafSpec("synthetic", ("pft_value",), (1, 2, 3), (1, 14, 3),
        "float64", 0, 6, "synthetic", "synthetic audit only", axis_names=("npts", "nvm", "ncarb"),
        selected_pft_indices=(0, 13))
    rest = StateLeafSpec("synthetic", ("rest",), (3848,), (3848,), "float64", 6, 3854,
                         "synthetic", "synthetic audit only")
    contract = SimpleNamespace(continuous_state_width=3854, state_leaves=(leaf, rest))
    def function(x):
        fields = reconstruct_state_fields_compiled(x, {}, contract)["synthetic"]
        return jnp.concatenate((jnp.take(fields["pft_value"], jnp.asarray([0, 13]), axis=1).reshape(-1),
                                fields["rest"]))
    x = np.linspace(-2., 2., 3854)
    dx = np.zeros_like(x)
    dx[[0, 5, 434, 2382, 2621, 200]] = 1.
    return derivative_check("canonical_PFT_inflate_select_roundtrip", function, x, dx,
                            lambda x, dx: (x, dx), fd=True)


def synthetic_checks():
    result = [reconstruction_check()]
    x = np.linspace(-2., 3., 3854, dtype=np.float32)
    dx = np.zeros_like(x)
    dx[[434, 2382, 2621, 200]] = 1.
    pool = np.zeros((4, 3854), bool)
    pool[0, :] = True
    pool[1, ::2] = True
    pool[2, [200, 434, 2382, 2621]] = True
    # Last pool is empty: finite zero primal and tangent is the contract.
    defined = np.ones(3854, bool)
    defined[11] = False
    x[11] = np.nan
    summary = lambda x: _masked_summary(pool, x, defined)[0]
    result.append(derivative_check("3854_masked_summary_smooth_and_empty", summary, x, dx,
        lambda x, dx: summary_oracle(x, dx, pool, defined), fd=True))
    tied = np.asarray([-1., -1., 2., 2., 7.], np.float32)
    direction = np.asarray([2., 0., 3., 0., 0.], np.float32)
    tie_pool = np.asarray([[1, 1, 1, 1, 0], [0, 0, 0, 0, 0]], bool)
    tie_defined = np.ones(5, bool)
    record = derivative_check("summary_extrema_ties", lambda x: _masked_summary(tie_pool, x, tie_defined)[0],
        tied, direction, lambda x, dx: summary_oracle(x, dx, tie_pool, tie_defined))
    record["real_directional_extrema"] = {"selected_AD_min_jvp": 1., "right_min_derivative": 0.,
        "selected_AD_max_jvp": 1.5, "right_max_derivative": 3.,
        "meaning": "ties are nonsmooth; equal-split AD is not an ordinary one-sided real derivative"}
    result.append(record)
    result.append(derivative_check("summary_constant_zero_variance", lambda x: _masked_summary(
        np.ones((1, 4), bool), x, np.ones(4, bool))[0], np.ones(4, np.float32),
        np.asarray([1., -1., 0., 0.], np.float32),
        lambda x, dx: summary_oracle(x, dx, np.ones((1, 4), bool), np.ones(4, bool))))
    for name, values in [("layer_norm_smooth_64", np.linspace(-2., 3., 64, dtype=np.float32)),
                         ("layer_norm_near_constant_64", np.full(64, 1., np.float32) +
                          np.arange(64, dtype=np.float32) * np.float32(2**-20))]:
        record = derivative_check(name, lambda x: _layer_norm(x, 1., 0.), values,
            np.linspace(-.01, .01, 64, dtype=np.float32), layer_norm_oracle, fd=True)
        if name == "layer_norm_near_constant_64":
            record["rounding_sensitivity"] = {
                "FP32_mean": float(jnp.mean(jnp.asarray(values))),
                "FP64_mean": float(np.mean(np.asarray(values, np.float64))),
                "variance_epsilon": 1e-6,
                "interpretation": "synthetic near-constant stress reveals mean-rounding amplification; no B1/B4 discrepancy or real head causality"}
        result.append(record)
    for name, values in [("stable_norm_smooth_64", np.linspace(-2., 3., 64, dtype=np.float32)),
                         ("stable_norm_zero_64", np.zeros(64, np.float32)),
                         ("stable_norm_max_abs_tie", np.asarray([-2., 2., .4, .5], np.float32))]:
        result.append(derivative_check(name, lambda x: _stable_head_l2_normalize(x, 1e-6),
            values, np.linspace(-.01, .01, values.size, dtype=np.float32), stable_norm_oracle,
            fd=name == "stable_norm_smooth_64"))
    # Includes opt-in selected tail, legacy coordinates and nonfinite values.
    raw = np.asarray([.048, 1e-7, np.nan, 0.], np.float64)
    scale = np.asarray([8e-12, 2e-7, 1., -1.], np.float64)
    tail = np.asarray([True, False, True, False])
    direction = np.asarray([1e-4, 1e-7, 0., 1.], np.float64)
    def tail_oracle(x, dx):
        safe = np.where(scale > 0., scale, 1.)
        z = x / safe
        magnitude = np.abs(z)
        y = np.where(tail & (magnitude > 16.), np.sign(z) * (16. + np.log1p(np.maximum(magnitude - 16., 0.))), z)
        dy = dx / safe / np.where(tail & (magnitude > 16.), 1. + np.maximum(magnitude - 16., 0.), 1.)
        return np.where(np.isfinite(x), y, 0.), np.where(np.isfinite(x), dy, 0.)
    result.append(derivative_check("selected_DOC_tail_and_legacy", lambda x: standardize_continuous(
        x, np.zeros(4), scale, 16., coordinate_mask=tail), raw, direction, tail_oracle, fd=True))
    mask = np.asarray(_defined(jnp.asarray([np.nan, np.inf, -np.inf, 0., 0.5e20,
                    np.nextafter(0.5e20, 0.)], jnp.float64)))
    result.append({"name": "canonical_defined_boundary", "fixture": "synthetic",
                   "passed": bool(np.array_equal(mask, [False, False, False, True, False, True]))})
    return result


def history_not_executed(diagnostics, reason):
    return {"mode": reason, "saved_artifact299_normalization": [], "saved_pool_tie_metadata": [],
        "rejected_complete_input_candidate": None,
        "assets": {name: {"path": str(Path(diagnostics) / name), "availability": "not_checked",
                           "executed": False} for name in (
            "T7-state-inputs-20261009.json", "T7-state-input-ties-20261009.json", "T7-state-repeats-20261009.npz")}}


def load_saved_history(diagnostics):
    """Optional external assets; missing files never imply executed evidence.

    Existing malformed files still fail clearly. No synthetic replacements are
    manufactured for saved scientific metadata or identity checks.
    """
    diagnostics = Path(diagnostics)
    history = history_not_executed(diagnostics, "optional_external_assets")
    for name, asset in history["assets"].items():
        asset["availability"] = "available" if (diagnostics / name).is_file() else "unavailable"
    metadata_path = diagnostics / "T7-state-inputs-20261009.json"
    metadata = None
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        history["saved_artifact299_normalization"] = saved_normalization_rows(metadata)
        history["assets"][metadata_path.name]["executed"] = True
    real_ties = []
    ties_path = diagnostics / "T7-state-input-ties-20261009.json"
    if ties_path.is_file():
        ties = json.loads(ties_path.read_text(encoding="utf-8"))
        for row in ties["rows"]:
            pools = row["field_summaries"] + row["axis_summaries"]
            real_ties.append({"coordinate": row["coordinate"], "pools": len(pools),
                "extrema_tied_pool_count": sum(p["minimum_tied"] or p["maximum_tied"] for p in pools),
                "coordinate_on_extrema_tie_count": sum(p["coordinate_on_minimum_tie"] or p["coordinate_on_maximum_tie"] for p in pools),
                "scope": "inherited exact saved metadata; no complete pool arrays available"})
        history["assets"][ties_path.name]["executed"] = True
    history["saved_pool_tie_metadata"] = real_ties
    candidate_path = diagnostics / "T7-state-repeats-20261009.npz"
    if candidate_path.is_file() and metadata is not None:
        with np.load(candidate_path, allow_pickle=False) as saved:
            candidate_hash = hashlib.sha256(saved["state00"][3].tobytes()).hexdigest()
        required_hash = metadata["identities"]["initial_state_sha256"]
        history["rejected_complete_input_candidate"] = {"path": str(candidate_path),
            "key_sample": "state00[3]", "actual_sha256": candidate_hash,
            "required_original_state_sha256": required_hash,
            "is_required_original_input": candidate_hash == required_hash,
            "meaning": "original-input identity comparison only; saved derivative result is not substituted"}
        history["assets"][candidate_path.name]["executed"] = True
    elif candidate_path.is_file():
        history["assets"][candidate_path.name]["not_executed_reason"] = "required original identity metadata unavailable"
    return history


def summarize_checks(checks, normalization):
    """All outcome claims derive from executed records, including empty scopes."""
    route_checks = [row for row in checks if "serial_vs_vmap_primal" in row]
    route_keys = ("serial_vs_vmap_primal", "serial_vs_vmap_jvp", "jvp_vjp_pairing")
    oracle_failures = [{"name": row["name"], "comparison": key, "failed_coordinates": row[key]["failed"]}
        for row in route_checks for key in ("FP64_oracle_primal", "FP64_oracle_jvp")
        if key in row and not row[key]["passed"]]
    fd_rows = [row for row in checks if "FP64_oracle_finite_difference" in row]
    fd_converged = [row["name"] for row in fd_rows
        if len(row["FP64_oracle_finite_difference"]) >= 2
        and all(step["passed"] for step in row["FP64_oracle_finite_difference"][-2:])]
    return {"total_checks": len(checks), "serial_vmap_pairing_cases": len(route_checks),
        "all_serial_vmap_and_pairing_checks_passed": all(row[key]["passed"] for row in route_checks for key in route_keys) if route_checks else None,
        "all_other_sample_tangents_exact_zero": all(row["other_sample_zero_tangent"] for row in route_checks) if route_checks else None,
        "saved_scalar_normalization_checks_executed": len(normalization),
        "all_saved_scalar_normalization_checks_passed": all(row[key]["passed"] for row in normalization
            for key in ("historical_primal", "cast_scale_analytic_jvp")) if normalization else None,
        "FP64_real_formula_sensitivity_failures_preserved": oracle_failures,
        "finite_difference_scope": "FP64 analytic oracle checks only",
        "finite_difference_cases_executed": len(fd_rows),
        "finite_difference_cases_converged_at_two_smallest_steps": fd_converged,
        "all_finite_difference_cases_converged_at_two_smallest_steps": len(fd_converged) == len(fd_rows) if fd_rows else None,
        "full_real_input_or_neural_model_acceptance": False}


def build_report(*, diagnostics=DIAGNOSTICS, include_history=True):
    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("input audit is CPU only")
    checks = synthetic_checks()
    history = load_saved_history(diagnostics) if include_history else history_not_executed(diagnostics, "disabled_synthetic_only")
    normalization = history["saved_artifact299_normalization"]
    summary = summarize_checks(checks, normalization)
    source = {
        "input_conditioning.py": "standardize_continuous:12-57; elementwise; FP32 legacy and selected FP64 tail then cast",
        "input_assembly.py": "assemble:149-208; FP64 reconstruction, fixed PFT selection; flatten only unbatched physical views",
        "successor_training.py": "rollout_input_from_template:161-185; validates unbatched (3854,), refreshes compact PFT views",
        "strong_baseline_architecture.py": "_masked_summary:957-973; per-pool coordinate reduction, centered variance; _layer_norm:890-894 last feature axis; _stable_head_l2_normalize:901-910 last head axis",
    }
    files = [ROOT / "research/daily_coarse_graining/daily_operator" / p for p in source]
    files += [ROOT / "research/daily_coarse_graining/daily_markov_contract.py", ROOT / "jax_orchidee/ad_primitives.py"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return {"schema_version": "T7_input_numerics_CPU_audit_v1", "status": "bounded_audit_complete",
        "precision_contract": "production unchanged: FP32 neural / FP64 physics",
        "tolerances": {"rtol": RTOL, "atol": ATOL},
        "runtime": {"jax_version": jax.__version__, "devices": [str(d) for d in jax.devices()],
                    "x64_enabled": bool(jax.config.x64_enabled), "module_root": str(ROOT)},
        "scope": "input reconstruction and elementary normalization/summary only; no real head, full context, rollout, training, GPU or remote work",
        "source_audit": source, "source_sha256": hashes,
        "graph_coverage": {"tier": "Verify source fallback", "project": "E-ORCHIDEE-MAN-JAX",
            "generation": "2026-10-10T08:46:50Z", "status": "daily runtime/worktrees excluded; direct source read"},
        "external_history": {"mode": history["mode"], "assets": history["assets"]},
        "saved_artifact299_normalization": normalization,
        "saved_pool_tie_metadata": history["saved_pool_tie_metadata"],
        "bounded_check_summary": summary,
        "metadata_inputs": [asset["path"] for asset in history["assets"].values() if asset["executed"]],
        "rejected_complete_input_candidate": history["rejected_complete_input_candidate"],
        "checks": checks,
        "findings": ["No cross-sample reduction or batch-dependent mathematical formula found in the bounded audited prefix; single-sample reshapes are under vmap.",
            "Source summary and layer norm use centered squared deviations, not E[x^2]-E[x]^2 subtraction.",
            "Exact extrema ties and zero variance have selected generalized AD tangents; scalar FD at kinks is not an ordinary derivative oracle.",
            "stable normalization real formula includes epsilon^2*max(abs(x))^2 and a zero-output norm threshold; neither boundary was certified for real head vectors."]
            + (["Executed saved scalar checks show small normalization scales amplify raw-coordinate tangents; no GPU B1/B4 cause established."] if normalization else [])
            + (["Synthetic near-constant LayerNorm had FP64 oracle failures, showing possible rounding sensitivity; no actual-model cause established."]
                if any(row["name"] == "layer_norm_near_constant_64" for row in summary["FP64_real_formula_sensitivity_failures_preserved"]) else []),
        "limitations": [f"Executed {len(normalization)} real raw/mean/scale coordinate checks; complete artifact299 input, normalization arrays and pool masks were not reconstructed.",
            "All actual-shape reconstruction/pooling and head-normalization vectors are synthetic; do not establish the original model failure cause.",
            "FP64 oracle finite differences validate analytic real formulas only; no tiny FP32 finite differences used as failure evidence.",
            "CPU checks cannot validate or exonerate changed GPU kernels, full-context projection, attention, full-state NaN, native AD or full-network gradients."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/T7-batch-numerics-20261011/input-audit.json")
    parser.add_argument("--diagnostics", type=Path, default=DIAGNOSTICS,
                        help="Optional saved metadata/NPZ directory; missing assets are reported as unexecuted")
    parser.add_argument("--synthetic-only", action="store_true", help="Do not inspect or load external historical assets")
    args = parser.parse_args()
    report = build_report(diagnostics=args.diagnostics, include_history=not args.synthetic_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"],
        "check_count": len(report["checks"]), "real_scalar_count": len(report["saved_artifact299_normalization"])}))


if __name__ == "__main__":
    main()
