"""Local gradient-training readiness gate for the daily coarse boundary.

Teacher targets are labels only.  This module intentionally keeps capture,
training, and checkpoint artifacts small enough for a Windows CPU smoke gate.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from research.daily_coarse_graining.replay_ceiling import (
    DEFAULT_CONFIG,
    DEFAULT_RUN_DEF,
    _block_until_ready,
    _load_state_cache,
)
from research.daily_coarse_graining.supervised_learnability_pilot import (
    BoundaryLeafSpec,
    BoundaryVectorSpec,
    _capture_days,
    _capture_days_compiled_blocks,
    _exact_discrete_metrics,
    _forcing_matrix,
    _make_sample,
    _safe_array,
    _state_input_vector,
    build_boundary_vector_spec,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_CONFIG = (
    ROOT / "research" / "daily_coarse_graining" / "configs" / "local_gradient_gate.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "performance"
    / "daily_coarse_graining"
    / "gradient_training_ready_gate.json"
)
DEFAULT_SAMPLE_MANIFEST = (
    ROOT
    / "outputs"
    / "performance"
    / "daily_coarse_graining"
    / "gradient_training_ready_samples.json"
)
DEFAULT_CHECKPOINT_DIR = (
    ROOT / "outputs" / "checkpoints" / "daily_coarse_graining" / "gradient_training_ready"
)
TEACHER_COMMIT = "7333b46c0b38650fb6c9250582876137831657b8"
DETERMINISTIC_DAILY_FIELDS = (
    "t2m_daily",
    "precip_daily",
    "snowfall_daily",
    "t2m_min_daily",
    "t2m_max_daily",
)
DETERMINISTIC_FINAL_FIELDS = ("t2mdiag",)


class ParameterCondition(NamedTuple):
    """Dynamic run.def-controlled parameter arrays; never JIT-static metadata."""

    vcmax25: Any
    maint_resp_slope_c: Any
    alloc_min: Any
    residence_time: Any


class ConditionedDayInput(NamedTuple):
    """Stable model input PyTree with no landpoint identifier or paper targets."""

    day_start_state: Any
    forcing_48: Any
    parameters: Any
    landpoint_physics: Any
    calendar: Any


class InputNormalization(NamedTuple):
    state_mean: Any
    state_scale: Any
    forcing_mean: Any
    forcing_scale: Any
    parameter_mean: Any
    parameter_scale: Any
    landpoint_mean: Any
    landpoint_scale: Any
    calendar_mean: Any
    calendar_scale: Any


class EncoderParameters(NamedTuple):
    state_weight: Any
    state_bias: Any
    state_position: Any
    forcing_input_weight: Any
    forcing_recurrent_weight: Any
    forcing_bias: Any
    condition_weight: Any
    condition_bias: Any
    fusion_weight: Any
    fusion_bias: Any
    latent_weight: Any
    latent_bias: Any


class GradientModelParameters(NamedTuple):
    encoder: EncoderParameters
    decoder_weights: tuple[Any, ...]
    decoder_biases: tuple[Any, ...]


class AdamState(NamedTuple):
    step: Any
    first_moment: Any
    second_moment: Any


def _normalization(values: np.ndarray, *, axis=0):
    mean = np.mean(values, axis=axis)
    scale = np.std(values, axis=axis)
    return mean, np.where(scale > 1.0e-6, scale, 1.0)


def _masked_normalization(values: np.ndarray, masks: np.ndarray):
    safe = np.where(masks, values, 0.0)
    count = np.sum(masks, axis=0)
    divisor = np.where(count > 0, count, 1)
    mean = np.sum(safe, axis=0) / divisor
    centered = np.where(masks, values - mean, 0.0)
    scale = np.sqrt(np.sum(centered * centered, axis=0) / divisor)
    return mean, np.where(scale > 1.0e-6, scale, 1.0)


def _parameter_vector(parameters: ParameterCondition):
    return jnp.concatenate(
        tuple(jnp.asarray(value, dtype=jnp.float32).reshape(-1) for value in parameters)
    )


def parameter_condition_from_context(context) -> ParameterCondition:
    values = teacher._compiled_stomate_parameter_values(context)
    return ParameterCondition(
        vcmax25=jnp.asarray(values.vcmax25, dtype=jnp.float32),
        maint_resp_slope_c=jnp.asarray(values.maint_resp_slope, dtype=jnp.float32)[:, 0],
        alloc_min=jnp.asarray(values.alloc_min, dtype=jnp.float32),
        residence_time=jnp.asarray(values.residence_time, dtype=jnp.float32),
    )


def landpoint_physics_from_context(context):
    """Return physical/static arrays without encoding a landpoint ID."""

    values = (
        context.hydrol_humcste,
        context.hydrol_throughfall_by_pft,
        context.hydrol_cwrr_ks,
        context.hydrol_zz_mm,
        context.hydrol_dz_mm,
        context.hydrol_reinf_slope,
        context.diaglev,
        context.ext_coeff_vegetfrac,
        context.sechiba_qsint,
        context.run_scalars.pref_soil_veg,
    )
    return jnp.concatenate(
        tuple(jnp.asarray(value, dtype=jnp.float32).reshape(-1) for value in values)
    )


def calendar_context(year: int, day_index: int):
    days = 366.0 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365.0
    angle = 2.0 * math.pi * (float(day_index) - 1.0) / days
    return jnp.asarray(
        [math.sin(angle), math.cos(angle), float(day_index) / days, days / 366.0],
        dtype=jnp.float32,
    )


def deterministic_boundary_fields(forcing, *, dt_sechiba: float, dt_stomate: float):
    """Compute the forcing-owned exact subset in Teacher source order."""

    temp = jnp.asarray(forcing.temp_air, dtype=jnp.float64)
    rain = jnp.asarray(forcing.precip_rain, dtype=jnp.float64)
    snow = jnp.asarray(forcing.precip_snow, dtype=jnp.float64)
    t2m = jnp.zeros_like(temp[0])
    precip = jnp.zeros_like(rain[0])
    snowfall = jnp.zeros_like(snow[0])
    tmin = jnp.full_like(temp[0], 1.0e33)
    tmax = jnp.full_like(temp[0], -1.0e33)
    dt = jnp.asarray(dt_sechiba, dtype=jnp.float64)
    day = jnp.asarray(dt_stomate, dtype=jnp.float64)
    precip_scale = jnp.asarray(86400.0 / dt_sechiba, dtype=jnp.float64)
    for index in range(int(temp.shape[0])):
        t2m = t2m + temp[index] * dt
        precip = precip + (rain[index] + snow[index]) * precip_scale * dt
        snowfall = snowfall + snow[index] * dt
        if index == int(temp.shape[0]) - 1:
            t2m = t2m / day
            precip = precip / day
            snowfall = snowfall / day
        tmin = jnp.minimum(tmin, temp[index])
        tmax = jnp.maximum(tmax, temp[index])
    return {
        "t2m_daily": t2m,
        "precip_daily": precip,
        "snowfall_daily": snowfall,
        "t2m_min_daily": tmin,
        "t2m_max_daily": tmax,
        "t2mdiag": temp[-1],
    }


def deterministic_exact_metrics(samples, *, dt_sechiba: float, dt_stomate: float):
    fields = {name: {"max_abs_error": 0.0, "exact": True} for name in (*DETERMINISTIC_DAILY_FIELDS, *DETERMINISTIC_FINAL_FIELDS)}
    for sample in samples:
        actual = deterministic_boundary_fields(
            sample.forcing, dt_sechiba=dt_sechiba, dt_stomate=dt_stomate
        )
        daily = sample.record.daily_fold.daily_fields
        final = sample.record.half_hour_transition.completed_entry_payloads[-1]
        for name in fields:
            expected = daily[name] if name in daily else final[name]
            lhs = np.asarray(actual[name])
            rhs = np.asarray(expected)
            error = float(np.max(np.abs(lhs - rhs)))
            fields[name]["max_abs_error"] = max(fields[name]["max_abs_error"], error)
            fields[name]["exact"] = fields[name]["exact"] and bool(np.array_equal(lhs, rhs))
    return {"fields": fields, "passed": all(value["exact"] for value in fields.values())}


def build_learned_boundary_spec(record) -> BoundaryVectorSpec:
    full = build_boundary_vector_spec(record)
    retained = [
        leaf
        for leaf in full.leaves
        if not (
            (leaf.family == "daily_interface" and leaf.name in DETERMINISTIC_DAILY_FIELDS)
            or (leaf.family == "final_diagnostics" and leaf.name in DETERMINISTIC_FINAL_FIELDS)
        )
    ]
    leaves = []
    cursor = 0
    for leaf in retained:
        size = int(np.prod(leaf.shape, dtype=np.int64))
        leaves.append(
            BoundaryLeafSpec(
                family=leaf.family,
                component=leaf.component,
                name=leaf.name,
                shape=leaf.shape,
                start=cursor,
                stop=cursor + size,
            )
        )
        cursor += size
    families = []
    for family in dict.fromkeys(leaf.family for leaf in leaves):
        selected = [leaf for leaf in leaves if leaf.family == family]
        families.append((family, selected[0].start, selected[-1].stop))
    return BoundaryVectorSpec(tuple(leaves), tuple(families), cursor)


def _residual_reference(sample, spec: BoundaryVectorSpec):
    values = []
    slow = sample.day_start_state.fields_by_component["slowproc_stomate_previous_step_state"]
    for leaf in spec.leaves:
        if leaf.component is not None:
            fields = sample.day_start_state.fields_by_component[leaf.component]
            value = fields.get(leaf.name, np.zeros(leaf.shape, dtype=np.float64))
        elif leaf.family == "daily_interface":
            value = np.zeros(leaf.shape, dtype=np.float64)
        elif leaf.family == "ok_leak":
            value = slow[leaf.name]
        else:
            value = sample.day_start_state.fields_by_component[
                "enerbil_previous_step_state"
            ]["temp_sol"]
        values.append(np.asarray(value, dtype=np.float64).reshape(-1))
    return np.concatenate(values)


def raw_conditioned_input(sample, spec, context, parameters, landpoint_physics):
    return ConditionedDayInput(
        day_start_state=jnp.asarray(
            _safe_array(_state_input_vector(sample.day_start_state, spec)), dtype=jnp.float32
        ),
        forcing_48=jnp.asarray(_safe_array(_forcing_matrix(sample.forcing)), dtype=jnp.float32),
        parameters=_parameter_vector(parameters),
        landpoint_physics=jnp.asarray(landpoint_physics, dtype=jnp.float32),
        calendar=calendar_context(sample.year, sample.day_index),
    )


def fit_input_normalization(inputs: Sequence[ConditionedDayInput]):
    state = np.stack([np.asarray(value.day_start_state) for value in inputs])
    forcing = np.stack([np.asarray(value.forcing_48) for value in inputs])
    parameters = np.stack([np.asarray(value.parameters) for value in inputs])
    landpoint = np.stack([np.asarray(value.landpoint_physics) for value in inputs])
    calendar = np.stack([np.asarray(value.calendar) for value in inputs])
    state_mean, state_scale = _normalization(state)
    forcing_mean, forcing_scale = _normalization(forcing.reshape((-1, forcing.shape[-1])))
    parameter_mean, parameter_scale = _normalization(parameters)
    landpoint_mean, landpoint_scale = _normalization(landpoint)
    calendar_mean, calendar_scale = _normalization(calendar)
    return InputNormalization(
        *(
            jnp.asarray(value, dtype=jnp.float32)
            for value in (
                state_mean,
                state_scale,
                forcing_mean,
                forcing_scale,
                parameter_mean,
                parameter_scale,
                landpoint_mean,
                landpoint_scale,
                calendar_mean,
                calendar_scale,
            )
        )
    )


def normalize_conditioned_input(value, normalization):
    return ConditionedDayInput(
        (value.day_start_state - normalization.state_mean) / normalization.state_scale,
        (value.forcing_48 - normalization.forcing_mean) / normalization.forcing_scale,
        (value.parameters - normalization.parameter_mean) / normalization.parameter_scale,
        (value.landpoint_physics - normalization.landpoint_mean) / normalization.landpoint_scale,
        (value.calendar - normalization.calendar_mean) / normalization.calendar_scale,
    )


def initialize_model(input_example, spec: BoundaryVectorSpec, *, seed: int):
    rng = np.random.default_rng(seed)
    chunk_size = 64
    state_width = 16
    forcing_width = 16
    condition_width = 16
    latent_width = 24
    chunks = (int(input_example.day_start_state.shape[0]) + chunk_size - 1) // chunk_size
    condition_size = int(
        input_example.parameters.size
        + input_example.landpoint_physics.size
        + input_example.calendar.size
    )

    def weight(shape, fan_in):
        return jnp.asarray(
            rng.normal(0.0, math.sqrt(2.0 / max(1, fan_in)), shape).astype(np.float32)
        )

    def bias(size):
        return jnp.zeros((size,), dtype=jnp.float32)

    encoder = EncoderParameters(
        state_weight=weight((chunk_size, state_width), chunk_size),
        state_bias=bias(state_width),
        state_position=weight((chunks, state_width), state_width),
        forcing_input_weight=weight((input_example.forcing_48.shape[1], forcing_width), input_example.forcing_48.shape[1]),
        forcing_recurrent_weight=weight((forcing_width, forcing_width), forcing_width),
        forcing_bias=bias(forcing_width),
        condition_weight=weight((condition_size, condition_width), condition_size),
        condition_bias=bias(condition_width),
        fusion_weight=weight((2 * state_width + 2 * forcing_width + condition_width, 48), 2 * state_width + 2 * forcing_width + condition_width),
        fusion_bias=bias(48),
        latent_weight=weight((48, latent_width), 48),
        latent_bias=bias(latent_width),
    )
    decoder_weights = tuple(
        weight((latent_width, stop - start), latent_width) * 0.05
        for _, start, stop in spec.family_ranges
    )
    decoder_biases = tuple(bias(stop - start) for _, start, stop in spec.family_ranges)
    return GradientModelParameters(encoder, decoder_weights, decoder_biases)


def encode_day(parameters: EncoderParameters, value: ConditionedDayInput):
    chunk_size = int(parameters.state_weight.shape[0])
    padded_size = int(parameters.state_position.shape[0]) * chunk_size
    padded = jnp.pad(value.day_start_state, (0, padded_size - value.day_start_state.shape[0]))
    tokens = jnp.tanh(
        padded.reshape((-1, chunk_size)) @ parameters.state_weight
        + parameters.state_bias
        + parameters.state_position
    )
    state_latent = jnp.concatenate((jnp.mean(tokens, axis=0), jnp.max(tokens, axis=0)))

    def forcing_step(carry, forcing_value):
        hidden = jnp.tanh(
            forcing_value @ parameters.forcing_input_weight
            + carry @ parameters.forcing_recurrent_weight
            + parameters.forcing_bias
        )
        return hidden, hidden

    initial = jnp.zeros_like(parameters.forcing_bias)
    last, sequence = jax.lax.scan(forcing_step, initial, value.forcing_48)
    forcing_latent = jnp.concatenate((last, jnp.mean(sequence, axis=0)))
    condition = jnp.concatenate(
        (value.parameters, value.landpoint_physics, value.calendar)
    )
    condition_latent = jnp.tanh(
        condition @ parameters.condition_weight + parameters.condition_bias
    )
    fused = jnp.concatenate((state_latent, forcing_latent, condition_latent))
    hidden = jnp.tanh(fused @ parameters.fusion_weight + parameters.fusion_bias)
    return jnp.tanh(hidden @ parameters.latent_weight + parameters.latent_bias)


def model_apply(parameters: GradientModelParameters, value: ConditionedDayInput):
    latent = encode_day(parameters.encoder, value)
    return jnp.concatenate(
        tuple(
            latent @ weight + bias
            for weight, bias in zip(
                parameters.decoder_weights, parameters.decoder_biases, strict=True
            )
        )
    )


def _stack_inputs(inputs):
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *inputs)


def _family_weights(spec: BoundaryVectorSpec):
    families = len(spec.family_ranges)
    values = np.zeros((spec.total_size,), dtype=np.float32)
    for _, start, stop in spec.family_ranges:
        values[start:stop] = 1.0 / (families * (stop - start))
    return jnp.asarray(values)


def _loss(parameters, inputs, targets, masks, weights):
    predictions = jax.vmap(model_apply, in_axes=(None, 0))(parameters, inputs)
    error = jnp.where(masks, predictions - targets, 0.0)
    return jnp.mean(jnp.sum(error * error * weights[None, :], axis=1))


def _adam_init(parameters):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    return AdamState(jnp.asarray(0, dtype=jnp.int32), zeros, zeros)


def _adam_update(parameters, gradients, state, *, learning_rate, beta1=0.9, beta2=0.999, epsilon=1.0e-8):
    step = state.step + 1
    first = jax.tree_util.tree_map(
        lambda old, grad: beta1 * old + (1.0 - beta1) * grad,
        state.first_moment,
        gradients,
    )
    second = jax.tree_util.tree_map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * grad * grad,
        state.second_moment,
        gradients,
    )
    first_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta1**step), first)
    second_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta2**step), second)
    updated = jax.tree_util.tree_map(
        lambda value, m, v: value - learning_rate * m / (jnp.sqrt(v) + epsilon),
        parameters,
        first_hat,
        second_hat,
    )
    return updated, AdamState(step, first, second)


def _tree_l2(tree):
    return float(
        np.sqrt(
            sum(float(np.sum(np.asarray(value, dtype=np.float64) ** 2)) for value in jax.tree_util.tree_leaves(tree))
        )
    )


def _family_metrics(predictions, targets, masks, spec):
    result = {}
    for family, start, stop in spec.family_ranges:
        error = predictions[:, start:stop] - targets[:, start:stop]
        selected = error[masks[:, start:stop]]
        result[family] = {
            "normalized_rmse": float(np.sqrt(np.mean(selected * selected))),
            "normalized_mae": float(np.mean(np.abs(selected))),
            "elements": stop - start,
            "evaluated_values": int(selected.size),
        }
    error = predictions - targets
    selected = error[masks]
    result["overall"] = {
        "normalized_rmse": float(np.sqrt(np.mean(selected * selected))),
        "normalized_mae": float(np.mean(np.abs(selected))),
        "elements": spec.total_size,
        "evaluated_values": int(selected.size),
    }
    return result


def train_model(inputs, targets, masks, spec, config, *, seed, resume_payload=None):
    stacked = _stack_inputs(inputs)
    targets = jnp.asarray(targets, dtype=jnp.float32)
    masks = jnp.asarray(masks, dtype=bool)
    weights = _family_weights(spec)
    if resume_payload is None:
        parameters = initialize_model(inputs[0], spec, seed=seed)
        optimizer = _adam_init(parameters)
    else:
        parameters = resume_payload["parameters"]
        optimizer = resume_payload["optimizer"]
    initial_parameters = parameters
    value_and_grad = jax.jit(jax.value_and_grad(_loss))

    initial_loss, initial_gradient = value_and_grad(parameters, stacked, targets, masks, weights)
    _block_until_ready((initial_loss, initial_gradient))
    history = [{"step": int(np.asarray(optimizer.step)), "loss": float(initial_loss)}]
    started = time.perf_counter()
    steps = int(config["steps"])
    log_every = int(config["log_every"])
    learning_rate = float(config["learning_rate"])
    for _ in range(steps):
        loss, gradients = value_and_grad(parameters, stacked, targets, masks, weights)
        parameters, optimizer = _adam_update(
            parameters, gradients, optimizer, learning_rate=learning_rate
        )
        if int(np.asarray(optimizer.step)) % log_every == 0:
            _block_until_ready((loss, parameters))
            history.append({"step": int(np.asarray(optimizer.step)), "loss": float(loss)})
    final_loss = _loss(parameters, stacked, targets, masks, weights)
    predictions = jax.vmap(model_apply, in_axes=(None, 0))(parameters, stacked)
    _block_until_ready((final_loss, predictions))
    training_seconds = time.perf_counter() - started
    encoder_change = _tree_l2(
        jax.tree_util.tree_map(
            lambda new, old: new - old, parameters.encoder, initial_parameters.encoder
        )
    )
    decoder_change = _tree_l2(
        (
            jax.tree_util.tree_map(
                lambda new, old: new - old,
                parameters.decoder_weights,
                initial_parameters.decoder_weights,
            ),
            jax.tree_util.tree_map(
                lambda new, old: new - old,
                parameters.decoder_biases,
                initial_parameters.decoder_biases,
            ),
        )
    )
    history.append({"step": int(np.asarray(optimizer.step)), "loss": float(final_loss)})
    return {
        "parameters": parameters,
        "optimizer": optimizer,
        "predictions": np.asarray(predictions),
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "initial_gradient_norm": _tree_l2(initial_gradient),
        "encoder_parameter_change_norm": encoder_change,
        "decoder_parameter_change_norm": decoder_change,
        "history": history,
        "training_seconds": training_seconds,
    }


def _parameter_count(parameters):
    return int(sum(np.asarray(value).size for value in jax.tree_util.tree_leaves(parameters)))


def _checkpoint_payload(parameters, optimizer, normalization, spec, config):
    return {
        "schema_version": "daily_coarse_gradient_checkpoint_v1",
        "teacher_commit": TEACHER_COMMIT,
        "parameters": jax.device_get(parameters),
        "optimizer": jax.device_get(optimizer),
        "normalization": jax.device_get(normalization),
        "learned_total_size": spec.total_size,
        "family_ranges": spec.family_ranges,
        "config": config,
    }


def _load_resume(path: Path | None, spec):
    if path is None:
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload["schema_version"] != "daily_coarse_gradient_checkpoint_v1":
        raise ValueError("unsupported checkpoint schema")
    if payload["teacher_commit"] != TEACHER_COMMIT or payload["learned_total_size"] != spec.total_size:
        raise ValueError("checkpoint does not match Teacher/schema contract")
    return payload


def _git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _artifact_path(path: Path) -> str:
    """Keep repository artifacts portable while allowing external output roots."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _training_readiness(
    *,
    initial_gradient_norm: float,
    encoder_parameter_change_norm: float,
    decoder_parameter_change_norm: float,
    initial_loss: float,
    final_loss: float,
    normalized_rmse: float,
    overfit_threshold: float,
    deterministic_exact_passed: bool,
    discrete_exact_passed: bool,
):
    """Separate executable training plumbing from a non-scientific fit diagnostic."""

    plumbing_ready = bool(
        initial_gradient_norm > 0.0
        and encoder_parameter_change_norm > 0.0
        and decoder_parameter_change_norm > 0.0
        and final_loss < initial_loss
        and deterministic_exact_passed
        and discrete_exact_passed
    )
    overfit_diagnostic_passed = bool(normalized_rmse <= overfit_threshold)
    return {
        "decision": (
            "training_pipeline_ready_for_gpu_smoke"
            if plumbing_ready
            else "training_pipeline_blocked_plumbing"
        ),
        "plumbing_ready": plumbing_ready,
        "overfit_diagnostic_passed": overfit_diagnostic_passed,
        "scientific_model_ready": False,
    }


def run_experiment(args):
    config = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    cache = _load_state_cache(args.state_cache)
    context = teacher.prepare_paper_1961_driver_context(
        args.teacher_config,
        used_run_def_path=args.run_def,
        reference_run_dir=args.reference_run_dir,
    )
    initial_state = teacher.rebase_driver_state_for_year_start(cache["state"])
    capture_started = time.perf_counter()
    capture = (
        _capture_days_compiled_blocks
        if args.teacher_capture_mode == "compiled-block"
        else _capture_days
    )
    capture_kwargs = {}
    if args.teacher_capture_mode == "compiled-block":
        capture_kwargs["block_size"] = args.teacher_capture_block_size
    states, forcings, records, _ = capture(
        config_path=args.teacher_config,
        context=context,
        previous_state=initial_state,
        year=args.year,
        start_day=1,
        days=int(config["train_days"]),
        **capture_kwargs,
    )
    capture_seconds = time.perf_counter() - capture_started
    full_spec = build_boundary_vector_spec(records[0])
    spec = build_learned_boundary_spec(records[0])
    samples = tuple(
        _make_sample(
            state=state,
            forcing=forcing,
            record=record,
            spec=spec,
            min_wind=context.min_wind,
        )
        for state, forcing, record in zip(states, forcings, records, strict=True)
    )
    parameters = parameter_condition_from_context(context)
    landpoint = landpoint_physics_from_context(context)
    raw_inputs = tuple(
        raw_conditioned_input(sample, spec, context, parameters, landpoint)
        for sample in samples
    )
    resume = _load_resume(args.resume, spec)
    normalization = (
        fit_input_normalization(raw_inputs)
        if resume is None
        else resume["normalization"]
    )
    inputs = tuple(normalize_conditioned_input(value, normalization) for value in raw_inputs)
    residuals = np.stack(
        [sample.target - _residual_reference(sample, spec) for sample in samples]
    )
    masks = np.stack([sample.finite_mask for sample in samples]) & np.isfinite(residuals)
    target_mean, target_scale = _masked_normalization(residuals, masks)
    targets = np.where(masks, (residuals - target_mean) / target_scale, 0.0)
    exact = deterministic_exact_metrics(
        samples,
        dt_sechiba=context.runtime.dt_sechiba,
        dt_stomate=context.runtime.dt_stomate,
    )
    if not exact["passed"]:
        raise RuntimeError("source-backed deterministic split failed pointwise exact validation")
    trained = train_model(
        inputs, targets, masks, spec, config, seed=args.seed, resume_payload=resume
    )
    metrics = _family_metrics(trained["predictions"], targets, masks, spec)
    discrete = _exact_discrete_metrics(samples, spec)
    threshold = float(config["overfit_normalized_rmse_threshold"])
    readiness = _training_readiness(
        initial_gradient_norm=trained["initial_gradient_norm"],
        encoder_parameter_change_norm=trained["encoder_parameter_change_norm"],
        decoder_parameter_change_norm=trained["decoder_parameter_change_norm"],
        initial_loss=trained["initial_loss"],
        final_loss=trained["final_loss"],
        normalized_rmse=metrics["overall"]["normalized_rmse"],
        overfit_threshold=threshold,
        deterministic_exact_passed=exact["passed"],
        discrete_exact_passed=discrete["passed"],
    )
    checkpoint_path = args.checkpoint_dir / "checkpoint.pkl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("wb") as handle:
        pickle.dump(
            _checkpoint_payload(
                trained["parameters"], trained["optimizer"], normalization, spec, config
            ),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    count = _parameter_count(trained["parameters"])
    parameter_bytes = sum(
        np.asarray(value).nbytes for value in jax.tree_util.tree_leaves(trained["parameters"])
    )
    backend = jax.default_backend()
    local_windows_cpu = platform.system() == "Windows" and backend == "cpu"
    return {
        "schema_version": "daily_coarse_gradient_training_ready_gate_v2",
        "decision": readiness["decision"],
        "readiness": {
            **readiness,
            "gpu_smoke_scope": "pipeline portability and execution only",
            "overfit_threshold_role": "nonblocking local optimization diagnostic",
        },
        "teacher_commit": TEACHER_COMMIT,
        "experiment_git_head": _git_head(),
        "constraints": {
            "local_windows_cpu_only": local_windows_cpu,
            "server_or_gpu_used": not local_windows_cpu,
            "teacher_targets_are_labels_only": True,
            "teacher_targets_as_inputs": False,
            "paper_agb_bgb_gpp_npp_as_inputs": False,
            "rollout_performed": False,
            "large_dataset_written": False,
        },
        "input_contract": {
            "pytree_type": "ConditionedDayInput",
            "state_elements": int(inputs[0].day_start_state.size),
            "forcing_shape": list(inputs[0].forcing_48.shape),
            "parameter_elements": int(inputs[0].parameters.size),
            "landpoint_physics_elements": int(inputs[0].landpoint_physics.size),
            "calendar_elements": int(inputs[0].calendar.size),
            "landpoint_id_included": False,
            "normalization_source": "train_split_only",
        },
        "output_split": {
            "before": {"elements": full_spec.total_size, "leaves": len(full_spec.leaves)},
            "learned": {"elements": spec.total_size, "leaves": len(spec.leaves)},
            "deterministic": {
                "elements": full_spec.total_size - spec.total_size,
                "leaves": len(full_spec.leaves) - len(spec.leaves),
                "fields": [*DETERMINISTIC_DAILY_FIELDS, *DETERMINISTIC_FINAL_FIELDS],
                "exact_validation": exact,
            },
        },
        "training": {
            "seed": args.seed,
            "days": int(config["train_days"]),
            "batching": "full_batch_local_gate",
            "steps": int(config["steps"]),
            "learning_rate": float(config["learning_rate"]),
            "dtype": "float32",
            "parameter_count": count,
            "parameter_bytes": parameter_bytes,
            "approx_peak_bytes": int(parameter_bytes * 4 + targets.nbytes * 4),
            "initial_gradient_norm": trained["initial_gradient_norm"],
            "encoder_parameter_change_norm": trained["encoder_parameter_change_norm"],
            "decoder_parameter_change_norm": trained["decoder_parameter_change_norm"],
            "initial_loss": trained["initial_loss"],
            "final_loss": trained["final_loss"],
            "loss_history": trained["history"],
            "overfit_threshold_normalized_rmse": threshold,
            "metrics": metrics,
            "exact_discrete": discrete,
            "checkpoint": _artifact_path(checkpoint_path),
            "checkpoint_committed": False,
            "resume_contract": "schema+Teacher commit+learned width must match",
        },
        "timing_seconds": {
            "teacher_capture": capture_seconds,
            "gradient_training": trained["training_seconds"],
        },
        "teacher_capture": {
            "mode": args.teacher_capture_mode,
            "block_size": (
                args.teacher_capture_block_size
                if args.teacher_capture_mode == "compiled-block"
                else None
            ),
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "sample_manifest": [
            {
                "year": sample.year,
                "day_index": sample.day_index,
                "input_state_tstep": sample.record.input_state_tstep,
                "parameter_condition_present": True,
                "landpoint_condition_present": True,
                "target_role": "supervision_label_only",
            }
            for sample in samples
        ],
        "limitations": [
            "Single landpoint and a fixed parameter vector test training plumbing, not parameter generalization.",
            "No holdout claim, free rollout, or scientific usability claim is made.",
            "The historical SVD pseudoinverse remains diagnostic evidence only.",
        ],
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-def", type=Path, default=DEFAULT_RUN_DEF)
    parser.add_argument("--reference-run-dir", type=Path)
    parser.add_argument(
        "--teacher-capture-mode",
        choices=("daily", "compiled-block"),
        default="daily",
    )
    parser.add_argument("--teacher-capture-block-size", type=int, default=7)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--year", type=int, default=1962)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-manifest-output", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)
    payload = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sample_payload = {
        "schema_version": "daily_coarse_gradient_samples_v1",
        "teacher_commit": payload["teacher_commit"],
        "experiment_git_head": payload["experiment_git_head"],
        "normalization_source": "train_split_only",
        "samples": payload["sample_manifest"],
        "binary_samples_committed": False,
    }
    args.sample_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.sample_manifest_output.write_text(
        json.dumps(sample_payload, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "normalized_rmse": payload["training"]["metrics"]["overall"]["normalized_rmse"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if payload["readiness"]["plumbing_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
