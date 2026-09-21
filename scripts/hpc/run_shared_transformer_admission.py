"""Bounded fresh shared-Transformer admission, never production training."""
from __future__ import annotations

import argparse
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def shared_config(specification):
    from research.daily_coarse_graining.daily_operator.shared_transformer import SharedTransformerConfig

    return SharedTransformerConfig(
        width=specification["width"], attention_heads=specification["heads"],
        ffn_width=specification["ffn_width"], backbone_blocks=specification["backbone_blocks"],
        coordinate_blocks=specification["decoder_blocks"],
        output_chunk_size=specification["query_chunk_size"],
        decoder_execution=specification.get("decoder_execution", "gather_reuse"),
    )


def hard_ok(auxiliary):
    return all(int(value) == 0 for value in auxiliary["hard"].values())


def memory_sizes(compiled):
    memory = compiled.memory_analysis()
    sizes = {key: int(getattr(memory, key)) for key in (
        "argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")}
    sizes["estimated_live_bytes"] = (sizes["argument_size_in_bytes"] + sizes["output_size_in_bytes"]
                                     + sizes["temp_size_in_bytes"] - sizes["alias_size_in_bytes"])
    return sizes


def persist_tree(path, jax, payload):
    """Keep completed expensive results before another graph compilation."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(jax.device_get(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def run(args):
    import json

    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
    from scripts.hpc.shared_transformer_runtime import (
        load_shared_runtime,
        prepare_shared_trajectory_scales,
        select_artifact_ids,
    )

    started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    report = dict(status="running", protocol=protocol, completed_updates=0,
                  scientific_acceptance=False, checkpoint_inherited=False,
                  source_git_head=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())

    def emit(stage, **values):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **values)
        _atomic_json(args.output, report)
        print(f'shared_admission stage={stage} seconds={report["elapsed_seconds"]:.1f}', flush=True)

    def reject(reason):
        emit("stopped", status="rejected", reason=reason)
        return report

    try:
        emit("load_metadata_and_train_rows")
        manifest = json.loads(args.prepared_manifest.read_text(encoding="utf-8"))
        ids = select_artifact_ids(manifest["entries"], protocol["calibration"]["artifact_count"])
        runtime, parent, template, artifacts, metadata = load_shared_runtime(args, ids)
        jax = runtime["jax"]
        import jax.numpy as jnp

        from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
        from research.daily_coarse_graining.daily_operator.shared_process_head import build_shared_process_head
        from research.daily_coarse_graining.daily_operator.shared_training import make_shared_daily_objective
        from research.daily_coarse_graining.daily_operator.shared_trajectory_loss import FLUX_KEYS, STOCK_KEYS
        from research.daily_coarse_graining.daily_operator.shared_transformer import (
            shared_transformer_static_coordinate_indices,
        )
        from research.daily_coarse_graining.daily_operator.target_oriented_training import (
            derive_target_oriented_scales,
            initialize_target_oriented_adamw,
        )
        from scripts.hpc.run_successor_fast_owner_gpu_probe import _tree_numeric_report

        emit("calibrate_train_scales", data=metadata)
        # Calibration is CPU work even when the model test runs on a GPU.
        with jax.default_device(jax.devices("cpu")[0]):
            scales, calibration = prepare_shared_trajectory_scales(
                runtime, artifacts,
                flux_floors={key: protocol["calibration"]["flux_floor_gC_m2_day"] for key in FLUX_KEYS},
                stock_change_floors={key: protocol["calibration"]["stock_change_floor_gC_m2_day"]
                                    for key in STOCK_KEYS})
        scale_mapping = {"flux_scales": scales.flux_scales,
                         "stock_change_scales": scales.stock_change_scales}
        emit("calibration_completed", calibration=calibration, scales=scale_mapping)
        head = build_shared_process_head(parent, template, config=shared_config(protocol["architecture"]))
        state_scales = derive_target_oriented_scales(
            runtime["base_statistics"].arrays["state_delta"].scale,
            runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
            runtime["contract"], flux_basis=FLUX_KEYS)
        objective = make_shared_daily_objective(runtime, head, template, state_scales, scale_mapping)
        # Row zero from two selected TRAIN artifacts exercises reuse; this is not a learning experiment.
        samples = [jax.tree_util.tree_map(lambda value: value[0], artifact.batch.one_day)
                   for artifact in artifacts[:2]]
        report["admission_rows"] = [next(row for row in metadata["calibration_daily_rows"]
                                         if row["artifact_id"] == identifier) for identifier in ids[:2]]
        def initializer():
            return head.initialize(runtime["probe_assembled"], seed=protocol["seed"])
        emit("parameter_shapes")
        shapes = jax.eval_shape(initializer)
        counts = {key: sum(int(np.prod(value.shape)) for value in jax.tree_util.tree_leaves(tree))
                  for key, tree in shapes.items()}
        if sum(counts.values()) != protocol["architecture"]["parameter_count"]:
            raise ValueError("shared parameter count differs from the declared full-schema topology")
        report.update(parameter_count=sum(counts.values()), trainable_parameter_count=sum(counts.values()),
                      parameter_groups=counts, learned_fields=len(head.learned_schema.fields),
                      learned_scalars=sum(int(np.prod(field.shape)) for field in head.learned_schema.fields))
        coordinate_plan = shared_transformer_static_coordinate_indices(
            schema=head.learned_schema, layout=head.layout)
        report["decoder_execution"] = dict(
            mode=head.shared_config.decoder_execution,
            rematerialization=False,
            active_pft_indices=list(head.layout.active_pft_indices),
            eligible_coordinates=sum(len(value) for value in coordinate_plan.values()),
            eligible_by_field={key: len(value) for key, value in coordinate_plan.items()},
        )
        if args.shape_only:
            for sample in samples:
                jax.eval_shape(objective, shapes, sample)
            jax.eval_shape(jax.value_and_grad(objective, has_aux=True), shapes, samples[0])
            emit("completed", status="passed", classification="canonical_one_day_forward_gradient_shapes_only")
            return report

        emit("initialize_fresh_parameters")
        parameters = jax.jit(initializer)()
        jax.block_until_ready(parameters)
        from scripts.hpc.shared_state_anchor_diagnostics import (
            state_coordinate_owners,
            summarize_state_anchor,
        )

        forward = jax.jit(make_shared_daily_objective(
            runtime, head, template, state_scales, scale_mapping,
            include_state_diagnostics=True))
        owners = state_coordinate_owners(head.contract)

        def forward_report(values):
            return [(loss, {key: value for key, value in auxiliary.items()
                            if key != "state_diagnostics"}) for loss, auxiliary in values]

        emit("initial_canonical_forward")
        initial = [forward(parameters, sample) for sample in samples]
        jax.block_until_ready(initial)
        attribution = [summarize_state_anchor(auxiliary["state_diagnostics"], state_scales, owners)
                       for _loss, auxiliary in initial]
        for detail, (_loss, auxiliary) in zip(attribution, initial, strict=True):
            detail["matches_state_anchor"] = bool(np.isclose(
                detail["total"], float(auxiliary["state_change_anchor"]), rtol=1e-10, atol=1e-10))
        emit("initial_forward_completed", initial=forward_report(initial),
             state_anchor_attribution=attribution,
             forward_device_memory=runtime["devices"][0].memory_stats())
        if not all(np.isfinite(float(value)) and hard_ok(aux) for value, aux in initial):
            return reject("initial_forward_physical")
        emit("compile_complete_parameter_gradient")
        gradient_function = jax.jit(jax.value_and_grad(objective, has_aux=True))
        compiled = gradient_function.lower(parameters, samples[0]).compile()
        sizes = memory_sizes(compiled)
        emit("gradient_compiled", compiled_gradient_memory=sizes)
        if sizes["estimated_live_bytes"] > protocol["admission"]["gradient_memory_limit_gib"] * 2**30:
            return reject("compiled_gradient_memory")
        (value, auxiliary), gradient = compiled(parameters, samples[0])
        jax.block_until_ready(gradient)
        numeric = _tree_numeric_report(jax, gradient)
        groups = {key: _tree_numeric_report(jax, tree) for key, tree in gradient.items()}
        emit("gradient_completed", gradient=numeric, gradient_groups=groups)
        if not numeric["all_finite"] or numeric["l2_norm"] <= 0 or not hard_ok(auxiliary):
            return reject("full_parameter_gradient")
        if not all(group["all_finite"] and group["l2_norm"] > 0 for group in groups.values()):
            return reject("shared_and_tokenizer_gradient_participation")
        saved = args.output.with_suffix(".gradient.pkl")
        persist_tree(saved, jax, dict(parameters=parameters, gradient=gradient, value=value,
                                      protocol=protocol, source_git_head=report["source_git_head"]))
        emit("gradient_preserved", gradient_artifact=str(saved))
        direction = jax.tree_util.tree_map(lambda g: (g / numeric["l2_norm"]).astype(g.dtype), gradient)
        tangent = jax.jit(lambda p, d, sample: jax.jvp(
            lambda current: objective(current, sample)[0], (p,), (d,))[1])
        emit("directional_jvp")
        jvp = float(tangent(parameters, direction, samples[0]))
        reverse = float(sum(jnp.vdot(g.astype(jnp.float64), d.astype(jnp.float64)) for g, d in zip(
            jax.tree_util.tree_leaves(gradient), jax.tree_util.tree_leaves(direction), strict=True)))
        direction_ok = bool(np.isfinite(jvp) and np.isclose(
            jvp, reverse, rtol=protocol["admission"]["direction_rtol"],
            atol=protocol["admission"]["direction_atol"]))
        emit("direction_completed", derivative=dict(jvp=jvp, reverse_projection=reverse,
             direction="normalized_full_parameter_gradient", passed=direction_ok))
        if not direction_ok:
            return reject("jvp_reverse_agreement")
        del direction, tangent, gradient
        optimizer = initialize_target_oriented_adamw(parameters)
        update = make_update(protocol)
        updates = []
        for iteration in range(protocol["admission"]["updates"]):
            sample = samples[iteration % len(samples)]
            tick = time.perf_counter()
            (before, aux), gradients = compiled(parameters, sample)
            if not hard_ok(aux):
                return reject("update_input_physical")
            parameters, optimizer, norm, clip = update(parameters, gradients, optimizer)
            jax.block_until_ready(parameters)
            after, aux_after = forward(parameters, sample)
            jax.block_until_ready(after)
            row = dict(update=iteration + 1, loss_before=before, loss_after=after,
                       gradient_norm=norm, clip_scale=clip,
                       seconds=time.perf_counter() - tick, hard=aux_after["hard"],
                       process_day=aux_after["process_day"], flux=aux_after["flux"], stock=aux_after["stock"])
            updates.append(row)
            emit("optimizer_update_completed", completed_updates=iteration + 1, updates=updates)
            if not np.isfinite(float(after)) or not np.isfinite(float(norm)) or not hard_ok(aux_after):
                return reject("updated_forward_physical")
        checkpoint = args.output.with_suffix(".checkpoint.pkl")
        persist_tree(checkpoint, jax, dict(parameters=parameters, optimizer=optimizer,
                                          protocol=protocol, scales=scale_mapping))
        with checkpoint.open("rb") as handle:
            restored = pickle.load(handle)
        parameters = jax.device_put(restored["parameters"])
        optimizer_equal = all(np.array_equal(np.asarray(left), np.asarray(right)) for left, right in zip(
            jax.tree_util.tree_leaves(optimizer), jax.tree_util.tree_leaves(restored["optimizer"]), strict=True))
        replay = [forward(parameters, sample) for sample in samples]
        jax.block_until_ready(replay)
        if not optimizer_equal or not all(np.isfinite(float(value)) and hard_ok(aux) for value, aux in replay):
            return reject("final_cross_sample_physical")
        emit("completed", status="passed", classification="one_day_two_update_technical_admission_only",
             checkpoint=str(checkpoint), checkpoint_optimizer_roundtrip_equal=optimizer_equal,
             final=forward_report(replay),
             device_memory=runtime["devices"][0].memory_stats(),
             next_gate="changed_joint_T7_T30_loss_native_AD_admission_then_bounded_learning")
        return report
    except Exception as error:
        emit("exception", status="error", error_type=type(error).__name__, error=str(error))
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-manifest", "acceptance-report", "base-statistics", "process-statistics",
                 "fast-statistics", "prepared-manifest", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--protocol", type=Path,
                        default=ROOT / "manifests/coarse_graining/shared_transformer_admission_v1.json")
    result.add_argument("--shape-only", action="store_true")
    return result


if __name__ == "__main__":
    outcome = run(parser().parse_args())
    raise SystemExit(0 if outcome["status"] == "passed" else 1)
