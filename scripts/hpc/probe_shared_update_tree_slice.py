"""Compare leaf and cached whole-tree slicing in complete AdamW updates."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

RTOL = 1e-5
ATOL = 1e-6
SOURCE_HEAD = "c599236bcf57b745e9bafed678298010f4c3b58f"
PROCESS_STARTED = time.perf_counter()


class _CompilationAttempt(BaseException):
    pass


def interleaved_orders(repeats: int = 3) -> tuple[tuple[str, str], ...]:
    if repeats < 3:
        raise ValueError("at least three matched full-proposal pairs are required")
    return tuple(("leaf", "tree") if index % 2 == 0 else ("tree", "leaf")
                 for index in range(repeats))


def profiled_update_admitted(parity_passed: bool, compile_counts: dict[str, int]) -> bool:
    return (
        parity_passed
        and compile_counts.get("compile_or_get_cached_calls") == 0
        and compile_counts.get("backend_compile_calls") == 0
    )


def reject_compilation_before_entry(compiler, counters: dict[str, int]):
    original_lookup, original_backend = compiler.compile_or_get_cached, compiler.backend_compile

    def reject_lookup(*args, **kwargs):
        counters["compile_or_get_cached_calls"] += 1
        raise _CompilationAttempt("compile_or_get_cached")

    def reject_backend(*args, **kwargs):
        counters["backend_compile_calls"] += 1
        raise _CompilationAttempt("backend_compile")

    compiler.compile_or_get_cached, compiler.backend_compile = reject_lookup, reject_backend

    def restore():
        compiler.compile_or_get_cached, compiler.backend_compile = original_lookup, original_backend

    return restore


def _annotated_call(operation, annotation_factory, annotation_name):
    def wrapped(*args, **kwargs):
        with annotation_factory(annotation_name):
            return operation(*args, **kwargs)
    return wrapped


def synchronized_evaluator_summary(root_wall_seconds: float, timings: dict[str, list[float]]):
    if len(timings.get("daily32_micro4", ())) != 8:
        raise ValueError("synchronized evaluator diagnostic requires exactly eight daily micro4 calls")
    if len(timings.get("T7", ())) != 1 or len(timings.get("T30", ())) != 1:
        raise ValueError("synchronized evaluator diagnostic requires one T7 and one T30 call")
    components = {name: float(sum(timings[name])) for name in ("daily32_micro4", "T7", "T30")}
    evaluator_sum = float(sum(components.values()))
    residual = max(0.0, float(root_wall_seconds) - evaluator_sum)
    return {
        "evaluator_wall_seconds": components,
        "synchronized_evaluator_sum_seconds": evaluator_sum,
        "outside_evaluator_wall_seconds": residual,
        "evaluator_sum_fraction_of_root_wall": evaluator_sum / root_wall_seconds if root_wall_seconds else None,
        "outside_fraction_of_root_wall": residual / root_wall_seconds if root_wall_seconds else None,
        "root_wall_seconds": float(root_wall_seconds),
    }


def _run_synchronized_evaluator_diagnostic(
    *, jax, jnp, compiler, evaluators, make_daily_accumulator, tree_helper,
    parameters, daily, device, device_helper_cache, execute, tree_reference,
    report, _compare, _passed, output,
):
    synchronized_times = {"daily32_micro4": [], "T7": [], "T30": []}

    def synchronized_evaluator(name, evaluator):
        def run(*call_args):
            jax.block_until_ready(call_args)
            began = time.perf_counter()
            result = evaluator(*call_args)
            jax.block_until_ready(result)
            synchronized_times[name].append(time.perf_counter() - began)
            return result
        return run

    synchronized_evaluators = dict(evaluators)
    synchronized_evaluators["daily"] = synchronized_evaluator("daily32_micro4", evaluators["daily"])
    synchronized_evaluators["T7"] = synchronized_evaluator("T7", evaluators["T7"])
    synchronized_evaluators["T30"] = synchronized_evaluator("T30", evaluators["T30"])
    synchronized_accumulator = make_daily_accumulator(
        jax=jax, jnp=jnp, microbatch_size=4, evaluator=synchronized_evaluators["daily"],
        expected_count=32, slice_execution="tree", _batch_slicer=tree_helper,
    )

    # First use compiles only the accumulator's small add/finish helpers. Keep
    # that warm-up separate from the compile-free observed proposal.
    warm_counts = {"compile_or_get_cached_calls": 0, "backend_compile_calls": 0}
    old_lookup, old_backend = compiler.compile_or_get_cached, compiler.backend_compile

    def warm_lookup(*args, **kwargs):
        warm_counts["compile_or_get_cached_calls"] += 1
        return old_lookup(*args, **kwargs)

    def warm_backend(*args, **kwargs):
        warm_counts["backend_compile_calls"] += 1
        return old_backend(*args, **kwargs)

    warm_started = time.perf_counter()
    compiler.compile_or_get_cached, compiler.backend_compile = warm_lookup, warm_backend
    try:
        with jax.default_device(device), device_helper_cache(device):
            warm_daily = synchronized_accumulator(parameters, daily)
            jax.block_until_ready(warm_daily)
    finally:
        compiler.compile_or_get_cached, compiler.backend_compile = old_lookup, old_backend
    warm_seconds = time.perf_counter() - warm_started
    warm_hard = _hard_counts(warm_daily[0][1])
    if any(warm_hard.values()):
        raise ValueError(f"synchronized accumulator warm-up returned nonzero hard counts: {warm_hard}")
    for rows in synchronized_times.values():
        rows.clear()
    del warm_daily

    compile_counts = {"compile_or_get_cached_calls": 0, "backend_compile_calls": 0}
    began = time.perf_counter()
    restore_guard = reject_compilation_before_entry(compiler, compile_counts)
    observed = None
    compile_error = None
    try:
        observed = execute(
            "tree", active_accumulators={"tree": synchronized_accumulator},
            evaluator_set=synchronized_evaluators,
        )
    except _CompilationAttempt as error:
        compile_error = str(error)
    finally:
        restore_guard()
    execute_call_wall_before_host_copy = time.perf_counter() - began
    sync_wall = float(observed["wall_seconds"]) if observed is not None else None
    synchronized_result = jax.device_get(observed) if observed is not None else None
    sync_parity = _compare(jax, tree_reference, synchronized_result) if observed is not None else None
    sync_passed = _passed(jax, sync_parity, tree_reference, synchronized_result) if observed is not None else False
    compile_free = profiled_update_admitted(True, compile_counts)
    report["synchronized_evaluator_diagnostic"] = {
        "status": "passed" if sync_passed and compile_free else "rejected",
        "classification": "one_altered_synchronization_fixed_weight_proposal_not_throughput_evidence",
        "accumulator_helper_warmup_seconds": warm_seconds,
        "accumulator_helper_warmup_compile_counters": warm_counts,
        "accumulator_helper_warmup_hard_counts": warm_hard,
        "blocking_proposal": synchronized_evaluator_summary(sync_wall, synchronized_times) if observed is not None else None,
        "ordinary_untraced_tree_median_seconds": report["matched_medians_seconds"]["tree"],
        "altered_to_clean_median_wall_ratio": sync_wall/report["matched_medians_seconds"]["tree"] if sync_wall is not None else None,
        "execute_call_wall_before_host_copy_seconds": execute_call_wall_before_host_copy,
        "compile_counters": compile_counts,
        "compile_free": compile_free,
        "compile_attempt_rejected_before_compiler_entry": compile_error,
        "parity": sync_parity,
        "all_hard_counts_zero": synchronized_result["all_hard_counts_zero"] if observed is not None else None,
        "limitations": [
            "Each timer blocks on the complete saved evaluator value, auxiliary mapping, and gradient; pending input slicing is blocked before the timer starts.",
            "This changes synchronization and measures combined forward, loss, and AD execution.",
            "Outside-evaluator wall includes microbatch slicing and accumulation, physical and finite gates, mixed-gradient work, and AdamW; it is not a neural-versus-physics split.",
            "The altered synchronization proposal is not a clean throughput measurement and may shift overlap or dispatch costs.",
        ],
    }
    _atomic_write(output, report)
    if not sync_passed:
        raise ValueError("synchronized evaluator diagnostic failed full proposal parity or compile-free admission")
    del observed, synchronized_result


def _load_overlay_module(name: str, path: Path):
    path = path.resolve(strict=True)
    existing = sys.modules.get(name)
    if existing is not None and Path(existing.__file__).resolve() == path:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load diagnostic overlay: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if existing is None:
            del sys.modules[name]
        else:
            sys.modules[name] = existing
        raise
    return module


def _prepare_runtime(args: argparse.Namespace):
    if not args.inspect_only:
        from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend

        initialize_gpu_backend()
    _load_overlay_module("scripts.hpc.shared_daily_microbatch", args.slice_helper_file)
    owner = _load_overlay_module(
        "scripts.hpc.probe_shared_serialized_dual", args.support_owner_file
    )
    if getattr(args, "learning_runner_file", None) is not None:
        _load_overlay_module("scripts.hpc.run_shared_learning", args.learning_runner_file)
    return owner


def _read_identity(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    owner = _prepare_runtime(args)
    profile_bytes = args.source_profile.read_bytes()
    profile = json.loads(profile_bytes)
    reference = json.loads(Path(profile["technical_update_report"]).read_text(encoding="utf-8"))
    owner.require_reference(profile, reference)
    current_head = owner._source_head()
    if current_head != SOURCE_HEAD or reference.get("source_git_head") != SOURCE_HEAD:
        raise ValueError("probe requires the admitted immutable c599236 source checkout")
    digest = hashlib.sha256(profile_bytes).hexdigest()
    with args.input_bundle.open("rb") as handle:
        bundle = pickle.load(handle)
    owner._validate_bundle(bundle, profile_sha256=digest, reference=reference,
                           source_head=current_head)
    admissions = owner.validate_admissions(
        profile, reference, owner.admission_report_paths(args), profile_sha256=digest
    )
    return profile, reference, bundle, digest, admissions


def inspect_assets(args: argparse.Namespace) -> dict[str, Any]:
    profile, reference, bundle, digest, admissions = _read_identity(args)
    ids = tuple(bundle["artifact_ids"])
    if ids != tuple(reference["method"]["training_plan"]["training_artifact_ids"][:2]):
        raise ValueError("input bundle artifact IDs differ from the saved profile")
    if len(bundle["payloads"]) != 2 or set(bundle["payloads"][0]["cases"]) != {7, 30}:
        raise ValueError("saved bundle must contain its 742 input and T7/T30 cases")
    daily32 = bundle["payloads"][0]["daily"]
    import jax

    leaves = jax.tree_util.tree_leaves(daily32)
    if not leaves or any(np.shape(leaf)[0] != 32 for leaf in leaves):
        raise ValueError("the first saved daily artifact must have exactly 32 rows")
    for name in ("daily", "T7", "T30"):
        selected = admissions.get(name, profile["executables"][name])
        path = Path(selected["executable"])
        if not path.is_file():
            raise FileNotFoundError(f"saved {name} executable is absent: {path}")
    return {
        "status": "passed",
        "scope": "cpu_saved_full_update_assets_only",
        "source_git_head": sys.modules["scripts.hpc.probe_shared_serialized_dual"]._source_head(),
        "source_profile_sha256": digest,
        "input_bundle": str(args.input_bundle),
        "artifact_ids": ids,
        "daily32_leaf_count": len(leaves),
        "recurrent_horizons": [7, 30],
        "admitted_overrides": admissions,
        "diagnostic_support_owner": str(args.support_owner_file.resolve()),
        "slice_helper_file": str(args.slice_helper_file.resolve()),
        "model_gradient_compilations": 0,
        "gpu_executables_loaded": False,
        "optimizer_updates": 0,
        "scientific_acceptance": False,
    }


def _hard_counts(auxiliary: Any) -> dict[str, int]:
    hard = auxiliary.get("hard") if isinstance(auxiliary, dict) else None
    if not isinstance(hard, dict):
        raise ValueError("saved branch auxiliary lacks its hard-count mapping")
    return {name: int(np.asarray(value)) for name, value in hard.items()}


def _make_update(args: argparse.Namespace, profile, reference, bundle, digest, admissions):
    import jax
    import jax.numpy as jnp
    from scripts.hpc.probe_shared_execution import compare_trees

    from jax.experimental.serialize_executable import deserialize_and_load
    from scripts.hpc import run_shared_learning as default_learning
    from scripts.hpc.checked_recurrent_execution import device_helper_cache
    from scripts.hpc.probe_shared_compile_cache import _require_supported_jax
    from scripts.hpc.probe_shared_dual import _serial_mixed, make_recurrent_branch
    from scripts.hpc.probe_shared_serialized_updates import load_evaluators
    from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update

    _require_supported_jax(jax)
    from jax._src import compiler

    helper = sys.modules["scripts.hpc.shared_daily_microbatch"]
    learning = sys.modules.get("scripts.hpc.run_shared_learning", default_learning)
    execute_learning_update = learning.execute_learning_update
    make_daily_accumulator = helper.make_daily_accumulator
    make_route_accumulator = getattr(learning, "make_learning_daily_accumulator", None)
    make_daily_batch_slicer = helper.make_daily_batch_slicer
    devices = tuple(jax.local_devices())
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("full-update replay requires exactly one visible GPU")
    device = devices[0]
    checkpoint_path = Path(bundle["initial_checkpoint"])
    checkpoint = learning.load_learning_checkpoint(
        checkpoint_path, expected_method=reference["method"]
    )
    if checkpoint["completed_updates"] != 0:
        raise ValueError("full-update admission must start from the retained update-0 state")
    with jax.default_device(device), device_helper_cache(device):
        parameters, optimizer = jax.device_put((checkpoint["parameters"], checkpoint["optimizer"]))
        operations = dict(learning.make_tree_operations(jax, jnp, 32))
    if not bool(np.asarray(operations["finite"]((parameters, optimizer)))):
        raise ValueError("update-0 parameters or AdamW state is nonfinite")

    started = PROCESS_STARTED
    load_events = []
    profiling = {"enabled": False}

    def annotation(name: str):
        if profiling["enabled"]:
            return jax.profiler.TraceAnnotation(name)
        return nullcontext()

    def emit(stage, **values):
        load_events.append({"stage": stage, "elapsed_seconds": time.perf_counter() - started, **values})

    selected = {name: admissions[name] for name in admissions}
    memory_limit_bytes = int(args.memory_limit_gib * 2**30)
    evaluators, executable_metadata = load_evaluators(
        profile, deserialize=deserialize_and_load, memory_sizes=learning.memory_sizes,
        memory_limit_bytes=memory_limit_bytes, emit=emit,
        overrides=selected,
    )
    daily = jax.device_put(bundle["payloads"][0]["daily"], device)
    seven_case = jax.device_put(bundle["payloads"][0]["cases"][7], device)
    thirty_case = jax.device_put(bundle["payloads"][0]["cases"][30], device)
    config = reference["method"]["daily_protocol"]
    update = make_update(config)
    report = {
        "status": "running", "scope": (
            "changing_weight_daily32_T7_T30_full_adamw_updates" if args.changing_updates
            else "unchanged_daily32_T7_T30_full_adamw_proposal"
        ),
        "source_profile": str(args.source_profile), "input_bundle": str(args.input_bundle),
        "source_profile_sha256": digest, "source_git_head": SOURCE_HEAD,
        "diagnostic_support_owner": str(args.support_owner_file.resolve()),
        "slice_helper_file": str(args.slice_helper_file.resolve()),
        "learning_runner_file": (
            str(args.learning_runner_file.resolve()) if args.learning_runner_file else None
        ),
        "artifact_id": int(bundle["artifact_ids"][0]),
        "daily_rows": 32, "daily_microbatch_size": 4,
        "recurrent_horizons": [7, 30], "fixed_weight_proposals": not args.changing_updates,
        "retained_training_updates": 0, "scientific_acceptance": False,
        "model_gradient_compilations": 0, "comparison_rtol": RTOL,
        "comparison_atol": ATOL, "admitted_overrides": admissions,
        "saved_executables": executable_metadata,
        "startup_seconds": time.perf_counter() - started,
        "executable_load_seconds": float(sum(
            row["read_seconds"] + row["deserialize_seconds"]
            for row in executable_metadata.values()
        )),
        "observed_memory_limit_bytes": memory_limit_bytes,
        "peak_memory_semantics": (
            "backend peak_bytes_in_use is process-cumulative; optional daily64 shares "
            "the same process high-water and is not an isolated workload peak"
        ),
        "recurrent_execution_scope": (
            "T7 and T30 use the same saved executables, input cases and execution path "
            "in both variants; only daily row slicing changes"
        ),
        "load_events": load_events, "timings": {},
    }
    _atomic_write(args.output, report)

    slicer_started = time.perf_counter()
    tree_slicer = make_daily_batch_slicer(jax, 4)
    tree_helper = tree_slicer.lower(daily, jnp.asarray(0, dtype=jnp.int32)).compile()
    tree_helper_seconds = time.perf_counter() - slicer_started
    if args.learning_runner_file and make_route_accumulator is None:
        raise ValueError("overlaid learning runner lacks the daily accumulator integration")
    accumulators = {
        mode: (
            make_route_accumulator(
                jax=jax, jnp=jnp, microbatch_size=4, evaluator=evaluators["daily"],
                expected_count=32, slice_execution=mode,
                batch_slicer=tree_helper if mode == "tree" else None,
            ) if args.learning_runner_file else make_daily_accumulator(
                jax=jax, jnp=jnp, microbatch_size=4, evaluator=evaluators["daily"],
                expected_count=32, slice_execution=mode,
                **({"_batch_slicer": tree_helper} if mode == "tree" else {}),
            )
        ) for mode in ("leaf", "tree")
    }
    report["tree_slice_helper_compile_seconds"] = tree_helper_seconds
    report["startup_seconds"] = time.perf_counter() - started

    def execute(mode, *, daily_payload=daily,
                operation_tree=operations, active_accumulators=accumulators,
                recurrent_cases=(seven_case, thirty_case), evaluator_set=evaluators):
        captured = {}
        current_operations = dict(operation_tree)
        combine = current_operations["combine_daily_mean"]

        def capture_combine(day, seven, thirty):
            result = combine(day, seven, thirty)
            captured["gradient"] = result
            return result

        current_operations["combine_daily_mean"] = capture_combine
        evaluator_aux = {}

        def daily_accumulator(parameters_arg, samples):
            with annotation("call_daily32_micro4"):
                (value, auxiliary), gradient = active_accumulators[mode](parameters_arg, samples)
            evaluator_aux["daily"] = auxiliary
            return (value, auxiliary), gradient

        wrapped = dict(evaluator_set)
        for name in ("T7", "T30"):
            evaluator = evaluator_set[name]

            def capture(*call_args, _name=name, _evaluator=evaluator):
                with annotation(f"call_{_name}"):
                    result = _evaluator(*call_args)
                evaluator_aux[_name] = result[0][1]
                return result
            wrapped[name] = capture
        annotated_update = _annotated_call(update, annotation, "call_adamw")
        began = time.perf_counter()
        with annotation("full_update"), jax.default_device(device), device_helper_cache(device):
            proposed_parameters, proposed_optimizer, row = execute_learning_update(
                jax=jax, parameters=parameters, optimizer=optimizer,
                daily_samples=daily_payload, seven_case=recurrent_cases[0],
                thirty_case=recurrent_cases[1],
                evaluators=wrapped, operations=current_operations, update=annotated_update,
                daily_accumulator=daily_accumulator,
            )
            jax.block_until_ready((captured["gradient"], proposed_parameters, proposed_optimizer))
        wall = time.perf_counter() - began
        hard = {name: _hard_counts(value) for name, value in evaluator_aux.items()}
        all_hard_zero = all(value == 0 for record in hard.values() for value in record.values())
        if not all_hard_zero:
            raise ValueError(f"{mode} route returned nonzero physical checks")
        memory = device.memory_stats() or {}
        observed = memory.get("peak_bytes_in_use", memory.get("bytes_in_use"))
        if observed is None:
            raise RuntimeError("backend did not report live or peak device memory")
        if int(observed) > memory_limit_bytes:
            raise MemoryError(
                f"observed device memory {int(observed)} exceeds the "
                f"{args.memory_limit_gib:g} GiB admission limit"
            )
        return {
            "gradient": captured["gradient"], "parameters": proposed_parameters,
            "optimizer": proposed_optimizer, "update": row,
            "auxiliary": evaluator_aux, "hard": hard,
            "all_hard_counts_zero": all_hard_zero, "wall_seconds": wall,
            "device_memory": memory,
        }

    # Compile accumulation helpers and perform the same full proposal on each route before timing.
    warm = {}
    for mode in ("leaf", "tree"):
        device_result = execute(mode)
        warm[mode] = jax.device_get(device_result)
        del device_result
    comparisons = _compare(jax, warm["leaf"], warm["tree"])
    if not _passed(jax, comparisons, warm["leaf"], warm["tree"]):
        raise ValueError("warm full-proposal leaf/tree admission parity failed")
    report["warm_parity"] = comparisons
    del comparisons

    # Independently check that the captured runner gradient is the serial mixed branch result.
    recurrent = make_recurrent_branch(
        jax=jax, device=device, evaluators={7: evaluators["T7"], 30: evaluators["T30"]},
        operations=operations,
    )
    for mode in ("leaf", "tree"):
        def daily_branch(parameters_arg, payload):
            began = time.perf_counter()
            (value, auxiliary), gradient = accumulators[mode](parameters_arg, payload)
            return {"gradient": gradient, "loss": float(value), "hard": _hard_counts(auxiliary),
                    "seconds": time.perf_counter() - began}

        serial_gradient, _details = _serial_mixed(
            parameters=parameters, daily_payload=daily, recurrent_payload={7: seven_case, 30: thirty_case},
            daily_branch=daily_branch, recurrent_branch=recurrent,
            combine=operations["combine_daily_mean"],
        )
        check = compare_trees(jax, warm[mode]["gradient"], serial_gradient, rtol=RTOL, atol=ATOL)
        if not check["passed"]:
            raise ValueError(f"{mode} captured update gradient differs from _serial_mixed")
        report.setdefault("serial_gradient_contract", {})[mode] = check
        del serial_gradient, _details, check
    del warm

    timed = {"leaf": [], "tree": []}
    changing_steps = []
    tree_reference = None
    for pair_index, order in enumerate(interleaved_orders(args.repeats), start=1):
        prior_step = int(np.asarray(optimizer.step))
        results = {}
        accepted_device_result = None
        for mode in order:
            device_result = execute(mode)
            results[mode] = jax.device_get(device_result)
            if args.changing_updates and mode == "tree":
                accepted_device_result = device_result
            else:
                del device_result
        parity = _compare(jax, results["leaf"], results["tree"])
        passed = _passed(jax, parity, results["leaf"], results["tree"])
        if args.changing_updates and passed:
            next_step = int(np.asarray(results["tree"]["optimizer"].step))
            parameter_delta = operations["norm"](jax.tree_util.tree_map(
                lambda next_value, previous: next_value - previous,
                accepted_device_result["parameters"], parameters,
            ))
            delta_norm = float(np.asarray(parameter_delta))
            if next_step != prior_step + 1 or not np.isfinite(delta_norm) or delta_norm <= 0:
                raise ValueError("changing-weight pair did not advance AdamW and parameters")
            changing_steps.append({
                "pair": pair_index, "input_optimizer_step": prior_step,
                "output_optimizer_step": next_step,
                "parameter_delta_norm": delta_norm,
            })
        report["timings"][str(pair_index)] = {
            "order": list(order), "status": "passed" if passed else "rejected",
            "comparison": parity,
            "routes": {mode: _timing_record(results[mode]) for mode in ("leaf", "tree")},
        }
        timed["leaf"].append(results["leaf"]["wall_seconds"])
        timed["tree"].append(results["tree"]["wall_seconds"])
        if pair_index == len(interleaved_orders(args.repeats)) and not args.changing_updates:
            tree_reference = results["tree"]
        _atomic_write(args.output, report)
        del results, parity
        if not passed:
            raise ValueError(f"matched full-proposal pair {pair_index} failed parity")
        if args.changing_updates:
            parameters = accepted_device_result["parameters"]
            optimizer = accepted_device_result["optimizer"]
            del accepted_device_result
    if args.changing_updates:
        report["changing_update_steps"] = changing_steps
        report["retained_training_updates"] = len(changing_steps)
        report["retention_scope"] = "in_process_only_no_checkpoint_or_formal_training"
    report["device_memory"] = device.memory_stats()
    report["matched_medians_seconds"] = {
        mode: float(np.median(rows))
        for mode, rows in timed.items()
    }
    tree_rows = [report["timings"][str(index)]["routes"]["tree"]
                 for index in range(1, args.repeats + 1)]
    timing_names = ("daily_32_microbatch_4", "T7", "T30", "optimizer")
    amdahl_rows = []
    for row in tree_rows:
        full_wall = float(row["wall_seconds"])
        component_seconds = {name: float(row["branch_and_optimizer_seconds"][name])
                             for name in timing_names}
        fractions = {name: min(1.0, value / full_wall) for name, value in component_seconds.items()}
        combined_fraction = min(1.0, sum(component_seconds.values()) / full_wall)
        amdahl_rows.append({
            "full_update_wall_seconds": full_wall,
            "component_seconds": component_seconds,
            "component_wall_fractions": fractions,
            "component_if_eliminated_speed_ceiling": {
                name: (None if value >= 1.0 else 1.0 / (1.0 - value))
                for name, value in fractions.items()
            },
            "measured_component_union_seconds": sum(component_seconds.values()),
            "unattributed_residual_seconds": max(0.0, full_wall - sum(component_seconds.values())),
            "all_measured_components_if_eliminated_speed_ceiling": (
                None if combined_fraction >= 1.0 else 1.0 / (1.0 - combined_fraction)
            ),
        })
    report["amdahl_style_optimistic_ceilings"] = {
        "basis": "per untraced tree execution: disjoint runner branch and AdamW wall timings divided by that same execution's complete proposal wall",
        "per_execution_scenarios": amdahl_rows,
        "summary_median_component_wall_fraction": {
            name: float(np.median([row["component_wall_fractions"][name] for row in amdahl_rows]))
            for name in timing_names
        },
        "summary_median_component_if_eliminated_speed_ceiling": {
            name: float(np.median([row["component_if_eliminated_speed_ceiling"][name] for row in amdahl_rows]))
            for name in timing_names
        },
        "summary_median_all_measured_components_if_eliminated_speed_ceiling": float(np.median([
            row["all_measured_components_if_eliminated_speed_ceiling"] for row in amdahl_rows
        ])),
        "optimistic_only": True,
        "limitations": [
            "Each ceiling assumes the named measured component can vanish without changing other work.",
            "The runner timers are disjoint but do not account for every host cost in the full-update wall.",
            "These ceilings are not hardware bounds or forecasts of achievable speedup.",
        ],
    }
    if args.synchronize_evaluators or args.profile_trace_dir is not None:
        _run_synchronized_evaluator_diagnostic(
            jax=jax, jnp=jnp, compiler=compiler, evaluators=evaluators,
            make_daily_accumulator=make_daily_accumulator, tree_helper=tree_helper,
            parameters=parameters, daily=daily, device=device,
            device_helper_cache=device_helper_cache, execute=execute,
            tree_reference=tree_reference, report=report, _compare=_compare,
            _passed=_passed, output=args.output,
        )
    else:
        report["synchronized_evaluator_diagnostic"] = {"status": "not_requested"}
    report["profiled_update"] = {"status": "not_requested"}
    if args.profile_trace_dir is not None:
        compile_counts = {"compile_or_get_cached_calls": 0, "backend_compile_calls": 0}
        trace_dir = Path(args.profile_trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=False)
        profiling["enabled"] = True
        profile_started = time.perf_counter()
        restore_guard = reject_compilation_before_entry(compiler, compile_counts)
        profile_error = None
        try:
            with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
                profiled_device_result = execute("tree")
            trace_total_wall = time.perf_counter() - profile_started
            profiled_result = jax.device_get(profiled_device_result)
        except _CompilationAttempt as error:
            profile_error = str(error)
            trace_total_wall = time.perf_counter() - profile_started
            profiled_device_result = None
            profiled_result = None
        finally:
            restore_guard()
            profiling["enabled"] = False
        profiled_blocking_wall = float(profiled_device_result["wall_seconds"]) if profiled_device_result is not None else None
        profile_parity = _compare(jax, tree_reference, profiled_result) if profiled_result is not None else None
        profile_passed = _passed(jax, profile_parity, tree_reference, profiled_result) if profiled_result is not None else False
        profile_admitted = profiled_update_admitted(profile_passed, compile_counts)
        report["profiled_update"] = {
            "status": "passed" if profile_admitted else "rejected",
            "trace_directory": str(trace_dir),
            "trace_total_wall_seconds_including_start_stop_export": trace_total_wall,
            "traced_blocking_full_update_seconds": profiled_blocking_wall,
            "untraced_tree_median_seconds": report["matched_medians_seconds"]["tree"],
            "profiler_overhead_during_blocking_update_seconds": (
                profiled_blocking_wall - report["matched_medians_seconds"]["tree"]
            ),
            "trace_start_stop_and_export_seconds_outside_blocking_update": (
                max(0.0, trace_total_wall-profiled_blocking_wall)
                if profiled_blocking_wall is not None else None
            ),
            "compile_counters": compile_counts,
            "compile_free": profiled_update_admitted(True, compile_counts),
            "compile_attempt_rejected_before_compiler_entry": profile_error,
            "parity": profile_parity,
            "all_hard_counts_zero": profiled_result["all_hard_counts_zero"] if profiled_result is not None else None,
            "trace_files": [str(path) for path in sorted(trace_dir.rglob("*")) if path.is_file()],
            "limitations": [
                "Trace start/stop and export time is separated from the traced blocking update; use untraced medians for throughput.",
                "Call annotation intervals are call spans only; they do not enclose later scalar or root blocking gates.",
            ],
        }
        _atomic_write(args.output, report)
        if not profile_passed:
            raise ValueError("profiled full proposal failed parity or encountered a compile attempt")
        if not profiled_update_admitted(True, compile_counts):
            raise ValueError(f"profiled warmed update triggered compilation: {compile_counts}")
        del profiled_device_result, profiled_result
    report["allocated_gpu_card_seconds"] = time.perf_counter() - PROCESS_STARTED
    report["warm_execution_card_seconds"] = float(sum(sum(rows) for rows in timed.values()))
    report["daily64_comparison"] = {
        "status": "not_run", "reason": "separate_non_adopted_workload_requires_explicit_flag"
    }
    if args.include_daily64:
        daily64 = jax.tree_util.tree_map(
            lambda left, right: jnp.concatenate((left, right), axis=0),
            jax.device_put(bundle["payloads"][0]["daily"], device),
            jax.device_put(bundle["payloads"][1]["daily"], device),
        )
        operations64 = dict(learning.make_tree_operations(jax, jnp, 64))
        tree_slicer64 = make_daily_batch_slicer(jax, 4)
        helper_started = time.perf_counter()
        tree_helper64 = tree_slicer64.lower(daily64, jnp.asarray(0, dtype=jnp.int32)).compile()
        helper64_seconds = time.perf_counter() - helper_started
        accumulators64 = {
            mode: make_daily_accumulator(
                jax=jax, jnp=jnp, microbatch_size=4, evaluator=evaluators["daily"],
                expected_count=64, slice_execution=mode,
                **({"_batch_slicer": tree_helper64} if mode == "tree" else {}),
            )
            for mode in ("leaf", "tree")
        }
        separate_rows = {"leaf": [], "tree": []}
        separate_pairs = []
        for pair_index, order in enumerate(interleaved_orders(args.repeats), start=1):
            results = {}
            for mode in order:
                device_result = execute(
                    mode, daily_payload=daily64,
                    operation_tree=operations64, active_accumulators=accumulators64,
                )
                results[mode] = jax.device_get(device_result)
                separate_rows[mode].append(results[mode]["wall_seconds"])
                del device_result
            parity = _compare(jax, results["leaf"], results["tree"])
            passed = _passed(jax, parity, results["leaf"], results["tree"])
            separate_pairs.append({
                "pair": pair_index, "order": list(order),
                "status": "passed" if passed else "rejected", "comparison": parity,
                "routes": {mode: _timing_record(results[mode]) for mode in ("leaf", "tree")},
            })
            del results, parity
            if not passed:
                raise ValueError(f"separate daily64 proposal pair {pair_index} failed parity")
        report["daily64_comparison"] = {
            "status": "passed", "classification": "separate_non_adopted_fixed_weight_workload",
            "daily_rows": 64, "daily_artifact_ids": list(bundle["artifact_ids"]),
            "recurrent_case_artifact_id": int(bundle["artifact_ids"][0]),
            "tree_slice_helper_compile_seconds": helper64_seconds,
            "pairs": separate_pairs,
            "matched_medians_seconds": {
                mode: float(np.median(rows))
                for mode, rows in separate_rows.items()
            },
            "production_adoption": False,
            "peak_memory_semantics": "same-process cumulative high-water, not isolated daily64 peak",
        }
    report.update(status="passed", completed_pairs=len(timed["leaf"]))
    _atomic_write(args.output, report)
    return report


def _compare(jax, left, right):
    from scripts.hpc.probe_shared_execution import compare_trees

    return {name: compare_trees(jax, left[name], right[name], rtol=RTOL, atol=ATOL)
            for name in ("gradient", "auxiliary", "parameters", "optimizer")}


def _passed(jax, comparisons, left, right):
    from scripts.hpc.probe_shared_execution import compare_trees

    scalars = compare_trees(
        jax, _numerics(left["update"]), _numerics(right["update"]), rtol=RTOL, atol=ATOL
    )
    comparisons["update_scalars"] = scalars
    return (all(item["passed"] for item in comparisons.values())
            and left["all_hard_counts_zero"] and right["all_hard_counts_zero"]
            and left["hard"] == right["hard"])


def _numerics(row):
    return {key: row[key] for key in (
        "loss", "gradient_norms", "optimizer_gradient_norm", "clip_scale"
    )}


def _timing_record(result):
    return {
        "wall_seconds": result["wall_seconds"],
        "branch_and_optimizer_seconds": result["update"]["timing_seconds"],
        "hard_counts": result["hard"],
        "all_hard_counts_zero": result["all_hard_counts_zero"],
        "device_memory": result["device_memory"],
    }


def _atomic_write(path: Path, value):
    sys.modules["scripts.hpc.probe_shared_serialized_dual"]._atomic_json(path, value)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-profile", type=Path, required=True)
    result.add_argument("--input-bundle", type=Path, required=True)
    result.add_argument("--slice-helper-file", type=Path, required=True)
    result.add_argument("--support-owner-file", type=Path, required=True)
    result.add_argument("--learning-runner-file", type=Path)
    result.add_argument("--daily-admission-report", type=Path, required=True)
    result.add_argument("--seven-admission-report", type=Path, required=True)
    result.add_argument("--thirty-admission-report", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--memory-limit-gib", type=float, default=24.0)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--inspect-only", action="store_true")
    result.add_argument("--include-daily64", action="store_true")
    result.add_argument("--changing-updates", action="store_true")
    result.add_argument("--profile-trace-dir", type=Path)
    result.add_argument("--synchronize-evaluators", action="store_true")
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    if args.repeats < 3:
        raise SystemExit("--repeats must be at least 3")
    if args.changing_updates and (args.include_daily64 or args.profile_trace_dir or args.synchronize_evaluators):
        raise SystemExit("changing updates cannot be combined with daily64 or profiling diagnostics")
    if args.inspect_only:
        result = inspect_assets(args)
        _atomic_write(args.output, result)
    else:
        result = _make_update(args, *_read_identity(args))
    raise SystemExit(0 if result["status"] == "passed" else 1)
