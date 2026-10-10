"""One B4 T7 compile; saved-only same-card full-AD comparison with serial B1.

Four distinct compatible training windows, unchanged public loss/physics,
shared candidate parameters, independent seven-day recurrences, zero updates.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np

try:
    from scripts.hpc import probe_shared_latent_reader_T7_pair as pair
except ImportError:
    # The declared immutable checkout predates this probe. Script execution
    # places the overlay directory on sys.path, where the exact pair lives.
    import probe_shared_latent_reader_T7_pair as pair
# Bind the new helper under its canonical package name from the same overlay,
# following the existing pair loader. Do not assume it exists in c599236.
batch_ad = pair.adjacent("shared_latent_T7_batch_ad")
FREE_WALL_SECONDS_CAP = 7200
PAID_PROBE_WALL_SECONDS_CAP = 10200


def parser():
    p = pair.parser()
    p.description = __doc__
    p._option_string_actions["--stage"].choices = ("cpu-readiness", "compile-and-compare", "saved-compare", "serial-saved")
    p._option_string_actions["--horizon"].choices = (7,)
    p._option_string_actions["--validate-only"].help = "compatibility alias; use only with --stage cpu-readiness"
    p.add_argument("--training-prepared-manifest", type=Path, required=True)
    p.add_argument("--dataset-manifest", type=Path, required=True)
    p.add_argument("--batch-executable", type=Path, required=True)
    p.add_argument("--cpu-readiness-report", type=Path)
    p.add_argument("--b1-memory-report", type=Path,
        help="existing exact B1 compile report; defaults to executable stem .json")
    p.add_argument("--load-seconds", type=float, default=360)
    p.add_argument("--post-compile-reserve-seconds", type=float, default=600)
    p.add_argument("--paid-ledger", type=Path,
        help="paid runner ledger binding this probe window to the current Slurm allocation")
    return p


def validate_deadline_scope(args, *, environ=None):
    environ = os.environ if environ is None else environ
    if args.paid_ledger is None:
        wall_cap = FREE_WALL_SECONDS_CAP
    else:
        ledger = json.loads(Path(args.paid_ledger).read_text(encoding="utf-8"))
        job_id = environ.get("SLURM_JOB_ID", "")
        actual_node = environ.get("SLURMD_NODENAME", "")
        expected = {
            "schema_version": "latent_T7_B4_paid_followup_v1",
            "approved": True,
            "funding_scope": "latent_reader_T7_B4_paid_followup_20261009",
            "SLURM_job_id": job_id,
            "node": actual_node,
            "execution_node": actual_node,
            "allocated_CPUs": 8,
            "GPUs": 1,
            "new_B4_graph_compile_cap": 1,
            "new_B4_compile_attempt_count": 0,
        }
        if not job_id or not actual_node:
            raise ValueError("paid probe requires an actual Slurm job and execution node")
        for key, value in expected.items():
            if type(ledger.get(key)) is not type(value) or ledger.get(key) != value:
                raise ValueError("paid probe ledger differs: " + key)
        allocation = {
            "SLURM_JOB_PARTITION": "gnall", "SLURM_CPUS_PER_TASK": "8",
            "SLURM_NTASKS": "1", "SLURM_GPUS_ON_NODE": "1",
            "SLURM_MEM_PER_NODE": "65536",
        }
        for key, value in allocation.items():
            if environ.get(key) != value:
                raise ValueError("paid probe allocation differs: " + key)
        if environ.get("SLURM_JOB_NUM_NODES", environ.get("SLURM_NNODES")) != "1":
            raise ValueError("paid probe requires one Slurm node")
        if not environ.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("paid probe requires the allocated visible GPU")
        total_wall = ledger.get("wall_seconds")
        probe_wall = ledger.get("probe_wall_seconds")
        if (type(total_wall) is not int or not 1 <= total_wall <= 10680
                or type(probe_wall) is not int or not 1 <= probe_wall <= PAID_PROBE_WALL_SECONDS_CAP
                or probe_wall > total_wall - 240 - 360):
            raise ValueError("paid probe ledger window exceeds the approved remaining allocation")
        wall_cap = probe_wall
    if (type(args.wall_seconds) not in (int, float)
            or int(args.wall_seconds) != args.wall_seconds
            or not 1 <= args.wall_seconds <= wall_cap
            or min(args.lower_seconds, args.backend_seconds, args.call_seconds,
                   args.load_seconds) <= 0 or args.pairs < 3
            or args.post_compile_reserve_seconds < 600):
        raise ValueError("bounded deadlines, >=600s compile reserve and >=3 pairs required")


def mathematical_identity(jax, base_identity, selected):
    """Exact dynamic inputs and source math; logging changes do not invalidate."""
    functions = {}
    for name in ("make_batch_objective", "make_batch_value_and_grad"):
        function = getattr(batch_ad, name)
        text = inspect.getsource(function).replace("\r\n", "\n")
        functions[name] = hashlib.sha256(text.encode()).hexdigest()
    base_identity = {k: v for k, v in base_identity.items() if k != "device_kind"}
    return json.loads(json.dumps({"schema_version": "shared_latent_T7_batch_math_v1",
        "saved_B1_mathematical_identity": base_identity, "batch_size": 4, "horizon": 7,
        "reduction": "value_and_grad(mean(vmap(public_checkpointed_objective)),argnums=(shared_parameters,B4_initial_states),has_aux=True)",
        "state_gradient_scaling": "each_sample_B1_state_gradient/4", "auxiliary": "unreduced_per_sample",
        "parameter_gradient_batch_axis": False, "sampling": selected["window_metadata"],
        "case_sha256": pair.tree_digest(jax, selected["stacked"]),
        "selector_contract": selected["contract"], "batch_math_sources": functions,
        "training_updates": 0, "optimizer_updates": 0}))


def memory_planning(args, sidecar, estimate):
    """Use saved B1 measurements for a heuristic, never a memory admission."""
    path = args.b1_memory_report or args.candidate_executable.with_name(
        args.candidate_executable.name.removesuffix(".executable.pkl")+".json")
    report = json.loads(path.read_text())
    cold = report.get("cold_measurements", {}).get("candidate", {})
    if (report.get("identities", {}).get("candidate") != sidecar["identity"]
            or not cold.get("executable")
            or Path(cold["executable"]).resolve() != args.candidate_executable.resolve()):
        raise ValueError("B1 memory metadata is not bound to this exact saved executable")
    memory = cold.get("compiled_memory", {})
    keys = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
            "alias_size_in_bytes", "estimated_live_bytes")
    if any(type(memory.get(key)) is not int or memory[key] < 0 for key in keys):
        raise ValueError("complete known B1 compiled-memory fields required")
    return {"known_B1_report": str(path), "known_B1_compiled_memory": memory,
        "heuristic_B4_workspace_bytes": 4*memory["temp_size_in_bytes"],
        "heuristic_B4_live_bytes_without_alias_credit": estimate["input_bytes"]+estimate["output_bytes"]+4*memory["temp_size_in_bytes"],
        "heuristic_formula": "B4 input arrays (one shared parameter tree) + B4 output arrays (one shared parameter gradient) + 4 * measured B1 temp bytes; no alias credit",
        "B4_workspace_linear_scaling_is_assumed_for_planning_only": True,
        "guaranteed_bound": False, "compiler_peak_estimated": False,
        "actual_post_compile_and_observed_peak_24GiB_gates_required": True}


def bounded_failure_evidence(jax, host, *, samples_per_leaf=16):
    """Complete nonfinite counts, bounded position samples; raw arrays persist."""
    records, total, nonfinite = [], 0, 0
    for path, value in jax.tree_util.tree_flatten_with_path(host)[0]:
        array = np.asarray(value)
        finite = np.isfinite(array)
        count = int(array.size-np.count_nonzero(finite))
        total += array.size
        nonfinite += count
        if not count:
            continue
        positions = []
        flat_finite = finite.reshape(-1)
        # Scan only until the bounded sample is filled. Never allocate an
        # array of all failed coordinates for a full-network NaN failure.
        for start in range(0, array.size, 8192):
            missing = samples_per_leaf-len(positions)
            if missing <= 0:
                break
            indices = np.flatnonzero(~flat_finite[start:start+8192])[:missing]+start
            positions.extend(int(index) for index in indices)
        records.append({"path": jax.tree_util.keystr(path), "shape": list(array.shape),
            "dtype": str(array.dtype), "coordinates": int(array.size),
            "finite_coordinates": int(array.size-count), "nonfinite_coordinates": count,
            "position_samples": [{"flat_index_C_order": index,
                "index": [int(item) for item in np.unravel_index(index, array.shape)],
                "value": str(array.reshape(-1)[index])} for index in positions],
            "position_samples_complete": len(positions) == count})
    return {"coordinates": int(total), "finite_coordinates": int(total-nonfinite),
        "nonfinite_coordinates": int(nonfinite), "counts_complete": True,
        "maximum_position_samples_per_leaf": samples_per_leaf,
        "nonfinite_leaves": records, "complete_values_in_preserved_raw_artifact": True}


class InsufficientBackendBudget(RuntimeError):
    """Stop before spending a backend window below the measured B1 minimum."""


def backend_budget_gate(available_seconds, previous_B1_backend_seconds):
    """Screen measured time after lowering and protecting the completion reserve."""
    if (not np.isfinite(previous_B1_backend_seconds)
            or previous_B1_backend_seconds <= 0 or not np.isfinite(available_seconds)):
        raise ValueError("finite positive measured B1 backend reference and finite available time required")
    return {"required_backend_seconds": float(previous_B1_backend_seconds),
        "available_backend_seconds": float(available_seconds),
        "minimum_additional_seconds": float(max(0.0, previous_B1_backend_seconds-available_seconds)),
        "passed": bool(available_seconds >= previous_B1_backend_seconds),
        "B4_backend_duration_guaranteed": False}


def _candidate_objective(args, owner, transformer, context, candidate_head, patches):
    from research.daily_coarse_graining.daily_operator import shared_process_head
    from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_recurrent_objective
    from jax_orchidee.stomate import carbon_kernels
    patches.enter_context(pair.install_turnover_overlay(carbon_kernels, args.turnover_overlay, pair.BASE_SHA))
    assembler = owner.load_backbone_assembler(args.backbone_adapter)
    patches.enter_context(pair.candidate_backbone_patches(shared_process_head,
        transformer.shared_transformer_raw_outputs, assembler))
    _, objective = make_shared_recurrent_objective(candidate_head, context["template"],
        context["state_scales"], context["science_scales"])
    return objective


def run(args):
    cpu = args.stage == "cpu-readiness"
    if args.validate_only and not cpu:
        raise ValueError("validate-only cannot request a GPU execution stage")
    if args.horizon != 7 or args.original_executable:
        raise ValueError("candidate-only B4 T7 scope required")
    validate_deadline_scope(args)
    if not args.candidate_executable:
        raise ValueError("exact saved B1 candidate full-state binary required; no B1 compile fallback")
    if cpu:
        os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or "," in visible or visible.strip() != visible:
            raise ValueError("one parent-allocated GPU required")
        if not args.cpu_readiness_report or not args.cpu_readiness_report.is_file():
            raise ValueError("completed CPU eval_shape readiness required before GPU occupation")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pair.adjacent("training_artifact_io")
    journal = pair.Journal(args.output, args.wall_seconds)
    journal.report.update(schema_version="shared_latent_T7_batch_diagnostic_v1", requested_stage=args.stage,
        gpu_count=0 if cpu else 1, horizon=7, batch_size=4, native_AD_argnums=[0, 1],
        model_compiles={"B1": 0, "B4": 0}, helper_compiles=0, optimizer_updates=0,
        training_updates=0, pairs=[], numerical_failures=[], repeat_failures=[],
        rtol=1e-5, atol=1e-6, timing_denominator="four_saved_B1_full_AD_calls_plus_device_loss_mean_parameter_mean_and_state_stack_div4_synchronized",
        runner_sha256=pair.sha(Path(__file__)), adoption=False, scientific_acceptance=False)
    try:
        if not cpu:
            pair.adjacent("orcjax_gpu_runtime").initialize_gpu_backend()
        import jax
        jax.config.update("jax_enable_x64", True)
        devices = jax.devices()
        if not cpu and (len(devices) != 1 or devices[0].platform != "gpu" or "V100" not in devices[0].device_kind):
            raise ValueError("exactly one allocated V100 required")
        if not cpu:
            journal.report["runtime_preflight"] = pair.adjacent("probe_shared_union_micro4").validate_runtime_report(args.runtime_preflight, jax.__version__)
        journal.emit("loading_candidate_and_four_training_windows")
        source, owner, transformer, context, head, _, trees, identities = pair.assets(jax, args)
        parameters = trees["candidate"]
        del trees
        selector = pair.adjacent("shared_latent_T7_batch_assets")
        manifest = pair.read_manifest(args)
        with jax.default_device(jax.devices("cpu")[0]):
            selected = selector.load_training_T7_cases(jax, prepared_manifest=args.training_prepared_manifest,
                dataset_manifest=args.dataset_manifest, input_bundle=manifest["input_bundle"],
                head=head, method=identities["candidate"]["method"], count=4)
        identity = mathematical_identity(jax, identities["candidate"], selected)
        journal.report["mathematical_identity"] = identity
        journal.report["window_metadata"] = selected["window_metadata"]
        base_identity = {**identities["candidate"], "device_kind": devices[0].device_kind}
        # CPU readiness cannot compare the GPU device kind, but verifies every
        # other exact B1 source/parameter/input identity before shape tracing.
        b1_sidecar = json.loads(args.candidate_executable.with_suffix(args.candidate_executable.suffix+".identity.json").read_text())
        expected_b1 = {"identity": base_identity, "component": "T7"}
        if cpu:
            supplied = dict(b1_sidecar.get("identity", {}))
            supplied.pop("device_kind", None)
            if b1_sidecar.get("component") != "T7" or supplied != identity["saved_B1_mathematical_identity"]:
                raise ValueError("CPU readiness saved B1 mathematical identity differs")
        elif b1_sidecar != expected_b1:
            raise ValueError("saved candidate B1 exact identity differs")
        host_arguments = (parameters, *selected["stacked"])
        if cpu:
            with ExitStack() as patches:
                objective = _candidate_objective(args, owner, transformer, context, head, patches)
                began = time.monotonic()
                with journal.deadline(args.lower_seconds, "CPU_eval_shape"):
                    shape = jax.eval_shape(batch_ad.make_batch_value_and_grad(jax, objective), *host_arguments)
            estimate = batch_ad.shape_memory_estimate(jax, shape, host_arguments)
            if (estimate["parameter_gradient_coordinates"] != 66123713
                    or estimate["parameter_gradient_leaves"] != 355
                    or estimate["state_gradient_shape"] != [4, 3854] or estimate["loss_shape"] != []):
                raise ValueError("full B4 output ABI shape differs")
            journal.report.update(status="CPU_readiness_passed", GPU_executed=False,
                shape_estimate=estimate, eval_shape_seconds=time.monotonic()-began,
                memory_planning=memory_planning(args, b1_sidecar, estimate),
                compile_planning={"previous_B1_lower_seconds": 324.935682,
                    "previous_B1_backend_seconds": 5543.315643, "previous_B1_load_seconds": 169.773898,
                    "post_compile_reserve_seconds": args.post_compile_reserve_seconds,
                    "B4_compile_or_memory_linear_scaling_guaranteed": False})
            journal.emit("CPU_shape_readiness_preserved")
            return journal.report
        readiness = json.loads(args.cpu_readiness_report.read_text())
        if (readiness.get("status") != "CPU_readiness_passed"
                or readiness.get("mathematical_identity") != identity
                or readiness.get("model_compiles") != {"B1": 0, "B4": 0}):
            raise ValueError("CPU shape readiness does not bind this exact B4 graph/input")
        journal.report["shape_estimate"] = readiness["shape_estimate"]
        journal.report["memory_planning"] = readiness["memory_planning"]
        from jax.experimental.serialize_executable import serialize, deserialize_and_load
        from scripts.hpc.run_shared_transformer_admission import memory_sizes
        abi = pair.adjacent("shared_full_state_ad_execution")
        gates = pair.adjacent("probe_shared_union_micro4")
        arguments = jax.device_put(host_arguments)
        singles = tuple((arguments[0], *jax.device_put(case)) for case in selected["cases"])
        executables, baseline_paths = {}, {}
        began = time.monotonic()
        with journal.deadline(args.load_seconds, "load_B1"), source.forbid_model_compile():
            serialized = pair.read_pickle(args.candidate_executable)
            executables["serial"] = deserialize_and_load(*serialized)
            for single in singles:
                abi.validate_full_state_abi(jax, serialized, executables["serial"], single, component="T7")
        del serialized
        journal.report["B1_load_seconds"] = time.monotonic()-began
        journal.emit("saved_B1_loaded_four_ABIs_validated")

        def preserve(kind, label, result, seconds, single_seconds=None):
            transfer_started = time.monotonic()
            host = jax.device_get(result)
            path = args.output.with_suffix("."+kind+"."+label+".result.pkl")
            if path.exists():
                raise FileExistsError(path)
            owner.atomic_pickle(path, {"mathematical_identity": identity, "implementation": kind,
                "label": label, "result": host})
            row = {"implementation": kind, "label": label, "artifact": str(path),
                "four_window_seconds": seconds, "trajectories_per_second": 4/seconds,
                "days_per_second": 28/seconds, "single_sample_seconds": single_seconds,
                "transfer_and_preserve_seconds": time.monotonic()-transfer_started}
            # Persist full arrays and measured timing BEFORE all host debug gates.
            journal.report.setdefault("raw_results", []).append(row.copy())
            journal.emit("raw_full_AD_and_timing_preserved", **row)
            row["validation"] = batch_ad.validate_batch_result(jax, host, parameters, selected["stacked"][0], gates)
            stats = devices[0].memory_stats()
            peak = stats.get("peak_bytes_in_use") if isinstance(stats, dict) else None
            row.update(JAX_device_memory_stats=stats, observed_peak_bytes=peak)
            if not isinstance(peak, (int, float)) or not np.isfinite(peak) or not 0 <= peak < 24*2**30:
                raise MemoryError("observed JAX peak unavailable or not below24GiB")
            if kind in baseline_paths:
                reference = pair.read_pickle(baseline_paths[kind])["result"]
                row["same_implementation_repeat"] = batch_ad.compare_full_results(jax, host, reference, pair, head.contract)
                if not row["same_implementation_repeat"]["passed"]:
                    journal.report["repeat_failures"].append({"implementation": kind, "label": label,
                        "artifact": str(path), "reference_artifact": str(baseline_paths[kind]),
                        "comparison": row["same_implementation_repeat"]})
            else:
                baseline_paths[kind] = path
            journal.report["results"].append(row)
            journal.emit("full_AD_hard_gates_checked", implementation=kind, label=label)
            return row

        def execute(kind, label):
            began = time.monotonic()
            single_seconds = None
            with journal.deadline(args.call_seconds, kind+"_"+label), source.forbid_model_compile():
                if kind == "serial":
                    results, single_seconds = [], []
                    for single in singles:
                        start = time.monotonic()
                        value = executables["serial"](*single)
                        jax.block_until_ready(value)
                        single_seconds.append(time.monotonic()-start)
                        results.append(value)
                    result = merge(*results)
                else:
                    result = executables["batch"](*arguments)
                jax.block_until_ready(result)
            elapsed = time.monotonic()-began
            return preserve(kind, label, result, elapsed, single_seconds)

        # One small reduction helper compile is allowed and timed separately.
        # Saved B1 full AD executes each real window before any expensive B4 compile.
        cold_outputs, cold_seconds = [], []
        for index, single in enumerate(singles):
            began = time.monotonic()
            with journal.deadline(args.call_seconds, "B1_cold_"+str(index)), source.forbid_model_compile():
                value = executables["serial"](*single)
                jax.block_until_ready(value)
            cold_seconds.append(time.monotonic()-began)
            cold_outputs.append(value)
            host = jax.device_get(value)
            raw_path = args.output.with_suffix(".serial.cold-sample"+str(index)+".result.pkl")
            if raw_path.exists():
                raise FileExistsError(raw_path)
            owner.atomic_pickle(raw_path, {"mathematical_identity": identity,
                "window_metadata": selected["window_metadata"][index], "result": host})
            record = {"sample": index, "full_result_artifact": str(raw_path),
                "synchronized_full_AD_seconds": cold_seconds[-1]}
            journal.report.setdefault("raw_B1_cold_results", []).append(record)
            journal.emit("B1_cold_raw_arrays_and_timing_preserved_before_gates", **record)
            try:
                record["validation"] = pair.validate_result(jax, host, parameters, "candidate", selected["cases"][index][0])
            except Exception as error:
                record["validation_failure"] = {"error_type": type(error).__name__, "error": str(error),
                    "evidence": bounded_failure_evidence(jax, host)}
                journal.emit("preserved_B1_cold_validation_failed", **record)
                raise
            del host
        began = time.monotonic()
        merge = jax.jit(lambda *rows: batch_ad.merge_serial_results(jax, rows)).lower(*cold_outputs).compile()
        journal.report.update(helper_compiles=1, reduction_helper_compile_seconds=time.monotonic()-began)
        began = time.monotonic()
        with source.forbid_model_compile():
            cold_merged = merge(*cold_outputs)
            jax.block_until_ready(cold_merged)
        preserve("serial", "cold-first", cold_merged, sum(cold_seconds)+time.monotonic()-began, cold_seconds)
        del cold_outputs, cold_merged

        if args.stage == "serial-saved":
            # Complete the already authorized serial denominator when the sole
            # B4 compile left no binary. This branch never loads/lowers B4.
            warm = [execute("serial", "warm"+str(index)) for index in range(1, args.pairs+1)]
            seconds = [row["four_window_seconds"] for row in warm]
            median = float(np.median(seconds))
            journal.report.update(status="numerical_failed" if journal.report["repeat_failures"] else "passed",
                B4_executed=False, B4_comparison_completed=False,
                timing_summary={"serial_warm_four_window_seconds": seconds,
                    "serial_four_window_median_seconds": median, "trajectories_per_second": 4/median,
                    "days_per_second": 28/median, "B4_speedup_measured": False,
                    "engineering_1_5x_gate_passed": None,
                    "complete_optimizer_update_throughput_measured": False})
            journal.emit("serial_saved_denominator_completed_no_B4")
            return journal.report

        batch_sidecar = {"identity": identity, "component": "B4_T7", "device_kind": devices[0].device_kind}
        path = args.batch_executable
        cold = {}
        if args.stage == "saved-compare":
            if json.loads(path.with_suffix(path.suffix+".identity.json").read_text()) != batch_sidecar:
                raise ValueError("saved B4 exact mathematical identity differs")
            began = time.monotonic()
            with journal.deadline(args.load_seconds, "load_B4"), source.forbid_model_compile():
                serialized = pair.read_pickle(path)
                executables["batch"] = deserialize_and_load(*serialized)
                batch_ad.validate_batch_abi(jax, serialized, executables["batch"], arguments)
            cold["load_seconds"] = time.monotonic()-began
        else:
            attempt = path.with_suffix(path.suffix+".compile-attempt.json")
            if path.exists() or attempt.exists() or path.with_suffix(path.suffix+".identity.json").exists():
                raise FileExistsError("B4 graph/attempt exists; saved-only recovery required, no second full compile")
            owner.atomic_json(attempt, {"mathematical_identity": identity, "full_B4_compile_attempts": 1})
            with ExitStack() as patches:
                objective = _candidate_objective(args, owner, transformer, context, head, patches)
                backward = jax.jit(batch_ad.make_batch_value_and_grad(jax, objective),
                    compiler_options=identities["candidate"]["compiler_options"])
                journal.emit("B4_lowering")
                began = time.monotonic()
                with journal.deadline(args.lower_seconds, "lower_B4"):
                    lowered = backward.lower(*arguments)
                cold["lower_seconds"] = time.monotonic()-began
                remaining = args.wall_seconds-(time.monotonic()-journal.started)-args.post_compile_reserve_seconds
                backend_deadline = min(args.backend_seconds, remaining)
                budget = backend_budget_gate(backend_deadline,
                    readiness["compile_planning"]["previous_B1_backend_seconds"])
                journal.report["backend_budget_gate"] = budget
                journal.emit("B4_backend_budget_screened_before_compile", **budget)
                if not budget["passed"]:
                    raise InsufficientBackendBudget("available backend window is below measured B1 minimum; minimum additional seconds: "
                        +str(budget["minimum_additional_seconds"]))
                journal.report["model_compiles"]["B4"] = 1
                journal.emit("B4_backend_compiling", deadline_seconds=backend_deadline,
                    protected_reserve_seconds=args.post_compile_reserve_seconds)
                began = time.monotonic()
                with journal.deadline(backend_deadline, "backend_B4"):
                    executables["batch"] = lowered.compile()
                cold["backend_seconds"] = time.monotonic()-began
                began = time.monotonic()
                serialized = serialize(executables["batch"])
                owner.atomic_pickle(path, serialized)
                owner.atomic_json(path.with_suffix(path.suffix+".identity.json"), batch_sidecar)
                cold["serialize_and_save_seconds"] = time.monotonic()-began
                journal.report["B4_cold"] = cold
                journal.emit("sole_B4_graph_saved_before_validation", executable=str(path), **cold)
                batch_ad.validate_batch_abi(jax, serialized, executables["batch"], arguments)
        del serialized
        cold["compiled_memory"] = memory_sizes(executables["batch"])
        journal.report["B4_cold"] = cold
        if cold["compiled_memory"]["estimated_live_bytes"] >= 24*2**30:
            raise MemoryError("compiled B4 live memory not below24GiB")

        execute("batch", "cold-first")
        for index in range(1, args.pairs+1):
            order = ("serial", "batch") if index % 2 else ("batch", "serial")
            rows = [execute(kind, "pair"+str(index)) for kind in order]
            by_kind = {row["implementation"]: row for row in rows}
            comparison = batch_ad.compare_full_results(jax,
                pair.read_pickle(by_kind["batch"]["artifact"])["result"],
                pair.read_pickle(by_kind["serial"]["artifact"])["result"], pair, head.contract)
            comparison_path = args.output.with_suffix(".pair"+str(index)+".strict-comparison.json")
            owner.atomic_json(comparison_path, {"mathematical_identity": identity, "pair": index,
                "batch_artifact": by_kind["batch"]["artifact"], "serial_artifact": by_kind["serial"]["artifact"],
                "comparison": comparison})
            if not comparison["passed"]:
                journal.report["numerical_failures"].append({"pair": index, "comparison_artifact": str(comparison_path), "comparison": comparison})
            journal.report["pairs"].append({"pair": index, "order": list(order), "measurements": rows,
                "strict_comparison_artifact": str(comparison_path), "strict_passed": comparison["passed"],
                "speedup": by_kind["serial"]["four_window_seconds"]/by_kind["batch"]["four_window_seconds"]})
            journal.emit("paired_measurement_preserved", pair=index, strict_passed=comparison["passed"])
        medians = {kind: float(np.median([next(row["four_window_seconds"] for row in item["measurements"]
            if row["implementation"] == kind) for item in journal.report["pairs"]])) for kind in ("serial", "batch")}
        speedup = medians["serial"]/medians["batch"]
        journal.report.update(status="numerical_failed" if journal.report["numerical_failures"] or journal.report["repeat_failures"] else "passed",
            timing_summary={"four_window_median_seconds": medians, "median_speedup": speedup,
                "trajectories_per_second": {kind: 4/value for kind, value in medians.items()},
                "days_per_second": {kind: 28/value for kind, value in medians.items()},
                "engineering_1_5x_gate_passed": speedup >= 1.5,
                "complete_optimizer_update_throughput_measured": False})
        journal.emit("completed")
        return journal.report
    except Exception as error:
        journal.report.update(status=("insufficient_backend_budget" if isinstance(error, InsufficientBackendBudget)
            else "resource_failed" if isinstance(error, MemoryError) else "error"),
            error_type=type(error).__name__, error=str(error))
        journal.emit("error")
        raise
    finally:
        journal.timer.cancel()


if __name__ == "__main__":
    run(parser().parse_args())
