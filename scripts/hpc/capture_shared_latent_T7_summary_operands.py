"""CPU-only canonical day0 -> state-field summary_input operand capture.

Does not call a neural head/context, backbone, reader, physical step or rollout.
Real mode is fail-closed on immutable historical assets. Fixture mode exercises
the same canonical reconstruction and preprocessing with explicit synthetic
objects; it never supplies real provenance. JVP covers preprocessing only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace

# Set before importing JAX or deserializing any saved JAX arrays. A caller's
# cuda/gpu environment cannot initialize a GPU through this CPU-only utility.
os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_compilation_cache", False)

from scripts.hpc import probe_shared_latent_T7_batch_numerics as probe
from scripts.hpc.shared_latent_T7_batch_assets import _selected_record, _training_provenance, native_window_dates
from research.daily_coarse_graining.daily_operator.input_conditioning import standardize_continuous
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import _masked_summary, _mask_inactive_pft_elements
from research.daily_coarse_graining.daily_operator.successor_training import rollout_input_from_template

HISTORICAL_HEAD = "c599236bcf57b745e9bafed678298010f4c3b58f"
T7_MANIFEST_SHA = "252d610a8df4b78963111244edf7ef3414471883bdfe126f436275beaef57bc4"
STATE299_SHA = "54d01d0f376b0ceed8a843740328dba9d61d2a8989507436bda1a0d16c624f5d"
ARTIFACT299_SHA = "ee7fa6e0f537fcd75dc8d02bbf4aa4f0438119f1ca73368e8f7f9d64af1f18aa"
PREPARED_IDENTITY = {
    "dataset_manifest_sha256": "5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99",
    "schedule_sha256": "ee1ff380a18e1ff3ea919c579455db5895fa95116f2f95d144d21a852da9e2a5",
    "source_git_head": "a243029e6043d71820c86eeb52a9a42215f9fa91",
    "training_contract_sha256": "91faf01123008a60c59038c074779a2a2b4553b82c3739475dd4829fe91ea25b",
}
HISTORICAL_OVERLAYS = {
    "shared_transformer.py": "aad25a9879de3fe5d79cf213f665998b921ba8b3599e4b577d8c4c49673c9b96",
    "shared_latent_reader.py": "02f44b30109664504f158cc55d6bed1e325b6782dcfdd35df626e6fcba5eeb20",
}
SCHEMA = "T7_summary_input_capture_receipt_v1"
BOUNDARY = "rollout_input_from_template -> CanonicalDailyOperatorInputAssembler.assemble -> standardize_continuous -> _mask_inactive_pft_elements -> _masked_summary -> tokenizer.summary_input operands"
PATH_POLICY = "On Windows only, saved pathlib.PosixPath is restored as PurePosixPath; historical lexical paths are preserved without binding remote paths to the host filesystem."


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_identity(value):
    value = np.ascontiguousarray(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256_bytes_C_order": hashlib.sha256(value.tobytes()).hexdigest()}


def require_file(path, expected_sha=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("required saved source unavailable: " + str(path))
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("saved source changed during identity verification: " + str(path))
    if expected_sha is not None and digest != expected_sha:
        raise ValueError("immutable saved source identity differs: " + str(path))
    return {"path": str(path), "sha256": digest, "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns}


class HistoricalAssetUnpickler(pickle.Unpickler):
    """Keep Linux path metadata lexical when reading unchanged assets on Windows."""

    def find_class(self, module, name):
        if os.name == "nt" and (module, name) == ("pathlib", "PosixPath"):
            return PurePosixPath
        return super().find_class(module, name)


def read_verified_pickle(path, expected_sha, *, verified_identity=None):
    # Hash before deserialization; no unverified pickle is opened in real mode.
    identity = verified_identity if verified_identity is not None else require_file(path, expected_sha)
    if verified_identity is not None:
        resolved, stat = Path(path).resolve(), Path(path).stat()
        if (identity["path"] != str(resolved) or identity["sha256"] != expected_sha
                or (identity["size_bytes"], identity["mtime_ns"]) != (stat.st_size, stat.st_mtime_ns)):
            raise ValueError("verified saved source identity changed before deserialization")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as stream:
        return HistoricalAssetUnpickler(stream).load(), identity


@dataclass(frozen=True)
class AssetPaths:
    parameters: Path
    context: Path
    t7_manifest: Path
    prepared_manifest: Path
    dataset_manifest: Path


def validate_historical_method(saved, context, manifest):
    identity = saved["identity"]
    method = identity["input_source_method"]
    ci = context["identity"]
    if (identity.get("source_head") != HISTORICAL_HEAD or identity.get("artifact_id") != 742
            or identity.get("candidate_source_sha256") != HISTORICAL_OVERLAYS):
        raise ValueError("saved candidate historical source/overlay identities differ")
    if (ci.get("schema_version") != "shared_expert_real_daily_context_v1"
            or ci.get("source_git_head") != HISTORICAL_HEAD or ci.get("method") != method
            or ci.get("artifact_id") != 742 or ci.get("daily_rows") != 4
            or ci.get("initial_checkpoint") != manifest.get("original_checkpoint")):
        raise ValueError("saved context historical method/checkpoint identity differs")
    if (list(method["training_plan"]["training_artifact_ids"][:4]) != probe.ARTIFACT_IDS
            or method["configuration"].get("recurrent_window_index") != 0
            or method["training_plan"]["prepared_manifest_identity"] != PREPARED_IDENTITY):
        raise ValueError("original TRAIN order/window/prepared method identity differs")
    return method


def validate_manifests(manifest, prepared, dataset, method):
    expected = {"schema_version": "shared_corrected_latent_reader_T7_pair_manifest_v1",
        "prepared_artifact_index": 0, "horizon": 7, "native_AD_argnums": [0, 1],
        "precision": "FP32_neural_FP64_physics", "training_updates": 0,
        "historical_original_is_corrected_pair_reference": False,
        "turnover_base_sha256": "bb8674f3d2ceadb28df907346d18f6300dd87f46d09ff6010c795f72dbee6485",
        "turnover_sha256": "f6240cf5ce64a47b9bbe0647b5e5715305e9a9230ea77f2bd9f42981ae35a93c"}
    if any(type(manifest.get(k)) is not type(v) or manifest[k] != v for k, v in expected.items()):
        raise ValueError("original T7 manifest protocol differs")
    if (prepared.get("schema_version") != "target_oriented_prepared_manifest_v1"
            or prepared.get("status") != "passed" or prepared.get("identity") != PREPARED_IDENTITY
            or prepared["identity"] != method["training_plan"]["prepared_manifest_identity"]):
        raise ValueError("prepared TRAIN manifest identity differs")
    if dataset.get("schema_version") != "daily_teacher_dataset_manifest_v6" or dataset.get("status") != "complete":
        raise ValueError("complete canonical v6 dataset manifest required")
    registry = {int(row["update"]): row for row in prepared["entries"]}
    declared = method["training_plan"]["training_artifact_ids"]
    if len(registry) != len(prepared["entries"]) or len(set(declared)) != len(declared):
        raise ValueError("duplicate prepared or declared TRAIN artifact ids")
    if any(identifier not in registry for identifier in probe.ARTIFACT_IDS):
        raise ValueError("one of the four declared TRAIN artifacts is absent")
    return registry


def first_day_template(inputs):
    shape = np.shape(inputs.continuous_state)
    if len(shape) != 3 or shape[0] < 1 or shape[1] < 7 or shape[2] != 3854:
        raise ValueError("saved prepared [window,day,3854] state required")
    if np.asarray(inputs.continuous_state).dtype != np.dtype("float64"):
        raise ValueError("saved physical state must be FP64")
    indices = np.asarray(inputs.day_index)[0, :7].reshape(-1)
    if indices.size != 7 or not np.all(np.diff(indices) == 1):
        raise ValueError("saved window0 must have seven consecutive model days")
    return jax.tree_util.tree_map(lambda value: np.asarray(value)[0, 0], inputs), indices


def load_real_assets(paths):
    """Only manifests + four prepared day0 templates + saved head metadata.

    No Teacher shard is loaded. No prepare_shared_window, pair assets(), model
    builder or executable is called. Full-file historical hashes authenticate
    provenance; current helper execution is recorded separately.
    """
    files = {"parameters": require_file(paths.parameters, probe.PARAMETERS_SHA256),
        "context": require_file(paths.context, probe.CONTEXT_SHA256),
        "t7_manifest": require_file(paths.t7_manifest, T7_MANIFEST_SHA),
        "prepared_manifest": require_file(paths.prepared_manifest),
        "dataset_manifest": require_file(paths.dataset_manifest, PREPARED_IDENTITY["dataset_manifest_sha256"])}
    manifest = json.loads(Path(paths.t7_manifest).read_text(encoding="utf-8"))
    # Stored remote path remains historical identity, not a silently remapped
    # executable path. Local filename must refer to the declared saved tree.
    if Path(manifest["candidate_parameters"]).name != Path(paths.parameters).name:
        raise ValueError("original manifest candidate parameter basename differs")
    saved, _ = read_verified_pickle(paths.parameters, probe.PARAMETERS_SHA256, verified_identity=files["parameters"])
    context, _ = read_verified_pickle(paths.context, probe.CONTEXT_SHA256, verified_identity=files["context"])
    method = validate_historical_method(saved, context, manifest)
    prepared = json.loads(Path(paths.prepared_manifest).read_text(encoding="utf-8"))
    dataset = json.loads(Path(paths.dataset_manifest).read_text(encoding="utf-8"))
    registry = validate_manifests(manifest, prepared, dataset, method)
    templates, selection, semantic_days = [], [], set()
    for identifier in probe.ARTIFACT_IDS:
        entry = registry[identifier]
        path = (Path(paths.prepared_manifest).parent / entry["path"]).resolve()
        path.relative_to(Path(paths.prepared_manifest).parent.resolve())
        if identifier == 299 and entry["sha256"] != ARTIFACT299_SHA:
            raise ValueError("original artifact299 manifest identity differs")
        artifact, file_identity = read_verified_pickle(path, entry["sha256"])
        if (int(artifact.schedule_entry["update"]) != identifier
                or int(artifact.schedule_entry["runtime_year"]) != int(entry["runtime_year"])
                or artifact.trace_signature != entry["trace_signature"]
                or artifact.training_contract_sha256 != PREPARED_IDENTITY["training_contract_sha256"]):
            raise ValueError("saved prepared artifact schedule/contract differs")
        record, start = _selected_record(artifact, 0)
        shard = _training_provenance(dataset, record, start, int(entry["runtime_year"]))
        days = {(str(record["landpoint_id"]), int(record["year"]), start + i) for i in range(7)}
        if semantic_days.intersection(days):
            raise ValueError("independent declared TRAIN windows overlap")
        semantic_days.update(days)
        template, indices = first_day_template(artifact.batch.seven_day.inputs)
        state_identity = array_identity(template.continuous_state)
        if identifier == 299 and state_identity["sha256_bytes_C_order"] != STATE299_SHA:
            raise ValueError("original sample299 window0/day0 state identity differs")
        templates.append(template)
        selection.append({"artifact_id": identifier, "window_index": 0, "day_index": 0,
            "prepared_artifact": file_identity, "state": state_identity,
            "spatial_split": shard["spatial_split"], "temporal_split": shard["temporal_split"],
            "landpoint_id": str(record["landpoint_id"]), "year": int(record["year"]),
            "start_row": start, "stop_row_exclusive": start + 7,
            "day_indices": indices.astype(int).tolist(), "model_dates": native_window_dates(record["year"], indices),
            "parent_shard_sha256": shard["shard_sha256"], "typed_shard_sha256": shard["typed_shard_sha256"]})
    head = context["head"]  # Deserialize only; never build/replace/call a head.
    if (head.input_assembler.contract != head.contract
            or tuple(head.layout.active_pft_indices) != tuple(head.contract.active_pft_indices)):
        raise ValueError("saved head assembler/contract/layout PFT identity differs")
    return {"templates": tuple(templates), "assembler": head.input_assembler,
        "normalization": head.input_normalization, "layout": head.layout,
        "projection": summary_projection(saved["parameters"]),
        "origin": "captured_component_operands", "provenance": {"files": files,
            "historical_path_deserialization_policy": PATH_POLICY,
            "historical_source_head": HISTORICAL_HEAD, "historical_overlays": HISTORICAL_OVERLAYS,
            "saved_candidate_identity": saved["identity"], "prepared_context_identity": context["identity"],
            "selection": selection, "original_manifest_declarations": manifest}}


def summary_projection(parameters):
    try:
        projection = parameters["tokenizer"]["summary_input"]
        kernel, bias = (np.asarray(projection[name]) for name in ("kernel", "bias"))
    except (KeyError, TypeError) as error:
        raise ValueError("saved tokenizer.summary_input leaf required") from error
    if (set(projection) != {"kernel", "bias"} or kernel.shape != (4, 384) or bias.shape != (384,)
            or kernel.dtype != np.dtype("float32") or bias.dtype != np.dtype("float32")
            or not np.all(np.isfinite(kernel)) or not np.all(np.isfinite(bias))):
        raise ValueError("exact finite FP32 saved summary_input[4,384]+bias[384] required")
    return {"kernel": kernel.copy(), "bias": bias.copy()}


def source_identities():
    paths = [Path(__file__), ROOT / "scripts/hpc/probe_shared_latent_T7_batch_numerics.py",
        ROOT / "scripts/hpc/shared_latent_T7_batch_assets.py", ROOT / "research/daily_coarse_graining/daily_markov_contract.py",
        ROOT / "jax_orchidee/ad_primitives.py"]
    paths += [ROOT / "research/daily_coarse_graining/daily_operator" / name for name in
              ("successor_training.py", "input_assembly.py", "input_conditioning.py", "strong_baseline_architecture.py")]
    result = {path.relative_to(ROOT).as_posix(): {"path": str(path), "sha256": sha256(path)} for path in paths}
    result["jax_orchidee/ad_primitives.py"]["used_function"] = "source_sqrt_with_finite_zero_tangent; registered _finite_zero_sqrt_jvp"
    return result


def capture_inputs(assets):
    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("capture permits CPU preprocessing only")
    templates = assets["templates"]
    if len(templates) != 4 or any(np.shape(t.continuous_state) != (3854,) for t in templates):
        raise ValueError("four ordered canonical day0 templates with state3854 required")
    assembler, normalization, layout = (assets[name] for name in ("assembler", "normalization", "layout"))
    pool = np.asarray(layout.state_field_pool)
    owner = np.asarray(layout.state_element_pft_index)
    if (assembler.contract.continuous_state_width != 3854 or pool.dtype != np.dtype("bool")
            or pool.ndim != 2 or pool.shape[1] != 3854 or owner.shape != (3854,)
            or owner.dtype.kind not in "iu" or np.any(owner < -2)
            or np.any(owner >= len(assembler.active_pft_indices))):
        raise ValueError("exact state-field pool and PFT-owner layout required")
    if pool.shape[0] != sum(not leaf.discrete for leaf in assembler.contract.state_leaves):
        raise ValueError("state-field pool rows differ from canonical continuous leaf count")
    for name in ("continuous_mean", "continuous_scale"):
        value = np.asarray(getattr(normalization, name))
        if value.shape != (3854,) or value.dtype.kind != "f":
            raise ValueError("saved continuous normalization width/type differs")
    tail_mask = normalization.continuous_tail_mask
    if tail_mask is not None and (np.shape(tail_mask) != (3854,) or np.asarray(tail_mask).dtype != np.dtype("bool")):
        raise ValueError("saved continuous tail coordinate mask differs")
    for template in templates:
        active = np.asarray(template.pft.active_pft_mask)
        if active.shape != (len(assembler.active_pft_indices),) or active.dtype != np.dtype("bool"):
            raise ValueError("canonical active PFT mask differs")
    states = np.stack([np.asarray(template.continuous_state) for template in templates])
    if states.dtype != np.dtype("float64"):
        raise ValueError("physical input states must remain FP64")
    stack = jax.tree_util.tree_map(lambda *values: jnp.asarray(np.stack(values)), *templates)

    def single(state, template):
        rebuilt = rollout_input_from_template(template, state, template.discrete_state, assembler.contract)
        inputs = assembler.assemble(rebuilt)
        values = standardize_continuous(inputs.state.other, normalization.continuous_mean,
            normalization.continuous_scale, normalization.continuous_tail_threshold,
            coordinate_mask=normalization.continuous_tail_mask)
        defined = _mask_inactive_pft_elements(inputs.state_defined.other, owner, inputs.pft.active_pft_mask)
        return _masked_summary(pool, values, defined)

    direction = np.zeros_like(states)
    direction[3, 434] = 1.
    x, dx, valid = jax.jvp(lambda state: jax.vmap(single)(state, stack),
        (jnp.asarray(states),), (jnp.asarray(direction),), has_aux=True)
    x, dx, valid = map(np.asarray, jax.device_get((x, dx, valid)))
    # Defer finite/isolation admission to write_capture's shared probe gate so
    # a rejected JVP still preserves every already-computed numeric operand.
    projection = summary_projection({"tokenizer": {"summary_input": assets["projection"]}})
    mask = np.broadcast_to(valid[..., None], (4, pool.shape[0], 384)).copy()
    cotangent = np.broadcast_to(np.linspace(.25, 1.25, 384, dtype=np.float32), mask.shape).copy()
    meta = {"schema_version": probe.SCHEMA, "origin": assets["origin"],
        "capture_boundary": BOUNDARY,
        "direction_coordinate": 434, "direction_sample": 3, "raw_state_direction_value": 1.,
        "window_index": 0, "day_index": 0,
        "mask_meaning": "actual canonical state-field valid mask broadcast across all384 output channels",
        "cotangent_meaning": "fixed diagnostic channel ramp0.25..1.25; not original T7 loss cotangent",
        "scope": "current canonical CPU preprocessing of declared sources; affine operands only",
        "capture_source_sha256": sha256(__file__)}
    if assets["origin"] == "captured_component_operands":
        meta.update(parameter_file_sha256=probe.PARAMETERS_SHA256,
                    prepared_context_sha256=probe.CONTEXT_SHA256,
                    artifact_ids=probe.ARTIFACT_IDS,
                    parameter_path=assets["provenance"]["files"]["parameters"]["path"])
    else:
        meta["fixture_sample_indices"] = [0, 1, 2, 3]
    # Receipt fields are added after the independently saved receipt is hashed.
    bundle = {"x": x, **projection, "direction": dx, "cotangent": cotangent, "mask": mask, "metadata": meta}
    detail = {"raw_states": array_identity(states), "raw_direction": array_identity(direction),
        "normalized_summary": array_identity(x), "summary_direction": array_identity(dx),
        "state_field_valid": {**array_identity(valid), "valid_counts_by_sample": np.count_nonzero(valid, axis=1).tolist()},
        "state_field_pool": array_identity(pool), "state_element_pft_index": array_identity(owner),
        "normalization": {name: array_identity(getattr(normalization, name)) for name in
                          ("continuous_mean", "continuous_scale")},
        "continuous_tail_threshold": normalization.continuous_tail_threshold,
        "continuous_tail_mask": None if normalization.continuous_tail_mask is None else array_identity(normalization.continuous_tail_mask),
        "projection": {name: array_identity(value) for name, value in projection.items()},
        "zero_other_sample_tangents": bool(np.all(dx[:3] == 0)),
        "validity_gate": "full state-field Boolean mask; full raw affine outputs preserved",
        "no_neural_forward_or_full_model_AD": True}
    return bundle, detail


def synthetic_assets():
    """Small real canonical objects, synthetic fields/normalization/parameters.

    Padding retains state3854; the canonical assembler and PFT refresh are
    exercised. Every scientific/source identity remains explicitly synthetic.
    """
    from research.daily_coarse_graining.daily_markov_contract import (
        DailyMarkovContract, StateLeafSpec, StaticConditionSpec, ConditionLeafSpec, NativeForcingSpec)
    from research.daily_coarse_graining.daily_operator.input_assembly import CanonicalDailyOperatorInputAssembler
    from research.daily_coarse_graining.daily_operator.neural_architecture import NeuralInputNormalization
    from research.daily_coarse_graining.daily_operator.types import (
        DailyOperatorInput, NativeForcingInput, PFTAxisInput, StaticRegistryInput)
    leaves, offset = [], 0
    def leaf(component, name, shape, pft_axis=None):
        nonlocal offset
        full = list(shape)
        axes = [f"synthetic_axis{i}" for i in range(len(shape))]
        if pft_axis is not None:
            full[pft_axis] = 14
            axes[pft_axis] = "nvm"
        size = int(np.prod(shape))
        leaves.append(StateLeafSpec(component, (name,), shape, tuple(full), "float64", offset, offset + size,
            "synthetic_fixture", "synthetic_method_fixture", axis_names=tuple(axes),
            selected_pft_indices=(0, 13) if pft_axis is not None else ()))
        offset += size
    hydro, slow = "hydrol_previous_step_state", "slowproc_stomate_previous_step_state"
    leaf(hydro, "qsintveg", (1, 2), 1)
    for name in ("mc", "mcl", "snow", "snow_nobio", "snowliq", "flood_res"):
        leaf(hydro, name, (1,))
    for name in ("litter_above", "litter_below"):
        leaf(slow, name, (1, 1, 2), 2)
    leaf(slow, "carbon_32l", (1, 3, 2, 1), 2)
    leaf(slow, "DOC", (1, 2, 1, 1), 1)
    for name in ("interception_storage", "bm_to_litter", "turnover_daily", "lignin_struc_above",
                 "lignin_struc_below", "veget_max", "lai"):
        leaf(slow, name, (1, 2), 1)
    leaf(slow, "biomass", (1, 2, 1), 1)
    leaf("enerbil_previous_step_state", "temp_sol", (1,))
    leaf(hydro, "synthetic_padding", (3854 - offset,))
    static = StaticConditionSpec((), (ConditionLeafSpec("clay_frac", (1,), "float64", 0, 1,
        "synthetic_static", "synthetic_method_fixture"),), (), ())
    contract = DailyMarkovContract("synthetic_method_fixture", tuple(leaves), (), (),
        NativeForcingSpec(("synthetic",), (1,), 1, 86400., 86400., 1, 1), static, (), (0, 13))
    assembler = CanonicalDailyOperatorInputAssembler.from_contract(contract)
    pool = np.zeros((len(leaves), 3854), bool)
    owner = np.full(3854, -1, np.int32)
    for index, spec in enumerate(leaves):
        pool[index, spec.start:spec.stop] = True
        if spec.selected_pft_indices:
            axis = spec.axis_names.index("nvm")
            indices = np.indices(spec.shape)[axis].reshape(-1)
            owner[spec.start:spec.stop] = indices
    owner[200], owner[201], owner[202] = -2, 0, 1
    mean, scale, tail = np.zeros(3854), np.ones(3854), np.zeros(3854, bool)
    scale[434], scale[435], tail[435] = 2e-7, 8e-12, True
    normalization = NeuralInputNormalization(mean, scale, np.zeros((1, 1)), np.ones((1, 1)),
        np.zeros(1), np.ones(1), np.zeros(1), np.ones(1), np.ones(1), np.ones(1), np.ones(1), 16., tail)
    templates = []
    undefined = next(spec.start for spec in leaves if spec.path == ("snowliq",))
    for sample in range(4):
        raw = np.linspace(.1 + .1 * sample, .3 + .1 * sample, 3854)
        raw[undefined] = 0.5e20
        raw[203] = np.nan
        raw[434], raw[435] = (sample + 1) * 1e-8, .048
        pft = PFTAxisInput(np.full((2, 3), -999.), np.ones((2, 1)), np.ones((2, 1)),
                          np.full(2, -999.), np.asarray([True, sample == 3]))
        templates.append(DailyOperatorInput(raw, {}, NativeForcingInput(np.ones((1, 1)), np.zeros(1),
            np.full(1, 86400.), np.zeros(1, bool), np.ones(1, bool)), pft,
            StaticRegistryInput(np.asarray([.2]), np.ones((1, 1)), np.ones(1)), np.zeros(1),
            np.asarray(1961), np.asarray(sample + 1)))
    kernel = np.linspace(-.025, .025, 4 * 384, dtype=np.float32).reshape(4, 384)
    bias = np.linspace(-.01, .01, 384, dtype=np.float32)
    return {"templates": tuple(templates), "assembler": assembler, "normalization": normalization,
        "layout": SimpleNamespace(state_field_pool=pool, state_element_pft_index=owner),
        "projection": summary_projection({"tokenizer": {"summary_input": {"kernel": kernel, "bias": bias}}}),
        "origin": "synthetic_method_fixture", "provenance": {"description": "fixed canonical-object method fixture; no real TRAIN data or inherited weights"}}


def write_capture(output, assets):
    output = Path(output).resolve()
    receipt_path, operand_path = Path(str(output) + ".capture.json"), Path(str(output) + ".operands.npz")
    failed_path = Path(str(output) + ".failed-arrays.npz")
    if receipt_path.exists() or operand_path.exists() or failed_path.exists():
        raise FileExistsError("use a new capture stem; preserve existing evidence")
    started = time.perf_counter()
    bundle, detail = capture_inputs(assets)
    sources = source_identities()
    receipt = {"schema_version": SCHEMA, "status": "operands_captured", "origin": assets["origin"],
        "capture_boundary": BOUNDARY, "operand_path": str(operand_path),
        "provenance": assets["provenance"], "current_execution": {
            "cwd": str(ROOT), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "jax_version": jax.__version__, "devices": [str(device) for device in jax.devices()], "source_files": sources},
        "preprocessing": detail, "array_identities": {name: array_identity(bundle[name]) for name in probe.ARRAY_KEYS},
        "direction_coordinate": 434, "direction_sample": 3, "cotangent": bundle["metadata"]["cotangent_meaning"],
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["Current CPU JAX helper evaluation is separate from historical deployed JAX/CUDA.",
            "Fixed diagnostic cotangent is not original T7 loss cotangent.",
            "No affine/whole-network forward, physical step, rollout or full-model gradient evaluated during capture.",
            "Synthetic origin, when selected, supplies method evidence only and never real artifact provenance."]}
    # Receipt binds numeric arrays and sources, not NPZ bytes: avoids a circular
    # receipt-hash/metadata-hash dependency. No executed failure is discarded.
    bundle["metadata"].update(capture_source_path=str(Path(__file__).resolve()),
        capture_receipt_path=str(receipt_path), capture_receipt_sha256="pending_receipt_hash")
    admission_error = None
    try:
        probe.validate_operands(bundle)
    except ValueError as error:
        admission_error = str(error)
        receipt.update(status="operand_admission_failed", admission_error=admission_error,
                       failed_arrays_path=str(failed_path), operands_admitted=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    bundle["metadata"]["capture_receipt_sha256"] = sha256(receipt_path)
    if admission_error is not None:
        np.savez_compressed(failed_path, **{name: bundle[name] for name in probe.ARRAY_KEYS},
                            metadata=np.asarray(json.dumps(bundle["metadata"], sort_keys=True)))
        raise ValueError("captured numerical/admission failure preserved: " + str(receipt_path))
    probe.save_operands(operand_path, bundle)
    return receipt, operand_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New unique capture stem")
    parser.add_argument("--synthetic-fixture", action="store_true")
    for name in AssetPaths.__annotations__:
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in AssetPaths.__annotations__}
    if args.synthetic_fixture and any(value is not None for value in paths.values()):
        parser.error("synthetic fixture cannot use real asset path arguments")
    if not args.synthetic_fixture and not all(value is not None for value in paths.values()):
        parser.error("real capture requires all five explicit saved source paths")
    failure_path = Path(str(args.output.resolve()) + ".failure.json")
    if failure_path.exists():
        raise FileExistsError("use a new stem; prior failure evidence is preserved")
    stage = "load_sources"
    try:
        assets = synthetic_assets() if args.synthetic_fixture else load_real_assets(AssetPaths(**paths))
        stage = "capture_preprocessing"
        receipt, operands = write_capture(args.output, assets)
    except (FileNotFoundError, ValueError) as error:
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps({"schema_version": SCHEMA,
            "status": "assets_unavailable" if isinstance(error, FileNotFoundError) else "capture_rejected",
            "execution_status": "not_executed" if stage == "load_sources" else "preprocessing_attempted_not_admitted",
            "asset_availability": "unavailable" if isinstance(error, FileNotFoundError) else "identity_or_numerical_rejection",
            "stage": stage,
            "origin": assets["origin"] if stage == "capture_preprocessing" else
                      ("synthetic_method_fixture" if args.synthetic_fixture else "real_capture_requested_not_executed"),
            "error_type": type(error).__name__, "error": str(error), "operands_saved": False,
            "current_execution": {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "jax_version": jax.__version__, "platforms": [device.platform for device in jax.devices()],
                "source_files": source_identities()},
            "source_paths": {name: str(path) if path is not None else None for name, path in paths.items()}},
            indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"failure_receipt": str(failure_path), "error": str(error)}))
        return 2
    print(json.dumps({"receipt": str(Path(str(args.output.resolve()) + ".capture.json")),
        "operands": str(operands), "origin": receipt["origin"], "summary_shape": receipt["preprocessing"]["normalized_summary"]["shape"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
