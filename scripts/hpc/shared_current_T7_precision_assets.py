"""Read-only artifact299 public-window loader, without a neural/model graph.

The immutable saved B1 mathematical receipt authenticates parameter/context
provenance; sampling[3] authenticates the independently prepared TRAIN window.
No old capture module is imported and no platform environment is changed.
Historical c599 source is asset provenance, never the current Git checkout.
Trusted historical pickles are hash-verified before deserialization; this is not
a general untrusted-pickle sandbox. Only public prepare_shared_window is called.
"""
from __future__ import annotations

from dataclasses import replace
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import pickle

import numpy as np

from scripts.hpc.shared_current_T7_precision_guard import CURRENT_MODULES, PACKAGE, SOURCE_HEAD
from scripts.hpc.shared_latent_T7_batch_assets import (
    _selected_record, _training_provenance, native_window_dates, tree_digest,
)

PARAMETERS_SHA = '4f95d18d97d2feb4dcab8e1dda6caeb242a4869f044a00b6a7ae6bc689806c14'
CONTEXT_SHA = 'ff3fdabe755d599bc2ceee72087c03983d9a629add93166519dcda1f56be9026'
ARTIFACT_SHA = 'ee7fa6e0f537fcd75dc8d02bbf4aa4f0438119f1ca73368e8f7f9d64af1f18aa'
CASE_SHA = '8c0ed07cd295a28a2f0ca0a9ae46b1904fc59328863aa4c89af9e6f83e820448'
STATE_SHA = '54d01d0f376b0ceed8a843740328dba9d61d2a8989507436bda1a0d16c624f5d'
RECEIPT_SHA = '5254cad5507607f68ed7e0bc6dde712b27a2d9ec5982ee2853da667b0e938eaf'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_file(path, expected_sha=None):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Asset changed during verification: '+str(path))
    if expected_sha is not None and digest != expected_sha:
        raise ValueError('Saved asset hash differs: '+str(path))
    return dict(path=str(path), sha256=digest, size_bytes=after.st_size, mtime_ns=after.st_mtime_ns)


class HistoricalAssetUnpickler(pickle.Unpickler):
    """Preserve remote path metadata lexically on Windows, without remapping."""
    def find_class(self, module, name):
        if os.name == 'nt' and (module, name) == ('pathlib', 'PosixPath'):
            return PurePosixPath
        return super().find_class(module, name)


def read_verified_pickle(path, expected_sha):
    identity = require_file(path, expected_sha)
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rb') as stream:
        value = HistoricalAssetUnpickler(stream).load()
    if require_file(path, expected_sha) != identity:
        raise ValueError('Asset changed during deserialization')
    return value, identity


def validate_receipt(receipt):
    if receipt.get('route') != 'B1':
        raise ValueError('Exact saved B1 receipt required')
    asset = receipt['identity']['asset_identity']
    math = asset['saved_B1_mathematical_identity']
    provenance = math['parameter_provenance']
    if (math.get('source_git_head') != SOURCE_HEAD or math.get('architecture') != 'shared_latent32x384_reader64x2'
            or math.get('parameter_count') != 66123713 or math.get('parameter_leaves') != 355
            or math.get('training_updates') != 0 or math.get('horizon') != 7
            or math.get('prepared_context_sha256') != CONTEXT_SHA or provenance.get('sha256') != PARAMETERS_SHA):
        raise ValueError('Current saved mathematical identity differs')
    if provenance['identity'].get('source_head') != SOURCE_HEAD:
        raise ValueError('Saved candidate asset source provenance differs')
    selection = asset['sampling'][3]
    if any(type(selection.get(key)) is not type(value) or selection.get(key) != value for key, value in dict(artifact_id=299, window_index=0, horizon=7,
            spatial_split='train', temporal_split='train', case_sha256=CASE_SHA,
            prepared_artifact_sha256=ARTIFACT_SHA).items()):
        raise ValueError('Exact artifact299/window0/T7 TRAIN sampling identity required')
    return math, selection


def validate_manifests(prepared, dataset, method):
    expected = method['training_plan']['prepared_manifest_identity']
    if (prepared.get('schema_version') != 'target_oriented_prepared_manifest_v1'
            or prepared.get('status') != 'passed' or prepared.get('identity') != expected):
        raise ValueError('Prepared training manifest identity differs')
    if dataset.get('schema_version') != 'daily_teacher_dataset_manifest_v6' or dataset.get('status') != 'complete':
        raise ValueError('Complete canonical v6 dataset manifest required')
    declared = method['training_plan']['training_artifact_ids']
    if len(set(declared)) != len(declared) or list(declared[:4]) != [742, 275, 482, 299]:
        raise ValueError('Original declared TRAIN order differs')
    if method['configuration'].get('recurrent_window_index') != 0:
        raise ValueError('Declared window0 required')
    registry = {int(row['update']): row for row in prepared['entries']}
    if len(registry) != len(prepared['entries']) or 299 not in registry:
        raise ValueError('Duplicate or absent artifact299 prepared entry')
    if registry[299]['sha256'] != ARTIFACT_SHA:
        raise ValueError('Artifact299 immutable identity differs')
    return registry[299]


def parameter_gate(jax, parameters):
    leaves = jax.tree_util.tree_leaves(parameters)
    if len(leaves) != 355 or sum(np.size(value) for value in leaves) != 66123713:
        raise ValueError('Current 355-leaf/66,123,713 parameter scope required')
    if any(np.asarray(value).dtype != np.dtype('float32') or not np.all(np.isfinite(value)) for value in leaves):
        raise ValueError('Saved current parameters must be finite FP32')


def validate_case(jax, case, selection):
    if len(case) != 5 or np.shape(case[0]) != (3854,) or np.asarray(case[0]).dtype != np.dtype('float64'):
        raise ValueError('Public five-part full FP64 initial-state case required')
    if np.shape(case.days[0].continuous_state)[0] != 7 or np.shape(case.targets['state'])[0] != 7:
        raise ValueError('Exact seven-day public inputs/targets required')
    indices = np.asarray(case.days[0].day_index).reshape(-1)
    if indices.tolist() != selection['day_indices'] or not np.all(np.diff(indices) == 1):
        raise ValueError('Prepared day indices differ from saved sampling')
    state_sha = hashlib.sha256(np.ascontiguousarray(case[0]).tobytes()).hexdigest()
    if state_sha != STATE_SHA or tree_digest(jax, case) != CASE_SHA:
        raise ValueError('Exact saved artifact299 state/public case identity differs')
    return dict(case_sha256=CASE_SHA, initial_state_sha256=state_sha,
                initial_state_shape=[3854], initial_state_dtype='float64')


def load_assets(jax, *, parameters, prepared_context, training_prepared_manifest,
                dataset_manifest, saved_identity_receipt):
    """Load exactly one saved artifact299 case; no objective or model call.

    Caller supplies its configured JAX runtime. No JIT/eval_shape/lower/compile
    API is invoked. prepare_shared_window performs the public input assembly
    and initial-stock-mask preparation, without invoking any neural head.
    """
    if not jax.config.x64_enabled:
        raise ValueError('Explicit x64-enabled runtime required')
    paths = dict(parameters=Path(parameters), prepared_context=Path(prepared_context),
                 training_prepared_manifest=Path(training_prepared_manifest),
                 dataset_manifest=Path(dataset_manifest), saved_identity_receipt=Path(saved_identity_receipt))
    files = {name: require_file(path) for name, path in paths.items()}
    if files['saved_identity_receipt']['sha256'] != RECEIPT_SHA:
        raise ValueError('Exact saved B1 mathematical receipt bytes required')
    receipt = json.loads(paths['saved_identity_receipt'].read_text(encoding='utf-8'))
    math, selection = validate_receipt(receipt)
    saved, files['parameters'] = read_verified_pickle(paths['parameters'], PARAMETERS_SHA)
    context, files['prepared_context'] = read_verified_pickle(paths['prepared_context'], CONTEXT_SHA)
    if saved['identity'] != math['parameter_provenance']['identity'] or context['identity'] != math['prepared_context_identity']:
        raise ValueError('Saved parameter/context full metadata differs from B1 mathematical receipt')
    method = saved['identity']['input_source_method']
    if context['identity']['method'] != method or method != math['method']:
        raise ValueError('Saved method identity differs')
    parameters_host = jax.device_get(saved['parameters'])
    parameter_gate(jax, parameters_host)
    prepared = json.loads(paths['training_prepared_manifest'].read_text(encoding='utf-8'))
    dataset = json.loads(paths['dataset_manifest'].read_text(encoding='utf-8'))
    entry = validate_manifests(prepared, dataset, method)
    if files['dataset_manifest']['sha256'] != prepared['identity']['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest immutable hash differs')
    artifact_path = (paths['training_prepared_manifest'].parent / entry['path']).resolve()
    artifact_path.relative_to(paths['training_prepared_manifest'].parent.resolve())
    artifact, files['prepared_artifact299'] = read_verified_pickle(artifact_path, ARTIFACT_SHA)
    if (int(artifact.schedule_entry['update']) != 299
            or int(artifact.schedule_entry['runtime_year']) != int(entry['runtime_year'])
            or artifact.trace_signature != entry['trace_signature']
            or artifact.training_contract_sha256 != prepared['identity']['training_contract_sha256']):
        raise ValueError('Artifact299 schedule/contract identity differs')
    record, start = _selected_record(artifact, 0)
    shard = _training_provenance(dataset, record, start, int(entry['runtime_year']))
    actual_selection = dict(landpoint_id=str(record['landpoint_id']), year=int(record['year']),
        start_row=start, stop_row_exclusive=start+7, parent_shard_sha256=shard['shard_sha256'],
        typed_shard_sha256=shard['typed_shard_sha256'])
    if any(selection[key] != value for key, value in actual_selection.items()):
        raise ValueError('Artifact299 TRAIN shard/window differs from saved sampling')
    modules = tuple(importlib.import_module(PACKAGE+name) for name in CURRENT_MODULES)
    sources = {}
    for module in (modules[2], modules[-1]):
        basename = Path(module.__file__).name
        sources[basename] = require_file(module.__file__, math['neural_module_sha256'][basename])
    latent = modules[-1]
    if (latent.LATENT_SLOTS, latent.LATENT_LAYERS, latent.READER_WIDTH, latent.READER_LAYERS) != (32, 2, 64, 2):
        raise ValueError('Current latent32/private reader64x2 layout differs')
    candidate_head = replace(context['head'], shared_config=latent.shared_latent_reader_config(context['head'].shared_config))
    if vars(candidate_head.shared_config) != saved['identity']['candidate_config']:
        raise ValueError('Current public candidate configuration differs')
    from research.daily_coarse_graining.daily_operator.shared_recurrent_training import prepare_shared_window
    case = jax.device_get(prepare_shared_window(candidate_head, artifact.batch.seven_day, window_index=0, horizon=7))
    case_identity = validate_case(jax, case, selection)
    if native_window_dates(record['year'], selection['day_indices']) != tuple(selection['model_date_labels']):
        raise ValueError('Saved noleap date labels differ')
    for name, identity in files.items():
        if require_file(identity['path'], identity['sha256']) != identity:
            raise ValueError('Original read-only asset changed: '+name)
    return dict(parameters=parameters_host, context=context, candidate_head=candidate_head,
        case=case, modules=modules, manifest_metadata=dict(saved_B1_mathematical_identity=math,
        sampling=selection, case_identity=case_identity, source_git_head_is_asset_provenance=True,
        current_git_head_not_overridden=True), original_assets=files, neural_sources=sources,
        current_graph_admitted=False, objective_called=False,
        required_execution_patches=dict(backbone='Install admitted current backbone assembler aliases when constructing objective.',
            turnover_base_sha256=math['turnover_base_sha256'], turnover_sha256=math['turnover_sha256'],
            turnover='Retain the accepted corrected turnover overlay; no physical patch is installed by this loader.'))
