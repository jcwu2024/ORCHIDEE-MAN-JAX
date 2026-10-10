"""Bounded actual first-T7-day raw-output mask check, without recurrent AD.

Reuses the admitted fresh parameters, actual prepared inputs and public raw
dispatcher. This checks one real head forward; it does not certify every
rollout day, every branch, or the undefined coordinates of physical states.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np


def raw_mask_statistics(raw, masks, schema):
    names = {field.name for field in schema.fields}
    if set(raw) != names or set(masks) != names:
        raise ValueError("raw output/mask field sets differ from public schema")
    fields = {}; inactive_total = 0
    for field in schema.fields:
        value = np.asarray(raw[field.name]); mask = np.asarray(masks[field.name])
        if (value.shape != field.shape or mask.shape != field.shape
                or value.dtype != np.dtype('float32') or mask.dtype != np.dtype('bool')
                or not np.all(np.isfinite(value))):
            raise ValueError(f"raw output shape/dtype/finite gate failed: {field.name}")
        inactive = value[~mask]
        nonzero = int(np.count_nonzero(inactive))
        fields[field.name] = {'shape':list(value.shape), 'active':int(np.count_nonzero(mask)),
            'inactive':int(inactive.size), 'inactive_nonzero':nonzero,
            'inactive_max_abs':float(np.max(np.abs(inactive), initial=0.0))}
        if nonzero:
            raise ValueError(f"inactive raw output is nonzero: {field.name}")
        inactive_total += inactive.size
    if inactive_total == 0:
        raise ValueError("actual fixture has no inactive coordinates to check")
    return {'all_raw_outputs_finite':True, 'all_inactive_raw_outputs_exact_zero':True,
            'inactive_coordinates':int(inactive_total), 'fields':fields}


def run(args):
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax, devices = initialize_gpu_backend(); jax.config.update('jax_enable_x64', True)
    from jax.experimental.serialize_executable import serialize
    # Import after PJRT registration, using the same frozen overlay as T7.
    from importlib.util import spec_from_file_location, module_from_spec
    import sys
    def load(path, name):
        spec = spec_from_file_location(name, path); module = module_from_spec(spec)
        sys.modules[name] = module; spec.loader.exec_module(module); return module
    overlay = Path(__file__).parent
    load(overlay/'training_artifact_io.py', 'scripts.hpc.training_artifact_io')
    package = 'research.daily_coarse_graining.daily_operator.'
    load(overlay/'shared_transformer.py', package+'shared_transformer')
    latent = load(overlay/'shared_latent_reader.py', package+'shared_latent_reader')
    util = load(overlay/'shared_full_state_ad_execution.py', 'scripts.hpc.shared_full_state_ad_execution')
    owner = load(args.patch_owner, 'scripts.hpc.probe_shared_expert_real_daily')
    gates = load(args.gate_owner, 'latent_reader_mask_gate_utilities')
    forward = load(args.forward_probe, 'scripts.hpc.probe_shared_forward_reverse')
    started = time.perf_counter()
    report = {'status':'running', 'scope':'actual_first_T7_day_public_raw_head_output_masks',
              'training_updates':0, 'full_T7_AD_compilations':0,
              'scientific_accuracy_admitted':False, 'production_adoption':False}
    if args.output.exists(): raise FileExistsError(args.output)
    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started, **fields)
        owner.atomic_json(args.output, report); print(stage, flush=True)
    try:
        emit('starting')
        if len(devices) != 1 or jax.default_backend() != 'gpu':
            raise ValueError('one actual free-node GPU required')
        gates.validate_runtime_report(args.runtime_preflight, jax.__version__)
        _, reference, bundle, context, _, _ = forward._assets(args)
        with args.parameters.open('rb') as stream: saved = pickle.load(stream)
        identity = saved['identity']
        admission = json.loads(args.candidate_admission.read_text())
        util.validate_candidate_admission(admission, 'T7', identity)
        head = replace(context['head'], shared_config=latent.shared_latent_reader_config(context['head'].shared_config))
        hashes = {name:hashlib.sha256((overlay/name).read_bytes()).hexdigest()
                  for name in ('shared_transformer.py','shared_latent_reader.py')}
        if (identity['candidate_source_sha256'] != hashes
                or identity['candidate_config'] != vars(head.shared_config)
                or identity['source_head'] != reference['source_git_head']
                or identity['input_source_method'] != reference['method']
                or identity['source_reuse_assembler_sha256'] != hashlib.sha256(args.backbone_module_file.read_bytes()).hexdigest()
                or identity['source_reuse_projection_sha256'] != hashlib.sha256(args.backbone_module_file.with_name('shared_backbone_projection_reuse.py').read_bytes()).hexdigest()):
            raise ValueError('actual forward source/parameter identity differs from admitted T7')
        parameters = jax.device_put(saved['parameters'])
        case = bundle['payloads'][0]['cases'][7]
        from research.daily_coarse_graining.daily_operator.successor_training import rollout_input_from_template
        from research.daily_coarse_graining.daily_operator import shared_process_head
        from research.daily_coarse_graining.daily_operator.shared_transformer import shared_transformer_raw_outputs
        from scripts.hpc.run_shared_transformer_admission import memory_sizes
        template = jax.tree_util.tree_map(lambda value:value[0], case.days[0])
        current = rollout_input_from_template(template, case.initial_continuous, case.initial_discrete, head.contract)
        inputs = head.input_assembler.assemble(current)
        arguments = jax.device_put((parameters, inputs)); jax.block_until_ready(arguments)
        def raw_forward(params, assembled):
            encoded = head.assemble_context(params, assembled)
            return shared_transformer_raw_outputs(params, encoded, assembled,
                schema=head.learned_schema, layout=head.layout,
                input_normalization=head.input_normalization, config=head.shared_config,
                carbon_state_layout=head.carbon_state_layout)
        original = shared_process_head.assemble_shared_transformer_context
        try:
            shared_process_head.assemble_shared_transformer_context = owner.load_backbone_assembler(args.backbone_module_file)
            emit('lowering', identity=identity, tested_artifact_id=int(bundle['artifact_ids'][0]),
                 tested_horizon=7, tested_day_index=0)
            with gates.deadline(args.lower_seconds, 'raw_head_lower', emit):
                begin=time.perf_counter(); lowered=jax.jit(raw_forward).lower(*arguments)
                lower_seconds=time.perf_counter()-begin
            emit('compiling', lower_seconds=lower_seconds)
            with gates.deadline(args.backend_seconds, 'raw_head_backend', emit):
                begin=time.perf_counter(); executable=lowered.compile()
                compile_seconds=time.perf_counter()-begin
        finally: shared_process_head.assemble_shared_transformer_context = original
        path=args.output.with_suffix('.executable.pkl')
        owner.atomic_pickle(path, serialize(executable))
        owner.atomic_json(path.with_suffix(path.suffix+'.identity.json'),
                          {'identity':identity,'component':'head_raw_masks_first_T7_day'})
        memory=memory_sizes(executable)
        emit('executable_saved', executable=str(path), compile_seconds=compile_seconds, compiled_memory=memory)
        if memory['estimated_live_bytes'] > 24*2**30:
            raise MemoryError('raw forward compiled memory exceeds24GiB')
        with gates.deadline(60, 'raw_head_result', emit):
            result=jax.device_get(jax.block_until_ready(executable(*arguments)))
        result_path=args.output.with_suffix('.result.pkl')
        owner.atomic_pickle(result_path, {'identity':identity,'result':result})
        validation=raw_mask_statistics(*result, head.learned_schema)
        stats=jax.local_devices()[0].memory_stats() or {}
        if stats.get('peak_bytes_in_use', 24*2**30+1) > 24*2**30:
            raise MemoryError('observed raw forward peak unavailable/exceeds24GiB')
        emit('completed', status='passed', validation=validation, result_artifact=str(result_path),
             device_memory=stats, bounded_first_day_only=True,
             undefined_physical_state_zero_certified=False)
    except BaseException as error:
        emit('exception', status='resource_failed' if isinstance(error,MemoryError) else 'error',
             error=str(error), error_type=type(error).__name__); raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--component', choices=('masks',), default='masks')
    for name in ('output','parameters','candidate-admission','patch-owner','gate-owner','forward-probe',
                 'source-profile','input-bundle','prepared-context','daily-admission-report',
                 'seven-admission-report','thirty-admission-report','support-owner-file',
                 'runtime-preflight','backbone-module-file'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--lower-seconds', type=int, default=120)
    p.add_argument('--backend-seconds', type=int, default=420)
    run(p.parse_args())

if __name__ == '__main__': main()
