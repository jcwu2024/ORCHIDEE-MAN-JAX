"""Read the fixed seven-day input boundary without evaluating a transition."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def summary(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): summary(v) for k, v in value.items()}
    if hasattr(value, '_asdict'):
        return summary(value._asdict())
    a = np.asarray(value)
    if a.dtype.kind not in 'biuf':
        return str(value)
    result = dict(shape=list(a.shape), dtype=str(a.dtype),
        sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
        finite=bool(np.all(np.isfinite(a))))
    if a.size <= 48:
        result['values'] = a.tolist()
    else:
        finite = a[np.isfinite(a)]
        result['finite_min_max'] = [float(finite.min()), float(finite.max())] if finite.size else []
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('code-root', 'parameters', 'prepared-context', 'training-prepared-manifest',
                 'dataset-manifest', 'saved-identity-receipt', 'source-cache', 'retained-data-root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve existing input-binding evidence')
    code = args.code_root.resolve(strict=True)
    sys.path.insert(0, str(code))
    os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
        PYTHONDONTWRITEBYTECODE='1', ORCHIDEE_REPO_ROOT=str(code),
        ORCHIDEE_DATA_ROOT=str(args.retained_data_root.resolve(strict=True)))
    import jax
    jax.config.update('jax_enable_x64', True)
    from scripts.hpc.shared_current_T7_precision_assets import load_assets
    from scripts.hpc.shared_current_T7_precision_guard import historical_expert_source_context
    from research.daily_coarse_graining.daily_operator.types import ConservativeUpdateResult
    from research.daily_coarse_graining.daily_operator.retained_tail import retained_tail_input
    from research.daily_coarse_graining.daily_markov_contract import reconstruct_state_fields_compiled
    from jax_orchidee.driver import orchestration as teacher

    with historical_expert_source_context(code, args.source_cache):
        assets = load_assets(jax, parameters=args.parameters, prepared_context=args.prepared_context,
            training_prepared_manifest=args.training_prepared_manifest,
            dataset_manifest=args.dataset_manifest, saved_identity_receipt=args.saved_identity_receipt)
        case = assets['case']
        template = assets['context']['template']
        owner = template.static_owner
        base_bundle = teacher._paper_1961_step_bundle_from_context(owner.context,
            year=int(owner.runtime_year), tstep=48)
        base_payload = teacher.build_intersurf_first_step_payload(base_bundle,
            driver_z0_for_wind=0.1, dt_sechiba=owner.context.dt_sechiba)
        contract = owner.contract
        report = dict(status='input_boundary_extracted_no_transition',
            source_checkout=dict(root=str(code), head=subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=code,text=True).strip()),
            source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            sampling=assets['manifest_metadata']['sampling'],
            case_identity=assets['manifest_metadata']['case_identity'],
            original_template_day_inputs=summary(template.day_inputs),
            original_first_bundle=dict(year=int(owner.context.first_step_bundle.year),
                domain_lon=summary(owner.context.first_step_bundle.domain.lon),
                domain_lat=summary(owner.context.first_step_bundle.domain.lat)),
            base_bundle=dict(year=int(base_bundle.year),co2_ppm=float(base_bundle.co2_ppm),
                payload=summary({name:getattr(base_payload,name) for name in base_payload.__dataclass_fields__
                    if name!='missing_fields'})),
            static_conditions_contract=asdict(contract.static_conditions) if is_dataclass(contract.static_conditions) else str(contract.static_conditions),
            native_forcing_contract=asdict(contract.native_forcing) if is_dataclass(contract.native_forcing) else str(contract.native_forcing),
            days=[], full_model_calls=0, full_model_traces=0, full_model_compiles=0,
            physical_transitions=0, neural_calls=0, override_functions_mocked=False,
            scope='Actual prepared input assembly, Adapter._dynamic_day_inputs, and driver transition-input construction only; no model transition, neural call, JIT, or loss evaluation.',
            lift_token_boundary='Uses each saved canonical day state for the token-only fast_update.state.other input. No forecast fast-update or prediction is constructed; unused handoff fields are None.')
        for day in range(7):
            canonical = jax.tree_util.tree_map(lambda x: x[day], case.days[0])
            prepared = jax.tree_util.tree_map(lambda x: x[day], case.days[1])
            assembled = owner.input_assembler.assemble(canonical)
            # This handoff supplies only the input token needed by the lift;
            # no updater, prediction, or scientific transition is evaluated.
            handoff = ConservativeUpdateResult(assembled.state, assembled.state_defined,
                canonical.discrete_state, None, None, None)
            value = retained_tail_input(assembled, None, handoff)
            effective_adapter = replace(template, day_inputs=prepared)
            effective = effective_adapter._dynamic_day_inputs(value)
            fields = reconstruct_state_fields_compiled(canonical.continuous_state,
                canonical.discrete_state, contract)
            previous = teacher.DriverPreviousStepStatePacket(
                tstep=int(owner.lifecycle_start_tstep)-1,
                fields_by_component=fields, provenance_by_component={})
            inputs = teacher._paper_1961_later_day_transition_inputs(
                context=owner.context, current_state=previous, year=int(owner.runtime_year),
                start=int(owner.lifecycle_start_tstep), steps_per_stomate=48,
                fixed_format_trace_dir=None, static_trace_fields=None, prebuild_day_payloads=True,
                prebound_hydrol_runtime_static_tables=teacher.HydrolRuntimeStaticTables(
                    mineral=teacher.MineralCWRRTables(**effective.hydrol_table_arrays._asdict(),
                        imin=int(owner.mineral_imin), imax=int(owner.mineral_imax)), peat=None),
                compiled_forcing_series=effective.compiled_forcing,
                compiled_landpoint_payload=effective.landpoint_payload)
            actual_payload = inputs.compiled_base_payload_template
            payload_matches = {name: bool(np.array_equal(np.asarray(getattr(actual_payload,name)),
                np.asarray(wanted),equal_nan=True)) for name,wanted in effective.landpoint_payload.items()}
            if not all(payload_matches.values()):
                raise ValueError('actual driver landpoint replacement mismatch')
            report['days'].append(dict(year=summary(canonical.year),day_index=summary(canonical.day_index),
                canonical_forcing=summary(canonical.forcing), annual_conditions=summary(canonical.annual_conditions),
                static_context=summary(canonical.static_registry.context_features),
                prepared_retained=summary(prepared),effective_retained=summary(effective),
                driver_payload=summary({name:getattr(actual_payload,name) for name in actual_payload.__dataclass_fields__
                    if name!='missing_fields'}),driver_override_exact_matches=payload_matches,
                driver_inputs_year=inputs.year,driver_inputs_start=inputs.start,
                actual_half_hour_input_count=len(inputs.half_hour_inputs)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=True)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'],days=len(report['days']),output=str(args.output))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
