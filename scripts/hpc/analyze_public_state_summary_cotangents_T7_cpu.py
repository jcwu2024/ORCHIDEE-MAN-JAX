"""CPU saved-array screen; no JAX/model evaluation or precision candidate.

Normalization provenance: strong_baseline_architecture._standardize 976-981;
input_conditioning.standardize_continuous 12-52. Only day zero's non-tail
direct normalization sensitivity is expressed in initial raw-state units.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import pickle
import numpy as np

COORDINATES = (33, 199, 428)
ATOL, RTOL = 1e-6, 1e-5
UNIT_ROUNDOFF = 2.0**-24
GAMMA7 = 7*UNIT_ROUNDOFF/(1-7*UNIT_ROUNDOFF)
CRITERIA = dict(
    valid="First capture passes complete zero-delta loss/aux/parameter/state bridge and hard/ABI/finite/memory gates.",
    repeat_stable="Both extra captures exist; selected partials and full original output scopes pass unchanged strict self-comparison; risk classifications agree.",
    material_screen="Day-zero selected coordinate has both positive and negative incoming partials, and gamma7*sum(abs(partials))/source_FP32_scale reaches its original raw-gradient strict allowance.",
    interpretation="Material means possible local accumulation sensitivity in saved FP32 partials, not observed GPU error or T7 causality. A conservative upper bound cannot prove a failure.",
    bound_assumptions="gamma7 assumes seven round-to-nearest normal FP32 additions of eight saved finite partials, without overflow/underflow. It excludes upstream partial-generation error and does not establish the actual compiled reduction order.",
    instrumentation_limit="Separate dynamic zero-delta arguments can change compiler optimization. The complete strict zero-delta bridge validates public outputs; it cannot prove identity of hidden original-binary cotangents.",
    no_material_stop="If valid and repeat-stable with no material selected day-zero screen, close this boundary diagnosis and return to structural complete-update optimization.",
    invalid_stop="Failed first bridge/hard gate is terminal; preserve artifacts, do not interpret as admitted original cotangents or retry the capture graph.",
    uncertain_stop="Missing/unstable repeats or a material bound allow only bounded uncertainty reporting; no automatic precision expansion, reference adoption, or exhaustive gradient decomposition.",
    later_days="Days one through six describe incoming normalized-state cotangents only; no mapping to initial raw gradients without recurrent Jacobians.")

def read(path):
    with path.open('rb') as stream:
        return pickle.load(stream)

def leaves(value, prefix=''):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from leaves(value[key], prefix+'/'+str(key))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from leaves(child, prefix+'/'+str(index))
    else:
        yield prefix, value

def tree_abi(value):
    if isinstance(value,dict):
        return ('dict',tuple((key,tree_abi(value[key])) for key in sorted(value)))
    if isinstance(value,(tuple,list)):
        return (type(value).__name__,tuple(tree_abi(child) for child in value))
    return ('static_None',) if value is None else ('numeric_leaf',)

def compare(actual, reference):
    if tree_abi(actual)!=tree_abi(reference):
        return dict(passed=False,ABI=False,reason='container or static-node tree ABI differs')
    aa, rr = list(leaves(actual)), list(leaves(reference))
    if len(aa) != len(rr):
        return dict(passed=False, ABI=False)
    failures, count, maximum, exact, nonfinite_errors = 0, 0, 0.0, True, 0
    numeric_leaves, static_paths = 0, []
    for (ap, a), (rp, r) in zip(aa, rr, strict=True):
        if a is None or r is None:
            if ap!=rp or a is not None or r is not None:
                return dict(passed=False,ABI=False,path=ap,reason='static None ABI differs')
            static_paths.append(ap)
            continue
        a,r=np.asarray(a),np.asarray(r)
        if ap != rp or a.shape != r.shape or a.dtype != r.dtype:
            return dict(passed=False, ABI=False, path=ap)
        if a.dtype.kind not in 'biuf':
            return dict(passed=False,ABI=False,path=ap,reason='unsupported nonnumeric static leaf',dtype=str(a.dtype))
        numeric_leaves+=1
        count += a.size
        exact = exact and bool(np.array_equal(a, r))
        if a.dtype.kind in 'biu':
            failures += int(np.count_nonzero(a != r))
        else:
            a64, r64 = a.astype(np.float64), r.astype(np.float64)
            with np.errstate(invalid='ignore',over='ignore'):
                error = np.abs(a64-r64)
            failures += int(np.count_nonzero(~(np.isfinite(a64)&np.isfinite(r64)&(error<=ATOL+RTOL*np.abs(r64)))))
            finite_error=np.isfinite(error)
            nonfinite_errors += int(np.count_nonzero(~finite_error))
            maximum = max(maximum, float(np.max(error[finite_error], initial=0)))
    return dict(passed=failures==0, ABI=True, failed_coordinates=failures,
                coordinates=count, max_abs_error=None if nonfinite_errors else maximum,
                max_finite_abs_error=maximum, nonfinite_error_coordinates=nonfinite_errors, array_equal=exact,
                numeric_leaves=numeric_leaves,static_nodes=len(static_paths),static_None_paths=static_paths,
                static_nodes_ABI_passed=True)

def ordered_sum(values):
    total = np.float32(0)
    with np.errstate(over='ignore',invalid='ignore'):
        for value in values:
            total = np.float32(total + value)
    return float(total) if np.isfinite(total) else None

def screen(partials, scale, raw_gradient):
    rows=[]
    if np.shape(partials)!=(7,8,3854):
        return []
    for day in range(7):
        for coordinate in COORDINATES:
            values=np.asarray(partials[day,:,coordinate], np.float32)
            if not np.isfinite(values).all():
                rows.append(dict(day_index=day,coordinate=coordinate,
                                 incoming_FP32=[float(v) if np.isfinite(v) else None for v in values],
                                 finite=False,material_screen=False,screen_valid=False))
                continue
            total=math.fsum(float(v) for v in values)
            absolute=math.fsum(abs(float(v)) for v in values)
            bound=GAMMA7*absolute
            both=bool(np.any(values>0) and np.any(values<0))
            row=dict(day_index=day, coordinate=coordinate, incoming_FP32=values.tolist(), finite=True,screen_valid=True,
                     analysis_FP64_sum=total, absolute_sum=absolute,
                     cancellation_ratio=None if total==0 else absolute/abs(total),
                     both_signs=both, gamma7_normalized_absolute_bound=bound,
                     illustrative_forward_order_sum_FP32=ordered_sum(values),
                     illustrative_reverse_order_sum_FP32=ordered_sum(values[::-1]),
                     actual_GPU_accumulation_order_known=False)
            if day==0:
                safe=float(np.float32(scale[coordinate]))
                if not math.isfinite(safe) or safe<=0:
                    safe=1.0
                allowance=ATOL+RTOL*abs(float(raw_gradient[coordinate]))
                row.update(source_FP32_safe_scale=safe,
                           direct_normalization_sum_raw_units=total/safe,
                           gamma7_direct_normalization_bound_raw_units=bound/safe,
                           original_public_raw_gradient=float(raw_gradient[coordinate]),
                           original_public_strict_allowance=allowance,
                           material_screen=both and bound/safe>=allowance,
                           complete_initial_gradient_reconstructed=False)
            rows.append(row)
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', type=Path)
    parser.add_argument('--component-inputs', type=Path)
    parser.add_argument('--write-criteria', type=Path)
    args=parser.parse_args()
    if args.write_criteria:
        args.write_criteria.parent.mkdir(parents=True,exist_ok=True)
        args.write_criteria.write_text(json.dumps(CRITERIA,indent=2)+'\n')
        return
    if args.family is None or args.component_inputs is None:
        parser.error('analysis requires family and component-inputs')
    model=args.family/'model'
    report=json.loads((model/'model.json').read_text())
    capture=read(model/'public-cotangents.result.pkl')
    baseline_record=read(model/'original-replay.pkl')
    baseline=baseline_record['result']
    inputs=read(args.component_inputs)
    if capture['identity'] != baseline_record['identity'] or capture['identity'] != inputs['identity']:
        raise ValueError('capture/replay/source-normalization case identity differs')
    for c in COORDINATES:
        if inputs['tail_mask'][c] or not inputs['defined'][c] or not np.isfinite(inputs['raw'][c]):
            raise ValueError('selected coordinate is not defined legacy normalization')
    result=capture['result']
    partials=np.asarray(result[1][2])
    saved_cotangent_valid=partials.shape==(7,8,3854) and partials.dtype==np.float32 and bool(np.isfinite(partials).all())
    bridge=dict(loss_aux=compare(result[0],baseline[0]),
                all_parameter_gradients=compare(result[1][0],baseline[1][0]),
                all_initial_state_gradients=compare(result[1][1],baseline[1][1]))
    parameters=bridge['all_parameter_gradients']
    parameters['declared_numeric_scope_passed']=(parameters.get('numeric_leaves')==300
                                               and parameters.get('coordinates')==64884481)
    parameters['passed']=parameters['passed'] and parameters['declared_numeric_scope_passed']
    rows=screen(partials,inputs['scale'],baseline[1][1])
    first_screen=[r['material_screen'] for r in rows if r['day_index']==0]
    valid=(saved_cotangent_valid and all(r['passed'] for r in bridge.values()) and report.get('hard_admission',{}).get('passed',False)
           and report.get('cotangent_ABI_passed',False) and report.get('cotangent_finite',False)
           and report.get('observed_memory_passed',False))
    output=dict(criteria=CRITERIA, first_capture_valid=valid, saved_cotangent_valid=saved_cotangent_valid, independent_complete_bridge=bridge,
                raw_unit_scaling_limit="Day-zero division is CPU analysis of direct sensitivity only; gamma7 excludes normalization-division roundoff, other direct state paths, and all recurrent Jacobian effects.",
                first_capture_screen=rows, gamma7=GAMMA7, CPU_analysis_only=True,
                FP64_reference_adopted=False, public_T7_cause_established=False,
                complete_update_speed_claim=False, repeats=[])
    # Extra files deliberately omit the 64.9M-parameter tree; runtime checks
    # retain its complete in-memory comparison against the first capture.
    for path in sorted(model.glob('public-cotangents.repeat-*.pkl')):
        repeat=read(path)
        if repeat['identity'] != capture['identity']:
            raise ValueError('repeat case identity differs')
        repeated=np.asarray(repeat['cotangents'])
        repeated_rows=screen(repeated,inputs['scale'],baseline[1][1])
        output['repeats'].append(dict(file=path.name,
            all_incoming_partials=compare(repeated,partials),
            selected_incoming_partials=compare(repeated[:,:,COORDINATES],partials[:,:,COORDINATES]),
            all_initial_state_gradients=compare(repeat['state_gradient'],result[1][1]),
            loss_aux=compare(repeat['loss_aux'],result[0]),
            runtime_complete_parameter_comparison=repeat['parameter_comparison'],
            material_classification_agrees=first_screen==[r['material_screen'] for r in repeated_rows if r['day_index']==0],
            selected_screen=repeated_rows))
    runtime_repeats=report.get('repeat_comparisons',[])
    stable=(len(runtime_repeats)==2 and all(r['passed'] for r in runtime_repeats)
            and len(output['repeats'])==2 and all(
        all(row[key]['passed'] for key in ('selected_incoming_partials','all_initial_state_gradients','loss_aux','runtime_complete_parameter_comparison'))
        and row['material_classification_agrees'] for row in output['repeats']))
    output.update(repeat_stable=stable, screen_interpretation_admitted=valid and stable,
                  material_day_zero_coordinates=[c for c,m in zip(COORDINATES,first_screen) if m],
                  decision='invalid_terminal' if not valid else 'repeat_stability_unverified_or_failed' if not stable else
                  'possible_local_accumulation_sensitivity_requires_causal_evidence' if any(first_screen) else
                  'no_material_selected_day_zero_boundary_screen_return_to_throughput')
    (args.family/'cancellation-analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:output[key] for key in ('decision','first_capture_valid','repeat_stable','material_day_zero_coordinates')},indent=2))

if __name__=='__main__':
    main()
