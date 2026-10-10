"""Host-only exact undefined-state gates for preserved dual slicing outputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pickle

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report','bundle','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior host gate')
    report = json.loads(args.report.read_text())
    with args.bundle.open('rb') as stream:
        bundle = pickle.load(stream)
    payload = bundle['payloads'][0]
    initial = {'daily32_micro4':np.asarray(payload['daily'].inputs.continuous_state).reshape(8,4,3854),
               **{f'T{h}':np.asarray(payload['cases'][h].initial_continuous) for h in (7,30)}}
    defined = {name:np.isfinite(value)&(np.abs(value)<0.5e20) for name,value in initial.items()}
    rows = []
    for artifact in report['result_artifacts']:
        for route,paths in artifact['routes'].items():
            with Path(paths['states']).open('rb') as stream:
                saved = pickle.load(stream)
            for name,mask in defined.items():
                gradient = np.asarray(saved['initial_state_gradients'][name])
                if gradient.shape != mask.shape:
                    raise ValueError('state gradient/mask shape drift')
                bad = ~mask & (gradient != 0)
                finite = bool(np.all(np.isfinite(gradient)))
                rows.append({'phase':artifact['phase'],'route':route,'component':name,
                    'artifact':paths['states'],'shape':list(gradient.shape),'coordinates':int(gradient.size),
                    'defined_coordinates':int(mask.sum()),'undefined_coordinates':int((~mask).sum()),
                    'undefined_exact_zero':bool(not np.any(bad)),'finite':finite,
                    'failed_coordinates':[{'index':[int(i) for i in index],'gradient':float(gradient[index])}
                                          for index in zip(*np.nonzero(bad),strict=True)]})
    import jax
    bool_masks = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(payload['daily'])
                  if np.asarray(leaf).dtype == np.bool_]
    expected_phases = {'bootstrap','update1','update2','update3'}
    complete = (len(rows)==24 and {x['phase'] for x in rows}==expected_phases
                and {x['route'] for x in rows}=={'leaf','tree'}
                and report.get('completed_pairs')==3)
    output = {'status':'passed' if complete and all(x['undefined_exact_zero'] and x['finite'] for x in rows) else 'failed',
              'scope':'host_only_defined_mask_and_undefined_exact_zero_for_all_saved_states',
              'execution_report':str(args.report),'prepared_bundle':str(args.bundle),'complete':complete,
              'defined_rule':'isfinite(initial_continuous) & (abs(initial_continuous) < 0.5e20)',
              'defined_rule_provenance':'research/daily_coarse_graining/daily_operator/input_assembly.py:_defined',
              'rows':rows,'model_executions':0,'model_compilations':0,
              'source_masks':{'daily_bool_leaves':len(bool_masks),
                  'daily_bool_coordinates':sum(int(x.size) for x in bool_masks),
                  'daily_all_eight_slices_byte_equal':len(report.get('microbatch_slice_checks',[]))==8
                      and all(x['all_leaves_exact_equal'] for x in report.get('microbatch_slice_checks',[])),
                  'daily_evidence':'Every leaf, including bool source masks, is dtype/shape/byte equal across routes at starts0,4,...28.',
                  'recurrent_evidence':'Both routes call the same original executable object with the same prepared T7/T30 case objects; no recurrent slicing or mask transformation.'}}
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'status':output['status'],'complete':complete,'rows':len(rows),
                      'mask_counts':{name:{'defined':int(mask.sum()),'undefined':int((~mask).sum())} for name,mask in defined.items()}},indent=2))
    return 0 if output['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
