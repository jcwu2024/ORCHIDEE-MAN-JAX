"""Dynamic power-of-two native-AD conditioning on six actual readout shapes."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import time
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('preserve previous diagnostic')
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    jax,_devices=initialize_gpu_backend()
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from jaxlib import xla_client
    from probe_shared_neural_precision import algorithm_linear,vector_metrics
    from shared_daily_dual_ad import make_scaled_daily_dual_value_and_grad
    from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import _linear
    report={'status':'running','scope':'synthetic_readout_dynamic_S_native_full_parameter_state_AD',
        'scale':256,'loss_weights_unchanged':True,'custom_AD':False,'scientific_acceptance':False,
        'xla_flags':os.environ.get('XLA_FLAGS',''),'rows':[]}
    rng=np.random.default_rng(20261001)
    for shape in ((4,128,256),(4,14,256),(4,28,256),(4,72,256),(4,24,256),(4,2,256)):
        p={'kernel':jnp.asarray(rng.normal(0,.03,(256,1)),jnp.float32),'bias':jnp.array([.02],jnp.float32)}
        x=jnp.asarray(rng.normal(size=shape),jnp.float64)
        mask=jnp.asarray((np.indices(shape[:-1]).sum(0)%3!=0)[...,None])
        row={'shape':list(shape),'cases':[],'compiled_evidence':{}}
        row['state_dtype']=str(x.dtype)
        row['scale_dtype']=str(jnp.asarray(1.,jnp.float64).dtype)
        binaries={}
        for policy,linear in (('FP32',_linear),('F16',algorithm_linear)):
            def objective(p,x,samples,_linear=linear):
                m,c=samples
                y=_linear(p,x)
                loss=jnp.sum(jnp.where(m,jnp.asarray(y,jnp.float64),0)**2)*c
                return loss,y
            began=time.perf_counter()
            function=make_scaled_daily_dual_value_and_grad(jax,objective)
            binary=function.lower(p,x,(mask,jnp.array(.01,jnp.float64)),jnp.array(1.,jnp.float64)).compile()
            binaries[policy]=binary
            prints=xla_client._xla.HloPrintOptions.short_parsable()
            prints.print_metadata=False;prints.print_large_constants=False;prints.print_operand_shape=True
            hlo=binary.runtime_executable().hlo_modules()[0].to_string(prints)
            matches=[line.strip()[:2200] for line in hlo.splitlines() if re.search(r'\b(?:multiply|divide|convert|parameter|custom-call)\(',line)]
            row['compiled_evidence'][policy]={'compile_seconds':time.perf_counter()-began,
                'dynamic_scale_flat_leaf_index':5,'leaf_count':len(jax.tree_util.tree_leaves((p,x,(mask,jnp.array(.01)),jnp.array(1.)))),
                'scalar_parameters':[line.strip()[:1200] for line in hlo.splitlines() if '= f64[]' in line and 'parameter(' in line],
                'FP16_converts':[line.strip()[:2200] for line in hlo.splitlines() if re.search(r'= f16\[.*\bconvert\(',line)],
                'multiply_divide_convert_samples':matches[:60],
                'compact_hlo_characters':len(hlo),'full_hlo_exported':False}
        for coefficient in (.01,1e-8):
            values={}
            for policy,scale in (('FP32',1.),('FP32',256.),('F16',1.),('F16',256.)):
                values[f'{policy}_S{int(scale)}']=jax.device_get(jax.block_until_ready(binaries[policy](
                    p,x,(mask,jnp.asarray(coefficient,jnp.float64)),jnp.asarray(scale,jnp.float64))))
            base=values['FP32_S1']
            case={'loss_coefficient':coefficient,'policies':{}}
            for name,value in values.items():
                case['policies'][name]={'all_finite':all(np.all(np.isfinite(v)) for v in jax.tree_util.tree_leaves(value)),
                    'loss':float(value[0][0]),'public_output_dtypes':[str(np.asarray(v).dtype) for v in jax.tree_util.tree_leaves(value)],
                    'parameters':vector_metrics(jax,value[1][0],base[1][0]),'states':vector_metrics(jax,value[1][1],base[1][1]),
                    'parameter_nonzero':sum(int(np.count_nonzero(v)) for v in jax.tree_util.tree_leaves(value[1][0])),
                    'state_nonzero':int(np.count_nonzero(value[1][1]))}
            case['F16_S1_S256_original_values_exact']=all(np.array_equal(a,b) for a,b in zip(
                jax.tree_util.tree_leaves(values['F16_S1'][0]),jax.tree_util.tree_leaves(values['F16_S256'][0]),strict=True))
            case['FP32_unscale_correct']=all(case['policies']['FP32_S256'][group]['relative_l2']<=1e-7 for group in ('parameters','states'))
            row['cases'].append(case)
        report['rows'].append(row)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'shape':shape,'all_finite':all(c['policies'][n]['all_finite'] for c in row['cases'] for n in c['policies']),
            'correct_FP32_unscale':all(c['FP32_unscale_correct'] for c in row['cases'])}),flush=True)
    report['status']='supported' if all(c['FP32_unscale_correct'] and c['F16_S1_S256_original_values_exact']
        and all(v['all_finite'] for v in c['policies'].values()) for row in report['rows'] for c in row['cases']) else 'failed'
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if report['status']=='supported' else 1)


if __name__=='__main__': main()
