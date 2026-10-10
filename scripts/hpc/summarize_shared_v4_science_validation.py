"""Reduce six saved small physical-curve JSON panels; never load a model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from research.daily_coarse_graining.daily_operator.forward_screen_metrics import summarize_errors

ARMS=('reference_a','candidate','reference_b')

def stats(p,t,m):
    return summarize_errors(np.asarray(p),np.asarray(t),np.asarray(m,dtype=bool))

def curve(windows,kind,key,label):
    rows=[w['physical_curves'][kind][key] for w in windows]
    if kind=='fluxes':
        p=[np.asarray(r['predicted']) for r in rows];t=[np.asarray(r['teacher']) for r in rows];m=[np.asarray(r['defined'],bool) for r in rows]
        if label=='cumulative':p=[v.sum(axis=0) for v in p];t=[v.sum(axis=0) for v in t];m=[v.all(axis=0) for v in m]
    else:
        fields={'level':('predicted_level','teacher_level','defined'),
                'daily_change':('predicted_daily_change','teacher_daily_change','daily_change_defined'),
                'trajectory_change':('predicted_change_from_common_initial','teacher_change_from_common_initial','defined')}
        pk,tk,mk=fields['level' if label=='final' else label]
        p=[np.asarray(r[pk]) for r in rows];t=[np.asarray(r[tk]) for r in rows];m=[np.asarray(r[mk],bool) for r in rows]
        if label=='final':p=[v[-1] for v in p];t=[v[-1] for v in t];m=[v[-1] for v in m]
    return tuple(np.concatenate([v.reshape(-1) for v in seq]) for seq in (p,t,m))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('family',type=Path);args=parser.parse_args();began=time.perf_counter()
    panels={(step,arm):json.loads((args.family/'model'/f'step-{step:02d}-{arm}.science.json').read_text()) for step in (8,16) for arm in ARMS}
    output=dict(status='six_panels_reduced',role='held_out_development',source_NPP=True,sealed_independent_test=False,
                scientific_acceptance=False,production_adoption=False,groups={},window_checks=[])
    for (step,arm),panel in panels.items():
        if len(panel['windows'])!=4 or not panel['original_execution_all_arms']:raise ValueError('Panel execution scope differs')
        for w in panel['windows']:
            rh=w['physical_curves']['fluxes']['heterotrophic_respiration'];teacher=np.asarray(rh['teacher']);mask=np.asarray(rh['defined'],bool)
            output['window_checks'].append(dict(step=step,arm=arm,ordinal=w['selection']['artifact_id'],horizon=w['horizon'],
                slice=w['selection']['slice_id'],physical_gate_passed=w['physical_gate_passed'],hard=w['hard'],
                teacher_RH_nonzero_daily_count=int(np.sum(mask & (np.abs(teacher)>1e-12))),teacher_RH_daily_count=int(mask.sum()),
                teacher_RH_total=float(np.sum(np.where(mask,teacher,0))),NPP_identity=w['source_npp_identity']))
    tables=[]
    fmt=lambda v:'—' if v is None else f'{v:.6g}'
    pair=lambda d:fmt(d['bias'])+' / '+fmt(d['rmse'])
    for step in (8,16):
        for horizon in (7,30):
            selected={a:[w for w in panels[step,a]['windows'] if w['horizon']==horizon] for a in ARMS}
            group={};output['groups'][f'step{step}_T{horizon}']=group
            for kind in ('fluxes','stocks'):
                labels=('daily','cumulative') if kind=='fluxes' else ('level','final','trajectory_change','daily_change')
                group[kind]={}
                for key in selected['reference_a'][0]['physical_curves'][kind]:
                    result={};group[kind][key]=result
                    for label in labels:
                        triples={a:curve(selected[a],kind,key,label) for a in ARMS}
                        for arm in ARMS:
                            if not np.array_equal(triples[arm][1],triples['reference_a'][1]):raise ValueError('Teacher changed between arms')
                        metric={a:stats(*triples[a]) for a in ARMS};result[label]=dict(accuracy=metric,output_difference={},accuracy_RMSE_difference={})
                        for left,right in (('candidate','reference_a'),('candidate','reference_b'),('reference_b','reference_a')):
                            p,t,m=triples[left];rp,rt,rm=triples[right];name=left+'-'+right
                            result[label]['output_difference'][name]=stats(p,rp,m & rm)
                            result[label]['accuracy_RMSE_difference'][name]=None if metric[left]['rmse'] is None or metric[right]['rmse'] is None else metric[left]['rmse']-metric[right]['rmse']
                title=f'Step {step}, T{horizon}: '+('fluxes' if kind=='fluxes' else 'stocks')
                tables.extend(['',f'### {title}',''])
                if kind=='fluxes':
                    tables.extend(['Daily bias / RMSE: gC m⁻² day⁻¹. Cumulative RMSE: gC m⁻². ΔRMSE is candidate minus each original; B−A is direct daily output RMS.','',
                        '| Flux | A daily | Candidate daily | B daily | ΔRMSE C−A / C−B | B−A RMS | Cumulative RMSE A / C / B | Daily / total counts |',
                        '|---|---:|---:|---:|---:|---:|---:|---:|'])
                    for key,r in group[kind].items():
                        d=r['daily'];m=d['accuracy'];deltas=d['accuracy_RMSE_difference'];c=r['cumulative']['accuracy']
                        tables.append('| '+ ' | '.join((key,pair(m['reference_a']),pair(m['candidate']),pair(m['reference_b']),fmt(deltas['candidate-reference_a'])+' / '+fmt(deltas['candidate-reference_b']),fmt(d['output_difference']['reference_b-reference_a']['rmse']), ' / '.join(fmt(c[a]['rmse']) for a in ARMS),str(m['candidate']['count'])+' / '+str(c['candidate']['count'])))+' |')
                else:
                    tables.extend(['Final bias / RMSE and level trajectory RMSE: gC m⁻², with the public joint prediction/Teacher/common-initial mask. Daily-change RMSE: gC m⁻² day⁻¹. Common initial makes final stock error equal endpoint change error.','',
                        '| Stock | A final | Candidate final | B final | Level RMSE A / C / B | Δlevel RMSE C−A / C−B | B−A level RMS | Daily-change RMSE A / C / B | Level / final counts |',
                        '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
                    for key,r in group[kind].items():
                        f=r['final']['accuracy'];l=r['level'];m=l['accuracy'];d=l['accuracy_RMSE_difference'];dc=r['daily_change']['accuracy']
                        tables.append('| '+' | '.join((key,pair(f['reference_a']),pair(f['candidate']),pair(f['reference_b']),' / '.join(fmt(m[a]['rmse']) for a in ARMS),fmt(d['candidate-reference_a'])+' / '+fmt(d['candidate-reference_b']),fmt(l['output_difference']['reference_b-reference_a']['rmse']),' / '.join(fmt(dc[a]['rmse']) for a in ARMS),str(m['candidate']['count'])+' / '+str(f['candidate']['count'])))+' |')
    output['reduction_seconds']=time.perf_counter()-began
    (args.family/'science-comparison.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    (args.family/'science-tables.md').write_text('\n'.join(tables)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=output['status'],reduction_seconds=output['reduction_seconds'])))

if __name__=='__main__':main()
