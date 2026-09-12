"""Physical-unit window metrics alongside the existing absolute science gates."""
import numpy as np


def summarize_errors(predicted,expected,defined):
    p,t,m = np.asarray(predicted),np.asarray(expected),np.asarray(defined,dtype=bool)
    nonfinite = int(np.sum(m & (~np.isfinite(p) | ~np.isfinite(t))))
    good = m & np.isfinite(p) & np.isfinite(t)
    error = (p[good]-t[good]).reshape(-1)
    return {'count':len(error),'nonfinite_defined':nonfinite,
            'rmse':float(np.sqrt(np.mean(error**2))) if len(error) else None,
            'bias':float(np.mean(error)) if len(error) else None,
            'mae':float(np.mean(abs(error))) if len(error) else None,
            'p95_absolute_error':float(np.quantile(abs(error),.95)) if len(error) else None}


def physical_window(fluxes,expected_fluxes,stocks,expected_stocks,start_stocks):
    output = {'fluxes':{},'stocks':{}}
    for name,p in fluxes.items():
        t = expected_fluxes[name]
        pv,tv = np.asarray(p.value),np.asarray(t.value)
        mask = np.asarray(p.defined)&np.asarray(t.defined)
        complete = np.all(mask,axis=0)
        output['fluxes'][name] = {'predicted_daily':pv,'teacher_daily':tv,'defined':mask,
            'daily_error':summarize_errors(pv,tv,mask),
            'predicted_total':np.sum(np.where(mask,pv,0.),axis=0),
            'teacher_total':np.sum(np.where(mask,tv,0.),axis=0),'complete':complete}
    for name,p in stocks.items():
        t = expected_stocks[name]
        pv,tv = np.asarray(p.value),np.asarray(t.value)
        start = start_stocks[name]
        mask = np.asarray(p.defined)&np.asarray(t.defined)
        complete = mask[-1]&np.asarray(start.defined)[0]
        output['stocks'][name] = {'predicted_daily':pv,'teacher_daily':tv,'defined':mask,
            'predicted_final':pv[-1],'teacher_final':tv[-1],
            'predicted_change':pv[-1]-np.asarray(start.value)[0],
            'teacher_change':tv[-1]-np.asarray(start.value)[0],'complete':complete}
    return output


def summarize_windows(windows):
    if not windows:
        return {}
    result = {'fluxes':{},'stocks':{}}
    for name in windows[0]['fluxes']:
        rows = [w['fluxes'][name] for w in windows]
        p,t,m = (np.concatenate([np.asarray(r[key]).reshape(-1) for r in rows])
                 for key in ('predicted_total','teacher_total','complete'))
        daily_p,daily_t,daily_m = (np.concatenate([np.asarray(r[key]).reshape(-1) for r in rows])
                                 for key in ('predicted_daily','teacher_daily','defined'))
        result['fluxes'][name] = {'cumulative_gC_m2':summarize_errors(p,t,m),
            'daily_gC_m2_day':summarize_errors(daily_p,daily_t,daily_m),
            'teacher_nonzero_totals':int(np.sum(np.asarray(m,dtype=bool)&(abs(t)>1e-12)))}
    for name in windows[0]['stocks']:
        rows = [w['stocks'][name] for w in windows]
        m = np.concatenate([np.asarray(r['complete']).reshape(-1) for r in rows])
        result['stocks'][name] = {}
        for kind,pkey,tkey in [('final_gC_m2','predicted_final','teacher_final'),('change_gC_m2','predicted_change','teacher_change')]:
            p,t = (np.concatenate([np.asarray(r[key]).reshape(-1) for r in rows]) for key in (pkey,tkey))
            result['stocks'][name][kind] = summarize_errors(p,t,m)
    return result


def complete_metric_gate(physical,science,horizon,window_count):
    """Missing or nonfinite observations cannot become a successful screen."""
    if window_count != 8 or len(science['keys']) != 13:
        return False
    if len(physical['fluxes']) != 8 or len(physical['stocks']) != 5:
        return False
    if any(row['evaluated_values'] != window_count for row in science['keys'].values()):
        return False
    for row in physical['fluxes'].values():
        for key,count in [('cumulative_gC_m2',window_count),('daily_gC_m2_day',window_count*horizon)]:
            if row[key]['nonfinite_defined'] or row[key]['count'] != count:
                return False
    for row in physical['stocks'].values():
        for value in row.values():
            if value['nonfinite_defined'] or value['count'] != window_count:
                return False
    return True
