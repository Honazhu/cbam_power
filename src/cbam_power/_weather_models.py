"""Port: V3/src/reused/weather_models.py. Frozen A/B/C; explicit context."""
import json
import numpy as np
import pandas as pd

from ._engine_context import Y,ACTUALS,sha,sample_hash,canonical_sample_hash
from pathlib import Path
from .design import calendar,local,seasonal,supports,F2_TERMS,F2_CONTRASTS
from .estimators import fit,NotEstimable,joint_covariance,contrast
from ._engine_context import FIELDS,DESTINATIONS

VARIANTS=['FULL_WEATHER_QUADRATIC','WEATHER_LINEAR_ONLY']

def features(g,centers,quadratic):
    out=pd.DataFrame(index=g.index)
    for field in FIELDS:
        out[field]=g[field]-centers[field]
        if quadratic and field!='tp_processed':out[field+'_squared']=out[field]**2
    return out

def synthetic_features(blocks,weights):
    """Nonlinear features are already formed separately at donor geographies."""
    assert set(blocks)==set(weights) and min(weights.values())>=0 and np.isclose(sum(weights.values()),1)
    first=next(iter(blocks.values()));out=pd.DataFrame(0.,index=first.index,columns=first.columns)
    valid=pd.Series(True,index=first.index)
    for key,w in weights.items():
        b=blocks[key].reindex(index=first.index,columns=first.columns)
        valid &= b.notna().all(axis=1);out += w*b
    return out.where(valid,axis=0)

def deduplicate(d,columns):
    keep=[];drops=[]
    for col in columns:
        if d[col].nunique(dropna=False)==1:
            drops.append(dict(column=col,reason='CONSTANT_COLLINEAR_WITH_INTERCEPT',duplicate_of='const'));continue
        same=next((x for x in keep if d[col].equals(d[x])),None)
        if same:drops.append(dict(column=col,reason='EXACT_DUPLICATE',duplicate_of=same))
        else:keep.append(col)
    return keep,drops

def assert_pair(b,c):
    tests=dict(timestamp_equality=b.sample.timestamp_utc.equals(c.sample.timestamp_utc),N_equality=len(b.y)==len(c.y),yearly_N_equality=b.sample.groupby('year').size().equals(c.sample.groupby('year').size()),outcome_equality=np.array_equal(b.y,c.y),kernel_equality=np.array_equal(b.weights,c.weights),FE_references_equality=b.audit['references']==c.audit['references'],core_design_equality=np.array_equal(b.x,c.x[:,[c.names.index(x) for x in b.names]]),pre_post_equality=(b.audit['n_pre'],b.audit['n_post'])==(c.audit['n_pre'],c.audit['n_post']),endpoint_equality=(b.sample.timestamp_utc.min(),b.sample.timestamp_utc.max())==(c.sample.timestamp_utc.min(),c.sample.timestamp_utc.max()))
    assert all(tests.values()),tests
    return tests

def scaled_independent_qa(f,weather_columns,pre_mask,lag):
    """Independent statsmodels WLS and explicit elapsed-hour HAC after units change.

    No new column, transformation or dropped regressor.  Scale constants are
    computed from the pre-cutoff reference only; the outcome is never scaled.
    """
    import statsmodels.api as sm  # independent QA oracle; never the scientific estimator
    scales=np.ones(len(f.names))
    for col in weather_columns:
        i=f.names.index(col);s=np.std(f.x[np.asarray(pre_mask),i],ddof=1)
        if np.isfinite(s) and s>0:scales[i]=s
    z=f.x/scales
    r=sm.WLS(f.y,z,weights=f.weights).fit()
    beta=r.params/scales
    scores=z*(f.weights*r.resid)[:,None]
    meat=scores.T@scores
    times=pd.DatetimeIndex(f.sample.timestamp_utc).as_unit('ns').asi8
    for h in range(1,lag+1):
        target=times-h*3600*10**9;left=np.searchsorted(times,target)
        right=np.arange(len(times));valid=left<len(times)
        valid[valid] &= times[left[valid]]==target[valid]
        a=scores[left[valid]].T@scores[right[valid]]
        meat+=(1-h/(lag+1))*(a+a.T)
    bread=np.asarray(r.normalized_cov_params)
    cov=(bread@meat@bread)*(len(f.y)/(len(f.y)-len(f.names)))
    cov=cov/scales[:,None]/scales[None,:]
    original=f.covariance(lag);core=['post'] if 'r' in f.names else [x for x in ['D25','D26','spread_c','D25_s','D26_s'] if x in f.names]
    rows=[]
    for term in core:
        i=f.names.index(term);diff=abs(beta[i]-f.beta[i]);sediff=abs(np.sqrt(cov[i,i])-np.sqrt(original[i,i]))
        rows.append(dict(term=term,original_coefficient=f.beta[i],rescaled_coefficient=beta[i],coefficient_abs_difference=diff,original_SE=np.sqrt(original[i,i]),rescaled_SE=np.sqrt(cov[i,i]),SE_abs_difference=sediff,original_condition_number=f.audit['condition_number'],rescaled_condition_number=float(np.linalg.cond(z*np.sqrt(f.weights)[:,None])),same_column_space=True,weather_columns_only_scaled=True,scale_reference='model policy-pre complete-case weather reference',outcome_rescaled=False,independent_path='statsmodels WLS + explicit elapsed-hour score pairs',max_fitted_value_difference=float(np.max(np.abs(f.x@f.beta-z@r.params))),status='PASS' if diff<1e-6 and sediff<1e-6 else 'FAIL'))
    return rows,[dict(column=c,scale=scales[i]) for i,c in enumerate(f.names) if c in weather_columns]

def run(context):
    ROOT=context.output;csv=context.csv;js=context.js
    weather=context.weather.copy()
    assert not weather.duplicated(['timestamp_utc','geography']).any()
    raw=context.panel();source_hash=context.identity('v2_panel');weather_hash=context.identity('weather')
    code_hash=sha(Path(__file__))
    rows=[];coef=[];adjust=[];paired=[];audit=[];centering=[];dropped=[];qa=[];scales=[];status=[];arrays={};index=[];members=[];correlations=[]
    for corridor in ACTUALS:
        origin=corridor[:2];units=[origin,DESTINATIONS[origin]];series=origin+'_actual'
        base=calendar(raw[raw.corridor_id.eq(corridor)])
        for model in ['R1','F1','F2']:
            q=local(base,90) if model=='R1' else seasonal(base)
            terms=['post','r','post_r'] if model=='R1' else ['D25','D26'] if model=='F1' else F2_TERMS
            fes=('hour','weekday') if model=='R1' else ('hour','weekday','month')
            weight='triangular' if model=='R1' else None
            contrastdefs={'post':{'post':1}} if model=='R1' else {'level_2026_minus_2025':{'D26':1,'D25':-1}} if model=='F1' else F2_CONTRASTS
            primary='post' if model=='R1' else 'level_2026_minus_2025' if model=='F1' else 'slope_2026_minus_2025'
            t=pd.DatetimeIndex(q.timestamp_utc)
            pre_t=pd.DatetimeIndex(q.loc[q.post.eq(0) if model=='R1' else q.year.eq(2024),'timestamp_utc'])
            byunit={};complete=pd.Series(True,index=t)
            for unit in units:
                g=weather[weather.geography.eq(unit)].set_index('timestamp_utc').reindex(t)
                complete &= g[FIELDS].notna().all(axis=1)
                pre=g.reindex(pre_t).dropna(subset=FIELDS)
                if len(pre):byunit[unit]=(g,pre[FIELDS].mean().to_dict(),pre)
            # Annual weather uses the full frozen sample; no unbalanced-years fallback.
            if len(byunit)!=2 or not complete.any() or (model!='R1' and not complete.all()):
                spatial_ready=all(u in set(weather.geography) for u in units)
                reason='BLOCKED_SPATIAL_COVERAGE: origin/zone full polygon grid not acquired' if not spatial_ready else 'BLOCKED_ANNUAL_COVERAGE: complete weather unavailable across all three same-season years' if model!='R1' else 'BLOCKED_TEMPORAL_COVERAGE: paired hours or pre-cutoff centering reference incomplete'
                status.append(dict(series=series,model=model,status='BLOCKED_WEATHER_PANEL',expected_N=len(q),complete_N=int(complete.sum()),reason=reason,geographies=';'.join(units)))
                continue
            for unit,(g,c,pre) in byunit.items():
                for field,value in c.items():centering.append(dict(series=series,model=model,geography=unit,variable=field,center=value,N_reference=len(pre),reference='local pre-cutoff' if model=='R1' else '2024 same-season',first_utc=str(pre.index.min()),last_utc=str(pre.index.max()),sample_hash=sample_hash(pre.index),weather_data_hash=weather_hash))
            support_frames,support_bounds=supports(q) if model=='F2' else ({'full':q},[])
            if support_bounds:csv(f'diagnostics/{series}_{model}_weather_support_bounds.csv',support_bounds)
            for support,z in support_frames.items():
                a=fit(z,Y,terms,fes,weight=weight)
                for variant in VARIANTS:
                    wf=pd.concat([features(g,c,variant=='FULL_WEATHER_QUADRATIC').add_prefix(u.replace('-','_')+'_') for u,(g,c,pre) in byunit.items()],axis=1)
                    frame=a.sample.set_index('timestamp_utc').join(wf,how='left');cols=list(wf)
                    include=frame[cols].notna().all(axis=1)
                    members.extend(dict(series=series,model=model,support=support,variant=variant,timestamp_utc=str(tt),included=bool(v)) for tt,v in include.items())
                    frame=frame[include].reset_index();keep,drops=deduplicate(frame,cols)
                    dropped.extend(dict(series=series,model=model,support=support,variant=variant,**v) for v in drops)
                    corr=frame[keep].corr()
                    for i,left in enumerate(keep):
                        for right in keep[i+1:]:correlations.append(dict(series=series,model=model,support=support,variant=variant,column_a=left,column_b=right,correlation=corr.loc[left,right],near_collinearity_abs_gt_095=bool(abs(corr.loc[left,right])>.95),action='DIAGNOSTIC_ONLY_NO_VARIABLE_DELETION'))
                    try:
                        b=fit(frame,Y,terms,fes,weight=weight);c=fit(frame,Y,terms+keep,fes,weight=weight)
                        checks=assert_pair(b,c)
                    except NotEstimable as exc:
                        status.append(dict(series=series,model=model,support=support,variant=variant,status='BLOCKED_WEATHER_MODEL',reason=exc.status));continue
                    paired.append(dict(series=series,model=model,support=support,variant=variant,status='PASS',N_A=len(a.y),N_B=len(b.y),N_C=len(c.y),N_pre_B=b.audit['n_pre'],N_post_B=b.audit['n_post'],sample_hash_B=b.sample_hash,sample_hash_C=c.sample_hash,HAC_identical=True,bandwidth_identical=True,donut_identical=True,**checks))
                    for arm,f in [('A',a),('B',b),('C',c)]:
                        mid=f'V2F__{series}__{model}__{support}__{variant}__{arm}'
                        context.capture_sample(mid,f.sample.timestamp_utc,f)
                        meta=dict(model_id=mid,series=series,corridor=corridor,model=model,support=support,variant=variant,arm=arm,weather_version='NO_WEATHER' if arm!='C' else variant,outcome=Y,weight_version='actual',data_hash=source_hash,weather_data_hash=weather_hash,code_version_hash=code_hash,sample_hash=f.sample_hash,FE='+'.join(fes),HAC='elapsed_hours',bandwidth=90 if model=='R1' else 0,donut=0,kernel='triangular' if model=='R1' else 'uniform',status='WEATHER_ADJUSTED_RUN' if arm=='C' else 'NO_WEATHER_BASELINE',N=len(f.y),N_pre=f.audit['n_pre'],N_post=f.audit['n_post'],year_N=json.dumps({str(k):int(v) for k,v in f.sample.groupby('year').size().items()},sort_keys=True),rank=f.audit['rank'],condition_number=f.audit['condition_number'],r2=f.audit['r2'],r2_type=f.audit['r2_type'],FE_reference=json.dumps(f.audit['references']),first_utc=str(f.sample.timestamp_utc.min()),last_utc=str(f.sample.timestamp_utc.max()))
                        meta.update(sample_hash_utc_ns=canonical_sample_hash(f.sample.timestamp_utc),sample_hash_integer_unit=str(pd.DatetimeIndex(f.sample.timestamp_utc).dtype))
                        audit.append(dict(meta,columns=json.dumps(f.names),omitted=json.dumps(drops if arm=='C' else []),absorbed='[]'))
                        for lag in [168,24]:
                            cv=f.covariance(lag);key=mid+'__HAC'+str(lag);arrays[key]=cv;index.append(dict(key=key,columns=json.dumps(f.names)))
                            coef.extend(dict(meta,hac_lag=lag,term=name,**f.contrast(cv,{name:1})) for name in f.names)
                            rows.extend(dict(meta,hac_lag=lag,estimand=name,**f.contrast(cv,con)) for name,con in contrastdefs.items())
                    for lag in [168,24]:
                        joint=joint_covariance([a,b,c],lag);beta=np.r_[a.beta,b.beta,c.beta];offsets=[0,len(a.names),len(a.names)+len(b.names)]
                        key=f'V2F__{series}__{model}__{support}__{variant}__JOINT_HAC{lag}';arrays[key]=joint;index.append(dict(key=key,columns=json.dumps([arm+':'+name for arm,f in [('A',a),('B',b),('C',c)] for name in f.names])))
                        for label,co in [('sample_change',[-1,1,0]),('weather_adjustment',[0,-1,1])]:
                            v=np.zeros(len(beta))
                            for k,f in enumerate([a,b,c]):
                                for term,mult in contrastdefs[primary].items():v[offsets[k]+f.names.index(term)]=co[k]*mult
                            result=contrast(beta,joint,v);baseline=b.contrast(b.covariance(lag),contrastdefs[primary])['coefficient']
                            adjust.append(dict(series=series,model=model,support=support,variant=variant,hac_lag=lag,estimand=primary,contrast=label,relative_weather_adjustment=result['coefficient']/abs(baseline) if label=='weather_adjustment' and baseline else np.nan,joint_cross_covariance_retained=True,**result))
                    reference=c.sample.post.eq(0) if model=='R1' else c.sample.year.eq(2024)
                    checks,s=scaled_independent_qa(c,keep,reference,168)
                    qa.extend(dict(series=series,model=model,support=support,variant=variant,**x) for x in checks)
                    scales.extend(dict(series=series,model=model,support=support,variant=variant,**x) for x in s)
                    status.append(dict(series=series,model=model,support=support,variant=variant,status='WEATHER_ADJUSTED_RUN',N=len(c.y),reason='Frozen bilateral controls; A/B/C paired; full joint elapsed-hour HAC'))
    for name,values in [('weather_abc_results',rows),('weather_coefficients',coef),('weather_adjustments',adjust)]:csv('tables/'+name+'.csv',values)
    for name,values in [('weather_sample_pairing',paired),('weather_model_design',audit),('weather_centering_constants',centering),('weather_omitted_columns',pd.DataFrame(dropped,columns=['series','model','support','variant','column','reason','duplicate_of'])),('weather_numerical_qa',qa),('weather_unit_scales',scales),('weather_model_status',status),('weather_membership',members),('weather_column_correlations',correlations)]:csv('diagnostics/'+name+'.csv',values)
    np.savez_compressed(ROOT/'tables/weather_covariances.npz',**arrays);csv('tables/weather_covariance_index.csv',index)
    assert all(x['status']=='PASS' for x in qa),'WEATHER_NUMERICAL_QA_FAILED'
    js('diagnostics/weather_execution.json',dict(status='PARTIAL' if any(x['status'].startswith('BLOCKED') for x in status) else 'RUN',paired_models=len(paired),numerical_checks=len(qa),numerical_QA='PASS',annual_unbalanced_samples_not_estimated=True,acquisition_status='ERA5_ACQUISITION_VERIFIED',weight_refitting=False))
    print('天气 A/B/C、联合 HAC 与单位缩放数值核验完成。')
    print(pd.DataFrame(rows).query('hac_lag==168 and arm=="C"')[['series','model','variant','estimand','coefficient','se','N']].to_string(index=False))


