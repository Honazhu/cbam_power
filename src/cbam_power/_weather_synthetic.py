"""Port: V3/src/reused/weather_synthetic.py. Frozen weights; explicit context."""
import json
import numpy as np
import pandas as pd
from ._engine_context import Y,sha,sample_hash,canonical_sample_hash
from pathlib import Path
from .design import calendar,local
from .estimators import fit,NotEstimable,joint_covariance,contrast
from ._weather_models import VARIANTS,features,synthetic_features,deduplicate,assert_pair,scaled_independent_qa
from ._engine_context import FIELDS

GEOGRAPHIES={'GR_IT':('GR','IT-SUD'),'AT_IT':('AT','IT-NORD')}

def run(context):
    ROOT=context.output;csv=context.csv;js=context.js
    s=context.synthetic.copy();s=calendar(s[s.weight_version.eq('GY050_final')]).sort_values('timestamp_utc')
    q=local(s,90)
    wr=context.weights.copy()
    wr=wr[(wr.target=='ME_IT')&wr.version.eq('GY050_final')]
    weights=wr.set_index('donor').weight.to_dict()
    assert set(weights)==set(GEOGRAPHIES) and min(weights.values())>=0 and np.isclose(sum(weights.values()),1,rtol=0,atol=1e-12)
    assert wr.trained_max_local_year.eq(2025).all()
    actual=calendar(context.panel().query('corridor_id == "ME_IT"')).set_index('timestamp_utc')
    index=pd.DatetimeIndex(q.timestamp_utc);preindex=pd.DatetimeIndex(q.loc[q.post.eq(0),'timestamp_utc'])
    weather=context.weather.copy()
    annual_times=context.annual_times;annual_coverage=[]
    for unit in ['ME','IT-CSUD','GR','IT-SUD','AT','IT-NORD']:
        g=weather[weather.geography.eq(unit)].set_index('timestamp_utc').reindex(annual_times)
        annual_coverage.append(dict(geography=unit,expected_hours=len(annual_times),complete_hours=int(g[FIELDS].notna().all(axis=1).sum())))
    annual_missing=[x['geography'] for x in annual_coverage if x['complete_hours']!=x['expected_hours']]
    annual_reason='Incomplete annual native bilateral weather: '+','.join(annual_missing) if annual_missing else 'Annual synthetic weather implementation not executed by R1-only module'
    annual_status=[dict(series=name,model=model,status='NOT_RUN',reason=annual_reason,weight_version='GY050_final',evidence_status='DESCRIPTIVE_ONLY_WEAK_PREFIT_WEAK_SUPPORT') for name,model in [('ME_synthetic','F1'),('ME_synthetic','F2'),('G_ME','F1')]]
    csv('diagnostics/weather_synthetic_annual_coverage.csv',annual_coverage)
    raw={};centers=[];coverage=[]
    for unit in ['ME','IT-CSUD','GR','IT-SUD','AT','IT-NORD']:
        g=weather[weather.geography.eq(unit)].set_index('timestamp_utc').reindex(index)
        valid=g[FIELDS].notna().all(axis=1)
        coverage.append(dict(geography=unit,expected_hours=len(index),complete_hours=int(valid.sum()),status='WEATHER_PANEL_COMPLETE' if valid.all() else 'BLOCKED_WEATHER_PANEL'))
        if valid.all():
            pre=g.reindex(preindex);c=pre[FIELDS].mean().to_dict();raw[unit]=(g,c)
            centers.extend(dict(geography=unit,variable=f,center=v,N_reference=len(pre),first_utc=str(preindex.min()),last_utc=str(preindex.max()),sample_hash=sample_hash(preindex),reference='R1 local pre-cutoff only') for f,v in c.items())
    csv('diagnostics/weather_synthetic_required_coverage.csv',coverage)
    incomplete=[x['geography'] for x in coverage if x['status']!='WEATHER_PANEL_COMPLETE']
    if incomplete:
        reason='Native origin/destination ERA5 panel incomplete: '+','.join(incomplete)
        csv('diagnostics/weather_synthetic_model_status.csv',[dict(series=name,model='R2' if name=='G_ME' else 'R1',status='NOT_RUN',reason=reason,weight_version='GY050_final',evidence_status='DESCRIPTIVE_ONLY_WEAK_PREFIT_WEAK_SUPPORT') for name in ['ME_synthetic','G_ME']]+annual_status)
        js('diagnostics/weather_synthetic_execution.json',dict(status='NOT_RUN',reason=reason,weights_refit=False,required_geographies=coverage))
        print('合成ME及G_ME天气未运行：'+reason);return
    rows=[];coefs=[];adjust=[];pairs=[];designs=[];omits=[];numerical=[];scale_rows=[];statuses=list(annual_status);arrays={};covindex=[]
    definitions={}
    for name in ['ME_synthetic','G_ME']:
        frame=q.copy()
        if name=='G_ME':
            aa=actual.reindex(index)
            frame[Y]=aa[Y].to_numpy()-q[Y].to_numpy()
        for variant in VARIANTS:
            uf={u:features(g,c,variant=='FULL_WEATHER_QUADRATIC') for u,(g,c) in raw.items()}
            df={d:uf[o].add_prefix('origin_').join(uf[z].add_prefix('destination_')) for d,(o,z) in GEOGRAPHIES.items()}
            syn=synthetic_features(df,weights).add_prefix('SYN_')
            block=uf['ME'].add_prefix('TARGET_ME_').join(uf['IT-CSUD'].add_prefix('TARGET_CSUD_')).join(syn) if name=='G_ME' else syn
            # Demonstrate the actual feature order with actual data, not only a toy test.
            naive_temperature=sum(weights[d]*uf[GEOGRAPHIES[d][0]]['temp_c'] for d in weights)
            order_difference=float(np.max(np.abs(syn.SYN_origin_temp_c_squared-naive_temperature**2))) if variant=='FULL_WEATHER_QUADRATIC' else None
            definitions[name+'__'+variant]=dict(donor_geographies=GEOGRAPHIES,weights=weights,nonlinear_before_weighting=True,max_difference_from_square_of_weighted_temperature=order_difference,columns=list(block),gap_estimated_directly=name=='G_ME',actual_synthetic_coefficient_identity_claimed=False)
            a=fit(frame,Y,['post','r','post_r'],weight='triangular')
            d=a.sample.set_index('timestamp_utc').join(block,how='left').dropna(subset=list(block)).reset_index();keep,drops=deduplicate(d,list(block))
            omits.extend(dict(series=name,variant=variant,**x) for x in drops)
            try:
                b=fit(d,Y,['post','r','post_r'],weight='triangular')
                c=fit(d,Y,['post','r','post_r',*keep],weight='triangular')
                checks=assert_pair(b,c)
            except NotEstimable as exc:
                statuses.append(dict(series=name,model='R2' if name=='G_ME' else 'R1',variant=variant,status='BLOCKED_WEATHER_MODEL',reason=exc.status));continue
            pairs.append(dict(series=name,model='R2' if name=='G_ME' else 'R1',variant=variant,N_A=len(a.y),N_B=len(b.y),N_C=len(c.y),sample_hash_B=b.sample_hash,sample_hash_C=c.sample_hash,status='PASS',**checks))
            for arm,f in [('A',a),('B',b),('C',c)]:
                mid=f'V2F__{name}__R1_WEATHER__{variant}__{arm}'
                context.capture_sample(mid,f.sample.timestamp_utc,f)
                meta=dict(model_id=mid,series=name,model='R2' if name=='G_ME' else 'R1',variant=variant,support='full',arm=arm,outcome=Y,weight_version='GY050_final',weather_version=variant if arm=='C' else 'NO_WEATHER',data_hash=context.identity("synthetic"),weight_source_hash=context.identity("weights"),weather_data_hash=context.identity("weather"),code_version_hash=sha(Path(__file__)),sample_hash=f.sample_hash,FE='hour+weekday',bandwidth=90,donut=0,kernel='triangular',N=len(f.y),N_pre=f.audit['n_pre'],N_post=f.audit['n_post'],rank=f.audit['rank'],condition_number=f.audit['condition_number'],r2=f.audit['r2'],r2_type=f.audit['r2_type'],year_N=json.dumps({str(k):int(v) for k,v in f.sample.groupby('year').size().items()}),FE_reference=json.dumps(f.audit['references']),status='WEATHER_ADJUSTED_RUN' if arm=='C' else 'NO_WEATHER_BASELINE',evidence_status='RELATIVE_BENCHMARK_DEVIATION_WEAK_SUPPORT' if name=='G_ME' else 'DESCRIPTIVE_ONLY_WEAK_PREFIT_WEAK_SUPPORT',weight_uncertainty='conditional_on_estimated_weights',gap_directly_regressed=name=='G_ME')
                meta.update(sample_hash_utc_ns=canonical_sample_hash(f.sample.timestamp_utc),sample_hash_integer_unit=str(pd.DatetimeIndex(f.sample.timestamp_utc).dtype),actual_data_hash=context.identity("v2_panel"))
                designs.append(dict(meta,columns=json.dumps(f.names),omitted=json.dumps(drops if arm=='C' else [])))
                for lag in [168,24]:
                    cov=f.covariance(lag);key=mid+'__HAC'+str(lag);arrays[key]=cov;covindex.append(dict(key=key,columns=json.dumps(f.names)))
                    rows.append(dict(meta,hac_lag=lag,estimand='post',**f.contrast(cov,{'post':1})))
                    coefs.extend(dict(meta,hac_lag=lag,term=t,**f.contrast(cov,{t:1})) for t in f.names)
            for lag in [168,24]:
                joint=joint_covariance([a,b,c],lag);beta=np.r_[a.beta,b.beta,c.beta];offsets=[0,len(a.names),len(a.names)+len(b.names)];key=f'{name}__{variant}__JOINT_HAC{lag}';arrays[key]=joint;covindex.append(dict(key=key,columns=json.dumps([arm+':'+n for arm,f in [('A',a),('B',b),('C',c)] for n in f.names])))
                for label,co in [('sample_change',[-1,1,0]),('weather_adjustment',[0,-1,1])]:
                    v=np.zeros(len(beta))
                    for k,f in enumerate([a,b,c]):v[offsets[k]+f.names.index('post')]=co[k]
                    result=contrast(beta,joint,v)
                    adjust.append(dict(series=name,model='R2' if name=='G_ME' else 'R1',variant=variant,support='full',estimand='post',contrast=label,hac_lag=lag,joint_cross_covariance_retained=True,**result))
            qa,scales=scaled_independent_qa(c,keep,c.sample.post.eq(0),168)
            numerical.extend(dict(series=name,variant=variant,**x) for x in qa)
            scale_rows.extend(dict(series=name,variant=variant,**x) for x in scales)
            statuses.append(dict(series=name,model='R2' if name=='G_ME' else 'R1',variant=variant,status='WEATHER_ADJUSTED_RUN',reason='All native donor/target weather complete; frozen weights and direct gap',weight_version='GY050_final'))
    for p,x in [('tables/weather_synthetic_abc_results.csv',rows),('tables/weather_synthetic_coefficients.csv',coefs),('tables/weather_synthetic_adjustments.csv',adjust),('diagnostics/weather_synthetic_pairing.csv',pairs),('diagnostics/weather_synthetic_design.csv',designs),('diagnostics/weather_synthetic_omitted.csv',omits),('diagnostics/weather_synthetic_numerical_qa.csv',numerical),('diagnostics/weather_synthetic_unit_scales.csv',scale_rows),('diagnostics/weather_synthetic_centers.csv',centers),('diagnostics/weather_synthetic_model_status.csv',statuses),('tables/weather_synthetic_covariance_index.csv',covindex)]:csv(p,x)
    np.savez_compressed(ROOT/'tables/weather_synthetic_covariances.npz',**arrays)
    assert all(x['status']=='PASS' for x in numerical)
    js('metadata/weather_synthetic_definition.json',definitions)
    js('diagnostics/weather_synthetic_execution.json',dict(status='RUN' if all(x['status']=='WEATHER_ADJUSTED_RUN' for x in statuses) else 'PARTIAL',R1_status='RUN' if all(x['status']=='WEATHER_ADJUSTED_RUN' for x in statuses if x['model'] in ['R1','R2']) else 'PARTIAL',models=statuses,weights_refit=False,required_geographies=coverage,annual_geographies=annual_coverage,full_covariance_retained=True))
    print('ME合成与直接G_ME天气A/B/C完成；权重未重估。')


