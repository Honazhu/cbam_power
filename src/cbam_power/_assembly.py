"""V3 compute.py assembly moved wholly into estimation, never reporting.
Source functions retained with explicit context and fresh-result-only QA separation.
"""
import json
import numpy as np
import pandas as pd
from ._engine_context import Y, ACTUALS, CFG, clean, sha, canonical_sample_hash
from .design import calendar, local
from .estimators import fit
from ._weather_models import features,deduplicate,assert_pair
from ._engine_context import FIELDS

def numeric(result):
    return dict(estimate=float(result['coefficient']),SE=float(result['se']),CI_low=float(result['ci_low']),CI_high=float(result['ci_high']),p_value=float(result['p_value']))

def canonical_id(meta,lag):
    series=meta['series'];fam=meta.get('model',meta.get('model_family'));name=series.split('_')[0]
    version=meta.get('weight_version','actual');outcome=meta.get('outcome',Y)
    bw=float(meta.get('bandwidth_days',meta.get('bandwidth',0)) or 0)
    cutoff=int(meta.get('cutoff',2026) or 2026);support=meta.get('support','full');spec=meta.get('spec','main')
    mid=None
    if outcome==Y and spec=='main' and float(meta.get('donut_days',0) or 0)==0:
        if version=='actual':
            if fam=='R1' and bw==90 and cutoff==2026:mid=f'V3_FE_{name}'
            elif fam=='R1_placebo' and bw==90 and cutoff==2025:mid=f'V3_PLACEBO2025_{name}'
            elif fam=='F1':mid=f'V3_F1_{name}'
            elif fam=='F2':mid=f'V3_F2_{name}_{support}'
        elif version=='GY050_final':
            if series=='ME_synthetic' and fam=='R1' and bw==90:mid='V3_SYNTHETIC_ME'
            elif series=='G_ME' and fam=='R2' and bw==90:mid='V3_GAP_ME'
    if mid is None:mid=meta['model_id'].replace('V2F','V3',1)
    return mid if lag==168 else mid+f'_HAC{lag}'

def notebook_for(fam,weather=False):
    if weather:return '04_weather_rdit.ipynb'
    if fam=='BASIC_RDiT':return '02_basic_rdit.ipynb'
    if fam in ['R1','R2']:return '03_rdit_calendar_fe.ipynb'
    return '06_placebo_same_season_spread.ipynb'

def from_fit(mid,f,series,family,fe,lag=168):
    cov=f.covariance(lag);z=numeric(f.contrast(cov,{'post':1}))
    terms=[dict(term=t,coefficient=float(f.beta[i]),SE=float(np.sqrt(max(0,cov[i,i]))),CI_low=float(f.beta[i]-1.96*np.sqrt(max(0,cov[i,i]))),CI_high=float(f.beta[i]+1.96*np.sqrt(max(0,cov[i,i]))),p_value=f.contrast(cov,{t:1})['p_value']) for i,t in enumerate(f.names)]
    return dict(model_id=mid,notebook=notebook_for(family),series=series,outcome=Y,model_family=family,cutoff=CFG['cutoff'],bandwidth=90,kernel='triangular',HAC_lag=lag,hour_FE='hour' in fe,weekday_FE='weekday' in fe,month_FE='month' in fe,weather=False,support='full',weight_version='actual',sample_start=str(f.sample.timestamp_utc.min()),sample_end=str(f.sample.timestamp_utc.max()),N=len(f.y),pre_N=f.audit['n_pre'],post_N=f.audit['n_post'],coefficient_name='post',status='ESTIMATED_DESCRIPTIVE',coefficients=terms,r2=f.audit['r2'],sample_hash=canonical_sample_hash(f.sample.timestamp_utc),legacy_sample_hash=f.sample_hash,**z)

def coefficient_list(group):
    return [dict(term=str(r.term),coefficient=r.coefficient,SE=r.se,CI_low=r.ci_low,CI_high=r.ci_high,p_value=r.p_value) for r in group.itertuples()]

def assemble_no_weather(context, ID_MAP):
    ROOT=context.output
    payload=json.loads((ROOT/'reporting/empirical_models.json').read_text())
    audits={a['model_id']:a for a in payload['audits']}
    coeff=pd.read_csv(ROOT/'tables/all_coefficients.csv');models=[]
    for (oldid,lag),g in coeff.groupby(['model_id','hac_lag'],sort=False):
        a=audits.get(oldid,{});r=g.iloc[0].to_dict();meta={**r,**a};mid=canonical_id(meta,int(lag));ID_MAP[(oldid,int(lag))]=mid
        family=meta['model'];primary='level_2026_minus_2025' if family=='F1' else 'slope_2026_minus_2025' if family=='F2' else 'post'
        if family=='F2_piecewise':primary='segment_0_slope_26_minus_25'
        match=g[g.term.eq(primary)]
        if len(match)!=1:raise AssertionError(f'Primary estimand {primary} absent/duplicated in {oldid} HAC{lag}')
        v=match.iloc[0].to_dict()
        fes=meta.get('fe','');s=meta['series'];syn=s in ['ME_synthetic','G_ME']
        model=dict(model_id=mid,source_model_id=oldid,notebook='05_synthetic_montenegro.ipynb' if syn else notebook_for(family),series=s,outcome=meta['outcome'],model_family=family,cutoff=f"{int(meta['cutoff'])}-01-01T00:00:00+01:00",bandwidth=meta.get('bandwidth_days',0),donut=meta.get('donut_days',0),kernel=meta['kernel'],HAC_lag=int(lag),hour_FE='hour' in fes.split('+'),weekday_FE='weekday' in fes.split('+'),month_FE='month' in fes.split('+'),week_FE='week' in fes.split('+'),weather=False,support=meta.get('support','full'),weight_version=meta.get('weight_version','actual'),sample_start=meta.get('first_utc'),sample_end=meta.get('last_utc'),N=meta.get('n'),pre_N=meta.get('n_pre'),post_N=meta.get('n_post'),coefficient_name=primary,status=meta.get('status'),coefficients=coefficient_list(g),r2=meta.get('r2'),sample_hash=meta.get('sample_hash_utc_ns',meta.get('sample_hash')),spec=meta.get('spec','main'),interpretation=meta.get('interpretation'),**numeric(v))
        models.append(model)
    return models,payload

def assemble_weather(context,ID_MAP,filename,coeffile,designfile,synthetic=False):
    ROOT=context.output;panel=context.panel
    rows=pd.read_csv(ROOT/'tables'/filename);coeff=pd.read_csv(ROOT/'tables'/coeffile)
    designs=pd.read_csv(ROOT/'diagnostics'/designfile).set_index('model_id');models=[]
    for (oldid,lag),g in rows.groupby(['model_id','hac_lag'],sort=False):
        r=g.iloc[0].to_dict();d=designs.loc[oldid].to_dict();meta={**r,**d};name=meta['series'].split('_')[0];arm=meta['arm'];variant=meta['variant'];fam=meta['model'];support=meta.get('support','full')
        if variant=='FULL_WEATHER_QUADRATIC' and fam in ['R1','R2'] and support=='full':
            stem='V3_GAP_WEATHER_ME' if meta['series']=='G_ME' else 'V3_SYNTHETIC_WEATHER_ME' if meta['series']=='ME_synthetic' else 'V3_WEATHER_'+name
            mid=stem+('' if arm=='C' else '_'+arm)
        else:mid=oldid.replace('V2F','V3',1)
        if lag!=168:mid+=f'_HAC{int(lag)}'
        ID_MAP[(oldid,int(lag))]=mid
        primary='post' if fam in ['R1','R2'] else 'level_2026_minus_2025' if fam=='F1' else 'slope_2026_minus_2025'
        primary_rows=g[g.estimand.eq(primary)]
        if len(primary_rows)!=1:raise AssertionError(f'Weather estimand {primary} absent/duplicated in {oldid} HAC{lag}')
        z=primary_rows.iloc[0].to_dict();cg=coeff[coeff.model_id.eq(oldid)&coeff.hac_lag.eq(lag)];fe=meta.get('FE','hour+weekday')
        model=dict(model_id=mid,source_model_id=oldid,notebook='05_synthetic_montenegro.ipynb' if synthetic else '04_weather_rdit.ipynb',series=meta['series'],outcome=Y,model_family=fam,cutoff=CFG['cutoff'],bandwidth=meta.get('bandwidth',90),kernel=meta.get('kernel','triangular'),HAC_lag=int(lag),hour_FE='hour' in fe.split('+'),weekday_FE='weekday' in fe.split('+'),month_FE='month' in fe.split('+'),weather=arm=='C',weather_variant=variant,arm=arm,support=support,weight_version=meta.get('weight_version','actual'),sample_start=meta.get('first_utc'),sample_end=meta.get('last_utc'),N=meta['N'],pre_N=meta['N_pre'],post_N=meta['N_post'],coefficient_name=primary,status=meta['status'],coefficients=coefficient_list(cg),r2=meta.get('r2'),sample_hash=meta.get('sample_hash_utc_ns',meta.get('sample_hash')),**numeric(z))
        if not model['sample_start']:
            q=local(panel().query('corridor_id == "ME_IT"'),90);model['sample_start']=str(q.timestamp_utc.min());model['sample_end']=str(q.timestamp_utc.max())
        models.append(model)
    return models

def synthetic_statistics(s):
    v=next(x for x in s['validation'] if x['year']==2025);c=s['scorecard'];e=next(x for x in s['convex_support'] if x['year']==2025)
    return dict(model_id='V3_SYNTHETIC_VALIDATION_2025',N=v['n'],rmse=v['rmse'],nrmse=v['nrmse'],mae=v['mae'],bias=v['bias'],validation_r2=c['validation_r2_2025'],rmse_improvement_pct=c['error_improvement_pct_2025'],target_zero_share=v['actual_zero_share'],synthetic_zero_share=v['predicted_zero_share'],outside_envelope_share=e['outside_share'],training_target_sd=v['training_target_sd'],naive_rmse=c['naive_rmse_2025'],status='DESCRIPTIVE_ONLY_WEAK_PREFIT_WEAK_SUPPORT',weight_version='GY050_train2024')

def adjustments(context):
    ROOT=context.output
    d=pd.read_csv(ROOT/'tables/weather_adjustments.csv');q=d[(d.series=='ME_actual')&(d.model=='R1')&(d.variant=='FULL_WEATHER_QUADRATIC')&(d.hac_lag==168)]
    out={}
    for kind in ['weather_adjustment','sample_change']:
        row=q[q.contrast==kind].iloc[0].to_dict();out[kind]=dict(**numeric(row),source_model_ids=['V3_WEATHER_ME_B','V3_WEATHER_ME'] if kind=='weather_adjustment' else ['V3_WEATHER_ME_A','V3_WEATHER_ME_B'],joint_cross_covariance_retained=True)
    return out

def export_covariance(context,ID_MAP,models,basic):
    ROOT=context.output;V3=context.output;machine=context.machine
    index=[]
    specs=[('covariance_index.csv','model_covariances.npz'),('weather_covariance_index.csv','weather_covariances.npz'),('weather_synthetic_covariance_index.csv','weather_synthetic_covariances.npz')]
    for indexfile,arrayfile in specs:
        df=pd.read_csv(ROOT/'tables'/indexfile)
        for r in df.to_dict('records'):
            key=r['key'];lag=r.get('hac_lag');oldid=r.get('model_id')
            if oldid is None and '__HAC' in key:oldid,lag=key.rsplit('__HAC',1)
            mid=ID_MAP.get((oldid,int(lag)),oldid) if oldid and pd.notna(lag) else oldid
            index.append(dict(model_id=mid,covariance_key=key,file_path=str((ROOT/'tables'/arrayfile).relative_to(V3)),columns=r['columns'],HAC_lag=lag,covariance_scope='joint' if 'JOINT' in key.upper() or 'JOINT' in str(oldid).upper() else 'full_parameter_covariance'))
    arrays={}
    for mid,f in basic.items():
        for lag in [168,24]:
            modelid=mid if lag==168 else mid+'_HAC24';arrays[modelid]=f.covariance(lag)
            index.append(dict(model_id=modelid,covariance_key=modelid,file_path='outputs/models/basic_covariances.npz',columns=json.dumps(f.names),HAC_lag=lag,covariance_scope='full_parameter_covariance'))
    np.savez_compressed(V3/'outputs/models/basic_covariances.npz',**arrays)
    lookup={r['model_id']:r for r in index if r['model_id']}
    for model in models:
        if model['model_id'] in lookup:model['covariance_reference']=lookup[model['model_id']]
    machine('T_covariance.csv',index)

def assemble(context):
    ROOT=context.output;V3=context.output;machine=context.machine;panel=lambda:context.frame.copy();ID_MAP={}
    d=panel();models,payload=assemble_no_weather(context,ID_MAP)
    models+=assemble_weather(context,ID_MAP,'weather_abc_results.csv','weather_coefficients.csv','weather_model_design.csv')
    models+=assemble_weather(context,ID_MAP,'weather_synthetic_abc_results.csv','weather_synthetic_coefficients.csv','weather_synthetic_design.csv',True)
    basic=basic_models(d)
    for mid,f in basic.items():
        for lag in [168,24]:models.append(from_fit(mid if lag==168 else mid+'_HAC24',f,mid.rsplit('_',1)[1]+'_actual','BASIC_RDiT',(),lag))
    for mid,f in basic.items():context.capture_sample(mid,f.sample.timestamp_utc,f)
    export_covariance(context,ID_MAP,models,basic)
    weather=context.weather.copy()
    a,b,c=fit_me(d,weather)
    syn=context.synthetic.copy();syn=calendar(syn[syn.weight_version.eq('GY050_final')]);gap=syn.copy();gap[Y]=gap.G_ME
    g=fit(local(gap,90),Y,['post','r','post_r'],weight='triangular')
    plot_data(context,{'V3_BASIC_ME':basic['V3_BASIC_ME'],'V3_FE_ME':a,'V3_WEATHER_ME':c,'V3_GAP_ME':g})
    main=dict(basic_me='V3_BASIC_ME',fe_me='V3_FE_ME',weather_me='V3_WEATHER_ME',synthetic_me='V3_SYNTHETIC_ME',gap_me='V3_GAP_ME',gap_weather='V3_GAP_WEATHER_ME',placebo_me='V3_PLACEBO2025_ME',f1_me='V3_F1_ME',f2_me='V3_F2_ME_full')
    for cid in ['AT','CH','SI','GR','ME']:
        main['basic_'+cid]='V3_BASIC_'+cid;main['fe_'+cid]='V3_FE_'+cid
        main['weather_'+cid]='V3_WEATHER_'+cid;main['f1_'+cid]='V3_F1_'+cid;main['f2_'+cid]='V3_F2_'+cid+'_full'
    for support in ['minmax','p5_p95','p10_p90']:main['f2_me_'+support]='V3_F2_ME_'+support
    s=context.synthetic_summary
    blocked=[]
    for fn in ['weather_model_status.csv','weather_synthetic_model_status.csv']:
        for row in pd.read_csv(ROOT/'diagnostics'/fn).to_dict('records'):
            if row['status'] not in ['WEATHER_ADJUSTED_RUN']:
                row['implementation_status']='NOT_IMPLEMENTED_ANNUAL_SYNTHETIC_WEATHER' if row['series'] in ['ME_synthetic','G_ME'] else 'IMPLEMENTED_BUT_BLOCKED_ANNUAL_WEATHER_COVERAGE'
                blocked.append(row)
    blocked+=payload['explicit_status']
    result=dict(version='V3_FORMAL_CBAM_1',panel=dict(path='data/processed/v3_hourly_panel.parquet',rows=len(d),columns=len(d.columns),corridors=5,hours_per_corridor=int(d.groupby('corridor_id').size().iloc[0]),first_utc=str(d.timestamp_utc.min()),last_utc=str(d.timestamp_utc.max()),timezone='UTC',calendar_timezone='Europe/Rome',sha256=context.identity("panel")),models=models,main=main,synthetic_validation=synthetic_statistics(s),weights={k:s['weights'][k] for k in ['GY050_train2024','GY050_final']},weight_learning=dict(objective='GY050 convex matching: .5 generation-share discrepancy plus .5 averaged flow-feature-block discrepancy',pure_hourly_least_squares_is_motivation_only=True,pre_policy_years=[2024,2025],numerically_refitted_and_verified=False,fixed_weights_only=True,uncertainty='All regression inference conditional on fitted frozen weights'),limitations=['Observed time discontinuities are descriptive and cannot isolate the policy from concurrent market changes.','2025 placebo has a non-zero discontinuity; calendar FE do not remove hydrology, outages, capacity or annual-contract effects.','Annual bilateral weather coverage is incomplete; annual synthetic-weather models additionally have no implemented estimator in the reused R1-only module.','Synthetic validation is weak; support and zero-state mismatch prevent interpreting the benchmark as an untreated causal counterfactual.','Same-season F1 compares Jan 1–Sep 16 means; it establishes neither recovery nor no policy effect.','F2 is a conditional spread-flow association in MW/(EUR/MWh), not an elasticity or causal supply curve.','Weights use the frozen GY050 feature objective, which differs from the pure hourly least-squares teaching formula in the request.','Input weather and generation vintages are retrospective; no claim to a real-time forecast.'],blocked_models=blocked,**adjustments(context))
    result['replication']={'status':'NOT_YET_VERIFIED','scope':'AUTHORIZED FULL NUMERICAL REPLICATION: NOT YET VERIFIED; run independent verify-baseline'}
    endpoints=[m['model_id'] for m in models if not m.get('sample_start') or not m.get('sample_end') or str(m.get('sample_start'))=='nan' or str(m.get('sample_end'))=='nan']
    if endpoints:raise AssertionError('Missing model endpoints: '+str(endpoints))
    result=clean(result)
    result_path=V3/'results.json'
    temporary=result_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False,default=str))
    temporary.replace(result_path)
    # Formal catalog: one row per fitted design/HAC estimand, plus explicit non-run rows.
    flat=[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in m.items() if k!='coefficients'} for m in models]
    validation_sample=d[d.corridor_id.eq('ME_IT')&d.year.eq(2025)]
    flat.append(dict(model_id='V3_SYNTHETIC_VALIDATION_2025',notebook='05_synthetic_montenegro.ipynb',series='ME_synthetic',outcome=Y,model_family='SYNTHETIC_VALIDATION',cutoff=None,bandwidth=None,kernel=None,HAC_lag=None,hour_FE=False,weekday_FE=False,month_FE=False,weather=False,weight_version='GY050_train2024',sample_start=str(validation_sample.timestamp_utc.min()),sample_end=str(validation_sample.timestamp_utc.max()),N=len(validation_sample),pre_N=len(validation_sample),post_N=0,coefficient_name=None,estimate=None,SE=None,CI_low=None,CI_high=None,p_value=None,status='DESCRIPTIVE_METRICS',sample_hash=canonical_sample_hash(validation_sample.timestamp_utc),diagnostic_metrics=json.dumps(result['synthetic_validation'])))
    for i,row in enumerate(blocked):
        flat.append(dict(model_id=f'V3_NOT_RUN_{i:03d}',notebook='04_weather_rdit.ipynb' if 'weather' in str(row).lower() else '06_placebo_same_season_spread.ipynb',series=row.get('series'),model_family=row.get('model'),status=row.get('status'),reason=row.get('reason'),implementation_status=row.get('implementation_status')))
    pd.DataFrame(flat).to_csv(V3/'model_catalog.csv',index=False)
    co=[]
    for m in models:
        for term in m['coefficients']:co.append(dict(model_id=m['model_id'],N=m['N'],HAC_lag=m['HAC_lag'],**term))
    machine('T_coefficients.csv',co)
    machine('T_model_sample_summary.csv',[{k:v for k,v in x.items() if k not in ['covariance_reference']} for x in flat])
    machine('T_main_results.csv',[{k:v for k,v in m.items() if k not in ['coefficients','covariance_reference']} for m in models if m['model_id'] in main.values()])
    machine('T_weather_ABC.csv',[{k:v for k,v in m.items() if k not in ['coefficients','covariance_reference']} for m in models if m['model_id'] in ['V3_WEATHER_ME_A','V3_WEATHER_ME_B','V3_WEATHER_ME','V3_GAP_WEATHER_ME_A','V3_GAP_WEATHER_ME_B','V3_GAP_WEATHER_ME']])
    machine('T_weather_adjustments.csv',[dict(contrast=k,**v) for k,v in adjustments(context).items()])
    print('V3 models:',len(models),'replication:',result['replication']['status'],flush=True)
    print(pd.DataFrame([m for m in models if m['model_id'] in main.values()])[['model_id','estimate','SE','CI_low','CI_high','N']].to_string(index=False),flush=True)
    context.export_samples(models,ID_MAP)
    return result

def basic_models(panel):
    return {f'V3_BASIC_{cid[:2]}':fit(local(calendar(panel[panel.corridor_id.eq(cid)]),90),Y,['post','r','post_r'],fe=(),weight='triangular') for cid in ACTUALS}

def plot_data(context,fits):
    machine=context.machine
    observations=[];lines=[]
    for mid,f in fits.items():
        core=['const','post','r','post_r'];ix=[f.names.index(c) for c in core]
        signal=f.x[:,ix]@f.beta[ix];nuisance=f.x@f.beta-signal
        for i,r in enumerate(f.sample.itertuples()):
            observations.append(dict(model_id=mid,timestamp_utc=str(r.timestamp_utc),r_days=r.r,Post=r.post,raw_outcome=f.y[i],adjusted_outcome=f.y[i]-nuisance[i],fitted=float(f.x[i]@f.beta),adjusted_fitted=signal[i],plot_reference='Calendar at omitted category; weather at policy-pre centered zero; fitted nuisance contribution removed'))
        cv=f.covariance(168)
        for post,xgrid in [(0,np.linspace(-90,-.001,100)),(1,np.linspace(.001,90,100))]:
            for r in xgrid:
                x=np.zeros(len(f.names));x[ix]=[1,post,r,post*r]
                estimate=float(x@f.beta);se=np.sqrt(max(0,float(x@cv@x)))
                lines.append(dict(model_id=mid,r_days=r,Post=post,adjusted_fitted=estimate,CI_low=estimate-1.96*se,CI_high=estimate+1.96*se))
    d=pd.DataFrame(observations);machine('T_rdit_plot_data.csv',d)
    d['bin_day']=np.floor(d.r_days)
    bins=d.groupby(['model_id','bin_day'],as_index=False).agg(r_days=('r_days','mean'),adjusted_outcome=('adjusted_outcome','mean'),raw_outcome=('raw_outcome','mean'),N=('r_days','size'))
    machine('T_rdit_binned.csv',bins);machine('T_rdit_lines.csv',lines)

def fit_me(frame,weather):
    q=local(calendar(frame[frame.corridor_id.eq('ME_IT')]),90)
    times=pd.DatetimeIndex(q.timestamp_utc);pre=pd.DatetimeIndex(q.loc[q.post.eq(0),'timestamp_utc'])
    blocks=[]
    for unit in ['ME','IT-CSUD']:
        w=weather[weather.geography.eq(unit)].set_index('timestamp_utc').reindex(times)
        centers=w.reindex(pre)[FIELDS].mean().to_dict()
        blocks.append(features(w,centers,True).add_prefix(unit.replace('-','_')+'_'))
    wf=pd.concat(blocks,axis=1);d=q.set_index('timestamp_utc').join(wf).dropna(subset=list(wf)).reset_index();keep,_=deduplicate(d,list(wf))
    a=fit(q,Y,['post','r','post_r'],weight='triangular');b=fit(d,Y,['post','r','post_r'],weight='triangular');c=fit(d,Y,['post','r','post_r',*keep],weight='triangular')
    assert_pair(b,c)
    return a,b,c
