"""Original deterministic artificial example, never calibrated to real research.

The small specification demonstrates software integration, not the V3 model
inventory. Every model ID is DEMO-prefixed and all output carries that label.
"""
import json
import numpy as np
import pandas as pd

from .pipeline import DEMO_LABEL, PipelineError, atomic_json, sha


def artificial_native():
    t=pd.date_range('2025-12-21T23:00Z',periods=480*4,freq='15min')
    hour=t.floor('h'); midpoint=hour+pd.Timedelta(minutes=30)
    r=(midpoint-pd.Timestamp('2025-12-31T23:00Z')).total_seconds()/86400
    post=(hour>=pd.Timestamp('2025-12-31T23:00Z')).astype(float)
    # Known local jump 4, pre slope .7 and slope change -.2. All positive,
    # artificial country offsets are not fitted to any real coefficient.
    y=200+4*post+.7*r-.2*post*r
    quarter=np.tile([-.3,.3,-.1,.1],len(t)//4)
    return t,y+quarter


def artificial_weather(times):
    from .weather import weighted_fields
    semantics={'variables':{}}
    for name,scale,offset,shift,out in [('t2m',1.,-273.15,0,'temp_c'),('u10',1.,0.,0,None),
            ('v10',1.,0.,0,None),('ssrd',1/3600,0.,-1,'ssrd_processed'),('tp',1000.,0.,-1,'tp_processed')]:
        semantics['variables'][name]=dict(scale=scale,offset=offset,delivery_start_offset_hours=shift,output=out)
    n=len(times); grids={'t2m':np.tile([273.15,293.15],(n,1)), 'u10':np.tile([3.,-3.],(n,1)),
        'v10':np.tile([4.,-4.],(n,1)), 'ssrd':np.tile([7200.,3600.],(n,1)), 'tp':np.tile([.002,.006],(n,1))}
    fields=weighted_fields(grids,[.25,.75],semantics)
    blocks=[]
    for geo in ['AT','CH','SI','GR','ME','IT-NORD','IT-CSUD','IT-SUD']:
        block=pd.DataFrame({'timestamp_utc':times,'geography':geo})
        for name,(values,shift) in fields.items():
            # Accumulated validity is next hour, then the frozen -1h rule.
            validity=times-pd.Timedelta(hours=shift)
            assert (validity+pd.Timedelta(hours=shift)).equals(times)
            block[name]=values
        blocks.append(block)
    return pd.concat(blocks,ignore_index=True)


def run_stage(run,stage,*,audience='public'):
    if stage=='validate-inputs':
        def validate(out):
            t,y=artificial_native()
            if not t.is_unique or not np.isfinite(y).all(): raise PipelineError('INVALID_DEMO_FIXTURE')
            atomic_json(out/'fixture.json',dict(label=DEMO_LABEL,rows=len(t),source='analytic artificial equations',
                target_jump=4.,weights={'GR_IT':.25,'AT_IT':.75},has_private_inputs=False))
            return dict(label=DEMO_LABEL,private_inputs_read=0)
        return run.execute(stage,validate)
    if stage=='build-weather':
        def weather(out):
            t,_=artificial_native();d=artificial_weather(pd.DatetimeIndex(t.floor('h').unique()))
            d.to_parquet(out/'weather.parquet',index=False)
            return dict(rows=len(d),label=DEMO_LABEL)
        return run.execute(stage,weather)
    if stage=='build-panel':
        def panel(out):
            from .market import integrate_native
            from .panel import assemble_panel
            t,y=artificial_native();blocks=[]
            for i,cid in enumerate(['AT_IT','CH_IT','SI_IT','GR_IT','ME_IT']):
                native=pd.DataFrame(dict(timestamp_utc=t,value=y+i*10,duration_minutes=15))
                h=integrate_native(native.timestamp_utc,native.value,native.duration_minutes,'commercial_net_MW')
                d=h.reset_index()[['timestamp_utc','commercial_net_MW']].copy()
                d['corridor_id']=cid;d['physical_net_MW']=d.commercial_net_MW*.95
                d['origin_price']=60.;d['destination_price']=70.;d['spread_center']=10.
                blocks.append(d)
            d=assemble_panel(pd.concat(blocks,ignore_index=True),pd.read_parquet(run.artifact('build-weather/weather.parquet')))
            d.to_parquet(out/'panel.parquet',index=False)
            return dict(rows=len(d),label=DEMO_LABEL)
        return run.execute(stage,panel)
    if stage=='build-synthetic':
        def synthetic(out):
            from .synthetic import construct_paths
            panel=pd.read_parquet(run.artifact('build-panel/panel.parquet'))
            weights={'DEMO_FIXED':{'GR_IT':.25,'AT_IT':.75}}
            d=construct_paths(panel,weights)
            d.to_parquet(out/'synthetic.parquet',index=False)
            return dict(weights_reestimated=False,label=DEMO_LABEL)
        return run.execute(stage,synthetic)
    if stage=='estimate-main':
        def estimate(out):
            from .design import calendar,local
            from .estimators import fit
            d=pd.read_parquet(run.artifact('build-panel/panel.parquet'))
            q=local(calendar(d[d.corridor_id.eq('ME_IT')]),10)
            f=fit(q,'scheduled_export_proxy',['post','r','post_r'],fe=('hour','weekday'),weight='triangular')
            cov=f.covariance(168);terms=[]
            for i,name in enumerate(f.names):
                z=f.contrast(cov,{name:1})
                terms.append(dict(term=name,coefficient=z['coefficient'],SE=z['se'],CI_low=z['ci_low'],CI_high=z['ci_high'],p_value=z['p_value']))
            z=f.contrast(cov,{'post':1})
            model=dict(model_id='DEMO_RDIT_ME',coefficient_name='post',estimate=z['coefficient'],SE=z['se'],
                CI_low=z['ci_low'],CI_high=z['ci_high'],p_value=z['p_value'],N=len(f.y),
                status='ARTIFICIAL_ESTIMATE',coefficients=terms,columns=f.names,covariance=cov.tolist())
            payload=dict(mode='demo',label=DEMO_LABEL,models=[model],private_inputs_read=0)
            atomic_json(out/'results.json',payload)
            pd.DataFrame(terms).to_csv(out/'coefficients.csv',index=False)
            np.savez_compressed(out/'covariance.npz',DEMO_RDIT_ME=cov)
            return dict(models=1,label=DEMO_LABEL)
        return run.execute(stage,estimate)
    if stage=='verify-main':
        def verify(out):
            payload=json.loads(run.artifact('estimate-main/results.json').read_text());m=payload['models'][0]
            if abs(m['estimate']-4.)>1e-9: raise PipelineError('DEMO_KNOWN_JUMP_FAILED')
            cov=np.asarray(m['covariance'])
            if not np.allclose(cov,cov.T,rtol=0,atol=1e-12): raise PipelineError('DEMO_COVARIANCE_ASYMMETRIC')
            result=dict(status='PASS',scope='artificial known-coefficient validation only',label=DEMO_LABEL,
                        known_jump=4.,absolute_error=abs(m['estimate']-4.),full_scientific_replication=False)
            atomic_json(out/'verification.json',result);return result
        return run.execute(stage,verify)
    if stage=='report':
        def report(out):
            from .reporting.reports import render_report
            p=run.artifact('estimate-main/results.json')
            from .release import RightsRegistry, demo_registry_for_run
            base = RightsRegistry(run.project_root/'manifests/release_rights.toml')
            registry, artifact_id = demo_registry_for_run(p, run.receipt, run.path, base)
            return render_report(p,out,mode='demo',audience=audience,expected_sha=sha(p),
                                 rights_registry=registry,artifact_id=artifact_id)
        return run.execute(stage,report,parameters={'audience':audience})
    raise PipelineError('DEMO_SPEC_NOT_REGISTERED')
