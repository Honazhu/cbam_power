"""Independent artificial mathematical checks; authorized test is inventory only."""
import json,os
import numpy as np
import pandas as pd
import pytest
from cbam_power import estimators as e
from cbam_power import design,models
from cbam_power._weather_models import features,synthetic_features,deduplicate,assert_pair


def frame(n=1000):
    t=pd.date_range('2025-12-12T03:00Z',periods=n,freq='h')
    d=design.local(design.calendar(pd.DataFrame({'timestamp_utc':t})),90)
    d['y']=5+3*d.post+.1*d.r-.03*d.post_r+.05*d.hour+.07*d.weekday
    return d


def test_elapsed_hour_hac_independent_all_pairs():
    times=pd.to_datetime(['2026-01-01T00:00Z','2026-01-01T01:00Z','2026-01-01T04:00Z','2026-01-01T06:00Z'])
    values=np.array([[1.,2.],[-2.,.5],[3.,-1.],[.2,4.]])
    lag=3;expected=np.zeros((2,2))
    for i,left in enumerate(times):
        for j,right in enumerate(times):
            distance=abs((left-right).total_seconds()/3600)
            if distance<=lag:expected+=(1-distance/(lag+1))*np.outer(values[i],values[j])
    np.testing.assert_allclose(e.hac(times,values,lag),expected,rtol=0,atol=1e-14)
    compressed=e.hac(pd.date_range(times[0],periods=4,freq='h'),values,lag)
    assert not np.allclose(compressed,expected)


def test_closed_form_weighted_fit_and_inference():
    d=frame(500);d['y']+=np.sin(np.arange(len(d))*.7)*.2
    f=e.fit(d,'y',['post','r','post_r'],fe=(),weight='triangular')
    x=np.column_stack([np.ones(len(d)),d[['post','r','post_r']]])
    expected=np.linalg.solve(x.T@(d.triangular.to_numpy()[:,None]*x),x.T@(d.triangular*d.y))
    np.testing.assert_allclose(f.beta,expected,rtol=1e-11,atol=1e-11)
    residual=d.y.to_numpy()-x@expected; bread=np.linalg.inv(x.T@(d.triangular.to_numpy()[:,None]*x))
    influence=(x*(d.triangular.to_numpy()*residual)[:,None])@bread*np.sqrt(len(d)/(len(d)-4))
    manual=np.zeros((4,4));lag=2
    for h in range(-lag,lag+1):
        a=influence[max(0,h):min(len(d),len(d)+h)]
        b=influence[max(0,-h):min(len(d),len(d)-h)]
        manual+=(1-abs(h)/(lag+1))*(a.T@b)
    np.testing.assert_allclose(f.covariance(lag),manual,rtol=1e-8,atol=1e-11)
    out=f.contrast(f.covariance(lag),{'post':1})
    assert out['ci_high']==pytest.approx(out['coefficient']+1.96*out['se'])
    assert 0<=out['p_value']<=1


def test_fixed_effects_references_and_exact_signal():
    d=frame();f=e.fit(d,'y',['post','r','post_r'],weight='triangular')
    assert f.names[:4]==['const','post','r','post_r']
    assert f.audit['references']=={'hour':'0','weekday':'0'}
    assert 'hour_0' not in f.names and 'weekday_0' not in f.names
    np.testing.assert_allclose(f.beta[:4],[5,3,.1,-.03],atol=1e-10)
    assert f.audit['rank']==len(f.names)


def test_midpoint_kernel_cutoff_timezone():
    d=design.calendar(pd.DataFrame({'timestamp_utc':pd.to_datetime(['2025-12-31T22:00Z','2025-12-31T23:00Z'])}))
    q=design.local(d,90)
    np.testing.assert_allclose(q.r,[-.5/24,.5/24])
    assert q.post.tolist()==[0.,1.] and q.hour.tolist()==[23,0]
    np.testing.assert_allclose(q.triangular,1-abs(q.r)/90)


def test_complete_case_preserves_elapsed_gaps():
    d=frame();d.loc[d.index[60],'y']=np.nan
    f=e.fit(d,'y',['post','r','post_r'])
    assert len(f.y)==999
    assert f.sample.timestamp_utc.diff().max()==pd.Timedelta(hours=2)
    assert f.covariance(168).shape==(33,33)


def test_joint_covariance_equals_direct_difference():
    d=frame();d['a']=d.y+.1*np.sin(np.arange(len(d)));d['b']=d.y*.3+.1*np.cos(np.arange(len(d)));d['gap']=d.a-d.b
    a=e.fit(d,'a',['post','r','post_r']);b=e.fit(d,'b',['post','r','post_r']);g=e.fit(d,'gap',['post','r','post_r'])
    joint=e.joint_covariance([a,b],24);v=np.zeros(len(a.names)*2);v[a.names.index('post')]=1;v[len(a.names)+b.names.index('post')]=-1
    indirect=e.contrast(np.r_[a.beta,b.beta],joint,v);direct=g.contrast(g.covariance(24),{'post':1})
    np.testing.assert_allclose([indirect['coefficient'],indirect['se']],[direct['coefficient'],direct['se']],atol=1e-10)


@pytest.mark.parametrize('kind',['duplicates','rank','small_sample','small_side'])
def test_fit_negative_designs(kind):
    d=frame()
    if kind=='duplicates':d=pd.concat([d,d.iloc[[0]]])
    elif kind=='rank':d['r']=1.
    elif kind=='small_sample':d=d.iloc[:90]
    elif kind=='small_side':d=d.iloc[:100]
    with pytest.raises((ValueError,e.NotEstimable)):e.fit(d,'y',['post','r','post_r'])


@pytest.mark.parametrize('times',[
 ['2026-01-01T01:00Z','2026-01-01T00:00Z'],
 ['2026-01-01T00:00Z','2026-01-01T00:00Z'],
 ['2026-01-01T00:00Z','2026-01-01T00:30Z']])
def test_hac_rejects_invalid_time(times):
    with pytest.raises(ValueError):e.hac(pd.to_datetime(times),np.ones((2,1)),2)


def test_weather_nonlinear_features_before_donor_weighting():
    a=pd.DataFrame({'temp_c':[1.,2.],'wind_ms':[2.,3.],'ssrd_processed':[3.,4.],'tp_processed':[4.,5.]})
    b=a*3;centers={c:0 for c in a}
    fa=features(a,centers,True);fb=features(b,centers,True)
    syn=synthetic_features({'a':fa,'b':fb},{'a':.25,'b':.75})
    assert syn.temp_c_squared.iloc[0]==7 and syn.temp_c.iloc[0]**2==6.25
    assert 'tp_processed_squared' not in syn
    fb.loc[1,'temp_c']=np.nan
    assert synthetic_features({'a':fa,'b':fb},{'a':.25,'b':.75}).iloc[1].isna().all()


def test_weather_exact_duplicate_rules():
    d=pd.DataFrame({'a':[1.,2.],'b':[1.,2.],'constant':[3.,3.],'almost':[1.,2.00000001]})
    keep,drops=deduplicate(d,list(d))
    assert keep==['a','almost'] and len(drops)==2


def test_full_schedule_from_loop_definitions():
    versions=[stem+suffix for stem in ['GY025','GY050','GY075','G'] for suffix in ['_train2024','_final','_final_loo_AT_IT','_final_loo_GR_IT']]
    # Source-defined version identities select both placebo and leave-one-out branches.
    ids=models.scheduled_model_ids(versions)
    assert len(ids)==1892 and len(set(ids))==1892
    assert 'V3_FE_ME' in ids and 'V3_GAP_ME' in ids
    assert not any('CH_synthetic' in s or 'FR_actual' in s for s in ids)


@pytest.mark.authorized
def test_authorized_full_schedule_matches_frozen_inventory_only():
    if os.environ.get('CBAM_WEATHER_AUTHORIZED')!='1':pytest.skip('authorized inventory requires private baseline')
    from cbam_power.inputs import InputCatalog
    catalog=InputCatalog()
    with catalog.open_input('v2_weights',purpose='process') as handle:weights=pd.read_csv(handle,float_precision='round_trip')
    versions=weights.loc[weights.target.eq('ME_IT'),'version'].unique()
    with catalog.open_input('v3_results',purpose='verify') as handle:expected=json.load(handle)
    scheduled=models.scheduled_model_ids(versions)
    assert len(scheduled)==1892
    assert set(scheduled)=={m['model_id'] for m in expected['models']}
    assert len(expected['blocked_models'])==35


def test_engine_record_retains_full_parameters_covariance_and_sample(tmp_path):
    from cbam_power._engine_context import EngineContext,FIELDS
    from cbam_power._empirical import Results
    base=frame();base['scheduled_export_proxy']=base.y
    panel=pd.concat([base.assign(corridor_id=c) for c in models.ACTUALS],ignore_index=True)
    weather=pd.DataFrame({'timestamp_utc':base.timestamp_utc,'geography':'ME',**{f:np.ones(len(base)) for f in FIELDS}})
    synthetic=base.assign(weight_version='ARTIFICIAL_TEST_ONLY')
    ctx=EngineContext(tmp_path,panel,weather,synthetic,pd.DataFrame(),{},
        {k:'0'*64 for k in ['v2_panel','panel','weather','synthetic','weights']},
        pd.DatetimeIndex(base.timestamp_utc),pd.DataFrame())
    res=Results(ctx)
    f=res.record(base,'y',['post','r','post_r'],series='ARTIFICIAL_TEST_ONLY',model='R1',bw=90,weighted=True)
    res.export()
    index=pd.read_csv(tmp_path/'tables/covariance_index.csv')
    coefficients=pd.read_csv(tmp_path/'tables/all_coefficients.csv')
    assert len(index)==2 and len(coefficients)==2*len(f.names)
    assert not coefficients.duplicated(['model_id','hac_lag','term']).any()
    with np.load(tmp_path/'tables/model_covariances.npz',allow_pickle=False) as values:
        for r in index.itertuples():
            assert json.loads(r.columns)==f.names
            assert values[r.key].shape==(len(f.names),len(f.names))
    assert len(ctx.samples)==1 and len(ctx.model_samples)==1


def test_engine_constants_match_frozen_scientific_contract():
    import tomllib
    from cbam_power.inputs import REPO
    from cbam_power._engine_context import CFG,ACTUALS,DESTINATIONS,FIELDS
    cfg=tomllib.loads((REPO/'config/main.toml').read_text())
    assert CFG['window_utc_end_exclusive']==cfg['time']['window_end_exclusive']
    assert CFG['cutoff']==cfg['time']['policy_cutoff_local']
    assert CFG['bandwidth_days']==cfg['rdit']['bandwidth_days']
    assert CFG['donut_days']==cfg['rdit']['donut_days']
    assert CFG['hac_elapsed_hours']==cfg['inference']['hac_elapsed_hours']
    assert ACTUALS==cfg['panel']['actuals']
    assert {k+'_IT':v for k,v in DESTINATIONS.items()}==cfg['panel']['zones']
    assert design.F2_TERMS==cfg['same_season']['F2_terms']
    assert design.F1_CONTRASTS==cfg['same_season']['F1_contrasts']
    assert design.F2_CONTRASTS==cfg['same_season']['F2_contrasts']


# Authorized inventory requires an explicit opt-in and is excluded by -m public.
for _name,_function in list(globals().items()):
    if _name.startswith('test_') and callable(_function) and not any(mark.name=='authorized' for mark in getattr(_function,'pytestmark',[])):
        globals()[_name]=pytest.mark.public(_function)


@pytest.mark.public
def test_artificial_weather_engine_assembly_covariance_and_sample_exports(tmp_path):
    """Actual migrated helpers, artificial values; deliberately not full V3 replication."""
    from cbam_power._engine_context import EngineContext,FIELDS,ACTUALS
    from cbam_power._empirical import Results
    from cbam_power import _weather_models,_weather_synthetic,_assembly
    rng=np.random.default_rng(411)
    base=frame();base['scheduled_export_proxy']=base.y
    base['commercial_net_MW']=base.y
    base['raw_price_spread']=rng.normal(size=len(base))
    base['spread_c']=base.raw_price_spread
    panel=pd.concat([base.assign(corridor_id=c) for c in ACTUALS],ignore_index=True)
    weather=pd.concat([pd.DataFrame({'timestamp_utc':base.timestamp_utc,'geography':geo,
        **{field:rng.normal(size=len(base))+(10 if field=='temp_c' else 0) for field in FIELDS}})
        for geo in ['AT','CH','SI','GR','ME','IT-NORD','IT-SUD','IT-CSUD']],ignore_index=True)
    syn=base.assign(weight_version='GY050_final')
    syn['scheduled_export_proxy']=syn.y*.8+.1*np.cos(np.arange(len(syn)))
    weights=pd.DataFrame({'target':['ME_IT','ME_IT'],'version':['GY050_final','GY050_final'],
        'donor':['GR_IT','AT_IT'],'weight':[.3,.7],'trained_max_local_year':[2025,2025]})
    context=EngineContext(tmp_path,panel,weather,syn,weights,{},
        {key:'0'*64 for key in ['v2_panel','panel','weather','synthetic','weights']},
        pd.date_range('2024-01-01',periods=24,freq='h',tz='UTC'),weights)
    result=Results(context)
    result.record(base,'scheduled_export_proxy',['post','r','post_r'],series='ME_actual',model='R1',bw=90,weighted=True)
    result.export()
    _weather_models.run(context)
    _weather_synthetic.run(context)
    id_map={}
    records,_=_assembly.assemble_no_weather(context,id_map)
    records+=_assembly.assemble_weather(context,id_map,'weather_abc_results.csv','weather_coefficients.csv','weather_model_design.csv')
    records+=_assembly.assemble_weather(context,id_map,'weather_synthetic_abc_results.csv','weather_synthetic_coefficients.csv','weather_synthetic_design.csv',True)
    basic=e.fit(base,'scheduled_export_proxy',['post','r','post_r'],fe=(),weight='triangular')
    context.capture_sample('V3_BASIC_ME',basic.sample.timestamp_utc,basic)
    for lag in [168,24]:
        records.append(_assembly.from_fit('V3_BASIC_ME'+('' if lag==168 else '_HAC24'),basic,'ME_actual','BASIC_RDiT',(),lag))
    _assembly.export_covariance(context,id_map,records,{'V3_BASIC_ME':basic})
    context.export_samples(records,id_map)
    index=pd.read_csv(tmp_path/'covariance_index.csv')
    membership=pd.read_csv(tmp_path/'model_sample_index.csv')
    timestamps=pd.read_parquet(tmp_path/'sample_timestamps.parquet')
    assert len(records)==88 and len({m['model_id'] for m in records})==88
    assert len(membership)==88 and not membership.model_id.duplicated().any()
    proof=pd.read_csv(tmp_path/'model_design_signatures.csv')
    assert len(proof)==88 and set(proof.model_id)==set(membership.model_id)
    assert proof.weights_sha256.str.fullmatch('[a-f0-9]{64}').all()
    assert not timestamps.duplicated(['sample_hash_utc_ns','timestamp_utc']).any()
    assert {'V3_FE_ME','V3_WEATHER_ME','V3_GAP_WEATHER_ME','V3_SYNTHETIC_WEATHER_ME','V3_BASIC_ME'}.issubset(membership.model_id)
    assert index.covariance_scope.eq('joint').sum()==28
    assert index.model_id.notna().sum()==88
    for r in index.itertuples():
        path=tmp_path/r.file_path
        assert path.is_file() and path.is_relative_to(tmp_path)
        with np.load(path,allow_pickle=False) as archive:
            assert archive[r.covariance_key].shape==(len(json.loads(r.columns)),)*2
    actual_status=pd.read_csv(tmp_path/'diagnostics/weather_model_status.csv')
    assert len(actual_status[actual_status.status.eq('BLOCKED_WEATHER_PANEL')])==10
    syn_status=pd.read_csv(tmp_path/'diagnostics/weather_synthetic_model_status.csv')
    assert syn_status.status.eq('NOT_RUN').sum()==3


@pytest.mark.public
def test_actual_fit_proof_binds_weights_design_and_timestamps(tmp_path):
    import hashlib
    from cbam_power._engine_context import EngineContext,FIELDS
    base=frame();base['scheduled_export_proxy']=base.y
    panel=pd.concat([base.assign(corridor_id=c) for c in models.ACTUALS],ignore_index=True)
    weather=pd.DataFrame({'timestamp_utc':base.timestamp_utc,'geography':'ME',**{f:np.ones(len(base)) for f in FIELDS}})
    ctx=EngineContext(tmp_path,panel,weather,base.assign(weight_version='ARTIFICIAL'),pd.DataFrame(),{},
        {k:'0'*64 for k in ['v2_panel','panel','weather','synthetic','weights']},pd.DatetimeIndex(base.timestamp_utc),pd.DataFrame())
    fitted=e.fit(base,'y',['post','r','post_r'],weight='triangular')
    ctx.capture_sample('DEMO_PROOF',fitted.sample.timestamp_utc,fitted)
    expected=hashlib.sha256(np.asarray(base.triangular,dtype='<f8').tobytes()).hexdigest()
    assert ctx.designs['DEMO_PROOF']['weights_sha256']==expected
    fitted.weights=fitted.weights.copy()
    fitted.weights[0]*=.5
    with pytest.raises(ValueError,match='MODEL_DESIGN_CHANGED'):
        ctx.capture_sample('DEMO_PROOF',fitted.sample.timestamp_utc,fitted)
    with pytest.raises(ValueError,match='FIT_SAMPLE_PROOF_MISMATCH'):
        ctx.capture_sample('DEMO_WRONG_TIME',fitted.sample.timestamp_utc+pd.Timedelta(hours=1),fitted)
    ctx.model_samples['DEMO_NO_PROOF']=ctx.model_samples['DEMO_PROOF']
    with pytest.raises(ValueError,match='MISSING_FIT_DESIGN_PROOF'):
        ctx.export_samples([{'model_id':'DEMO_NO_PROOF','sample_hash':ctx.model_samples['DEMO_PROOF']}],{})
