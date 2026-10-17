"""Artificial polynomial design and diagnostic checks; no full authorized fit."""
import numpy as np
import pandas as pd
import pytest
from cbam_power import polynomial as p
from cbam_power import estimators,design
from cbam_power._assembly import from_fit

pytestmark=pytest.mark.public

def data(h=10):
    cut=pd.Timestamp('2025-12-31T23:00Z')
    t=pd.date_range(cut-pd.Timedelta(days=h),periods=48*h,freq='h')
    r=(t+pd.Timedelta(minutes=30)-cut).total_seconds().to_numpy()/86400
    post=(t>=cut).astype(float)
    y=25+4*post+.2*r+.08*post*r+.004*r*r+.01*np.sin(np.arange(len(t))*.7)
    d=pd.DataFrame({'timestamp_utc':t,'corridor_id':'AT_IT','scheduled_export_proxy':y,
                    'commercial_net_MW':y,'delivery_interval_minutes':60,'destination_bidding_zone':'IT-NORD'})
    return p.prepare_panel(d)


@pytest.mark.parametrize('order',[1,2,3])
def test_complete_post_interactions(order):
    q=p.sample(data(), 'AT',10,2026);z,names=p.terms(q,order)
    assert names==['post']+[v for k in range(1,order+1) for v in [f'z{k}',f'post_z{k}']]
    for k in range(1,order+1):
        np.testing.assert_allclose(z[f'z{k}'],(z.r/10)**k)
        np.testing.assert_array_equal(z[f'post_z{k}'],z.post*z[f'z{k}'])


def test_three_orders_real_fit_shape_and_samples():
    existing={};hashes=[]
    for order in [1,2,3]:
        model,lines,bins=p.fit_model(data(),'AT',order,'BASIC',10,2026,existing)
        existing[model['model_id']]=model;hashes.append((model['sample_hash'],model['weight_hash']))
        assert model['number_of_regressors']==2+2*order
        assert len(lines)==482 and len(bins)>0
        assert abs(model['tau']-4)<.05
        assert model['raw_scaled_max_fitted_difference']<1e-5
        assert model['covariance_status']=='PASS'
        assert 'turning_points' in model['shape_diagnostics']
        assert model['status'] in ['ESTIMATED','ESTIMATED_FUNCTIONALLY_UNSTABLE']
    assert len(set(hashes))==1


def test_coefficient_and_covariance_change_of_basis():
    m={'columns':['const','post','z1','post_z1','z2','post_z2'],
       'bandwidth_days':10,'beta':[1.,2.,30.,40.,500.,600.],
       'covariance':np.arange(36,dtype=float).reshape(6,6)}
    names,beta,cov=p.transform_to_elapsed_days(m)
    assert names==['const','post','r','post_r','r2','post_r2']
    np.testing.assert_allclose(beta,[1,2,3,4,5,6])
    scale=np.array([1,1,.1,.1,.01,.01])
    for i in range(6):
        for j in range(6):assert cov[i,j]==pytest.approx(m['covariance'][i,j]*scale[i]*scale[j])


def test_p1_parent_linear_gate_full_coefficients():
    d=data(90);m,_,_=p.fit_model(d,'AT',1,'BASIC',90,2026,{})
    q=design.local(design.calendar(d),90)
    f=estimators.fit(q,'scheduled_export_proxy',['post','r','post_r'],fe=(),weight='triangular')
    parent=from_fit('V3_BASIC_AT',f,'AT_actual','BASIC_RDiT',())
    check=p.baseline_check(m,{'V3_BASIC_AT':parent})
    assert check['status']=='PASS' and check['sample_match']
    parent['sample_hash']='wrong'
    with pytest.raises(RuntimeError,match='SAMPLE_MISMATCH'):p.baseline_check(m,{'V3_BASIC_AT':parent})


def test_schedule_all_orders_and_preserved_2024_statuses():
    ids=[p.model_id(*config) for order in [1,2,3] for config in p.configurations(order)]
    assert len(ids)==90 and len(set(ids))==90
    blocked=p.not_estimated_models()
    assert len(blocked)==15 and {r['status'] for r in blocked}=={'INSUFFICIENT_PRE_SUPPORT'}
    assert all('2024' in r['model_id'] for r in blocked)
    assert all('HAC24' not in i for i in ids)


def test_polynomial_requires_verified_parent_before_creating_outputs(tmp_path):
    path=tmp_path/'poly'
    with pytest.raises(ValueError,match='MAIN_BASELINE_VERIFICATION_REQUIRED'):
        p.estimate_polynomial(data(),{},path,source_hashes={},parent_verification={'status':'NOT_YET_VERIFIED'})
    assert not path.exists()


@pytest.mark.parametrize('kind',['duplicate','gap','missing_outcome','incorrect_interval','incorrect_positive_part'])
def test_invalid_polynomial_sample(kind):
    d=data()
    if kind=='duplicate':d=pd.concat([d,d.iloc[[0]]])
    elif kind=='gap':d=d.drop(d.index[15])
    elif kind=='missing_outcome':d.loc[0,'scheduled_export_proxy']=np.nan
    elif kind=='incorrect_interval':d.loc[0,'delivery_interval_minutes']=15
    else:d.loc[0,'scheduled_export_proxy']+=5
    with pytest.raises(AssertionError):
        d=p.prepare_panel(d)
        p.sample(d,'AT',10,2026)


def test_curve_cutoff_matches_machine_tau():
    m,_,_=p.fit_model(data(),'AT',3,'BASIC',10,2026,{})
    left,_=p.curve(m,0,np.array([0.]));right,_=p.curve(m,1,np.array([0.]))
    assert float(right[0]-left[0])==pytest.approx(m['tau'],abs=1e-10)


def test_shape_flags_are_retained_not_filtered():
    m,_,_=p.fit_model(data(),'AT',2,'BASIC',10,2026,{})
    # An artificial quadratic has a real derivative root and it is reported.
    assert isinstance(m['shape_diagnostics']['flags'],list)
    assert set(m['shape_diagnostics']['flags']).issubset({
        'NEGATIVE_FITTED_HOURLY_OUTCOME','EXTREME_ENDPOINT_FITTED_VALUE',
        'CUBIC_TWO_TURNING_POINTS_ON_ONE_SIDE','TURNING_POINT_LOW_LOCAL_SUPPORT',
        'TURNING_POINT_IN_LOW_WEIGHT_ENDPOINT_REGION','CI_EXPANSION_VS_LINEAR',
        'CONDITION_NUMBER_EXPLOSION'})


def test_original_independent_polynomial_recovery_and_gapped_hac_oracle():
    checks=p.known_polynomials()
    assert len(checks)==7 and all(row['status']=='PASS' for row in checks)


def test_polynomial_constants_match_frozen_scientific_contract():
    import tomllib
    from cbam_power.inputs import REPO
    cfg=tomllib.loads((REPO/'config/polynomial.toml').read_text())
    assert p.THRESHOLDS==cfg['thresholds']
    assert p.CODES==cfg['design']['codes']
    assert p.ZONES==cfg['design']['zones']


@pytest.mark.parametrize('stop_order,mode',[(1,'FAIL'),(1,'missing_status'),(1,'raise'),(2,'FAIL')])
def test_independent_order_gate_stops_before_higher_order(monkeypatch,tmp_path,stop_order,mode):
    """Orchestration spy only: no fixture value is used as scientific expected data."""
    import json
    d=data(90);q=p.sample(d,'AT',90,2026)
    sample_hash=p.canonical_sample_hash(q.timestamp_utc)
    parent={'models':[{'model_id':'V3_BASIC_AT','sample_hash':sample_hash}]}
    monkeypatch.setattr(p,'CODES',['AT'])
    # Fifteen tiny spies satisfy the original P1 gate cardinality while keeping
    # this test focused on sequencing. Production configurations are unchanged.
    monkeypatch.setattr(p,'configurations',lambda order: [('AT',order,'BASIC',90,2026)]*15)
    fit_orders=[]
    def fake_fit(data,code,order,family,bandwidth,year,existing):
        fit_orders.append(order)
        model={'model_id':f'DEMO_GATE_P{order}_{len(fit_orders)}','sample_hash':sample_hash,'polynomial_order':order}
        return model,[{'model_id':model['model_id'],'fitted':0.}],pd.DataFrame({'model_id':[model['model_id']],'N':[len(q)]})
    monkeypatch.setattr(p,'fit_model',fake_fit)
    monkeypatch.setattr(p,'baseline_check',lambda model,parent: {'status':'PASS'} if model['polynomial_order']==1 else None)
    checked=[]
    def independent(order,obj,path):
        checked.append(order)
        assert str(order) not in obj['stages']
        assert (path/'model_sample_index.csv').is_file()
        assert (path/'sample_timestamps.parquet').is_file()
        assert (path/f'tables/plot_curves_P{order}.csv').is_file()
        assert len(pd.read_csv(path/'model_sample_index.csv'))==15*order
        if order==stop_order:
            if mode=='raise':raise RuntimeError('FROZEN_ORDER_MISMATCH')
            return {'status':'FAIL'} if mode=='FAIL' else {}
        return {'status':'PASS','order':order}
    output=tmp_path/'polynomial'
    with pytest.raises(RuntimeError):
        p.estimate_polynomial(d,parent,output,source_hashes={'panel':'artificial','main_results':'artificial'},
            parent_verification={'status':'PASS'},order_validator=independent)
    assert checked==list(range(1,stop_order+1))
    assert fit_orders==[order for order in range(1,stop_order+1) for _ in range(15)]
    saved=json.loads((output/'results.json').read_text())
    assert str(stop_order) not in saved['stages']
    assert not (output/f'tables/plot_curves_P{stop_order+1}.csv').exists()
