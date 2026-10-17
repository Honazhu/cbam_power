"""Independent artificial validation cases; no expected research values here."""
from copy import deepcopy
from contextlib import contextmanager
from io import BytesIO
import hashlib
import json
import gzip
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cbam_power.validation import (ValidationError, assert_close, exact_columns, keyed_compare,
                                   timestamp_proof, verify_covariance, verify_models, verify_panel)
from cbam_power.validation import verify_v2f
from cbam_power.validation import primary_consistency
from cbam_power.validation import Evidence,comparison_context,elapsed_polynomial,compare_nested,_supplemental_csv
from cbam_power.validation import verify_design_signatures,verify_polynomial_parent_p1

pytestmark=pytest.mark.public
NUMERIC=dict(atol=1e-6,rtol=1e-8)


def artificial_model():
    row=dict(model_id='DEMO_VALIDATION',series='ME_actual',outcome='scheduled_export_proxy',
             model_family='R1',HAC_lag=168,N=200,pre_N=100,post_N=100,kernel='triangular',
             coefficient_name='post',status='ARTIFICIAL',weather=False,hour_FE=True,
             weekday_FE=True,month_FE=False,week_FE=False,cutoff='2026-01-01T00:00:00+01:00',
             bandwidth=90,donut=0,support='full',weight_version='actual',spec='main',
             estimate=4.,SE=1.,CI_low=2.04,CI_high=5.96,p_value=.001)
    row['coefficients']=[dict(term='post',coefficient=4.,SE=1.,CI_low=2.04,CI_high=5.96,p_value=.001)]
    return dict(models=[row],blocked_models=[])


@pytest.mark.parametrize('field,value',[('bandwidth',30),('donut',3),('cutoff','2025-01-01'),
    ('weight_version','wrong'),('support','trimmed'),('month_FE',True),('week_FE',True),
    ('spec','other'),('weather_variant','NEW'),('arm','C')])
def test_changed_scientific_definition_cannot_pass_equal_coefficients(field,value):
    expected=artificial_model();actual=deepcopy(expected);actual['models'][0][field]=value
    with pytest.raises(ValidationError,match='MODEL_DEFINITION_CHANGED'):
        verify_models(actual,expected,NUMERIC)


def test_model_and_status_identity():
    expected=artificial_model()
    assert verify_models(expected,deepcopy(expected),NUMERIC)['model_records']==1
    actual=deepcopy(expected);actual['blocked_models']=[dict(status='NOT_RUN',reason='ARTIFICIAL')]
    with pytest.raises(ValidationError,match='MODEL_STATUS_CHANGED'): verify_models(actual,expected,NUMERIC)
    actual=deepcopy(expected);actual['models']=[]
    with pytest.raises(ValidationError,match='MODEL_SET_CHANGED'): verify_models(actual,expected,NUMERIC)


def test_missing_mask_and_infinity_rejected():
    with pytest.raises(ValidationError,match='NULL_MASK'): assert_close([np.nan],[0.],NUMERIC)
    with pytest.raises(ValidationError,match='NONFINITE'): assert_close([np.inf],[np.inf],NUMERIC)


def test_primary_result_identity_uses_stricter_internal_contract():
    model=artificial_model()['models'][0]
    assert primary_consistency([model],dict(atol=1e-10,rtol=0))==1
    model['estimate']+=1e-7
    with pytest.raises(ValidationError,match='NUMERICAL'):
        primary_consistency([model],dict(atol=1e-10,rtol=0))


def artificial_panel():
    return pd.DataFrame(dict(timestamp_utc=pd.date_range('2025-01-01',periods=2,freq='h',tz='UTC'),
        corridor_id='ME_IT',origin_price=[20.,21.],destination_bidding_zone='IT-CSUD',
        origin_weather_temp_c=[0.,np.nan],destination_weather_temp_c=[10.,11.]))


def test_weather_uses_its_own_strict_tolerance():
    expected=artificial_panel();actual=expected.copy()
    tolerances=dict(core_panel=dict(atol=1e-8,rtol=1e-11),weather=dict(strict_abs_less_than=1e-9))
    actual.loc[0,'origin_weather_temp_c']=5e-10
    assert verify_panel(actual,expected,tolerances)['origin_weather_temp_c']==5e-10
    actual.loc[0,'origin_weather_temp_c']=1e-9
    with pytest.raises(ValidationError,match='NUMERICAL'): verify_panel(actual,expected,tolerances)


def test_panel_geographic_metadata_is_exact():
    expected=artificial_panel();actual=expected.copy();actual.loc[0,'destination_bidding_zone']='IT-NORD'
    with pytest.raises(ValidationError,match='SCIENTIFIC_DEFINITION_CHANGED'):
        verify_panel(actual,expected,dict(core_panel=NUMERIC,weather=dict(strict_abs_less_than=1e-9)))


def test_us_ns_encoding_requires_actual_timestamp_proof():
    values=pd.date_range('2025-01-01',periods=3,freq='h',tz='UTC').as_unit('ns').asi8
    legacy=hashlib.sha256((values//1000).astype('<i8').tobytes()).hexdigest()
    assert timestamp_proof(values,values.copy(),legacy)=='us'
    changed=values.copy();changed[1]+=1_000_000_000
    with pytest.raises(ValidationError,match='TIMESTAMPS_CHANGED'): timestamp_proof(changed,values,legacy)
    with pytest.raises(ValidationError,match='MEMBERSHIP_HASH'): timestamp_proof(values,values,'0'*64)


def test_comparison_key_duplicate_and_missing_rejected():
    expected=pd.DataFrame(dict(id=['one','two'],value=[1.,2.]))
    with pytest.raises(ValidationError,match='DUPLICATE'):
        keyed_compare(pd.concat([expected,expected.iloc[[0]]]),expected,['id'],['value'],NUMERIC)
    with pytest.raises(ValidationError,match='SET_CHANGED'):
        exact_columns(expected.iloc[:1],expected,['id'],['value'])


def covariance_fixture(tmp_path):
    old=tmp_path/'old.npz';new=tmp_path/'new.npz'
    np.savez(old,artificial_joint=np.eye(2));np.savez(new,artificial_joint=np.eye(2))
    row=dict(model_id='DEMO_JOINT',covariance_key='artificial_joint',file_path='outputs/models/engine/tables/old.npz',
             columns=json.dumps(['const','post']),HAC_lag=168,covariance_scope='joint')
    expected=pd.DataFrame([row]);actual=expected.copy();actual['file_path']='new.npz'
    catalog=SimpleNamespace(baselines={'files':[dict(id='old',legacy_source='v3_formal_cbam/'+row['file_path'])]},
                            resolve=lambda aid,purpose:old)
    return catalog,actual,expected


@pytest.mark.parametrize('field,value',[('model_id','WRONG_MODEL'),('HAC_lag',24),('covariance_scope','full_parameter_covariance')])
def test_covariance_key_does_not_excuse_wrong_model_link(tmp_path,field,value):
    cat,actual,expected=covariance_fixture(tmp_path)
    assert verify_covariance(cat,tmp_path,actual,expected,NUMERIC)['references']==1
    actual.loc[0,field]=value
    with pytest.raises(ValidationError,match='COVARIANCE_MODEL_LINK_CHANGED'):
        verify_covariance(cat,tmp_path,actual,expected,NUMERIC)


def test_covariance_missing_reference_and_array_detected(tmp_path):
    cat,actual,expected=covariance_fixture(tmp_path)
    cat.baselines['files']=[]
    with pytest.raises(ValidationError,match='BENCHMARK_REFERENCE'):
        verify_covariance(cat,tmp_path,actual,expected,NUMERIC)
    cat,actual,expected=covariance_fixture(tmp_path)
    np.savez(tmp_path/'new.npz',wrong=np.eye(2))
    with pytest.raises(ValidationError,match='KEY_MISSING'):
        verify_covariance(cat,tmp_path,actual,expected,NUMERIC)


def test_complete_v2f_archives_and_timestamp_translation(tmp_path):
    """Small independent fixtures exercise every full-benchmark table/archive path."""
    table_dir=tmp_path/'tables';table_dir.mkdir()
    values=pd.date_range('2025-01-01',periods=3,freq='h',tz='UTC').as_unit('ns').asi8
    ns=hashlib.sha256(values.astype('<i8').tobytes()).hexdigest()
    us=hashlib.sha256((values//1000).astype('<i8').tobytes()).hexdigest()
    quantities=dict(coefficient=2.,se=.5,ci_low=1.02,ci_high=2.98,p_value=.1)
    expected={};paths={}
    for stem in ['all_coefficients','weather_abc_results','weather_coefficients',
                 'weather_synthetic_abc_results','weather_synthetic_coefficients']:
        selector='estimand' if stem.endswith('abc_results') else 'term'
        row=dict(model_id='SOURCE_DEMO',hac_lag=168,n=3,sample_hash=us,sample_hash_utc_ns=ns,
                 **{selector:'post'},**quantities)
        expected['v2f_'+stem]=pd.DataFrame([row]).to_csv(index=False).encode()
        row['sample_hash']=ns
        pd.DataFrame([row]).to_csv(table_dir/(stem+'.csv'),index=False)
    for stem in ['model_covariances','weather_covariances','weather_synthetic_covariances']:
        paths['v2f_'+stem]=tmp_path/('old_'+stem+'.npz')
        np.savez(paths['v2f_'+stem],joint_known=np.array([[.25]]))
        np.savez(table_dir/(stem+'.npz'),joint_known=np.array([[.25]]))
    row=dict(kind='DEMO',series_a='A',series_b='B',series='ME',weight_version='DEMO_FIXED',hac_lag=168,
             covariance=.25,n=3,**quantities)
    joint=pd.DataFrame([row]);joint.to_csv(table_dir/'joint_covariance_contrasts.csv',index=False)
    expected['v2f_joint_covariance_contrasts']=joint.to_csv(index=False).encode()
    for stem in ['weather_adjustments','weather_synthetic_adjustments']:
        row=dict(series='ME',model='R1',support='full',variant='DEMO',hac_lag=168,
                 estimand='post',contrast='C-B',joint_cross_covariance_retained=True,**quantities)
        frame=pd.DataFrame([row]);frame.to_csv(table_dir/(stem+'.csv'),index=False)
        expected['v2f_'+stem]=frame.to_csv(index=False).encode()

    class ArtificialBenchmarks:
        @contextmanager
        def open_input(self,aid,purpose):
            assert purpose=='verify'
            yield BytesIO(expected[aid])

        def resolve(self,aid,purpose):
            assert purpose=='verify'
            return paths[aid]

    models=[dict(model_id='DEMO',source_model_id='SOURCE_DEMO')]
    report=verify_v2f(ArtificialBenchmarks(),tmp_path,models,{'DEMO':values},NUMERIC)
    assert len(report)==11
    assert report['model_covariances']['matrices']==1
    np.savez(table_dir/'model_covariances.npz',joint_known=np.array([[.5]]))
    with pytest.raises(ValidationError,match='NUMERICAL_EQUIVALENCE_FAILED'):
        verify_v2f(ArtificialBenchmarks(),tmp_path,models,{'DEMO':values},NUMERIC)


def test_outer_comparison_records_missing_and_unexpected_terms(tmp_path):
    actual=pd.DataFrame(dict(model_id=['DEMO','DEMO'],term=['post','unexpected'],coefficient=[2.,3.]))
    expected=pd.DataFrame(dict(model_id=['DEMO','DEMO'],term=['post','missing'],coefficient=[2.,4.]))
    with Evidence(tmp_path) as evidence,comparison_context(object='coefficients',source_module='artificial',failure_category='ESTIMATOR_DRIFT'):
        with pytest.raises(ValidationError,match='SET_CHANGED'):
            keyed_compare(actual,expected,['model_id','term'],['coefficient'],NUMERIC)
    rows=pd.read_csv(tmp_path/'comparison_cells.csv.gz')
    assert (rows.quantity=='outer_presence').sum()==2
    cells=rows[rows.quantity=='coefficient']
    assert len(cells)==3
    assert set(cells.term)=={'post','missing','unexpected'}
    assert (cells.status=='FAIL').sum()==2
    assert evidence.failed>=4


def test_every_covariance_cell_including_off_diagonal_is_recorded(tmp_path):
    actual=np.array([[1.,.5],[.5,4.]])
    expected=np.array([[1.,.4],[.4,4.]])
    with Evidence(tmp_path),comparison_context(object='covariance',model_id='DEMO',covariance_key='joint',
            columns=['a','b'],quantity='covariance',source_module='estimators.hac',failure_category='COVARIANCE_DRIFT'):
        with pytest.raises(ValidationError):assert_close(actual,expected,NUMERIC)
    rows=pd.read_csv(tmp_path/'comparison_cells.csv.gz')
    assert len(rows)==4 and (rows.status=='FAIL').sum()==2
    assert set(zip(rows.row_term,rows.column_term))=={('a','a'),('a','b'),('b','a'),('b','b')}
    assert rows.loc[rows.status=='FAIL','failure_category'].eq('COVARIANCE_DRIFT').all()
    assert np.isclose(rows.absolute_difference.max(),.1)


def test_evidence_relative_zero_and_frozen_combined_tolerance(tmp_path):
    with Evidence(tmp_path),comparison_context(quantity='beta'):
        assert_close([5e-7,1000000.005],[0.,1000000.],NUMERIC)
    rows=pd.read_csv(tmp_path/'comparison_cells.csv.gz')
    assert np.isinf(rows.relative_difference.iloc[0])
    assert rows.allowed_tolerance.iloc[0]==1e-6
    assert rows.allowed_tolerance.iloc[1]==.010001
    assert rows.status.eq('PASS').all()


def test_independent_scaled_polynomial_transform():
    model=dict(bandwidth_days=2,columns=['const','post','z1','post_z1','z2','post_z2'],
               beta=[10.,4.,6.,8.,12.,16.],covariance=np.eye(6).tolist())
    names,beta,cov=elapsed_polynomial(model)
    assert names==['const','post','r','post_r','r2','post_r2']
    np.testing.assert_array_equal(beta,[10.,4.,3.,4.,3.,4.])
    np.testing.assert_array_equal(np.diag(cov),[1.,1.,.25,.25,.0625,.0625])


def test_shape_turning_point_and_status_drift_rejected():
    expected=dict(flags=['WEAK'],turning_points=[dict(r_days=2.,weak_support=True)])
    actual=deepcopy(expected);actual['turning_points'][0]['r_days']=3.
    with pytest.raises(ValidationError):compare_nested(actual,expected,NUMERIC)
    actual=deepcopy(expected);actual['flags']=[]
    with pytest.raises(ValidationError,match='LENGTH_CHANGED'):compare_nested(actual,expected,NUMERIC)


def test_supplemental_registered_sidecars_require_exact_original_identity(tmp_path):
    path=tmp_path/'reference.csv';path.write_text('model_id,value\nDEMO,4\n')
    original=path.read_bytes()
    cat=SimpleNamespace(assets={'sidecar':dict(classification='P',size_bytes=len(original),sha256=hashlib.sha256(original).hexdigest())})
    assert _supplemental_csv(cat,'sidecar',{'sidecar':path}).value.iloc[0]==4
    path.write_text('model_id,value\nDEMO,9\n')
    with pytest.raises(ValidationError,match='HASH_CHANGED'):_supplemental_csv(cat,'sidecar',{'sidecar':path})
    with pytest.raises(ValidationError,match='REFERENCE_MISSING'):_supplemental_csv(cat,'sidecar',{})


def test_panel_sorted_order_is_a_contract():
    expected=artificial_panel();actual=expected.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValidationError,match='KEY_ORDER_CHANGED'):
        verify_panel(actual,expected,dict(core_panel=NUMERIC,weather=dict(strict_abs_less_than=1e-9)))


def test_actual_fit_kernel_signature_uses_elapsed_midpoints(tmp_path):
    times=pd.date_range('2025-12-31T22:00Z',periods=2,freq='h').as_unit('ns').asi8
    digest=hashlib.sha256(times.astype('<i8').tobytes()).hexdigest()
    names=['const','post','r','post_r']
    model=dict(model_id='DEMO',sample_hash=digest,kernel='triangular',bandwidth=90,
               cutoff='2026-01-01T00:00:00+01:00',covariance_reference={'columns':json.dumps(names)})
    # Both observations have half-hour midpoint distance: 1 - (0.5/24)/90.
    weights=np.asarray([1-(1800/86400)/90]*2,dtype='<f8')
    row=dict(model_id='DEMO',N=2,rank=4,columns=json.dumps(names),FE_references='{}',
             weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),sample_hash_utc_ns=digest,
             design_sha256='1'*64,outcome_sha256='2'*64,encoding='little_endian_float64_C_order')
    path=tmp_path/'model_design_signatures.csv';pd.DataFrame([row]).to_csv(path,index=False)
    assert verify_design_signatures(tmp_path,[model],{'DEMO':times})['models']==1
    row['weights_sha256']='0'*64;pd.DataFrame([row]).to_csv(path,index=False)
    with pytest.raises(ValidationError,match='SIGNATURE_CHANGED'):
        verify_design_signatures(tmp_path,[model],{'DEMO':times})
    path.unlink()
    with pytest.raises(ValidationError,match='BLOCKED_'):
        verify_design_signatures(tmp_path,[model],{'DEMO':times})


def test_p1_checks_independent_frozen_v3_parent_not_new_parent(tmp_path):
    times=pd.date_range('2025-01-01',periods=3,freq='h',tz='UTC').as_unit('ns').asi8
    digest=hashlib.sha256(times.astype('<i8').tobytes()).hexdigest()
    names=['const','post','r','post_r'];cov=tmp_path/'parent.npz';np.savez(cov,known=np.eye(4))
    parent=dict(model_id='V3_BASIC_ME',N=3,pre_N=2,post_N=1,estimate=4.,SE=1.,sample_hash=digest,
                coefficients=[dict(term=k,coefficient=v) for k,v in zip(names,[10.,4.,1.,2.])],
                covariance_reference=dict(file_path='old.npz',covariance_key='known',columns=json.dumps(names)))
    model=dict(model_id='DEMO_P1',polynomial_order=1,corridor='ME',bandwidth_days=90,cutoff_year=2026,
               family='BASIC',N=3,N_pre=2,N_post=1,tau=4.,se=1.,columns=['const','post','z1','post_z1'],
               beta=[10.,4.,90.,180.],covariance=np.diag([1.,1.,8100.,8100.]).tolist())
    class FrozenParent:
        baselines={'files':[dict(legacy_source='v3_formal_cbam/old.npz',id='parent_cov')],
                   'validation':dict(polynomial_p1_parent_tau_se=dict(atol=1e-8,rtol=1e-10),
                        polynomial_p1_all_coefficients=dict(strict_abs_less_than=1e-7),full_frozen=NUMERIC)}
        @contextmanager
        def open_input(self,aid,purpose):
            assert aid=='v3_results' and purpose=='verify'
            yield BytesIO(json.dumps(dict(models=[parent])).encode())
        def resolve(self,aid,purpose):return cov
    assert len(verify_polynomial_parent_p1([model],FrozenParent(),{'DEMO_P1':times}))==1
    model['tau']=4.1
    with pytest.raises(ValidationError,match='NUMERICAL'):
        verify_polynomial_parent_p1([model],FrozenParent(),{'DEMO_P1':times})


@pytest.mark.parametrize('field,value',[
    ('file_path','missing.npz'),('covariance_key','unknown'),('columns','["post", "const"]'),
    ('model_id','WRONG_MODEL'),('HAC_lag',24),('covariance_scope','joint')])
def test_json_covariance_reference_must_bind_index_and_frozen_role(tmp_path,field,value):
    from cbam_power.validation import verify_model_covariance_links
    (tmp_path/'tables').mkdir()
    np.savez(tmp_path/'tables'/'known.npz',known=np.eye(2))
    oldref=dict(model_id='DEMO',HAC_lag=168,covariance_key='known',
        columns=json.dumps(['const','post']),covariance_scope='full_parameter_covariance',
        file_path='outputs/models/engine/tables/known.npz')
    newref=dict(oldref,file_path='tables/known.npz')
    old=[dict(model_id='DEMO',HAC_lag=168,covariance_reference=oldref)]
    new=[dict(model_id='DEMO',HAC_lag=168,covariance_reference=deepcopy(newref))]
    args=(pd.DataFrame([newref]),pd.DataFrame([oldref]))
    assert verify_model_covariance_links(tmp_path,new,old,*args)['model_references']==1
    new[0]['covariance_reference'][field]=value
    with pytest.raises(ValidationError,match='MODEL_COVARIANCE'):
        verify_model_covariance_links(tmp_path,new,old,*args)
    new[0]['covariance_reference']=dict(newref,file_path='wrong_role.npz')
    np.savez(tmp_path/'wrong_role.npz',known=np.eye(2))
    wrong=pd.DataFrame([new[0]['covariance_reference']])
    with pytest.raises(ValidationError,match='ARCHIVE_ROLE_CHANGED'):
        verify_model_covariance_links(tmp_path,new,old,wrong,args[1])
