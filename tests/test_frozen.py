"""PR5 independent-run evidence and complete saved-result regression checks.

Public tests exercise failure detection using artificial files. Real tests require
CBAM_PR5_AUTHORIZED=1 and CBAM_PR5_RUN_ROOT containing independently computed
run_a/run_b. Opting in with missing evidence fails; it never becomes a skip.
No estimator is called, and no expected value or tolerance is updated.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbam_power import pipeline, synthetic, validation, weather
from cbam_power.inputs import InputCatalog, REPO, relative_path

REQUIRED_STAGES = ('validate-inputs', 'build-weather', 'build-panel', 'build-synthetic',
                   'estimate-main', 'verify-main', 'estimate-polynomial', 'verify-polynomial')
CORE_HASHES = {
    'v3_panel': 'a0eea50d8ac966fa9833eadc91d33e2f8f3f07c09bb1c025a5d79ed57bddc305',
    'v3_results': '997264dbc1748ce42e06fdd3b3693e7ee977899890c6752f30c62ed3db076c3c',
    'polynomial_results': '3857fbffc63e817ae9aea88e23cf9e6698ba9076ed9741ec18f3bc854b7437a9',
}
MINIMUM_OUTPUTS = {
    'build-weather': {'weather.parquet', 'v23_view.parquet', 'diagnostics.json'},
    'build-panel': {'panel.parquet'},
    'build-synthetic': {'all_versions.parquet', 'main.parquet', 'weights.csv',
                        'energy_mix.csv', 'objective_features.csv', 'summary.json'},
    'estimate-main': {'results.json', 'coefficients.csv', 'model_catalog.csv',
                      'covariance_index.csv', 'model_sample_index.csv',
                      'sample_timestamps.parquet', 'model_design_signatures.csv'},
    'estimate-polynomial': {'results.json', 'model_catalog.csv', 'not_estimated.csv',
                            'model_sample_index.csv', 'sample_timestamps.parquet'},
    'verify-main': {'verification.json'}, 'verify-polynomial': {'verification.json'},
}


class EvidenceError(AssertionError):
    pass


def require(condition, code):
    if not condition:
        raise EvidenceError(code)


def file_sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_json(path):
    require(Path(path).is_file(), 'REQUIRED_EVIDENCE_MISSING: '+Path(path).name)
    return json.loads(Path(path).read_text())


def local_artifact(root, relative):
    path = Path(root).joinpath(*relative_path(relative).parts)
    require(not any(p.is_symlink() for p in [path, *path.parents]), 'OUTPUT_SYMLINK')
    require(path.is_file(), 'REQUIRED_ARTIFACT_MISSING: '+relative)
    require(path.stat().st_nlink == 1, 'SHARED_OUTPUT_HARDLINK')
    return path


def check_receipt(root, receipt, *, expected_identity=None, required_stages=REQUIRED_STAGES,
                  minimum_outputs=MINIMUM_OUTPUTS):
    require(receipt.get('mode') == 'authorized', 'AUTHORIZED_RECEIPT_REQUIRED')
    require(receipt.get('rights') == 'PRIVATE_ONLY_PR6_PENDING', 'RELEASE_BOUNDARY_CHANGED')
    require(not receipt.get('errors'), 'RUN_CONTAINS_ERRORS')
    require('identity' in receipt, 'RUN_IDENTITY_MISSING')
    if expected_identity is not None:
        require(receipt['identity'] == expected_identity, 'RUN_IDENTITY_CHANGED')
    records = receipt.get('stages', {})
    require(set(required_stages) <= set(records), 'REQUIRED_STAGE_MISSING')
    completed = receipt.get('completed_stages', [])
    require([s for s in completed if s in required_stages] == list(required_stages),
            'STAGE_COMPLETION_ORDER_CHANGED')
    signature = pipeline.digest(receipt['identity'])
    previous_end = None
    for stage in required_stages:
        record = records[stage]
        require(record.get('status') == 'PASS', 'REQUIRED_STAGE_NOT_PASSED: '+stage)
        require(record.get('identity') == signature, 'STAGE_IDENTITY_CHANGED: '+stage)
        start, end = map(datetime.fromisoformat, (record['started_utc'], record['ended_utc']))
        require(start <= end and (previous_end is None or previous_end <= start), 'STAGE_TIME_ORDER')
        previous_end = end
        outputs = record.get('outputs', {})
        needed = {stage+'/'+name for name in minimum_outputs.get(stage, ())}
        require(needed <= set(outputs), 'INCOMPLETE_STAGE_ARTIFACTS: '+stage)
        for relative, expected_sha in outputs.items():
            require(relative.startswith(stage+'/'), 'ARTIFACT_STAGE_ESCAPE')
            require(file_sha(local_artifact(root, relative)) == expected_sha, 'ARTIFACT_HASH_CHANGED: '+relative)
    return signature


def check_audit(root, receipt, assets, *, required_stages=REQUIRED_STAGES):
    isolation = load_json(Path(root)/'isolation.json')
    require(isolation.get('status') == 'PASS', 'ISOLATION_NOT_PASSED')
    require(all(isolation.get(k) == 'DENY' for k in ('network','legacy','cross_run')),
            'ISOLATION_BOUNDARY_MISSING')
    require(isolation.get('protected_inputs') == 'READ_ONLY', 'INPUT_PROTECTION_MISSING')
    invocations = isolation.get('invocations', [])
    require(bool(invocations), 'NO_ACTUAL_EXECUTION_INVOCATIONS')
    require(all(i.get('status') == 'PASS' and not i.get('violations') for i in invocations),
            'ISOLATION_INVOCATION_FAILED')
    require(len({i['harness_sha256'] for i in invocations}) == 1, 'AUDIT_HARNESS_CHANGED_WITHIN_RUN')
    actual_completed = [s for i in invocations for s in i['completed_stages']]
    require(Counter(actual_completed) == Counter(required_stages), 'INDEPENDENT_STAGE_EXECUTION_MISSING_OR_REUSED')
    events_path = local_artifact(root, 'access_audit.jsonl')
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    require(not any(e['event'] == 'denied' for e in events), 'DENIED_ACCESS_DURING_REPLICATION')
    invocation_ids = {i['invocation'] for i in invocations}
    require({e['invocation'] for e in events} == invocation_ids, 'UNRECORDED_AUDIT_INVOCATION')
    starts = [e for e in events if e['event'] == 'stage_start']
    ends = [e for e in events if e['event'] == 'stage_end']
    require([e['stage'] for e in starts] == list(required_stages), 'ACTUAL_STAGE_STARTS_MISSING')
    require([e['stage'] for e in ends] == list(required_stages), 'ACTUAL_STAGE_COMPLETIONS_MISSING')
    for start, end in zip(starts, ends):
        require(start['invocation'] == end['invocation'], 'STAGE_PROCESS_IDENTITY_CHANGED')
        require(start['code_identity'] == receipt['identity']['code'], 'AUDITED_CODE_IDENTITY_CHANGED')
        require(end['status'] == 'PASS' and end['elapsed_ns'] > 0, 'STAGE_CALLBACK_NOT_EXECUTED')
        require(end['outputs'] == receipt['stages'][end['stage']]['outputs'], 'AUDIT_RECEIPT_OUTPUTS_CHANGED')
    opened = [e for e in events if e['event'] == 'input_open']
    verified = [e for e in events if e['event'] == 'input_verified']
    require(bool(verified) and len(opened) == len(verified), 'LOGICAL_INPUT_VERIFICATION_INCOMPLETE')
    identity = lambda e: (e['invocation'], e['stage'], e['asset_id'], e['purpose'], e['sha256'])
    require(Counter(map(identity, opened)) == Counter(map(identity, verified)), 'UNPAIRED_LOGICAL_INPUT_READ')
    for event in [*verified, *[e for e in events if e['event']=='supplemental_reference_verified']]:
        check_logical_read(event, assets)
    for stage in ('build-weather','build-panel','build-synthetic'):
        require(any(e['stage']==stage and e['classification']=='F' for e in verified),
                'RECONSTRUCTION_WITHOUT_RAW_OR_FROZEN_UPSTREAM_INPUTS: '+stage)
    netcdf = {aid for event in events if event['event']=='native_decode'
              for aid in event['asset_ids'] if assets[aid]['format']=='netcdf'}
    expected_netcdf = {aid for aid,a in assets.items() if a['format']=='netcdf' and a['required']}
    require(netcdf == expected_netcdf and len(netcdf)==50, 'NATIVE_RECONSTRUCTION_INCOMPLETE')
    for stage in ('estimate-main','estimate-polynomial'):
        fits=[e for e in events if e['event']=='fit_start' and e['stage']==stage]
        require(bool(fits), 'REAL_ESTIMATION_NOT_AUDITED: '+stage)
        require(all(not e['validation_phase'] for e in fits), 'FIT_DURING_FROZEN_VALIDATION')
        done=[e for e in events if e['event']=='fit_end' and e['stage']==stage]
        require(len(done)==len(fits), 'FIT_COMPLETION_AUDIT_MISSING')
    order_starts=[e['order'] for e in events if e['event']=='validation_start']
    order_ends=[e['order'] for e in events if e['event']=='validation_end']
    require(order_starts == order_ends == [1,2,3], 'POLYNOMIAL_SEQUENTIAL_VALIDATION_MISSING')
    return isolation, events


def check_logical_read(event, assets):
    aid = event['asset_id']
    require(aid in assets, 'UNREGISTERED_RUNTIME_INPUT')
    require(event['sha256'] == assets[aid]['sha256'], 'AUDITED_SOURCE_IDENTITY_CHANGED')
    require(event['classification'] == assets[aid]['classification'], 'AUDITED_SOURCE_ROLE_CHANGED')
    if event.get('root_identity_only'):
        require(aid == 'v3_panel' and event['purpose'] == 'verify', 'INVALID_ROOT_IDENTITY_EXCEPTION')
    elif event['classification'] in {'K','P'}:
        require(event['stage']=='validate-inputs' or event['stage'].startswith('verify-')
                or (event['stage']=='estimate-polynomial' and event.get('validation_phase')),
                'EXPECTED_VALUE_LEAKAGE_IN_COMPUTATION')
        require(event['purpose']=='verify', 'BENCHMARK_NON_VERIFICATION_PURPOSE')


def authorized_paths(environ):
    if environ.get('CBAM_PR5_AUTHORIZED') != '1':return None
    root=environ.get('CBAM_PR5_RUN_ROOT');input_root=environ.get('CBAM_INPUT_ROOT')
    require(bool(root),'BLOCKED_CBAM_PR5_RUN_ROOT_REQUIRED')
    require(bool(input_root),'BLOCKED_CBAM_INPUT_ROOT_REQUIRED')
    root=Path(root).absolute()
    require(root.is_dir(),'BLOCKED_PR5_RUN_ROOT_MISSING')
    paths=[root/'run_a',root/'run_b']
    require(all(p.is_dir() for p in paths),'BLOCKED_INDEPENDENT_RUN_MISSING')
    return root,paths,input_root


def compare_numeric_frame(left, right, contract, *, keys=None):
    require(list(left.columns)==list(right.columns), 'CROSS_RUN_SCHEMA_CHANGED')
    require(left.shape==right.shape, 'CROSS_RUN_SHAPE_CHANGED')
    if keys:
        require(not left.duplicated(keys).any() and not right.duplicated(keys).any(), 'CROSS_RUN_DUPLICATE_KEY')
        left=left.sort_values(keys).reset_index(drop=True)
        right=right.sort_values(keys).reset_index(drop=True)
    require(left.isna().equals(right.isna()), 'CROSS_RUN_NULL_MASK_CHANGED')
    differences={}
    for col in left:
        if pd.api.types.is_numeric_dtype(left[col]) and pd.api.types.is_numeric_dtype(right[col]):
            differences[col]=validation.assert_close(left[col],right[col],contract)
        else:
            a=left[col].astype(object).where(left[col].notna(),None).tolist()
            b=right[col].astype(object).where(right[col].notna(),None).tolist()
            require(a==b,'CROSS_RUN_DEFINITION_CHANGED: '+col)
    return differences


def compare_nested(left, right, contract, location='root'):
    if isinstance(left, dict):
        require(isinstance(right,dict) and left.keys()==right.keys(), 'NESTED_KEYS_CHANGED: '+location)
        return max((compare_nested(left[k],right[k],contract,location+'/'+k) for k in left),default=0.)
    if isinstance(left,list):
        require(isinstance(right,list) and len(left)==len(right), 'NESTED_LENGTH_CHANGED: '+location)
        return max((compare_nested(a,b,contract,location+'/'+str(i)) for i,(a,b) in enumerate(zip(left,right))),default=0.)
    if isinstance(left,(int,float)) and not isinstance(left,bool):
        require(isinstance(right,(int,float)) and not isinstance(right,bool), 'NESTED_TYPE_CHANGED: '+location)
        return validation.assert_close(left,right,contract)
    require(left==right,'NESTED_DEFINITION_CHANGED: '+location)
    return 0.


def cross_covariance(left_dir, right_dir, contract):
    ia=pd.read_csv(Path(left_dir)/'covariance_index.csv')
    ib=pd.read_csv(Path(right_dir)/'covariance_index.csv')
    def index(frame):
        result={}
        for row in frame.to_dict('records'):
            key=(row['covariance_key'],tuple(json.loads(row['columns'])))
            require(key not in result,'DUPLICATE_COVARIANCE_REFERENCE')
            result[key]=row
        return result
    a,b=index(ia),index(ib)
    require(a.keys()==b.keys(),'CROSS_RUN_COVARIANCE_SET_CHANGED')
    archives={};cells=0;maximum=0.
    try:
        for key,x in a.items():
            y=b[key]
            for field in ('model_id','HAC_lag','covariance_scope'):
                require((pd.isna(x[field]) and pd.isna(y[field])) or x[field]==y[field],
                        'CROSS_RUN_COVARIANCE_LINK_CHANGED')
            matrices=[]
            for root,row in ((left_dir,x),(right_dir,y)):
                path=local_artifact(root,row['file_path'])
                if path not in archives: archives[path]=np.load(path,allow_pickle=False)
                require(key[0] in archives[path], 'COVARIANCE_KEY_MISSING')
                matrix=archives[path][key[0]]
                require(matrix.shape==(len(key[1]),len(key[1])), 'COVARIANCE_PARAMETER_ORDER_CHANGED')
                matrices.append(matrix)
            maximum=max(maximum,validation.assert_close(*matrices,contract))
            cells+=matrices[0].size
    finally:
        for archive in archives.values(): archive.close()
    return dict(references=len(a),cells=cells,max_absolute_difference=maximum)


def sample_arrays(directory, models):
    index=pd.read_csv(Path(directory)/'model_sample_index.csv')
    membership=pd.read_parquet(Path(directory)/'sample_timestamps.parquet')
    require(not index.model_id.duplicated().any(),'DUPLICATE_MODEL_SAMPLE_LINK')
    require(set(index.model_id)=={m['model_id'] for m in models},'MODEL_SAMPLE_SET_CHANGED')
    require(not membership.duplicated(['sample_hash_utc_ns','timestamp_utc']).any(),'DUPLICATE_SAMPLE_TIMESTAMP')
    arrays={}
    for key,group in membership.groupby('sample_hash_utc_ns',sort=False):
        arr=pd.DatetimeIndex(group.timestamp_utc).as_unit('ns').asi8.astype('<i8')
        require(len(arr)>0 and np.all(np.diff(arr)>0),'NONCANONICAL_SAMPLE_ORDER')
        require(hashlib.sha256(arr.tobytes()).hexdigest()==key,'SAMPLE_HASH_DOES_NOT_REPRESENT_TIMESTAMPS')
        arrays[key]=arr
    require(set(index.sample_hash_utc_ns)==set(arrays),'UNREFERENCED_OR_MISSING_SAMPLE_CONTENT')
    links=dict(zip(index.model_id,index.sample_hash_utc_ns))
    for model in models:
        require(model['sample_hash']==links[model['model_id']],'RESULT_SAMPLE_LINK_CHANGED')
    return {mid:arrays[digest] for mid,digest in links.items()}


@pytest.mark.public
def test_receipt_rejects_missing_authorized_stage(tmp_path):
    receipt={'mode':'authorized','rights':'PRIVATE_ONLY_PR6_PENDING','errors':[],
             'identity':{},'stages':{},'completed_stages':[]}
    with pytest.raises(EvidenceError,match='REQUIRED_STAGE_MISSING'):
        check_receipt(tmp_path,receipt)


def artificial_receipt(tmp_path):
    folder=tmp_path/'sample';folder.mkdir();(folder/'value.txt').write_text('artificial')
    identity={'mode':'authorized','code':{'artificial':'identity'}}
    receipt={'mode':'authorized','rights':'PRIVATE_ONLY_PR6_PENDING','errors':[],
        'identity':identity,'completed_stages':['sample'],'stages':{
        'sample':{'status':'PASS','identity':pipeline.digest(identity),
                  'started_utc':'2026-01-01T00:00:00+00:00','ended_utc':'2026-01-01T00:00:01+00:00',
                  'outputs':{'sample/value.txt':file_sha(folder/'value.txt')}}}}
    return receipt


@pytest.mark.public
def test_receipt_checks_actual_artifact_bytes(tmp_path):
    receipt=artificial_receipt(tmp_path)
    check_receipt(tmp_path,receipt,required_stages=('sample',),minimum_outputs={})
    (tmp_path/'sample/value.txt').write_text('modified')
    with pytest.raises(EvidenceError,match='ARTIFACT_HASH_CHANGED'):
        check_receipt(tmp_path,receipt,required_stages=('sample',),minimum_outputs={})


@pytest.mark.public
def test_receipt_rejects_stale_code_identity(tmp_path):
    receipt=artificial_receipt(tmp_path)
    with pytest.raises(EvidenceError,match='RUN_IDENTITY_CHANGED'):
        check_receipt(tmp_path,receipt,expected_identity={'code':'new'},required_stages=('sample',),minimum_outputs={})


@pytest.mark.public
def test_receipt_rejects_unfinished_stage(tmp_path):
    receipt=artificial_receipt(tmp_path);receipt['stages']['sample']['status']='RUNNING'
    with pytest.raises(EvidenceError,match='REQUIRED_STAGE_NOT_PASSED'):
        check_receipt(tmp_path,receipt,required_stages=('sample',),minimum_outputs={})


@pytest.mark.public
def test_artifact_rejects_symlink_and_hardlink(tmp_path):
    original=tmp_path/'original';original.write_text('artificial')
    (tmp_path/'symbolic').symlink_to(original)
    with pytest.raises(EvidenceError,match='OUTPUT_SYMLINK'):local_artifact(tmp_path,'symbolic')
    os.link(original,tmp_path/'hard')
    with pytest.raises(EvidenceError,match='SHARED_OUTPUT_HARDLINK'):local_artifact(tmp_path,'hard')


@pytest.mark.public
def test_cross_run_values_and_masks_are_independent_checks():
    a=pd.DataFrame({'key':['a','b'],'value':[1.,np.nan]})
    compare_numeric_frame(a,a.copy(),{'atol':1e-6,'rtol':1e-8},keys=['key'])
    b=a.copy();b.loc[1,'value']=0
    with pytest.raises(EvidenceError,match='NULL_MASK'):compare_numeric_frame(a,b,{'atol':1e-6,'rtol':1e-8})
    b=a.copy();b.loc[0,'value']=2
    with pytest.raises(validation.ValidationError,match='NUMERICAL'):compare_numeric_frame(a,b,{'atol':1e-6,'rtol':1e-8})


@pytest.mark.public
def test_nested_shape_flags_cannot_be_hidden_by_numeric_tolerance():
    a={'turning_points':[.1],'flags':['unstable_shape']}
    b={'turning_points':[.1],'flags':[]}
    with pytest.raises(EvidenceError,match='NESTED_LENGTH'):compare_nested(a,b,{'atol':1e-6,'rtol':1e-8})


@pytest.mark.public
def test_missing_isolation_evidence_is_not_independence(tmp_path):
    with pytest.raises(EvidenceError,match='REQUIRED_EVIDENCE_MISSING'):check_audit(tmp_path,{}, {})


@pytest.mark.public
def test_duplicate_covariance_reference_rejected(tmp_path):
    dirs=[tmp_path/'a',tmp_path/'b']
    rows=[dict(model_id='artificial',covariance_key='joint',file_path='joint.npz',
               columns='["x", "z"]',HAC_lag=2,covariance_scope='joint')]*2
    for d in dirs:d.mkdir();pd.DataFrame(rows).to_csv(d/'covariance_index.csv',index=False)
    with pytest.raises(EvidenceError,match='DUPLICATE_COVARIANCE_REFERENCE'):
        cross_covariance(*dirs,{'atol':1e-6,'rtol':1e-8})


@pytest.mark.public
def test_full_covariance_detects_off_diagonal_drift(tmp_path):
    dirs=[tmp_path/'a',tmp_path/'b']
    row=dict(model_id='artificial',covariance_key='joint',file_path='joint.npz',
             columns='["x", "z"]',HAC_lag=2,covariance_scope='joint')
    for d in dirs:
        d.mkdir();pd.DataFrame([row]).to_csv(d/'covariance_index.csv',index=False)
        np.savez(d/'joint.npz',joint=np.array([[1.,.2],[.2,2.]]))
    assert cross_covariance(*dirs,{'atol':1e-6,'rtol':1e-8})['cells']==4
    np.savez(dirs[1]/'joint.npz',joint=np.array([[1.,.3],[.3,2.]]))
    with pytest.raises(validation.ValidationError,match='NUMERICAL'):
        cross_covariance(*dirs,{'atol':1e-6,'rtol':1e-8})


@pytest.mark.public
def test_authorized_request_missing_evidence_fails_instead_of_skipping(tmp_path):
    with pytest.raises(EvidenceError,match='BLOCKED_CBAM_PR5_RUN_ROOT_REQUIRED'):
        authorized_paths({'CBAM_PR5_AUTHORIZED':'1'})
    with pytest.raises(EvidenceError,match='BLOCKED_INDEPENDENT_RUN_MISSING'):
        authorized_paths({'CBAM_PR5_AUTHORIZED':'1','CBAM_PR5_RUN_ROOT':str(tmp_path),'CBAM_INPUT_ROOT':str(tmp_path/'inputs')})


@pytest.mark.public
def test_expected_values_are_rejected_as_estimator_reads():
    assets={'expected':{'sha256':'a'*64,'classification':'K'}}
    event=dict(asset_id='expected',sha256='a'*64,classification='K',purpose='verify',
               stage='estimate-main',validation_phase=[],root_identity_only=False)
    with pytest.raises(EvidenceError,match='EXPECTED_VALUE_LEAKAGE'):
        check_logical_read(event,assets)
    check_logical_read({**event,'stage':'verify-main'},assets)
    check_logical_read({**event,'stage':'estimate-polynomial','validation_phase':[['polynomial_order',1]]},assets)


@pytest.mark.public
def test_unknown_or_modified_runtime_source_is_rejected():
    assets={'source':{'sha256':'a'*64,'classification':'F'}}
    event=dict(asset_id='source',sha256='b'*64,classification='F',purpose='process',stage='build-panel')
    with pytest.raises(EvidenceError,match='SOURCE_IDENTITY_CHANGED'):check_logical_read(event,assets)
    with pytest.raises(EvidenceError,match='UNREGISTERED_RUNTIME_INPUT'):
        check_logical_read({**event,'asset_id':'unregistered'},assets)


@pytest.mark.public
def test_only_frozen_panel_identity_hash_may_use_root_anchor_exception():
    assets={'coefficient':{'sha256':'a'*64,'classification':'K'}}
    event=dict(asset_id='coefficient',sha256='a'*64,classification='K',purpose='verify',
               stage='estimate-main',root_identity_only=True)
    with pytest.raises(EvidenceError,match='INVALID_ROOT_IDENTITY_EXCEPTION'):check_logical_read(event,assets)


@pytest.fixture(scope='module')
def authorized_runs():
    request=authorized_paths(os.environ)
    if request is None:
        pytest.skip('Authorized PR5 evidence is opt-in; this skip is never G5 acceptance.')
    root,paths,input_root=request
    catalog=InputCatalog(input_root=input_root,output_root=root)
    current_identity=pipeline.identity('authorized',REPO)
    receipts=[load_json(p/'run.json') for p in paths]
    for p,r in zip(paths,receipts):check_receipt(p,r,expected_identity=current_identity)
    require(receipts[0]['run_id']=='run_a' and receipts[1]['run_id']=='run_b','INDEPENDENT_RUN_NAMES')
    return dict(root=root,paths=paths,catalog=catalog,receipts=receipts,
                tolerance=catalog.baselines['validation'])


@pytest.mark.authorized
@pytest.mark.frozen
def test_both_runs_completed_with_identical_final_code_and_sources(authorized_runs):
    c=authorized_runs
    assert c['receipts'][0]['identity']==c['receipts'][1]['identity']
    for aid,expected in CORE_HASHES.items():
        assert c['catalog'].assets[aid]['sha256']==expected
        with c['catalog'].open_input(aid,purpose='verify') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==expected
    for stage in REQUIRED_STAGES:
        oa=c['receipts'][0]['stages'][stage]['outputs'];ob=c['receipts'][1]['stages'][stage]['outputs']
        require(set(oa)==set(ob),'CROSS_RUN_ARTIFACT_SET_CHANGED')
        for relative in oa:
            sa=(c['paths'][0]/relative).stat();sb=(c['paths'][1]/relative).stat()
            require((sa.st_dev,sa.st_ino)!=(sb.st_dev,sb.st_ino),'SHARED_CROSS_RUN_ARTIFACT_INODE')


@pytest.mark.authorized
@pytest.mark.frozen
def test_actual_execution_and_no_benchmark_or_legacy_leakage(authorized_runs):
    c=authorized_runs
    checked=[check_audit(p,r,c['catalog'].assets) for p,r in zip(c['paths'],c['receipts'])]
    a,b=checked
    assert {i['harness_sha256'] for i in a[0]['invocations']}=={i['harness_sha256'] for i in b[0]['invocations']}
    # Both computations consume the same source IDs; they never consume each
    # other's output paths. Logical IDs are recorded only after hash verification.
    def sources(events):return {(e['stage'],e['asset_id'],e['sha256'],e['purpose']) for e in events if e['event']=='input_verified'}
    assert sources(a[1])==sources(b[1])
    assert set(i['invocation'] for i in a[0]['invocations']).isdisjoint(i['invocation'] for i in b[0]['invocations'])


@pytest.mark.authorized
@pytest.mark.frozen
@pytest.mark.parametrize('name',['run_a','run_b'])
def test_reconstructed_panel_and_weather_against_independent_frozen_references(authorized_runs,name):
    c=authorized_runs;root=c['root']/name;catalog=c['catalog']
    panel=pd.read_parquet(root/'build-panel/panel.parquet')
    assert panel.shape==(118795,40)
    with catalog.open_input('v3_panel',purpose='verify') as f:expected=pd.read_parquet(f)
    validation.verify_panel(panel,expected,c['tolerance'])
    for aid,filename in [('engine_weather_panel_hourly','weather.parquet'),
                         ('weather_v2f_benchmark','weather.parquet'),
                         ('weather_v23_benchmark','v23_view.parquet')]:
        actual=pd.read_parquet(root/'build-weather'/filename)
        with catalog.open_input(aid,purpose='verify') as f:expected=pd.read_parquet(f)
        report,differences=weather.compare_panel(actual,expected,tolerance=c['tolerance']['weather']['strict_abs_less_than'])
        assert report['status']=='PASS',differences
    actual=pd.read_parquet(root/'build-weather/weather.parquet')
    assert actual.shape==(41486,6)
    assert actual[weather.FIELDS].isna().sum().to_dict()==dict.fromkeys(weather.FIELDS,16)


@pytest.mark.authorized
@pytest.mark.frozen
@pytest.mark.parametrize('name',['run_a','run_b'])
def test_all_synthetic_versions_objectives_diagnostics_and_weight_identity(authorized_runs,name):
    c=authorized_runs;root=c['root']/name
    checks=synthetic.validate_saved(root/'build-synthetic',c['catalog'])
    assert len(checks)==142 and all(r['status']=='PASS' for r in checks)
    weights=pd.read_csv(root/'build-synthetic/weights.csv',float_precision='round_trip')
    assert set(weights.version)==set(synthetic.ALL_VERSIONS) and len(weights)==32
    paths=pd.read_parquet(root/'build-synthetic/all_versions.parquet')
    assert set(paths.weight_version)==set(synthetic.ALL_VERSIONS)
    assert load_json(root/'build-synthetic/summary.json')['weights_reestimated'] is False


@pytest.mark.authorized
@pytest.mark.frozen
@pytest.mark.parametrize('name',['run_a','run_b'])
@pytest.mark.parametrize('spec',['main','polynomial'])
def test_fresh_full_baseline_verification_is_not_a_cached_pass(authorized_runs,name,spec):
    c=authorized_runs;root=c['root']/name
    # Executes all saved-result, sample, coefficient, covariance and mapped V2F
    # comparisons again. Expected inputs are read only after actual estimation.
    references = None
    if spec == 'polynomial':
        reference_root=Path(os.environ.get('CBAM_REFERENCE_ROOT',str(c['root']/'references')))
        references={f'polynomial_{kind}_p{order}':reference_root/f'polynomial_{kind}_p{order}.csv'
                    for kind in ('daily_bins','plot_curves') for order in (1,2,3)}
    report=validation.verify_baseline(c['catalog'],root/('estimate-'+spec),spec=spec,
                                      panel_path=root/'build-panel/panel.parquet',
                                      supplemental_references=references)
    assert report['status']=='PASS'
    assert report['model_records']==(1892 if spec=='main' else 90)
    assert report['sample_proofs']==(1892 if spec=='main' else 90)
    saved=load_json(root/('verify-'+spec)/'verification.json')
    assert saved['status']=='PASS'


@pytest.mark.authorized
@pytest.mark.frozen
def test_cross_run_panel_weather_synthetic_and_all_16_weight_versions(authorized_runs):
    c=authorized_runs;a,b=c['paths'];tol=c['tolerance']
    validation.verify_panel(pd.read_parquet(a/'build-panel/panel.parquet'),
                            pd.read_parquet(b/'build-panel/panel.parquet'),tol)
    for name in ['weather.parquet','v23_view.parquet']:
        x=pd.read_parquet(a/'build-weather'/name);y=pd.read_parquet(b/'build-weather'/name)
        checked,diff=weather.compare_panel(x,y,tolerance=tol['weather']['strict_abs_less_than'])
        assert checked['status']=='PASS',diff
    for name in ['main.parquet','all_versions.parquet']:
        compare_numeric_frame(pd.read_parquet(a/'build-synthetic'/name),pd.read_parquet(b/'build-synthetic'/name),
                              tol['core_panel'],keys=['weight_version','timestamp_utc'])
    compare_numeric_frame(pd.read_csv(a/'build-synthetic/weights.csv',float_precision='round_trip'),
                          pd.read_csv(b/'build-synthetic/weights.csv',float_precision='round_trip'),
                          tol['frozen_weights_csv_json'],keys=['version','donor'])


@pytest.mark.authorized
@pytest.mark.frozen
def test_cross_run_all_v3_models_coefficients_covariance_samples_and_statuses(authorized_runs):
    c=authorized_runs;a,b=[p/'estimate-main' for p in c['paths']];tol=c['tolerance']['full_frozen']
    x,y=[load_json(p/'results.json') for p in (a,b)]
    assert len(x['models'])==len(y['models'])==1892
    validation.verify_models(x,y,tol)
    coefficients=[pd.read_csv(p/'coefficients.csv') for p in (a,b)]
    assert all(len(d)==77574 for d in coefficients)
    compare_numeric_frame(*coefficients,tol,keys=['model_id','term'])
    catalogs=[pd.read_csv(p/'model_catalog.csv') for p in (a,b)]
    assert all(len(d)==1928 for d in catalogs)
    compare_numeric_frame(*catalogs,tol,keys=['model_id'])
    assert len(x['blocked_models'])==len(y['blocked_models'])==35
    cov=cross_covariance(a,b,tol)
    assert cov['references']==1926 and cov['cells']>77574
    sa=sample_arrays(a,x['models']);sb=sample_arrays(b,y['models'])
    assert sa.keys()==sb.keys()
    for mid in sa:np.testing.assert_array_equal(sa[mid],sb[mid],err_msg=mid)
    # Actual X/y/kernel/FE signatures are emitted by the fit observer, not read
    # from a baseline or inferred merely from a model identifier.
    signatures=[pd.read_csv(p/'model_design_signatures.csv') for p in (a,b)]
    assert len(signatures[0])>0
    pd.testing.assert_frame_equal(*signatures,check_dtype=False,check_exact=True)


@pytest.mark.authorized
@pytest.mark.frozen
def test_cross_run_polynomial_beta_covariance_shape_curves_bins_and_statuses(authorized_runs):
    c=authorized_runs;a,b=[p/'estimate-polynomial' for p in c['paths']];tol=c['tolerance']['full_frozen']
    x,y=[load_json(p/'results.json') for p in (a,b)]
    assert len(x['models'])==len(y['models'])==90
    assert len(x['not_estimated'])==len(y['not_estimated'])==15
    validation.verify_models(x,y,tol,polynomial=True)
    ma={m['model_id']:m for m in x['models']};mb={m['model_id']:m for m in y['models']}
    for mid,m in ma.items():compare_nested(m['shape_diagnostics'],mb[mid]['shape_diagnostics'],tol,mid)
    sa=sample_arrays(a,x['models']);sb=sample_arrays(b,y['models'])
    for mid in sa:np.testing.assert_array_equal(sa[mid],sb[mid],err_msg=mid)
    for order in (1,2,3):
        for kind in ('plot_curves','daily_bins'):
            name=f'{kind}_P{order}.csv'
            compare_numeric_frame(pd.read_csv(a/'tables'/name),pd.read_csv(b/'tables'/name),tol)
    for name in ('model_catalog.csv','not_estimated.csv'):
        compare_numeric_frame(pd.read_csv(a/name),pd.read_csv(b/name),tol)
