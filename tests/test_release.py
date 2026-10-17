"""Artificial release-policy and content-security tests; no private research reads."""
import copy
import gzip
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import zipfile
import tomllib
from collections import Counter

import pytest

from cbam_power import release


pytestmark = pytest.mark.public


def evidence(eid='demo-approval', *, kind='user_demo_authorization', group='demo', **changes):
    value = dict(id=eid, kind=kind, status='VERIFIED', source='Explicit artificial test authorization',
                 scope=['group:'+group], applicability='NOT_APPLICABLE',
                 permissions=['public_repository', 'research_publication'])
    value.update(changes)
    return value


def node(nid='demo', content=b'Artificial example\n', path='demo.txt', **changes):
    value = dict(id=nid, kind='artifact', origin='ARTIFICIAL', status='APPROVED_PUBLIC',
        dependencies=[], lineage_complete=True, lineage_kind='artificial_generator', review_group='demo',
        evidence=['demo-approval'], conditions=[], permissions=['public_repository', 'research_publication'],
        sha256=release.sha256(content), path=path, format='text', reason='Reviewed artificial fixture')
    value.update(changes)
    return value


def registry(nodes=None, records=None):
    return release.RightsRegistry(dict(nodes=nodes or [node()], evidence=records or [evidence()]))


def test_default_denial_statuses():
    for status in ['RESTRICTED', 'UNVERIFIED', 'PENDING_OWNER_APPROVAL', 'EXCLUDED']:
        with pytest.raises(release.ReleaseError, match='RIGHTS_NOT_APPROVED'):
            registry([node(status=status)]).authorize('demo')


def test_transitive_restriction_cannot_be_laundered_as_derived():
    nodes = [node('raw', status='RESTRICTED'), node('variable', kind='variable', dependencies=['raw'], lineage_kind='derived'),
             node('figure', dependencies=['variable'], lineage_kind='derived')]
    with pytest.raises(release.ReleaseError, match='RIGHTS_NOT_APPROVED'):
        registry(nodes).authorize('figure')


@pytest.mark.parametrize('change,error', [
    ({'dependencies':['absent']}, 'MISSING_RIGHTS_DEPENDENCY'),
    ({'status':'CONDITIONAL'}, 'INVALID_RIGHTS_STATUS'),
    ({'evidence':['absent']}, 'MISSING_EVIDENCE_REFERENCE'),
    ({'sha256':'bad'}, 'INVALID_RIGHTS_SHA256'),
    ({'path':'../private'}, 'INVALID_RELEASE_PATH'),
    ({'permissions':['open_source']}, 'INVALID_NODE_PERMISSIONS'),
])
def test_registry_schema_negative(change,error):
    with pytest.raises(release.ReleaseError, match=error): registry([node(**change)])


def test_duplicate_and_cycle_rejected():
    with pytest.raises(release.ReleaseError, match='DUPLICATE_NODE_ID'): registry([node(),node()])
    with pytest.raises(release.ReleaseError, match='RIGHTS_DEPENDENCY_CYCLE'):
        registry([node('a',dependencies=['b']),node('b',dependencies=['a'])])


def test_missing_lineage_and_unpinned_bytes():
    with pytest.raises(release.ReleaseError, match='INCOMPLETE_PUBLICATION_LINEAGE'):
        registry([node(lineage_complete=False)]).authorize('demo')
    with pytest.raises(release.ReleaseError, match='DERIVED_LINEAGE_MISSING'):
        registry([node(kind='variable',lineage_kind='derived')]).authorize('demo')
    with pytest.raises(release.ReleaseError, match='UNPINNED_RELEASE_BYTES'):
        registry([node(sha256='UNVERIFIED')]).authorize('demo','0'*64)


def test_verified_evidence_requires_scope_and_permission():
    for record in [evidence(status='UNVERIFIED'),evidence(scope=['other']),evidence(permissions=['research_publication'])]:
        with pytest.raises(release.ReleaseError,match='ARTIFICIAL_AUTHORIZATION_REQUIRED'):
            registry(records=[record]).authorize('demo')
    with pytest.raises(release.ReleaseError,match='UNBOUNDED_EVIDENCE_APPROVAL'):
        registry(records=[evidence(scope=['*'])])


def test_explicit_conditions_are_evidence_bound():
    conditional = node(status='APPROVED_WITH_CONDITIONS',conditions=['attribution'])
    records=[evidence(),evidence('attribution-proof',kind='condition',condition_id='attribution')]
    rights=registry([conditional],records)
    with pytest.raises(release.ReleaseError,match='UNSATISFIED_RELEASE_CONDITION'):
        rights.authorize('demo')
    with pytest.raises(release.ReleaseError,match='CONDITION_EVIDENCE_REQUIRED'):
        rights.authorize('demo',satisfied_conditions={'attribution':True})
    with pytest.raises(release.ReleaseError,match='INVALID_CONDITION_EVIDENCE'):
        rights.authorize('demo',satisfied_conditions={'attribution':'demo-approval'})
    assert rights.authorize('demo',satisfied_conditions={'attribution':'attribution-proof'})['status']=='APPROVED'


def test_owner_approval_and_context_are_separate():
    source=node('code',kind='source',origin='SOURCE',lineage_kind='source',evidence=['owner'],review_group='code')
    rights=registry([source],[evidence('owner',kind='owner_approval',group='code')])
    assert rights.authorize('code')['status']=='APPROVED'
    with pytest.raises(release.ReleaseError,match='CONTEXT_NOT_AUTHORIZED'):
        rights.authorize('code',context='commercial_reuse')
    wrong=copy.deepcopy(rights.data);wrong['evidence'][0]['kind']='license'
    with pytest.raises(release.ReleaseError,match='OWNER_APPROVAL_REQUIRED'):
        release.RightsRegistry(wrong).authorize('code')


def test_current_license_does_not_approve_frozen_vintage():
    asset=node('real',kind='asset',origin='REAL',lineage_kind='raw',review_group='real',evidence=['license'])
    record=evidence('license',kind='license',group='real',applicability='CURRENT_ONLY')
    with pytest.raises(release.ReleaseError,match='FROZEN_VINTAGE_LICENSE_UNVERIFIED'):
        registry([asset],[record]).authorize('real')
    record['applicability']='FROZEN_VINTAGE_CONFIRMED'
    assert registry([asset],[record]).authorize('real')['status']=='APPROVED'


def test_real_artifact_needs_publication_specific_approval():
    raw=node('real',kind='asset',origin='REAL',lineage_kind='raw',review_group='real',evidence=['license'])
    artifact=node('result',origin='REAL',lineage_kind='derived',dependencies=['real'],review_group='real',evidence=['license'])
    records=[evidence('license',kind='license',group='real',applicability='FROZEN_VINTAGE_CONFIRMED')]
    with pytest.raises(release.ReleaseError,match='ARTIFACT_PUBLICATION_APPROVAL_REQUIRED'):
        registry([raw,artifact],records).authorize('result')
    artifact['evidence'].append('publication')
    records.append(evidence('publication',kind='publication_approval',group='real'))
    assert registry([raw,artifact],records).authorize('result')['status']=='APPROVED'


def test_real_dependency_cannot_be_relabelled_demo():
    raw=node('raw',kind='asset',origin='REAL',lineage_kind='raw')
    fake=node('fake',dependencies=['raw'],lineage_kind='derived')
    with pytest.raises(release.ReleaseError,match='REAL_OR_UNREVIEWED_SOURCE_IN_DEMO'):
        registry([raw,fake]).authorize('fake')


def test_demo_payload_requires_registered_bytes_not_claimed_mode():
    content=json.dumps(dict(mode='demo',label=release.DEMO_LABEL,models=[{'model_id':'DEMO_FIXTURE'}])).encode()
    rights=registry([node(content=content,format='json')])
    assert release.validate_demo_result_content(content,rights,'demo')['mode']=='demo'
    with pytest.raises(release.ReleaseError,match='RELEASE_CONTENT_IDENTITY_CHANGED'):
        release.validate_demo_result_content(content+b' ',rights,'demo')
    data=json.loads(content);data['models'][0]['model_id']='V3_FE_ME';bad=json.dumps(data).encode()
    with pytest.raises(release.ReleaseError,match='REAL_MODEL_IN_DEMO'):
        release.validate_demo_result_content(bad,registry([node(content=bad,format='json')]),'demo')


@pytest.mark.parametrize('content,error',[
    (b'PAR1private panel', 'RESTRICTED_BINARY_MAGIC'),
    (b'\x89HDF\r\n\x1a\nprivate weather', 'RESTRICTED_BINARY_MAGIC'),
    (b'\x93NUMPYprivate array', 'RESTRICTED_BINARY_MAGIC'),
    (b'%PDF-private report', 'RESTRICTED_BINARY_MAGIC'),
    (b'\xff\x00payload', 'UNREVIEWED_BINARY_CONTENT'),
    (b'text\x00binary', 'BINARY_CONTENT_DISGUISED_AS_TEXT'),
    (b'file '+b'/' + b'Users/'+b'exampleperson/private/file', 'PERSONAL_ABSOLUTE_PATH'),
    (b'-----BEGIN '+b'PRIVATE KEY-----', 'CREDENTIAL_CONTENT'),
    (b'api_key = "'+b'abcdefghijklmno' + b'"', 'CREDENTIAL_CONTENT'),
    (b'{"models":[{"estimate":1.23}]}', 'STRUCTURED_CONTENT_MASQUERADE'),
    (b'timestamp,price\n2026-01-01,87.5\n', 'TABULAR_CONTENT_MASQUERADE'),
    (b'timestamp\tprice\n2026-01-01\t87.5\n', 'TABULAR_CONTENT_MASQUERADE'),
    (b'timestamp;price\n2026-01-01;87.5\n', 'TABULAR_CONTENT_MASQUERADE'),
])
def test_content_cannot_masquerade_as_safe_text(content,error):
    with pytest.raises(release.ReleaseError,match=error):
        release.scan_content(content,registry([node(content=content,format='markdown')]),'demo')


def test_notebook_output_detection_ignores_suffix():
    content=json.dumps(dict(nbformat=4,cells=[dict(cell_type='code',source=['1+1'],outputs=[{'text':'2'}],execution_count=1)])).encode()
    with pytest.raises(release.ReleaseError,match='NOTEBOOK_EXECUTION_OUTPUTS'):
        release.scan_content(content,registry([node(content=content,format='json')]),'demo')


def test_active_report_payload_rejected():
    content=b'<svg><script>alert(1)</script></svg>'
    with pytest.raises(release.ReleaseError,match='ACTIVE_OR_EMBEDDED_REPORT_CONTENT'):
        release.scan_content(content,registry([node(content=content,format='svg')]),'demo')


@pytest.mark.parametrize('parameter',['X-Amz-Signature','X-Amz-Credential','X-Goog-Signature',
                                     'access_token','token','sig'])
def test_signed_url_credentials_detected(parameter):
    content=('https://example.invalid/object?'+parameter+'='+'privateCredential1234567890').encode()
    with pytest.raises(release.ReleaseError,match='SIGNED_URL_CREDENTIAL_CONTENT'):
        release.scan_content(content,registry([node(content=content)]),'demo')


def test_public_hash_and_unsigned_provider_url_are_not_secret_false_positives():
    content=b'identity '+b'a'*64+b'\nhttps://example.invalid/data?year=2026&country=IT\n'
    assert release.scan_content(content,registry([node(content=content)]),'demo')['status']=='SECURITY_CLEAN'


def zip_bytes(name,content,*,symlink=False):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        info=zipfile.ZipInfo(name)
        if symlink:info.external_attr=(stat.S_IFLNK|0o777)<<16
        archive.writestr(info,content)
    return buffer.getvalue()


def archive_registry(content,*,fmt='zip',member='inside.txt',inside=b'artificial text\n'):
    return registry([node('inner',content=inside,path=member),
        node('outer',content=content,path='reviewed.'+fmt,format=fmt,
             members={member:'inner'},dependencies=['inner'],lineage_kind='derived')])


def test_registered_archive_members_scanned_recursively():
    inside=b'artificial text\n';content=zip_bytes('inside.txt',inside)
    assert release.scan_content(content,archive_registry(content),'outer')['status']=='SECURITY_CLEAN'
    bad=b'/' + b'home/'+b'privateperson/result.csv';content=zip_bytes('inside.txt',bad)
    with pytest.raises(release.ReleaseError,match='PERSONAL_ABSOLUTE_PATH'):
        release.scan_content(content,archive_registry(content,inside=bad),'outer')


def test_unknown_or_disguised_archive_denied():
    content=zip_bytes('inside.txt',b'artificial text\n')
    with pytest.raises(release.ReleaseError,match='ARCHIVE_FORMAT_MASQUERADE'):
        release.scan_content(content,registry([node(content=content)]),'demo')
    with pytest.raises(release.ReleaseError,match='UNREVIEWED_ARCHIVE_CONTENTS'):
        release.scan_content(content,registry([node(content=content,format='zip')]),'demo')


def test_archive_comments_are_scanned_too():
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as archive:
        archive.writestr('inside.txt',b'artificial text\n')
        archive.comment=b'password="'+b'not-a-public-password'+b'"'
    content=buffer.getvalue()
    with pytest.raises(release.ReleaseError,match='CREDENTIAL_CONTENT'):
        release.scan_content(content,archive_registry(content),'outer')


@pytest.mark.parametrize('member',['../escape.txt','/absolute.txt','folder\\escape.txt'])
def test_archive_path_traversal_denied(member):
    content=zip_bytes(member,b'artificial text\n')
    with pytest.raises(release.ReleaseError,match='INVALID_RELEASE_PATH'):
        release.scan_content(content,archive_registry(content),'outer')


def test_archive_symlink_hardlink_and_size_limits():
    content=zip_bytes('inside.txt',b'artificial text\n',symlink=True)
    with pytest.raises(release.ReleaseError,match='UNSAFE_ARCHIVE_MEMBER'):
        release.scan_content(content,archive_registry(content),'outer')
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as archive:
        info=tarfile.TarInfo('inside.txt');info.type=tarfile.LNKTYPE;info.linkname='outside';archive.addfile(info)
    content=buffer.getvalue()
    with pytest.raises(release.ReleaseError,match='UNSAFE_ARCHIVE_MEMBER'):
        release.scan_content(content,archive_registry(content,fmt='tar'),'outer')
    inside=b'a'*10000;content=gzip.compress(inside)
    with pytest.raises(release.ReleaseError,match='RELEASE_SCAN_SIZE_LIMIT'):
        release.scan_content(content,archive_registry(content,fmt='gzip',inside=inside),'outer',
            limits=release.ScanLimits(max_compression_ratio=2))


def test_nested_archive_requires_all_member_identities():
    leaf=b'artificial\n';inner=zip_bytes('leaf.txt',leaf);outer=gzip.compress(inner)
    rights=registry([node('leaf',content=leaf,path='leaf.txt'),
        node('inner',content=inner,path='inner.zip',format='zip',members={'leaf.txt':'leaf'},dependencies=['leaf'],lineage_kind='derived'),
        node('outer',content=outer,path='outer.gz',format='gzip',members={'inner.zip':'inner'},dependencies=['inner'],lineage_kind='derived')])
    assert release.scan_content(outer,rights,'outer')['members'][0]['format']=='zip'
    with pytest.raises(release.ReleaseError,match='RELEASE_SCAN_RECURSION_LIMIT'):
        release.scan_content(outer,rights,'outer',limits=release.ScanLimits(max_depth=1))


def test_security_audit_is_not_owner_approval(tmp_path):
    root=tmp_path/'source';root.mkdir();content=b'print("safe code")\n';(root/'example.py').write_bytes(content)
    rights=registry([node('code',content=content,path='example.py',format='python',kind='source',origin='SOURCE',
        lineage_kind='source',status='PENDING_OWNER_APPROVAL',evidence=[])])
    audit=release.audit_candidate(rights,root,whitelist={'example.py':'code'})
    assert audit['status']=='SECURITY_CLEAN' and audit['publication_approved'] is False
    with pytest.raises(release.ReleaseError,match='RIGHTS_NOT_APPROVED'):
        release.stage_public(rights,root,tmp_path/'public',whitelist={'example.py':'code'})
    assert not (tmp_path/'public').exists()


def test_stage_is_explicit_byte_pinned_and_atomic(tmp_path):
    root=tmp_path/'source';root.mkdir();content=b'artificial text\n';(root/'demo.txt').write_bytes(content)
    (root/'unregistered.csv').write_text('private,values\n1,2\n')
    rights=registry([node(content=content)])
    output=tmp_path/'public'
    report=release.stage_public(rights,root,output,whitelist={'demo.txt':'demo'})
    assert report['published'] is False and list(output.iterdir())==[output/'demo.txt']
    assert (output/'demo.txt').read_bytes()==content
    with pytest.raises(release.ReleaseError,match='RELEASE_DESTINATION_MUST_BE_NEW'):
        release.stage_public(rights,root,output,whitelist={'demo.txt':'demo'})
    (root/'demo.txt').write_bytes(content+b'changed')
    with pytest.raises(release.ReleaseError,match='RELEASE_CONTENT_IDENTITY_CHANGED'):
        release.stage_public(rights,root,tmp_path/'bad',whitelist={'demo.txt':'demo'})
    assert not (tmp_path/'bad').exists() and not list(tmp_path.glob('.cbam-release-*'))


@pytest.mark.parametrize('kind',['symlink','hardlink'])
def test_source_links_are_not_safe_copies(tmp_path,kind):
    source=tmp_path/'source';source.mkdir();payload=tmp_path/'payload';payload.write_bytes(b'fixture')
    if kind=='symlink':(source/'demo.txt').symlink_to(payload)
    else:os.link(payload,source/'demo.txt')
    rights=registry([node(content=b'fixture')])
    with pytest.raises(release.ReleaseError,match='RELEASE_SOURCE_'):
        release.stage_public(rights,source,tmp_path/'public',whitelist={'demo.txt':'demo'})
    assert not (tmp_path/'public').exists()


def test_fail_before_any_staging_when_later_file_is_secret(tmp_path):
    source=tmp_path/'source';source.mkdir()
    safe=b'Artificial text\n';secret=b'/' + b'Users/'+b'example/private';(source/'a.txt').write_bytes(safe);(source/'b.txt').write_bytes(secret)
    rights=registry([node('a',content=safe,path='a.txt'),node('b',content=secret,path='b.txt')])
    with pytest.raises(release.ReleaseError,match='PERSONAL_ABSOLUTE_PATH'):
        release.stage_public(rights,source,tmp_path/'public',whitelist={'a.txt':'a','b.txt':'b'})
    assert not (tmp_path/'public').exists() and not list(tmp_path.glob('.cbam-release-*'))


def test_stage_io_failure_removes_only_temporary_output(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();content=b'artificial\n';(source/'demo.txt').write_bytes(content)
    rights=registry([node(content=content)])
    monkeypatch.setattr(release.os,'rename',lambda *a,**kw:(_ for _ in ()).throw(OSError('artificial I/O failure')))
    with pytest.raises(release.ReleaseError,match='RELEASE_STAGING_IO_FAILED'):
        release.stage_public(rights,source,tmp_path/'public',whitelist={'demo.txt':'demo'})
    assert (source/'demo.txt').read_bytes()==content
    assert not (tmp_path/'public').exists() and not list(tmp_path.glob('.cbam-release-*'))


def test_no_source_ownership_bypass_via_artificial_generator():
    content=b'print("demo")\n'
    with pytest.raises(release.ReleaseError,match='SOURCE_OWNERSHIP_CLASSIFICATION_REQUIRED'):
        release.scan_content(content,registry([node(content=content,format='python')]),'demo')


def prepared_demo_receipt(tmp_path, monkeypatch):
    code=tmp_path/'installed';code.mkdir();run=tmp_path/'run';run.mkdir()
    names=['_demo.py','pipeline.py','market.py','panel.py','weather.py','synthetic.py',
           'design.py','estimators.py','_engine_context.py']
    hashes={}
    for name in names:
        content=('# Artificial approved-code identity fixture: '+name+'\n').encode()
        (code/name).write_bytes(content);hashes[name]=release.sha256(content)
    monkeypatch.setattr(release,'__file__',str(code/'release.py'))
    generator=node('demo-generator',content=(code/'_demo.py').read_bytes(),path='src/cbam_power/_demo.py',
                   format='python',generator_code_hashes=hashes)
    rights=registry([generator])
    identity=dict(mode='demo',code=hashes,contracts={},environment={})
    digest=release.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode())
    columns=['const','post','r','post_r']+['hour_'+str(x) for x in range(1,24)]+['weekday_'+str(x) for x in range(1,7)]
    payload=dict(mode='demo',label=release.DEMO_LABEL,private_inputs_read=0,models=[dict(
        model_id='DEMO_RDIT_ME',N=480,coefficient_name='post',status='ARTIFICIAL_ESTIMATE',columns=columns,estimate=4.)])
    json_bytes=lambda value:json.dumps(value,sort_keys=True).encode()
    outputs={
        'validate-inputs':{'fixture.json':json_bytes(dict(label=release.DEMO_LABEL,rows=1920,
            source='analytic artificial equations',target_jump=4.,weights={'GR_IT':.25,'AT_IT':.75},has_private_inputs=False))},
        'build-weather':{'weather.parquet':b'ARTIFICIAL RECEIPT FILE; no weather calculation in this unit test'},
        'build-panel':{'panel.parquet':b'ARTIFICIAL RECEIPT FILE; no market calculation in this unit test'},
        'build-synthetic':{'synthetic.parquet':b'ARTIFICIAL RECEIPT FILE; no synthetic calculation in this unit test'},
        'estimate-main':{'results.json':json_bytes(payload),'coefficients.csv':b'term,coefficient\npost,4\n','covariance.npz':b'ARTIFICIAL RECEIPT FILE'},
        'verify-main':{'verification.json':json_bytes(dict(status='PASS',label=release.DEMO_LABEL,
            known_jump=4.,absolute_error=0.,full_scientific_replication=False))},
    }
    receipt=dict(mode='demo',label=release.DEMO_LABEL,rights='ARTIFICIAL_DEMO_ONLY',identity=identity,stages={})
    for stage,files in outputs.items():
        (run/stage).mkdir()
        saved={}
        for filename,content in files.items():
            (run/stage/filename).write_bytes(content);saved[stage+'/'+filename]=release.sha256(content)
        receipt['stages'][stage]=dict(status='PASS',identity=digest,parameters={},outputs=saved)
    return run,receipt,rights,code


def test_demo_registry_requires_full_receipt_artifact_and_code_binding(tmp_path,monkeypatch):
    run,receipt,rights,code=prepared_demo_receipt(tmp_path,monkeypatch)
    dynamic,nid=release.demo_registry_for_run(run/'estimate-main/results.json',receipt,run,rights)
    assert dynamic.node(nid)['dependencies']==['demo-generator']
    assert dynamic.authorize(nid)['status']=='APPROVED'
    assert rights.nodes.keys()=={'demo-generator'}
    (code/'market.py').write_text('# changed code\n')
    with pytest.raises(release.ReleaseError,match='INSTALLED_DEMO_CODE_IDENTITY_CHANGED'):
        release.demo_registry_for_run(run/'estimate-main/results.json',receipt,run,rights)


@pytest.mark.parametrize('mutation,error',[
    ('missing_stage','DEMO_STAGE_SET_CHANGED'),('failed_stage','INCOMPLETE_OR_CHANGED_DEMO_STAGE'),
    ('unknown_output','DEMO_STAGE_ARTIFACT_SET_CHANGED'),('changed_output','DEMO_RECEIPT_ARTIFACT_CHANGED'),
    ('real_input','REAL_INPUT_IDENTITY_IN_DEMO'),('mode_only','DEMO_RECEIPT_REQUIRED'),
    ('wrong_model','DEMO_TOY_SPECIFICATION_CHANGED'),('wrong_fixture','DEMO_ANALYTIC_SOURCE_CHANGED'),
])
def test_demo_factory_negative_evidence(tmp_path,monkeypatch,mutation,error):
    run,receipt,rights,code=prepared_demo_receipt(tmp_path,monkeypatch)
    if mutation=='missing_stage':del receipt['stages']['verify-main']
    elif mutation=='failed_stage':receipt['stages']['build-panel']['status']='FAIL'
    elif mutation=='unknown_output':receipt['stages']['build-panel']['outputs']['build-panel/private.csv']='0'*64
    elif mutation=='changed_output':(run/'build-panel/panel.parquet').write_bytes(b'changed')
    elif mutation=='real_input':receipt['input_root']='private authorized input'
    elif mutation=='mode_only':receipt['rights']='PRIVATE_ONLY_PR6_PENDING'
    else:
        path='estimate-main/results.json' if mutation=='wrong_model' else 'validate-inputs/fixture.json'
        obj=json.loads((run/path).read_text())
        if mutation=='wrong_model':obj['models'][0]['N']=4320
        else:obj['has_private_inputs']=True
        content=json.dumps(obj).encode();(run/path).write_bytes(content)
        receipt['stages'][path.split('/')[0]]['outputs'][path]=release.sha256(content)
    with pytest.raises(release.ReleaseError,match=error):
        release.demo_registry_for_run(run/'estimate-main/results.json',receipt,run,rights)


def test_import_performs_no_io_or_network(monkeypatch):
    import importlib
    import socket
    def deny(*a,**kw): raise AssertionError('Unexpected import effect')
    with monkeypatch.context() as patch:
        patch.setattr(Path,'open',deny);patch.setattr(socket,'socket',deny)
        importlib.reload(release)


def actual_registry():
    from cbam_power.inputs import REPO
    root=Path(REPO)
    data=tomllib.loads((root/'manifests/release_rights.toml').read_text())
    inputs=tomllib.loads((root/'manifests/inputs.toml').read_text())
    return root,release.RightsRegistry(data),inputs


def test_actual_rights_inventory_matches_every_frozen_input_identity():
    root,rights,inputs=actual_registry()
    assets={n['asset_id']:n for n in rights.nodes.values() if n['kind']=='asset'}
    frozen={n['asset_id']:n for n in inputs['assets']}
    assert len(assets)==752 and assets.keys()==frozen.keys()
    assert Counter(n['status'] for n in assets.values())=={
        'APPROVED_WITH_CONDITIONS':97,'RESTRICTED':7,'EXCLUDED':167,'UNVERIFIED':481}
    for key,node_value in assets.items():
        assert node_value['id']=='asset:'+key
        assert node_value['sha256']==frozen[key]['sha256']
        assert node_value['path']==frozen[key]['relative_path']
        assert node_value['input_classification']==frozen[key]['classification']
    assert rights.data['input_manifest_sha256']==release.sha256((root/'manifests/inputs.toml').read_bytes())
    assert rights.data['baseline_registry_sha256']==release.sha256((root/'manifests/baselines.toml').read_bytes())
    with pytest.raises(release.ReleaseError,match='UNKNOWN_RELEASE_NODE'):
        rights.authorize('asset:unregistered_new_input')


def test_actual_conditional_and_real_artifacts_default_deny():
    _,rights,_=actual_registry()
    real=[n for n in rights.nodes.values() if n['kind']=='artifact' and n['origin']=='REAL']
    assert len(real)==114
    # Polynomial uses actual-flow outcomes, not the F2 spread regressors.
    assert set(rights.node('artifact:pr5-run-a:estimate-polynomial/results.json')['dependencies']) == {
        'variable:'+co+'_IT.scheduled_export_proxy' for co in ['AT','CH','SI','GR','ME']
    } | {'variable:calendar'}
    assert rights.node('artifact:pr5-run-a:build-synthetic/energy_mix.csv')['dependencies'] == ['variable:energy_mix']
    for n in real:
        with pytest.raises(release.ReleaseError):rights.authorize(n['id'])
    conditional=[n for n in rights.nodes.values() if n['kind']=='asset' and n['status']=='APPROVED_WITH_CONDITIONS']
    for n in conditional:
        assert n['conditions']
        with pytest.raises(release.ReleaseError):rights.authorize(n['id'])


def test_actual_source_mit_authorization_requires_the_notice():
    _,rights,_=actual_registry()
    sources=[n for n in rights.nodes.values() if n['kind']=='source']
    assert len(sources)==66 and {n['status'] for n in sources}=={'APPROVED_WITH_CONDITIONS'}
    assert {n['path'] for n in sources}==set(rights.data['candidate_whitelist'])
    for n in sources:
        with pytest.raises(release.ReleaseError,match='UNSATISFIED_RELEASE_CONDITION'):
            rights.authorize(n['id'])
        proof=rights.authorize(n['id'], satisfied_conditions={'retain_mit_notice':'mit-notice-in-candidate'})
        assert proof['status']=='APPROVED'
        assert 'hona-zhu-mit-authorization' in n['evidence']


def test_actual_core_frozen_hashes_and_contracts_are_unchanged():
    root,rights,_=actual_registry()
    anchors={
        'v3_panel':'a0eea50d8ac966fa9833eadc91d33e2f8f3f07c09bb1c025a5d79ed57bddc305',
        'v3_results':'997264dbc1748ce42e06fdd3b3693e7ee977899890c6752f30c62ed3db076c3c',
        'polynomial_results':'3857fbffc63e817ae9aea88e23cf9e6698ba9076ed9741ec18f3bc854b7437a9',
    }
    for aid,wanted in anchors.items():assert rights.node('asset:'+aid)['sha256']==wanted
    contracts={
        'config/main.toml':'9dae2a98f7b28d713fd5782c25fc51b3c8f9711dfd125bb6e62d374a2eebf267',
        'config/polynomial.toml':'f781899685b3666269ae674b72357d2e98d55708b1a2aa59cdffa86b605ce379',
        'manifests/baselines.toml':'abab449441409bcfee729a30fd0eb65f9e90239099e917b47828a104e0f56a32',
        'manifests/inputs.toml':'d6fa98e13f9a565051d14fcb086d8f280c066d3551b66ccea382a83ea7722310',
    }
    for name,wanted in contracts.items():assert release.sha256((root/name).read_bytes())==wanted


def test_actual_historical_presentation_and_incomplete_lineage_never_inherit_approval():
    _,rights,_=actual_registry()
    incomplete=[n for n in rights.nodes.values() if not n['lineage_complete']]
    historical=[n for n in rights.nodes.values() if n.get('input_classification') in {'H','P'}]
    assert incomplete and historical
    for n in incomplete+historical:
        with pytest.raises(release.ReleaseError):rights.authorize(n['id'])


def test_actual_cli_release_check_is_nonzero_for_real_and_unknown(capsys):
    from cbam_power import cli
    root,_,_=actual_registry()
    for artifact in ['artifact-class:coefficients','asset:unknown']:
        assert cli.main(['release-check','--registry',str(root/'manifests/release_rights.toml'),
                         '--artifact',artifact])==2
        assert capsys.readouterr().err


def test_actual_cli_export_refuses_missing_mit_notice_proof_before_creating_tree(tmp_path,capsys):
    from cbam_power import cli
    root,_,_=actual_registry()
    selection=tmp_path/'selection.json';selection.write_text(json.dumps({'.gitignore':'source:.gitignore'}))
    target=tmp_path/'public'
    assert cli.main(['export-public','--registry',str(root/'manifests/release_rights.toml'),
        '--source-root',str(Path(__file__).resolve().parents[1]),'--staging',str(target),'--selection',str(selection)])==2
    assert 'UNSATISFIED_RELEASE_CONDITION' in capsys.readouterr().err
    assert not target.exists()


# Every value below is an explicitly invalid, synthetic scanner fixture. No
# account token, connection file, endpoint or private credential is accessed.
INVALID_CREDENTIAL_FIXTURE = 'SYNTHETIC-INVALID-NOT-A-CREDENTIAL'


@pytest.mark.parametrize('key',['ENTSOE_TOKEN','ENTSOE_API_TOKEN','CDSAPI_KEY','CDSAPI_RC'])
@pytest.mark.parametrize('style',['unquoted','export','quoted','json'])
def test_energy_api_credentials_detected_with_redacted_errors(key,style):
    if style=='json':text=json.dumps({key:INVALID_CREDENTIAL_FIXTURE})
    elif style=='quoted':text=key+'="'+INVALID_CREDENTIAL_FIXTURE+'"'
    else:text=('export ' if style=='export' else '')+key+'='+INVALID_CREDENTIAL_FIXTURE
    content=text.encode()
    with pytest.raises(release.ReleaseError) as caught:
        release.scan_content(content,registry([node(content=content,format='json' if style=='json' else 'text')]),'demo')
    assert str(caught.value) in {'CREDENTIAL_CONFIGURATION_CONTENT','CREDENTIAL_CONTENT'}
    assert INVALID_CREDENTIAL_FIXTURE not in str(caught.value)
    assert key not in str(caught.value)


@pytest.mark.parametrize('key',['Authorization','authorization','apiKey','access_token','password','connection_string'])
def test_nested_connection_json_and_short_authorization_are_rejected(key):
    value='Basic TEST' if key.lower()=='authorization' else INVALID_CREDENTIAL_FIXTURE
    content=json.dumps({'connections':[{'headers':{key:value}}]}).encode()
    with pytest.raises(release.ReleaseError) as caught:
        release.scan_content(content,registry([node(content=content,format='json')]),'demo')
    assert release.public_error_code(caught.value)=='CREDENTIAL_CONFIGURATION_CONTENT'
    assert value not in str(caught.value)


def test_json_escaped_credential_key_and_cds_connection_semantics():
    key,value='authorization','Basic TEST'
    content=json.dumps({key:value}).replace(key,'\\u0061uthorization').encode()
    with pytest.raises(release.ReleaseError,match='CREDENTIAL_CONFIGURATION_CONTENT'):
        release.scan_content(content,registry([node(content=content,format='json')]),'demo')
    content=json.dumps(dict(url='https://cds.climate.copernicus.eu/api',key=INVALID_CREDENTIAL_FIXTURE)).encode()
    with pytest.raises(release.ReleaseError,match='CREDENTIAL_CONFIGURATION_CONTENT'):
        release.scan_content(content,registry([node(content=content,format='json')]),'demo')
    content=('url: https://cds.climate.copernicus.eu/api\n'+'key: '+INVALID_CREDENTIAL_FIXTURE+'\n').encode()
    with pytest.raises(release.ReleaseError,match='CREDENTIAL_CONFIGURATION_CONTENT'):
        release.scan_content(content,registry([node(content=content)]),'demo')


@pytest.mark.parametrize('filename',['.env','.env.local','.cdsapirc','.cdsapirc.backup','.netrc',
    'credentials.json','connections.json','kernel-invalid-fixture.json','jpserver-123.json'])
def test_credential_configuration_filenames_fail_even_with_benign_content(filename):
    content=b'SYNTHETIC INVALID FILE NAME FIXTURE; no secret value\n'
    with pytest.raises(release.ReleaseError,match='CREDENTIAL_FILE_NAME'):
        release.scan_content(content,registry([node(content=content,path=filename)]),'demo')


def test_archive_credential_filename_cannot_hide_behind_safe_node_path():
    inside=b'SYNTHETIC INVALID FILE NAME FIXTURE';content=zip_bytes('.env',inside)
    rights=registry([node('inner',content=inside,path='safe-name.txt'),
        node('outer',content=content,path='archive.zip',format='zip',members={'.env':'inner'},
             dependencies=['inner'],lineage_kind='derived')])
    with pytest.raises(release.ReleaseError,match='CREDENTIAL_FILE_NAME'):
        release.scan_content(content,rights,'outer')


@pytest.mark.parametrize('host',['localhost','127.0.0.1','10.1.2.3','172.20.1.2','192.168.1.8','[::1]',
                              'fixture.internal','fixture.local'])
def test_actual_private_endpoint_urls_fail_but_generic_mentions_are_safe(host):
    content=('https'+'://'+host+'/synthetic-invalid-endpoint').encode()
    with pytest.raises(release.ReleaseError,match='PRIVATE_ENDPOINT_CONTENT'):
        release.scan_content(content,registry([node(content=content)]),'demo')
    content=('Generic hostname mention only: '+host).encode()
    assert release.scan_content(content,registry([node(content=content)]),'demo')['status']=='SECURITY_CLEAN'


def test_userinfo_url_and_private_json_host_are_redacted():
    content=('https'+ '://invalid-user:'+INVALID_CREDENTIAL_FIXTURE+'@example.invalid').encode()
    with pytest.raises(release.ReleaseError,match='CREDENTIAL_CONFIGURATION_CONTENT') as caught:
        release.scan_content(content,registry([node(content=content)]),'demo')
    assert INVALID_CREDENTIAL_FIXTURE not in release.public_error_code(caught.value)
    content=json.dumps(dict(connection=dict(host='localhost',port=1234))).encode()
    with pytest.raises(release.ReleaseError,match='PRIVATE_ENDPOINT_CONTENT'):
        release.scan_content(content,registry([node(content=content,format='json')]),'demo')


def test_runtime_environment_references_and_hashes_are_not_credential_values():
    content=('import os\nvalue = os.environ.get('+repr('ENTSOE_TOKEN')+')\n'
             'other = os.getenv('+repr('CDSAPI_KEY')+')\n').encode()
    source=node(content=content,path='example.py',format='python',kind='source',origin='SOURCE',
                lineage_kind='source',status='PENDING_OWNER_APPROVAL',evidence=[])
    assert release.scan_content(content,registry([source]),'demo')['status']=='SECURITY_CLEAN'


def test_error_messages_never_include_rejected_node_values_or_io_paths(tmp_path):
    malicious_id=INVALID_CREDENTIAL_FIXTURE
    rights=registry([node(malicious_id,status='RESTRICTED')])
    with pytest.raises(release.ReleaseError) as caught:rights.authorize(malicious_id)
    assert str(caught.value)=='RIGHTS_NOT_APPROVED'
    assert malicious_id not in repr(caught.value)
    with pytest.raises(release.ReleaseError) as caught:
        release._safe_read(tmp_path,malicious_id+'.txt',1024)
    assert str(caught.value)=='RELEASE_SOURCE_UNAVAILABLE'
    assert malicious_id not in str(caught.value)
    assert release.public_error_code(ValueError(malicious_id))=='RELEASE_OPERATION_FAILED'


def candidate_fixture(tmp_path, *, content=b'print("artificial source")\n', filename='example.py', fmt='python'):
    source=tmp_path/'source';source.mkdir();(source/filename).write_bytes(content)
    record=node('source-fixture',content=content,path=filename,format=fmt,kind='source',origin='SOURCE',
                lineage_kind='source',status='PENDING_OWNER_APPROVAL',evidence=[])
    rights=registry([record]);rights.data['candidate_whitelist']=[filename]
    rights.data['candidate_preparation_authorized']=True
    return source,rights,{filename:'source-fixture'}


def test_prepare_candidate_allows_security_review_without_public_approval(tmp_path):
    source,rights,selection=candidate_fixture(tmp_path)
    destination=tmp_path/'review'
    report=release.prepare_candidate(rights,source,destination,whitelist=selection)
    assert report['status']=='SECURITY_REVIEW_CANDIDATE'
    assert report['release_status']=='PUBLIC_AUTH_PENDING'
    assert report['publication_approved'] is False and report['published'] is False
    assert (destination/'example.py').read_bytes()==(source/'example.py').read_bytes()
    with pytest.raises(release.ReleaseError,match='RIGHTS_NOT_APPROVED'):
        release.stage_public(rights,source,tmp_path/'public',whitelist=selection)
    assert not (tmp_path/'public').exists()


@pytest.mark.parametrize('change,error',[
    ('missing','EXACT_SOURCE_CANDIDATE_CLOSURE_REQUIRED'),
    ('unknown','UNKNOWN_RELEASE_NODE'),('raw','CANDIDATE_SOURCE_ONLY'),
    ('archive','CANDIDATE_SOURCE_FORMAT_ONLY'),('existing','RELEASE_DESTINATION_MUST_BE_NEW'),
    ('source_symlink','RELEASE_SOURCE_SYMLINK'),('destination_symlink','RELEASE_DESTINATION_SYMLINK'),
    ('changed_bytes','RELEASE_CONTENT_IDENTITY_CHANGED'),('protected','RELEASE_PROTECTED_ROOT_OVERLAP'),
])
def test_prepare_candidate_denies_partial_unknown_data_and_unsafe_paths(tmp_path,change,error):
    source,rights,selection=candidate_fixture(tmp_path)
    destination=tmp_path/'review';protected=()
    if change=='missing':rights.data['candidate_whitelist'].append('other.py')
    elif change=='unknown':selection['example.py']='unknown'
    elif change=='raw':rights.nodes['source-fixture']['kind']='asset';rights.nodes['source-fixture']['origin']='REAL'
    elif change=='archive':rights.nodes['source-fixture']['format']='zip'
    elif change=='existing':destination.mkdir()
    elif change=='source_symlink':
        (source/'example.py').unlink();(source/'example.py').symlink_to(tmp_path/'absent')
    elif change=='destination_symlink':destination.symlink_to(tmp_path/'absent')
    elif change=='changed_bytes':(source/'example.py').write_text('# modified synthetic fixture\n')
    elif change=='protected':protected=(tmp_path,)
    with pytest.raises(release.ReleaseError,match=error):
        release.prepare_candidate(rights,source,destination,whitelist=selection,protected_roots=protected)
    assert not list(tmp_path.glob('.cbam-candidate-*'))


def test_prepare_candidate_scans_all_bytes_before_creating_review_tree(tmp_path):
    source,rights,selection=candidate_fixture(tmp_path,content=b'PAR1SYNTHETIC-INVALID-PARQUET',filename='example.md',fmt='markdown')
    with pytest.raises(release.ReleaseError,match='RESTRICTED_BINARY_MAGIC'):
        release.prepare_candidate(rights,source,tmp_path/'review',whitelist=selection)
    assert not (tmp_path/'review').exists() and not list(tmp_path.glob('.cbam-candidate-*'))


def test_unknown_exception_codes_cannot_echo_payloads():
    for message in ['SYNTHETIC_INVALID_UPPERCASE', 'malformed invalid fixture', 'WRONG:payload']:
        error=release.ReleaseError(message)
        assert str(error)=='RELEASE_OPERATION_FAILED'
        assert release.public_error_code(error)=='RELEASE_OPERATION_FAILED'


def test_source_only_first_release_policy_excludes_even_licensed_real_assets():
    n=node('raw',kind='asset',origin='REAL',lineage_kind='raw',evidence=['license'],
           permissions=['public_repository','research_publication','commercial_reuse'])
    ev=evidence('license',kind='license',applicability='FROZEN_VINTAGE_CONFIRMED',
                permissions=['public_repository','research_publication','commercial_reuse'])
    data=dict(nodes=[n],evidence=[ev],publication_scope='SOURCE_AND_ARTIFICIAL_ONLY')
    rights=release.RightsRegistry(data)
    for context in release.CONTEXTS:
        with pytest.raises(release.ReleaseError,match='REAL_DATA_EXCLUDED_FROM_RELEASE'):
            rights.authorize('raw',context=context)
    del data['publication_scope']
    assert release.RightsRegistry(data).authorize('raw')['status']=='APPROVED'


def test_candidate_preparation_requires_separate_explicit_permission(tmp_path):
    source,rights,selection=candidate_fixture(tmp_path)
    del rights.data['candidate_preparation_authorized']
    with pytest.raises(release.ReleaseError,match='CANDIDATE_PREPARATION_NOT_AUTHORIZED'):
        release.prepare_candidate(rights,source,tmp_path/'review',whitelist=selection)
    assert not (tmp_path/'review').exists()


@pytest.mark.parametrize('operation',['prepare_candidate','stage_public'])
def test_destination_parent_traversal_cannot_bypass_protected_root(tmp_path,operation):
    source,rights,selection=candidate_fixture(tmp_path)
    outside=tmp_path/'outside';outside.mkdir()
    protected=tmp_path/'protected';protected.mkdir()
    destination=outside/'..'/'protected'/'forbidden'
    with pytest.raises(release.ReleaseError,match='INVALID_RELEASE_DESTINATION'):
        getattr(release,operation)(rights,source,destination,whitelist=selection,protected_roots=[protected])
    assert not (protected/'forbidden').exists()
    assert not list(protected.iterdir())


def test_decoded_toml_credential_key_cannot_bypass_scanning():
    encoded_key='access'+chr(92)+'u005f'+'token'
    content=('"'+encoded_key+'"="'+INVALID_CREDENTIAL_FIXTURE+'"\n').encode()
    source=node(content=content,path='example.toml',format='toml',kind='source',origin='SOURCE',
                lineage_kind='source',status='PENDING_OWNER_APPROVAL',evidence=[])
    with pytest.raises(release.ReleaseError,match='CREDENTIAL_CONFIGURATION_CONTENT') as caught:
        release.scan_content(content,registry([source]),'demo')
    assert INVALID_CREDENTIAL_FIXTURE not in str(caught.value)


@pytest.mark.parametrize('fmt',['json','toml'])
@pytest.mark.parametrize('location',['key','value'])
def test_decoded_structured_personal_paths_cannot_bypass_scanning(fmt,location):
    private='/'+'Users'+'/'+'synthetic_fixture'+'/'+'private'
    key,value=(private,'artificial') if location=='key' else ('note',private)
    if fmt=='json': text=json.dumps({key:value})
    else:text=json.dumps(key)+'='+json.dumps(value)+'\n'
    content=text.replace('/',chr(92)+'u002f').encode()
    record=node(content=content,format=fmt)
    if fmt=='toml':record.update(path='example.toml',kind='source',origin='SOURCE',
                                lineage_kind='source',status='PENDING_OWNER_APPROVAL',evidence=[])
    with pytest.raises(release.ReleaseError,match='PERSONAL_ABSOLUTE_PATH') as caught:
        release.scan_content(content,registry([record]),'demo')
    assert private not in str(caught.value)


@pytest.mark.parametrize('fmt',['zip','tar'])
@pytest.mark.parametrize('location',['separate_benign_members','payload','metadata'])
def test_archive_cds_credential_context_is_scoped_to_member_or_metadata(fmt,location):
    doc=b'Provider URL: https://cds.climate.copernicus.eu/api\n'
    python=b'key = "artificial_lookup"\n'
    configuration=doc+b'key: '+INVALID_CREDENTIAL_FIXTURE.encode()+b'\n'
    entries={'provider.txt':doc,'example.py':python}
    if location=='payload':entries['settings.txt']=configuration
    buffer=io.BytesIO()
    if fmt=='zip':
        with zipfile.ZipFile(buffer,'w') as archive:
            for name,content in entries.items():archive.writestr(name,content)
            if location=='metadata':archive.comment=configuration
    else:
        metadata={'comment':configuration.decode()} if location=='metadata' else {}
        with tarfile.open(fileobj=buffer,mode='w',format=tarfile.PAX_FORMAT,pax_headers=metadata) as archive:
            for name,content in entries.items():
                info=tarfile.TarInfo(name);info.size=len(content);archive.addfile(info,io.BytesIO(content))
    content=buffer.getvalue();children=[];members={}
    for i,(name,payload) in enumerate(entries.items()):
        nid='member-'+str(i);members[name]=nid
        children.append(node(nid,content=payload,path=name,format='python' if name.endswith('.py') else 'text',
                             kind='source',origin='SOURCE',lineage_kind='source',status='PENDING_OWNER_APPROVAL',evidence=[]))
    rights=registry(children+[node('archive',content=content,path='reviewed.'+fmt,format=fmt,
                                  members=members,dependencies=list(members.values()),lineage_kind='derived')])
    if location=='separate_benign_members':
        assert release.scan_content(content,rights,'archive')['status']=='SECURITY_CLEAN'
    else:
        with pytest.raises(release.ReleaseError,match='CREDENTIAL_CONFIGURATION_CONTENT') as caught:
            release.scan_content(content,rights,'archive')
        assert INVALID_CREDENTIAL_FIXTURE not in str(caught.value)


def test_gzip_wrapped_tar_keeps_member_scoped_cds_checks():
    entries={'provider.txt':b'Provider URL: https://cds.climate.copernicus.eu/api\n',
             'example.py':b'key = "artificial_lookup"\n'}
    buffer=io.BytesIO();children=[];members={}
    with tarfile.open(fileobj=buffer,mode='w') as archive:
        for i,(name,payload) in enumerate(entries.items()):
            info=tarfile.TarInfo(name);info.size=len(payload);archive.addfile(info,io.BytesIO(payload))
            nid='member-'+str(i);members[name]=nid
            children.append(node(nid,content=payload,path=name,format='python' if name.endswith('.py') else 'text',
                                 kind='source',origin='SOURCE',lineage_kind='source',status='PENDING_OWNER_APPROVAL',evidence=[]))
    tar_content=buffer.getvalue();compressed=gzip.compress(tar_content)
    children.append(node('tar',content=tar_content,path='sources.tar',format='tar',members=members,
                         dependencies=list(members.values()),lineage_kind='derived'))
    children.append(node('gzip',content=compressed,path='sources.tar.gz',format='gzip',
                         members={'sources.tar':'tar'},dependencies=['tar'],lineage_kind='derived'))
    assert release.scan_content(compressed,registry(children),'gzip')['status']=='SECURITY_CLEAN'


def test_gzip_contextual_cds_credentials_in_header_comment_are_rejected():
    inside=b'artificial text\n';compressed=gzip.compress(inside)
    comment=b'https://cds.climate.copernicus.eu/api\n'+b'key: '+INVALID_CREDENTIAL_FIXTURE.encode()
    header=bytearray(compressed[:10]);header[3]|=16
    content=bytes(header)+comment+b'\x00'+compressed[10:]
    assert gzip.decompress(content)==inside
    with pytest.raises(release.ReleaseError,match='CREDENTIAL_CONFIGURATION_CONTENT'):
        release.scan_content(content,archive_registry(content,fmt='gzip',inside=inside),'outer')


def test_unregistered_concatenated_gzip_stream_metadata_is_rejected():
    first=b'artificial first\n';second=b'artificial second\n'
    content=gzip.compress(first)+gzip.compress(second)
    with pytest.raises(release.ReleaseError,match='GZIP_SINGLE_MEMBER_IDENTITY_REQUIRED'):
        release.scan_content(content,archive_registry(content,fmt='gzip',inside=first+second),'outer')


def test_zip_extra_metadata_is_rejected_without_echoing_payload():
    # This deliberately invalid fixture is never a real service credential.
    import struct
    hidden=(b'https://cds.climate.copernicus.eu/api\nkey: '
            + INVALID_CREDENTIAL_FIXTURE.encode() + b'\n')
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as archive:
        entry=zipfile.ZipInfo('inside.txt')
        entry.extra=struct.pack('<HH',0xffff,len(hidden))+hidden
        archive.writestr(entry,b'artificial text\n')
    content=buffer.getvalue()
    with pytest.raises(release.ReleaseError,match='ZIP_EXTRA_METADATA_UNREVIEWED') as caught:
        release.scan_content(content,archive_registry(content),'outer')
    assert release.public_error_code(caught.value)=='ZIP_EXTRA_METADATA_UNREVIEWED'
    assert INVALID_CREDENTIAL_FIXTURE not in str(caught.value)


def test_even_benign_zip_extra_metadata_requires_explicit_review():
    import struct
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as archive:
        entry=zipfile.ZipInfo('inside.txt')
        entry.extra=struct.pack('<HH',0xffff,0)
        archive.writestr(entry,b'artificial text\n')
    content=buffer.getvalue()
    with pytest.raises(release.ReleaseError,match='ZIP_EXTRA_METADATA_UNREVIEWED'):
        release.scan_content(content,archive_registry(content),'outer')


def test_local_header_only_zip_extra_metadata_cannot_bypass_review():
    import struct
    hidden=(b'https://cds.climate.copernicus.eu/api\nkey: '
            + INVALID_CREDENTIAL_FIXTURE.encode() + b'\n')
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as archive:
        entry=zipfile.ZipInfo('inside.txt')
        entry.extra=struct.pack('<HH',0xffff,len(hidden))+hidden
        archive.writestr(entry,b'artificial text\n')
    content=bytearray(buffer.getvalue())
    end=content.rfind(b'PK\x05\x06')
    central=int.from_bytes(content[end+16:end+20],'little')
    name_length=int.from_bytes(content[central+28:central+30],'little')
    extra_length=int.from_bytes(content[central+30:central+32],'little')
    start=central+46+name_length
    del content[start:start+extra_length]
    content[central+30:central+32]=b'\x00\x00'
    end-=extra_length
    size=int.from_bytes(content[end+12:end+16],'little')-extra_length
    content[end+12:end+16]=size.to_bytes(4,'little')
    content=bytes(content)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assert archive.infolist()[0].extra==b''
        assert archive.read('inside.txt')==b'artificial text\n'
    with pytest.raises(release.ReleaseError,match='ZIP_EXTRA_METADATA_UNREVIEWED'):
        release.scan_content(content,archive_registry(content),'outer')
