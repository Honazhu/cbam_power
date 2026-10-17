"""Saved-output reporting, with independent sentinels against hidden calculation."""
from contextlib import ExitStack
from copy import deepcopy
import csv
import hashlib
import importlib
import json
from pathlib import Path
import socket
import urllib.request
from unittest.mock import patch

import pytest

from cbam_power.reporting import figures, reports, tables

pytestmark = pytest.mark.public


def saved_payload():
    # Artificial numbers are specified directly, not produced by an estimator.
    coefficient = dict(term='post', coefficient=3., SE=1., CI_low=1.04,
                       CI_high=4.96, p_value=.0027)
    model = dict(model_id='DEMO_MANUAL', coefficient_name='post', estimate=3.,
                 SE=1., CI_low=1.04, CI_high=4.96, p_value=.0027,
                 N=48, status='ARTIFICIAL_ESTIMATE', coefficients=[coefficient])
    return dict(mode='demo', label=reports.DEMO_LABEL, models=[model])


def write_saved(tmp_path, payload=None):
    path = tmp_path.resolve()/'saved.json'
    path.write_text(json.dumps(saved_payload() if payload is None else payload))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def reviewed_artificial_registry(source):
    """Explicit permission for original manual test numbers, never real results."""
    from cbam_power.release import RightsRegistry
    identity=hashlib.sha256(source.read_bytes()).hexdigest()
    common=dict(kind='artifact',origin='ARTIFICIAL',status='APPROVED_PUBLIC',
        lineage_complete=True,review_group='report-fixtures',evidence=['fixture-approval'],
        conditions=[],permissions=['public_repository'],reason='Reviewed original artificial unit-test numbers')
    generator=dict(common,id='manual-fixture',dependencies=[],lineage_kind='artificial_generator',
        sha256=hashlib.sha256(b'Original manual test fixture: jump3; SE1; N48').hexdigest(),format='text')
    result=dict(common,id='fixture-result',dependencies=['manual-fixture'],lineage_kind='derived',
        sha256=identity,path=source.name,format='json')
    evidence=dict(id='fixture-approval',kind='user_demo_authorization',status='VERIFIED',
        source='Task explicitly authorizes original artificial fixtures and public-demo tests',
        scope=['group:report-fixtures'],applicability='NOT_APPLICABLE',permissions=['public_repository'])
    return RightsRegistry(dict(nodes=[generator,result],evidence=[evidence]))


def public_fixture_options(source):
    return dict(rights_registry=reviewed_artificial_registry(source),artifact_id='fixture-result')


def render(tmp_path, payload=None, **kwargs):
    source, identity = write_saved(tmp_path, payload)
    out = tmp_path.resolve()/'report'
    result = reports.render_report(source, out, mode=kwargs.pop('mode','demo'),
                                   audience=kwargs.pop('audience','public'), expected_sha=identity,
                                   **public_fixture_options(source), **kwargs)
    return result, source, out


def test_reporting_estimator_optimizer_network_and_input_sentinels(tmp_path):
    from cbam_power import estimators, synthetic
    from cbam_power.inputs import InputCatalog
    source, identity = write_saved(tmp_path)
    out = tmp_path.resolve()/'report'
    with ExitStack() as stack:
        mocks = []
        for owner, name in [(estimators, 'fit'), (estimators, 'lstsq'),
                            (estimators, 'hac'), (estimators, 'joint_covariance'),
                            (synthetic, 'fit_weights'), (synthetic, 'simplex_least_squares'),
                            (socket, 'socket'), (urllib.request, 'urlopen'),
                            (urllib.request, 'urlretrieve'), (InputCatalog, 'open_input')]:
            mocks.append(stack.enter_context(patch.object(owner, name, create=True,
                side_effect=AssertionError('Reporting invoked forbidden '+name))))
        result = reports.render_report(source,out,mode='demo',audience='public',expected_sha=identity,
                                       **public_fixture_options(source))
        for sentinel in mocks:
            sentinel.assert_not_called()
    assert result['estimation_performed'] is False
    assert result['network_access'] is False
    assert (out/'report.html').is_file()


def test_numeric_sidecar_traces_saved_values_and_byte_identity(tmp_path):
    result, source, out = render(tmp_path)
    sidecar = json.loads((out/'numeric_sidecar.json').read_text())
    assert sidecar['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert sidecar['rows'] == tables.rows(saved_payload())
    assert sidecar['estimation_performed'] is False
    with (out/'table.csv').open() as handle:
        row = next(csv.DictReader(handle))
    assert row['label'] == reports.DEMO_LABEL
    assert float(row['estimate']) == 3.
    assert float(row['CI_low']) == 1.04
    assert float(row['CI_high']) == 4.96
    assert result['label'] == reports.DEMO_LABEL
    assert reports.DEMO_LABEL in (out/'report.html').read_text()
    assert reports.DEMO_LABEL in (out/'coefficients.svg').read_text()


def test_tampered_result_hash_fails_before_output(tmp_path):
    source, identity = write_saved(tmp_path)
    source.write_text(source.read_text()+' ')
    out = tmp_path.resolve()/'report'
    with pytest.raises(ValueError,match='IDENTITY_CHANGED'):
        reports.render_report(source,out,mode='demo',audience='public',expected_sha=identity,
                              **public_fixture_options(source))
    assert not out.exists()


def test_authorized_public_export_is_refused_before_any_read(tmp_path):
    with patch.object(Path,'read_bytes',side_effect=AssertionError('Private source read')):
        with pytest.raises(PermissionError,match='PR6_ARTIFACT_RIGHTS_PENDING'):
            reports.render_report(tmp_path/'private.json',tmp_path/'out',
                                  mode='authorized',audience='public',expected_sha='0'*64)
    assert not (tmp_path/'out').exists()


def test_local_authorized_report_retains_private_label(tmp_path):
    payload = saved_payload()
    payload['mode'] = 'authorized'
    payload['models'][0]['model_id'] = 'ARTIFICIAL_LOCAL_REPORT_FIXTURE'
    result, _, out = render(tmp_path,payload,mode='authorized',audience='local')
    assert 'PRIVATE RESEARCH RESULTS' in result['label']
    assert 'release rights unverified' in (out/'report.html').read_text()
    with (out/'table.csv').open() as handle:
        assert 'label' not in next(csv.DictReader(handle))


@pytest.mark.parametrize('change', ['label','mode','model_id'])
def test_real_or_unlabeled_results_cannot_impersonate_demo(tmp_path,change):
    payload=saved_payload()
    if change=='label': payload['label']='Research results'
    elif change=='mode': payload['mode']='authorized'
    else: payload['models'][0]['model_id']='V3_REAL_MODEL'
    with pytest.raises(PermissionError):
        render(tmp_path,payload)
    assert not (tmp_path/'report').exists()


def test_demo_cannot_impersonate_authorized_analysis(tmp_path):
    with pytest.raises(ValueError,match='DEMO_IS_NOT_AUTHORIZED'):
        render(tmp_path,mode='authorized',audience='local')


@pytest.mark.parametrize('field', ['estimate','SE','CI_low','CI_high','p_value'])
def test_mismatched_primary_number_and_sidecar_rejected(tmp_path,field):
    payload=saved_payload(); payload['models'][0][field] += .01
    with pytest.raises(ValueError,match='NUMERIC_SIDECAR_MISMATCH'):
        render(tmp_path,payload)
    assert not (tmp_path/'report').exists()


@pytest.mark.parametrize('value', [float('nan'),float('inf'),-float('inf'),'3.0',True])
def test_nonfinite_or_nonnumeric_values_cannot_pass_sidecar_check(tmp_path,value):
    payload=saved_payload()
    payload['models'][0]['estimate']=value
    payload['models'][0]['coefficients'][0]['coefficient']=value
    with pytest.raises(ValueError,match='NONFINITE_OR_INVALID_REPORT_VALUE'):
        render(tmp_path,payload)
    assert not (tmp_path/'report').exists()


@pytest.mark.parametrize('kind', ['missing','duplicated_term','unknown_term','duplicated_model','empty'])
def test_incomplete_or_duplicate_sidecars_fail(tmp_path,kind):
    payload=saved_payload(); model=payload['models'][0]
    if kind=='missing': del model['coefficients']
    elif kind=='duplicated_term': model['coefficients'] *= 2
    elif kind=='unknown_term': model['coefficients'][0]['term']='OTHER'
    elif kind=='duplicated_model': payload['models'].append(deepcopy(model))
    else: payload['models']=[]
    with pytest.raises((ValueError,PermissionError)):
        render(tmp_path,payload)
    assert not (tmp_path/'report').exists()


def test_report_only_repeat_does_not_overwrite_existing_artifacts(tmp_path):
    _, source, out = render(tmp_path)
    before={p.name:p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError,match='REPORT_OUTPUT_EXISTS'):
        reports.render_report(source,out,mode='demo',audience='public',
                              expected_sha=hashlib.sha256(source.read_bytes()).hexdigest(),
                              **public_fixture_options(source))
    assert before == {p.name:p.read_bytes() for p in out.iterdir()}


@pytest.mark.parametrize('ancestor', [False,True])
def test_report_refuses_output_symlink_and_ancestor_escape(tmp_path,ancestor):
    source, identity=write_saved(tmp_path)
    real=tmp_path.resolve()/'real'; real.mkdir()
    linked=tmp_path.resolve()/'link'; linked.symlink_to(real,target_is_directory=True)
    target=linked/'report' if ancestor else linked
    with pytest.raises(ValueError,match='REPORT_OUTPUT_SYMLINK'):
        reports.render_report(source,target,mode='demo',audience='public',expected_sha=identity,
                              **public_fixture_options(source))
    assert list(real.iterdir()) == []


def test_reporting_imports_do_not_read_payloads_or_write_outputs():
    with ExitStack() as stack:
        for name in ['read_bytes','read_text','write_bytes','write_text','mkdir','open']:
            stack.enter_context(patch.object(Path,name,side_effect=AssertionError('Import I/O '+name)))
        for module in [tables,figures,reports]:
            importlib.reload(module)


def test_report_escapes_untrusted_display_labels(tmp_path):
    payload=saved_payload();payload['models'][0]['model_id']='DEMO_<script>alert(1)</script>'
    _,_,out=render(tmp_path,payload)
    html=(out/'report.html').read_text();svg=(out/'coefficients.svg').read_text()
    assert '<script>' not in html and '<script>' not in svg
    assert '&lt;script&gt;' in html and '&lt;script&gt;' in svg


def prepared_report_run(tmp_path, *, verified=True, mode='demo'):
    """Artificial receipts isolate report orchestration from scientific fitting."""
    from cbam_power.pipeline import Run, atomic_json
    run=Run(mode=mode,run_id='artificial_report_fixture',
            output_root=tmp_path.resolve()/'runs',
            input_root=(tmp_path.resolve()/'unused_authorized_inputs') if mode=='authorized' else None)
    for stage in ['validate-inputs','build-weather','build-panel','build-synthetic']:
        run.execute(stage,lambda out: {'fixture':'ARTIFICIAL_REPORT_STAGE_PRECONDITION'})
    payload=saved_payload()
    if mode=='authorized':
        payload['mode']='authorized'
        payload['models'][0]['model_id']='ARTIFICIAL_LOCAL_REPORT_FIXTURE'
    def estimated(out):
        atomic_json(out/'results.json',payload)
        return {'fixture':'SAVED_MANUAL_NUMBERS_NO_ESTIMATION'}
    run.execute('estimate-main',estimated)
    if verified:
        run.execute('verify-main',lambda out: {'fixture':'ARTIFICIAL_REPORT_STAGE_PRECONDITION'})
    return run


def test_pipeline_report_rejects_tampered_estimate_after_verification(tmp_path):
    from cbam_power.pipeline import run_stage, PipelineError
    run=prepared_report_run(tmp_path)
    source=run.artifact('estimate-main/results.json')
    source.write_text(source.read_text()+' ')
    with pytest.raises(PipelineError,match='ARTIFACT_IDENTITY_CHANGED'):
        run_stage(run,'report',audience='public')
    assert not run.artifact('report').exists()


def test_pipeline_report_requires_verified_stage(tmp_path):
    from cbam_power.pipeline import run_stage, PipelineError
    run=prepared_report_run(tmp_path,verified=False)
    with pytest.raises(PipelineError,match='MISSING_STAGE'):
        run_stage(run,'report',audience='public')
    assert not run.artifact('report').exists()


def test_pipeline_cannot_publicly_export_authorized_results(tmp_path):
    from cbam_power.pipeline import run_stage
    run=prepared_report_run(tmp_path,mode='authorized')
    with pytest.raises(PermissionError,match='PR6_ARTIFACT_RIGHTS_PENDING'):
        run_stage(run,'report',audience='public')
    assert list(run.artifact('report').iterdir()) == []
    assert run.receipt['stages']['report']['status']=='FAIL'


def test_demo_label_without_reviewed_provenance_is_denied(tmp_path):
    source,identity=write_saved(tmp_path)
    with pytest.raises(PermissionError,match='REVIEWED_ARTIFACT_REQUIRED'):
        reports.render_report(source,tmp_path/'out',mode='demo',audience='public',expected_sha=identity)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('field,value',[('kind','source'),('origin','REAL'),('format','text')])
def test_public_report_rejects_wrong_source_registry_kind(tmp_path,field,value):
    from cbam_power.release import RightsRegistry
    source,identity=write_saved(tmp_path);data=reviewed_artificial_registry(source).data
    data['nodes'][1][field]=value
    registry=RightsRegistry(data)
    with patch.object(Path,'read_bytes',side_effect=AssertionError('Unapproved bytes read')):
        with pytest.raises(PermissionError,match='ARTIFACT_KIND_MISMATCH'):
            reports.render_report(source,tmp_path/'out',mode='demo',audience='public',
                expected_sha=identity,rights_registry=registry,artifact_id='fixture-result')
    assert not (tmp_path/'out').exists()


def test_public_report_rejects_swapped_registered_bytes(tmp_path):
    source,identity=write_saved(tmp_path);registry=reviewed_artificial_registry(source)
    source.write_text(source.read_text()+' ')
    current=hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(PermissionError,match='CONTENT_IDENTITY_CHANGED'):
        reports.render_report(source,tmp_path/'out',mode='demo',audience='public',expected_sha=current,
            rights_registry=registry,artifact_id='fixture-result')
    assert not (tmp_path/'out').exists()


def test_public_report_retains_all_four_artifact_permission_proofs(tmp_path):
    result,_,out=render(tmp_path)
    assert set(result['publication_proofs'])==set(reports.OUTPUT_FORMATS)
    for name,proof in result['publication_proofs'].items():
        assert proof['status']=='APPROVED'
        assert proof['security']['status']=='SECURITY_CLEAN'
        assert proof['security']['sha256']==hashlib.sha256((out/name).read_bytes()).hexdigest()
        assert 'fixture-result' in {item['node_id'] for item in proof['lineage']}


def test_authorized_saved_result_approval_does_not_approve_new_report(tmp_path):
    from cbam_power.release import RightsRegistry
    raw=dict(id='raw',kind='asset',origin='REAL',status='APPROVED_PUBLIC',dependencies=[],
        lineage_complete=True,lineage_kind='raw',review_group='real-fixture',evidence=['license'],
        conditions=[],permissions=['public_repository'],reason='Artificial policy fixture, no actual research data')
    result=dict(raw,id='result',kind='artifact',dependencies=['raw'],lineage_kind='derived',
        evidence=['publication'],sha256='0'*64,path='results.json',format='json')
    evidence=[dict(id='license',kind='license',status='VERIFIED',source='Artificial license test',
        scope=['raw'],applicability='FROZEN_VINTAGE_CONFIRMED',permissions=['public_repository']),
        dict(id='publication',kind='publication_approval',status='VERIFIED',source='Artificial publication test',
        scope=['result'],applicability='NOT_APPLICABLE',permissions=['public_repository'])]
    registry=RightsRegistry(dict(nodes=[raw,result],evidence=evidence))
    with patch.object(Path,'read_bytes',side_effect=AssertionError('No output approval, no read')):
        with pytest.raises(PermissionError,match='EXACT_OUTPUT_ARTIFACTS_REQUIRED'):
            reports.render_report(tmp_path/'private.json',tmp_path/'out',mode='authorized',audience='public',
                expected_sha='0'*64,rights_registry=registry,artifact_id='result')
    assert not (tmp_path/'out').exists()
