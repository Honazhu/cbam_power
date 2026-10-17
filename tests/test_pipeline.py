"""PR4 public orchestration and installed-package tests; no real input reads."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from unittest.mock import patch

import pytest
from cbam_power.inputs import REPO
from cbam_power.pipeline import Run, PipelineError, demo, doctor, sha

pytestmark=pytest.mark.public
SOURCE_ROOT = Path(__file__).resolve().parents[1]


def test_order_missing_predecessor(tmp_path):
    r=Run(mode='demo',run_id='ordering',output_root=tmp_path)
    with pytest.raises(PipelineError,match='MISSING_STAGE'):r.execute('build-panel',lambda _:None)
    assert r.receipt['stages']=={}


def test_failed_stage_is_recorded(tmp_path):
    r=Run(mode='demo',run_id='failure',output_root=tmp_path)
    def fail(_):raise ValueError('intentional fixture failure')
    with pytest.raises(ValueError):r.execute('validate-inputs',fail)
    receipt=json.loads((r.path/'run.json').read_text())
    assert receipt['stages']['validate-inputs']['status']=='FAIL'
    with pytest.raises(PipelineError):r.execute('validate-inputs',lambda _:None)


def test_output_isolation_and_traversal(tmp_path):
    from cbam_power.inputs import InputError
    for root in [REPO,REPO/'outputs',REPO.parent]:
        with pytest.raises(InputError):Run(mode='demo',output_root=root)
    with pytest.raises((InputError,PipelineError)):Run(mode='demo',output_root=tmp_path,run_id='../escape')
    outside=tmp_path/'elsewhere';outside.mkdir();(tmp_path/'linked').symlink_to(outside,target_is_directory=True)
    with pytest.raises(PipelineError,match='SYMLINK'):Run(mode='demo',output_root=tmp_path/'linked')


def test_cache_identity_parameters_and_artifacts(tmp_path):
    r=Run(mode='demo',run_id='cache',output_root=tmp_path)
    def first(out):(out/'test.txt').write_text('artificial')
    r.execute('validate-inputs',first,parameters={'a':1})
    with patch('cbam_power.pipeline.atomic_json',side_effect=AssertionError('unexpected write')):
        r.execute('validate-inputs',lambda _:pytest.fail('cache recomputed'),parameters={'a':1})
    with pytest.raises(PipelineError):r.execute('validate-inputs',first,parameters={'a':2})
    (r.path/'validate-inputs/test.txt').write_text('changed')
    with pytest.raises(PipelineError,match='ARTIFACT'):r.require('validate-inputs')
    receipt=json.loads((r.path/'run.json').read_text());receipt['identity']['code']['pipeline.py']='wrong'
    (r.path/'run.json').write_text(json.dumps(receipt))
    with pytest.raises(PipelineError,match='IDENTITY'):Run(mode='demo',run_id='cache',output_root=tmp_path)


def test_transitive_stage_integrity(tmp_path):
    r=Run(mode='demo',output_root=tmp_path)
    def first(out):(out/'input.txt').write_text('artificial')
    r.execute('validate-inputs',first);r.execute('build-weather',lambda _:None)
    (r.path/'validate-inputs/input.txt').write_text('tamper')
    with pytest.raises(PipelineError,match='ARTIFACT'):r.require('build-weather')


def test_network_disabled_demo_end_to_end_and_repeat(tmp_path,monkeypatch):
    monkeypatch.setenv('CBAM_INPUT_ROOT',str(tmp_path/'must-never-be-read'))
    from cbam_power.inputs import InputCatalog
    with patch.object(InputCatalog,'__init__',side_effect=AssertionError('demo read private input')),patch.object(socket,'socket',side_effect=AssertionError('network')):
        r=Run(mode='demo',run_id='first',output_root=tmp_path);receipt=demo(r)
        other=Run(mode='demo',run_id='second',output_root=tmp_path);demo(other)
    assert list(receipt['stages'])==['validate-inputs','build-weather','build-panel','build-synthetic','estimate-main','verify-main','report']
    assert all(v['status']=='PASS' for v in receipt['stages'].values())
    assert sha(r.path/'estimate-main/results.json')==sha(other.path/'estimate-main/results.json')
    assert 'SYNTHETIC DATA — NOT EMPIRICAL RESEARCH RESULTS' in (r.path/'report/report.html').read_text()
    assert receipt['stages']['verify-main']['detail']['full_scientific_replication'] is False


def test_doctor_read_only_no_input_catalog(tmp_path):
    from cbam_power.inputs import InputCatalog
    before=set(tmp_path.iterdir())
    with patch.object(InputCatalog,'__init__',side_effect=AssertionError('not doctor responsibility')):
        d=doctor(mode='demo')
    assert d['python_312'];assert set(tmp_path.iterdir())==before


def test_imported_wheel_is_fresh():
    import cbam_power
    assert 'site-packages' in str(Path(cbam_power.__file__))
    installed=Path(cbam_power.__file__).parent
    source = SOURCE_ROOT/'src/cbam_power'
    expected = {p.relative_to(source): sha(p) for p in source.rglob('*.py')}
    actual = {p.relative_to(installed): sha(p) for p in installed.rglob('*.py')}
    assert expected and actual == expected
    import importlib.metadata
    import tomllib
    expected_version = tomllib.loads((SOURCE_ROOT/'pyproject.toml').read_text())['project']['version']
    assert importlib.metadata.version('cbam-power') == expected_version


def test_cross_cwd_cli_and_clean_import(tmp_path):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None);env.pop('CBAM_INPUT_ROOT',None)
    modules=[str(p.relative_to(SOURCE_ROOT/'src')).replace('/','.')[:-3] for p in (SOURCE_ROOT/'src/cbam_power').rglob('*.py') if p.name!='__init__.py']
    assert modules, 'No source modules were inspected'
    script='import importlib\n'+''.join(f'importlib.import_module({m!r})\n' for m in modules)
    imported=subprocess.run([sys.executable,'-B','-c',script],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert imported.returncode==0,imported.stderr
    cli=Path(sys.executable).parent/'cbam-power'
    p=subprocess.run([str(cli),'doctor','--mode','demo'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['python_312']
    assert list(tmp_path.iterdir())==[]
    bad=subprocess.run([str(cli),'estimate','--spec','main','--run-id','missing'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert bad.returncode!=0 and 'BLOCKED_CBAM_INPUT_ROOT_REQUIRED' in bad.stderr


def test_package_no_legacy_import_or_optimizer_call():
    files=list((SOURCE_ROOT/'src/cbam_power').rglob('*.py'))
    assert files, 'No source files were inspected'
    for p in files:
        text=p.read_text();tree=ast.parse(text)
        assert 'sys.path.insert' not in text and 'sys.path.append' not in text
        assert '/Users/' not in text
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom):
                assert not (n.module or '').startswith(('v3_formal_cbam','v2_synthetic_cbam'))
            if isinstance(n,ast.Call):
                name=getattr(n.func,'attr',getattr(n.func,'id',''))
                assert name not in {'minimize','fit_weights','urlretrieve','urlopen','requests.get'}


def test_validation_independent_timestamp_bridge_and_nulls():
    import numpy as np
    from cbam_power.validation import timestamp_proof,assert_close,ValidationError
    values=np.array([0,3600*10**9,7200*10**9],dtype='<i8')
    oldhash=hashlib.sha256((values//1000).astype('<i8').tobytes()).hexdigest()
    assert timestamp_proof(values,values.copy(),oldhash)=='us'
    with pytest.raises(ValidationError):timestamp_proof(values,values+1,oldhash)
    with pytest.raises(ValidationError):assert_close([1.,np.nan],[1.,0.],{'atol':1e-6,'rtol':1e-8})


def test_authorized_verify_dispatch_real_weather_interface(tmp_path):
    """Artificial saved artifacts exercise verify orchestration, not frozen values."""
    from contextlib import contextmanager
    import io
    import pandas as pd
    from cbam_power.pipeline import run_stage
    weather=pd.DataFrame({'timestamp_utc':pd.date_range('2026-01-01',periods=3,freq='h',tz='UTC'),
        'geography':['ME']*3,'temp_c':[1.,2.,3.],'wind_ms':[2.,3.,4.],
        'ssrd_processed':[1.,1.,1.],'tp_processed':[0.,0.,0.]})
    class FakeCatalog:
        baselines={'validation':{'weather':{'strict_abs_less_than':1e-9}}}
        @contextmanager
        def open_input(self, aid, *, purpose):
            assert purpose=='verify'
            b=io.BytesIO();weather.to_parquet(b,index=False);b.seek(0);yield b
    run=Run(mode='authorized',run_id='artificial_dispatch',output_root=tmp_path/'runs',input_root=tmp_path/'inputs')
    run.execute('validate-inputs',lambda _:None)
    def saved_weather(out):
        weather.to_parquet(out/'weather.parquet',index=False)
        weather.to_parquet(out/'v23_view.parquet',index=False)
    run.execute('build-weather',saved_weather)
    run.execute('build-panel',lambda out:weather.to_parquet(out/'panel.parquet',index=False))
    run.execute('build-synthetic',lambda _:None)
    run.execute('estimate-main',lambda _:None)
    with patch.object(run,'catalog',return_value=FakeCatalog()), \
         patch('cbam_power.validation.verify_baseline',return_value={'status':'PASS'}), \
         patch('cbam_power.synthetic.validate_saved',return_value=[]), \
         patch('cbam_power.weather.coverage',return_value={'artificial':True}):
        result=run_stage(run,'verify-main')
    assert result['status']=='PASS'
    assert len(result['detail']['weather'])==3
    assert all(c['status']=='PASS' for c in result['detail']['weather'].values())



def test_thread_settings_part_of_environment_identity(monkeypatch):
    from cbam_power.pipeline import environment_identity
    monkeypatch.delenv('OMP_NUM_THREADS', raising=False)
    before=environment_identity()
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    after=environment_identity()
    assert before['thread_settings']['OMP_NUM_THREADS']=='UNSET'
    assert after['thread_settings']['OMP_NUM_THREADS']=='1'
    assert before!=after


def test_polynomial_references_explicit_private_boundary(tmp_path,monkeypatch):
    from cbam_power.pipeline import supplemental_references
    from cbam_power.inputs import InputError
    run=Run(mode='demo',run_id='references',output_root=tmp_path/'runs')
    monkeypatch.delenv('CBAM_REFERENCE_ROOT',raising=False)
    with pytest.raises(PipelineError,match='REFERENCE_ROOT_REQUIRED'):supplemental_references(run)
    monkeypatch.setenv('CBAM_REFERENCE_ROOT',str(run.path))
    with pytest.raises(InputError):supplemental_references(run)
    refs=tmp_path/'refs';refs.mkdir()
    monkeypatch.setenv('CBAM_REFERENCE_ROOT',str(refs))
    mapping=supplemental_references(run)
    assert len(mapping)==6 and all(p.parent==refs for p in mapping.values())
    link=tmp_path/'link';link.symlink_to(refs,target_is_directory=True)
    monkeypatch.setenv('CBAM_REFERENCE_ROOT',str(link))
    with pytest.raises(PipelineError,match='SYMLINK'):supplemental_references(run)
