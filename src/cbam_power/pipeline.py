"""Explicit run orchestration. No archive access, implicit fitting, or baseline updates.

Scientific functions live in the domain modules. This module owns only A (run
outputs); I/B are accessed through InputCatalog. Caches require the complete
input/config/code/environment/parameter identity and verified output bytes.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import tomllib
import uuid

from .inputs import InputCatalog, InputError, REPO, disjoint, relative_path

DEMO_LABEL = 'SYNTHETIC DATA — NOT EMPIRICAL RESEARCH RESULTS'
AUTHORIZED_STATUS = 'AUTHORIZED FULL NUMERICAL REPLICATION: NOT YET VERIFIED'
STAGES = {'validate-inputs': (), 'build-weather': ('validate-inputs',),
          'build-panel': ('build-weather',), 'build-synthetic': ('build-panel',),
          'estimate-main': ('build-synthetic',), 'verify-main': ('estimate-main',),
          'estimate-polynomial': ('verify-main',),
          'verify-polynomial': ('estimate-polynomial',), 'report': ('verify-main',)}


class PipelineError(ValueError):
    pass


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def clean(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.generic):
        return clean(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8') as f:
        temporary = Path(f.name)
        json.dump(clean(obj), f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def environment_identity():
    names = ['cbam-power', 'numpy', 'pandas', 'scipy', 'pyarrow', 'xarray',
             'netCDF4', 'h5netcdf', 'h5py', 'shapely', 'pyproj', 'statsmodels']
    versions = {}
    for name in names:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'UNAVAILABLE'
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions,
                thread_settings={name: os.environ.get(name, 'UNSET') for name in
                    ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']})


def validate_scientific_contracts(project_root):
    expected = {'config/main.toml': '7bb0fdada81087c3dc8b07f466f365d0eeb42269e073b46076379cf8afce3378',
                'config/polynomial.toml': '2fb4ee49b144ba9fa3ba28d9f6bdb93e081cb8439caaf83aad13dba85a720a56'}
    for name, wanted in expected.items():
        obj = tomllib.loads((Path(project_root)/name).read_text())
        encoded = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
        if hashlib.sha256(encoded).hexdigest() != wanted:
            raise PipelineError('SCIENTIFIC_CONFIGURATION_CHANGED: '+name)


def identity(mode, project_root=REPO):
    root = Path(project_root)
    validate_scientific_contracts(root)
    paths = ['config/main.toml', 'config/polynomial.toml', 'manifests/baselines.toml',
             'manifests/inputs.toml', 'uv.lock']
    contracts = {p: sha(root/p) for p in paths}
    package = Path(__file__).parent
    code = {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob('*.py'))}
    return dict(mode=mode, contracts=contracts, code=code, environment=environment_identity())


def _safe_tree(root, path):
    """Reject symlinks in A, including already existing parent components."""
    root, path = Path(root).absolute(), Path(path).absolute()
    if not path.is_relative_to(root): raise PipelineError('OUTPUT_PATH_ESCAPE')
    for p in [path, *path.parents]:
        if p.is_symlink(): raise PipelineError('OUTPUT_SYMLINK_FORBIDDEN')
    return path


class Run:
    def __init__(self, *, mode='authorized', run_id=None, output_root=None,
                 input_root=None, project_root=REPO):
        if mode not in {'authorized', 'demo'}: raise PipelineError('UNKNOWN_MODE')
        self.mode, self.project_root = mode, Path(project_root).resolve()
        self.input_root = (input_root or os.environ.get('CBAM_INPUT_ROOT')) if mode == 'authorized' else None
        if mode == 'authorized' and not self.input_root:
            raise PipelineError('BLOCKED_CBAM_INPUT_ROOT_REQUIRED')
        default = Path(self.input_root).parent/'runs' if mode == 'authorized' else Path(tempfile.gettempdir()).resolve()/'cbam-power-demo-runs'
        self.output_root = Path(output_root or os.environ.get('CBAM_OUTPUT_ROOT') or default).absolute()
        disjoint(self.output_root, self.project_root)
        if self.input_root: disjoint(self.output_root, self.input_root)
        run_id = run_id or f'{mode}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:10]}'
        if len(relative_path(run_id).parts) != 1: raise PipelineError('INVALID_RUN_ID')
        self.path = _safe_tree(self.output_root, self.output_root/run_id)
        self.identity = identity(mode, self.project_root)
        self.path.mkdir(parents=True, exist_ok=True)
        receipt = self.path/'run.json'
        if receipt.exists():
            self.receipt = json.loads(receipt.read_text())
            if self.receipt['identity'] != self.identity:
                raise PipelineError('RUN_IDENTITY_CHANGED_USE_NEW_RUN')
        else:
            self.receipt = dict(run_id=run_id, mode=mode, identity=self.identity,
                created_utc=datetime.now(timezone.utc).isoformat(), stages={}, errors=[],
                label=DEMO_LABEL if mode == 'demo' else AUTHORIZED_STATUS,
                rights='ARTIFICIAL_DEMO_ONLY' if mode == 'demo' else 'PRIVATE_ONLY_PR6_PENDING')
            self.save()

    def save(self):
        self.receipt['completed_stages'] = [name for name, record in sorted(
            self.receipt['stages'].items(), key=lambda item: item[1]['started_utc'])
            if record['status'] == 'PASS']
        atomic_json(_safe_tree(self.path, self.path/'run.json'), self.receipt)

    def catalog(self):
        if self.mode != 'authorized': raise PipelineError('DEMO_PRIVATE_ACCESS_FORBIDDEN')
        return InputCatalog(input_root=self.input_root, output_root=self.output_root,
            manifest_path=self.project_root/'manifests/inputs.toml',
            baseline_manifest_path=self.project_root/'manifests/baselines.toml')

    def artifact(self, relative):
        p = self.path.joinpath(*relative_path(relative).parts)
        return _safe_tree(self.path, p)

    def require(self, stage):
        for dep in STAGES[stage]: self.require(dep)
        record = self.receipt['stages'].get(stage)
        if not record or record['status'] != 'PASS':
            raise PipelineError('MISSING_STAGE: '+stage)
        if record['identity'] != digest(self.identity): raise PipelineError('CACHE_IDENTITY_CHANGED')
        for rel, expected in record['outputs'].items():
            p = self.artifact(rel)
            if not p.is_file() or sha(p) != expected: raise PipelineError('ARTIFACT_IDENTITY_CHANGED: '+rel)
        return record

    def execute(self, stage, callback, *, parameters=None):
        if stage not in STAGES: raise PipelineError('UNKNOWN_STAGE')
        for dep in STAGES[stage]: self.require(dep)
        parameters = parameters or {}
        prior = self.receipt['stages'].get(stage)
        if prior:
            if prior['status'] == 'PASS' and prior['parameters'] == parameters:
                self.require(stage); return prior
            raise PipelineError('STAGE_EXISTS_USE_NEW_RUN')
        directory = self.artifact(stage)
        if directory.exists(): raise PipelineError('UNREGISTERED_STAGE_OUTPUT')
        directory.mkdir()
        record = dict(status='RUNNING', identity=digest(self.identity), parameters=parameters,
                      started_utc=datetime.now(timezone.utc).isoformat(), outputs={})
        self.receipt['stages'][stage] = record; self.save()
        try:
            detail = callback(directory) or {}
            record['outputs'] = {str(p.relative_to(self.path)): sha(p)
                for p in sorted(directory.rglob('*')) if p.is_file() and not p.is_symlink()}
            # A symlink is never a registered output or a reusable cache.
            if any(p.is_symlink() for p in directory.rglob('*')): raise PipelineError('OUTPUT_SYMLINK_FORBIDDEN')
            record.update(status='PASS', detail=clean(detail))
        except Exception as error:
            record.update(status='FAIL', error_type=type(error).__name__, error=str(error))
            self.receipt['errors'].append(dict(stage=stage, error=str(error)))
            raise
        finally:
            record['ended_utc'] = datetime.now(timezone.utc).isoformat(); self.save()
        return record


def doctor(*, mode='demo', project_root=REPO, input_root=None, output_root=None):
    """Read-only capabilities; never creates a run or validates by computing."""
    import sys
    result = environment_identity()
    result.update(mode=mode, installed_module=str(Path(__file__).resolve()),
        python_312=sys.version_info[:2] == (3, 12), numerical_replication=AUTHORIZED_STATUS,
        network='OFFLINE', optional_reporting='XeLaTeX/fonts not needed for HTML/SVG reports')
    root = Path(project_root)
    result['lock_sha256'] = sha(root/'uv.lock')
    locked = {p['name'].lower(): p['version'] for p in tomllib.loads((root/'uv.lock').read_text())['package']}
    for name, version in result['packages'].items():
        if version == 'UNAVAILABLE' or locked.get(name.lower()) != version:
            raise PipelineError('ENVIRONMENT_DEPENDENCY_MISMATCH: '+name)
    source = root/'src/cbam_power'
    if source.is_dir():
        for p in source.rglob('*.py'):
            installed = Path(__file__).parent/p.relative_to(source)
            if not installed.is_file() or sha(p) != sha(installed):
                raise PipelineError('INSTALLED_WHEEL_STALE_REINSTALL_REQUIRED')
    result['locked_versions_match'] = True
    if mode == 'authorized':
        value = input_root or os.environ.get('CBAM_INPUT_ROOT')
        if not value or not Path(value).is_dir(): raise PipelineError('BLOCKED_CBAM_INPUT_ROOT_REQUIRED')
        disjoint(value, root)
        disjoint(output_root or os.environ.get('CBAM_OUTPUT_ROOT') or Path(value).parent/'runs', value)
    if not result['python_312']: raise PipelineError('PYTHON_312_REQUIRED')
    return result


def supplemental_references(run):
    """Explicit private, manifest-pinned plot references; never estimator inputs.

    PR2 registered these P assets as optional metadata. PR5's complete numeric
    sidecar verification requires authorized copies, without changing I/B or
    allowing any implicit historical-directory fallback.
    """
    value = os.environ.get('CBAM_REFERENCE_ROOT')
    if not value:
        raise PipelineError('BLOCKED_POLYNOMIAL_SIDECAR_REFERENCE_ROOT_REQUIRED')
    root = Path(value).absolute()
    if any(p.is_symlink() for p in [root, *root.parents]):
        raise PipelineError('REFERENCE_SYMLINK_FORBIDDEN')
    disjoint(root, run.path)
    disjoint(root, run.project_root)
    return {f'polynomial_{kind}_p{p}': root/f'polynomial_{kind}_p{p}.csv'
            for p in [1, 2, 3] for kind in ['daily_bins', 'plot_curves']}


def run_stage(run, stage, *, spec='main', audience='local'):
    """Dispatch calculations only from their explicit stage; imports are lazy."""
    import pandas as pd
    if run.mode == 'demo': return _demo_stage(run, stage, audience=audience)
    if stage == 'validate-inputs':
        return run.execute(stage, lambda out: run.catalog().validate_required())
    if stage == 'build-weather':
        def weather(out):
            from .weather import reconstruct
            r = reconstruct(run.catalog())
            r.panel.to_parquet(out/'weather.parquet', index=False)
            r.v23_panel.to_parquet(out/'v23_view.parquet', index=False)
            atomic_json(out/'diagnostics.json', r.diagnostics)
            return dict(rows=len(r.panel), scope='raw ERA5 reconstruction; frozen comparison is independent')
        return run.execute(stage, weather)
    if stage == 'build-panel':
        def panel(out):
            from .panel import build_panel
            config = tomllib.loads((run.project_root/'config/main.toml').read_text())
            d = build_panel(run.catalog(), pd.read_parquet(run.artifact('build-weather/weather.parquet')), config=config)
            d.to_parquet(out/'panel.parquet', index=False)
            return dict(rows=len(d), columns=len(d.columns))
        return run.execute(stage, panel)
    if stage == 'build-synthetic':
        def synthetic(out):
            from .synthetic import build_synthetic
            r = build_synthetic(run.catalog())
            r.all_versions.to_parquet(out/'all_versions.parquet', index=False)
            r.main.to_parquet(out/'main.parquet', index=False)
            r.weight_rows.to_csv(out/'weights.csv', index=False)
            r.energy_mix.to_csv(out/'energy_mix.csv', index=False)
            r.objective_features.to_csv(out/'objective_features.csv', index=False)
            atomic_json(out/'summary.json', r.payload)
            return dict(versions=r.all_versions.weight_version.nunique(), weights_reestimated=False)
        return run.execute(stage, synthetic)
    if stage == 'estimate-main':
        def main(out):
            from .models import estimate_main
            catalog = run.catalog()
            with catalog.open_input('sample_ME_F') as f:
                annual_times = pd.to_datetime(pd.read_csv(f).timestamp_utc, utc=True)
            with catalog.open_input('v2_weights') as f:
                weights_default = pd.read_csv(f)
            source_hashes = dict(v2_panel=catalog.assets['frozen_v2_panel']['sha256'],
                panel=sha(run.artifact('build-panel/panel.parquet')),
                weather=sha(run.artifact('build-weather/weather.parquet')),
                synthetic=sha(run.artifact('build-synthetic/all_versions.parquet')),
                weights=catalog.assets['v2_weights']['sha256'])
            result = estimate_main(pd.read_parquet(run.artifact('build-panel/panel.parquet')),
                pd.read_parquet(run.artifact('build-weather/weather.parquet')),
                pd.read_parquet(run.artifact('build-synthetic/all_versions.parquet')),
                pd.read_csv(run.artifact('build-synthetic/weights.csv'), float_precision='round_trip'),
                json.loads(run.artifact('build-synthetic/summary.json').read_text()), out,
                source_hashes=source_hashes, annual_times=annual_times, weights_default=weights_default)
            return dict(models=len(result['models']), numerical_replication=AUTHORIZED_STATUS)
        return run.execute(stage, main)
    if stage == 'estimate-polynomial':
        def polynomial(out):
            from .polynomial import estimate_polynomial
            from .validation import validate_polynomial_order
            catalog = run.catalog()
            references = supplemental_references(run)
            def check_order(order, completed, directory):
                # Explicit post-fit benchmark phase, auditable independently of
                # estimation. No validation result supplies a design or outcome.
                sys.audit('cbam.validation.start', 'polynomial_order', order)
                try:
                    checked = validate_polynomial_order(order, completed, directory, catalog,
                        supplemental_references=references,
                        evidence_dir=directory/'order_validation'/f'p{order}')
                    if checked['status'] != 'PASS':
                        raise PipelineError('POLYNOMIAL_ORDER_FROZEN_VALIDATION_FAILED')
                    return checked
                finally:
                    sys.audit('cbam.validation.end', 'polynomial_order', order)
            result = estimate_polynomial(pd.read_parquet(run.artifact('build-panel/panel.parquet')),
                json.loads(run.artifact('estimate-main/results.json').read_text()), out,
                source_hashes={'panel':sha(run.artifact('build-panel/panel.parquet')),
                    'main_results':sha(run.artifact('estimate-main/results.json'))},
                parent_verification=json.loads(run.artifact('verify-main/verification.json').read_text()),
                order_validator=check_order)
            return dict(models=len(result['models']), numerical_replication=AUTHORIZED_STATUS)
        return run.execute(stage, polynomial)
    if stage in {'verify-main', 'verify-polynomial'}:
        def verify(out):
            from .validation import verify_baseline
            selected = stage.split('-', 1)[1]
            catalog = run.catalog()
            report = verify_baseline(catalog, run.artifact('estimate-'+selected), spec=selected,
                panel_path=run.artifact('build-panel/panel.parquet'), evidence_dir=out/'comparison',
                supplemental_references=supplemental_references(run) if selected == 'polynomial' else None)
            if selected == 'main' and report['status'] == 'PASS':
                from .synthetic import validate_saved
                from .weather import compare_panel, coverage
                report['synthetic'] = validate_saved(run.artifact('build-synthetic'), catalog)
                weather = pd.read_parquet(run.artifact('build-weather/weather.parquet'))
                report['weather'] = {}
                for aid, actual in [('engine_weather_panel_hourly', weather),
                                    ('weather_v2f_benchmark', weather),
                                    ('weather_v23_benchmark', pd.read_parquet(run.artifact('build-weather/v23_view.parquet')))]:
                    with catalog.open_input(aid, purpose='verify') as f:
                        expected = pd.read_parquet(f)
                    checked, differences = compare_panel(actual, expected,
                        tolerance=catalog.baselines['validation']['weather']['strict_abs_less_than'])
                    report['weather'][aid] = checked
                    if checked['status'] != 'PASS':
                        report['status'] = 'FAIL'
                        atomic_json(out/(aid+'_differences.json'), differences)
                report['weather_coverage'] = coverage(weather, catalog)
            atomic_json(out/'verification.json', report)
            if report['status'] != 'PASS': raise PipelineError('FROZEN_VALIDATION_'+report['status'])
            return report
        return run.execute(stage, verify)
    if stage == 'report':
        def report(out):
            from .reporting.reports import render_report
            return render_report(run.artifact('estimate-main/results.json'), out,
                mode=run.mode, audience=audience, expected_sha=run.receipt['stages']['estimate-main']['outputs']['estimate-main/results.json'])
        return run.execute(stage, report, parameters={'audience': audience})
    raise PipelineError('UNKNOWN_STAGE')


def reproduce(run):
    if run.mode != 'authorized': raise PipelineError('AUTHORIZED_MODE_REQUIRED')
    for stage in ['validate-inputs', 'build-weather', 'build-panel', 'build-synthetic',
                  'estimate-main', 'verify-main', 'estimate-polynomial', 'verify-polynomial', 'report']:
        run_stage(run, stage)
    return run.receipt


def demo(run):
    if run.mode != 'demo': raise PipelineError('DEMO_MODE_REQUIRED')
    for stage in ['validate-inputs', 'build-weather', 'build-panel', 'build-synthetic',
                  'estimate-main', 'verify-main', 'report']:
        run_stage(run, stage, audience='public')
    return run.receipt


def _demo_stage(run, stage, *, audience):
    from . import _demo
    return _demo.run_stage(run, stage, audience=audience)
