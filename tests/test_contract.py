"""PR1 integrity/contract tests only; never import or run a legacy estimator.

Python >= 3.11, standard library only. Run from the candidate root:
  CBAM_INPUT_ROOT=<authorized input root> python -B -m unittest discover \
      -s tests -p 'test_contract.py'

Required immutable files live under CBAM_INPUT_ROOT/baselines, outside this
tree. CBAM_OUTPUT_ROOT, if set, must also be disjoint; otherwise it resolves to
CBAM_INPUT_ROOT's sibling 'runs'. Missing inputs are errors, never skips.
No expected-value update command exists. Digests below are independent pinned
review anchors; tests never calculate new expected values from an estimator.
A green result proves PR1 only, not full scientific replication or release rights.
"""
import ast
from array import array
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import tempfile
import tomllib
import unittest
import zipfile

REPO = Path(__file__).resolve().parents[1]
CONTRACT_DIGESTS = {'config/main.toml': '7bb0fdada81087c3dc8b07f466f365d0eeb42269e073b46076379cf8afce3378', 'config/polynomial.toml': '2fb4ee49b144ba9fa3ba28d9f6bdb93e081cb8439caaf83aad13dba85a720a56', 'manifests/baselines.toml': 'f28a5b6ed9007ba8af0fd3e5ae5dd7168ebaf53f56f1af150833a59ce0c4157b'}
CORE_SHA256 = {
    'v3_panel': 'a0eea50d8ac966fa9833eadc91d33e2f8f3f07c09bb1c025a5d79ed57bddc305',
    'v3_results': '997264dbc1748ce42e06fdd3b3693e7ee977899890c6752f30c62ed3db076c3c',
    'polynomial_results': '3857fbffc63e817ae9aea88e23cf9e6698ba9076ed9741ec18f3bc854b7437a9',
}
CANDIDATE_FILES = {'.github/workflows/ci.yml', '.gitignore', '.python-version', 'CITATION.cff', 'LICENSE', 'Makefile', 'README.md', 'config/main.toml', 'config/polynomial.toml', 'docs/data_access.md', 'docs/figures/corridors.svg', 'docs/figures/era5_aggregation.svg', 'docs/figures/rdit_demo.png', 'docs/figures/synthetic_demo.png', 'docs/figures/workflow.svg', 'docs/index.html', 'docs/methodology.md', 'manifests/baselines.toml', 'manifests/inputs.toml', 'manifests/release_rights.toml', 'notebooks/01_data_and_weather.ipynb', 'notebooks/02_rdit_and_hac.ipynb', 'notebooks/03_synthetic_control.ipynb', 'notebooks/04_robustness_and_validation.ipynb', 'pyproject.toml', 'src/cbam_power/__init__.py', 'src/cbam_power/_assembly.py', 'src/cbam_power/_demo.py', 'src/cbam_power/_empirical.py', 'src/cbam_power/_engine_context.py', 'src/cbam_power/_weather_models.py', 'src/cbam_power/_weather_synthetic.py', 'src/cbam_power/cli.py', 'src/cbam_power/design.py', 'src/cbam_power/estimators.py', 'src/cbam_power/inputs.py', 'src/cbam_power/market.py', 'src/cbam_power/methods.py', 'src/cbam_power/models.py', 'src/cbam_power/panel.py', 'src/cbam_power/pipeline.py', 'src/cbam_power/polynomial.py', 'src/cbam_power/portfolio.py', 'src/cbam_power/release.py', 'src/cbam_power/reporting/__init__.py', 'src/cbam_power/reporting/figures.py', 'src/cbam_power/reporting/reports.py', 'src/cbam_power/reporting/tables.py', 'src/cbam_power/site.py', 'src/cbam_power/synthetic.py', 'src/cbam_power/validation.py', 'src/cbam_power/weather.py', 'tests/test_contract.py', 'tests/test_estimators.py', 'tests/test_frozen.py', 'tests/test_inputs.py', 'tests/test_market_panel.py', 'tests/test_methods.py', 'tests/test_pipeline.py', 'tests/test_polynomial.py', 'tests/test_portfolio.py', 'tests/test_release.py', 'tests/test_reporting.py', 'tests/test_site.py', 'tests/test_synthetic.py', 'tests/test_validation.py', 'tests/test_weather.py', 'uv.lock'}


class ContractError(AssertionError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_sha(path, expected):
    require(Path(path).is_file(), 'BLOCKED_REQUIRED_BASELINE_MISSING')
    require(file_sha(path) == expected, 'FROZEN_SHA256_MISMATCH')


def validate_configuration(main, polynomial):
    for config in (main['time'], polynomial['design']):
        require(config['policy_cutoff_local'] == '2026-01-01T00:00:00+01:00',
                'POLICY_CUTOFF_CHANGED')
        require(config['policy_cutoff_utc'] == '2025-12-31T23:00:00Z',
                'POLICY_CUTOFF_CHANGED')
    for rel, obj in [('config/main.toml', main), ('config/polynomial.toml', polynomial)]:
        require(digest(obj) == CONTRACT_DIGESTS[rel], 'SCIENTIFIC_CONFIGURATION_CHANGED: ' + rel)


def input_root(environ):
    value = environ.get('CBAM_INPUT_ROOT')
    require(bool(value), 'BLOCKED_CBAM_INPUT_ROOT_REQUIRED; no implicit archive fallback')
    root = Path(value).expanduser().resolve()
    require(root.is_dir(), 'BLOCKED_INPUT_ROOT_MISSING')
    return root


def disjoint(left, right):
    left, right = Path(left).resolve(), Path(right).resolve()
    require(not (left.is_relative_to(right) or right.is_relative_to(left)),
            'INPUT_BASELINE_OUTPUT_REPOSITORY_OVERLAP')


def confined(root, relative):
    part = PurePosixPath(relative)
    require(not part.is_absolute() and '..' not in part.parts and '\\' not in relative,
            'INVALID_PRIVATE_RELATIVE_PATH')
    require(bool(part.parts), 'EMPTY_PRIVATE_PATH')
    path = Path(root)
    for name in part.parts:
        path /= name
        require(not path.is_symlink(), 'BASELINE_SYMLINK_NOT_ALLOWED')
    require(path.resolve().is_relative_to(Path(root).resolve()), 'BASELINE_PATH_ESCAPE')
    return path


def index_unique(rows, keys):
    indexed = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        require(all(v not in (None, '') for v in key), 'EMPTY_PRIMARY_KEY')
        require(key not in indexed, 'DUPLICATE_PRIMARY_KEY')
        indexed[key] = row
    return indexed


def validate_model_set(models, count, expected_digest):
    ids = [m['model_id'] for m in models]
    require(len(ids) == count and len(set(ids)) == count, 'MODEL_SET_COUNT_OR_DUPLICATE')
    require(digest(sorted(ids)) == expected_digest, 'MODEL_SET_IDENTITY_MISMATCH')


def validate_weights(rows, versions, expected_donors):
    index_unique(rows, ['target', 'version', 'donor'])
    me = [r for r in rows if r['target'] == 'ME_IT']
    require(len(me) == 32, 'ME_WEIGHT_ROW_COUNT')
    require(sorted({r['version'] for r in me}) == sorted(versions), 'ME_WEIGHT_VERSION_MISMATCH')
    grouped = defaultdict(list)
    for row in me:
        grouped[row['version']].append(row)
    for version, group in grouped.items():
        require({r['donor'] for r in group} == set(expected_donors), 'ME_DONOR_POOL_CHANGED')
        values = [float(r['weight']) for r in group]
        require(all(math.isfinite(v) and v >= 0 for v in values), 'INVALID_FROZEN_WEIGHT')
        require(math.isclose(sum(values), 1, rel_tol=0, abs_tol=1e-13), 'WEIGHT_SUM_CHANGED')
        year = 2024 if version.endswith('train2024') else 2025
        require(all(float(r['trained_max_local_year']) == year for r in group), 'WEIGHT_TRAINING_CHANGED')
    return grouped


def covariance_reference(row, archives):
    key = (row['file_path'], row['covariance_key'])
    require(key[0] in archives and key[1] in archives[key[0]], 'COVARIANCE_REFERENCE_MISSING')
    columns = json.loads(row['columns'])
    require(isinstance(columns, list) and all(isinstance(c, str) and c for c in columns),
            'INVALID_COVARIANCE_COLUMNS')
    require(len(columns) == len(set(columns)), 'DUPLICATE_COVARIANCE_COLUMN')
    require(archives[key[0]][key[1]] == (len(columns), len(columns)),
            'COVARIANCE_COLUMN_DIMENSION_MISMATCH')
    return columns


def inspect_npz(path):
    """Decode numeric NPY headers and payloads without NumPy, pickle or extraction."""
    shapes = {}
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        require(len(members) == len(set(members)), 'DUPLICATE_NPZ_MEMBER')
        for name in members:
            require(name.endswith('.npy') and '/' not in name, 'INVALID_NPZ_MEMBER')
            with archive.open(name) as handle:
                require(handle.read(6) == b'\x93NUMPY', 'INVALID_NPY_MAGIC')
                version = tuple(handle.read(2))
                require(version in [(1, 0), (2, 0), (3, 0)], 'UNSUPPORTED_NPY_VERSION')
                size = 2 if version == (1, 0) else 4
                length = int.from_bytes(handle.read(size), 'little')
                require(0 < length < 65536, 'UNEXPECTED_NPY_HEADER_SIZE')
                header = ast.literal_eval(handle.read(length).decode('utf8' if version == (3, 0) else 'latin1'))
                shape = header['shape']
                require(isinstance(shape, tuple) and len(shape) == 2 and
                        shape[0] == shape[1] and shape[0] > 0, 'INVALID_COVARIANCE_SHAPE')
                dtype = header['descr']
                require(dtype in ('<f8', '>f8', '=f8'), 'UNEXPECTED_COVARIANCE_DTYPE')
                payload = handle.read()
                require(len(payload) == shape[0] * shape[1] * 8, 'INVALID_NPY_PAYLOAD_LENGTH')
                vals = array('d')
                require(vals.itemsize == 8, 'UNSUPPORTED_NATIVE_DOUBLE_SIZE')
                vals.frombytes(payload)
                if dtype != '=f8' and ((dtype[0] == '<') != (sys.byteorder == 'little')):
                    vals.byteswap()
                require(all(math.isfinite(v) for v in vals), 'NONFINITE_COVARIANCE')
                shapes[name[:-4]] = shape
    return shapes


def timestamp_integers(timestamps, unit):
    require(timestamps is not None and len(timestamps) > 0, 'ACTUAL_TIMESTAMPS_REQUIRED')
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    values = []
    for text in timestamps:
        # Reject precision unsupported by datetime rather than silently truncating.
        require(not re.search(r'\.\d{7,}', text), 'SUBMICROSECOND_TIMESTAMP_REQUIRES_NS_DECODER')
        dt = datetime.fromisoformat(text.replace('Z', '+00:00'))
        require(dt.utcoffset() is not None, 'NAIVE_TIMESTAMP_REJECTED')
        delta = dt.astimezone(timezone.utc) - epoch
        us = (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds
        values.append(us if unit == 'us' else us * 1000)
    require(values == sorted(set(values)), 'TIMESTAMPS_NOT_SORTED_UNIQUE')
    return values


def timestamp_digest(values):
    return hashlib.sha256(b''.join(struct.pack('<q', v) for v in values)).hexdigest()


def prove_encoding_bridge(old_timestamps, new_timestamps, legacy_us_hash, canonical_ns_hash):
    old_ns = timestamp_integers(old_timestamps, 'ns')
    new_ns = timestamp_integers(new_timestamps, 'ns')
    require(old_ns == new_ns, 'SAMPLE_TIMESTAMPS_CHANGED')
    require(timestamp_digest(timestamp_integers(old_timestamps, 'us')) == legacy_us_hash,
            'LEGACY_SAMPLE_HASH_MISMATCH')
    require(timestamp_digest(new_ns) == canonical_ns_hash, 'CANONICAL_SAMPLE_HASH_MISMATCH')


def ast_literal(path, variable):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'dict':
                return {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}
            return ast.literal_eval(node.value)
    raise ContractError('SOURCE_LITERAL_NOT_FOUND: ' + variable)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = tomllib.loads((REPO / 'config/main.toml').read_text())
        cls.poly = tomllib.loads((REPO / 'config/polynomial.toml').read_text())
        cls.manifest = tomllib.loads((REPO / 'manifests/baselines.toml').read_text())
        cls.root = input_root(os.environ)
        cls.output = Path(os.environ.get('CBAM_OUTPUT_ROOT', cls.root.parent / 'runs')).resolve()
        cls.files = {f['id']: f for f in cls.manifest['files']}
        cls.csv_cache = {}
        cls.json_cache = {}
        cls.npz_cache = {}

    @classmethod
    def path(cls, id):
        require(id in cls.files, 'BASELINE_ID_NOT_REGISTERED: ' + id)
        path = confined(cls.root, cls.files[id]['private_relative_path'])
        require(path.is_file(), 'BLOCKED_REQUIRED_BASELINE_MISSING: ' + id)
        return path

    @classmethod
    def rows(cls, id):
        if id not in cls.csv_cache:
            with cls.path(id).open(newline='') as handle:
                cls.csv_cache[id] = list(csv.DictReader(handle))
        return cls.csv_cache[id]

    @classmethod
    def obj(cls, id):
        if id not in cls.json_cache:
            cls.json_cache[id] = json.loads(cls.path(id).read_text())
        return cls.json_cache[id]

    @classmethod
    def npz(cls, id):
        if id not in cls.npz_cache:
            cls.npz_cache[id] = inspect_npz(cls.path(id))
        return cls.npz_cache[id]

    def test_configuration_is_frozen(self):
        validate_configuration(self.main, self.poly)
        self.assertEqual(digest(self.manifest), CONTRACT_DIGESTS['manifests/baselines.toml'])

    def test_three_core_sha256(self):
        for id, expected in CORE_SHA256.items():
            with self.subTest(id=id):
                self.assertEqual(self.files[id]['sha256'], expected)
                verify_sha(self.path(id), expected)
        poly = self.obj('polynomial_results')
        self.assertEqual(poly['input_sha256'], CORE_SHA256['v3_panel'])
        self.assertEqual(poly['baseline_results_sha256'], CORE_SHA256['v3_results'])
        self.assertEqual(self.obj('v3_results')['panel']['sha256'], CORE_SHA256['v3_panel'])

    def test_all_registered_files_integrity_schema_and_read_only(self):
        entries = self.manifest['files']
        self.assertEqual(len(self.files), len(entries))
        index_unique(entries, ['private_relative_path'])
        index_unique(entries, ['legacy_source'])
        for f in entries:
            with self.subTest(id=f['id']):
                path = self.path(f['id'])
                self.assertTrue(f['required'] and f['read_only'])
                self.assertEqual(f['release_policy'], 'PRIVATE_ONLY')
                self.assertEqual(path.stat().st_size, f['bytes'])
                self.assertEqual(path.stat().st_mode & 0o222, 0, 'Frozen file is writable')
                self.assertEqual(path.stat().st_nlink, 1, 'Do not hardlink archive baselines')
                verify_sha(path, f['sha256'])
                if f['format'] == 'csv':
                    rows = self.rows(f['id'])
                    self.assertEqual(len(rows), f['rows'])
                    with path.open(newline='') as handle:
                        self.assertEqual(csv.DictReader(handle).fieldnames or [], f['columns'])
                    self.assertTrue(all(None not in row for row in rows), 'CSV column overflow')
                    if f.get('primary_key'):
                        index_unique(rows, f['primary_key'])
                elif f['format'] == 'json':
                    obj = self.obj(f['id'])
                    self.assertEqual(type(obj).__name__, f['json_root_type'])
                    if isinstance(obj, dict):
                        self.assertEqual(list(obj), f['required_top_level_keys'])
                elif f['format'] == 'npz':
                    arrays = self.npz(f['id'])
                    self.assertEqual(len(arrays), f['array_count'])
                    self.assertEqual(digest(sorted(arrays)), f['array_keys_sha256'])
                elif f['format'] == 'parquet':
                    with path.open('rb') as handle:
                        self.assertEqual(handle.read(4), b'PAR1')
                        handle.seek(-4, 2)
                        self.assertEqual(handle.read(), b'PAR1')
                    self.assertEqual(len(f['columns']), len(f['dtypes']))
        baseline = self.root / self.manifest['baseline_subdirectory']
        actual = {str(p.relative_to(self.root)) for p in baseline.rglob('*') if p.is_file()}
        self.assertEqual(actual, {x['private_relative_path'] for x in entries})
        for directory in [baseline, *(p for p in baseline.rglob('*') if p.is_dir())]:
            metadata = directory.stat()
            # Some file-provider volumes reset directory modes. An OS immutable
            # flag protects entries more strongly; accept that explicit seal,
            # never a writable unsealed directory. File write bits must remain 0.
            immutable = getattr(metadata, 'st_flags', 0) & getattr(stat, 'UF_IMMUTABLE', 0)
            self.assertTrue(metadata.st_mode & 0o222 == 0 or immutable,
                            'Frozen directory is writable and not immutable')

    def test_v3_full_model_ids(self):
        validate_model_set(self.obj('v3_results')['models'], 1892,
                           self.main['inventory']['expected_model_ids_sha256'])
        models = self.obj('v3_results')['models']
        self.assertEqual(Counter(m['model_family'] for m in models), self.main['inventory']['families'])
        self.assertEqual(Counter(m['status'] for m in models), self.main['inventory']['estimated_status_counts'])
        self.assertEqual(self.obj('v3_results')['main'], self.main['inventory']['formal_main_ids'])
        self.assertFalse(any(m['series'] in ['FR_actual', 'CH_synthetic', 'G_CH', 'D'] for m in models))

    def test_catalog_status_and_not_run_reasons(self):
        rows = self.rows('v3_catalog')
        catalog = index_unique(rows, ['model_id'])
        results = self.obj('v3_results')
        self.assertEqual(len(rows), 1928)
        expected = {m['model_id'] for m in results['models']}
        expected |= {'V3_SYNTHETIC_VALIDATION_2025'}
        expected |= {f'V3_NOT_RUN_{i:03d}' for i in range(35)}
        self.assertEqual({k[0] for k in catalog}, expected)
        for m in results['models']:
            row = catalog[(m['model_id'],)]
            self.assertEqual(row['status'], m['status'])
            for field in ['sample_hash', 'sample_start', 'sample_end', 'coefficient_name']:
                self.assertEqual(row[field], m[field])
            for field in ['N', 'pre_N', 'post_N', 'HAC_lag']:
                self.assertEqual(float(row[field]), m[field])
        blocks = results['blocked_models']
        self.assertEqual(len(blocks), 35)
        self.assertEqual(Counter(b['status'] for b in blocks), self.main['inventory']['not_estimated_status_counts'])
        for i, (block, contract) in enumerate(zip(blocks, self.main['not_estimated'])):
            row = catalog[(f'V3_NOT_RUN_{i:03d}',)]
            for field in ['status', 'reason', 'implementation_status']:
                self.assertEqual(row[field], block.get(field) or '')
            for key, val in contract.items():
                if key not in ('catalog_model_id', 'frozen_json_pointer'):
                    self.assertEqual(val, block[key])
            self.assertTrue(all(row[k] == '' for k in ['estimate', 'SE', 'N']))
        self.assertEqual(catalog[('V3_SYNTHETIC_VALIDATION_2025',)]['status'], 'DESCRIPTIVE_METRICS')

    def test_all_coefficients_unique_complete_and_traceable(self):
        rows = self.rows('v3_coefficients')
        actual = index_unique(rows, ['model_id', 'term'])
        self.assertEqual(len(rows), 77574)
        expected_keys = set()
        for m in self.obj('v3_results')['models']:
            for c in m['coefficients']:
                key = (m['model_id'], c['term'])
                self.assertNotIn(key, expected_keys)
                expected_keys.add(key)
                self.assertIn(key, actual)
                row = actual[key]
                self.assertEqual(int(row['N']), m['N'])
                self.assertEqual(int(row['HAC_lag']), m['HAC_lag'])
                for field in ['coefficient', 'SE', 'CI_low', 'CI_high', 'p_value']:
                    self.assertTrue(math.isclose(float(row[field]), c[field], abs_tol=1e-10, rel_tol=0),
                                    'Coefficient table/JSON mismatch: ' + m['model_id'])
            primary = [c for c in m['coefficients'] if c['term'] == m['coefficient_name']]
            self.assertEqual(len(primary), 1)
            for field in ['estimate', 'SE', 'CI_low', 'CI_high', 'p_value']:
                self.assertTrue(math.isclose(m[field], primary[0]['coefficient' if field == 'estimate' else field],
                                            abs_tol=1e-10, rel_tol=0))
        self.assertEqual(set(actual), expected_keys)

    def test_covariance_references_and_joint_blocks(self):
        files = ['engine_model_covariances', 'engine_weather_covariances',
                 'engine_weather_synthetic_covariances', 'v3_basic_covariances']
        arrays = {self.files[id]['legacy_source'].removeprefix('v3_formal_cbam/'): self.npz(id) for id in files}
        rows = self.rows('v3_covariance_index')
        index = index_unique(rows, ['file_path', 'covariance_key'])
        self.assertEqual(len(rows), 1926)
        self.assertEqual(sum(r['covariance_scope'] == 'joint' for r in rows), 32)
        self.assertEqual(sum(not r['model_id'] for r in rows), 28)
        self.assertLess(len({r['model_id'] for r in rows}), len(rows))
        self.assertEqual(set(index), {(file, key) for file, content in arrays.items() for key in content})
        for row in rows:
            columns = covariance_reference(row, arrays)
            if row['covariance_scope'] == 'joint':
                self.assertTrue(all(':' in c for c in columns))
        for m in self.obj('v3_results')['models']:
            ref = m['covariance_reference']
            key = (ref['file_path'], ref['covariance_key'])
            self.assertIn(key, index)
            row = index[key]
            self.assertEqual(row['model_id'], m['model_id'])
            self.assertEqual(int(row['HAC_lag']), m['HAC_lag'])
            self.assertEqual(row['columns'], ref['columns'])
            self.assertEqual(row['covariance_scope'], ref['covariance_scope'])
            self.assertTrue(set(json.loads(row['columns'])).issubset({c['term'] for c in m['coefficients']}))
        for prefix in ['engine', 'v2f']:
            for index_id in ['covariance_index', 'weather_covariance_index', 'weather_synthetic_covariance_index']:
                id = prefix + '_' + index_id
                archive_id = self.files[id]['covariance_file_id']
                indexed = index_unique(self.rows(id), ['key'])
                shapes = self.npz(archive_id)
                self.assertEqual({k[0] for k in indexed}, set(shapes))
                for row in self.rows(id):
                    normalized = {'file_path': archive_id, 'covariance_key': row['key'], 'columns': row['columns']}
                    covariance_reference(normalized, {archive_id: shapes})
                    if prefix == 'engine':
                        file = self.files[archive_id]['legacy_source'].removeprefix('v3_formal_cbam/')
                        self.assertEqual(index[(file, row['key'])]['columns'], row['columns'])

    def test_samples_parameters_and_references_are_locatable(self):
        designs = {}
        for id in ['engine_model_catalog', 'engine_weather_model_design', 'engine_weather_synthetic_design']:
            for row in self.rows(id):
                self.assertNotIn(row['model_id'], designs)
                designs[row['model_id']] = row
                self.assertRegex(row['sample_hash'], r'^[0-9a-f]{64}$')
                self.assertRegex(row['sample_hash_utc_ns'], r'^[0-9a-f]{64}$')
                self.assertIn('datetime64[', row['sample_hash_integer_unit'])
                refs = json.loads(row.get('references', row.get('FE_reference')))
                self.assertIn('hour', refs)
                self.assertIn('weekday', refs)
                self.assertTrue(json.loads(row['columns']))
                self.assertGreater(sum(json.loads(row['year_N']).values()), 0)
        for m in self.obj('v3_results')['models']:
            self.assertRegex(m['sample_hash'], r'^[0-9a-f]{64}$')
            self.assertTrue(m['sample_start'] and m['sample_end'])
            self.assertEqual(m['N'], m['pre_N'] + m['post_N'])
            if 'source_model_id' in m:
                row = designs[m['source_model_id']]
                self.assertEqual(m['sample_hash'], row['sample_hash_utc_ns'])
                self.assertEqual(m['N'], int(row.get('n', row.get('N'))))
                self.assertEqual(json.loads(row['columns']), json.loads(m['covariance_reference']['columns']))
            else:
                self.assertEqual(m['model_family'], 'BASIC_RDiT')
                self.assertEqual([c['term'] for c in m['coefficients']], ['const', 'post', 'r', 'post_r'])
                self.assertEqual(m['N'], 4320)
        summary = index_unique(self.rows('v3_sample_summary'), ['model_id'])
        self.assertEqual(set(summary), set(index_unique(self.rows('v3_catalog'), ['model_id'])))
        panel = self.files['v3_panel']
        dictionary = self.rows('v3_dictionary')
        self.assertEqual(panel['rows'], 118795)
        self.assertEqual(panel['columns'], [d['variable'] for d in dictionary])
        self.assertIn('timestamp[us, tz=UTC]', panel['dtypes'])

    def test_us_ns_bridge_uses_actual_frozen_timestamp_rows(self):
        expected = [r['timestamp_utc'] for r in self.rows('sample_ME_R1')]
        self.assertEqual(len(expected), 4320)
        groups = defaultdict(list)
        for row in self.rows('engine_weather_membership'):
            if row['included'] == 'True':
                groups[(row['series'], row['model'], row['support'], row['variant'])].append(row['timestamp_utc'])
        for row in self.rows('engine_weather_model_design'):
            key = (row['series'], row['model'], row['support'], row['variant'])
            prove_encoding_bridge(groups[key], expected, row['sample_hash'], row['sample_hash_utc_ns'])
        # This proves these actual local-weather memberships only. Other model
        # timestamp/missing-mask reconstructions remain a PR5 obligation.

    def test_synthetic_all_sixteen_versions_and_frozen_values(self):
        rows = self.rows('v2_weights')
        groups = validate_weights(rows, self.main['synthetic']['versions'], self.main['synthetic']['donors'])
        self.assertEqual(len(rows), 92)
        self.assertEqual(sum(r['weight'] == '' for r in rows), 60)
        frozen = self.obj('engine_synthetic_results')['weights']
        self.assertEqual(set(frozen), set(groups))
        for version, group in groups.items():
            for row in group:
                self.assertTrue(math.isclose(float(row['weight']), frozen[version][row['donor']],
                                            rel_tol=0, abs_tol=1e-15))
                if version in self.main['synthetic']['main_versions']:
                    self.assertTrue(math.isclose(float(row['weight']), self.obj('v3_results')['weights'][version][row['donor']],
                                                rel_tol=0, abs_tol=1e-15))
        copy = index_unique(self.rows('engine_synthetic_frozen_weights'), ['target', 'version', 'donor'])
        self.assertEqual(set(copy), {(r['target'], r['version'], r['donor']) for group in groups.values() for r in group})
        for group in groups.values():
            for row in group:
                self.assertEqual(row['weight'], copy[(row['target'], row['version'], row['donor'])]['weight'])
        self.assertEqual(self.files['engine_me_synthetic_all_frozen_versions']['rows'], 16 * 23759)

    def test_v2f_benchmarks_registered_with_full_tables(self):
        self.assertEqual(len(self.rows('v2f_model_catalog')), 899)
        for name in ['all_coefficients', 'weather_abc_results', 'weather_coefficients',
                     'weather_synthetic_abc_results', 'weather_synthetic_coefficients']:
            old = self.rows('v2f_' + name)
            new = self.rows('engine_' + name)
            key = ['model_id', 'hac_lag', 'estimand' if 'abc_results' in name else 'term']
            self.assertEqual(set(index_unique(old, key)), set(index_unique(new, key)))
        self.assertTrue(self.obj('v2f_results')['permanently_excluded'])
        # Registration and key closure only, not rerunning historical QA.

    def test_polynomial_complete_grid_parameters_samples_and_statuses(self):
        result = self.obj('polynomial_results')
        models = result['models']
        validate_model_set(models, 90, self.poly['inventory']['expected_model_ids_sha256'])
        expected = set()
        for order in [1, 2, 3]:
            for code in self.poly['design']['codes']:
                expected.add(f'V3_2_BASIC_P{order}_{code}')
                expected.add(f'V3_2_PLACEBO2025_FE_P{order}_{code}')
                for h in [30, 60, 90, 180]:
                    expected.add(f'V3_2_FE_P{order}_{code}' + (f'_BW{h}' if h != 90 else ''))
        self.assertEqual({m['model_id'] for m in models}, expected)
        rows = index_unique(self.rows('polynomial_catalog'), ['model_id'])
        self.assertEqual({k[0] for k in rows}, expected)
        self.assertEqual(Counter(m['status'] for m in models), self.poly['inventory']['estimated_status_counts'])
        groups = defaultdict(list)
        coefficients = index_unique(self.rows('polynomial_coefficients'), ['model_id', 'term'])
        all_terms = set()
        for m in models:
            p, h = m['polynomial_order'], m['bandwidth_days']
            terms = ['const', 'post'] + [term for k in range(1, p + 1) for term in [f'z{k}', f'post_z{k}']]
            if m['family'] == 'FE':
                terms += [f'hour_{i}' for i in range(1, 24)] + [f'weekday_{i}' for i in range(1, 7)]
            self.assertEqual(m['columns'], terms)
            self.assertEqual(len(m['beta']), len(terms))
            self.assertEqual(len(m['covariance']), len(terms))
            self.assertTrue(all(len(row) == len(terms) and all(math.isfinite(v) for v in row) for row in m['covariance']))
            self.assertEqual(m['N'], 48 * h)
            self.assertEqual(m['N_pre'], 24 * h)
            self.assertEqual(m['N_post'], 24 * h)
            self.assertEqual(m['HAC'], 168)
            self.assertEqual(m['kernel'], 'triangular')
            self.assertFalse(m['weather_controls'])
            self.assertEqual(m['cutoff'], f"{m['cutoff_year']}-01-01T00:00:00+01:00")
            self.assertEqual(m['rank'], len(terms))
            row = rows[(m['model_id'],)]
            for field in ['status', 'functional_form_status', 'sample_hash', 'weight_hash']:
                self.assertEqual(row[field], m[field])
            self.assertTrue(set(m['shape_diagnostics']['flags']).issubset(self.poly['diagnostics']['shape_flags']))
            for term, beta in zip(terms, m['beta']):
                key = (m['model_id'], term)
                all_terms.add(key)
                self.assertIn(key, coefficients)
                self.assertTrue(math.isclose(float(coefficients[key]['coefficient']), beta, rel_tol=0, abs_tol=1e-10))
            groups[(m['corridor'], m['family'], h, m['cutoff_year'])].append(m)
        self.assertEqual(set(coefficients), all_terms)
        for group in groups.values():
            self.assertEqual(len(group), 3)
            self.assertEqual(len({m['sample_hash'] for m in group}), 1)
            self.assertEqual(len({m['weight_hash'] for m in group}), 1)
        self.assertEqual(result['not_estimated'], self.poly['not_estimated'])
        self.assertEqual(len(result['not_estimated']), 15)
        status = index_unique(self.rows('polynomial_not_estimated'), ['model_id'])
        self.assertEqual(set(status), {(m['model_id'],) for m in result['not_estimated']})
        for m in result['not_estimated']:
            self.assertEqual(status[(m['model_id'],)]['status'], m['status'])
            self.assertEqual(status[(m['model_id'],)]['detail'], m['detail'])

    def test_source_literals_and_tolerances(self):
        sources = {f['legacy_source']: self.path(f['id']) for f in self.files.values()}
        cfg = json.loads(sources[self.main['evidence']['design']].read_text())
        for target, source in [('main_bandwidth_days', 'main_bandwidth'), ('bandwidth_days', 'bandwidth_days'),
                               ('donut_days', 'donut_days'), ('fixed_effects', 'rdit_fe')]:
            self.assertEqual(self.main['rdit'][target], cfg[source])
        self.assertEqual(self.main['time']['policy_cutoff_local'], cfg['cutoff'])
        self.assertEqual(self.main['inference']['hac_elapsed_hours'], cfg['hac_elapsed_hours'])
        self.assertEqual(self.main['synthetic']['donors'], cfg['formal_donors'])
        for variable in ['F1_CONTRASTS', 'F2_CONTRASTS', 'F2_TERMS']:
            key = {'F1_CONTRASTS': 'F1_contrasts', 'F2_CONTRASTS': 'F2_contrasts', 'F2_TERMS': 'F2_terms'}[variable]
            self.assertEqual(self.main['same_season'][key], ast_literal(sources[self.main['evidence']['timing']], variable))
        self.assertEqual(self.poly['thresholds'], ast_literal(sources[self.poly['evidence']['configuration']], 'THRESHOLDS'))
        tolerance = self.manifest['validation']
        self.assertEqual((tolerance['full_frozen']['atol'], tolerance['full_frozen']['rtol']), (1e-6, 1e-8))
        self.assertEqual((tolerance['core_panel']['atol'], tolerance['core_panel']['rtol']), (1e-8, 1e-11))
        self.assertEqual(tolerance['primary_internal']['atol'], 1e-10)
        self.assertEqual(tolerance['weather']['strict_abs_less_than'], 1e-9)
        self.assertEqual(tolerance['frozen_weights_csv_json']['atol'], 1e-15)
        self.assertEqual((tolerance['polynomial_p1_parent_tau_se']['atol'], tolerance['polynomial_p1_parent_tau_se']['rtol']), (1e-8, 1e-10))
        self.assertEqual(tolerance['polynomial_p1_all_coefficients']['strict_abs_less_than'], 1e-7)
        self.assertTrue(tolerance['sample_identity']['actual_timestamp_proof_required'])
        self.assertFalse(tolerance['file_integrity']['update_expected_from_new_run'])

    def test_private_inputs_outputs_and_candidate_tree_are_isolated(self):
        disjoint(REPO, self.root)
        disjoint(REPO, self.output)
        disjoint(self.root, self.output)
        disjoint(self.root / 'baselines', self.output)
        actual = set()
        for path in REPO.rglob('*'):
            if '.git' in path.relative_to(REPO).parts or path.relative_to(REPO).parts[0] == '.venv':
                continue
            self.assertFalse(path.is_symlink(), 'No symlinks in candidate tree')
            if path.is_file():
                actual.add(str(path.relative_to(REPO)))
                self.assertEqual(path.stat().st_nlink, 1)
                if path.suffix == '.png':
                    # PR7 adds two explicitly registered artificial illustrations.
                    # Validate their exact approved bytes, never decode arbitrary
                    # binary payloads as text or exempt them based on extension.
                    from cbam_power.release import RightsRegistry, scan_content
                    rights = RightsRegistry(tomllib.loads((REPO / 'manifests/release_rights.toml').read_text()))
                    relative = path.relative_to(REPO).as_posix()
                    node_id = rights.data['portfolio_artifacts'][relative]
                    self.assertEqual(rights.node(node_id)['origin'], 'ARTIFICIAL')
                    rights.authorize(node_id)
                    scan_content(path.read_bytes(), rights, node_id)
                    continue
                text = path.read_text()
                self.assertNotRegex(text, r'/(?:Users|home)/[^/\s]+/')
                self.assertNotRegex(text, r'[A-Za-z]:\\Users\\')
                self.assertNotIn(str(self.root), text)
        self.assertEqual(actual, CANDIDATE_FILES, 'Unreviewed file in public candidate tree')
        rules = [line for line in (REPO / '.gitignore').read_text().splitlines() if line and not line.startswith('#')]
        allowed = set(CANDIDATE_FILES)
        for candidate in CANDIDATE_FILES:
            for parent in Path(candidate).parents:
                if str(parent) != '.':
                    allowed.add(str(parent) + '/')
        self.assertEqual(rules, ['*'] + ['!' + name for name in sorted(allowed)])


    def test_negative_wrong_sha_is_rejected(self):
        with self.assertRaisesRegex(ContractError, 'SHA256_MISMATCH'):
            verify_sha(self.path('v3_results'), '0' * 64)

    def test_negative_missing_model_is_rejected(self):
        with self.assertRaisesRegex(ContractError, 'MODEL_SET'):
            validate_model_set(self.obj('v3_results')['models'][:-1], 1892,
                               self.main['inventory']['expected_model_ids_sha256'])

    def test_negative_renamed_model_same_count_is_rejected(self):
        models = [dict(model_id=m['model_id']) for m in self.obj('v3_results')['models']]
        models[0]['model_id'] = 'UNAUTHORIZED_MODEL'
        with self.assertRaisesRegex(ContractError, 'MODEL_SET_IDENTITY'):
            validate_model_set(models, 1892, self.main['inventory']['expected_model_ids_sha256'])

    def test_negative_wrong_synthetic_version_is_rejected(self):
        rows = deepcopy(self.rows('v2_weights'))
        next(r for r in rows if r['target'] == 'ME_IT')['version'] = 'GY999_final'
        with self.assertRaisesRegex(ContractError, 'WEIGHT_VERSION'):
            validate_weights(rows, self.main['synthetic']['versions'], self.main['synthetic']['donors'])

    def test_negative_missing_covariance_reference_is_rejected(self):
        with self.assertRaisesRegex(ContractError, 'COVARIANCE_REFERENCE_MISSING'):
            covariance_reference(self.rows('v3_covariance_index')[0], {})

    def test_negative_wrong_covariance_column_index_is_rejected(self):
        row = self.rows('v3_covariance_index')[0]
        with self.assertRaisesRegex(ContractError, 'COLUMN_DIMENSION_MISMATCH'):
            covariance_reference(row, {row['file_path']: {row['covariance_key']: (1, 1)}})

    def test_negative_wrong_cutoff_is_rejected(self):
        main = deepcopy(self.main)
        main['time']['policy_cutoff_utc'] = '2025-12-31T22:00:00Z'
        with self.assertRaisesRegex(ContractError, 'POLICY_CUTOFF_CHANGED'):
            validate_configuration(main, self.poly)

    def test_negative_unapproved_hac_change_is_rejected(self):
        main = deepcopy(self.main)
        main['inference']['hac_elapsed_hours'] = [24]
        with self.assertRaisesRegex(ContractError, 'SCIENTIFIC_CONFIGURATION_CHANGED'):
            validate_configuration(main, self.poly)

    def test_negative_missing_baseline_and_environment_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ContractError, 'BASELINE_MISSING'):
                verify_sha(Path(directory) / 'absent.json', CORE_SHA256['v3_results'])
        with self.assertRaisesRegex(ContractError, 'CBAM_INPUT_ROOT_REQUIRED'):
            input_root({})

    def test_negative_sample_encoding_without_timestamps_fails(self):
        with self.assertRaisesRegex(ContractError, 'ACTUAL_TIMESTAMPS_REQUIRED'):
            prove_encoding_bridge(None, None, '0' * 64, '1' * 64)

    def test_negative_changed_actual_timestamps_fail(self):
        left = ['2025-12-31T22:00:00Z']
        right = ['2025-12-31T23:00:00Z']
        with self.assertRaisesRegex(ContractError, 'SAMPLE_TIMESTAMPS_CHANGED'):
            prove_encoding_bridge(left, right, '0' * 64, '1' * 64)

    def test_negative_output_overlap_and_path_escape_fail(self):
        with self.assertRaisesRegex(ContractError, 'OVERLAP'):
            disjoint(self.root, self.root / 'baselines' / 'new_run')
        for bad in ['../outside', '/absolute/baseline', 'baselines/../../outside']:
            with self.assertRaisesRegex(ContractError, 'INVALID_PRIVATE_RELATIVE_PATH'):
                confined(self.root, bad)


if __name__ == '__main__':
    unittest.main()
