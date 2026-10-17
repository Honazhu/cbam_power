"""PR2 static closure/security checks, never full scientific reproduction.

Run with CBAM_INPUT_ROOT set, PYTHONPATH=src and python -B. Real required
inputs are mandatory: missing inputs fail, never skip. Negative cases use only
tiny artificial fixtures outside the candidate and protected input directories.
"""
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch
import zipfile
import xml.etree.ElementTree as ET

from cbam_power.inputs import (InputCatalog, InputError, REPO, disjoint,
                              relative_path, resolve_input, stream_sha,
                              validate_manifest)

MANIFEST_DIGEST = 'b1dabea1d074eeea73a4ec0a5eca475e6357f9390f4fe74c17a5a81249de4ffd'


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def toml_value(value):
    """Fixture serialization only, never an expected-value update facility."""
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return '[' + ', '.join(toml_value(v) for v in value) + ']'
    if isinstance(value, dict):
        return '{' + ', '.join(toml_value(k) + '=' + toml_value(v) for k, v in value.items()) + '}'
    raise TypeError(value)


def write_toml(path, data):
    path.write_text('\n'.join(toml_value(k) + '=' + toml_value(v) for k, v in data.items()) + '\n')


class ClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = InputCatalog()
        cls.data = cls.catalog.manifest
        cls.assets = cls.catalog.assets

    def read_json(self, aid):
        with self.catalog.open_input(aid, purpose='verify') as h:
            return json.load(h)

    def read_csv(self, aid):
        with self.catalog.open_input(aid, purpose='verify') as h:
            return list(csv.DictReader(io.StringIO(h.read().decode())))

    def test_manifest_identity_and_classification(self):
        self.assertEqual(canonical_digest(self.data), MANIFEST_DIGEST)
        self.assertEqual(len(self.assets), 752)
        self.assertEqual(Counter(a['classification'] for a in self.assets.values()),
                         {'F': 440, 'K': 145, 'P': 79, 'H': 88})
        validate_manifest(self.data, self.catalog.baselines)

    def test_all_required_file_bytes_sizes_and_containers(self):
        result = self.catalog.validate_required()
        self.assertEqual(len(result['verified_assets']), 585)
        self.assertEqual(set(result['verified_assets']),
                         {a['asset_id'] for a in self.assets.values() if a['required']})

    def test_private_copy_dedup_and_read_only_boundary(self):
        copied = [a for a in self.assets.values() if a['materialization'] == 'private_copy']
        self.assertEqual(len(copied), 457)
        self.assertEqual(sum(a['size_bytes'] for a in copied), 493215324)
        required_paths = {a['relative_path'] for a in self.assets.values() if a['required']}
        self.assertEqual(len(required_paths), 584)
        for rel in required_paths:
            p = self.catalog.input_root / rel
            self.assertFalse(p.is_symlink())
            self.assertEqual(p.stat().st_nlink, 1)
            self.assertEqual(p.stat().st_mode & 0o222, 0)
        closure = self.catalog.input_root / 'closure'
        actual = {str(p.relative_to(self.catalog.input_root)) for p in closure.rglob('*') if p.is_file()}
        self.assertEqual(actual, {a['relative_path'] for a in copied})
        for directory in [closure, *(p for p in closure.rglob('*') if p.is_dir())]:
            mode = directory.stat()
            self.assertTrue(mode.st_mode & 0o222 == 0 or
                            getattr(mode, 'st_flags', 0) & getattr(stat, 'UF_IMMUTABLE', 2))

    def test_pr1_references_are_complete_and_unchanged(self):
        refs = {a['baseline_id']: a for a in self.assets.values() if 'baseline_id' in a}
        self.assertEqual(len(refs), 127)
        self.assertTrue({'v3_panel', 'v3_results', 'polynomial_results', 'v2_weights'} <= refs.keys())
        for baseline in self.catalog.baselines['files']:
            asset = refs[baseline['id']]
            self.assertEqual(asset['relative_path'], baseline['private_relative_path'])
            self.assertEqual(asset['sha256'], baseline['sha256'])
            self.assertEqual(asset['size_bytes'], baseline['bytes'])

    def test_all_543_legacy_sources_have_explicit_decisions(self):
        source_rows = self.read_csv('legacy_source_inventory')
        self.assertEqual(len(source_rows), 585)
        original = {r['relative_path']: r['sha256'] for r in source_rows}
        self.assertEqual(len(original), 543)
        decisions = self.data['legacy_inventory']['decisions']
        self.assertEqual(len(decisions), 543)
        self.assertEqual({r['legacy_source'] for r in decisions}, set(original))
        for row in decisions:
            asset = self.assets[row['asset_id']]
            self.assertEqual(row['archived_sha256'], original[row['legacy_source']])
            self.assertEqual(asset['sha256'], row['archived_sha256'])
            self.assertEqual(asset['classification'], row['classification'])
            self.assertIn(row['legacy_source'], asset['legacy_sources'])

    def test_v1_v2_lineage_and_synthetic_identity(self):
        main = tomllib.loads((REPO / 'config/main.toml').read_text())
        versions = self.data['synthetic_versions']
        self.assertEqual(versions, main['synthetic']['versions'])
        weights = self.read_csv('v2_weights')
        me = [w for w in weights if w['target'] == 'ME_IT']
        self.assertEqual(len(me), 32)
        self.assertEqual(sorted({w['version'] for w in me}), versions)
        self.assertEqual({w['donor'] for w in me}, {'AT_IT', 'GR_IT'})
        self.assertEqual(len(weights), 92)  # blocked CH rows preserved as well
        for aid in ['frozen_v1_panel', 'frozen_v2_panel', 'v2_weight_feature_freeze']:
            self.assertEqual(self.assets[aid]['classification'], 'F')
        self.assertEqual(self.assets['frozen_v2_panel']['upstream_assets'], ['frozen_v1_panel'])
        self.assertTrue(self.assets['frozen_v1_panel']['upstream_assets'])
        self.assertIn('UNVERIFIED', self.assets['frozen_v1_panel']['lineage_limitation'])

    def test_fr_ch_and_benchmark_estimator_separation(self):
        self.assertEqual(self.data['formal_actuals'], ['AT_IT', 'CH_IT', 'SI_IT', 'GR_IT', 'ME_IT'])
        self.assertEqual(self.data['synthetic_donors'], ['GR_IT', 'AT_IT'])
        self.assertFalse(self.data['CH_synthetic_enabled'])
        for aid, asset in self.assets.items():
            if asset['classification'] in {'K', 'P', 'H'}:
                self.assertNotIn('process', asset['allowed_uses'])
        with self.assertRaisesRegex(InputError, 'INPUT_ROLE_VIOLATION'):
            self.catalog.resolve('v3_results')
        with self.assertRaisesRegex(InputError, 'INPUT_ROLE_VIOLATION'):
            self.catalog.resolve('weather_v2f_benchmark')

    def test_era5_selection_reconstructed_from_frozen_requests(self):
        expected = []
        states = {}
        for label in ['v23', 'v2f']:
            document = self.read_json('era5_requests_' + label)
            for req in document['requests']:
                selected = bool(req.get('downloaded')) and (label == 'v2f' or not req.get('superseded_by'))
                rid = label + ':' + req['id']
                states[rid] = selected
                if selected:
                    expected.extend((rid, f['sha256'], f['bytes']) for f in req.get('files', []))
        actual = [(r['request_id'], r['expected_sha256'], r['size_bytes']) for r in self.data['era5_lineage']]
        self.assertEqual(Counter(actual), Counter(expected))
        self.assertEqual(len(actual), 58)
        self.assertEqual(len(states), 44)
        self.assertEqual({r['request_id']: r['selected'] for r in self.data['era5_requests']}, states)
        for req in self.data['era5_requests']:
            expected_ids = [r['asset_id'] for r in self.data['era5_lineage'] if r['request_id'] == req['request_id']]
            self.assertEqual(req['content_ids'], expected_ids)

    def test_era5_unique_content_and_22_recovered_paths(self):
        lineage = self.data['era5_lineage']
        contents = {row['asset_id'] for row in lineage}
        self.assertEqual(len(contents), 50)
        self.assertEqual(len({self.assets[aid]['sha256'] for aid in contents}), 50)
        broken = [r for r in lineage if r['legacy_path_broken']]
        self.assertEqual(len(broken), 30)
        self.assertEqual(len({r['legacy_logical_path'] for r in broken}), 22)
        for row in broken:
            self.assertIn('/v22_weather/', row['legacy_physical_path'])
            self.assertIn('/v23/', row['legacy_logical_path'])
        for aid in contents:
            asset = self.assets[aid]
            self.assertEqual(asset['format'], 'netcdf')
            self.assertEqual(asset['classification'], 'F')
            self.assertEqual(set(asset['request_ids']),
                             {r['request_id'] for r in lineage if r['asset_id'] == aid})
            self.catalog.resolve(aid)

    def test_superseded_and_unknown_request_fail(self):
        rejected = [r for r in self.data['era5_requests'] if not r['selected']]
        self.assertTrue(rejected)
        for req in rejected:
            with self.assertRaisesRegex(InputError, 'REQUEST_NOT_SELECTED'):
                self.catalog.resolve_request(req['request_id'])
        with self.assertRaisesRegex(InputError, 'UNKNOWN_REQUEST_SOURCE'):
            self.catalog.resolve_request('unregistered_request')

    def test_market_archives_freeze_member_revision_rules(self):
        for choice in self.data['archive_selections']:
            with self.catalog.open_input(choice['asset_id']) as handle, zipfile.ZipFile(handle) as archive:
                names = archive.namelist()
                selected = []
                if choice['rule'] == 'GME_PREZZI15_PRIORITY':
                    for name in sorted(names):
                        if not name.endswith(('Prezzi.xml', 'Prezzi15.xml')):
                            continue
                        quarter = name.endswith('Prezzi15.xml')
                        records = ET.fromstring(archive.read(name)).findall('Prezzi15' if quarter else 'Prezzi')
                        if not records:
                            continue
                        date = records[0].findtext('Data')
                        if int(date[:4]) not in [2024, 2025, 2026]:
                            continue
                        if not quarter and name.replace('Prezzi.xml', 'Prezzi15.xml') in names:
                            continue
                        selected.append(name)
                else:
                    groups = {}
                    for name in names:
                        if '/DAM/' in name and re.search(r'\d{8}_EL-DAM_ResultsSummary_EN_v\d+\.xlsx$', name):
                            groups.setdefault(Path(name).name[:8], []).append(name)
                    selected = [max(group, key=lambda s: int(re.search(r'_v(\d+)', s).group(1)))
                                for day, group in sorted(groups.items())]
                self.assertEqual(selected, choice['selected_members'])

    def test_price_source_status_and_generation_missing_states(self):
        selected = {(r['country'], r['year']) for r in self.data['price_mirror_selection'] if r['selected']}
        self.assertEqual(selected, {('SI', 2024), ('GR', 2024)})
        for row in self.data['price_mirror_selection']:
            aid = row['asset_id'] + '_metadata' if row['selected'] else row['metadata_asset_id']
            metadata = self.read_json(aid)
            self.assertEqual(metadata['status'], row['status'])
            if row['selected']:
                self.assertEqual(metadata['sha256'], self.assets[row['asset_id']]['sha256'])
        self.assertEqual(sum(aid.startswith('generation_') for aid in self.assets), 6)
        self.assertEqual(sum(aid.startswith('smard_SI_') for aid in self.assets), 90)
        self.assertEqual(sum(aid.startswith('henex_results_') for aid in self.assets), 259)

    def test_portable_reads_without_network_archive_or_writes(self):
        # Fresh interpreter, unrelated cwd, bytecode disabled; audit hook fails
        # any payload read outside the new manifest paths and the authorized I.
        script = '''
import os, sys
from pathlib import Path
from cbam_power.inputs import InputCatalog
repo, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
def audit(event, args):
    if event.startswith('socket.'):
        raise AssertionError('NETWORK_ATTEMPT')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        p = Path(os.fsdecode(args[0]))
        mode, flags = args[1], args[2]
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise AssertionError('INPUT_WRITE_ATTEMPT')
        # Relative opens are confined via directory descriptors in inputs.py.
        if p.is_absolute() and not (p == root or root in p.parents or repo in p.parents):
            raise AssertionError('UNREGISTERED_EXTERNAL_READ')
sys.addaudithook(audit)
c = InputCatalog(input_root=root)
c.resolve('frozen_v1_panel')
c.resolve('v3_results', purpose='verify')
print('OFFLINE_READ_ONLY_PORTABLE')
'''
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ, PYTHONPATH=str(REPO / 'src'), PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run([sys.executable, '-B', '-c', script, str(REPO), str(self.catalog.input_root)],
                                    cwd=temp, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('OFFLINE_READ_ONLY_PORTABLE', result.stdout)

    def test_public_boundary_and_optional_references(self):
        for asset in self.assets.values():
            self.assertNotEqual(asset['redistribution_status'], 'APPROVED_PUBLIC')
            self.assertFalse(asset['relative_path'].startswith('/'))
            if asset['classification'] in {'P', 'H'}:
                self.assertFalse(asset['required'])
                self.assertEqual(asset['materialization'], 'reference_only')
                self.assertFalse((self.catalog.input_root / asset['relative_path']).exists())
        with self.assertRaisesRegex(InputError, 'OPTIONAL_INPUT_NOT_MATERIALIZED'):
            self.catalog.resolve('gis_countries', purpose='presentation')
        disjoint(self.catalog.input_root, REPO)
        disjoint(self.catalog.input_root, self.catalog.output_root)

    def test_manifest_negative_missing_baseline_reference(self):
        data = deepcopy(self.data)
        del next(a for a in data['assets'] if a['asset_id'] == 'v3_results')['baseline_id']
        with self.assertRaisesRegex(InputError, 'INCOMPLETE_BASELINE_REFERENCE'):
            validate_manifest(data, self.catalog.baselines)

    def test_manifest_negative_era5_wrong_hash_or_source(self):
        for field, value, error in [('expected_sha256', '0' * 64, 'ERA5_CONTENT_IDENTITY'),
                                    ('request_id', 'unknown', 'UNKNOWN_REQUEST_SOURCE')]:
            data = deepcopy(self.data)
            data['era5_lineage'][0][field] = value
            with self.assertRaisesRegex(InputError, error):
                validate_manifest(data, self.catalog.baselines)

    def test_manifest_negative_benchmark_in_estimation_stage(self):
        data = deepcopy(self.data)
        next(s for s in data['stages'] if s['stage_id'] == 'estimate')['inputs'] = ['v3_results']
        with self.assertRaisesRegex(InputError, 'BENCHMARK_AS_ESTIMATOR_INPUT'):
            validate_manifest(data, self.catalog.baselines)


class NegativeInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name).resolve()
        self.root = self.parent / 'authorized'
        self.root.mkdir()
        (self.root / 'baselines').mkdir()
        self.payload = b'{"revision":1}\n'
        self.path = self.root / 'baselines/probe.json'
        self.path.write_bytes(self.payload)
        sha = hashlib.sha256(self.payload).hexdigest()
        self.baselines = {'files': [{'id': 'probe', 'private_relative_path': 'baselines/probe.json',
                                     'sha256': sha, 'bytes': len(self.payload)}]}
        self.baseline_path = self.parent / 'baselines.toml'
        write_toml(self.baseline_path, self.baselines)
        asset = dict(asset_id='probe', provider='Artificial fixture', dataset='Fixture',
                     legacy_source='fixture.json', legacy_sources=['fixture.json'],
                     relative_path='baselines/probe.json', source_vintage='fixture', sha256=sha,
                     size_bytes=len(self.payload), format='json', schema_ref='json',
                     temporal_coverage='not applicable', frequency='static', units=[],
                     classification='F', kind='frozen_upstream', required=True,
                     required_when=[], materialization='reuse_pr1', allowed_uses=['process', 'verify'],
                     downstream=['test'], variables=[], transformation='none', upstream_assets=[],
                     license_status='artificial', redistribution_status='UNVERIFIED',
                     license_evidence='fixture', read_only=True, baseline_id='probe')
        self.data = dict(offline_only=True, allow_network=False, allow_legacy_fallback=False,
                         baseline_subdirectory='baselines', root_identity_asset='probe',
                         baseline_registry_sha256=hashlib.sha256(self.baseline_path.read_bytes()).hexdigest(),
                         assets=[asset], schemas={'json': {'root_type': 'dict', 'top_level_keys': ['revision']}},
                         stages=[], era5_requests=[], era5_lineage=[], archive_selections=[])
        self.manifest_path = self.parent / 'inputs.toml'
        self.save()

    def save(self):
        write_toml(self.manifest_path, self.data)

    def catalog(self, **kwargs):
        return InputCatalog(input_root=kwargs.pop('input_root', self.root),
                            manifest_path=self.manifest_path, **kwargs)

    def test_artificial_fixture_and_read_only_handle(self):
        c = self.catalog()
        with c.open_input('probe') as handle:
            self.assertEqual(handle.read(), self.payload)
            with self.assertRaises(io.UnsupportedOperation):
                handle.write(b'forbidden')
        self.assertEqual(resolve_input('probe', input_root=self.root, manifest_path=self.manifest_path), self.path)

    def test_missing_input_fails(self):
        self.path.unlink()
        with self.assertRaisesRegex(InputError, 'BLOCKED_INPUT_MISSING'):
            self.catalog()

    def test_modified_file_and_wrong_revision_fail(self):
        c = self.catalog()
        self.path.write_bytes(b'{"revision":2}\n')  # same size, different content
        with self.assertRaisesRegex(InputError, 'BLOCKED_INPUT_HASH'):
            c.resolve('probe')

    def test_wrong_size_fails(self):
        self.path.write_bytes(b'wrong')
        with self.assertRaisesRegex(InputError, 'BLOCKED_INPUT_SIZE'):
            self.catalog()

    def test_duplicate_asset_id_fails(self):
        self.data['assets'].append(deepcopy(self.data['assets'][0]))
        self.save()
        with self.assertRaisesRegex(InputError, 'DUPLICATE_ASSET_ID'):
            self.catalog()

    def test_conflicting_same_path_fails(self):
        other = deepcopy(self.data['assets'][0])
        other.update(asset_id='other', sha256='0' * 64)
        del other['baseline_id']
        self.data['assets'].append(other)
        self.save()
        with self.assertRaisesRegex(InputError, 'CONFLICTING_PATH_IDENTITY'):
            self.catalog()

    def test_invalid_hash_fails(self):
        for value in ['invalid', '0' * 63, 'G' * 64]:
            with self.subTest(value=value):
                self.data['assets'][0]['sha256'] = value
                self.save()
                with self.assertRaisesRegex(InputError, 'INVALID_HASH'):
                    self.catalog()

    def test_absolute_traversal_glob_and_windows_paths_fail(self):
        for path in ['.', '', '/private/forbidden', '../escape', 'nested/../../escape', 'a//b',
                     './a', 'a/./b', 'a\\b', 'C:/private/a', 'a/*', 'a\x00b']:
            with self.subTest(path=path), self.assertRaisesRegex(InputError, 'INVALID_RELATIVE_PATH'):
                relative_path(path)

    def test_file_symlink_even_inside_root_fails(self):
        original = self.path.with_name('other.json')
        self.path.rename(original)
        self.path.symlink_to(original)
        with self.assertRaisesRegex(InputError, 'BLOCKED_INPUT_PATH'):
            self.catalog()

    def test_parent_symlink_escape_fails(self):
        outside = self.parent / 'outside'
        (self.root / 'baselines').rename(outside)
        (self.root / 'baselines').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(InputError, 'BLOCKED_INPUT_PATH'):
            self.catalog()

    def test_hardlink_to_other_file_fails(self):
        os.link(self.path, self.parent / 'alias')
        with self.assertRaisesRegex(InputError, 'INPUT_HARDLINK_FORBIDDEN'):
            self.catalog()

    def test_unknown_asset_fails(self):
        with self.assertRaisesRegex(InputError, 'UNKNOWN_ASSET_ID'):
            self.catalog().resolve('unregistered')

    def test_wrong_root_no_legacy_fallback(self):
        other = self.parent / 'wrong'
        other.mkdir()
        with self.assertRaisesRegex(InputError, 'BLOCKED_INPUT_MISSING'):
            self.catalog(input_root=other)
        with self.assertRaisesRegex(InputError, 'BLOCKED_INPUT_ROOT_MISSING'):
            self.catalog(input_root=self.parent / 'absent')

    def test_missing_root_env_fails(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(InputError, 'BLOCKED_INPUT_ROOT_REQUIRED'):
                InputCatalog(manifest_path=self.manifest_path)

    def test_input_output_and_repo_overlap_fail(self):
        for output in [self.root, self.root / 'results', self.root / 'baselines/out', self.parent, REPO]:
            with self.subTest(output=output), self.assertRaisesRegex(InputError, 'ROOTS_OVERLAP'):
                self.catalog(output_root=output)
        with self.assertRaisesRegex(InputError, 'ROOTS_OVERLAP'):
            self.catalog(input_root=REPO)

    def test_symlink_input_root_fails(self):
        link = self.parent / 'link'
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(InputError, 'INPUT_ROOT_SYMLINK'):
            self.catalog(input_root=link)

    def test_baseline_registry_corruption_fails(self):
        self.baseline_path.write_text('# wrong revision\n' + self.baseline_path.read_text())
        with self.assertRaisesRegex(InputError, 'BASELINE_REGISTRY_HASH'):
            self.catalog()

    def test_missing_required_field_fails(self):
        del self.data['assets'][0]['provider']
        self.save()
        with self.assertRaisesRegex(InputError, 'INCOMPLETE_ASSET_FIELDS'):
            self.catalog()

    def test_incomplete_benchmark_reference_fails(self):
        del self.data['assets'][0]['baseline_id']
        self.save()
        with self.assertRaisesRegex(InputError, 'INCOMPLETE_BASELINE_REFERENCE'):
            self.catalog()

    def test_invalid_class_fails(self):
        self.data['assets'][0]['classification'] = 'UNKNOWN'
        self.save()
        with self.assertRaisesRegex(InputError, 'INVALID_CLASS'):
            self.catalog()

    def test_network_or_fallback_policy_fails(self):
        for key in ['allow_network', 'allow_legacy_fallback']:
            data = deepcopy(self.data)
            data[key] = True
            with self.assertRaisesRegex(InputError, 'UNSAFE_INPUT_POLICY'):
                validate_manifest(data, self.baselines)

    def test_basic_format_and_schema_fail(self):
        for payload in [b'not JSON', b'{"changed":1}']:
            with self.subTest(payload=payload):
                self.path.write_bytes(payload)
                digest = hashlib.sha256(payload).hexdigest()
                self.data['assets'][0].update(sha256=digest, size_bytes=len(payload))
                self.baselines['files'][0].update(sha256=digest, bytes=len(payload))
                write_toml(self.baseline_path, self.baselines)
                self.data['baseline_registry_sha256'] = hashlib.sha256(self.baseline_path.read_bytes()).hexdigest()
                self.save()
                with self.assertRaisesRegex(InputError, 'INPUT_FORMAT|INPUT_SCHEMA'):
                    self.catalog()


if __name__ == '__main__':
    unittest.main()
