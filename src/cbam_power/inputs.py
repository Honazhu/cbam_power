"""Offline, read-only logical inputs. Python >= 3.11; standard library only.

I = CBAM_INPUT_ROOT; B = I/baselines; A = CBAM_OUTPUT_ROOT (default I's
sibling runs). No function here creates outputs, repairs inputs, changes a
manifest, discovers a newer revision, or consults an archived project.
"""
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tomllib
import zipfile

# Support both source imports and a regular wheel installed into this project's
# .venv. This searches only enclosing candidate directories, never legacy data.
# An external installation can still supply an explicit manifest_path.
REPO = next((root for root in Path(__file__).resolve().parents
             if (root / 'src/cbam_power/inputs.py').is_file()
             and (root / 'manifests/inputs.toml').is_file()),
            Path(__file__).resolve().parent / '_resources')
FIELDS = frozenset({
    'asset_id', 'provider', 'dataset', 'legacy_source', 'legacy_sources',
    'relative_path', 'source_vintage', 'sha256', 'size_bytes', 'format',
    'schema_ref', 'temporal_coverage', 'frequency', 'units', 'classification',
    'kind', 'required', 'required_when', 'materialization', 'allowed_uses',
    'downstream', 'variables', 'transformation', 'upstream_assets',
    'license_status', 'redistribution_status', 'license_evidence', 'read_only',
})
FORMATS = {'csv', 'json', 'geojson', 'parquet', 'netcdf', 'zip', 'xlsx',
           'npz', 'html', 'py', 'md', 'tex', 'txt', 'text', 'pdf', 'png'}


class InputError(ValueError):
    """A stable failure code, without disclosing local private paths."""

    def __init__(self, code, asset_id=''):
        self.code = code
        self.asset_id = asset_id
        super().__init__(code + (': ' + asset_id if asset_id else ''))


def require(condition, code, asset_id=''):
    if not condition:
        raise InputError(code, asset_id)


def relative_path(value):
    """Canonical POSIX paths only; reject Windows paths on POSIX as well."""
    require(isinstance(value, str) and bool(value), 'INVALID_RELATIVE_PATH')
    path = PurePosixPath(value)
    require(bool(path.parts) and not path.is_absolute() and path.as_posix() == value
            and not any(p in {'..', '.'} for p in path.parts)
            and not any(c in value for c in '\\:*?[]')
            and not any(ord(c) < 32 for c in value), 'INVALID_RELATIVE_PATH')
    return path


def stream_sha(handle):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def disjoint(left, right):
    left, right = Path(left).resolve(), Path(right).resolve()
    require(left != right and left not in right.parents and right not in left.parents,
            'ROOTS_OVERLAP')


def validate_manifest(data, baselines):
    """Validate references and roles before any payload can be resolved."""
    require(data.get('offline_only') is True and data.get('allow_network') is False
            and data.get('allow_legacy_fallback') is False, 'UNSAFE_INPUT_POLICY')
    require(data.get('baseline_subdirectory') == 'baselines', 'INVALID_BASELINE_ROOT')
    assets = data.get('assets', [])
    require(isinstance(assets, list) and assets, 'INVALID_MANIFEST')
    by_id, paths, baseline_refs = {}, {}, {}
    for asset in assets:
        require(FIELDS <= asset.keys(), 'INCOMPLETE_ASSET_FIELDS')
        aid = asset['asset_id']
        require(isinstance(aid, str) and bool(re.fullmatch(r'[A-Za-z0-9_]+', aid)),
                'INVALID_ASSET_ID')
        require(aid not in by_id, 'DUPLICATE_ASSET_ID', aid)
        require(asset['classification'] in {'F', 'K', 'P', 'H'}, 'INVALID_CLASS', aid)
        require(bool(re.fullmatch('[0-9a-f]{64}', asset['sha256'])), 'INVALID_HASH', aid)
        require(type(asset['size_bytes']) is int and asset['size_bytes'] >= 0,
                'INVALID_SIZE', aid)
        require(asset['format'] in FORMATS, 'INVALID_FORMAT', aid)
        require(asset['schema_ref'] in data.get('schemas', {}), 'MISSING_SCHEMA', aid)
        require(type(asset['required']) is bool and asset['read_only'] is True,
                'INVALID_ASSET_POLICY', aid)
        require(asset['materialization'] in {'reuse_pr1', 'private_copy',
                    'existing_verified_object', 'reference_only'}, 'INVALID_STORAGE', aid)
        path = str(relative_path(asset['relative_path']))
        for source in [asset['legacy_source'], *asset['legacy_sources']]:
            relative_path(source)  # Documentary provenance; never a file lookup.
        identity = (asset['sha256'], asset['size_bytes'], asset['format'])
        require(path not in paths or paths[path] == identity, 'CONFLICTING_PATH_IDENTITY', aid)
        paths[path] = identity
        uses = asset['allowed_uses']
        require(isinstance(uses, list) and set(uses) <=
                {'process', 'verify', 'presentation', 'frozen_panel'}, 'INVALID_USE', aid)
        require('process' not in uses or asset['classification'] == 'F',
                'BENCHMARK_AS_ESTIMATOR_INPUT', aid)
        require(asset['classification'] != 'H' or not uses, 'HISTORICAL_RUNTIME_INPUT', aid)
        require(asset['classification'] not in {'P', 'H'} or not asset['required'],
                'OPTIONAL_CLASS_REQUIRED', aid)
        require(not asset['required'] or asset['materialization'] != 'reference_only',
                'REQUIRED_INPUT_UNMATERIALIZED', aid)
        if 'baseline_id' in asset:
            require(asset['baseline_id'] not in baseline_refs, 'DUPLICATE_BASELINE_REFERENCE', aid)
            baseline_refs[asset['baseline_id']] = asset
        by_id[aid] = asset
    expected = {row['id']: row for row in baselines['files']}
    require(set(baseline_refs) == set(expected), 'INCOMPLETE_BASELINE_REFERENCE')
    for bid, source in expected.items():
        asset = baseline_refs[bid]
        require((asset['sha256'], asset['size_bytes'], asset['relative_path']) ==
                (source['sha256'], source['bytes'], source['private_relative_path']),
                'BASELINE_IDENTITY_CHANGED', asset['asset_id'])
        require(asset['required'], 'BASELINE_NOT_REQUIRED', asset['asset_id'])
    require(data.get('root_identity_asset') in by_id, 'MISSING_ROOT_IDENTITY')
    for asset in assets:
        require(set(asset['upstream_assets']) <= by_id.keys(), 'UNKNOWN_UPSTREAM_ASSET', asset['asset_id'])
    stages = data.get('stages', [])
    require(len({s['stage_id'] for s in stages}) == len(stages), 'DUPLICATE_STAGE_ID')
    for stage in stages:
        require(set(stage['inputs']) <= by_id.keys(), 'UNKNOWN_STAGE_INPUT')
        if stage['stage_id'] in {'market', 'panel', 'weather', 'generation', 'synthetic', 'estimate'}:
            require(all('process' in by_id[aid]['allowed_uses'] for aid in stage['inputs']),
                    'BENCHMARK_AS_ESTIMATOR_INPUT')
    requests = data.get('era5_requests', [])
    require(len({r['request_id'] for r in requests}) == len(requests), 'DUPLICATE_REQUEST_ID')
    request_index = {r['request_id']: r for r in requests}
    for request in requests:
        require(request['manifest_asset_id'] in by_id, 'UNKNOWN_REQUEST_SOURCE')
        require(set(request['content_ids']) <= by_id.keys(), 'UNKNOWN_REQUEST_CONTENT')
        require(request['selected'] or not request['content_ids'], 'UNSELECTED_REQUEST_CONTENT')
    for row in data.get('era5_lineage', []):
        require(row['request_id'] in request_index, 'UNKNOWN_REQUEST_SOURCE')
        request = request_index[row['request_id']]
        require(request['selected'] and row['asset_id'] in request['content_ids'],
                'UNSELECTED_REQUEST_CONTENT')
        asset = by_id[row['asset_id']]
        require(asset['format'] == 'netcdf' and asset['classification'] == 'F'
                and asset['sha256'] == row['expected_sha256']
                and asset['size_bytes'] == row['size_bytes'], 'ERA5_CONTENT_IDENTITY')
        relative_path(row['legacy_logical_path'])
        relative_path(row['legacy_physical_path'])
    for selection in data.get('archive_selections', []):
        require(selection['asset_id'] in by_id, 'UNKNOWN_ARCHIVE_INPUT')
        require(selection['selected_members'] == by_id[selection['asset_id']]['selected_members'],
                'ARCHIVE_SELECTION_CHANGED')
        for member in selection['selected_members']:
            relative_path(member)
    return by_id


class InputCatalog:
    """An explicit manifest and authorized input root; default use is computation.

    Verification inputs require purpose='verify'. A frozen final panel can only
    be requested with explicit purpose='frozen_panel' or for verification/reporting.
    Optional reference-only artifacts are not silently fetched from the archive.
    """

    def __init__(self, *, input_root=None, manifest_path=None, output_root=None,
                 baseline_manifest_path=None):
        manifest_path = Path(manifest_path) if manifest_path else REPO / 'manifests/inputs.toml'
        baseline_path = (Path(baseline_manifest_path) if baseline_manifest_path else
                         manifest_path.parent / 'baselines.toml')
        try:
            self.manifest = tomllib.loads(manifest_path.read_text())
            baseline_bytes = baseline_path.read_bytes()
            self.baselines = tomllib.loads(baseline_bytes.decode())
        except (OSError, ValueError) as error:
            raise InputError('BLOCKED_MANIFEST_INVALID_OR_MISSING') from error
        require(hashlib.sha256(baseline_bytes).hexdigest() ==
                self.manifest['baseline_registry_sha256'], 'BASELINE_REGISTRY_HASH')
        self.assets = validate_manifest(self.manifest, self.baselines)
        value = input_root if input_root is not None else os.environ.get('CBAM_INPUT_ROOT')
        require(bool(value), 'BLOCKED_INPUT_ROOT_REQUIRED')
        raw_root = Path(value).expanduser().absolute()
        require(not any(p.is_symlink() for p in [raw_root, *raw_root.parents]), 'INPUT_ROOT_SYMLINK')
        self.input_root = raw_root.resolve()
        require(self.input_root.is_dir(), 'BLOCKED_INPUT_ROOT_MISSING')
        self.baseline_root = self.input_root / 'baselines'
        value = output_root if output_root is not None else os.environ.get('CBAM_OUTPUT_ROOT')
        self.output_root = Path(value).expanduser().resolve() if value else self.input_root.parent / 'runs'
        disjoint(self.input_root, self.output_root)
        disjoint(self.baseline_root, self.output_root)
        disjoint(REPO, self.input_root)
        disjoint(REPO, self.output_root)
        # An existing unrelated directory is not accepted as an authorized root.
        with self.open_input(self.manifest['root_identity_asset'], purpose='verify'):
            pass

    @contextmanager
    def _open_confined(self, asset):
        """Use no-follow directory descriptors, preventing symlink traversal races.

        The input root and every child are opened read-only. Parent descriptors
        remain open until the child is acquired. Non-regular files are refused.
        """
        aid, parts = asset['asset_id'], relative_path(asset['relative_path']).parts
        descriptors = []
        handle = None
        try:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            descriptor = os.open(self.input_root, flags)
            descriptors.append(descriptor)
            for part in parts[:-1]:
                descriptor = os.open(part, flags, dir_fd=descriptor)
                descriptors.append(descriptor)
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=descriptor)
            descriptors.append(descriptor)
            info = os.fstat(descriptor)
            require(stat.S_ISREG(info.st_mode), 'INPUT_NOT_REGULAR', aid)
            require(info.st_nlink == 1, 'INPUT_HARDLINK_FORBIDDEN', aid)
            handle = os.fdopen(descriptor, 'rb')
            descriptors.pop()
            yield handle
        except FileNotFoundError as error:
            raise InputError('BLOCKED_INPUT_MISSING', aid) from error
        except OSError as error:
            raise InputError('BLOCKED_INPUT_PATH', aid) from error
        finally:
            if handle is not None:
                handle.close()
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def _format(self, handle, asset):
        aid, fmt = asset['asset_id'], asset['format']
        schema = self.manifest['schemas'][asset['schema_ref']]
        head = handle.read(8)
        handle.seek(0)
        try:
            if fmt in {'json', 'geojson'}:
                obj = json.load(handle)
                require(type(obj).__name__ == schema['root_type'], 'INPUT_SCHEMA', aid)
                if 'top_level_keys' in schema:
                    require(sorted(obj) == schema['top_level_keys'], 'INPUT_SCHEMA', aid)
            elif fmt == 'csv':
                text = io.TextIOWrapper(handle, encoding='utf-8', newline='')
                try:
                    require((csv.DictReader(text).fieldnames or []) == schema['columns'], 'INPUT_SCHEMA', aid)
                finally:
                    text.detach()
            elif fmt in {'zip', 'xlsx', 'npz'}:
                with zipfile.ZipFile(handle) as archive:
                    names = archive.namelist()
                    require(set(asset.get('selected_members', [])) <= set(names), 'ARCHIVE_MEMBER_MISSING', aid)
                    require(len(names) == len(set(names)), 'DUPLICATE_ARCHIVE_MEMBER', aid)
                    if fmt == 'xlsx':
                        require('[Content_Types].xml' in names, 'INPUT_FORMAT', aid)
            elif fmt == 'netcdf':
                require(head == b'\x89HDF\r\n\x1a\n' or head[:4] in
                        {b'CDF\x01', b'CDF\x02', b'CDF\x05'}, 'INPUT_FORMAT', aid)
            elif fmt == 'parquet':
                handle.seek(-4, os.SEEK_END)
                require(head[:4] == b'PAR1' and handle.read(4) == b'PAR1', 'INPUT_FORMAT', aid)
            elif fmt == 'pdf':
                require(head.startswith(b'%PDF-'), 'INPUT_FORMAT', aid)
            elif fmt == 'png':
                require(head == b'\x89PNG\r\n\x1a\n', 'INPUT_FORMAT', aid)
            else:
                require(b'\x00' not in head, 'INPUT_FORMAT', aid)
        except (ValueError, KeyError, UnicodeError, csv.Error, zipfile.BadZipFile, OSError) as error:
            if isinstance(error, InputError):
                raise
            raise InputError('INPUT_FORMAT', aid) from error
        finally:
            handle.seek(0)

    @contextmanager
    def open_input(self, asset_id, *, purpose='process'):
        """Yield a verified read-only binary handle (prefer over reopening a path)."""
        require(asset_id in self.assets, 'UNKNOWN_ASSET_ID', asset_id)
        asset = self.assets[asset_id]
        require(purpose in asset['allowed_uses'], 'INPUT_ROLE_VIOLATION', asset_id)
        require(asset['materialization'] != 'reference_only', 'OPTIONAL_INPUT_NOT_MATERIALIZED', asset_id)
        with self._open_confined(asset) as handle:
            require(os.fstat(handle.fileno()).st_size == asset['size_bytes'], 'BLOCKED_INPUT_SIZE', asset_id)
            require(stream_sha(handle) == asset['sha256'], 'BLOCKED_INPUT_HASH', asset_id)
            handle.seek(0)
            self._format(handle, asset)
            yield handle

    def resolve(self, asset_id, *, purpose='process'):
        """Return a confined path after byte/size/container verification."""
        with self.open_input(asset_id, purpose=purpose):
            pass
        return self.input_root.joinpath(*relative_path(self.assets[asset_id]['relative_path']).parts)

    def resolve_request(self, request_id):
        requests = {row['request_id']: row for row in self.manifest['era5_requests']}
        require(request_id in requests, 'UNKNOWN_REQUEST_SOURCE', request_id)
        request = requests[request_id]
        require(request['selected'], 'REQUEST_NOT_SELECTED', request_id)
        return [self.resolve(aid) for aid in request['content_ids']]

    def validate_required(self):
        """Fail at the first invalid required input; no skips or auto-repairs."""
        verified = []
        for aid, asset in self.assets.items():
            if asset['required']:
                self.resolve(aid, purpose='verify')
                verified.append(aid)
        return {'verified_assets': verified, 'scope': 'static input identity only'}


def resolve_input(asset_id, *, input_root=None, manifest_path=None, output_root=None,
                  purpose='process'):
    return InputCatalog(input_root=input_root, manifest_path=manifest_path,
                        output_root=output_root).resolve(asset_id, purpose=purpose)
