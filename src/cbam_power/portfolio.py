"""Offline checks and fresh-kernel execution for the reviewed teaching portfolio.

Security review and artificial-output permission do not grant source ownership.
Kernel socket guards supplement the caller's OS isolation; they are not a sandbox.
"""
from __future__ import annotations

import argparse
import base64
import copy
import csv
import gzip
import hashlib
import io
import json
import os
import re
from pathlib import Path
import stat
import sys
import tarfile
import tomllib
import zipfile

from . import release


class PortfolioError(RuntimeError):
    """Stable local error identifier; never include a payload or personal path."""


def require(condition, code):
    if not condition:
        raise PortfolioError(code)


def inspect_source(source_root):
    """Inspect the exact reviewed SOURCE + ARTIFICIAL portfolio file closure."""
    root = Path(source_root).absolute()
    require('..' not in root.parts and not any(p.is_symlink() for p in [root, *root.parents]),
            'PORTFOLIO_SOURCE_PATH_UNSAFE')
    root = root.resolve()
    raw = release._safe_read(root, 'manifests/release_rights.toml', 16*1024*1024)
    data = tomllib.loads(raw.decode())
    # A registry cannot contain its own byte hash; this one source metadata
    # object receives a detached identity for the current security audit.
    for node in data['nodes']:
        if node['id'] == 'source:manifests/release_rights.toml':
            require(node['kind'] == 'source' and node['origin'] == 'SOURCE'
                    and node['path'] == 'manifests/release_rights.toml', 'PORTFOLIO_REGISTRY_SELF_BINDING_INVALID')
            node['sha256'] = release.sha256(raw)
    registry = release.RightsRegistry(data)
    names = data.get('candidate_whitelist')
    require(isinstance(names, list) and len(names) == len(set(names)), 'PORTFOLIO_SOURCE_ALLOWLIST_REQUIRED')
    selection = {name: 'source:'+name for name in names}
    artifacts = data.get('portfolio_artifacts', {})
    require(isinstance(artifacts, dict) and not selection.keys() & artifacts.keys(), 'PORTFOLIO_ARTIFACT_ALLOWLIST_INVALID')
    for name, key in selection.items():
        node = registry.node(key)
        require(node['kind'] == 'source' and node['origin'] == 'SOURCE'
                and node.get('path') == name, 'PORTFOLIO_SOURCE_CLASSIFICATION_INVALID')
    for name, key in artifacts.items():
        node = registry.node(key)
        require(node['kind'] == 'artifact' and node['origin'] == 'ARTIFICIAL'
                and node.get('path') == name, 'PORTFOLIO_ARTIFACT_CLASSIFICATION_INVALID')
        registry.authorize(key)
    selection.update(artifacts)
    files = set()
    for path in root.rglob('*'):
        # Git administrative state is not a source/publication payload. This
        # command makes no claim about Git history or credential configuration.
        relative = path.relative_to(root)
        if relative.parts[0] == '.git':
            continue
        require(not path.is_symlink(), 'PORTFOLIO_SOURCE_LINK_REJECTED')
        if path.is_file():
            files.add(relative.as_posix())
    require(files == selection.keys(), 'PORTFOLIO_SOURCE_CLOSURE_CHANGED')
    report = release.audit_candidate(registry, root, whitelist=selection)
    for name, key in selection.items():
        if registry.node(key).get('format') == 'notebook':
            document = json.loads(release._safe_read(root, name, 4*1024*1024))
            inspect_notebook_links(document, root, name)
    site_nodes=[n for n in registry.nodes.values() if n.get('content_role')=='GENERATED_RESEARCH_SITE']
    if site_nodes:
        from . import site
        for n in site_nodes:
            require(site.render_site(root)==release._safe_read(root,n['path'],4*1024*1024).decode(), 'PORTFOLIO_SITE_BUILD_STALE')
    return root, registry, selection, report


def inspect_notebook_links(document, root, name):
    for cell in document['cells']:
        if cell['cell_type'] != 'markdown': continue
        source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        for image, target in re.findall(r'(!?)\[[^\]]*\]\(([^\s)]+)\)', source):
            if target.startswith(('https://', 'http://')):
                require(not image, 'PORTFOLIO_REMOTE_NOTEBOOK_IMAGE_REJECTED')
                continue
            target = target.split('#', 1)[0]
            if not target: continue
            require(not Path(target).is_absolute(), 'PORTFOLIO_NOTEBOOK_LINK_OUTSIDE_SOURCE')
            path = (root/Path(name).parent/target).resolve()
            require(path.is_relative_to(root), 'PORTFOLIO_NOTEBOOK_LINK_OUTSIDE_SOURCE')
            require(path.is_file(), 'PORTFOLIO_NOTEBOOK_LINK_MISSING')


def inspect_installed_code(root, selection):
    package = Path(__file__).parent
    count = 0
    for name in selection:
        if name.startswith('src/cbam_power/') and name.endswith('.py'):
            relative = name.removeprefix('src/cbam_power/')
            installed = release._safe_read(package, relative, 4*1024*1024)
            source = release._safe_read(root, name, 4*1024*1024)
            require(installed == source, 'PORTFOLIO_INSTALLED_CODE_STALE')
            count += 1
    require(count > 0, 'PORTFOLIO_PACKAGE_SOURCE_MISSING')
    return count


def inspect_packages(directory, root, registry, selection):
    """Bind actual wheel/sdist members to the audited source; scan recursively."""
    project = tomllib.loads((root/'pyproject.toml').read_text())
    prefix = 'cbam_power-'+project['project']['version']
    dist = prefix+'.dist-info/'
    expected = {prefix+'-py3-none-any.whl', prefix+'.tar.gz'}
    paths = list(Path(directory).iterdir())
    require({p.name for p in paths} == expected and all(p.is_file() and not p.is_symlink() for p in paths),
            'PORTFOLIO_PACKAGE_SET_CHANGED')
    wheelmap = {name.removeprefix('src/'):name for name in selection if name.startswith('src/')}
    includes = project['tool']['hatch']['build']['targets']['wheel'].get('force-include', {})
    for source, target in includes.items():
        release.relative(source);release.relative(target)
        matches = [n for n in selection if n == source or n.startswith(source+'/')]
        require(matches, 'PORTFOLIO_WHEEL_INCLUDE_UNKNOWN')
        for name in matches:
            wheelmap[target+name[len(source):]] = name
    license_files = project['project'].get('license-files', [])
    require(isinstance(license_files, list) and all(isinstance(n, str) for n in license_files)
            and len(license_files) == len(set(license_files)), 'PORTFOLIO_LICENSE_FILES_INVALID')
    for name in license_files:
        release.relative(name)  # Explicit relative paths only; no glob expansion.
        require(name in selection, 'PORTFOLIO_LICENSE_FILE_UNREGISTERED')
        node = registry.node(selection[name])
        require(node['kind'] == 'source' and node['origin'] == 'SOURCE'
                and node.get('path') == name, 'PORTFOLIO_LICENSE_FILE_NOT_SOURCE')
        wheelmap[dist+'licenses/'+name] = name
    metadata = {dist+n for n in ['METADATA', 'WHEEL', 'entry_points.txt', 'RECORD']}
    reports = []
    for path in sorted(paths):
        content = release._safe_read(Path(directory), path.name, 32*1024*1024)
        data = copy.deepcopy(registry.data)
        def add(key, payload, fmt, name, members=None):
            record = dict(id=key, kind='source', origin='SOURCE', status='PENDING_OWNER_APPROVAL',
                dependencies=list(members.values()) if members else [], lineage_complete=True, lineage_kind='source',
                evidence=[], conditions=[], permissions=[], sha256=release.sha256(payload), path=name, format=fmt,
                reason='Private packaging security inspection; no source publication grant.')
            if members is not None:record['members'] = members
            data['nodes'].append(record)
            return key
        if path.suffix == '.whl':
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                infos = archive.infolist()
                require(len({i.filename for i in infos}) == len(infos)
                        and len(infos) <= 500 and sum(i.file_size for i in infos) <= 128*1024*1024
                        and all(not i.is_dir() and stat.S_IFMT(i.external_attr >> 16) in {0, stat.S_IFREG}
                                and i.file_size <= 32*1024*1024
                                and i.file_size <= max(i.compress_size, 1)*200 for i in infos), 'PORTFOLIO_PACKAGE_MEMBERS_INVALID')
                contents = {i.filename:archive.read(i) for i in infos}
            require(set(contents) == set(wheelmap) | metadata, 'PORTFOLIO_PACKAGE_MEMBERS_CHANGED')
            mapping = {name:wheelmap.get(name) for name in contents}
            rows = list(csv.reader(io.StringIO(contents[dist+'RECORD'].decode())))
            require(len(rows) == len(contents) and {r[0] for r in rows} == contents.keys(), 'PORTFOLIO_WHEEL_RECORD_INVALID')
            for name, digest, size in rows:
                if name == dist+'RECORD': require(digest == size == '', 'PORTFOLIO_WHEEL_RECORD_INVALID')
                else:
                    encoded = base64.urlsafe_b64encode(hashlib.sha256(contents[name]).digest()).decode().rstrip('=')
                    require(digest == 'sha256='+encoded and size == str(len(contents[name])), 'PORTFOLIO_WHEEL_RECORD_INVALID')
        else:
            with gzip.GzipFile(fileobj=io.BytesIO(content)) as compressed:
                tarbytes = compressed.read(64*1024*1024+1)
            require(len(tarbytes) <= 64*1024*1024, 'PORTFOLIO_PACKAGE_SIZE_LIMIT')
            with tarfile.open(fileobj=io.BytesIO(tarbytes), mode='r:') as archive:
                infos = archive.getmembers()
                require(len({i.name for i in infos}) == len(infos) and len(infos) <= 500
                        and sum(i.size for i in infos) <= 128*1024*1024 and all(i.isfile() and i.size <= 32*1024*1024 for i in infos), 'PORTFOLIO_PACKAGE_MEMBERS_INVALID')
                contents = {i.name:archive.extractfile(i).read() for i in infos}
            require(all(n.startswith(prefix+'/') for n in contents)
                    and {n.removeprefix(prefix+'/') for n in contents} == set(selection) | {'PKG-INFO'},
                    'PORTFOLIO_PACKAGE_MEMBERS_CHANGED')
            mapping = {name:None if name == prefix+'/PKG-INFO' else name.removeprefix(prefix+'/') for name in contents}
        members = {}
        for name, payload in contents.items():
            source = mapping[name]
            if source:
                require(payload == release._safe_read(root, source, 32*1024*1024), 'PORTFOLIO_PACKAGE_SOURCE_CHANGED')
                members[name] = selection[source]
            else:
                members[name] = add('package-metadata:'+name, payload, 'csv' if name.endswith('/RECORD') else 'text', name)
        if path.suffix == '.whl': parent = add('package:'+path.name, content, 'zip', path.name, members)
        else:
            inner = add('tar:'+path.name, tarbytes, 'tar', path.name[:-3], members)
            parent = add('package:'+path.name, content, 'gzip', path.name, {path.name[:-3]:inner})
        release.scan_content(content, release.RightsRegistry(data), parent)
        reports.append(dict(file=path.name, sha256=release.sha256(content), members=len(contents), status='SECURITY_CLEAN'))
    return reports


_KERNEL_GUARD = r'''
import ipaddress as _ip, socket as _socket
import urllib.request as _request
from cbam_power import inputs as _inputs

def _local(address):
    host = address[0] if isinstance(address, tuple) else address
    if host == 'localhost': return True
    try: return _ip.ip_address(host).is_loopback
    except ValueError: return False

_original_connect, _original_connect_ex = _socket.socket.connect, _socket.socket.connect_ex
_original_sendto, _original_getaddrinfo = _socket.socket.sendto, _socket.getaddrinfo

def _connect(self, address):
    if self.family != _socket.AF_UNIX and not _local(address): raise RuntimeError('EXTERNAL_NETWORK_BLOCKED')
    return _original_connect(self, address)
def _connect_ex(self, address):
    if self.family != _socket.AF_UNIX and not _local(address): raise RuntimeError('EXTERNAL_NETWORK_BLOCKED')
    return _original_connect_ex(self, address)
def _sendto(self, data, *args):
    if self.family != _socket.AF_UNIX and not _local(args[-1]): raise RuntimeError('EXTERNAL_NETWORK_BLOCKED')
    return _original_sendto(self, data, *args)
def _getaddrinfo(host, *args, **kwargs):
    if not _local(host): raise RuntimeError('EXTERNAL_NETWORK_BLOCKED')
    return _original_getaddrinfo(host, *args, **kwargs)
def _deny(*args, **kwargs): raise RuntimeError('PRIVATE_INPUT_OR_NETWORK_ACCESS_BLOCKED')
_socket.socket.connect, _socket.socket.connect_ex = _connect, _connect_ex
_socket.socket.sendto, _socket.getaddrinfo = _sendto, _getaddrinfo
_request.urlopen = _deny
_inputs.InputCatalog.__init__ = _deny
_inputs.resolve_input = _deny
'''


def inspect_notebook_outputs(notebook):
    """Inspect private execution outputs without granting publication rights."""
    count = 0
    for cell in notebook['cells']:
        for output in cell.get('outputs', []):
            release._check_structured_credentials(output.get('metadata', {}))
            require(output.get('output_type') in {'stream', 'display_data', 'execute_result'}, 'PORTFOLIO_NOTEBOOK_ERROR_OUTPUT')
            if output['output_type'] == 'stream':
                text = output.get('text', '')
                text = ''.join(text) if isinstance(text, list) else text
                require(isinstance(text, str) and len(text) <= 1024*1024, 'PORTFOLIO_OUTPUT_SIZE_LIMIT')
                release._check_credentials(text)
            else:
                require(set(output.get('data', {})) <= {'text/plain', 'image/png'}, 'PORTFOLIO_OUTPUT_MIME_REJECTED')
                for mime, value in output.get('data', {}).items():
                    value = ''.join(value) if isinstance(value, list) else value
                    require(isinstance(value, str) and len(value) <= 32*1024*1024, 'PORTFOLIO_OUTPUT_SIZE_LIMIT')
                    if mime == 'text/plain': release._check_credentials(value)
                    else: release._scan_png(base64.b64decode(value, validate=True))
            count += 1
    return count


def compare_checkpoints(actual, expected):
    """Artificial demonstration contract only; no frozen scientific tolerance changes."""
    import math
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and actual.keys()==expected.keys(),'PORTFOLIO_CHECKPOINT_KEYS_CHANGED')
        for key in expected:compare_checkpoints(actual[key],expected[key])
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),'PORTFOLIO_CHECKPOINT_KEYS_CHANGED')
        for x,y in zip(actual,expected):compare_checkpoints(x,y)
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        require(isinstance(actual,(int,float)) and not isinstance(actual,bool) and math.isfinite(actual)
                and math.isfinite(expected) and math.isclose(actual,expected,abs_tol=1e-10,rel_tol=1e-10),'PORTFOLIO_CHECKPOINT_VALUE_CHANGED')
    else:require(type(actual) is type(expected) and actual==expected,'PORTFOLIO_CHECKPOINT_VALUE_CHANGED')


def execute_notebooks(root, registry, selection, output_root):
    require(registry.data.get('notebook_execution_authorized') is True, 'PORTFOLIO_NOTEBOOK_EXECUTION_NOT_AUTHORIZED')
    notebooks = registry.data.get('portfolio_notebooks')
    require(isinstance(notebooks, list) and len(notebooks) == len(set(notebooks)) == 4
            and set(notebooks) <= selection.keys(), 'PORTFOLIO_FOUR_NOTEBOOKS_REQUIRED')
    target = Path(output_root).absolute()
    require('..' not in target.parts and not any(p.is_symlink() for p in [target, *target.parents]), 'PORTFOLIO_OUTPUT_PATH_UNSAFE')
    target = target.resolve()
    require(not (target.is_relative_to(root) or root.is_relative_to(target)), 'PORTFOLIO_OUTPUT_SOURCE_OVERLAP')
    if os.environ.get('CBAM_INPUT_ROOT'):
        protected = Path(os.environ['CBAM_INPUT_ROOT']).resolve()
        require(not (target.is_relative_to(protected) or protected.is_relative_to(target)), 'PORTFOLIO_OUTPUT_INPUT_OVERLAP')
    require(not target.exists() or target.is_dir() and not list(target.iterdir()), 'PORTFOLIO_OUTPUT_MUST_BE_EMPTY')
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    target.mkdir(parents=True, exist_ok=True)
    for name in ['runtime', 'config', 'cache', 'ipython', 'tmp', 'home', 'kernels/cbam-portfolio']:
        (target/name).mkdir(parents=True, exist_ok=True)
    kernel_spec = dict(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                       display_name='Reviewed artificial portfolio', language='python')
    (target/'kernels/cbam-portfolio/kernel.json').write_text(json.dumps(kernel_spec))
    environment = dict(HOME=str(target/'home'), PATH=os.defpath, LANG='C.UTF-8', TZ='UTC', PYTHONDONTWRITEBYTECODE='1',
        PYTHONNOUSERSITE='1', MPLBACKEND='module://matplotlib_inline.backend_inline',
        MPLCONFIGDIR=str(target/'cache'), XDG_CACHE_HOME=str(target/'cache'),
        JUPYTER_CONFIG_DIR=str(target/'config'), JUPYTER_CONFIG_PATH='', JUPYTER_RUNTIME_DIR=str(target/'runtime'),
        IPYTHONDIR=str(target/'ipython'), TMPDIR=str(target/'tmp'))
    reports = []
    for number, name in enumerate(notebooks):
        node = registry.node(selection[name])
        require(node.get('format') == 'notebook' and node.get('content_role') in {'UNEXECUTED_NOTEBOOK','REVIEWED_ARTIFICIAL_NOTEBOOK'}, 'PORTFOLIO_NOTEBOOK_SOURCE_REQUIRED')
        content = release._safe_read(root, name, 4*1024*1024)
        release.scan_content(content, registry, node['id'])
        notebook = nbformat.from_dict(release.notebook_source(json.loads(content)))
        notebook.cells.insert(0, nbformat.v4.new_code_cell(_KERNEL_GUARD))
        work = target/('notebook-'+str(number+1));work.mkdir()
        publish = registry.data.get('notebook_output_publication_authorized') is True
        trials, checkpoints = [], []
        for repeat in range(2 if publish else 1):
            trial = nbformat.from_dict(release.notebook_source(json.loads(content)))
            trial.cells.insert(0, nbformat.v4.new_code_cell(_KERNEL_GUARD))
            if publish:
                trial.cells.append(nbformat.v4.new_code_cell(
                    'import json as _checkpoint_json\nprint("__CBAM_CHECKPOINTS__" + _checkpoint_json.dumps(PUBLIC_NUMERICAL_CHECKPOINTS, sort_keys=True, allow_nan=False))'))
            manager = KernelManager(kernel_name='cbam-portfolio', connection_file=str(target/'runtime'/('kernel-'+str(number+1)+'-'+str(repeat)+'.json')),
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(target/'kernels')], ensure_native_kernel=False))
            client = NotebookClient(trial, km=manager, timeout=180, allow_errors=False, record_timing=False, resources={'metadata':{'path':str(work)}})
            try:
                executed = client.execute(cwd=str(work), env=environment)
                outputs = inspect_notebook_outputs(executed)
                if publish:
                    capture = ''.join(output.get('text','') for output in executed.cells[-1].get('outputs',[]) if output.get('output_type')=='stream')
                    require(capture.startswith('__CBAM_CHECKPOINTS__'), 'PORTFOLIO_CHECKPOINTS_MISSING')
                    values = json.loads(capture.removeprefix('__CBAM_CHECKPOINTS__'))
                    require(isinstance(values,dict) and bool(values), 'PORTFOLIO_CHECKPOINTS_MISSING')
                    json.dumps(values,allow_nan=False)
                    release._check_structured_credentials(values)
                    checkpoints.append(values)
                clean = release.notebook_source(json.loads(content))
                index = 0
                for source_cell, result_cell in zip(clean['cells'],executed.cells[1:]):
                    if source_cell['cell_type']=='code':
                        index += 1;source_cell['execution_count']=index
                        source_cell['outputs']=copy.deepcopy(result_cell.get('outputs',[]))
                        for output in source_cell['outputs']:
                            if 'metadata' in output: output['metadata']={}
                            if output.get('output_type')=='execute_result':output['execution_count']=index
                release.scan_notebook_outputs(clean)
                trials.append(clean)
            except Exception:
                raise PortfolioError('PORTFOLIO_NOTEBOOK_EXECUTION_FAILED') from None
            finally:
                if manager.has_kernel: manager.shutdown_kernel(now=True)
        review = None
        if publish:
            require(checkpoints[0]==checkpoints[1], 'PORTFOLIO_CHECKPOINT_REEXECUTION_DIFFERENCE')
            require(release.notebook_outputs_sha256(trials[0])==release.notebook_outputs_sha256(trials[1]), 'PORTFOLIO_OUTPUT_REEXECUTION_DIFFERENCE')
            previous=node.get('execution_review',{}).get('checkpoints')
            if previous is not None: compare_checkpoints(checkpoints[0],previous)
            counts=release.scan_notebook_outputs(trials[0])
            require(counts['png']>=1 and counts['text']>=1,'PORTFOLIO_RICH_OUTPUT_REQUIRED')
            review=dict(artificial_only=True,fresh_kernel_runs=2,program_sha256=release.notebook_program_sha256(trials[0]),
                outputs_sha256=release.notebook_outputs_sha256(trials[0]),checkpoints=checkpoints[0],output_counts=counts)
        # Only canonical original cells are retained; private guards/capture are removed.
        nbformat.write(nbformat.from_dict(trials[0]), work/'executed.ipynb')
        require(content == release._safe_read(root, name, 4*1024*1024), 'PORTFOLIO_NOTEBOOK_SOURCE_CHANGED')
        reports.append(dict(notebook=name, source_sha256=release.sha256(content), cells=len(trials[0]['cells']),
                            outputs_checked=release.scan_notebook_outputs(trials[0])['outputs'], status='PASS',
                            source_outputs_saved=False, execution_review=review))
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-root')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--packages')
    args = parser.parse_args(argv)
    try:
        root, registry, selection, audit = inspect_source(args.source_root)
        installed = inspect_installed_code(root, selection)
        packages = inspect_packages(args.packages, root, registry, selection) if args.packages else []
        require(args.check_only or args.output_root, 'PORTFOLIO_EXTERNAL_OUTPUT_REQUIRED')
        notebooks = [] if args.check_only else execute_notebooks(root, registry, selection, args.output_root)
        result = dict(status='PASS', security='SECURITY_CLEAN', publication_approved=False,
            source_files=len(registry.data['candidate_whitelist']), artificial_artifacts=len(registry.data.get('portfolio_artifacts', {})),
            installed_modules=installed, notebooks=notebooks, packages=packages,
            scientific_replication='NOT_REEXECUTED', git_history='NOT_INSPECTED')
        if notebooks:(Path(args.output_root)/'portfolio_evidence.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, ensure_ascii=False));return 0
    except Exception as error:
        code = str(error) if isinstance(error, PortfolioError) else release.public_error_code(error)
        print(code, file=sys.stderr);return 2


if __name__ == '__main__':
    raise SystemExit(main())
