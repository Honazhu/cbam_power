"""Saved-output reporting with explicit artifact-level publication permission.

No estimator or input provider is called. Local authorized reports stay private.
Public exports require byte identities and complete reviewed rights lineage;
labels alone never establish artificial provenance or permission.
"""
import hashlib
import csv
import io
from html import escape
import json
from pathlib import Path
import tempfile

from .tables import rows, write_table
from .figures import coefficient_svg

DEMO_LABEL = 'SYNTHETIC DATA — NOT EMPIRICAL RESEARCH RESULTS'
OUTPUT_FORMATS = {'table.csv': 'csv', 'coefficients.svg': 'svg',
                  'report.html': 'html', 'numeric_sidecar.json': 'json'}


def _rights_preflight(registry, artifact_id, mode, output_ids, conditions):
    from ..release import RightsRegistry, ReleaseError
    if not isinstance(registry, RightsRegistry) or not artifact_id:
        if mode == 'authorized':
            raise PermissionError('PUBLIC_EXPORT_BLOCKED_PR6_ARTIFACT_RIGHTS_PENDING')
        raise PermissionError('PUBLIC_DEMO_REVIEWED_ARTIFACT_REQUIRED')
    node = registry.node(artifact_id)
    origin = 'ARTIFICIAL' if mode == 'demo' else 'REAL'
    if node['kind'] != 'artifact' or node['origin'] != origin or node.get('format') != 'json':
        raise ReleaseError('REPORT_SOURCE_ARTIFACT_KIND_MISMATCH')
    registry.authorize(artifact_id, satisfied_conditions=conditions)
    if mode == 'authorized':
        if not isinstance(output_ids, dict) or set(output_ids) != set(OUTPUT_FORMATS):
            raise PermissionError('PUBLIC_EXPORT_BLOCKED_PR6_EXACT_OUTPUT_ARTIFACTS_REQUIRED')
        for name, output_id in output_ids.items():
            outnode = registry.node(output_id)
            if (outnode['kind'] != 'artifact' or outnode['origin'] != 'REAL'
                    or outnode.get('format') != OUTPUT_FORMATS[name]
                    or Path(outnode.get('path', '')).name != name):
                raise ReleaseError('REPORT_OUTPUT_ARTIFACT_KIND_MISMATCH')
            proof = registry.authorize(output_id, satisfied_conditions=conditions)
            if artifact_id not in {r['node_id'] for r in proof['lineage']} or output_id == artifact_id:
                raise ReleaseError('REPORT_OUTPUT_SOURCE_LINEAGE_MISMATCH')
    return node


def _render_bytes(records, label, source_sha):
    # Reuse the tested renderers in a private temporary directory. Publication
    # identities are checked before any byte is written to the requested output.
    with tempfile.TemporaryDirectory(prefix='cbam-report-') as temporary:
        temp = Path(temporary)
        write_table(temp/'table.csv', records)
        coefficient_svg(temp/'coefficients.svg', records, label)
        content = {name: (temp/name).read_bytes() for name in ['table.csv', 'coefficients.svg']}
    if label == DEMO_LABEL:
        # A detached CSV must retain the artificial-data label on its own.
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=['label', *records[0]])
        writer.writeheader()
        writer.writerows(dict(label=label, **record) for record in records)
        content['table.csv'] = stream.getvalue().encode('utf-8')
    sidecar = dict(label=label, source_sha256=source_sha, rows=records,
                   display_rounding_only=True, estimation_performed=False)
    content['numeric_sidecar.json'] = (json.dumps(sidecar, indent=2, allow_nan=False)+'\n').encode()
    table = ''.join('<tr>'+''.join(f'<td>{escape(str(r[k]))}</td>' for k in r)+'</tr>' for r in records)
    html = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>'+escape(label)+'</title>'
            '<h1>'+escape(label)+'</h1><p>Saved results only. No estimation during reporting. '
            'This output does not establish full V3 scientific replication.</p>'
            '<img src="coefficients.svg" alt="Saved HAC estimates and confidence intervals">'
            '<table><tr>'+''.join('<th>'+escape(k)+'</th>' for k in records[0])+'</tr>'+table+'</table></html>')
    content['report.html'] = html.encode('utf-8')
    return content


def _public_outputs(registry, source_id, content, mode, output_ids, conditions):
    from ..release import RightsRegistry, scan_content
    if mode == 'demo':
        # Explicitly reviewed artificial authorization covers this presentation
        # only. Research/code ownership is not approved by this derivative node.
        source = registry.node(source_id)
        data = json.loads(json.dumps(registry.data))
        output_ids = {}
        for name, value in content.items():
            digest = hashlib.sha256(value).hexdigest()
            key = 'demo-report:'+name+':'+digest
            output_ids[name] = key
            data['nodes'].append(dict(id=key, kind='artifact', origin='ARTIFICIAL',
                status='APPROVED_PUBLIC', dependencies=[source_id], lineage_complete=True,
                lineage_kind='derived', review_group=source.get('review_group', 'public-demo'),
                evidence=list(source['evidence']), conditions=list(source['conditions']),
                permissions=list(source['permissions']), sha256=digest, path=name,
                format=OUTPUT_FORMATS[name], reason='Presentation of byte-pinned reviewed artificial results only.'))
        registry = RightsRegistry(data)
    proofs = {}
    for name, value in content.items():
        proofs[name] = registry.authorize(output_ids[name], hashlib.sha256(value).hexdigest(),
                                          satisfied_conditions=conditions)
        proofs[name]['security'] = scan_content(value, registry, output_ids[name])
    return proofs


def render_report(result_path, output_dir, *, mode, audience, expected_sha,
                  rights_registry=None, artifact_id=None, output_artifact_ids=None,
                  satisfied_conditions=None):
    if mode not in {'demo', 'authorized'} or audience not in {'local', 'public'}:
        raise ValueError('INVALID_REPORT_MODE')
    if audience == 'public':
        _rights_preflight(rights_registry, artifact_id, mode, output_artifact_ids, satisfied_conditions)
    path = Path(result_path)
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError('REPORT_SOURCE_SYMLINK')
    content = path.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected_sha: raise ValueError('REPORT_RESULT_IDENTITY_CHANGED')
    if audience == 'public': rights_registry.identity(artifact_id, actual)
    payload = json.loads(content)
    if mode == 'demo':
        if audience == 'public':
            from ..release import validate_demo_result_content
            payload = validate_demo_result_content(content, rights_registry, artifact_id,
                                                   satisfied_conditions=satisfied_conditions)
        if payload.get('label') != DEMO_LABEL or payload.get('mode') != 'demo':
            raise PermissionError('DEMO_PROVENANCE_REQUIRED')
        if any(not m['model_id'].startswith('DEMO_') for m in payload['models']):
            raise PermissionError('REAL_MODEL_IN_DEMO')
        label = DEMO_LABEL
    else:
        if payload.get('mode') == 'demo': raise ValueError('DEMO_IS_NOT_AUTHORIZED_RESEARCH')
        label = ('REVIEWED RESEARCH RESULTS' if audience == 'public'
                 else 'PRIVATE RESEARCH RESULTS — release rights unverified')
    records = rows(payload)
    out = Path(output_dir)
    if any(p.is_symlink() for p in [out, *out.parents]):
        raise ValueError('REPORT_OUTPUT_SYMLINK')
    if any((out/n).exists() for n in OUTPUT_FORMATS): raise FileExistsError('REPORT_OUTPUT_EXISTS')
    rendered = _render_bytes(records, label, actual)
    proofs = (_public_outputs(rights_registry, artifact_id, rendered, mode,
                              output_artifact_ids, satisfied_conditions)
              if audience == 'public' else {})
    out.mkdir(parents=True, exist_ok=True)
    for name, value in rendered.items():
        with (out/name).open('xb') as handle: handle.write(value)
    return dict(label=label, models=len(records), source_sha256=actual,
                estimation_performed=False, network_access=False, audience=audience,
                publication_proofs=proofs)
