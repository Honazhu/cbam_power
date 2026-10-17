"""Offline release controls, independent of research computation.

The rights registry is a reviewed policy input, not a license inference engine.
Unknown rights, incomplete lineage, unmet conditions and pending ownership deny
export. Security-clean bytes do not imply legal permission to publish them.
"""
from __future__ import annotations

from dataclasses import dataclass
import ast
import base64
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import xml.etree.ElementTree as ET
import tarfile
import tempfile
import tomllib
import zipfile
import zlib


STATUSES = frozenset({'APPROVED_PUBLIC', 'APPROVED_WITH_CONDITIONS', 'RESTRICTED',
                     'UNVERIFIED', 'PENDING_OWNER_APPROVAL', 'EXCLUDED'})
CONTEXTS = frozenset({'public_repository', 'research_publication', 'commercial_reuse'})
KINDS = frozenset({'asset', 'variable', 'artifact', 'source'})
ORIGINS = frozenset({'REAL', 'ARTIFICIAL', 'SOURCE'})
EVIDENCE_KINDS = frozenset({'license', 'owner_approval', 'publication_approval',
                          'user_demo_authorization', 'condition'})
FORMATS = frozenset({'python', 'toml', 'markdown', 'makefile', 'text', 'json', 'csv',
                    'html', 'svg', 'notebook', 'yaml', 'png', 'zip', 'tar', 'gzip'})
DEMO_LABEL = 'SYNTHETIC DATA — NOT EMPIRICAL RESEARCH RESULTS'
SHA_PATTERN = re.compile(r'^[0-9a-f]{64}$')


_PUBLIC_ERROR_CODES = frozenset("""
ZIP_EXTRA_METADATA_UNREVIEWED
NOTEBOOK_RESULT_COUNT
NOTEBOOK_CHECKPOINT_PROOF_REQUIRED
NOTEBOOK_OUTPUT_COUNTS_CHANGED
HTML_UNBALANCED_STRUCTURE
FINDER_METADATA_EXCLUDED
NOTEBOOK_OUTPUT_SCHEMA
NOTEBOOK_ERROR_OUTPUT
NOTEBOOK_OUTPUT_METADATA
NOTEBOOK_OUTPUT_MIME
NOTEBOOK_OUTPUT_SIZE
NOTEBOOK_PNG_ENCODING
NOTEBOOK_TRACEBACK_OUTPUT
NOTEBOOK_CONTROL_CHARACTER
NOTEBOOK_REEXECUTION_PROOF_REQUIRED
NOTEBOOK_EXECUTION_IDENTITY_CHANGED
NOTEBOOK_ARTIFICIAL_LINEAGE_REQUIRED
NOTEBOOK_DEMO_LABEL_REQUIRED
NOTEBOOK_RICH_OUTPUT_REQUIRED
NOTEBOOK_EXECUTION_COUNT
NOTEBOOK_TEMPORARY_GUARD
REVIEWED_HTML_SITE_REQUIRED
HTML_UNREVIEWED_ELEMENT
HTML_DUPLICATE_ATTRIBUTE
HTML_UNREVIEWED_ATTRIBUTE
HTML_UNREVIEWED_NAMESPACE
HTML_ACTIVE_META
HTML_UNSAFE_URL
HTML_EXTERNAL_RESOURCE
HTML_UNREVIEWED_DECLARATION
HTML_EXTERNAL_STYLE
REVIEWED_NOTEBOOK_SOURCE_REQUIRED
STRUCTURED_BINARY_PAYLOAD
NOTEBOOK_HIDDEN_METADATA
NOTEBOOK_PYTHON_REQUIRED
INVALID_NOTEBOOK_PYTHON_SOURCE
REVIEWED_SCHEMATIC_REQUIRED
UNSAFE_SVG_CONTENT
INVALID_SVG_CONTENT
INVALID_PNG_CONTENT
PNG_CRC_MISMATCH
PNG_UNREVIEWED_METADATA
PNG_UNSUPPORTED_IMAGE_PROFILE
REVIEWED_YAML_SOURCE_REQUIRED
YAML_SCANNER_DEPENDENCY_MISSING
INVALID_YAML_SOURCE_CONTENT
ARTIFICIAL_PNG_REQUIRED
ACTIVE_OR_EMBEDDED_REPORT_CONTENT
ARCHIVE_FORMAT_MASQUERADE
ARCHIVE_MEMBER_LINEAGE_MISSING
ARCHIVE_MEMBER_SET_CHANGED
ARTIFACT_PUBLICATION_APPROVAL_REQUIRED
ARTIFICIAL_AUTHORIZATION_REQUIRED
BINARY_CONTENT_DISGUISED_AS_TEXT
CANDIDATE_COPY_IO_FAILED
CANDIDATE_PREPARATION_NOT_AUTHORIZED
CANDIDATE_SOURCE_FORMAT_ONLY
CANDIDATE_SOURCE_ONLY
CANDIDATE_SOURCE_STATUS_NOT_REVIEWABLE
CONDITIONAL_APPROVAL_WITHOUT_CONDITIONS
CONDITION_EVIDENCE_REQUIRED
CONTEXT_NOT_AUTHORIZED
CREDENTIAL_CONFIGURATION_CONTENT
CREDENTIAL_CONTENT
CREDENTIAL_FILE_NAME
DEMO_ANALYTIC_SOURCE_CHANGED
DEMO_ARTIFACT_BINDING_REQUIRED
DEMO_GENERATOR_CODE_IDENTITY_CHANGED
DEMO_IDENTITY_REQUIRED
DEMO_PRIVATE_INPUT_OR_MODEL_SET_CHANGED
DEMO_PROVENANCE_REQUIRED
DEMO_RECEIPT_ARTIFACT_CHANGED
DEMO_RECEIPT_REQUIRED
DEMO_RESULT_OUTSIDE_REVIEWED_RUN
DEMO_STAGE_ARTIFACT_SET_CHANGED
DEMO_STAGE_SET_CHANGED
DEMO_TOY_SPECIFICATION_CHANGED
DEMO_VALIDATION_NOT_PASSED
DERIVED_LINEAGE_MISSING
DUPLICATE_ARCHIVE_MEMBER
DUPLICATE_EVIDENCE_ID
DUPLICATE_NODE_ID
ENCRYPTED_ARCHIVE_UNSCANNABLE
EVIDENCE_SOURCE_MISSING
EXACT_SOURCE_CANDIDATE_CLOSURE_REQUIRED
EXPLICIT_RELEASE_WHITELIST_REQUIRED
FROZEN_VINTAGE_LICENSE_UNVERIFIED
GZIP_SINGLE_MEMBER_IDENTITY_REQUIRED
INCOMPLETE_DEMO_CODE_PROVENANCE
INCOMPLETE_OR_CHANGED_DEMO_STAGE
INCOMPLETE_PUBLICATION_LINEAGE
INSTALLED_DEMO_CODE_IDENTITY_CHANGED
INVALID_ARCHIVE_REGISTRY
INVALID_CANDIDATE_PREPARATION_AUTHORIZATION
INVALID_CONDITION_EVIDENCE
INVALID_DEMO_RESULT_CONTENT
INVALID_EVIDENCE_APPLICABILITY
INVALID_EVIDENCE_ID
INVALID_EVIDENCE_KIND
INVALID_EVIDENCE_LIST
INVALID_EVIDENCE_PERMISSIONS
INVALID_EVIDENCE_SCOPE
INVALID_EVIDENCE_STATUS
INVALID_JSON_CONTENT
INVALID_LINEAGE_KIND
INVALID_NODE_CONDITIONS
INVALID_NODE_EVIDENCE
INVALID_NODE_ID
INVALID_NODE_LIST
INVALID_NODE_PERMISSIONS
INVALID_PUBLICATION_SCOPE
INVALID_PYTHON_SOURCE_CONTENT
INVALID_RELEASE_CONTEXT
INVALID_RELEASE_DESTINATION
INVALID_RELEASE_FORMAT
INVALID_RELEASE_PATH
INVALID_RIGHTS_DEPENDENCIES
INVALID_RIGHTS_NODE_KIND
INVALID_RIGHTS_SHA256
INVALID_RIGHTS_STATUS
INVALID_TOML_SOURCE_CONTENT
LINEAGE_COMPLETENESS_UNSPECIFIED
MALFORMED_NOTEBOOK
MISSING_EVIDENCE_REFERENCE
MISSING_RIGHTS_DEPENDENCY
NOTEBOOK_EMBEDDED_PAYLOAD
NOTEBOOK_EXECUTION_OUTPUTS
OWNER_APPROVAL_REQUIRED
PERSONAL_ABSOLUTE_PATH
PRIVATE_ENDPOINT_CONTENT
REAL_DATA_ARTIFACT_NOT_RELEASE_SUPPORTED
REAL_DATA_EXCLUDED_FROM_RELEASE
REAL_INPUT_IDENTITY_IN_DEMO
REAL_MODEL_IN_DEMO
REAL_OR_UNREVIEWED_SOURCE_IN_DEMO
RELEASE_CONTENT_IDENTITY_CHANGED
RELEASE_DESTINATION_APPEARED
RELEASE_DESTINATION_MUST_BE_NEW
RELEASE_DESTINATION_SYMLINK
RELEASE_OPERATION_FAILED
RELEASE_PATH_IDENTITY_CHANGED
RELEASE_PROTECTED_ROOT_OVERLAP
RELEASE_SCAN_RECURSION_LIMIT
RELEASE_SCAN_SIZE_LIMIT
RELEASE_SOURCE_CHANGED_DURING_OPEN
RELEASE_SOURCE_NOT_INDEPENDENT_FILE
RELEASE_SOURCE_SYMLINK
RELEASE_SOURCE_UNAVAILABLE
RELEASE_STAGING_IO_FAILED
REPORT_OUTPUT_ARTIFACT_KIND_MISMATCH
REPORT_OUTPUT_SOURCE_LINEAGE_MISMATCH
REPORT_SOURCE_ARTIFACT_KIND_MISMATCH
RESTRICTED_BINARY_MAGIC
REVIEWED_ARTIFICIAL_GENERATOR_REQUIRED
REVIEWED_DEMO_REGISTRY_REQUIRED
RIGHTS_DEPENDENCY_CYCLE
RIGHTS_NOT_APPROVED
RIGHTS_REASON_MISSING
RIGHTS_REGISTRY_INVALID
RIGHTS_REGISTRY_UNREADABLE
SCAN_BYTES_REQUIRED
SIGNED_URL_CREDENTIAL_CONTENT
SOURCE_OWNERSHIP_CLASSIFICATION_REQUIRED
STRUCTURED_CONTENT_MASQUERADE
STRUCTURED_SCAN_COMPLEXITY_LIMIT
TABULAR_CONTENT_MASQUERADE
UNBOUNDED_EVIDENCE_APPROVAL
UNKNOWN_RELEASE_NODE
UNPINNED_RELEASE_BYTES
UNREADABLE_RELEASE_ARCHIVE
UNREGISTERED_ARCHIVE_MEMBER
UNREVIEWED_ARCHIVE_CONTENTS
UNREVIEWED_BINARY_CONTENT
UNREVIEWED_CONTENT_FORMAT
UNSAFE_ARCHIVE_MEMBER
UNSATISFIED_RELEASE_CONDITION
""".split())


class ReleaseError(PermissionError):
    """A stable redacted denial code, with no rejected payload or path value."""

    def __init__(self, message):
        # Node/condition identifiers can themselves be attacker-controlled.
        # Preserve the rule identifier, never append their contents to errors.
        candidate = str(message).split(':', 1)[0]
        self.code = candidate if candidate in _PUBLIC_ERROR_CODES else 'RELEASE_OPERATION_FAILED'
        super().__init__(self.code)


def public_error_code(error):
    """Safe diagnostic for release CLI boundaries, including parser/I/O errors."""
    return error.code if isinstance(error, ReleaseError) and error.code in _PUBLIC_ERROR_CODES else 'RELEASE_OPERATION_FAILED'


def require(condition, code):
    if not condition:
        raise ReleaseError(code)


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def relative(value):
    require(isinstance(value, str) and bool(value), 'INVALID_RELEASE_PATH')
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.as_posix() == value and bool(path.parts)
            and all(p not in {'.', '..'} for p in path.parts)
            and not any(c in value for c in '\\:*?[]')
            and all(ord(c) >= 32 for c in value), 'INVALID_RELEASE_PATH')
    return path


def _strings(value, code):
    require(isinstance(value, list) and all(isinstance(v, str) and v for v in value)
            and len(value) == len(set(value)), code)
    return value


class RightsRegistry:
    """Validated transitive rights graph over files, variables and artifacts.

    Construct from a TOML path or an already parsed mapping. Registry/evidence
    entries must have been reviewed by the responsible human; this class only
    enforces their explicit permissions, identities and conditions.
    """

    def __init__(self, source):
        if isinstance(source, (str, os.PathLike)):
            try:
                with Path(source).open('rb') as handle:
                    source = tomllib.load(handle)
            except (OSError, ValueError):
                raise ReleaseError('RIGHTS_REGISTRY_UNREADABLE') from None
        require(isinstance(source, dict), 'RIGHTS_REGISTRY_INVALID')
        # Freeze a detached JSON-compatible copy so later caller mutations do
        # not alter a policy object after its graph has been checked.
        self.data = json.loads(json.dumps(source, allow_nan=False))
        require(self.data.get('publication_scope') in {None, 'SOURCE_AND_ARTIFICIAL_ONLY'}, 'INVALID_PUBLICATION_SCOPE')
        require(type(self.data.get('candidate_preparation_authorized', False)) is bool, 'INVALID_CANDIDATE_PREPARATION_AUTHORIZATION')
        self.nodes = self._index(self.data.get('nodes'), 'NODE')
        self.evidence = self._index(self.data.get('evidence', []), 'EVIDENCE')
        for record in self.evidence.values():
            record.setdefault('permissions', [])
            require(record.get('kind') in EVIDENCE_KINDS, 'INVALID_EVIDENCE_KIND')
            require(record.get('status') in {'VERIFIED', 'UNVERIFIED'}, 'INVALID_EVIDENCE_STATUS')
            require(isinstance(record.get('source'), str) and bool(record['source'].strip()), 'EVIDENCE_SOURCE_MISSING')
            scopes = _strings(record.get('scope'), 'INVALID_EVIDENCE_SCOPE')
            require(scopes and '*' not in scopes, 'UNBOUNDED_EVIDENCE_APPROVAL')
            require(record.get('applicability') in {'FROZEN_VINTAGE_CONFIRMED', 'CURRENT_ONLY', 'NOT_APPLICABLE'},
                    'INVALID_EVIDENCE_APPLICABILITY')
            permissions = _strings(record.get('permissions', []), 'INVALID_EVIDENCE_PERMISSIONS')
            require(set(permissions) <= CONTEXTS, 'INVALID_EVIDENCE_PERMISSIONS')
        for node in self.nodes.values():
            for field in ['evidence', 'conditions', 'permissions']:
                node.setdefault(field, [])
            require(node.get('kind') in KINDS and node.get('origin') in ORIGINS, 'INVALID_RIGHTS_NODE_KIND')
            require(node.get('status') in STATUSES, 'INVALID_RIGHTS_STATUS')
            require(type(node.get('lineage_complete')) is bool, 'LINEAGE_COMPLETENESS_UNSPECIFIED')
            require(node.get('lineage_kind') in {'raw', 'derived', 'artificial_generator', 'source'}, 'INVALID_LINEAGE_KIND')
            require(isinstance(node.get('reason'), str) and bool(node['reason'].strip()), 'RIGHTS_REASON_MISSING')
            dependencies = _strings(node.get('dependencies'), 'INVALID_RIGHTS_DEPENDENCIES')
            require(set(dependencies) <= self.nodes.keys(), 'MISSING_RIGHTS_DEPENDENCY')
            records = _strings(node.get('evidence', []), 'INVALID_NODE_EVIDENCE')
            require(set(records) <= self.evidence.keys(), 'MISSING_EVIDENCE_REFERENCE')
            _strings(node.get('conditions', []), 'INVALID_NODE_CONDITIONS')
            permissions = _strings(node.get('permissions', []), 'INVALID_NODE_PERMISSIONS')
            require(set(permissions) <= CONTEXTS, 'INVALID_NODE_PERMISSIONS')
            if node.get('sha256') not in (None, 'UNVERIFIED'):
                require(isinstance(node['sha256'], str) and SHA_PATTERN.fullmatch(node['sha256']), 'INVALID_RIGHTS_SHA256')
            if 'path' in node and node['path']:
                relative(node['path'])
            if 'format' in node:
                # Raw/restricted formats remain describable in the inventory;
                # scan_content has a narrower explicit publication allowlist.
                require(isinstance(node['format'], str) and re.fullmatch(r'[A-Za-z0-9_.+-]+', node['format']),
                        'INVALID_RELEASE_FORMAT')
            members = node.get('members', {})
            require(isinstance(members, dict), 'INVALID_ARCHIVE_REGISTRY')
            for member, child in members.items():
                relative(member)
                require(child in self.nodes and child in dependencies, 'ARCHIVE_MEMBER_LINEAGE_MISSING')
        self._check_cycles()

    @staticmethod
    def _index(records, kind):
        require(isinstance(records, list), 'INVALID_'+kind+'_LIST')
        out = {}
        for record in records:
            require(isinstance(record, dict) and isinstance(record.get('id'), str)
                    and bool(record['id']), 'INVALID_'+kind+'_ID')
            require(record['id'] not in out, 'DUPLICATE_'+kind+'_ID')
            out[record['id']] = record
        return out

    def _check_cycles(self):
        visited, visiting = set(), set()
        def visit(key):
            require(key not in visiting, 'RIGHTS_DEPENDENCY_CYCLE')
            if key in visited:
                return
            visiting.add(key)
            for child in self.nodes[key]['dependencies']:
                visit(child)
            visiting.remove(key)
            visited.add(key)
        for key in self.nodes:
            visit(key)

    def node(self, key):
        require(key in self.nodes, 'UNKNOWN_RELEASE_NODE')
        return self.nodes[key]

    def identity(self, key, content_sha256):
        node = self.node(key)
        require(isinstance(node.get('sha256'), str) and SHA_PATTERN.fullmatch(node['sha256']), 'UNPINNED_RELEASE_BYTES')
        require(node['sha256'] == content_sha256, 'RELEASE_CONTENT_IDENTITY_CHANGED')
        return node

    @staticmethod
    def _scope(record, node):
        return node['id'] in record['scope'] or ('review_group' in node
                and 'group:'+node['review_group'] in record['scope'])

    def _evidence_for(self, node, kind, context):
        return [self.evidence[key] for key in node['evidence']
                if self.evidence[key]['kind'] == kind and self.evidence[key]['status'] == 'VERIFIED'
                and self._scope(self.evidence[key], node)
                and context in self.evidence[key]['permissions']]

    def authorize(self, node_id, content_sha256=None, *, context='public_repository', satisfied_conditions=None):
        require(context in CONTEXTS, 'INVALID_RELEASE_CONTEXT')
        conditions = {} if satisfied_conditions is None else satisfied_conditions
        require(isinstance(conditions, dict) and all(isinstance(k, str) and isinstance(v, str)
                for k, v in conditions.items()), 'CONDITION_EVIDENCE_REQUIRED')
        if content_sha256 is not None:
            self.identity(node_id, content_sha256)
        visited, proof = set(), []
        def visit(key):
            if key in visited:
                return
            node = self.node(key)
            require(not (self.data.get('publication_scope') == 'SOURCE_AND_ARTIFICIAL_ONLY'
                         and node['origin'] == 'REAL'), 'REAL_DATA_EXCLUDED_FROM_RELEASE')
            require(node['status'] in {'APPROVED_PUBLIC', 'APPROVED_WITH_CONDITIONS'}, 'RIGHTS_NOT_APPROVED:'+key)
            require(context in node['permissions'], 'CONTEXT_NOT_AUTHORIZED:'+key)
            require(node['lineage_complete'], 'INCOMPLETE_PUBLICATION_LINEAGE:'+key)
            dependencies = node['dependencies']
            if not dependencies:
                permitted_leaf = (node['kind'] == 'asset' and node['lineage_kind'] == 'raw'
                                  or node['kind'] == 'source' and node['lineage_kind'] == 'source'
                                  or node['origin'] == 'ARTIFICIAL' and node['lineage_kind'] == 'artificial_generator')
                require(permitted_leaf, 'DERIVED_LINEAGE_MISSING:'+key)
            if node['origin'] == 'SOURCE':
                require(node['kind'] == 'source' and self._evidence_for(node, 'owner_approval', context),
                        'OWNER_APPROVAL_REQUIRED:'+key)
            elif node['origin'] == 'ARTIFICIAL':
                require(self._evidence_for(node, 'user_demo_authorization', context), 'ARTIFICIAL_AUTHORIZATION_REQUIRED:'+key)
                require(all(self.node(child)['origin'] == 'ARTIFICIAL' for child in dependencies),
                        'REAL_OR_UNREVIEWED_SOURCE_IN_DEMO:'+key)
            else:
                if node['kind'] == 'asset' and node['lineage_kind'] == 'raw':
                    licenses = self._evidence_for(node, 'license', context)
                    require(any(r['applicability'] == 'FROZEN_VINTAGE_CONFIRMED' for r in licenses),
                            'FROZEN_VINTAGE_LICENSE_UNVERIFIED:'+key)
                if node['kind'] == 'artifact':
                    require(self._evidence_for(node, 'publication_approval', context), 'ARTIFACT_PUBLICATION_APPROVAL_REQUIRED:'+key)
            if node['status'] == 'APPROVED_WITH_CONDITIONS':
                require(bool(node['conditions']), 'CONDITIONAL_APPROVAL_WITHOUT_CONDITIONS:'+key)
            for condition in node['conditions']:
                evidence_id = conditions.get(condition)
                require(evidence_id in self.evidence, 'UNSATISFIED_RELEASE_CONDITION:'+condition)
                record = self.evidence[evidence_id]
                require(record['kind'] == 'condition' and record['status'] == 'VERIFIED'
                        and record.get('condition_id') == condition and self._scope(record, node)
                        and context in record['permissions'], 'INVALID_CONDITION_EVIDENCE:'+condition)
            for child in dependencies:
                visit(child)
            visited.add(key)
            proof.append(dict(node_id=key, status=node['status'], origin=node['origin'],
                              evidence=list(node['evidence']),
                              satisfied_conditions={c:conditions[c] for c in node['conditions']}))
        visit(node_id)
        return dict(status='APPROVED', context=context, node_id=node_id, lineage=proof)


def validate_demo_result_content(content, registry, node_id, *, context='public_repository', satisfied_conditions=None):
    """Validate a byte-pinned, explicitly authorized artificial saved result.

    A caller must independently bind any runtime registry to its artificial
    generator, receipt and output identities. A payload's own mode is no grant.
    """
    node = registry.identity(node_id, sha256(content))
    require(node['origin'] == 'ARTIFICIAL' and node['kind'] == 'artifact', 'DEMO_ARTIFACT_BINDING_REQUIRED')
    registry.authorize(node_id, sha256(content), context=context, satisfied_conditions=satisfied_conditions)
    try:
        payload = json.loads(content)
    except (ValueError, UnicodeError) as error:
        raise ReleaseError('INVALID_DEMO_RESULT_CONTENT') from None
    require(isinstance(payload, dict) and payload.get('label') == DEMO_LABEL and payload.get('mode') == 'demo',
            'DEMO_PROVENANCE_REQUIRED')
    models = payload.get('models')
    require(isinstance(models, list) and bool(models) and all(isinstance(m, dict)
            and isinstance(m.get('model_id'), str) and m['model_id'].startswith('DEMO_') for m in models),
            'REAL_MODEL_IN_DEMO')
    return payload


def demo_registry_for_run(result_path, receipt, run_root, base_registry):
    """Bind public-demo permission to complete, byte-checked artificial stages.

    This verifies saved computation evidence without recomputing an estimator.
    It does not treat a mutable receipt as a cryptographic attestation against
    a malicious local operator. A user-supplied mode/label alone never suffices.
    """
    require(isinstance(base_registry, RightsRegistry), 'REVIEWED_DEMO_REGISTRY_REQUIRED')
    generator = base_registry.node('demo-generator')
    require(generator['origin'] == 'ARTIFICIAL' and generator['lineage_kind'] == 'artificial_generator',
            'REVIEWED_ARTIFICIAL_GENERATOR_REQUIRED')
    base_registry.authorize('demo-generator')
    require(isinstance(receipt, dict) and receipt.get('mode') == 'demo'
            and receipt.get('label') == DEMO_LABEL and receipt.get('rights') == 'ARTIFICIAL_DEMO_ONLY',
            'DEMO_RECEIPT_REQUIRED')
    identity = receipt.get('identity', {})
    require(identity.get('mode') == 'demo', 'DEMO_IDENTITY_REQUIRED')
    approved_code = generator.get('generator_code_hashes')
    require(isinstance(approved_code, dict) and approved_code
            and identity.get('code') == approved_code, 'DEMO_GENERATOR_CODE_IDENTITY_CHANGED')
    required_code = {'_demo.py', 'pipeline.py', 'market.py', 'panel.py', 'weather.py',
                     'synthetic.py', 'design.py', 'estimators.py', '_engine_context.py'}
    require(required_code <= approved_code.keys()
            and approved_code['_demo.py'] == generator['sha256'], 'INCOMPLETE_DEMO_CODE_PROVENANCE')
    package = Path(__file__).parent
    for name, expected in approved_code.items():
        content = _safe_read(package, name, 4*1024*1024)
        require(sha256(content) == expected, 'INSTALLED_DEMO_CODE_IDENTITY_CHANGED')
    require(not identity.get('input_assets') and not identity.get('authorized_inputs')
            and not receipt.get('input_root'), 'REAL_INPUT_IDENTITY_IN_DEMO')
    stages = receipt.get('stages', {})
    files = {
        'validate-inputs': {'fixture.json'}, 'build-weather': {'weather.parquet'},
        'build-panel': {'panel.parquet'}, 'build-synthetic': {'synthetic.parquet'},
        'estimate-main': {'results.json', 'coefficients.csv', 'covariance.npz'},
        'verify-main': {'verification.json'},
    }
    require(set(files) <= set(stages) and set(stages) <= {*files, 'report'}, 'DEMO_STAGE_SET_CHANGED')
    expected_identity = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                                  allow_nan=False).encode()).hexdigest()
    verified_bytes = {}
    for stage, names in files.items():
        record = stages[stage]
        require(record.get('status') == 'PASS' and record.get('identity') == expected_identity
                and record.get('parameters') == {}, 'INCOMPLETE_OR_CHANGED_DEMO_STAGE')
        expected_paths = {stage+'/'+name for name in names}
        require(set(record.get('outputs', {})) == expected_paths, 'DEMO_STAGE_ARTIFACT_SET_CHANGED')
        for name in expected_paths:
            content = _safe_read(run_root, name, ScanLimits().max_file_bytes)
            require(sha256(content) == record['outputs'][name], 'DEMO_RECEIPT_ARTIFACT_CHANGED')
            verified_bytes[name] = content
    root = Path(run_root).absolute()
    require(Path(result_path).absolute() == root/'estimate-main/results.json', 'DEMO_RESULT_OUTSIDE_REVIEWED_RUN')
    fixture = json.loads(verified_bytes['validate-inputs/fixture.json'])
    require(fixture.get('label') == DEMO_LABEL and fixture.get('rows') == 1920
            and fixture.get('source') == 'analytic artificial equations' and fixture.get('target_jump') == 4.
            and fixture.get('weights') == {'GR_IT': .25, 'AT_IT': .75}
            and fixture.get('has_private_inputs') is False, 'DEMO_ANALYTIC_SOURCE_CHANGED')
    validation = json.loads(verified_bytes['verify-main/verification.json'])
    require(validation.get('status') == 'PASS' and validation.get('label') == DEMO_LABEL
            and validation.get('known_jump') == 4. and validation.get('full_scientific_replication') is False
            and isinstance(validation.get('absolute_error'), (int, float))
            and validation['absolute_error'] < 1e-9, 'DEMO_VALIDATION_NOT_PASSED')
    content = verified_bytes['estimate-main/results.json']
    payload = json.loads(content)
    require(payload.get('private_inputs_read') == 0 and len(payload.get('models', [])) == 1,
            'DEMO_PRIVATE_INPUT_OR_MODEL_SET_CHANGED')
    model = payload['models'][0]
    expected_columns = ['const', 'post', 'r', 'post_r'] + ['hour_'+str(x) for x in range(1,24)] + ['weekday_'+str(x) for x in range(1,7)]
    require(model.get('model_id') == 'DEMO_RDIT_ME' and model.get('N') == 480
            and model.get('coefficient_name') == 'post' and model.get('status') == 'ARTIFICIAL_ESTIMATE'
            and model.get('columns') == expected_columns and abs(model.get('estimate', float('inf'))-4.) < 1e-9,
            'DEMO_TOY_SPECIFICATION_CHANGED')
    result_id = 'demo-result:'+sha256(content)
    data = json.loads(json.dumps(base_registry.data))
    data['nodes'].append(dict(id=result_id, kind='artifact', origin='ARTIFICIAL',
        status='APPROVED_PUBLIC', dependencies=['demo-generator'], lineage_complete=True, lineage_kind='derived',
        review_group=generator.get('review_group', 'public-demo'), evidence=list(generator['evidence']),
        conditions=list(generator['conditions']), permissions=list(generator['permissions']),
        sha256=sha256(content), path='estimate-main/results.json', format='json',
        reason='All six artificial-stage receipts and saved artifacts verified against the reviewed generator identity.'))
    registry = RightsRegistry(data)
    validate_demo_result_content(content, registry, result_id)
    return registry, result_id


@dataclass(frozen=True)
class ScanLimits:
    max_file_bytes: int = 32 * 1024 * 1024
    max_total_bytes: int = 128 * 1024 * 1024
    max_members: int = 500
    max_depth: int = 4
    max_compression_ratio: int = 200


class _Budget:
    def __init__(self, limits):
        self.limits, self.total, self.members = limits, 0, 0
    def charge(self, size, depth):
        self.total += size
        self.members += 1
        require(size <= self.limits.max_file_bytes and self.total <= self.limits.max_total_bytes,
                'RELEASE_SCAN_SIZE_LIMIT')
        require(self.members <= self.limits.max_members and depth <= self.limits.max_depth,
                'RELEASE_SCAN_RECURSION_LIMIT')


_PRIVATE_PATH = re.compile(r'(?:/(?:Users|home)/[A-Za-z0-9_.-]+/|[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+\\)')
_SECRET = re.compile(r'(?:-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|'
                     r'\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}\b|\bsk-[A-Za-z0-9]{20,}\b|\bxox[baprs]-[A-Za-z0-9-]{12,}|'
                     r'(?i:Bearer\s+[A-Za-z0-9._~+/-]{20,})|'
                     r'(?i:(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*[:=]\s*["\'][^"\'\s]{8,}["\']))')
_SIGNED_URL = re.compile(r'(?i)[?&](?:x-amz-(?:signature|credential|security-token)|'
                         r'x-goog-(?:signature|credential)|access_token|api_key|token|sig|signature)='
                         r'[^&#\s"\'<>]+')
_CREDENTIAL_KEYS = frozenset({'entsoetoken', 'entsoeapitoken', 'cdsapikey', 'cdsapirc',
    'apikey', 'accesstoken', 'refreshtoken', 'idtoken', 'clientsecret', 'password',
    'authorization', 'bearertoken', 'privatekey', 'personalaccesstoken', 'securitytoken',
    'awsaccesskeyid', 'awssecretaccesskey', 'secretaccesskey', 'connectionstring', 'token'})
_NAMED_CREDENTIAL = re.compile(
    r'(?im)(?<![A-Za-z0-9_])["\']?(?:ENTSOE_(?:API_)?TOKEN|CDSAPI_(?:KEY|RC)|'
    r'api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|authorization)'
    r'["\']?\s*[:=]\s*(?:["\'][^"\'\r\n]+["\']|'
    r'[-A-Za-z0-9_./:+%@=\\]+(?=\s*(?:[,;}\]]|\#.*$|$)))')
_AUTH_HEADER = re.compile(r'(?im)\bauthorization["\']?\s*:\s*["\']?(?:Basic|Bearer)\s+[^\s"\'<>]+')
_URL_USERINFO = re.compile(r'(?i)\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@')
_PRIVATE_HOST = re.compile(r'(?i)^(?:localhost|0\.0\.0\.0|127(?:\.\d{1,3}){3}|10(?:\.\d{1,3}){3}|'
    r'192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|'
    r'169\.254(?:\.\d{1,3}){2}|\[?::1\]?|[a-z0-9.-]+\.(?:internal|local))$')
_URL_HOST = re.compile(r'(?i)\b[a-z][a-z0-9+.-]*://(?:[^\s/@]+@)?(\[[^\]]+\]|[^\s/:?#"\'<>]+)')
_MAGIC = (b'PAR1', b'\x89HDF\r\n\x1a\n', b'CDF\x01', b'CDF\x02', b'CDF\x05', b'\x93NUMPY',
          b'SQLite format 3\x00', b'%PDF-', b'\x89PNG\r\n\x1a\n', b'\xff\xd8\xff', b'GIF87a', b'GIF89a',
          b'\x80\x04', b'\x80\x05')


def _check_credential_name(name):
    if not name:
        return
    parts = PurePosixPath(name).parts
    require(not any(part.lower() in {'.ds_store','__macosx'} or part.startswith('._') for part in parts), 'FINDER_METADATA_EXCLUDED')
    leaf = PurePosixPath(name).name.lower()
    require(not (leaf == '.env' or leaf.startswith('.env.') or leaf == '.cdsapirc'
                 or leaf.startswith('.cdsapirc.') or leaf == '.netrc'
                 or leaf in {'credentials.json', 'connection.json', 'connections.json'}
                 or re.fullmatch(r'(?:kernel(?:-.*)?|jpserver-.*|nbserver-.*)\.json', leaf)),
            'CREDENTIAL_FILE_NAME')


def _check_credentials(text, *, contextual_cds=True):
    require(not _PRIVATE_PATH.search(text), 'PERSONAL_ABSOLUTE_PATH')
    require(not _SECRET.search(text), 'CREDENTIAL_CONTENT')
    require(not _SIGNED_URL.search(text), 'SIGNED_URL_CREDENTIAL_CONTENT')
    require(not (_NAMED_CREDENTIAL.search(text) or _AUTH_HEADER.search(text)
                 or _URL_USERINFO.search(text)), 'CREDENTIAL_CONFIGURATION_CONTENT')
    require(not any(_PRIVATE_HOST.fullmatch(match.group(1)) for match in _URL_HOST.finditer(text)),
            'PRIVATE_ENDPOINT_CONTENT')
    if contextual_cds and re.search(r'cdsapi|cds\.climate\.copernicus\.eu|ads\.atmosphere\.copernicus\.eu', text, re.I):
        require(not re.search(r'(?im)^\s*key\s*[:=]\s*\S+', text), 'CREDENTIAL_CONFIGURATION_CONTENT')


def _check_structured_credentials(value):
    # JSON escapes must not hide sensitive key names or connection settings.
    # Values are never included in errors or returned risk diagnostics.
    pending = [(value, 0)]
    checked = 0
    while pending:
        current, depth = pending.pop()
        checked += 1
        require(depth <= 64 and checked <= 200000, 'STRUCTURED_SCAN_COMPLEXITY_LIMIT')
        if isinstance(current, dict):
            cds_context = any(isinstance(v, str) and re.search(
                r'cdsapi|cds\.climate\.copernicus\.eu|ads\.atmosphere\.copernicus\.eu', v, re.I)
                for v in current.values())
            for key, child in current.items():
                _check_credentials(str(key))
                normalized = re.sub(r'[^a-z0-9]', '', str(key).lower())
                sensitive = normalized in _CREDENTIAL_KEYS or cds_context and normalized == 'key'
                require(not (sensitive and isinstance(child, str) and child.strip()), 'CREDENTIAL_CONFIGURATION_CONTENT')
                if normalized in {'host', 'hostname', 'server', 'endpoint', 'baseurl', 'url'} and isinstance(child, str):
                    require(not _PRIVATE_HOST.fullmatch(child.strip()), 'PRIVATE_ENDPOINT_CONTENT')
                pending.append((child, depth+1))
        elif isinstance(current, list):
            pending.extend((child, depth+1) for child in current)
        elif isinstance(current, str):
            _check_credentials(current)
        elif isinstance(current, (bytes, bytearray)):
            raise ReleaseError('STRUCTURED_BINARY_PAYLOAD')


def notebook_source(document):
    """Canonical, guard-free program identity; exclude all execution state."""
    import copy
    result = copy.deepcopy(document)
    result['metadata'] = {'kernelspec': {'name':'python3', 'display_name':'Python 3', 'language':'python'},
                          'language_info': {'name':'python'}}
    for cell in result['cells']:
        cell['metadata'] = {}
        if isinstance(cell['source'], list): cell['source'] = ''.join(cell['source'])
        if cell['cell_type'] == 'code': cell.update(execution_count=None, outputs=[])
    return result


def notebook_program_sha256(document):
    return sha256(json.dumps(notebook_source(document),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())


def notebook_outputs_sha256(document):
    """Bind output content independently of notebook JSON multiline encoding.

    nbformat writes multiline text as a list of lines and reads it as one
    string. Normalize only those specified text/PNG fields, preserving every
    character, output order, count and metadata value in the identity.
    """
    import copy
    outputs = copy.deepcopy([c.get('outputs',[]) for c in document['cells']])
    for cell in outputs:
        for output in cell:
            if isinstance(output.get('text'),list) and all(isinstance(v,str) for v in output['text']):
                output['text']=''.join(output['text'])
            for mime,value in output.get('data',{}).items():
                if mime in {'text/plain','image/png'} and isinstance(value,list) and all(isinstance(v,str) for v in value):
                    output['data'][mime]=''.join(value)
    return sha256(json.dumps(outputs,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())


def scan_notebook_outputs(document):
    """Strict text/PNG only; HTML and JavaScript are deliberately unsupported."""
    import base64
    counts = {'outputs':0, 'png':0, 'text':0}
    total = 0
    for cell in document['cells']:
        for output in cell.get('outputs',[]):
            require(isinstance(output,dict), 'NOTEBOOK_OUTPUT_SCHEMA')
            kind = output.get('output_type')
            require(kind in {'stream','display_data','execute_result'}, 'NOTEBOOK_ERROR_OUTPUT')
            keys = {'output_type','name','text'} if kind=='stream' else {'output_type','data','metadata'} | ({'execution_count'} if kind=='execute_result' else set())
            require(set(output)<=keys and output.get('metadata',{})=={}, 'NOTEBOOK_OUTPUT_METADATA')
            if kind=='execute_result':
                require(type(output.get('execution_count')) is int and output['execution_count']==cell.get('execution_count'), 'NOTEBOOK_RESULT_COUNT')
            if kind=='stream':
                require(output.get('name') in {'stdout','stderr'}, 'NOTEBOOK_OUTPUT_SCHEMA')
                values = {'text/plain':output.get('text','')}
            else:
                values = output.get('data');require(isinstance(values,dict) and values and set(values)<={'text/plain','image/png'}, 'NOTEBOOK_OUTPUT_MIME')
            for mime,value in values.items():
                value=''.join(value) if isinstance(value,list) and all(isinstance(x,str) for x in value) else value
                require(isinstance(value,str), 'NOTEBOOK_OUTPUT_SCHEMA')
                total+=len(value);require(total<=4*1024*1024, 'NOTEBOOK_OUTPUT_SIZE')
                if mime=='image/png':
                    try: content=base64.b64decode(value,validate=True)
                    except ValueError: raise ReleaseError('NOTEBOOK_PNG_ENCODING') from None
                    _scan_png(content);counts['png']+=1
                else:
                    _check_credentials(value)
                    require('__CBAM_CHECKPOINTS__' not in value, 'NOTEBOOK_TEMPORARY_GUARD')
                    require(not re.search(r'Traceback \(most recent call last\)|Cell In\[\d+\]|File ["\'].*line \d+',value), 'NOTEBOOK_TRACEBACK_OUTPUT')
                    require(not any(ord(c)<32 and c not in '\n\r\t' for c in value),'NOTEBOOK_CONTROL_CHARACTER')
                    counts['text']+=1
            counts['outputs']+=1
    return counts


def _scan_notebook(document, node):
    """Accept only explicit source or reviewed artificial execution profiles."""
    executed = node.get('content_role') == 'REVIEWED_ARTIFICIAL_NOTEBOOK'
    require(node['kind'] == 'source' and node['origin'] == 'SOURCE'
            and node.get('content_role') in {'UNEXECUTED_NOTEBOOK','REVIEWED_ARTIFICIAL_NOTEBOOK'}, 'REVIEWED_NOTEBOOK_SOURCE_REQUIRED')
    require(isinstance(document, dict) and set(document) <= {'nbformat', 'nbformat_minor', 'metadata', 'cells'}
            and document.get('nbformat') == 4 and type(document.get('nbformat_minor')) is int
            and 0 <= document['nbformat_minor'] <= 5, 'MALFORMED_NOTEBOOK')
    metadata = document.get('metadata', {})
    require(isinstance(metadata, dict) and set(metadata) <= {'kernelspec', 'language_info'}, 'NOTEBOOK_HIDDEN_METADATA')
    for field, allowed in [('kernelspec', {'name', 'display_name', 'language'}),
                           ('language_info', {'name', 'version'})]:
        value = metadata.get(field, {})
        require(isinstance(value, dict) and set(value) <= allowed
                and all(isinstance(v, str) for v in value.values()), 'NOTEBOOK_HIDDEN_METADATA')
    require(metadata.get('kernelspec', {}).get('language', 'python') == 'python'
            and metadata.get('language_info', {}).get('name', 'python') == 'python', 'NOTEBOOK_PYTHON_REQUIRED')
    cells = document.get('cells')
    require(isinstance(cells, list) and 0 < len(cells) <= 200, 'MALFORMED_NOTEBOOK')
    if executed:
        review=node.get('execution_review',{})
        require(review.get('artificial_only') is True and type(review.get('fresh_kernel_runs')) is int and review['fresh_kernel_runs']>=2, 'NOTEBOOK_REEXECUTION_PROOF_REQUIRED')
        require(review.get('program_sha256')==notebook_program_sha256(document) and review.get('outputs_sha256')==notebook_outputs_sha256(document), 'NOTEBOOK_EXECUTION_IDENTITY_CHANGED')
        require(node.get('dependencies'), 'NOTEBOOK_ARTIFICIAL_LINEAGE_REQUIRED')
        require(DEMO_LABEL in json.dumps(document,ensure_ascii=False), 'NOTEBOOK_DEMO_LABEL_REQUIRED')
        checkpoint=review.get('checkpoints')
        require(isinstance(checkpoint,dict) and bool(checkpoint), 'NOTEBOOK_CHECKPOINT_PROOF_REQUIRED')
        try: encoded=json.dumps(checkpoint,allow_nan=False)
        except (ValueError,TypeError): raise ReleaseError('NOTEBOOK_CHECKPOINT_PROOF_REQUIRED') from None
        require(len(encoded)<=64*1024, 'NOTEBOOK_CHECKPOINT_PROOF_REQUIRED')
        _check_structured_credentials(checkpoint)
        counts=scan_notebook_outputs(document)
        require(review.get('output_counts')==counts, 'NOTEBOOK_OUTPUT_COUNTS_CHANGED')
        require(counts['png']>=1 and counts['text']>=1, 'NOTEBOOK_RICH_OUTPUT_REQUIRED')
    identities = set()
    code_count = 0
    for cell in cells:
        require(isinstance(cell, dict) and cell.get('cell_type') in {'markdown', 'code'}, 'MALFORMED_NOTEBOOK')
        allowed = {'cell_type', 'id', 'metadata', 'source'} | ({'execution_count', 'outputs'} if cell['cell_type'] == 'code' else set())
        require(set(cell) <= allowed and cell.get('metadata', {}) == {}, 'NOTEBOOK_HIDDEN_METADATA')
        identifier = cell.get('id')
        if identifier is not None:
            require(isinstance(identifier, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,64}', identifier)
                    and identifier not in identities, 'MALFORMED_NOTEBOOK')
            identities.add(identifier)
        source = cell.get('source')
        require(isinstance(source, str) or isinstance(source, list)
                and all(isinstance(v, str) for v in source), 'MALFORMED_NOTEBOOK')
        source = source if isinstance(source, str) else ''.join(source)
        require(len(source) <= 256*1024, 'RELEASE_SCAN_SIZE_LIMIT')
        _check_credentials(source)
        if cell['cell_type'] == 'code':
            code_count += 1
            if executed:
                require(type(cell.get('execution_count')) is int and cell['execution_count']==code_count, 'NOTEBOOK_EXECUTION_COUNT')
                require(not any(x in source for x in ['_original_connect','PRIVATE_INPUT_OR_NETWORK_ACCESS_BLOCKED','__CBAM_CHECKPOINTS__','_checkpoint_json']), 'NOTEBOOK_TEMPORARY_GUARD')
            else: require(cell.get('outputs') == [] and cell.get('execution_count') is None, 'NOTEBOOK_EXECUTION_OUTPUTS')
            try: ast.parse(source)
            except SyntaxError: raise ReleaseError('INVALID_NOTEBOOK_PYTHON_SOURCE') from None
        else:
            require(not re.search(r'<\s*(?:script|iframe|img|object|embed)\b|data:|javascript:', source, re.I),
                    'NOTEBOOK_EMBEDDED_PAYLOAD')


def _scan_svg(text, node):
    require(node['origin'] == 'ARTIFICIAL' or node['kind'] == 'source' and node['origin'] == 'SOURCE'
            and node.get('content_role') == 'ORIGINAL_SCHEMATIC', 'REVIEWED_SCHEMATIC_REQUIRED')
    require(not re.search(r'<!DOCTYPE|<!ENTITY', text, re.I), 'UNSAFE_SVG_CONTENT')
    try: root = ET.fromstring(text)
    except ET.ParseError: raise ReleaseError('INVALID_SVG_CONTENT') from None
    require(root.tag.rsplit('}', 1)[-1] == 'svg', 'INVALID_SVG_CONTENT')
    forbidden = {'script', 'foreignObject', 'iframe', 'object', 'embed', 'image', 'audio', 'video'}
    for element in root.iter():
        require(element.tag.rsplit('}', 1)[-1] not in forbidden, 'UNSAFE_SVG_CONTENT')
        for attr_name, value in element.attrib.items():
            attr_name = attr_name.rsplit('}', 1)[-1]
            require(not attr_name.lower().startswith('on') and (attr_name != 'href' or value.startswith('#')),
                    'UNSAFE_SVG_CONTENT')
            for target in re.findall(r'url\((.*?)\)', value, re.I):
                require(target.strip(' \"\'').startswith('#'), 'UNSAFE_SVG_CONTENT')
        if element.tag.rsplit('}', 1)[-1] == 'style':
            require(not re.search(r'@import|url\(\s*[\"]?(?!#)', element.text or '', re.I), 'UNSAFE_SVG_CONTENT')


def _scan_png(content, *, maximum_pixels=16_000_000, maximum_bytes=64*1024*1024):
    """Validate a bounded 8-bit PNG, without trusting suffixes or a decoder."""
    require(content.startswith(b'\x89PNG\r\n\x1a\n'), 'INVALID_PNG_CONTENT')
    position, kinds, compressed = 8, [], bytearray()
    width = height = channels = None
    while position < len(content):
        require(position+12 <= len(content), 'INVALID_PNG_CONTENT')
        length = struct.unpack('>I', content[position:position+4])[0]
        kind = content[position+4:position+8]; end = position+12+length
        require(end <= len(content) and length <= maximum_bytes, 'INVALID_PNG_CONTENT')
        value = content[position+8:position+8+length]
        expected = struct.unpack('>I', content[position+8+length:end])[0]
        require(zlib.crc32(kind+value) & 0xffffffff == expected, 'PNG_CRC_MISMATCH')
        require(kind in {b'IHDR', b'IDAT', b'IEND', b'tEXt', b'pHYs', b'sRGB', b'gAMA'}, 'PNG_UNREVIEWED_METADATA')
        if not kinds: require(kind == b'IHDR', 'INVALID_PNG_CONTENT')
        if kind == b'IHDR':
            require(not kinds and length == 13, 'INVALID_PNG_CONTENT')
            width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', value)
            require(width > 0 and height > 0 and width*height <= maximum_pixels
                    and depth == 8 and color in {0, 2, 4, 6}
                    and compression == filtering == interlace == 0, 'PNG_UNSUPPORTED_IMAGE_PROFILE')
            channels = {0:1, 2:3, 4:2, 6:4}[color]
        elif kind == b'IDAT':
            require(b'IDAT' not in kinds or kinds[-1] == b'IDAT', 'INVALID_PNG_CONTENT')
            compressed.extend(value)
        elif kind == b'IEND':
            require(length == 0 and b'IDAT' in kinds and end == len(content), 'INVALID_PNG_CONTENT')
        elif kind == b'tEXt':
            require(kind not in kinds and b'IDAT' not in kinds and value.startswith(b'Software\x00'), 'PNG_UNREVIEWED_METADATA')
            software = value.split(b'\x00', 1)[1].decode('latin1')
            require(software == 'Matplotlib' or re.fullmatch(r'Matplotlib version[0-9.]+, https://matplotlib.org/', software), 'PNG_UNREVIEWED_METADATA')
        elif kind == b'pHYs':
            require(length == 9 and kind not in kinds and value[-1] in {0, 1}, 'PNG_UNREVIEWED_METADATA')
        elif kind == b'sRGB': require(length == 1 and value[0] <= 3 and kind not in kinds, 'PNG_UNREVIEWED_METADATA')
        elif kind == b'gAMA': require(length == 4 and kind not in kinds, 'PNG_UNREVIEWED_METADATA')
        kinds.append(kind);position = end
    require(kinds and kinds[-1] == b'IEND', 'INVALID_PNG_CONTENT')
    expected_size = height*(1+width*channels)
    require(expected_size <= maximum_bytes, 'RELEASE_SCAN_SIZE_LIMIT')
    try:
        decoder = zlib.decompressobj()
        pixels = decoder.decompress(compressed, expected_size+1)
    except zlib.error: raise ReleaseError('INVALID_PNG_CONTENT') from None
    require(decoder.eof and not decoder.unused_data and len(pixels) == expected_size, 'INVALID_PNG_CONTENT')
    row_size = width*channels+1
    require(all(pixels[i] <= 4 for i in range(0, len(pixels), row_size)), 'INVALID_PNG_CONTENT')
    return dict(status='SECURITY_CLEAN', format='png', bytes=len(content), width=width, height=height)


def _scan_static_html(text, node):
    """Non-active academic HTML/MathML: no scripts, resource URLs or forms."""
    from html.parser import HTMLParser
    from urllib.parse import urlsplit, unquote
    require(node.get('content_role')=='GENERATED_RESEARCH_SITE' and node['origin']=='SOURCE', 'REVIEWED_HTML_SITE_REQUIRED')
    allowed=set('html head body title meta style main header footer nav aside section article div span p a img figure figcaption h1 h2 h3 h4 h5 h6 ul ol li table thead tbody tr th td caption blockquote pre code em strong br hr details summary sup sub dl dt dd math mrow mi mn mo mfrac msup msub msubsup mroot msqrt munder mover munderover mstyle mspace mtext mtable mtr mtd mpadded mphantom menclose'.split())
    attributes=set('id class lang charset name content http-equiv href src alt title width height loading decoding open role aria-label aria-labelledby aria-hidden tabindex colspan rowspan scope xmlns display mathvariant stretchy fence separator lspace rspace accent accentunder columnalign rowalign columnspacing rowspacing linethickness notation scriptlevel displaystyle encoding rel form'.split())
    def css(value):
        require(not re.search(r'url\s*\(|@import|@font-face|expression\s*\(|\\|<!--',value,re.I),'HTML_EXTERNAL_STYLE')
    class Check(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True);self.stack=[];self.css_text=''
        def handle_starttag(self,tag,attrs):
            require(tag in allowed,'HTML_UNREVIEWED_ELEMENT')
            require(len(attrs)==len({k for k,v in attrs}), 'HTML_DUPLICATE_ATTRIBUTE')
            for key,value in attrs:
                require(key in attributes and not key.startswith('on'), 'HTML_UNREVIEWED_ATTRIBUTE')
                value=value or '';_check_credentials(value)
                if key=='xmlns':require(value=='http://www.w3.org/1998/Math/MathML','HTML_UNREVIEWED_NAMESPACE')
                if key=='http-equiv':require(value.lower()=='content-security-policy','HTML_ACTIVE_META')
                if key in {'href','src'}:
                    decoded=unquote(value);parts=urlsplit(decoded)
                    require('\\' not in decoded and not decoded.startswith('//') and parts.scheme in {'','https'},'HTML_UNSAFE_URL')
                    if key=='src': require(not parts.scheme and not parts.netloc and not decoded.startswith('/') and '..' not in PurePosixPath(parts.path).parts,'HTML_EXTERNAL_RESOURCE')
            if tag not in {'meta','img','br','hr'}:self.stack.append(tag)
        def handle_startendtag(self,tag,attrs):
            self.handle_starttag(tag,attrs)
            if tag not in {'meta','img','br','hr'}:self.handle_endtag(tag)
        def handle_endtag(self,tag):
            require(tag in allowed and self.stack and self.stack[-1]==tag,'HTML_UNBALANCED_STRUCTURE')
            if tag=='style':css(self.css_text);self.css_text=''
            self.stack.pop()
        def handle_data(self,data):
            if self.stack and self.stack[-1]=='style':self.css_text+=data;css(self.css_text)
        def handle_decl(self,decl):require(decl.lower()=='doctype html','HTML_UNREVIEWED_DECLARATION')
    require(not re.search(r'<\?|<!\[|<!ENTITY',text,re.I),'HTML_UNREVIEWED_DECLARATION')
    parser=Check();parser.feed(text);parser.close();require(not parser.stack,'HTML_UNBALANCED_STRUCTURE')



def _text_scan(content, node):
    require(not any(content.startswith(magic) for magic in _MAGIC), 'RESTRICTED_BINARY_MAGIC')
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError as error:
        raise ReleaseError('UNREVIEWED_BINARY_CONTENT') from None
    require('\x00' not in text, 'BINARY_CONTENT_DISGUISED_AS_TEXT')
    require(not _PRIVATE_PATH.search(text), 'PERSONAL_ABSOLUTE_PATH')
    _check_credentials(text)
    stripped = text.lstrip('\ufeff \t\r\n')
    structured = None
    if stripped.startswith(('{', '[')):
        try: structured = json.loads(stripped)
        except ValueError: pass
        except RecursionError: raise ReleaseError('STRUCTURED_SCAN_COMPLEXITY_LIMIT') from None
    if structured is not None:
        _check_structured_credentials(structured)
    if isinstance(structured, dict) and ('nbformat' in structured or 'cells' in structured) and not (node.get('format')=='notebook' and node.get('content_role')=='REVIEWED_ARTIFICIAL_NOTEBOOK'):
        cells = structured.get('cells')
        require(isinstance(cells, list), 'MALFORMED_NOTEBOOK')
        for cell in cells:
            require(isinstance(cell, dict) and not cell.get('outputs') and cell.get('execution_count') is None,
                    'NOTEBOOK_EXECUTION_OUTPUTS')
            require(not cell.get('attachments'), 'NOTEBOOK_EMBEDDED_PAYLOAD')
    data_kind = node.get('format') in {'json', 'csv', 'html', 'svg', 'notebook'}
    if node.get('format') in {'python', 'toml', 'makefile'}:
        require(node['kind'] == 'source' and node['origin'] == 'SOURCE', 'SOURCE_OWNERSHIP_CLASSIFICATION_REQUIRED')
    if node.get('format') == 'python':
        try: ast.parse(text)
        except SyntaxError as error: raise ReleaseError('INVALID_PYTHON_SOURCE_CONTENT') from None
    if node.get('format') == 'toml':
        try: parsed_toml = tomllib.loads(text)
        except ValueError as error: raise ReleaseError('INVALID_TOML_SOURCE_CONTENT') from None
        _check_structured_credentials(parsed_toml)
    if node.get('format') == 'yaml':
        require(node['kind'] == 'source' and node['origin'] == 'SOURCE'
                and node.get('content_role') in {'CI_WORKFLOW', 'CITATION_METADATA'}, 'REVIEWED_YAML_SOURCE_REQUIRED')
        try:
            import yaml
        except ImportError: raise ReleaseError('YAML_SCANNER_DEPENDENCY_MISSING') from None
        try: parsed_yaml = yaml.safe_load(text)
        except yaml.YAMLError: raise ReleaseError('INVALID_YAML_SOURCE_CONTENT') from None
        _check_structured_credentials(parsed_yaml)
    if node.get('format') in {'json', 'notebook'}:
        require(structured is not None, 'INVALID_JSON_CONTENT')
    if node.get('format') == 'notebook': _scan_notebook(structured, node)
    if structured is not None and node.get('format') not in {'json', 'notebook'}:
        raise ReleaseError('STRUCTURED_CONTENT_MASQUERADE')
    lines = [line for line in text.splitlines() if line.strip()]
    for delimiter in [',', '\t', ';']:
        if len(lines) < 2 or delimiter not in lines[0]: continue
        try: records = list(csv.reader(lines[:4], delimiter=delimiter))
        except csv.Error: records = []
        if records and len(records[0]) >= 2 and len(records) >= 2 and all(len(row) == len(records[0]) for row in records):
            if node.get('format') != 'csv':
                # Code and TOML receive independent syntax checks above.
                numeric_row = any(re.search(r'\d', field) for row in records[1:] for field in row)
                if numeric_row and node.get('format') not in {'python', 'toml', 'makefile'}:
                    raise ReleaseError('TABULAR_CONTENT_MASQUERADE')
    if node['kind'] != 'source' and data_kind:
        require(node['origin'] == 'ARTIFICIAL', 'REAL_DATA_ARTIFACT_NOT_RELEASE_SUPPORTED')
    if node.get('format') in {'html', 'svg'}:
        require(not re.search(r'<\s*(?:script|iframe|object|embed|foreignObject)\b|\bon\w+\s*=|javascript:|data:', text, re.I),
                'ACTIVE_OR_EMBEDDED_REPORT_CONTENT')
        if node.get('format') == 'svg': _scan_svg(text, node)
        elif node.get('content_role') == 'GENERATED_RESEARCH_SITE': _scan_static_html(text, node)
    return dict(format=node.get('format'), bytes=len(content), status='SECURITY_CLEAN')


def _check_gzip_header_metadata(content):
    """Check metadata of the one reviewed gzip stream separately from payload."""
    require(len(content) >= 10, 'UNREADABLE_RELEASE_ARCHIVE')
    flags, position = content[3], 10
    if flags & 4:
        require(position+2 <= len(content), 'UNREADABLE_RELEASE_ARCHIVE')
        position += 2+int.from_bytes(content[position:position+2], 'little')
        require(position <= len(content), 'UNREADABLE_RELEASE_ARCHIVE')
    for flag in (8, 16):
        if flags & flag:
            end = content.find(b'\x00', position)
            require(end >= position, 'UNREADABLE_RELEASE_ARCHIVE')
            value = content[position:end].decode('utf-8', errors='ignore')
            _check_credentials(value)
            if flag == 8:
                _check_credential_name(value)
            position = end+1


def scan_content(content, registry, node_id, *, limits=None, _budget=None, _depth=0):
    """Bounded content inspection; archive paths never become filesystem paths.

    All archive members need explicit byte identities and graph edges. A zip,
    gzip or tar header under a text suffix is rejected, not treated as text.
    """
    require(isinstance(content, bytes), 'SCAN_BYTES_REQUIRED')
    budget = _budget or _Budget(limits or ScanLimits())
    budget.charge(len(content), _depth)
    node = registry.identity(node_id, sha256(content))
    _check_credential_name(node.get('path'))
    fmt = node.get('format')
    require(fmt in FORMATS, 'UNREVIEWED_CONTENT_FORMAT')
    # Container comments, member metadata and gzip filenames are bytes too;
    # inspecting only decompressed member payloads would miss leaked secrets.
    if fmt=='notebook' and node.get('content_role')=='REVIEWED_ARTIFICIAL_NOTEBOOK':
        require(node.get('dependencies'), 'NOTEBOOK_ARTIFICIAL_LINEAGE_REQUIRED')
        for child in node['dependencies']:
            generator=registry.node(child);review=node.get('execution_review',{})
            require(generator['origin']=='ARTIFICIAL' and generator['lineage_kind']=='artificial_generator'
                    and generator.get('program_sha256')==review.get('program_sha256')
                    and generator.get('sha256')==review.get('outputs_sha256'), 'NOTEBOOK_ARTIFICIAL_LINEAGE_REQUIRED')
            registry.authorize(child)
    if fmt == 'png':
        require(node['origin'] == 'ARTIFICIAL' and node['kind'] == 'artifact', 'ARTIFICIAL_PNG_REQUIRED')
        registry.authorize(node_id, sha256(content))
        return dict(node_id=node_id, sha256=sha256(content), **_scan_png(content))
    visible_bytes = content.decode('utf-8', errors='ignore')
    require(not _PRIVATE_PATH.search(visible_bytes), 'PERSONAL_ABSOLUTE_PATH')
    # Raw container bytes join unrelated member documents. Contextual generic
    # CDS keys are checked within each decoded file or metadata object below.
    _check_credentials(visible_bytes, contextual_cds=False)
    zipped = content.startswith((b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08'))
    gzipped = content.startswith(b'\x1f\x8b')
    tarred = len(content) > 262 and content[257:262] == b'ustar'
    detected = 'zip' if zipped else 'gzip' if gzipped else 'tar' if tarred else None
    if detected or fmt in {'zip', 'gzip', 'tar'}:
        require(fmt == detected or fmt == 'tar' and detected is None, 'ARCHIVE_FORMAT_MASQUERADE')
        members = node.get('members', {})
        require(bool(members), 'UNREVIEWED_ARCHIVE_CONTENTS')
        seen, reports = set(), []
        def inspect(name, data):
            relative(name)
            _check_credential_name(name)
            require(name not in seen, 'DUPLICATE_ARCHIVE_MEMBER')
            seen.add(name)
            require(name in members, 'UNREGISTERED_ARCHIVE_MEMBER')
            reports.append(scan_content(data, registry, members[name], _budget=budget, _depth=_depth+1))
        try:
            if fmt == 'zip':
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    _check_credentials(archive.comment.decode('utf-8', errors='ignore'))
                    require(len(archive.infolist()) <= budget.limits.max_members, 'RELEASE_SCAN_RECURSION_LIMIT')
                    for item in archive.infolist():
                        # Extra fields are an unreviewed binary metadata channel.
                        # Current approved wheels and clean ZIPs do not require them.
                        require(not item.extra, 'ZIP_EXTRA_METADATA_UNREVIEWED')
                        local = content[item.header_offset:item.header_offset+30]
                        require(len(local) == 30 and local[:4] == b'PK\x03\x04', 'UNREADABLE_RELEASE_ARCHIVE')
                        require(int.from_bytes(local[28:30], 'little') == 0, 'ZIP_EXTRA_METADATA_UNREVIEWED')
                        _check_credentials(item.comment.decode('utf-8', errors='ignore'))
                        name = item.filename.rstrip('/') if item.is_dir() else item.filename
                        relative(name)
                        _check_credential_name(name)
                        require(stat.S_IFMT(item.external_attr >> 16) in {0, stat.S_IFREG, stat.S_IFDIR},
                                'UNSAFE_ARCHIVE_MEMBER')
                        require(not item.flag_bits & 1, 'ENCRYPTED_ARCHIVE_UNSCANNABLE')
                        if item.is_dir(): continue
                        require(item.file_size <= budget.limits.max_file_bytes
                                and item.file_size <= max(item.compress_size, 1) * budget.limits.max_compression_ratio,
                                'RELEASE_SCAN_SIZE_LIMIT')
                        with archive.open(item) as handle:
                            data = handle.read(budget.limits.max_file_bytes+1)
                        inspect(item.filename, data)
            elif fmt == 'tar':
                with tarfile.open(fileobj=io.BytesIO(content), mode='r:') as archive:
                    _check_structured_credentials(archive.pax_headers)
                    count = 0
                    for item in archive:
                        _check_structured_credentials(item.pax_headers)
                        count += 1
                        require(count <= budget.limits.max_members, 'RELEASE_SCAN_RECURSION_LIMIT')
                        relative(item.name.rstrip('/') if item.isdir() else item.name)
                        _check_credential_name(item.name)
                        if item.isdir(): continue
                        require(item.isfile() and not item.issym() and not item.islnk(), 'UNSAFE_ARCHIVE_MEMBER')
                        require(item.size <= budget.limits.max_file_bytes, 'RELEASE_SCAN_SIZE_LIMIT')
                        with archive.extractfile(item) as handle:
                            data = handle.read(budget.limits.max_file_bytes+1)
                        inspect(item.name, data)
            else:
                _check_gzip_header_metadata(content)
                require(len(members) == 1, 'GZIP_SINGLE_MEMBER_IDENTITY_REQUIRED')
                decoder = zlib.decompressobj(wbits=31)
                data = decoder.decompress(content, budget.limits.max_file_bytes+1)
                require(len(data) <= budget.limits.max_file_bytes, 'RELEASE_SCAN_SIZE_LIMIT')
                require(decoder.eof, 'UNREADABLE_RELEASE_ARCHIVE')
                require(not decoder.unused_data, 'GZIP_SINGLE_MEMBER_IDENTITY_REQUIRED')
                require(len(data) <= max(len(content), 1)*budget.limits.max_compression_ratio, 'RELEASE_SCAN_SIZE_LIMIT')
                inspect(next(iter(members)), data)
        except (OSError, EOFError, ValueError, zipfile.BadZipFile, tarfile.TarError, zlib.error) as error:
            if isinstance(error, ReleaseError): raise
            raise ReleaseError('UNREADABLE_RELEASE_ARCHIVE') from None
        require(seen == set(members), 'ARCHIVE_MEMBER_SET_CHANGED')
        return dict(node_id=node_id, format=fmt, sha256=sha256(content), members=reports, status='SECURITY_CLEAN')
    return dict(node_id=node_id, sha256=sha256(content), **_text_scan(content, node))


def _safe_read_checked(root, relative_path, maximum):
    root = Path(root).absolute()
    path = root.joinpath(*relative(relative_path).parts)
    require(not any(p.is_symlink() for p in [path, *path.parents]), 'RELEASE_SOURCE_SYMLINK')
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'RELEASE_SOURCE_NOT_INDEPENDENT_FILE')
    require(info.st_size <= maximum, 'RELEASE_SCAN_SIZE_LIMIT')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as handle:
        opened = os.fstat(handle.fileno())
        require((opened.st_dev, opened.st_ino) == (info.st_dev, info.st_ino)
                and opened.st_nlink == 1 and stat.S_ISREG(opened.st_mode), 'RELEASE_SOURCE_CHANGED_DURING_OPEN')
        content = handle.read(maximum+1)
    require(len(content) <= maximum, 'RELEASE_SCAN_SIZE_LIMIT')
    return content


def _safe_read(root, relative_path, maximum):
    try:
        return _safe_read_checked(root, relative_path, maximum)
    except ReleaseError:
        raise
    except OSError:
        raise ReleaseError('RELEASE_SOURCE_UNAVAILABLE') from None


def _inspect_candidate(registry, source_root, whitelist, context, satisfied_conditions, limits, legal):
    require(isinstance(whitelist, dict) and bool(whitelist), 'EXPLICIT_RELEASE_WHITELIST_REQUIRED')
    budget = _Budget(limits)
    contents, rows = {}, []
    for path, node_id in sorted(whitelist.items()):
        relative(path)
        node = registry.node(node_id)
        require(node.get('path') == path, 'RELEASE_PATH_IDENTITY_CHANGED')
        content = _safe_read(source_root, path, limits.max_file_bytes)
        identity = sha256(content)
        registry.identity(node_id, identity)
        approval = registry.authorize(node_id, identity, context=context,
            satisfied_conditions=satisfied_conditions) if legal else None
        scan = scan_content(content, registry, node_id, _budget=budget)
        contents[path] = content
        rows.append(dict(path=path, node_id=node_id, sha256=identity, bytes=len(content), scan=scan, approval=approval))
    return contents, rows


def audit_candidate(registry, source_root, *, whitelist, limits=None):
    """Security-only audit: pending ownership never becomes publication approval."""
    _, rows = _inspect_candidate(registry, source_root, whitelist, None, None, limits or ScanLimits(), False)
    return dict(status='SECURITY_CLEAN', publication_approved=False, files=rows)


def prepare_candidate(registry, source_root, destination, *, whitelist, protected_roots=(), limits=None):
    """Copy an exact source-only closure into a new private review directory.

    This is not public staging and grants no ownership or publication rights.
    Pending ownership is permitted only for security review of registered source
    text. Data assets, unknown additions and partial candidate closures fail.
    """
    require(isinstance(whitelist, dict) and bool(whitelist), 'EXPLICIT_RELEASE_WHITELIST_REQUIRED')
    require(registry.data.get('candidate_preparation_authorized') is True, 'CANDIDATE_PREPARATION_NOT_AUTHORIZED')
    expected = registry.data.get('candidate_whitelist')
    require(isinstance(expected, list) and len(expected) == len(set(expected))
            and set(whitelist) == set(expected), 'EXACT_SOURCE_CANDIDATE_CLOSURE_REQUIRED')
    for path, node_id in whitelist.items():
        node = registry.node(node_id)
        require(node['kind'] == 'source' and node['origin'] == 'SOURCE'
                and node.get('path') == path and node['lineage_kind'] == 'source'
                and node['lineage_complete'], 'CANDIDATE_SOURCE_ONLY')
        require(node['status'] in {'PENDING_OWNER_APPROVAL', 'APPROVED_PUBLIC', 'APPROVED_WITH_CONDITIONS'},
                'CANDIDATE_SOURCE_STATUS_NOT_REVIEWABLE')
        require(node.get('format') in {'python', 'toml', 'markdown', 'makefile', 'text', 'yaml', 'notebook', 'svg'},
                'CANDIDATE_SOURCE_FORMAT_ONLY')
    destination = Path(destination).absolute()
    require('..' not in destination.parts, 'INVALID_RELEASE_DESTINATION')
    require(not any(p.is_symlink() for p in [destination, *destination.parents]), 'RELEASE_DESTINATION_SYMLINK')
    destination = destination.resolve(strict=False)
    require(not destination.exists() and destination.parent.is_dir(), 'RELEASE_DESTINATION_MUST_BE_NEW')
    for root in [source_root, *protected_roots]:
        root = Path(root).resolve()
        require(not (destination.is_relative_to(root) or root.is_relative_to(destination)), 'RELEASE_PROTECTED_ROOT_OVERLAP')
    contents, rows = _inspect_candidate(registry, source_root, whitelist, None, None,
                                       limits or ScanLimits(), False)
    temporary = Path(tempfile.mkdtemp(prefix='.cbam-candidate-', dir=destination.parent))
    try:
        for relative_path, content in contents.items():
            path = temporary.joinpath(*relative(relative_path).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        require(not destination.exists(), 'RELEASE_DESTINATION_APPEARED')
        os.rename(temporary, destination)
    except BaseException as error:
        shutil.rmtree(temporary)
        if isinstance(error, OSError) and not isinstance(error, ReleaseError):
            raise ReleaseError('CANDIDATE_COPY_IO_FAILED') from None
        raise
    return dict(status='SECURITY_REVIEW_CANDIDATE', release_status='PUBLIC_AUTH_PENDING',
                publication_approved=False, published=False, files=rows)


def stage_public(registry, source_root, destination, *, whitelist, context='public_repository',
                 satisfied_conditions=None, protected_roots=(), limits=None):
    """Create a new reviewed staging directory only after every check succeeds.

    Exact bytes are held after validation and written into a sibling temporary
    directory, then renamed. No network, estimator, input recovery or cleaning.
    """
    destination = Path(destination).absolute()
    require('..' not in destination.parts, 'INVALID_RELEASE_DESTINATION')
    require(not any(p.is_symlink() for p in [destination, *destination.parents]), 'RELEASE_DESTINATION_SYMLINK')
    destination = destination.resolve(strict=False)
    require(not destination.exists() and destination.parent.is_dir(), 'RELEASE_DESTINATION_MUST_BE_NEW')
    for root in [source_root, *protected_roots]:
        root = Path(root).resolve()
        require(not (destination.is_relative_to(root) or root.is_relative_to(destination)), 'RELEASE_PROTECTED_ROOT_OVERLAP')
    contents, rows = _inspect_candidate(registry, source_root, whitelist, context, satisfied_conditions,
                                       limits or ScanLimits(), True)
    temporary = Path(tempfile.mkdtemp(prefix='.cbam-release-', dir=destination.parent))
    try:
        for relative_path, content in contents.items():
            path = temporary.joinpath(*relative(relative_path).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        require(not destination.exists(), 'RELEASE_DESTINATION_APPEARED')
        os.rename(temporary, destination)
    except BaseException as error:
        shutil.rmtree(temporary)
        if isinstance(error, OSError) and not isinstance(error, ReleaseError):
            raise ReleaseError('RELEASE_STAGING_IO_FAILED') from None
        raise
    return dict(status='STAGED_REVIEWED_BYTES', publication_approved=True, published=False,
                context=context, files=rows)
