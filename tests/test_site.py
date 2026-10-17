"""Public artificial tests for a deterministic source-only offline website."""
import hashlib
from html.parser import HTMLParser
import importlib.util
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

import pytest

from cbam_power import site

pytestmark = pytest.mark.public


@pytest.fixture
def source(tmp_path):
    root = tmp_path.resolve() / 'source'
    (root/'docs/figures').mkdir(parents=True)
    (root/'src/cbam_power').mkdir(parents=True)
    (root/'README.md').write_text('Artificial source fixture.\n')
    (root/'docs/data_access.md').write_text('Artificial data only.\n')
    (root/site.GENERATOR).write_bytes(Path(site.__file__).read_bytes())
    api = []
    for name in site.PUBLIC_API:
        if name in {'Corridor', 'RDiTSpec'}:
            api.append('@dataclass(kw_only=True)\nclass '+name+':\n    """Explicit toy settings."""\n    value: str\n')
        else:
            api.append('def '+name+'(value, *, option):\n    """Artificial API signature."""\n    raise AssertionError("MUST NOT EXECUTE")\n')
    (root/site.API_SOURCE).write_text('\n'.join(api))
    blocks = ['# Artificial research guide', 'A public teaching fixture.', '## Methods',
              'An inline equation $x^2$ and a [reference](https://example.org/method).',
              '$$\nx=\\frac{1}{2}\n$$']
    for n,name in enumerate(site.FIGURES,1):
        (root/'docs/figures'/name).write_bytes(b'ARTIFICIAL FIGURE '+name.encode())
        blocks += ['![Artificial figure](figures/'+name+')',
                   '**Figure '+str(n)+'. Teaching figure.** **Units:** artificial. **Interpretation:** toy example. **Provenance:** generated fixture. **Limits:** no empirical evidence.']
    blocks += ['## API', '```api-signatures\ncbam_power.methods\n```']
    (root/site.SOURCE).write_text('\n\n'.join(blocks)+'\n')
    return root


def edit(source, old, new):
    p = source/site.SOURCE
    p.write_text(p.read_text().replace(old,new))


def test_deterministic_and_read_only_render(source):
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    first, evidence = site.render_site_with_evidence(source)
    assert site.render_site(source).encode() == first
    assert evidence['equations'] == 1 and evidence['inline_math'] == 1
    assert evidence['api_tables'] == 1 and len(evidence['figures']) == 5
    assert evidence['html_sha256'] == hashlib.sha256(first).hexdigest()
    assert before == {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}


def test_mathml_responsive_and_offline_resources(source):
    html = site.render_site(source)
    for text in ['<math ', 'MathML', 'aria-label=', 'Content-Security-Policy',
                 '@media(max-width:560px)', 'prefers-reduced-motion', 'Skip to research narrative',
                 '<figure id="figure-5">', '<figcaption>', 'tabindex="0"', 'role="region"', 'Corridor(*, value: str)']:
        assert text in html
    assert '<script' not in html and 'cdn' not in html and 'url(' not in site.CSS
    assert 'https://example.org/method' in html  # Citation anchor, never fetched.
    assert html.count('<figure ') == 5 and html.count('<figcaption>') == 5
    assert 'SHA-256 ' in html and 'Source binding:' in html


def test_no_network_or_api_execution(source):
    with patch.object(socket,'create_connection',side_effect=AssertionError('NETWORK')), \
         patch.object(socket.socket,'connect',side_effect=AssertionError('NETWORK')):
        html = site.render_site(source)
    assert 'Artificial API signature' in html  # AST documentation; undefined decorator is not executed.


def test_build_and_stale_check_only(source):
    first = site.build_site(source)
    assert first['status'] == 'PASS' and site.build_site(source,check=True)['status'] == 'PASS'
    unchanged = (source/site.OUTPUT).read_bytes()
    edit(source,'A public teaching fixture.','A revised public teaching fixture.')
    with pytest.raises(site.SiteError,match='SITE_GENERATED_HTML_STALE'):
        site.build_site(source,check=True)
    assert (source/site.OUTPUT).read_bytes() == unchanged
    assert site.build_site(source)['source_sha256'] != first['source_sha256']
    assert not list((source/'docs').glob('.site-*'))


@pytest.mark.parametrize('relative',[site.API_SOURCE, 'docs/figures/workflow.svg', 'README.md'])
def test_all_referenced_sources_are_bound(source,relative):
    before = site.render_site_with_evidence(source)[1]
    with (source/relative).open('ab') as f: f.write(b'\n')
    after = site.render_site_with_evidence(source)[1]
    assert before['sources'][relative] != after['sources'][relative]
    assert before['binding_sha256'] != after['binding_sha256']


def test_installed_generator_must_match(source):
    (source/site.GENERATOR).write_text('# Stale or altered generator.\n')
    with pytest.raises(site.SiteError,match='SITE_INSTALLED_GENERATOR_STALE'): site.render_site(source)


@pytest.mark.parametrize('target',[
    '/private/secret', '../../outside', '%2fprivate/secret', 'file:///private/secret',
    ''.join(('https', '://', 'INVALID-TEST-USER', ':', 'INVALID-TEST-PASSWORD', '@example.org/a')), 'http://example.org/a', '//example.org/a',
    'data:text/html,hello', 'javascript:alert(1)',
])
def test_unsafe_link_rejected(source,target):
    with pytest.raises(site.SiteError): site._local_link(source,target)


def test_broken_local_reference_rejected(source):
    edit(source,'https://example.org/method','missing.md')
    with pytest.raises(site.SiteError,match='SITE_SOURCE_MISSING'): site.render_site(source)


def test_external_image_rejected(source):
    edit(source,'figures/workflow.svg','https://example.org/image.svg')
    with pytest.raises(site.SiteError,match='SITE_UNSAFE_LINK'): site.render_site(source)


def test_unreviewed_figure_rejected(source):
    edit(source,'figures/workflow.svg','figures/extra.svg')
    with pytest.raises(site.SiteError,match='SITE_UNREVIEWED_FIGURE'): site.render_site(source)


@pytest.mark.parametrize('kind',['symlink','hardlink'])
def test_input_links_rejected(source,tmp_path,kind):
    target = source/'docs/figures/workflow.svg'
    original = target.read_bytes();target.unlink()
    other = tmp_path/'outside';other.write_bytes(original)
    if kind == 'symlink': target.symlink_to(other)
    else: os.link(other,target)
    with pytest.raises(site.SiteError,match='SITE_SOURCE_(SYMLINK|FILE_UNSAFE)'): site.render_site(source)


def test_output_symlink_rejected(source,tmp_path):
    target=tmp_path/'protected';target.write_text('UNCHANGED')
    (source/site.OUTPUT).symlink_to(target)
    with pytest.raises(site.SiteError,match='SITE_OUTPUT_LINK'): site.build_site(source)
    assert target.read_text() == 'UNCHANGED'


def test_output_hardlink_rejected(source,tmp_path):
    target=tmp_path/'protected';target.write_text('UNCHANGED')
    os.link(target,source/site.OUTPUT)
    with pytest.raises(site.SiteError,match='SITE_OUTPUT_UNSAFE'): site.build_site(source)
    assert target.read_text() == 'UNCHANGED'


def test_caption_contract(source):
    edit(source,'**Units:** artificial.','Missing unit declaration.')
    with pytest.raises(site.SiteError,match='SITE_FIGURE_CAPTION_CONTRACT'): site.render_site(source)


def test_duplicate_and_missing_headings(source):
    edit(source,'## API','## Methods')
    with pytest.raises(site.SiteError,match='SITE_DUPLICATE_HEADING'): site.render_site(source)


def test_missing_internal_anchor(source):
    edit(source,'https://example.org/method','#not-present')
    with pytest.raises(site.SiteError,match='SITE_ANCHOR_MISSING'): site.render_site(source)


def test_missing_api_definition(source):
    p=source/site.API_SOURCE;p.write_text(p.read_text().replace('def infer_hac(', 'def renamed('))
    with pytest.raises(site.SiteError,match='SITE_API_SURFACE_INCOMPLETE'): site.render_site(source)


@pytest.mark.parametrize('tex',[r'\href{https://example.org}{x}', r'\htmlClass{x}{y}', r'\input{x}'])
def test_unsafe_math_commands(tex):
    with pytest.raises(site.SiteError,match='SITE_UNSAFE_MATH'): site._math(tex,True)


def test_unclosed_math(source):
    edit(source,'x=\\frac{1}{2}\n$$','x=\\frac{1}{2}')
    with pytest.raises(site.SiteError,match='SITE_UNCLOSED_MATH'): site.render_site(source)


def test_raw_html_is_text_not_active(source):
    edit(source,'A public teaching fixture.','A public teaching fixture. <script>alert(1)</script>')
    html=site.render_site(source)
    assert '<script>' not in html and '&lt;script&gt;' in html


def test_relative_root_traversal_rejected(source):
    with pytest.raises(site.SiteError,match='SITE_SOURCE_PATH_UNSAFE'):
        site.render_site(source/'docs'/'..')


def test_module_import_has_no_io_or_scientific_imports():
    p=Path(site.__file__)
    spec=importlib.util.spec_from_file_location('site_import_probe',p)
    module=importlib.util.module_from_spec(spec)
    before=set(sys.modules)
    with patch.object(Path,'read_bytes',side_effect=AssertionError('IMPORT READ')), \
         patch.object(Path,'write_bytes',side_effect=AssertionError('IMPORT WRITE')):
        spec.loader.exec_module(module)
    assert not any(name.startswith(('cbam_power.models','cbam_power.methods')) for name in set(sys.modules)-before)


class _BalancedHTML(HTMLParser):
    """Independent structural oracle, including the generated MathML subtree."""
    void = {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.void:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass  # XML-style empty MathML nodes have no pending closing tag.

    def handle_endtag(self, tag):
        assert self.stack and self.stack[-1] == tag, ('UNBALANCED_HTML', tag, tuple(self.stack))
        self.stack.pop()

    def validate(self, content):
        self.feed(content)
        self.close()
        assert not self.stack, ('UNCLOSED_HTML', tuple(self.stack))


def test_generated_html_has_independently_balanced_structure(source):
    html = site.render_site(source)
    assert '<table class="api-table">' in html
    _BalancedHTML().validate(html)


def test_structure_oracle_rejects_missing_api_wrapper_close(source):
    html = site.render_site(source)
    assert '</tbody></table></div>' in html
    malformed = html.replace('</tbody></table></div>', '</tbody></table>', 1)
    with pytest.raises(AssertionError, match='UNBALANCED_HTML'):
        _BalancedHTML().validate(malformed)
