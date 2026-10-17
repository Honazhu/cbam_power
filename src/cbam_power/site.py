"""Deterministic offline research website, maintained in docs/methodology.md.

This presentation tool reads reviewed source, teaching figures and API signatures.
It does not import the research engine, open empirical inputs, fit models, or fetch
remote assets. Markdown and LaTeX are converted locally; generated HTML has no JS.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from html import escape
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import tempfile
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET


class SiteError(ValueError):
    """Stable error codes; never echo local paths or untrusted input content."""


SOURCE = 'docs/methodology.md'
OUTPUT = 'docs/index.html'
GENERATOR = 'src/cbam_power/site.py'
API_SOURCE = 'src/cbam_power/methods.py'
FIGURES = ('workflow.svg', 'corridors.svg', 'era5_aggregation.svg',
           'rdit_demo.png', 'synthetic_demo.png')
PUBLIC_API = ('Corridor', 'RDiTSpec', 'validate_inputs', 'aggregate_flows',
              'construct_rdit', 'fit_rdit', 'infer_hac', 'aggregate_era5',
              'predict_synthetic', 'evaluate_fixed_objective')

CSS = """
:root{color-scheme:light;--ink:#18354a;--muted:#536a79;--teal:#126c78;--paper:#fbfaf7;--line:#d7e1e6}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:1rem}body{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:17px;line-height:1.72}
a{color:var(--teal);text-decoration-thickness:.08em;text-underline-offset:.18em}a:hover{color:#093e48}a:focus-visible,summary:focus-visible,.equation:focus-visible{outline:3px solid #b97f2b;outline-offset:4px}
.skip{position:absolute;left:1rem;top:-5rem;background:white;padding:.6rem;z-index:9}.skip:focus{top:1rem}
.masthead{border-top:5px solid var(--teal);background:#eef4f5;border-bottom:1px solid var(--line)}.masthead-inner{max-width:1320px;margin:auto;padding:1.2rem 2rem;display:flex;justify-content:space-between;align-items:center;gap:1.5rem;flex-wrap:wrap}.brand{font-size:.82rem;letter-spacing:.14em;font-weight:700;text-transform:uppercase}.utility{display:flex;gap:1.1rem;font-size:.86rem;flex-wrap:wrap}
.layout{max-width:1320px;margin:auto;display:grid;grid-template-columns:245px minmax(0,1fr);gap:3.6rem;padding:2.8rem 2rem 4rem}.contents{align-self:start;position:sticky;top:1.2rem;max-height:calc(100vh - 2.4rem);overflow:auto;font-size:.85rem;line-height:1.5;border-right:1px solid var(--line);padding-right:1.3rem}.contents summary{font-weight:700;cursor:pointer;list-style:none;margin-bottom:.9rem}.contents ol{list-style:none;padding:0;margin:0}.contents li{margin:.65rem 0}.contents a{text-decoration:none;color:var(--muted)}.contents a:hover{text-decoration:underline;color:var(--teal)}
main{min-width:0;max-width:945px}h1,h2,h3{line-height:1.22;letter-spacing:-.025em;color:var(--ink)}h1{font-family:Georgia,"Times New Roman",serif;font-size:clamp(2.2rem,4vw,3.6rem);font-weight:500;margin:0 0 1.5rem;max-width:22ch}h2{font-size:1.65rem;margin:3rem 0 1rem;padding-top:1.2rem;border-top:1px solid var(--line)}h3{font-size:1.16rem;margin:1.8rem 0 .8rem}p{margin:.9rem 0 1.2rem}main>p:first-of-type{font-size:1.12rem;color:#37576f}blockquote{margin:1.4rem 0;padding:.1rem 1.2rem;border-left:4px solid var(--teal);background:#edf5f4}blockquote p{margin:.7rem 0}
figure{margin:1.8rem 0 2rem;background:white;border:1px solid var(--line);border-radius:8px;overflow:hidden}figure img{display:block;width:100%;height:auto}figcaption{padding:1.1rem 1.3rem;font-size:.86rem;line-height:1.65;border-top:1px solid var(--line);color:var(--muted)}figcaption strong:first-child{color:var(--ink)}.figure-source{display:block;font-size:.73rem;margin-top:.5rem;overflow-wrap:anywhere}.figure-source a{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
.table-wrap{overflow-x:auto;margin:1.4rem 0;border:1px solid var(--line);border-radius:6px}table{width:100%;border-collapse:collapse;font-size:.87rem;line-height:1.55}th,td{text-align:left;vertical-align:top;padding:.8rem .9rem;border-bottom:1px solid var(--line)}th{background:#edf3f5;font-weight:650}tr:last-child td{border-bottom:0}td:first-child{min-width:9rem}td p{margin:.2rem 0}.api-table td:first-child{min-width:12rem}
code,pre{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.86em}code{background:#edf1f3;padding:.12em .25em;border-radius:3px;overflow-wrap:anywhere}pre{background:#eff3f5;padding:1.1rem 1.2rem;overflow-x:auto;border-radius:6px;line-height:1.6}pre code{padding:0;background:transparent;overflow-wrap:normal}
.equation{overflow-x:auto;max-width:100%;padding:1rem .3rem;margin:1.3rem 0;background:#fff;border-radius:6px}math{font-family:"STIX Two Math","Cambria Math",serif;font-size:1.08em}math[display=block]{margin:.35rem auto}math[display=inline]{vertical-align:baseline}li{margin:.35rem 0}hr{border:0;border-top:1px solid var(--line);margin:2rem 0}
.build-record{font-size:.78rem;color:var(--muted);margin-top:3rem;border-top:1px solid var(--line);padding-top:1rem}.build-record summary{cursor:pointer;font-weight:600}.build-record dd{margin:.2rem 0 .8rem;overflow-wrap:anywhere}.footer{border-top:1px solid var(--line);padding:1.5rem 2rem;text-align:center;color:var(--muted);font-size:.82rem}
@media(max-width:1000px){.layout{grid-template-columns:1fr;gap:1.8rem;max-width:900px}.contents{position:static;max-height:none;border:1px solid var(--line);padding:1rem 1.2rem;border-radius:6px}.contents ol{columns:2;column-gap:2rem}.contents li{break-inside:avoid}.contents summary{margin-bottom:.3rem}main{max-width:none}}
@media(max-width:560px){body{font-size:16px}.layout{padding:1.5rem 1rem 2.5rem}.masthead-inner{padding:1rem}.utility{gap:.8rem}h1{font-size:2.25rem}h2{font-size:1.4rem;margin-top:2.2rem}.contents ol{columns:1;max-height:10rem;overflow:auto}.contents summary:after{content:" · scroll sections";font-weight:400;color:var(--muted);font-size:.8rem}.contents li{margin:.4rem 0}figcaption{padding:.9rem;font-size:.82rem}th,td{padding:.65rem}.equation{font-size:.93rem}}
@media print{.masthead,.contents,.footer,.build-record,.skip{display:none}.layout{display:block;padding:0;max-width:none}main{max-width:none}body{background:white;color:black;font-size:10pt}h2,figure,table{break-inside:avoid}a{color:inherit}figure img{max-height:20cm;object-fit:contain}h1{font-size:26pt}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
""".strip()


def require(condition, code):
    if not condition:
        raise SiteError(code)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _root(path):
    root = Path(path).absolute()
    require('..' not in root.parts and not any(p.is_symlink() for p in [root, *root.parents]),
            'SITE_SOURCE_PATH_UNSAFE')
    require(root.is_dir(), 'SITE_SOURCE_ROOT_MISSING')
    return root.resolve()


def _read(root, relative, limit=16*1024*1024):
    parts = PurePosixPath(relative)
    require(not parts.is_absolute() and '..' not in parts.parts and '\\' not in relative,
            'SITE_RELATIVE_PATH_UNSAFE')
    path = root.joinpath(*parts.parts)
    require(not any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(root)),
            'SITE_SOURCE_SYMLINK')
    require(path.is_file(), 'SITE_SOURCE_MISSING')
    require(path.stat().st_nlink == 1 and path.stat().st_size <= limit, 'SITE_SOURCE_FILE_UNSAFE')
    return path.read_bytes()


def _local_link(root, target, *, image=False):
    parsed = urlsplit(target)
    require(not parsed.netloc or parsed.scheme == 'https', 'SITE_UNSAFE_LINK')
    if parsed.scheme:
        require(not image and parsed.scheme == 'https' and bool(parsed.netloc)
                and not parsed.username and not parsed.password, 'SITE_UNSAFE_LINK')
        return None
    path = unquote(parsed.path)
    require(not path.startswith(('/', '\\')) and '\\' not in path and not parsed.query,
            'SITE_UNSAFE_LINK')
    if not path:
        require(not image, 'SITE_FIGURE_PATH_INVALID')
        return None
    relative = posixpath.normpath('docs/'+path)
    require(not relative.startswith('../') and relative != '..', 'SITE_LINK_OUTSIDE_SOURCE')
    if image:
        require(relative in {'docs/figures/'+name for name in FIGURES}, 'SITE_UNREVIEWED_FIGURE')
    local = root/relative
    if local.is_dir():
        require(not local.is_symlink() and local.resolve().is_relative_to(root), 'SITE_UNSAFE_LINK')
        return None
    _read(root, relative)
    return relative


def _math(tex, display):
    require(len(tex) <= 16000 and not re.search(r'\\(?:href|url|html\w*|includegraphics|input|include|class|style)\b', tex),
            'SITE_UNSAFE_MATH')
    try:
        from latex2mathml.converter import convert
        raw = convert(tex, display='block' if display else 'inline')
        tree = ET.fromstring(raw)
    except Exception as error:
        raise SiteError('SITE_MATH_CONVERSION_FAILED') from None
    tags = {'math','mrow','mi','mn','mo','mtext','mfrac','msqrt','mroot','msub','msup','msubsup',
            'munder','mover','munderover','mtable','mtr','mtd','mstyle','mspace','mpadded','mphantom',
            'menclose','mfenced','mmultiscripts','mprescripts','none'}
    attrs = {'display','displaystyle','scriptlevel','mathvariant','mathsize','stretchy','fence','separator',
             'form','lspace','rspace','maxsize','minsize','accent','accentunder','columnalign','rowalign',
             'rowspacing','columnspacing','columnspan','rowspan','linethickness','notation','width','height',
             'depth','voffset','lquote','rquote','open','close','separators','align','equalrows','equalcolumns'}
    for node in tree.iter():
        require(node.tag.startswith('{http://www.w3.org/1998/Math/MathML}')
                and node.tag.rsplit('}',1)[-1] in tags, 'SITE_UNSAFE_MATHML')
        require(set(node.attrib) <= attrs and all(not re.search(r'url\s*\(|https?:|javascript:', v, re.I)
                                                 for v in node.attrib.values()), 'SITE_UNSAFE_MATHML')
    # The converter output has already been parsed and restricted. A text
    # alternative improves equation inspection without a JS rendering layer.
    raw = raw.replace('<math ', '<math aria-label="'+escape(tex, quote=True)+'" ', 1)
    return '<div class="equation" tabindex="0" role="region" aria-label="Mathematical equation; scroll horizontally if needed">'+raw+'</div>\n' if display else raw


def _api_table(root):
    data = _read(root, API_SOURCE)
    tree = ast.parse(data.decode('utf-8'))
    nodes = {node.name:node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    require(set(PUBLIC_API) <= nodes.keys(), 'SITE_API_SURFACE_INCOMPLETE')
    rows = []
    for name in PUBLIC_API:
        node = nodes[name]
        if isinstance(node, ast.ClassDef):
            fields = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    term = item.target.id+': '+ast.unparse(item.annotation)
                    if item.value is not None: term += ' = '+ast.unparse(item.value)
                    fields.append(term)
            keyword_only = any(isinstance(d, ast.Call) and any(k.arg == 'kw_only' and isinstance(k.value, ast.Constant) and k.value.value is True for k in d.keywords) for d in node.decorator_list)
            signature = name+'('+('*'+', ' if keyword_only and fields else '')+', '.join(fields)+')'
        else:
            signature = name+'('+ast.unparse(node.args)+')'
            if node.returns: signature += ' -> '+ast.unparse(node.returns)
        description = (ast.get_docstring(node) or 'See the source contract.').split('\n\n', 1)[0]
        rows.append('<tr><td><code>'+escape(signature)+'</code></td><td>'+escape(description)+'</td></tr>')
    return ('<div class="table-wrap"><table class="api-table"><thead><tr><th>Source signature</th>'
            '<th>Contract</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>\n')


def render_site_with_evidence(source_root):
    """Return (deterministic HTML bytes, public build evidence); perform no writes."""
    root = _root(source_root)
    source = _read(root, SOURCE)
    generator = _read(root, GENERATOR)
    require(generator == Path(__file__).read_bytes(), 'SITE_INSTALLED_GENERATOR_STALE')
    from markdown_it import MarkdownIt
    md = MarkdownIt('commonmark', {'html':False, 'linkify':False}).enable('table')
    md.renderer.rules['table_open'] = lambda tokens, idx, options, env: '<div class="table-wrap"><table>\n'
    md.renderer.rules['table_close'] = lambda tokens, idx, options, env: '</table></div>\n'
    links = {'README.md', 'docs/data_access.md'}
    for relative in links: _read(root, relative)
    counts = {'equations':0, 'inline_math':0, 'api_tables':0}

    def inline_math(state, silent):
        start = state.pos
        if state.src[start:start+1] != '$' or state.src[start:start+2] == '$$': return False
        end = start+1
        while end < state.posMax:
            if state.src[end] == '$' and (end == 0 or state.src[end-1] != '\\'): break
            if state.src[end] == '\n': return False
            end += 1
        require(end < state.posMax and end > start+1, 'SITE_UNCLOSED_MATH')
        if not silent:
            token = state.push('math_inline', 'math', 0)
            token.content = state.src[start+1:end]
        state.pos = end+1
        return True

    def block_math(state, start, end, silent):
        def line(i): return state.src[state.bMarks[i]+state.tShift[i]:state.eMarks[i]].strip()
        if line(start) != '$$': return False
        stop = start+1
        while stop < end and line(stop) != '$$': stop += 1
        require(stop < end, 'SITE_UNCLOSED_MATH')
        if silent: return True
        token = state.push('math_block', 'math', 0)
        token.block = True
        token.map = [start, stop+1]
        token.content = '\n'.join(line(i) for i in range(start+1,stop))
        state.line = stop+1
        return True

    md.inline.ruler.before('escape', 'math_inline', inline_math)
    md.block.ruler.before('fence', 'math_block', block_math, {'alt':['paragraph','reference','blockquote','list']})
    def math_rule(tokens, idx, options, env):
        block = tokens[idx].type == 'math_block'
        counts['equations' if block else 'inline_math'] += 1
        return _math(tokens[idx].content, block)
    md.renderer.rules['math_inline'] = math_rule
    md.renderer.rules['math_block'] = math_rule

    old_fence = md.renderer.rules['fence']
    def fence(tokens, idx, options, env):
        if tokens[idx].info.strip() == 'api-signatures':
            require(tokens[idx].content.strip() == 'cbam_power.methods', 'SITE_API_REFERENCE_INVALID')
            counts['api_tables'] += 1
            return _api_table(root)
        return old_fence(tokens, idx, options, env)
    md.renderer.rules['fence'] = fence

    tokens = md.parse(source.decode('utf-8'))
    heading_ids = set()
    toc = []
    title = None
    for i, token in enumerate(tokens):
        if token.type == 'heading_open':
            text = tokens[i+1].content
            slug = re.sub(r'[^a-z0-9]+','-',text.lower()).strip('-')
            require(slug and slug not in heading_ids, 'SITE_DUPLICATE_HEADING')
            token.attrSet('id', slug);heading_ids.add(slug)
            if token.tag == 'h1':
                require(title is None, 'SITE_MULTIPLE_TITLES');title = text
            if token.tag == 'h2': toc.append((slug, text))
        for child in token.children or []:
            if child.type in {'link_open','image'}:
                target = child.attrGet('src' if child.type == 'image' else 'href')
                ref = _local_link(root,target,image=child.type == 'image')
                if ref: links.add(ref)
                if child.type == 'link_open' and urlsplit(target).scheme:
                    child.attrSet('rel','noreferrer noopener')
    require(title and toc, 'SITE_STRUCTURE_MISSING')
    for token in tokens:
        for child in token.children or []:
            if child.type == 'link_open':
                target = child.attrGet('href')
                if target.startswith('#'):
                    require(unquote(target[1:]) in heading_ids, 'SITE_ANCHOR_MISSING')

    # Figures and their maintained prose captions form a single accessible unit.
    rendered = []
    figures = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        children = tokens[i+1].children if i+1 < len(tokens) and tokens[i+1].type == 'inline' else []
        if t.type == 'paragraph_open' and children and len(children) == 1 and children[0].type == 'image':
            require(i+5 < len(tokens) and tokens[i+3].type == 'paragraph_open'
                    and tokens[i+4].type == 'inline', 'SITE_FIGURE_CAPTION_MISSING')
            caption = tokens[i+4].content
            number = len(figures)+1
            require(caption.startswith('**Figure '+str(number)+'. ')
                    and all(word in caption for word in ['Units:', 'Interpretation:', 'Provenance:', 'Limits:']),
                    'SITE_FIGURE_CAPTION_CONTRACT')
            image = children[0]
            relative = _local_link(root, image.attrGet('src'), image=True)
            figures.append(relative)
            image.attrSet('loading', 'lazy');image.attrSet('decoding','async')
            image_html = md.renderer.renderInline([image],md.options,{})
            caption_html = md.renderer.renderInline(tokens[i+4].children,md.options,{})
            provenance = '<span class="figure-source">Source: <a href="'+escape(image.attrGet('src'),quote=True)+'">'+escape(relative)+'</a> · SHA-256 '+digest(_read(root,relative))+'</span>'
            rendered.append('<figure id="figure-'+str(number)+'">'+image_html+'<figcaption>'+caption_html+provenance+'</figcaption></figure>\n')
            i += 6
        else:
            rendered.append(md.renderer.render([t], md.options, {}));i += 1
    require(figures == ['docs/figures/'+name for name in FIGURES], 'SITE_FIGURE_SET_CHANGED')
    require(counts['api_tables'] == 1 and counts['equations'] > 0, 'SITE_TEACHING_COMPONENT_MISSING')
    links.add(API_SOURCE)
    evidence = dict(source_sha256=digest(source),generator_sha256=digest(generator),
                    sources={name:digest(_read(root,name)) for name in sorted(links|{SOURCE,GENERATOR})},
                    renderer_versions={name:importlib.metadata.version(name) for name in ['markdown-it-py','latex2mathml']},
                    figures=figures, **counts)
    binding = digest(json.dumps(evidence,sort_keys=True,separators=(',',':')).encode())
    evidence['binding_sha256'] = binding
    navigation = ''.join('<li><a href="#'+slug+'">'+escape(text)+'</a></li>' for slug,text in toc)
    provenance = ''.join('<dt>'+escape(name)+'</dt><dd><code>'+value+'</code></dd>' for name,value in evidence['sources'].items())
    html = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\'; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">'
            '<meta name="referrer" content="no-referrer">'
            '<meta name="cbam-source-sha256" content="'+evidence['source_sha256']+'">'
            '<meta name="cbam-generator-sha256" content="'+evidence['generator_sha256']+'">'
            '<meta name="cbam-binding-sha256" content="'+binding+'">'
            '<title>'+escape(title)+' · CBAM power research</title><style>'+CSS+'</style></head><body>'
            '<a class="skip" href="#main">Skip to research narrative</a>'
            '<header class="masthead"><div class="masthead-inner"><span class="brand">CBAM · Electricity research</span>'
            '<nav class="utility" aria-label="Repository resources"><a href="../README.md">Repository guide</a>'
            '<a href="methodology.md">Markdown source</a><a href="data_access.md">Data access</a></nav></div></header>'
            '<div class="layout"><aside class="contents"><details open><summary>In this guide</summary>'
            '<nav aria-label="Research sections"><ol>'+navigation+'</ol></nav></details></aside>'
            '<main id="main">'+''.join(rendered)+'<details class="build-record"><summary>Build provenance · generated from one Markdown source</summary>'
            '<p>Offline conversion; no estimation, external fonts, scripts or data requests. Source binding: <code>'+binding+'</code>.</p><dl>'+provenance+'</dl></details></main></div>'
            '<footer class="footer">Original teaching material · artificial illustrations · empirical outputs remain private.</footer></body></html>\n')
    require('<script' not in html.lower() and '@import' not in CSS and 'url(' not in CSS.lower(), 'SITE_ACTIVE_RESOURCE')
    result = html.encode('utf-8')
    evidence['html_sha256'] = digest(result)
    return result, evidence


def render_site(source_root):
    """Return the deterministic UTF-8 HTML string without writing any files."""
    content, _ = render_site_with_evidence(source_root)
    return content.decode('utf-8')


def build_site(source_root, *, check=False):
    """Build only docs/index.html, or fail if its deterministic bytes are stale."""
    root = _root(source_root)
    content,evidence = render_site_with_evidence(root)
    output = root/OUTPUT
    require(not output.is_symlink() and not output.parent.is_symlink(), 'SITE_OUTPUT_LINK')
    if check:
        require(output.is_file() and output.read_bytes() == content, 'SITE_GENERATED_HTML_STALE')
    else:
        require(not output.exists() or output.is_file() and output.stat().st_nlink == 1, 'SITE_OUTPUT_UNSAFE')
        descriptor,name = tempfile.mkstemp(prefix='.site-',suffix='.tmp',dir=output.parent)
        try:
            with os.fdopen(descriptor,'wb') as handle:
                handle.write(content);handle.flush();os.fsync(handle.fileno())
            os.replace(name,output)
        finally:
            if Path(name).exists(): Path(name).unlink()
    return dict(status='PASS', generated=OUTPUT, check_only=check, **evidence)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv)
    try:
        print(json.dumps(build_site(args.source_root,check=args.check),sort_keys=True));return 0
    except (SiteError,ImportError):
        print('SITE_BUILD_FAILED',file=__import__('sys').stderr);return 2


if __name__ == '__main__':
    raise SystemExit(main())
