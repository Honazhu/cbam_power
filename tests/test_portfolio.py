"""Artificial PR7 format and execution tests; no empirical data or credentials."""
import base64
import copy
import io
import json
from pathlib import Path
import struct
import tomllib
import zlib

import pytest
from cbam_power import release, portfolio

pytestmark = pytest.mark.public
INVALID_FIXTURE = 'SYNTHETIC-INVALID-NOT-A-CREDENTIAL'


def record(key='source', content=b'example\n', fmt='text', **changes):
    value = dict(id=key,kind='source',origin='SOURCE',status='PENDING_OWNER_APPROVAL',
        dependencies=[],lineage_complete=True,lineage_kind='source',evidence=[],conditions=[],permissions=[],
        sha256=release.sha256(content),path='example.txt',format=fmt,reason='Independent artificial test fixture')
    value.update(changes);return value


def registry(records):
    evidence=dict(id='artificial-permission',kind='user_demo_authorization',status='VERIFIED',
        source='Explicit synthetic test authorization',scope=['group:teaching'],applicability='NOT_APPLICABLE',
        permissions=['public_repository'])
    return release.RightsRegistry(dict(nodes=records,evidence=[evidence],publication_scope='SOURCE_AND_ARTIFICIAL_ONLY'))


def notebook(code='value = 2 + 3\nassert value == 5\n'):
    return dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),
        language_info=dict(name='python')),cells=[dict(id='teaching-cell',cell_type='code',metadata={},source=code,
                                                     execution_count=None,outputs=[])])


def notebook_record(document):
    content=json.dumps(document).encode()
    return content,record(content=content,fmt='notebook',path='notebooks/example.ipynb',content_role='UNEXECUTED_NOTEBOOK')


def test_reviewed_unexecuted_notebook_is_source_not_public_permission():
    content,node=notebook_record(notebook())
    rights=registry([node])
    assert release.scan_content(content,rights,'source')['status']=='SECURITY_CLEAN'
    with pytest.raises(release.ReleaseError,match='RIGHTS_NOT_APPROVED'):rights.authorize('source')


@pytest.mark.parametrize('change,error',[
    ('outputs','NOTEBOOK_EXECUTION_OUTPUTS'),('metadata','NOTEBOOK_HIDDEN_METADATA'),
    ('cell_metadata','NOTEBOOK_HIDDEN_METADATA'),('attachments','NOTEBOOK_EMBEDDED_PAYLOAD'),
    ('invalid_code','INVALID_NOTEBOOK_PYTHON_SOURCE'),('credential','CREDENTIAL_CONFIGURATION_CONTENT'),
    ('role','REVIEWED_NOTEBOOK_SOURCE_REQUIRED'),('changed','RELEASE_CONTENT_IDENTITY_CHANGED'),
    ('raw_cell','MALFORMED_NOTEBOOK'),('language','NOTEBOOK_PYTHON_REQUIRED')])
def test_notebook_hidden_state_and_content_rejected(change,error):
    document=notebook()
    if change=='outputs':document['cells'][0]['outputs']=[{'output_type':'stream','name':'stdout','text':'saved'}]
    elif change=='metadata':document['metadata']['hidden']={'payload':'not permitted'}
    elif change=='cell_metadata':document['cells'][0]['metadata']['execution']={'time':'not permitted'}
    elif change=='attachments':document['cells'][0]['attachments']={'plot':'not permitted'}
    elif change=='invalid_code':document['cells'][0]['source']='%shell_command'
    elif change=='credential':document['cells'][0]['source']='ENTSOE_'+'TOKEN = '+repr(INVALID_FIXTURE)
    elif change=='raw_cell':document['cells'][0]['cell_type']='raw'
    elif change=='language':document['metadata']['language_info']['name']='javascript'
    content,node=notebook_record(document)
    if change=='role':node['content_role']='unreviewed'
    elif change=='changed':content+=b'\n'
    with pytest.raises(release.ReleaseError,match=error):release.scan_content(content,registry([node]),'source')


def test_yaml_source_decodes_escaped_credentials_and_rejects_tags():
    content=b'name: teaching\non: [push]\njobs:\n  tests:\n    runs-on: ubuntu-latest\n'
    node=record(content=content,fmt='yaml',content_role='CI_WORKFLOW',path='.github/workflows/test.yml')
    assert release.scan_content(content,registry([node]),'source')['status']=='SECURITY_CLEAN'
    escaped='access'+chr(92)+'u005f'+'token'
    for bad,error in [(('"'+escaped+'": "'+INVALID_FIXTURE+'"\n').encode(),'CREDENTIAL_CONFIGURATION_CONTENT'),
                      (b'value: !!python/object:invalid {}\n','INVALID_YAML_SOURCE_CONTENT')]:
        node['sha256']=release.sha256(bad)
        with pytest.raises(release.ReleaseError,match=error):release.scan_content(bad,registry([node]),'source')


def test_original_svg_schematic_keeps_pending_ownership():
    content=b'<svg xmlns="http://www.w3.org/2000/svg"><rect x="1" y="1" width="20" height="10"/></svg>'
    node=record(content=content,fmt='svg',content_role='ORIGINAL_SCHEMATIC',path='schematic.svg')
    rights=registry([node]);assert release.scan_content(content,rights,'source')['status']=='SECURITY_CLEAN'
    with pytest.raises(release.ReleaseError,match='RIGHTS_NOT_APPROVED'):rights.authorize('source')


@pytest.mark.parametrize('fragment',[
    '<image href="https://example.invalid/image.png"/>', '<use href="https://example.invalid/element"/>',
    '<style>@import "https://example.invalid/style";</style>',
    '<rect fill="url(https://example.invalid/pixel)"/>',
    '<!DOCTYPE svg [<!ENTITY hidden "payload">]>'])
def test_svg_external_or_embedded_content_rejected(fragment):
    content=('<svg xmlns="http://www.w3.org/2000/svg">'+fragment+'</svg>').encode()
    node=record(content=content,fmt='svg',content_role='ORIGINAL_SCHEMATIC')
    with pytest.raises(release.ReleaseError,match='UNSAFE_SVG_CONTENT|INVALID_SVG_CONTENT'):
        release.scan_content(content,registry([node]),'source')


def chunk(kind,value):
    return struct.pack('>I',len(value))+kind+value+struct.pack('>I',zlib.crc32(kind+value)&0xffffffff)


def png(*,metadata=b'',width=1,height=1,pixels=b'\x00\xff\x00\x00\xff'):
    header=struct.pack('>IIBBBBB',width,height,8,6,0,0,0)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',header)+metadata+chunk(b'IDAT',zlib.compress(pixels))+chunk(b'IEND',b'')


def png_record(content):
    return record('figure',content=content,fmt='png',path='docs/figures/example.png',kind='artifact',origin='ARTIFICIAL',
        status='APPROVED_PUBLIC',lineage_kind='artificial_generator',review_group='teaching',
        evidence=['artificial-permission'],permissions=['public_repository'])


@pytest.mark.parametrize('software',[b'Matplotlib',b'Matplotlib version3.11.1, https://matplotlib.org/'])
def test_reviewed_artificial_png_and_minimal_software_metadata(software):
    content=png(metadata=chunk(b'tEXt',b'Software\x00'+software))
    result=release.scan_content(content,registry([png_record(content)]),'figure')
    assert result['width']==result['height']==1
    assert result['status']=='SECURITY_CLEAN'


@pytest.mark.parametrize('change,error',[
    ('real','ARTIFICIAL_PNG_REQUIRED'),('pending','RIGHTS_NOT_APPROVED'),('crc','PNG_CRC_MISMATCH'),
    ('trailing','INVALID_PNG_CONTENT'),('metadata','PNG_UNREVIEWED_METADATA'),
    ('size','PNG_UNSUPPORTED_IMAGE_PROFILE'),('pixels','INVALID_PNG_CONTENT'),
    ('changed','RELEASE_CONTENT_IDENTITY_CHANGED'),('disguised','RESTRICTED_BINARY_MAGIC')])
def test_png_cannot_hide_unknown_real_data_or_metadata(change,error):
    content=png()
    if change=='crc':content=content[:-1]+bytes([content[-1]^1])
    elif change=='trailing':content+=b'EXTRA'
    elif change=='metadata':content=png(metadata=chunk(b'tEXt',b'Comment\x00'+INVALID_FIXTURE.encode()))
    elif change=='size':content=png(width=100000,height=100000)
    elif change=='pixels':content=png(pixels=b'\x00'*10000)
    node=png_record(content)
    if change=='real':node['origin']='REAL'
    elif change=='pending':node['status']='UNVERIFIED'
    elif change=='changed':content+=b'changed'
    elif change=='disguised':node.update(kind='source',origin='SOURCE',format='markdown',path='example.md')
    with pytest.raises(release.ReleaseError,match=error):release.scan_content(content,registry([node]),'figure')


def test_private_notebook_outputs_are_scanned_without_publication_grant():
    document=notebook();document['cells'][0]['outputs']=[dict(output_type='stream',name='stdout',text='Teaching output\n'),
        dict(output_type='display_data',data={'image/png':base64.b64encode(png()).decode(),'text/plain':'Artificial figure'},metadata={})]
    assert portfolio.inspect_notebook_outputs(document)==2
    document['cells'][0]['outputs'][0]['text']='ENTSOE_'+'TOKEN='+INVALID_FIXTURE
    with pytest.raises(release.ReleaseError):portfolio.inspect_notebook_outputs(document)


@pytest.mark.parametrize('mime',['text/html','application/javascript','application/octet-stream'])
def test_executed_notebook_unknown_mime_is_not_approved(mime):
    document=notebook();document['cells'][0]['outputs']=[dict(output_type='display_data',data={mime:'not permitted'})]
    with pytest.raises(portfolio.PortfolioError,match='PORTFOLIO_OUTPUT_MIME_REJECTED'):
        portfolio.inspect_notebook_outputs(document)


def toml_value(value):
    if isinstance(value,str):return json.dumps(value)
    if isinstance(value,bool):return str(value).lower()
    if isinstance(value,int):return str(value)
    if isinstance(value,list):return '['+', '.join(toml_value(v) for v in value)+']'
    if isinstance(value,dict):return '{'+', '.join(toml_value(k)+' = '+toml_value(v) for k,v in value.items())+'}'
    raise TypeError('unsupported artificial fixture')


def source_fixture(tmp_path):
    root=tmp_path/'source';(root/'manifests').mkdir(parents=True)
    notebook_paths=[];records=[]
    for i in range(4):
        name='notebooks/teaching_'+str(i)+'.ipynb';notebook_paths.append(name)
        document=notebook('import os, socket\nfrom pathlib import Path\n'
            'assert "previous_notebook_marker" not in globals()\nprevious_notebook_marker = True\n'
            'assert Path.home().name == "home"\n'
            'assert "ENTSOE_TOKEN" not in os.environ\n'
            'try:\n    socket.getaddrinfo("example.invalid", 443)\n'
            'except RuntimeError as error:\n    assert str(error) == "EXTERNAL_NETWORK_BLOCKED"\n'
            'else:\n    raise AssertionError("Network guard absent")\n'
            'from cbam_power.inputs import InputCatalog\n'
            'try:\n    InputCatalog()\n'
            'except RuntimeError as error:\n    assert str(error) == "PRIVATE_INPUT_OR_NETWORK_ACCESS_BLOCKED"\n'
            'else:\n    raise AssertionError("Input guard absent")\n'
            'value = 2 + 3\nassert value == 5\nprint("Artificial teaching result", value)\n')
        content=json.dumps(document).encode();path=root/name;path.parent.mkdir(exist_ok=True);path.write_bytes(content)
        records.append(record('source:'+name,content=content,fmt='notebook',path=name,content_role='UNEXECUTED_NOTEBOOK'))
    name='manifests/release_rights.toml'
    records.append(record('source:'+name,fmt='toml',path=name,sha256='UNVERIFIED'))
    data=registry(records).data
    data.update(candidate_whitelist=[n['path'] for n in records],portfolio_notebooks=notebook_paths,
                portfolio_artifacts={},notebook_execution_authorized=True)
    text='\n'.join(k+' = '+toml_value(v) for k,v in data.items())+'\n'
    (root/name).write_text(text)
    return root


def test_source_closure_rejects_unknown_files_and_preserves_pending_rights(tmp_path):
    source=source_fixture(tmp_path)
    _,rights,selection,report=portfolio.inspect_source(source)
    assert len(selection)==5 and report['publication_approved'] is False
    with pytest.raises(release.ReleaseError,match='RIGHTS_NOT_APPROVED'):rights.authorize(next(iter(selection.values())))
    (source/'unexpected.csv').write_text('unknown\n')
    with pytest.raises(portfolio.PortfolioError,match='PORTFOLIO_SOURCE_CLOSURE_CHANGED'):portfolio.inspect_source(source)


def test_notebook_execution_requires_external_new_output_and_explicit_permission(tmp_path):
    source=source_fixture(tmp_path);root,rights,selection,_=portfolio.inspect_source(source)
    with pytest.raises(portfolio.PortfolioError,match='PORTFOLIO_OUTPUT_SOURCE_OVERLAP'):
        portfolio.execute_notebooks(root,rights,selection,source/'outputs')
    rights.data['notebook_execution_authorized']=False
    with pytest.raises(portfolio.PortfolioError,match='PORTFOLIO_NOTEBOOK_EXECUTION_NOT_AUTHORIZED'):
        portfolio.execute_notebooks(root,rights,selection,tmp_path/'outputs')


def test_four_artificial_notebooks_execute_in_fresh_kernels_without_source_changes(tmp_path,monkeypatch):
    monkeypatch.setenv('ENTSOE_TOKEN',INVALID_FIXTURE)
    source=source_fixture(tmp_path);root,rights,selection,_=portfolio.inspect_source(source)
    before={name:(root/name).read_bytes() for name in selection}
    reports=portfolio.execute_notebooks(root,rights,selection,tmp_path/'execution')
    assert len(reports)==4 and all(r['status']=='PASS' and r['outputs_checked']==1 for r in reports)
    assert all((root/name).read_bytes()==content for name,content in before.items())
    assert len(list((tmp_path/'execution').glob('notebook-*/executed.ipynb')))==4


@pytest.mark.parametrize('target,error',[('../../outside.txt','PORTFOLIO_NOTEBOOK_LINK_OUTSIDE_SOURCE'),
    ('missing.md','PORTFOLIO_NOTEBOOK_LINK_MISSING'),('![image](https://example.invalid/image.png)','PORTFOLIO_REMOTE_NOTEBOOK_IMAGE_REJECTED')])
def test_notebook_relative_links_cannot_escape_or_load_remote_images(tmp_path,target,error):
    document=notebook();document['cells']=[dict(cell_type='markdown',source=target if target.startswith('!') else '[reference]('+target+')',metadata={})]
    with pytest.raises(portfolio.PortfolioError,match=error):
        portfolio.inspect_notebook_links(document,tmp_path,'notebooks/example.ipynb')


def test_notebook_valid_relative_document_link(tmp_path):
    (tmp_path/'docs').mkdir();(tmp_path/'docs/method.md').write_text('Teaching methods\n')
    document=notebook();document['cells']=[dict(cell_type='markdown',source='[Methods](../docs/method.md)',metadata={})]
    portfolio.inspect_notebook_links(document,tmp_path,'notebooks/example.ipynb')


def test_yaml_binary_payload_cannot_hide_in_source_metadata():
    content=b'payload: !!binary QUJDRA==\n'
    node=record(content=content,fmt='yaml',content_role='CI_WORKFLOW')
    with pytest.raises(release.ReleaseError,match='STRUCTURED_BINARY_PAYLOAD'):
        release.scan_content(content,registry([node]),'source')


def test_executed_notebook_output_metadata_is_scanned():
    credential_key='authorization'
    document=notebook();document['cells'][0]['outputs']=[dict(output_type='display_data',data={'text/plain':'artificial'},
        metadata={credential_key:INVALID_FIXTURE})]
    with pytest.raises(release.ReleaseError,match='CREDENTIAL_CONFIGURATION_CONTENT'):
        portfolio.inspect_notebook_outputs(document)


def licensed_package_fixture(tmp_path, *, license_files=None, modified_license=False):
    import csv, gzip, hashlib, tarfile, zipfile
    source=tmp_path/'source';source.mkdir();packages=tmp_path/'packages';packages.mkdir()
    selected=['LICENSE'] if license_files is None else license_files
    project={'project':{'version':'0.0.0','license':'MIT','license-files':selected},
             'tool':{'hatch':{'build':{'targets':{'wheel':{}}}}}}
    files={'src/cbam_power/__init__.py':b'"""Artificial package fixture."""\n',
           'LICENSE':b'Artificial license identity fixture; no third-party data grant.\n',
           'pyproject.toml':('\n'.join(k+' = '+toml_value(v) for k,v in project.items())+'\n').encode()}
    records=[]
    for name,content in files.items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        fmt='python' if name.endswith('.py') else 'toml' if name.endswith('.toml') else 'text'
        records.append(record('source:'+name,content=content,fmt=fmt,path=name))
    prefix='cbam_power-0.0.0';dist=prefix+'.dist-info/'
    contents={'cbam_power/__init__.py':files['src/cbam_power/__init__.py'],
              dist+'licenses/LICENSE':b'Changed artificial license\n' if modified_license else files['LICENSE'],
              dist+'METADATA':b'Metadata-Version: 2.4\nName: cbam-power\nVersion: 0.0.0\n',
              dist+'WHEEL':b'Wheel-Version: 1.0\n',dist+'entry_points.txt':b'[console_scripts]\n'}
    text=io.StringIO();writer=csv.writer(text,lineterminator='\n')
    for name,content in contents.items():
        digest=base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip('=')
        writer.writerow([name,'sha256='+digest,str(len(content))])
    writer.writerow([dist+'RECORD','','']);contents[dist+'RECORD']=text.getvalue().encode()
    with zipfile.ZipFile(packages/(prefix+'-py3-none-any.whl'),'w') as archive:
        for name,content in contents.items():archive.writestr(name,content)
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as archive:
        for name,content in {**files,'PKG-INFO':contents[dist+'METADATA']}.items():
            info=tarfile.TarInfo(prefix+'/'+name);info.size=len(content);archive.addfile(info,io.BytesIO(content))
    (packages/(prefix+'.tar.gz')).write_bytes(gzip.compress(buffer.getvalue()))
    return packages,source,registry(records),{n['path']:n['id'] for n in records}


def test_wheel_license_member_is_bound_to_exact_registered_source(tmp_path):
    packages,source,rights,selection=licensed_package_fixture(tmp_path)
    results=portfolio.inspect_packages(packages,source,rights,selection)
    assert len(results)==2 and all(r['status']=='SECURITY_CLEAN' for r in results)


@pytest.mark.parametrize('names,error',[(['*.txt'],'INVALID_RELEASE_PATH'),
    (['unknown.txt'],'PORTFOLIO_LICENSE_FILE_UNREGISTERED'),
    (['../LICENSE'],'INVALID_RELEASE_PATH'),(['LICENSE','LICENSE'],'PORTFOLIO_LICENSE_FILES_INVALID')])
def test_wheel_license_file_rules_are_explicit_and_registered(tmp_path,names,error):
    packages,source,rights,selection=licensed_package_fixture(tmp_path,license_files=names)
    with pytest.raises((portfolio.PortfolioError,release.ReleaseError),match=error):
        portfolio.inspect_packages(packages,source,rights,selection)


def test_modified_wheel_license_cannot_pass_source_identity(tmp_path):
    packages,source,rights,selection=licensed_package_fixture(tmp_path,modified_license=True)
    with pytest.raises(portfolio.PortfolioError,match='PORTFOLIO_PACKAGE_SOURCE_CHANGED'):
        portfolio.inspect_packages(packages,source,rights,selection)


def test_license_file_cannot_be_reclassified_as_data(tmp_path):
    packages,source,rights,selection=licensed_package_fixture(tmp_path)
    rights.nodes['source:LICENSE']['origin']='REAL';rights.nodes['source:LICENSE']['kind']='asset'
    with pytest.raises(portfolio.PortfolioError,match='PORTFOLIO_LICENSE_FILE_NOT_SOURCE'):
        portfolio.inspect_packages(packages,source,rights,selection)

# PR7.1: published execution state is a separately reviewed, byte-pinned object.
def executed_fixture():
    document=notebook('print("'+release.DEMO_LABEL+'")\n')
    document['cells'][0]['execution_count']=1
    document['cells'][0]['outputs']=[dict(output_type='stream',name='stdout',text='metric  value\nmean    2.0\n'),
        dict(output_type='display_data',data={'image/png':base64.b64encode(png()).decode(),'text/plain':'Artificial plot'},metadata={})]
    content,node=notebook_record(document)
    node.update(content_role='REVIEWED_ARTIFICIAL_NOTEBOOK',dependencies=['artificial-calculation'],
        execution_review=dict(artificial_only=True,fresh_kernel_runs=2,
            program_sha256=release.notebook_program_sha256(document),outputs_sha256=release.notebook_outputs_sha256(document),checkpoints={'mean':2.0},output_counts=release.scan_notebook_outputs(document)))
    generator=png_record(b'artificial program');generator.update(id='artificial-calculation',format='text',program_sha256=node['execution_review']['program_sha256'],sha256=node['execution_review']['outputs_sha256'])
    return document,node,generator


def test_executed_artificial_notebook_keeps_png_and_table():
    document,node,generator=executed_fixture();content=json.dumps(document).encode();node['sha256']=release.sha256(content)
    report=release.scan_content(content,registry([node,generator]),'source')
    assert report['status']=='SECURITY_CLEAN'
    assert release.scan_notebook_outputs(document)=={'outputs':2,'png':1,'text':2}
    assert document['cells'][0]['outputs']


@pytest.mark.parametrize('change,error',[
    ('unverified','RIGHTS_NOT_APPROVED'),('lineage','NOTEBOOK_ARTIFICIAL_LINEAGE_REQUIRED'),
    ('once','NOTEBOOK_REEXECUTION_PROOF_REQUIRED'),('real','REAL_DATA_EXCLUDED_FROM_RELEASE|NOTEBOOK_ARTIFICIAL_LINEAGE_REQUIRED'),
    ('program','NOTEBOOK_EXECUTION_IDENTITY_CHANGED'),('outputs','NOTEBOOK_EXECUTION_IDENTITY_CHANGED'),
    ('error','NOTEBOOK_ERROR_OUTPUT'),('html','NOTEBOOK_OUTPUT_MIME'),('metadata','NOTEBOOK_OUTPUT_METADATA'),
    ('path','PERSONAL_ABSOLUTE_PATH'),('token','CREDENTIAL_CONFIGURATION_CONTENT'),
    ('traceback','NOTEBOOK_TRACEBACK_OUTPUT'),('counts','NOTEBOOK_EXECUTION_COUNT'),
    ('empty','NOTEBOOK_OUTPUT_COUNTS_CHANGED|NOTEBOOK_RICH_OUTPUT_REQUIRED'),('guard','NOTEBOOK_TEMPORARY_GUARD'),
    ('label','NOTEBOOK_DEMO_LABEL_REQUIRED'),('hidden','NOTEBOOK_HIDDEN_METADATA')])
def test_published_notebook_rejects_unsafe_or_unproven_execution(change,error):
    doc,node,gen=executed_fixture()
    if change=='unverified':gen['status']='UNVERIFIED'
    elif change=='lineage':node['dependencies']=[]
    elif change=='once':node['execution_review']['fresh_kernel_runs']=1
    elif change=='real':gen['origin']='REAL'
    elif change=='program':doc['cells'][0]['source']+='changed = True\n'
    elif change=='outputs':doc['cells'][0]['outputs'][0]['text']='changed\n'
    elif change=='error':doc['cells'][0]['outputs']=[dict(output_type='error',ename='Error',evalue='redacted',traceback=[])]
    elif change=='html':doc['cells'][0]['outputs'][1]['data']={'text/html':'<table><tr><td>2</td></tr></table>'}
    elif change=='metadata':doc['cells'][0]['outputs'][1]['metadata']={'hidden':'value'}
    elif change=='path':doc['cells'][0]['outputs'][0]['text']='/'+'Users'+'/synthetic-fixture/private.txt'
    elif change=='token':doc['cells'][0]['outputs'][0]['text']='ENTSOE_'+'TOKEN='+INVALID_FIXTURE
    elif change=='traceback':doc['cells'][0]['outputs'][0]['text']='Traceback (most recent call last):\n'
    elif change=='counts':doc['cells'][0]['execution_count']=7
    elif change=='empty':doc['cells'][0]['outputs']=[]
    elif change=='guard':doc['cells'][0]['source']+='_original_connect = None\n'
    elif change=='label':doc['cells'][0]['source']='print("no label")\n'
    elif change=='hidden':doc['cells'][0]['metadata']={'execution':{'private':'value'}}
    if change not in {'program','outputs'}:
        node['execution_review']['program_sha256']=release.notebook_program_sha256(doc)
        node['execution_review']['outputs_sha256']=release.notebook_outputs_sha256(doc)
    gen['program_sha256']=node['execution_review']['program_sha256'];gen['sha256']=node['execution_review']['outputs_sha256']
    content=json.dumps(doc).encode();node['sha256']=release.sha256(content)
    with pytest.raises(release.ReleaseError,match=error):release.scan_content(content,registry([node,gen]),'source')


def test_artificial_checkpoint_comparison_is_separate_and_exact_in_structure():
    portfolio.compare_checkpoints({'effect':[2.0,3.0],'status':'BLOCKED_RANK'},{'effect':[2.0+1e-12,3.0],'status':'BLOCKED_RANK'})
    for changed in [{'effect':[9.,3.],'status':'BLOCKED_RANK'},{'effect':[2.,3.],'status':'PASS'}, {'effect':[2.]}]:
        with pytest.raises(portfolio.PortfolioError):portfolio.compare_checkpoints(changed,{'effect':[2.,3.],'status':'BLOCKED_RANK'})


@pytest.mark.parametrize('name',['.DS_Store','__MACOSX/._report.csv','docs/._index.html','__MACOSX/'])
def test_finder_metadata_is_rejected_even_if_registered(name):
    content=b'innocuous bytes'
    node=record(content=content,path=name.rstrip('/'))
    with pytest.raises(release.ReleaseError,match='FINDER_METADATA_EXCLUDED'):
        release.scan_content(content,registry([node]),'source')


@pytest.mark.parametrize('fragment',[
    '<script>alert(1)</script>','<img src="https://example.invalid/x.png" alt="x">',
    '<style>@import "https://example.invalid/x";</style>','<iframe src="x"></iframe>',
    '<p onclick="alert(1)">x</p>','<meta http-equiv="refresh" content="0; url=https://example.invalid">',
    '<img src="//example.invalid/x.png">','<a href="javascript:alert(1)">x</a>',
    '<style>p{background:u\\72l(x)}</style>'])
def test_research_html_rejects_active_or_remote_resources(fragment):
    payload=('<!doctype html><html lang="en"><head><title>Research</title></head><body>'+fragment+'</body></html>').encode()
    node=record(content=payload,fmt='html',content_role='GENERATED_RESEARCH_SITE',path='docs/index.html')
    with pytest.raises(release.ReleaseError):release.scan_content(payload,registry([node]),'source')


def test_mathml_and_relative_illustration_are_safe_offline_html():
    content=b'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Research</title><style>img{max-width:100%}</style></head><body><h1>Methods</h1><img src="figures/artificial.svg" alt="Artificial schematic"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi><mo>=</mo><mn>2</mn></math></body></html>'
    node=record(content=content,fmt='html',content_role='GENERATED_RESEARCH_SITE',path='docs/index.html')
    assert release.scan_content(content,registry([node]),'source')['status']=='SECURITY_CLEAN'


@pytest.mark.parametrize('change,error',[
    ('result_count','NOTEBOOK_RESULT_COUNT'),('checkpoints','NOTEBOOK_CHECKPOINT_PROOF_REQUIRED'),
    ('output_counts','NOTEBOOK_OUTPUT_COUNTS_CHANGED'),('generator','NOTEBOOK_ARTIFICIAL_LINEAGE_REQUIRED'),
    ('capture','NOTEBOOK_TEMPORARY_GUARD')])
def test_reviewed_execution_proof_is_complete_and_capture_free(change,error):
    doc,node,gen=executed_fixture()
    if change=='result_count':
        out=doc['cells'][0]['outputs'][1];out['output_type']='execute_result';out['execution_count']=999
    elif change=='checkpoints':node['execution_review'].pop('checkpoints')
    elif change=='output_counts':node['execution_review']['output_counts']={}
    elif change=='generator':gen['program_sha256']='f'*64
    elif change=='capture':doc['cells'][0]['source']+='print("__CBAM_CHECKPOINTS__")\n'
    if change in {'result_count','capture'}:
        node['execution_review']['program_sha256']=release.notebook_program_sha256(doc)
        node['execution_review']['outputs_sha256']=release.notebook_outputs_sha256(doc)
        gen['program_sha256']=node['execution_review']['program_sha256'];gen['sha256']=node['execution_review']['outputs_sha256']
    content=json.dumps(doc).encode();node['sha256']=release.sha256(content)
    with pytest.raises(release.ReleaseError,match=error):release.scan_content(content,registry([node,gen]),'source')


def test_unclosed_html_style_cannot_load_external_css():
    content=b'<!doctype html><html><head><style>@import "https://example.invalid/hidden.css";'
    node=record(content=content,fmt='html',content_role='GENERATED_RESEARCH_SITE',path='docs/index.html')
    with pytest.raises(release.ReleaseError,match='HTML_EXTERNAL_STYLE|HTML_UNBALANCED_STRUCTURE'):
        release.scan_content(content,registry([node]),'source')


@pytest.mark.public
def test_reviewed_execution_survives_notebook_json_roundtrip():
    import nbformat
    document, node, generator = executed_fixture()
    # This is the real .ipynb serialization path; text becomes line arrays.
    serialized = nbformat.writes(nbformat.from_dict(document)).encode()
    reloaded = json.loads(serialized)
    assert release.notebook_outputs_sha256(reloaded) == node['execution_review']['outputs_sha256']
    node['sha256'] = release.sha256(serialized)
    authority = registry([node, generator])
    release.scan_content(serialized, authority, node['id'])
    original = release.notebook_outputs_sha256(reloaded)
    for cell in reloaded['cells']:
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'stream':
                value = output['text']
                output['text'] = (''.join(value) if isinstance(value,list) else value) + ' CHANGED'
                assert release.notebook_outputs_sha256(reloaded) != original
                return
    raise AssertionError('fixture must exercise real multiline text')
