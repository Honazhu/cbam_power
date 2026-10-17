"""Independent saved-output verification; expected files are never modified.

Comparison semantics are from V3 src/validation.py and PR1 baselines.toml.
No estimator is imported here. Hash equality and numerical equivalence are
separate checks. A missing timestamp proof is BLOCKED, never an implicit pass.
"""
import hashlib
import json
from pathlib import Path
import tomllib
from contextlib import contextmanager
from contextvars import ContextVar
import csv
import gzip

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    pass


_EVIDENCE=ContextVar('cbam_validation_evidence',default=None)
_DETAIL=ContextVar('cbam_validation_detail',default={})


@contextmanager
def comparison_context(**values):
    token=_DETAIL.set({**_DETAIL.get(),**values})
    try: yield
    finally: _DETAIL.reset(token)


class Evidence:
    """Streaming complete CSV evidence, compressed to avoid multi-GB repetition.

    Every numeric cell is emitted, including off-diagonal covariance elements.
    Expected values are read-only; the caller supplies an isolated private A path.
    """
    fields=['object','quantity','model_id','term','covariance_key','row_term','column_term',
            'key','actual','expected','absolute_difference','relative_difference',
            'atol','rtol','strict_abs_less_than','allowed_tolerance','status','sample_identity',
            'source_module','failure_category']

    def __init__(self,directory):
        self.directory=Path(directory);self.summary={};self.rows=0;self.failed=0

    def __enter__(self):
        self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'comparison_cells.csv.gz'
        if self.path.exists(): raise ValidationError('EVIDENCE_EXISTS_USE_NEW_DIRECTORY')
        self.file=gzip.open(self.path,'wt',encoding='utf-8',newline='',compresslevel=1)
        self.writer=csv.DictWriter(self.file,fieldnames=self.fields,extrasaction='ignore')
        self.writer.writeheader();self.token=_EVIDENCE.set(self);return self

    def __exit__(self,kind,value,tb):
        _EVIDENCE.reset(self.token);self.file.close()
        out=dict(rows=self.rows,passed=self.rows-self.failed,failed=self.failed,groups=self.summary,
                 complete=kind is None,error=None if value is None else str(value))
        (self.directory/'comparison_summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')

    def row(self,*,actual,expected,status,quantity='',**extra):
        context={**_DETAIL.get(),**extra};context.pop('labels',None);context.pop('columns',None)
        record={**context,'quantity':quantity or context.get('quantity',''),'actual':actual,'expected':expected,'status':status}
        if status=='PASS': record['failure_category']=''
        elif not record.get('failure_category'): record['failure_category']='UNCLASSIFIED_VALIDATION_FAILURE'
        self.writer.writerow(record);self.rows+=1;self.failed+=status!='PASS'
        group=str(record.get('object',''))+'/'+str(record.get('quantity',''))
        entry=self.summary.setdefault(group,dict(compared=0,passed=0,failed=0,max_absolute_difference=0.,max_relative_difference=0.))
        entry['compared']+=1;entry['passed']+=status=='PASS';entry['failed']+=status!='PASS'
        for field,target in [('absolute_difference','max_absolute_difference'),('relative_difference','max_relative_difference')]:
            v=record.get(field)
            if isinstance(v,(int,float)) and np.isfinite(v):entry[target]=max(entry[target],float(v))
        if np.isinf(record.get('relative_difference',0)):
            entry['infinite_relative_difference_count']=entry.get('infinite_relative_difference_count',0)+1

    def numeric(self,a,b,contract):
        detail=_DETAIL.get();labels=detail.get('labels');columns=detail.get('columns')
        for index in np.ndindex(a.shape):
            left,right=float(a[index]),float(b[index]);both_nan=np.isnan(left) and np.isnan(right)
            delta=abs(left-right) if np.isfinite(left) and np.isfinite(right) else np.nan
            relative=delta/abs(right) if np.isfinite(delta) and right else (0. if delta==0 else np.inf)
            allowed=contract.get('strict_abs_less_than',contract.get('atol',0)+contract.get('rtol',0)*abs(right))
            ok=both_nan or (np.isfinite(delta) and (delta<allowed if 'strict_abs_less_than' in contract else delta<=allowed))
            extra={}
            if labels is not None and index:extra.update(labels[index[0]])
            if columns is not None:
                if len(index)==2:extra.update(row_term=columns[index[0]],column_term=columns[index[1]])
                elif len(index)==1:extra['term']=columns[index[0]]
            self.row(actual=left,expected=right,status='PASS' if ok else 'FAIL',
                absolute_difference=delta,relative_difference=relative,allowed_tolerance=allowed,
                atol=contract.get('atol',''),rtol=contract.get('rtol',''),
                strict_abs_less_than=contract.get('strict_abs_less_than',''),**extra)


def evidence_error(message,category='VALIDATION_FAILURE'):
    evidence=_EVIDENCE.get()
    if evidence: evidence.row(actual=str(message),expected='VALID',status='FAIL',quantity='structural_check',failure_category=category)


def assert_close(actual, expected, contract):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    if a.shape != b.shape:
        evidence_error(f'actual={a.shape}; expected={b.shape}','SHAPE_DRIFT')
        raise ValidationError('SHAPE_OR_NULL_MASK_CHANGED')
    evidence=_EVIDENCE.get()
    if evidence:evidence.numeric(a,b,contract)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        raise ValidationError('SHAPE_OR_NULL_MASK_CHANGED')
    if np.isinf(a).any() or np.isinf(b).any():
        raise ValidationError('NONFINITE_NUMERICAL_VALUE')
    finite = ~np.isnan(a)
    if 'strict_abs_less_than' in contract:
        ok = np.all(np.abs(a[finite]-b[finite]) < contract['strict_abs_less_than'])
    else:
        ok = np.allclose(a, b, atol=contract['atol'], rtol=contract['rtol'], equal_nan=True)
    if not ok: raise ValidationError('NUMERICAL_EQUIVALENCE_FAILED')
    return float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.


def keyed_compare(actual, expected, keys, numerical, contract):
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        evidence_error('Duplicate key: '+','.join(keys),'SAMPLE_OR_MODEL_DRIFT')
        raise ValidationError('DUPLICATE_COMPARISON_KEY')
    a=actual.set_index(keys).sort_index(); b=expected.set_index(keys).sort_index()
    errors=[];result={};evidence=_EVIDENCE.get()
    union=a.index.union(b.index).sort_values()
    if not a.index.equals(b.index):
        errors.append('SAMPLE_OR_MODEL_SET_CHANGED')
        if evidence:
            for key in union:
                if (key in a.index)!=(key in b.index):
                    evidence.row(actual=key in a.index,expected=key in b.index,status='FAIL',quantity='outer_presence',
                                 key=str(key),failure_category='MODEL_OR_TERM_MISSING')
    a=a.reindex(union);b=b.reindex(union)
    labels=[]
    for key in union:
        values=key if isinstance(key,tuple) else (key,)
        fields=dict(zip(keys,values));label={'key':str(fields)}
        for name in ['model_id','term']: 
            if name in fields:label[name]=fields[name]
        if 'term' not in label and 'estimand' in fields:label['term']=fields['estimand']
        if 'sample_hash' in b and key in b.index:label['sample_identity']=b.loc[key,'sample_hash']
        labels.append(label)
    for column in numerical:
        if column not in a or column not in b:
            errors.append('COMPARISON_COLUMN_MISSING:'+column);evidence_error(errors[-1],'SCHEMA_DRIFT');continue
        try:
            with comparison_context(quantity=column,labels=labels):result[column]=assert_close(a[column],b[column],contract)
        except ValidationError as error: errors.append(str(error)+':'+column)
    if errors:raise ValidationError('; '.join(errors))
    return result


def exact_columns(actual, expected, keys, columns):
    """Compare stable scientific definitions, permitting only missing-value encoding."""
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValidationError('DUPLICATE_COMPARISON_KEY')
    a=actual.set_index(keys).sort_index(); b=expected.set_index(keys).sort_index()
    if not a.index.equals(b.index): raise ValidationError('SAMPLE_OR_MODEL_SET_CHANGED')
    for column in columns:
        if column not in a or column not in b:
            raise ValidationError('DEFINITION_COLUMN_MISSING: '+column)
        # pandas nullable strings may differ by dtype, but not by their values.
        av=a[column].astype(object).where(a[column].notna(), None).tolist()
        bv=b[column].astype(object).where(b[column].notna(), None).tolist()
        evidence=_EVIDENCE.get()
        if evidence:
            for key,left,right in zip(a.index,av,bv):
                evidence.row(actual=left,expected=right,status='PASS' if left==right else 'FAIL',
                             quantity=column,key=str(key),failure_category='DESIGN_OR_STATUS_DRIFT')
        if av!=bv: raise ValidationError('SCIENTIFIC_DEFINITION_CHANGED: '+column)


def verify_panel(actual, expected, tolerances):
    if list(actual.columns)!=list(expected.columns): raise ValidationError('PANEL_SCHEMA_CHANGED')
    a=actual.copy();b=expected.copy()
    evidence=_EVIDENCE.get()
    for column in a:
        left,right=a[column].dtype,b[column].dtype
        # Timestamp resolution and integer width are storage representations;
        # exact timestamps/values below remain mandatory and are never waived.
        def family(dtype):
            if isinstance(dtype,pd.DatetimeTZDtype):return 'datetime_utc'
            if pd.api.types.is_bool_dtype(dtype):return 'bool'
            if pd.api.types.is_integer_dtype(dtype):return 'integer'
            if pd.api.types.is_float_dtype(dtype):return 'float'
            if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):return 'string'
            return str(dtype)
        ok=family(left)==family(right)
        if evidence:evidence.row(actual=str(left),expected=str(right),status='PASS' if ok else 'FAIL',
                                 quantity='dtype',term=column,key='storage representation; exact values checked separately',
                                 failure_category='SCHEMA_DRIFT')
        if not ok:raise ValidationError('PANEL_DTYPE_FAMILY_CHANGED:'+column)
    for d in (a,b):
        d['timestamp_utc']=pd.to_datetime(d.timestamp_utc,utc=True).astype('datetime64[ns, UTC]')
    keys=['corridor_id','timestamp_utc']
    ordered=a[keys].reset_index(drop=True).equals(b[keys].reset_index(drop=True))
    if evidence:evidence.row(actual=ordered,expected=True,status='PASS' if ordered else 'FAIL',quantity='ordered_keys',failure_category='TIME_INDEX_DRIFT')
    if not ordered:raise ValidationError('PANEL_KEY_ORDER_CHANGED')
    weather=[c for c in b if c.startswith(('origin_weather_', 'destination_weather_'))]
    numeric=[c for c in b if pd.api.types.is_numeric_dtype(b[c]) and c not in weather]
    categorical=[c for c in b if c not in {*keys,*numeric,*weather}]
    result=keyed_compare(a,b,keys,numeric,tolerances['core_panel'])
    result.update(keyed_compare(a,b,keys,weather,tolerances['weather']))
    exact_columns(a,b,keys,categorical)
    return result


def primary_consistency(models, contract):
    """V3 validate_results primary-coefficient identity, before benchmark QA."""
    for model in models:
        rows=[r for r in model['coefficients'] if r['term']==model['coefficient_name']]
        if len(rows)!=1: raise ValidationError('PRIMARY_COEFFICIENT_REFERENCE')
        for field in ['estimate','SE','CI_low','CI_high','p_value']:
            term='coefficient' if field=='estimate' else field
            assert_close(model[field],rows[0][term],contract)
    return len(models)


def timestamp_proof(actual_ns, expected_ns, frozen_hash):
    a=np.asarray(actual_ns, dtype='<i8'); b=np.asarray(expected_ns, dtype='<i8')
    if not np.array_equal(a,b) or len(a)==0 or not np.all(np.diff(a)>0):
        raise ValidationError('SAMPLE_TIMESTAMPS_CHANGED')
    hashes={hashlib.sha256(b.tobytes()).hexdigest(): 'ns'}
    if np.all(b%1000==0): hashes[hashlib.sha256((b//1000).astype('<i8').tobytes()).hexdigest()]='us'
    if frozen_hash not in hashes: raise ValidationError('SAMPLE_MEMBERSHIP_HASH_CHANGED')
    return hashes[frozen_hash]


def verify_models(actual, expected, contract, *, polynomial=False):
    def index(rows):
        out={m['model_id']:m for m in rows}
        if len(out)!=len(rows): raise ValidationError('DUPLICATE_MODEL_ID')
        return out
    a=index(actual['models']); b=index(expected['models']);evidence=_EVIDENCE.get()
    if a.keys()!=b.keys():
        if evidence:
            for mid in sorted(a.keys()|b.keys()):
                evidence.row(actual=mid in a,expected=mid in b,status='PASS' if mid in a and mid in b else 'FAIL',
                    quantity='model_presence',model_id=mid,failure_category='MODEL_MISSING')
        raise ValidationError('MODEL_SET_CHANGED')
    exact=(['corridor','destination_zone','family','polynomial_order','bandwidth_days','cutoff','cutoff_year',
            'N','N_pre','N_post','HAC','kernel','columns','status','functional_form_status',
            'hour_FE','weekday_FE','weather_controls','number_of_regressors','rank','covariance_status'] if polynomial else
           ['series','outcome','model_family','HAC_lag','N','pre_N','post_N','kernel','coefficient_name',
            'status','weather','hour_FE','weekday_FE','month_FE','week_FE','cutoff','bandwidth','donut',
            'support','weight_version','spec','weather_variant','arm','sample_start','sample_end'])
    maximum=0.;errors=[]
    for mid,x in a.items():
        y=b[mid]
        with comparison_context(model_id=mid,sample_identity=y.get('sample_hash','UNVERIFIED')):
            for key in exact:
                ok=x.get(key)==y.get(key)
                if evidence:evidence.row(actual=x.get(key),expected=y.get(key),quantity=key,status='PASS' if ok else 'FAIL',failure_category='DESIGN_MATRIX_DRIFT')
                if not ok:errors.append('MODEL_DEFINITION_CHANGED: '+mid+'/'+key)
            try:
                if polynomial:
                    with comparison_context(quantity='beta',columns=y['columns']):maximum=max(maximum,assert_close(x['beta'],y['beta'],contract))
                    with comparison_context(quantity='covariance',columns=y['columns'],failure_category='COVARIANCE_DRIFT'):
                        maximum=max(maximum,assert_close(x['covariance'],y['covariance'],contract))
                    for key in ['tau','se','ci_low','ci_high','p_value']:
                        with comparison_context(quantity=key):maximum=max(maximum,assert_close(x[key],y[key],contract))
                    if x['shape_diagnostics']['flags']!=y['shape_diagnostics']['flags']:
                        errors.append('POLYNOMIAL_SHAPE_FLAGS_CHANGED:'+mid)
                else:
                    xa=pd.DataFrame(x['coefficients']);ya=pd.DataFrame(y['coefficients'])
                    if xa['term'].tolist()!=ya['term'].tolist():errors.append('COEFFICIENT_ORDER_CHANGED: '+mid)
                    differences=keyed_compare(xa,ya,['term'],['coefficient','SE','CI_low','CI_high','p_value'],contract)
                    maximum=max(maximum,*differences.values())
                    for key in ['estimate','SE','CI_low','CI_high','p_value']:
                        with comparison_context(quantity='primary_'+key):maximum=max(maximum,assert_close(x[key],y[key],contract))
            except ValidationError as error:errors.append(mid+': '+str(error))
    status_key='not_estimated' if polynomial else 'blocked_models'
    left,right=actual[status_key],expected[status_key]
    if evidence:
        for ordinal in range(max(len(left),len(right))):
            x=left[ordinal] if ordinal<len(left) else None;y=right[ordinal] if ordinal<len(right) else None
            evidence.row(actual=json.dumps(x,sort_keys=True),expected=json.dumps(y,sort_keys=True),
                status='PASS' if x==y else 'FAIL',quantity='not_estimated_status',key=str(ordinal),failure_category='MODEL_STATUS_DRIFT')
    if left!=right:errors.append('MODEL_STATUS_CHANGED')
    if errors:raise ValidationError('; '.join(errors[:20]))
    return dict(model_records=len(a),max_abs_numeric_difference=maximum,model_status='EXACT',not_estimated_records=len(right))


def _read(catalog, aid, reader):
    with catalog.open_input(aid, purpose='verify') as f: return reader(f)


def _confined(root, rel):
    from .inputs import relative_path
    p=Path(root).joinpath(*relative_path(rel).parts)
    if any(x.is_symlink() for x in [p,*p.parents]): raise ValidationError('OUTPUT_SYMLINK')
    return p


def verify_model_covariance_links(directory, models, expected_models, actual_index, expected_index):
    """Bind each result reference to its indexed matrix and frozen archive role.

    The sole path migration is the removed historical engine directory prefix.
    A valid CSV matrix must not conceal a broken JSON reference used by consumers.
    """
    def identity(row):
        return (row['file_path'], row['covariance_key'], tuple(json.loads(row['columns'])))
    def indexed(frame):
        rows={}
        for row in frame.to_dict('records'):
            key=identity(row)
            if key in rows: raise ValidationError('DUPLICATE_COVARIANCE_REFERENCE')
            rows[key]=row
        return rows
    new_index,old_index=indexed(actual_index),indexed(expected_index)
    expected={m['model_id']:m for m in expected_models}
    evidence=_EVIDENCE.get()
    for model in models:
        mid=model['model_id']; ref=model.get('covariance_reference')
        oldref=expected[mid].get('covariance_reference')
        if not isinstance(ref,dict) or not isinstance(oldref,dict):
            raise ValidationError('MODEL_COVARIANCE_REFERENCE_MISSING: '+mid)
        try: row=new_index[identity(ref)]; oldrow=old_index[identity(oldref)]
        except KeyError as error: raise ValidationError('MODEL_COVARIANCE_INDEX_LINK_MISSING: '+mid) from error
        for item,indexrow in [(ref,row),(oldref,oldrow)]:
            for field in ['model_id','HAC_lag','covariance_scope']:
                left,right=item.get(field),indexrow.get(field)
                if field=='HAC_lag': left,right=float(left),float(right)
                if left!=right:
                    raise ValidationError('MODEL_COVARIANCE_JSON_INDEX_MISMATCH: '+mid+'/'+field)
            if item['model_id']!=mid or float(item['HAC_lag'])!=float(model['HAC_lag']):
                raise ValidationError('MODEL_COVARIANCE_MODEL_LINK_CHANGED: '+mid)
        expected_path=oldref['file_path']
        prefix='outputs/models/engine/'
        if expected_path.startswith(prefix): expected_path=expected_path[len(prefix):]
        if ref['file_path']!=expected_path:
            raise ValidationError('MODEL_COVARIANCE_ARCHIVE_ROLE_CHANGED: '+mid)
        for field in ['covariance_key','columns','covariance_scope']:
            left=json.loads(ref[field]) if field=='columns' else ref[field]
            right=json.loads(oldref[field]) if field=='columns' else oldref[field]
            if left!=right: raise ValidationError('MODEL_COVARIANCE_FROZEN_LINK_CHANGED: '+mid+'/'+field)
        if not _confined(directory,ref['file_path']).is_file():
            raise ValidationError('MODEL_COVARIANCE_FILE_MISSING: '+mid)
        if evidence:
            evidence.row(model_id=mid,covariance_key=ref['covariance_key'],quantity='json_index_covariance_reference',
                actual=json.dumps(ref,sort_keys=True),expected=json.dumps(oldref,sort_keys=True),status='PASS',
                failure_category='COVARIANCE_REFERENCE_DRIFT')
    return dict(model_references=len(models),json_index_links='EXACT',archive_role='EXACT_PORTABLE_MAPPING')


def verify_covariance(catalog, directory, actual_index, expected_index, contract, sample_lookup=None):
    """Identity is archive content role + key + ordered columns, never model ID alone."""
    lookup={f['legacy_source']:f['id'] for f in catalog.baselines['files']}
    def keyed(frame):
        out={}
        for r in frame.to_dict('records'):
            columns=json.loads(r['columns'])
            # New file paths are portable, so match keys with scientific scope
            # and ordered columns, then verify every explicit archive reference.
            key=(r['covariance_key'], tuple(columns))
            if key in out: raise ValidationError('DUPLICATE_COVARIANCE_REFERENCE')
            out[key]=r
        return out
    a,b=keyed(actual_index),keyed(expected_index)
    if a.keys()!=b.keys(): raise ValidationError('COVARIANCE_REFERENCE_SET_CHANGED')
    maximum=0.; archives={};cells=0;off_diagonal=0
    try:
        for key,r in a.items():
            old=b[key]; legacy='v3_formal_cbam/'+old['file_path']
            for column in ['model_id','HAC_lag','covariance_scope']:
                left,right=r.get(column),old.get(column)
                if pd.isna(left) and pd.isna(right): continue
                if left!=right: raise ValidationError('COVARIANCE_MODEL_LINK_CHANGED: '+column)
            if legacy not in lookup: raise ValidationError('INCOMPLETE_BENCHMARK_REFERENCE')
            aid=lookup[legacy]
            if aid not in archives:
                # resolve validates content identity before np.load; no writes.
                archives[aid]=np.load(catalog.resolve(aid,purpose='verify'),allow_pickle=False)
            path=_confined(directory,r['file_path'])
            if str(path) not in archives: archives[str(path)]=np.load(path,allow_pickle=False)
            new=archives[str(path)]; old_npz=archives[aid]; covkey=key[0]
            if covkey not in new or covkey not in old_npz: raise ValidationError('COVARIANCE_KEY_MISSING')
            if new[covkey].shape!=(len(key[1]),len(key[1])): raise ValidationError('COVARIANCE_COLUMNS_CHANGED')
            with comparison_context(model_id=r.get('model_id'),covariance_key=covkey,columns=list(key[1]),
                    sample_identity=(sample_lookup or {}).get(r.get('model_id'),'joint/compound sample; see indexed model and ordered block columns'),
                    key=old['file_path'],quantity='covariance',source_module='estimators.hac/joint_covariance',failure_category='COVARIANCE_DRIFT'):
                maximum=max(maximum,assert_close(new[covkey],old_npz[covkey],contract))
            cells+=new[covkey].size;off_diagonal+=new[covkey].size-new[covkey].shape[0]
    finally:
        for item in archives.values(): item.close()
    return dict(references=len(a),cells=cells,off_diagonal_cells=off_diagonal,max_abs_difference=maximum)


def verify_v2f(catalog, directory, actual_models, samples_by_model, contract):
    """Full historical coefficient/covariance QA, ported from validate_full_frozen_outputs.

    V2F is read exclusively after fitting. The complete archive key sets and joint
    blocks are compared. Any us/ns sample encoding difference requires timestamp
    proof from independently captured model membership; no digest is waived.
    """
    tables=Path(directory)/'tables'
    specifications=[('all_coefficients',['model_id','hac_lag','term']),
                    ('weather_abc_results',['model_id','hac_lag','estimand']),
                    ('weather_coefficients',['model_id','hac_lag','term']),
                    ('weather_synthetic_abc_results',['model_id','hac_lag','estimand']),
                    ('weather_synthetic_coefficients',['model_id','hac_lag','term'])]
    source_samples={}
    for m in actual_models:
        source=m.get('source_model_id')
        if source:
            values=samples_by_model[m['model_id']]
            if source in source_samples and not np.array_equal(values,source_samples[source]):
                raise ValidationError('SOURCE_MODEL_SAMPLE_CONFLICT')
            source_samples[source]=values
    report={}
    numeric=['coefficient','se','ci_low','ci_high','p_value']
    for stem,keys in specifications:
        new=pd.read_csv(tables/(stem+'.csv'))
        old=_read(catalog,'v2f_'+stem,pd.read_csv)
        with comparison_context(object='v2f_'+stem,source_module='_empirical/_weather_models/_weather_synthetic',failure_category='ESTIMATOR_DRIFT'):
            report[stem]=keyed_compare(new,old,keys,numeric,contract)
        identity=[c for c in ['series','model','outcome','weight_version','bandwidth_days','donut_days',
            'cutoff','fe','weather_version','weather','kernel','support','spec','status','n','n_pre','n_post',
            'N','N_pre','N_post','rank','year_N','reference_categories','FE_reference','FE','bandwidth','donut',
            'arm','variant','first_utc','last_utc'] if c in old]
        with comparison_context(object='v2f_'+stem,source_module='_empirical/_weather_models/_weather_synthetic'):
            exact_columns(new,old,keys,identity)
        if 'sample_hash' in old:
            for frame in [new,old]:
                for source,group in frame.groupby('model_id',sort=False):
                    if source not in source_samples: raise ValidationError('V2F_SAMPLE_PROOF_MISSING: '+source)
                    values=source_samples[source]
                    for digest in group.sample_hash.drop_duplicates():
                        encoding=timestamp_proof(values,values,digest)
                        evidence=_EVIDENCE.get()
                        if evidence:evidence.row(actual=encoding,expected='timestamp-proven us/ns',status='PASS',object='v2f_'+stem,
                            model_id=source,quantity='sample_identity',sample_identity=digest)
                    if 'sample_hash_utc_ns' in group:
                        for digest in group.sample_hash_utc_ns.drop_duplicates():
                            timestamp_proof(values,values,digest)
    for stem in ['model_covariances','weather_covariances','weather_synthetic_covariances']:
        old_path=catalog.resolve('v2f_'+stem,purpose='verify')
        maximum=0.;cells=0
        with np.load(tables/(stem+'.npz'),allow_pickle=False) as new, np.load(old_path,allow_pickle=False) as old:
            if set(new.files)!=set(old.files): raise ValidationError('V2F_COVARIANCE_KEY_SET_CHANGED')
            for key in old.files:
                with comparison_context(object='v2f_'+stem,covariance_key=key,quantity='covariance',
                        columns=[str(i) for i in range(old[key].shape[0])],
                        source_module='estimators.hac/joint_covariance',failure_category='COVARIANCE_DRIFT'):
                    maximum=max(maximum,assert_close(new[key],old[key],contract));cells+=old[key].size
            report[stem]=dict(matrices=len(old.files),cells=cells,max_abs_difference=maximum)
    # Contrasts use the same joint covariance blocks; retain their numerical QA.
    extra=[('joint_covariance_contrasts',['kind','series_a','series_b','series','weight_version','hac_lag']),
           ('weather_adjustments',['series','model','support','variant','hac_lag','estimand','contrast']),
           ('weather_synthetic_adjustments',['series','model','support','variant','hac_lag','estimand','contrast'])]
    for stem,keys in extra:
        new=pd.read_csv(tables/(stem+'.csv'));old=_read(catalog,'v2f_'+stem,pd.read_csv)
        fields=[c for c in [*numeric,'covariance','relative_weather_adjustment'] if c in old]
        with comparison_context(object='v2f_'+stem,source_module='estimators.contrast/joint_covariance',failure_category='COVARIANCE_DRIFT'):
            report[stem]=keyed_compare(new,old,keys,fields,contract)
            exact_columns(new,old,keys,[c for c in ['joint_cross_covariance_retained','n'] if c in old])
    return report


def _verify_baseline_complete(catalog,directory,*,spec,panel_path,supplemental_references=None):
    if spec not in {'main','polynomial'}:raise ValidationError('UNKNOWN_SPEC')
    directory=Path(directory);actual=json.loads((directory/'results.json').read_text())
    if actual.get('mode')=='demo':raise ValidationError('DEMO_IS_NOT_FROZEN_REPLICATION')
    expected=_read(catalog,'v3_results' if spec=='main' else 'polynomial_results',json.load)
    tolerances=catalog.baselines['validation'];result={};errors=[]
    def check(name,callback,*,module='validation',category='ESTIMATOR_DRIFT'):
        with comparison_context(object=name,source_module=module,failure_category=category):
            try:result[name]=callback();return result[name]
            except (ValidationError,KeyError,ValueError,FileNotFoundError) as error:
                evidence_error(str(error),category);errors.append(dict(check=name,error=str(error),failure_category=category));return None
    check('models',lambda:verify_models(actual,expected,tolerances['full_frozen'],polynomial=spec=='polynomial'),module='models/polynomial')
    if spec=='main':check('primary_consistency',lambda:primary_consistency(actual['models'],tolerances['primary_internal']),module='_assembly')
    panel=pd.read_parquet(panel_path);frozen_panel=_read(catalog,'v3_panel',pd.read_parquet)
    check('panel',lambda:verify_panel(panel,frozen_panel,tolerances),module='market/panel/weather',category='DATA_OR_TIME_INDEX_DRIFT')
    proof=check('samples',lambda:sample_proofs(actual,expected,directory,frozen_panel),module='design/polynomial.sample',category='SAMPLE_DRIFT')
    samples_by_model=proof[1] if proof is not None else None
    if proof is not None:result['samples']=dict(proven=len(proof[0]),encoding_counts=pd.Series(proof[0]).value_counts().to_dict())
    index=pd.read_csv(directory/'model_sample_index.csv')
    if set(index.model_id)!=set(m['model_id'] for m in expected['models']):
        errors.append(dict(check='samples',error='MODEL_SAMPLE_REFERENCE_SET_CHANGED',failure_category='SAMPLE_DRIFT'))
    if spec=='main':
        if samples_by_model is not None:
            check('actual_fit_design',lambda:verify_design_signatures(directory,actual['models'],samples_by_model),module='estimators.fit/design',category='DESIGN_MATRIX_DRIFT')
        new_coef=pd.read_csv(directory/'coefficients.csv');old_coef=_read(catalog,'v3_coefficients',pd.read_csv)
        check('coefficients',lambda:keyed_compare(new_coef,old_coef,['model_id','term'],
            ['coefficient','SE','CI_low','CI_high','p_value'],tolerances['full_frozen']),module='estimators/_assembly')
        check('coefficient_identity',lambda:exact_columns(new_coef,old_coef,['model_id','term'],['N','HAC_lag']),module='_assembly',category='SAMPLE_OR_DESIGN_DRIFT')
        check('covariance_links',lambda:verify_model_covariance_links(directory,actual['models'],expected['models'],
            pd.read_csv(directory/'covariance_index.csv'),_read(catalog,'v3_covariance_index',pd.read_csv)),
            module='_assembly/covariance_index',category='COVARIANCE_REFERENCE_DRIFT')
        check('covariance',lambda:verify_covariance(catalog,directory,pd.read_csv(directory/'covariance_index.csv'),
            _read(catalog,'v3_covariance_index',pd.read_csv),tolerances['full_frozen'],
            {m['model_id']:m['sample_hash'] for m in actual['models']}),module='estimators.hac/joint_covariance',category='COVARIANCE_DRIFT')
        if samples_by_model is not None:
            check('v2f',lambda:verify_v2f(catalog,directory,actual['models'],samples_by_model,tolerances['full_frozen']),module='legacy_validate_full_frozen_outputs')
        new_catalog=pd.read_csv(directory/'model_catalog.csv');old_catalog=_read(catalog,'v3_catalog',pd.read_csv)
        check('catalog',lambda:exact_columns(new_catalog,old_catalog,['model_id'],[
            'series','outcome','model_family','cutoff','bandwidth','donut','kernel','HAC_lag',
            'hour_FE','weekday_FE','month_FE','week_FE','weather','support','weight_version',
            'sample_start','sample_end','N','pre_N','post_N','coefficient_name','status','spec',
            'weather_variant','arm','reason','implementation_status']),module='_assembly',category='MODEL_STATUS_OR_DESIGN_DRIFT')
        result['inventory']=dict(estimated_records=len(actual['models']),catalog_rows=len(new_catalog),
                                 coefficient_rows=len(new_coef),not_estimated_records=len(actual['blocked_models']))
        check('design_diagnostics',lambda:verify_design_diagnostics(catalog,directory,tolerances),module='design/weather_models')
    else:
        for order in (1,2,3):
            check('polynomial_p'+str(order),lambda order=order:_verify_polynomial_order(order,actual,directory,catalog,supplemental_references),module='polynomial')
        result['inventory']=dict(estimated_records=len(actual['models']),not_estimated_records=len(actual['not_estimated']))
    blocked=errors and all('BLOCKED_' in e['error'] or 'UNVERIFIED_' in e['error'] for e in errors)
    if result.get('models') is not None:
        result.update({key:result['models'][key] for key in ['model_records','max_abs_numeric_difference','model_status']})
    if result.get('samples') is not None:result['sample_proofs']=result['samples']['proven']
    return dict(status=('BLOCKED' if blocked else 'FAIL') if errors else 'PASS',scope='complete independently frozen saved-output comparison',
                errors=errors,failed_checks=len(errors),**result)


def verify_baseline(catalog,directory,*,spec,panel_path,evidence_dir=None,supplemental_references=None):
    """Validate saved outputs with optional complete private per-cell evidence."""
    def perform():return _verify_baseline_complete(catalog,directory,spec=spec,panel_path=panel_path,
                                                  supplemental_references=supplemental_references)
    if evidence_dir is None:return perform()
    path=Path(evidence_dir).resolve()
    if hasattr(catalog,'input_root'):
        from .inputs import disjoint
        disjoint(path,catalog.input_root)
        if not path.is_relative_to(catalog.output_root.resolve()):raise ValidationError('EVIDENCE_OUTSIDE_PRIVATE_OUTPUT_ROOT')
    with Evidence(path):return perform()


def compare_nested(actual, expected, contract, path=''):
    """Compare complete saved shape diagnostics, including every turning point."""
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or actual.keys()!=expected.keys():
            evidence_error(path,'SHAPE_DIAGNOSTIC_STRUCTURE_DRIFT');raise ValidationError('SHAPE_DIAGNOSTIC_STRUCTURE_CHANGED:'+path)
        return max([compare_nested(actual[k],v,contract,path+'/'+k) for k,v in expected.items()]+[0.])
    if isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):
            evidence_error(path,'SHAPE_DIAGNOSTIC_STRUCTURE_DRIFT');raise ValidationError('SHAPE_DIAGNOSTIC_LENGTH_CHANGED:'+path)
        return max([compare_nested(a,b,contract,path+'/'+str(i)) for i,(a,b) in enumerate(zip(actual,expected))]+[0.])
    with comparison_context(quantity=path):
        if isinstance(expected,(int,float)) and not isinstance(expected,bool):
            return assert_close(actual,expected,contract)
        ok=actual==expected;evidence=_EVIDENCE.get()
        if evidence:evidence.row(actual=actual,expected=expected,status='PASS' if ok else 'FAIL',failure_category='SHAPE_STATUS_DRIFT')
        if not ok:raise ValidationError('SHAPE_DIAGNOSTIC_IDENTITY_CHANGED:'+path)
    return 0.


def elapsed_polynomial(model):
    """Independent diagonal parameter transform for z=r/h; no estimator call."""
    names=[];scale=[];h=model['bandwidth_days']
    for name in model['columns']:
        if name.startswith('post_z') and name[6:].isdigit():
            power=int(name[6:]);names.append('post_r' if power==1 else f'post_r{power}');scale.append(h**(-power))
        elif name.startswith('z') and name[1:].isdigit():
            power=int(name[1:]);names.append('r' if power==1 else f'r{power}');scale.append(h**(-power))
        else:names.append(name);scale.append(1.)
    d=np.asarray(scale,float)
    return names,np.asarray(model['beta'])*d,np.asarray(model['covariance'])*d[:,None]*d[None,:]


def sample_proofs(actual,expected,directory,frozen_panel):
    directory=Path(directory);sample_path=directory/'sample_timestamps.parquet';index_path=directory/'model_sample_index.csv'
    if not sample_path.is_file() or not index_path.is_file():raise ValidationError('UNVERIFIED_SAMPLE_IDENTITY')
    index=pd.read_csv(index_path).set_index('model_id')
    if index.index.has_duplicates:raise ValidationError('DUPLICATE_SAMPLE_REFERENCE')
    current={m['model_id']:m for m in actual['models']};frozen={m['model_id']:m for m in expected['models']}
    # Order validation can receive cumulative sidecars, but every selected model
    # must be present and the final full verification requires exact membership.
    if not set(frozen)<=set(index.index):raise ValidationError('MODEL_SAMPLE_REFERENCE_SET_CHANGED')
    table=pd.read_parquet(sample_path)
    if table.duplicated(['sample_hash_utc_ns','timestamp_utc']).any():raise ValidationError('DUPLICATE_SAMPLE_TIMESTAMP')
    arrays={str(k):pd.DatetimeIndex(g.timestamp_utc).as_unit('ns').asi8 for k,g in table.groupby('sample_hash_utc_ns',sort=False)}
    for key,values in arrays.items():
        if hashlib.sha256(np.asarray(values,dtype='<i8').tobytes()).hexdigest()!=key:raise ValidationError('CANONICAL_SAMPLE_HASH_CHANGED')
    result={};samples={};evidence=_EVIDENCE.get()
    for mid,model in frozen.items():
        key=str(index.loc[mid,'sample_hash_utc_ns'])
        if key not in arrays:raise ValidationError('SAMPLE_ARRAY_MISSING:'+mid)
        values=arrays[key]
        if current[mid]['sample_hash']!=key:raise ValidationError('MODEL_SAMPLE_LINK_CHANGED:'+mid)
        code=(model.get('corridor') or model.get('series','ME').split('_')[0]).replace('_IT','')
        if code in {'G','ME'}:code='ME'
        old=pd.DatetimeIndex(frozen_panel.loc[frozen_panel.corridor_id.eq(code+'_IT'),'timestamp_utc']).as_unit('ns').asi8
        recovered=old[np.isin(old,values)]
        with comparison_context(model_id=mid,sample_identity=model['sample_hash'],quantity='sample_identity',failure_category='SAMPLE_DRIFT'):
            result[mid]=timestamp_proof(values,recovered,model['sample_hash'])
            if evidence:evidence.row(actual=key,expected=model['sample_hash'],status='PASS',key='actual timestamps verified; '+result[mid])
        samples[mid]=values
    return result,samples


def _supplemental_csv(catalog,asset_id,references):
    if references is None or asset_id not in references:raise ValidationError('BLOCKED_SUPPLEMENTAL_REFERENCE_MISSING:'+asset_id)
    asset=catalog.assets[asset_id]
    if asset['classification']!='P':raise ValidationError('SUPPLEMENTAL_REFERENCE_ROLE_CHANGED')
    path=Path(references[asset_id])
    if not path.is_absolute() or any(p.is_symlink() for p in [path,*path.parents]):raise ValidationError('SUPPLEMENTAL_REFERENCE_PATH_UNSAFE')
    if not path.is_file():raise ValidationError('BLOCKED_SUPPLEMENTAL_REFERENCE_MISSING:'+asset_id)
    if path.stat().st_size!=asset['size_bytes']:raise ValidationError('SUPPLEMENTAL_REFERENCE_SIZE_CHANGED:'+asset_id)
    with path.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=asset['sha256']:raise ValidationError('SUPPLEMENTAL_REFERENCE_HASH_CHANGED:'+asset_id)
    return pd.read_csv(path)


def verify_polynomial_parent_p1(models,catalog,samples_by_model):
    frozen=_read(catalog,'v3_results',json.load);parents={m['model_id']:m for m in frozen['models']}
    lookup={f['legacy_source']:f['id'] for f in catalog.baselines['files']}
    tolerances=catalog.baselines['validation'];rows=[]
    for model in models:
        if model['polynomial_order']!=1:continue
        code=model['corridor'];h=model['bandwidth_days']
        if model['cutoff_year']==2025:parent_id=f'V3_PLACEBO2025_{code}'
        elif h==90:parent_id=f'V3_{model["family"]}_{code}'
        else:parent_id=f'V3__R1__{code}_actual__actual__scheduled_export_proxy__{h}__0__2026__hour+weekday__full__main'
        if parent_id not in parents:raise ValidationError('P1_FROZEN_PARENT_MISSING:'+parent_id)
        parent=parents[parent_id];values=samples_by_model[model['model_id']]
        timestamp_proof(values,values,parent['sample_hash'])
        for left,right in [('N','N'),('N_pre','pre_N'),('N_post','post_N')]:
            if model[left]!=parent[right]:raise ValidationError('P1_FROZEN_PARENT_SAMPLE_CHANGED')
        names,beta,cov=elapsed_polynomial(model)
        with comparison_context(object='polynomial_p1_frozen_v3',model_id=model['model_id'],sample_identity=parent['sample_hash'],
                key=parent_id,source_module='polynomial.fit_model/elapsed_polynomial',failure_category='ESTIMATOR_DRIFT'):
            with comparison_context(quantity='tau_SE'):
                primary=assert_close([model['tau'],model['se']],[parent['estimate'],parent['SE']],tolerances['polynomial_p1_parent_tau_se'])
            terms=[r['term'] for r in parent['coefficients']]
            if set(names)!=set(terms):raise ValidationError('P1_FROZEN_PARENT_PARAMETER_SET_CHANGED')
            order=[names.index(term) for term in terms]
            with comparison_context(quantity='transformed_beta',columns=terms):
                error=assert_close(beta[order],[r['coefficient'] for r in parent['coefficients']],tolerances['polynomial_p1_all_coefficients'])
            ref=parent['covariance_reference'];aid=lookup['v3_formal_cbam/'+ref['file_path']]
            columns=json.loads(ref['columns']);order=[names.index(term) for term in columns]
            with np.load(catalog.resolve(aid,purpose='verify'),allow_pickle=False) as archive:
                with comparison_context(quantity='transformed_covariance',columns=columns,covariance_key=ref['covariance_key'],failure_category='COVARIANCE_DRIFT'):
                    covariance_error=assert_close(cov[np.ix_(order,order)],archive[ref['covariance_key']],tolerances['full_frozen'])
        rows.append(dict(model_id=model['model_id'],frozen_parent_id=parent_id,status='PASS',
                         tau_se_max_abs=primary,beta_max_abs=error,covariance_max_abs=covariance_error))
    return rows


def _verify_polynomial_order(order,obj,output_dir,catalog,supplemental_references):
    if order not in (1,2,3):raise ValidationError('INVALID_POLYNOMIAL_ORDER')
    baseline=_read(catalog,'polynomial_results',json.load)
    actual={**obj,'models':[m for m in obj['models'] if m['polynomial_order']==order]}
    expected={**baseline,'models':[m for m in baseline['models'] if m['polynomial_order']==order]}
    contract=catalog.baselines['validation']['full_frozen']
    with comparison_context(object=f'polynomial_p{order}',source_module='polynomial.fit_model',failure_category='ESTIMATOR_DRIFT'):
        report=verify_models(actual,expected,contract,polynomial=True)
    frozen_panel=_read(catalog,'v3_panel',pd.read_parquet)
    with comparison_context(object=f'polynomial_p{order}',source_module='polynomial.sample'):
        proofs,samples=sample_proofs(actual,expected,output_dir,frozen_panel)
    report['sample_proofs']=len(proofs)
    old={m['model_id']:m for m in expected['models']}
    maximum=0.
    for model in actual['models']:
        ref=old[model['model_id']]
        with comparison_context(object=f'polynomial_p{order}',model_id=model['model_id'],sample_identity=ref['sample_hash'],
                source_module='polynomial.shape_qa',failure_category='SHAPE_DIAGNOSTIC_DRIFT'):
            maximum=max(maximum,compare_nested(model['shape_diagnostics'],ref['shape_diagnostics'],contract,'shape'))
            names,beta,cov=elapsed_polynomial(model);old_names,old_beta,old_cov=elapsed_polynomial(ref)
            if names!=old_names:raise ValidationError('TRANSFORMED_POLYNOMIAL_COLUMNS_CHANGED')
            with comparison_context(quantity='transformed_beta',columns=names):assert_close(beta,old_beta,contract)
            with comparison_context(quantity='transformed_covariance',columns=names,failure_category='COVARIANCE_DRIFT'):assert_close(cov,old_cov,contract)
            for field in ['condition_number','weighted_R2','raw_condition_number','covariance_min_eigenvalue',
                          'raw_scaled_tau_difference','raw_scaled_max_fitted_difference','qr_max_coefficient_difference',
                          'cutoff_prediction_tau_difference']:
                with comparison_context(quantity=field):assert_close(model[field],ref[field],contract)
            if model['weight_hash']!=ref['weight_hash']:raise ValidationError('POLYNOMIAL_KERNEL_WEIGHT_CHANGED')
            if model['reference_categories']!=ref['reference_categories']:raise ValidationError('POLYNOMIAL_FE_REFERENCE_CHANGED')
    report['shape_max_abs_difference']=maximum
    if order==1:report['independent_frozen_v3_parent']=verify_polynomial_parent_p1(actual['models'],catalog,samples)
    for kind,keys,numeric in [('plot_curves',['model_id','post','r_days','z'],['fitted','se','ci_low','ci_high']),
                              ('daily_bins',['model_id','bin_day'],['r_days','raw_mean','adjusted_mean','N'])]:
        aid=f'polynomial_{kind}_p{order}'
        old_frame=_supplemental_csv(catalog,aid,supplemental_references)
        new=pd.read_csv(Path(output_dir)/'tables'/f'{kind}_P{order}.csv')
        with comparison_context(object=aid,source_module='polynomial.shape_qa',failure_category='REPORTING_NUMERIC_DRIFT'):
            report[kind]=keyed_compare(new,old_frame,keys,numeric,contract)
            if kind=='daily_bins':exact_columns(new,old_frame,keys,['N'])
    return dict(status='PASS',order=order,**report)


def validate_polynomial_order(order,obj,output_dir,catalog,*,supplemental_references=None,evidence_dir=None):
    """Post-fit independent order gate, called before any next-order estimator."""
    if evidence_dir is not None:
        with Evidence(evidence_dir):return _verify_polynomial_order(order,obj,output_dir,catalog,supplemental_references)
    return _verify_polynomial_order(order,obj,output_dir,catalog,supplemental_references)


def verify_design_diagnostics(catalog,directory,tolerances):
    """Compare executed design, sample-pairing, centering and support diagnostics.

    File/code identities intentionally differ after refactoring. Their original
    and new identities remain in receipts; they are not scientific matrix values.
    Sample digests are handled only by timestamp_proofs, never string coercion.
    """
    cases=[('model_catalog','engine_model_catalog',['model_id'],['condition_number','r2']),
        ('weather_model_design','engine_weather_model_design',['model_id'],['condition_number','r2']),
        ('weather_synthetic_design','engine_weather_synthetic_design',['model_id'],['condition_number','r2']),
        ('weather_sample_pairing','engine_weather_sample_pairing',['series','model','support','variant'],[]),
        ('weather_synthetic_pairing','engine_weather_synthetic_pairing',['series','model','variant'],[]),
        ('weather_membership','engine_weather_membership',['series','model','support','variant','timestamp_utc'],[]),
        ('weather_centering_constants','engine_weather_centering_constants',['series','model','geography','variable'],['center']),
        ('weather_synthetic_centers','engine_weather_synthetic_centers',['geography','variable'],['center']),
        ('weather_omitted_columns','engine_weather_omitted_columns',['series','model','support','variant','column'],[]),
        ('weather_synthetic_omitted','engine_weather_synthetic_omitted',[],[])]
    result={}
    def csv_file(reader):
        try:return reader()
        except pd.errors.EmptyDataError:return pd.DataFrame()
    for stem,aid,keys,numeric in cases:
        new=csv_file(lambda:pd.read_csv(Path(directory)/'diagnostics'/(stem+'.csv')))
        old=csv_file(lambda:_read(catalog,aid,pd.read_csv))
        if new.empty and old.empty:result[stem]=dict(rows=0,status='PASS');continue
        if not keys:raise ValidationError('UNEXPECTED_OMITTED_DESIGN_COLUMNS')
        if 'timestamp_utc' in keys:
            for frame in [new,old]:frame['timestamp_utc']=pd.to_datetime(frame.timestamp_utc,utc=True).astype('datetime64[ns, UTC]')
        ignored=[c for c in old if 'hash' in c or c=='sample_hash_integer_unit']
        exact=[c for c in old if c not in {*keys,*numeric,*ignored}]
        contract=tolerances['weather'] if numeric==['center'] else tolerances['full_frozen']
        with comparison_context(object='design_'+stem,source_module='_empirical/_weather_models/_weather_synthetic',failure_category='DESIGN_MATRIX_DRIFT'):
            result[stem]=dict(rows=len(new),numeric=keyed_compare(new,old,keys,numeric,contract))
            exact_columns(new,old,keys,exact)
    new=pd.read_csv(Path(directory)/'tables/support_restrictions.csv')
    old=_read(catalog,'engine_support_restrictions',pd.read_csv)
    keys=['series','weight_version','support','year']
    for frame in [new,old]:frame['ordinal']=frame.groupby(keys,dropna=False).cumcount()
    keys=keys+['ordinal'];numeric=['lower','upper','deletion_rate','hours_per_eur_mwh']
    with comparison_context(object='design_support_restrictions',source_module='design.supports',failure_category='SAMPLE_OR_SUPPORT_DRIFT'):
        result['support']=keyed_compare(new,old,keys,numeric,tolerances['core_panel'])
        exact_columns(new,old,keys,['n_original','n_kept','n_deleted','status'])
    return result


def verify_design_signatures(directory,models,samples_by_model):
    """Check actual-fit signatures against exact sample/kernel/FE definitions.

    X/y hashes prove captured run identity and are compared across independent
    runs; they are never presented as pre-existing frozen design-array hashes.
    """
    path=Path(directory)/'model_design_signatures.csv'
    if not path.is_file():raise ValidationError('BLOCKED_MODEL_DESIGN_SIGNATURES_MISSING')
    signatures=pd.read_csv(path).set_index('model_id')
    if signatures.index.has_duplicates or set(signatures.index)!=set(m['model_id'] for m in models):
        raise ValidationError('MODEL_DESIGN_SIGNATURE_SET_CHANGED')
    evidence=_EVIDENCE.get()
    for model in models:
        mid=model['model_id'];row=signatures.loc[mid];values=samples_by_model[mid]
        time=pd.DatetimeIndex(values,tz='UTC');local=time.tz_convert('Europe/Rome')
        if model['kernel']=='triangular':
            cutoff=pd.Timestamp(model['cutoff']).tz_convert('UTC')
            r=(time+pd.Timedelta(minutes=30)-cutoff).total_seconds()/86400
            weights=1-np.abs(r)/model['bandwidth']
        elif model['kernel']=='uniform':weights=np.ones(len(time),dtype=float)
        else:raise ValidationError('UNKNOWN_FROZEN_KERNEL')
        digest=hashlib.sha256(np.ascontiguousarray(weights,dtype='<f8').tobytes()).hexdigest()
        names=json.loads(row['columns']);references=json.loads(row['FE_references'])
        expected_columns=json.loads(model['covariance_reference']['columns'])
        levels={'hour':local.hour,'weekday':local.weekday,'month':local.month,'week':local.isocalendar().week.to_numpy()}
        expected_references={name:str(np.min(value)) for name,value in levels.items() if model.get(name+'_FE')}
        checks=dict(weights_sha256=(row['weights_sha256'],digest),columns=(names,expected_columns),
            FE_references=(references,expected_references),N=(row['N'],len(values)),rank=(row['rank'],len(names)),
            sample_hash_utc_ns=(row['sample_hash_utc_ns'],hashlib.sha256(np.asarray(values,dtype='<i8').tobytes()).hexdigest()),
            encoding=(row['encoding'],'little_endian_float64_C_order'))
        with comparison_context(object='actual_fit_design_signatures',model_id=mid,sample_identity=model['sample_hash'],
                source_module='estimators.fit/design.local',failure_category='DESIGN_OR_KERNEL_DRIFT'):
            for field,(left,right) in checks.items():
                ok=left==right
                if evidence:evidence.row(actual=left,expected=right,status='PASS' if ok else 'FAIL',quantity=field)
                if not ok:raise ValidationError('ACTUAL_FIT_DESIGN_SIGNATURE_CHANGED:'+mid+'/'+field)
        for field in ['design_sha256','outcome_sha256']:
            digest=str(row[field])
            if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValidationError('INVALID_CAPTURED_ARRAY_IDENTITY')
    return dict(models=len(signatures),kernel='exact reconstructed frozen weights',FE='exact frozen references and columns',
                design_outcome='actual-fit hashes captured; independent cross-run equality required')
