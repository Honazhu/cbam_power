"""Explicit engine context; adapted from V3/src/reused/common.py.

No imported configuration, legacy paths, mutable global roots or benchmark reads.
The caller supplies already validated processing inputs and an isolated run directory.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib, json, math
import numpy as np
import pandas as pd
Y = 'scheduled_export_proxy'
ACTUALS = ['AT_IT', 'CH_IT', 'SI_IT', 'GR_IT', 'ME_IT']
FIELDS = ['temp_c', 'wind_ms', 'ssrd_processed', 'tp_processed']
DESTINATIONS = {'AT':'IT-NORD','CH':'IT-NORD','SI':'IT-NORD','GR':'IT-SUD','ME':'IT-CSUD'}
CFG = {'window_utc_end_exclusive':'2026-09-16T22:00:00Z', 'cutoff':'2026-01-01T00:00:00+01:00',
       'bandwidth_days':[14,30,60,90,180], 'donut_days':[0,3], 'hac_elapsed_hours':[168,24]}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float) and not math.isfinite(v):return None
 return v
def sample_hash(t):return hashlib.sha256(pd.DatetimeIndex(t).asi8.tobytes()).hexdigest()
def canonical_sample_hash(t):return hashlib.sha256(pd.DatetimeIndex(t).as_unit('ns').asi8.tobytes()).hexdigest()
def write_json(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')
@dataclass
class EngineContext:
 output: Path
 frame: pd.DataFrame
 weather: pd.DataFrame
 synthetic: pd.DataFrame
 weights: pd.DataFrame
 synthetic_summary: dict
 source_hashes: dict
 annual_times: pd.DatetimeIndex
 weights_default: pd.DataFrame
 def __post_init__(self):
  self.output=Path(self.output).resolve()
  self.samples={};self.model_samples={};self.designs={}
  for name in ['tables','diagnostics','metadata','reporting','outputs/models','outputs/tables/machine']:
   (self.output/name).mkdir(parents=True,exist_ok=True)
  if set(self.frame.corridor_id.unique()) != set(ACTUALS): raise ValueError('FORMAL_ACTUAL_CORRIDORS_MISMATCH')
  if self.frame.duplicated(['corridor_id','timestamp_utc']).any():raise ValueError('DUPLICATE_PANEL_KEY')
  if self.weather.duplicated(['geography','timestamp_utc']).any():raise ValueError('DUPLICATE_WEATHER_KEY')
  if self.synthetic.duplicated(['weight_version','timestamp_utc']).any():raise ValueError('DUPLICATE_SYNTHETIC_KEY')
  for name in ['v2_panel','panel','weather','synthetic','weights']:
   if name not in self.source_hashes:raise ValueError('MISSING_SOURCE_IDENTITY:'+name)
 def panel(self):
  return self.frame[[c for c in self.frame if not c.startswith(('origin_weather_','destination_weather_'))]].copy()
 def csv(self,path,value):
  p=self.output/path;p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(value).to_csv(p,index=False)
 def js(self,path,value):write_json(self.output/path,value)
 def machine(self,name,value):
  self.csv('outputs/tables/machine/'+name,value)
  if name=='T_coefficients.csv':self.csv('coefficients.csv',value)
  if name=='T_covariance.csv':self.csv('covariance_index.csv',value)
 def capture_sample(self,model_id,timestamps,fit):
  # Bind evidence to the actual design given to the frozen estimator. This is
  # diagnostic capture only; no values, sample, weights or covariance change.
  actual_times=pd.DatetimeIndex(fit.sample.timestamp_utc).as_unit('ns')
  provided_times=pd.DatetimeIndex(timestamps).as_unit('ns')
  if not np.array_equal(actual_times.asi8,provided_times.asi8):raise ValueError('FIT_SAMPLE_PROOF_MISMATCH:'+model_id)
  if fit.x.shape!=(len(fit.y),len(fit.names)) or len(fit.weights)!=len(fit.y):raise ValueError('FIT_DESIGN_SHAPE_MISMATCH:'+model_id)
  def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value,dtype='<f8').tobytes()).hexdigest()
  signature=dict(N=len(fit.y),rank=fit.audit['rank'],columns=json.dumps(fit.names),
   FE_references=json.dumps(fit.audit['references'],sort_keys=True),
   weights_sha256=array_sha(fit.weights),design_sha256=array_sha(fit.x),outcome_sha256=array_sha(fit.y),
   encoding='little_endian_float64_C_order')
  if model_id in self.designs and self.designs[model_id]!=signature:raise ValueError('MODEL_DESIGN_CHANGED:'+model_id)
  self.designs[model_id]=signature
  times=pd.DatetimeIndex(timestamps).as_unit('ns');key=canonical_sample_hash(times)
  if model_id in self.model_samples and self.model_samples[model_id]!=key:raise ValueError('MODEL_SAMPLE_CHANGED:'+model_id)
  self.model_samples[model_id]=key
  if key not in self.samples:self.samples[key]=times
 def export_samples(self,models,id_map):
  rows=[];design_rows=[]
  for m in models:
   source=m.get('source_model_id',m['model_id'].removesuffix('_HAC24'))
   key=self.model_samples[source]
   if key!=m['sample_hash']:raise ValueError('SAMPLE_HASH_ENCODING_NOT_CANONICAL:'+m['model_id'])
   rows.append(dict(model_id=m['model_id'],sample_hash_utc_ns=key))
   if source not in self.designs:raise ValueError('MISSING_FIT_DESIGN_PROOF:'+source)
   design_rows.append(dict(model_id=m['model_id'],source_model_id=source,sample_hash_utc_ns=key,**self.designs[source]))
  self.csv('model_sample_index.csv',rows)
  self.csv('model_design_signatures.csv',design_rows)
  table=pd.concat([pd.DataFrame({'sample_hash_utc_ns':key,'timestamp_utc':times}) for key,times in self.samples.items()],ignore_index=True)
  table.to_parquet(self.output/'sample_timestamps.parquet',index=False)

 def identity(self,name):return self.source_hashes[name]
