"""Port: V3/src/reused/empirical.py. Full no-weather grid; explicit context."""
import json,hashlib
import numpy as np
import pandas as pd
from ._engine_context import Y,ACTUALS,CFG,sha,canonical_sample_hash
from pathlib import Path
from .design import calendar,local,seasonal,supports,F1_CONTRASTS,F2_TERMS,F2_CONTRASTS
from .estimators import fit,NotEstimable,joint_covariance,contrast
ZONES={'AT_IT':'IT-NORD','CH_IT':'IT-NORD','SI_IT':'IT-NORD','GR_IT':'IT-SUD','ME_IT':'IT-CSUD'}
class Results:
 def __init__(self,context):self.context=context;self.rows=[];self.audits=[];self.arrays={};self.index=[];self.support=[];self.joint=[];self.gap_checks=[];self.status=[]
 def record(self,q,outcome,terms,*,series,model,version='actual',bw=0,donut=0,cutoff=2026,fe=('hour','weekday'),weighted=False,support='full',spec='main',contrasts=None):
  modelid='__'.join(map(str,['V2F',model,series,version,outcome,bw,donut,cutoff,'+'.join(fe),support,spec]));gap=series=='G_ME';synthetic=series=='ME_synthetic'
  kind='relative_deviation' if gap else 'synthetic_descriptive' if synthetic else 'actual'
  meta=dict(model_id=modelid,series=series,model=model,weight_version=version,data_hash=self.context.identity("v2_panel"),code_version_hash=sha(Path(__file__)),outcome=outcome,bandwidth_days=bw,donut_days=donut,cutoff=cutoff,fe='+'.join(fe),weather_version='NO_WEATHER',weather='NO_WEATHER',kernel='triangular' if weighted else 'uniform',support=support,spec=spec,series_type=kind,weight_uncertainty='conditional_on_estimated_weights' if version!='actual' else 'not_applicable',validation_status='DESCRIPTIVE_ONLY_WEAK_PREFIT_WEAK_SUPPORT' if synthetic or gap else 'NOT_APPLICABLE',interpretation='relative deviation from fixed EU-internal benchmark' if gap else 'index-response association' if synthetic and model=='F2' else 'descriptive synthetic benchmark' if synthetic else 'local time discontinuity' if model.startswith('R1') else 'same-season conditional level difference' if model=='F1' else 'spread-flow conditional association')
  try:f=fit(q,outcome,terms,fe,weight='triangular' if weighted else None)
  except NotEstimable as e:
   self.audits.append(dict(meta,status=e.status,**e.audit));self.status.append(dict(meta,status=e.status,reason='Design/data gate failed'))
   for lag in CFG['hac_elapsed_hours']:
    for term in (contrasts or {'post':{'post':1}}):self.rows.append(dict(meta,hac_lag=lag,term=term,status=e.status,coefficient=None,se=None,ci_low=None,ci_high=None,p_value=None))
   return None
  self.context.capture_sample(modelid,f.sample.timestamp_utc,f)
  audit=dict(meta,status='ESTIMATED',**f.audit,sample_hash=f.sample_hash,sample_hash_utc_ns=canonical_sample_hash(f.sample.timestamp_utc),sample_hash_integer_unit=str(pd.DatetimeIndex(f.sample.timestamp_utc).dtype),first_utc=str(f.sample.timestamp_utc.min()),last_utc=str(f.sample.timestamp_utc.max()),year_N={str(k):int(v) for k,v in f.sample.groupby('year').size().items()},missing_elapsed_hours=int((f.sample.timestamp_utc.max()-f.sample.timestamp_utc.min()).total_seconds()/3600)+1-len(f.y))
  self.audits.append(audit)
  info=dict(meta,status='ESTIMATED_DESCRIPTIVE',n=len(f.y),n_pre=f.audit['n_pre'],n_post=f.audit['n_post'],r2=f.audit['r2'],r2_type=f.audit['r2_type'],sample_hash=f.sample_hash,sample_hash_utc_ns=audit['sample_hash_utc_ns'],rank=f.audit['rank'],condition_number=f.audit['condition_number'],year_N=json.dumps(audit['year_N']),reference_categories=json.dumps(f.audit['references']))
  for lag in CFG['hac_elapsed_hours']:
   cov=f.covariance(lag);key='cov_'+str(len(self.arrays));self.arrays[key]=cov;self.index.append(dict(key=key,model_id=modelid,hac_lag=lag,columns=json.dumps(f.names)))
   for term in f.names:self.rows.append(dict(info,hac_lag=lag,term=term,**f.contrast(cov,{term:1})))
   for term,con in (contrasts or {}).items():self.rows.append(dict(info,hac_lag=lag,term=term,**f.contrast(cov,con)))
  return f
 def export(self):
  ROOT=self.context.output;csv=self.context.csv;js=self.context.js
  CORE_DATA_HASH=self.context.identity("v2_panel");CODE_HASH=sha(Path(__file__))
  csv('tables/all_coefficients.csv',self.rows)
  flat=[]
  for a in self.audits:flat.append({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in a.items()})
  csv('diagnostics/model_catalog.csv',flat);csv('tables/covariance_index.csv',self.index)
  np.savez_compressed(ROOT/'tables/model_covariances.npz',**self.arrays)
  csv('tables/support_restrictions.csv',self.support);csv('tables/joint_covariance_contrasts.csv',self.joint);csv('diagnostics/gap_equivalence.csv',self.gap_checks);csv('diagnostics/explicit_model_status.csv',self.status)
  js('reporting/empirical_models.json',dict(status='RUN',model_count=len(self.audits),estimated=sum(a['status']=='ESTIMATED' for a in self.audits),blocked=sum(a['status']!='ESTIMATED' for a in self.audits),coefficient_rows=len(self.rows),architecture=['AT_actual','CH_actual','SI_actual','GR_actual','ME_actual','ME_synthetic','G_ME'],outcomes=[Y,'commercial_net_MW'],sources=[dict(path='frozen_v2_panel',sha256=CORE_DATA_HASH)],code_version_hash=CODE_HASH,records=self.rows,audits=self.audits,joint=self.joint,gap_checks=self.gap_checks,support=self.support,explicit_status=self.status))

def native_series(res,d,series,version='actual',comprehensive=True):
 mainfit=None
 for outcome in [Y,'commercial_net_MW']:
  for bw in CFG['bandwidth_days']:
   f=res.record(local(d,bw),outcome,['post','r','post_r'],series=series,model='R1',version=version,bw=bw,weighted=True)
   if bw==90 and outcome==Y:mainfit=f
  for donut in [v for v in CFG['donut_days'] if v!=0]:res.record(local(d,90,donut=donut),outcome,['post','r','post_r'],series=series,model='R1',version=version,bw=90,donut=donut,weighted=True,spec='donut')
  for fe,spec,drop in [(('hour','weekday','month'),'main',True),(('hour','weekday','week'),'week_FE',True),(('hour','weekday','month'),'include_leap_day',False)]:
   q=seasonal(d,drop)
   res.record(q,outcome,['D25','D26'],series=series,model='F1',version=version,fe=fe,spec=spec,contrasts=F1_CONTRASTS)
   res.record(q,outcome,F2_TERMS,series=series,model='F2',version=version,fe=fe,spec=spec,contrasts=F2_CONTRASTS)
  res.record(local(d,90),outcome,['post','r','post_r','spread_c','post_s'],series=series,model='local_spread_appendix',version=version,bw=90,weighted=True,spec='frozen_appendix',contrasts={'post_price_slope':{'spread_c':1,'post_s':1}})
 if version=='actual' or 'train2024' in version:
  for bw in CFG['bandwidth_days']:res.record(local(d,bw,2025),Y,['post','r','post_r'],series=series,model='R1_placebo',version=version,bw=bw,cutoff=2025,weighted=True)
 res.status.append(dict(series=series,model='R1_placebo',cutoff=2024,status='INSUFFICIENT_PRE_SUPPORT',reason='No 2023 left-side observations; not fitted'))
 q=seasonal(d);sets,bounds=supports(q);res.support.extend(dict(series=series,weight_version=version,**b) for b in bounds)
 for s,z in sets.items():
  if s!='full':res.record(z,Y,F2_TERMS,series=series,model='F2',version=version,fe=('hour','weekday','month'),support=s,contrasts=F2_CONTRASTS)
 # Preserve the frozen low-dimensional native-spread piecewise appendix.
 pre=d.loc[d.year<2026,'raw_price_spread'].dropna();knots=sorted(set([0.,*pre.quantile([.25,.5,.75]).tolist()]));terms=F2_TERMS.copy();z=q.copy()
 for i,k in enumerate(knots):
  z[f'hinge{i}']=(z.raw_price_spread-k).clip(lower=0)
  for year in [25,26]:z[f'D{year}_hinge{i}']=z[f'D{year}']*z[f'hinge{i}']
  terms.extend([f'hinge{i}',f'D25_hinge{i}',f'D26_hinge{i}'])
 piecewise={}
 for segment in range(len(knots)+1):
  con={'D26_s':1,'D25_s':-1}
  for i in range(segment):con.update({f'D26_hinge{i}':1,f'D25_hinge{i}':-1})
  piecewise[f'segment_{segment}_slope_26_minus_25']=con
 res.record(z,Y,terms,series=series,model='F2_piecewise',version=version,fe=('hour','weekday','month'),spec='fixed_pre_knots',contrasts=piecewise)
 res.support.extend(dict(series=series,weight_version=version,support='PRE_FIXED_KNOT',lower=k,year=0) for k in knots)
 return mainfit

def gaps(res,actual,syn):
 me=actual['ME_IT'].set_index('timestamp_utc');gap_main=None
 for version,sv in syn.groupby('weight_version'):
  s=calendar(sv).set_index('timestamp_utc');common=me.dropna(subset=[Y,'commercial_net_MW']).index.intersection(s.dropna(subset=[Y,'commercial_net_MW']).index)
  q=me.loc[common].copy()
  for y in [Y,'commercial_net_MW']:q[y]=me.loc[common,y]-s.loc[common,y]
  q=q.reset_index()
  for y in [Y,'commercial_net_MW']:
   for bw in CFG['bandwidth_days']:
    f=res.record(local(q,bw),y,['post','r','post_r'],series='G_ME',model='R2',version=version,bw=bw,weighted=True)
    if bw==90 and y==Y and version=='GY050_final':gap_main=f
   for donut in [v for v in CFG['donut_days'] if v!=0]:res.record(local(q,90,donut=donut),y,['post','r','post_r'],series='G_ME',model='R2',version=version,bw=90,donut=donut,weighted=True,spec='donut')
  if 'train2024' in version:
   for bw in CFG['bandwidth_days']:res.record(local(q,bw,2025),Y,['post','r','post_r'],series='G_ME',model='R2_placebo',version=version,bw=bw,cutoff=2025,weighted=True)
  if version in ['GY050_final','GY050_train2024']:
   for fe,spec,drop in [(('hour','weekday','month'),'main',True),(('hour','weekday','week'),'week_FE',True),(('hour','weekday','month'),'include_leap_day',False)]:
    for y in [Y,'commercial_net_MW']:res.record(seasonal(q,drop),y,['D25','D26'],series='G_ME',model='F1',version=version,fe=fe,spec=spec,contrasts=F1_CONTRASTS)
  if version=='GY050_final':
   a=fit(local(me.loc[common].reset_index(),90),Y,['post','r','post_r'],weight='triangular');b=fit(local(s.loc[common].reset_index(),90),Y,['post','r','post_r'],weight='triangular')
   for lag in CFG['hac_elapsed_hours']:
    joint=joint_covariance([a,b],lag);v=np.zeros(len(a.names)+len(b.names));v[a.names.index('post')]=1;v[len(a.names)+b.names.index('post')]=-1
    r=contrast(np.r_[a.beta,b.beta],joint,v);direct=gap_main.contrast(gap_main.covariance(lag),{'post':1})
    np.testing.assert_allclose([r['coefficient'],r['se']],[direct['coefficient'],direct['se']],rtol=1e-8,atol=1e-7)
    res.gap_checks.append(dict(version=version,hac_lag=lag,status='PASS',same_design=True,sample_hash=gap_main.sample_hash,joint_variance=r['se']**2,direct_variance=direct['se']**2,actual_synthetic_cross_covariance=float(joint[a.names.index('post'),len(a.names)+b.names.index('post')])))
    key='joint_gap_'+str(lag);res.arrays[key]=joint;res.index.append(dict(key=key,model_id='G_ME_joint_R1_GY050_final',hac_lag=lag,columns=json.dumps(['ME_actual:'+n for n in a.names]+['ME_synthetic:'+n for n in b.names])))
 res.status.append(dict(series='G_ME',model='F2',status='NOT_RUN_UNSPECIFIED_GAP_SPREAD',reason='A gap has no frozen unique native price index; no new spread regressor chosen. Native-donor Theta retained separately.'))
 return gap_main

def run(context):
 res=Results(context);data=calendar(context.panel());actual={k:v.copy() for k,v in data.groupby('corridor_id')};r1=[]
 for cid in ACTUALS:
  r1.append(native_series(res,actual[cid],cid[:2]+'_actual'));print('真实走廊已完成：'+cid,flush=True)
 # Retain joint covariance; do not infer a difference from separate significance.
 for lag in CFG['hac_elapsed_hours']:
  joint=joint_covariance(r1,lag);off=np.cumsum([0,*[len(f.names) for f in r1]])
  for i,ci in enumerate(ACTUALS):
   for j,cj in enumerate(ACTUALS):res.joint.append(dict(kind='actual_R1_tau_covariance',series_a=ci,series_b=cj,hac_lag=lag,covariance=float(joint[off[i]+r1[i].names.index('post'),off[j]+r1[j].names.index('post')])))
  key='joint_actual_R1_'+str(lag);res.arrays[key]=joint;res.index.append(dict(key=key,model_id='FIVE_ACTUAL_R1_JOINT',hac_lag=lag,columns=json.dumps([cid+':'+n for cid,f in zip(ACTUALS,r1) for n in f.names])))
 syn=calendar(context.synthetic.copy())
 for version in sorted(syn.weight_version.unique()):
  native_series(res,syn[syn.weight_version==version],'ME_synthetic',version);print('冻结ME合成版本完成：'+version,flush=True)
 gaps(res,actual,syn)
 # Frozen native donor slope contrast; joint time-series covariance, no synthetic-index substitution.
 ws=context.weights_default.copy();w=ws[(ws.target=='ME_IT')&(ws.version=='GY050_final')].set_index('donor').weight;members=['ME_IT',*w.index]
 frames=[seasonal(actual[c]).dropna(subset=[Y,'spread_c']) for c in members];common=set(frames[0].timestamp_utc)
 for frame in frames[1:]:common&=set(frame.timestamp_utc)
 fs=[fit(f[f.timestamp_utc.isin(common)],Y,F2_TERMS,('hour','weekday','month')) for f in frames];beta=np.concatenate([f.beta for f in fs]);offset=0;v=np.zeros(len(beta))
 for scale,f in zip([1,*(-w.to_numpy())],fs):
  v[offset+f.names.index('D26_s')]=scale;v[offset+f.names.index('D25_s')]=-scale;offset+=len(f.names)
 for lag in CFG['hac_elapsed_hours']:
  cov=joint_covariance(fs,lag);res.joint.append(dict(kind='native_donor_slope_Theta',series='ME_IT',weight_version='GY050_final',hac_lag=lag,n=len(common),sample_hash=fs[0].sample_hash,**contrast(beta,cov,v)))
  key='native_Theta_'+str(lag);res.arrays[key]=cov;res.index.append(dict(key=key,model_id='ME_NATIVE_F2_THETA',hac_lag=lag,columns=json.dumps([cid+':'+n for cid,f in zip(members,fs) for n in f.names])))
 res.export();print(f'正式无天气模块完成：{len(res.audits)}个设计；所有协方差和FE保留。',flush=True)

