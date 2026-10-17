"""Faithful port: v3_formal_cbam/src/reused/design.py, all named functions.
No fitting or data reads at import; frozen calendar definitions retained.
"""
import numpy as np
import pandas as pd
from ._engine_context import CFG,sample_hash
F2_TERMS=['D25','D26','spread_c','D25_s','D26_s']
F1_CONTRASTS={'level_2026_minus_2025':{'D26':1,'D25':-1}}
F2_CONTRASTS={'slope_2024':{'spread_c':1},'slope_2025':{'spread_c':1,'D25_s':1},'slope_2026':{'spread_c':1,'D26_s':1},'slope_2025_minus_2024':{'D25_s':1},'slope_2026_minus_2024':{'D26_s':1},'slope_2026_minus_2025':{'D26_s':1,'D25_s':-1},'slope_2026_minus_pre_average':{'D26_s':1,'D25_s':-.5}}
def calendar(d):
 d=d.copy();d['timestamp_utc']=pd.to_datetime(d.timestamp_utc,utc=True);t=d.timestamp_utc.dt.tz_convert('Europe/Rome')
 for key in ['year','month','hour','weekday','day']:d[key]=getattr(t.dt,key)
 d['week']=t.dt.isocalendar().week.astype(int);return d
def local(d,bw,cutoff=2026,donut=0):
 q=d.copy();cut=pd.Timestamp(f'{cutoff}-01-01',tz='Europe/Rome').tz_convert('UTC')
 q['r']=(q.timestamp_utc+pd.Timedelta(minutes=30)-cut).dt.total_seconds()/86400
 q['post']=q.timestamp_utc.ge(cut).astype(float);q['post_r']=q.post*q.r
 if 'spread_c' in q:q['post_s']=q.post*q.spread_c
 q=q[q.r.abs().lt(bw)&q.r.abs().ge(donut)].copy();q['triangular']=1-q.r.abs()/bw;return q
def seasonal(d,drop_leap=True):
 t=d.timestamp_utc.dt.tz_convert('Europe/Rome');end=pd.Timestamp(CFG['window_utc_end_exclusive']).tz_convert('Europe/Rome')
 limits=pd.Series([end.replace(year=int(y)) for y in t.dt.year],index=d.index)
 q=d[t<limits].copy()
 if drop_leap:q=q[~((q.month==2)&(q.day==29))].copy()
 q['D25']=q.year.eq(2025).astype(float);q['D26']=q.year.eq(2026).astype(float)
 q['D25_s']=q.D25*q.spread_c;q['D26_s']=q.D26*q.spread_c;q['post']=q.D26;return q
def supports(q):
 result={'full':q.copy()};bounds=[]
 if not {2024,2025,2026}.issubset(set(q.loc[q.raw_price_spread.notna(),'year'])):return result,[dict(support=s,status='BLOCKED_MISSING_YEAR') for s in ['minmax','p5_p95','p10_p90']]
 for year,v in q.groupby('year'):bounds.append(dict(support='full',year=int(year),lower=None,upper=None,n_original=len(v),n_kept=int(v.raw_price_spread.notna().sum()),n_deleted=int(v.raw_price_spread.isna().sum()),deletion_rate=float(v.raw_price_spread.isna().mean()),status='COMPLETE_CASE_REQUIRED'))
 for name,lo,hi in [('minmax',0,1),('p5_p95',.05,.95),('p10_p90',.1,.9)]:
  grouped=q.groupby('year').raw_price_spread;lower=grouped.quantile(lo).max();upper=grouped.quantile(hi).min();z=q[q.raw_price_spread.between(lower,upper)].copy();result[name]=z
  for year,v in q.groupby('year'):
   kept=z[z.year==year];span=upper-lower
   bounds.append(dict(support=name,year=int(year),lower=float(lower),upper=float(upper),n_original=len(v),n_kept=len(kept),n_deleted=len(v)-len(kept),deletion_rate=1-len(kept)/len(v),hours_per_eur_mwh=len(kept)/span if span>0 else np.nan,sample_hash=sample_hash(kept.timestamp_utc),status='RUN'))
 return result,bounds
