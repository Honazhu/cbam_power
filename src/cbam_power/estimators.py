"""Faithful port: v3_formal_cbam/src/reused/estimators.py.
Scientific fit, influence, covariance and contrast bodies are unchanged.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd
from scipy.linalg import lstsq, block_diag
from scipy.signal import fftconvolve
from scipy.stats import norm

class NotEstimable(ValueError):
    def __init__(self, status, audit):
        super().__init__(status); self.status=status; self.audit=audit

@dataclass
class Fit:
    sample: pd.DataFrame
    names: list
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    beta: np.ndarray
    residuals: np.ndarray
    bread: np.ndarray
    audit: dict
    @property
    def sample_hash(self):
        return hashlib.sha256(pd.DatetimeIndex(self.sample.timestamp_utc).asi8.tobytes()).hexdigest()
    def influence(self):
        return (self.x*(self.weights*self.residuals)[:,None])@self.bread * np.sqrt(len(self.y)/(len(self.y)-len(self.names)))
    def covariance(self,lag):
        return hac(self.sample.timestamp_utc,self.influence(),lag)
    def contrast(self,cov,terms):
        a=np.zeros(len(self.names))
        for k,v in terms.items(): a[self.names.index(k)]=v
        return contrast(self.beta,cov,a)

def contrast(beta,cov,a):
    b=float(a@beta);var=float(a@cov@a)
    if var < -1e-7: raise ArithmeticError('negative contrast variance')
    se=float(np.sqrt(max(0,var)))
    return dict(coefficient=b,se=se,ci_low=b-1.96*se,ci_high=b+1.96*se,p_value=float(2*norm.sf(abs(b/se))) if se else np.nan)

def hac(t,influence,lag):
    t=pd.DatetimeIndex(t)
    if t.has_duplicates or not t.is_monotonic_increasing: raise ValueError('UTC must be sorted unique')
    elapsed=(t-t[0]).total_seconds()/3600
    if not np.allclose(elapsed,np.round(elapsed)): raise ValueError('not hourly grid')
    slots=np.round(elapsed).astype(int)
    scores=np.zeros((slots[-1]+1,influence.shape[1]));scores[slots]=influence
    h=np.arange(-lag,lag+1);kernel=(1-np.abs(h)/(lag+1))[:,None]
    cov=scores.T@fftconvolve(scores,kernel,mode='same',axes=0)
    return (cov+cov.T)/2

def joint_covariance(fits,lag):
    times=pd.DatetimeIndex(sorted(set().union(*(set(f.sample.timestamp_utc) for f in fits))))
    cols=[]
    for f in fits:
        cols.append(pd.DataFrame(f.influence(),index=pd.DatetimeIndex(f.sample.timestamp_utc)).reindex(times,fill_value=0).to_numpy())
    return hac(times,np.column_stack(cols),lag)

def fit(frame,y,terms,fe=('hour','weekday'),weight=None,minimum_side=24):
    needed=list(dict.fromkeys(['timestamp_utc',y,*terms,*fe]+([weight] if weight else [])))
    d=frame.dropna(subset=needed).sort_values('timestamp_utc').copy()
    if weight: d=d[d[weight]>0]
    audit=dict(n=len(d),fe='+'.join(fe),omitted_terms='',absorbed_terms='',references={},columns=[])
    if len(d)<100: raise NotEstimable('INSUFFICIENT_SUPPORT',audit)
    if d.timestamp_utc.duplicated().any(): raise ValueError('duplicate UTC rows')
    x=d[list(terms)].astype(float).copy()
    for c in fe:
        levels=sorted(d[c].unique());audit['references'][c]=str(levels[0])
        for v in levels[1:]: x[f'{c}_{v}']=d[c].eq(v).astype(float)
    x.insert(0,'const',1.)
    a=x.to_numpy();v=d[y].to_numpy(float);w=d[weight].to_numpy(float) if weight else np.ones(len(d))
    aw=a*np.sqrt(w)[:,None];vw=v*np.sqrt(w)
    beta,_,_,s=lstsq(aw,vw,lapack_driver='gelsd')
    scales=np.linalg.norm(aw,axis=0);scaled=aw/np.where(scales>0,scales,1)
    rank=np.linalg.matrix_rank(scaled)
    audit.update(columns=list(x.columns),rank=int(rank),k=a.shape[1],condition_number=float(s[0]/s[-1]),n_pre=int(d.post.eq(0).sum()) if 'post' in d else None,n_post=int(d.post.eq(1).sum()) if 'post' in d else None)
    if rank<a.shape[1]: raise NotEstimable('BLOCKED_RANK',audit)
    if minimum_side and 'post' in d and min(audit['n_pre'],audit['n_post'])<minimum_side: raise NotEstimable('INSUFFICIENT_PRE_SUPPORT',audit)
    _,r=np.linalg.qr(aw,mode='reduced');ri=np.linalg.solve(r,np.eye(len(r)));bread=ri@ri.T
    residual=v-a@beta;den=np.sum(w*(v-np.average(v,weights=w))**2)
    audit['r2']=1-float(np.sum(w*residual**2)/den) if den else np.nan
    audit['r2_type']='weighted_R2' if weight else 'R2'
    return Fit(d,list(x.columns),a,v,w,beta,residual,bread,audit)
