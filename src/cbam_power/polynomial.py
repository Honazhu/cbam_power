"""Faithful V3.2 estimate.py/common.py port, using the same WLS/HAC engine.

Only orchestration, source identity and file boundaries changed. Shape diagnostics
are calculated during estimation. The parent result is used solely for post-fit
linear validation; no benchmark coefficient enters a fitting design or outcome.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from . import estimators as engine
from ._engine_context import sha,write_json,clean,canonical_sample_hash
ENGINE=Path(engine.__file__)
CODES=['AT','CH','SI','GR','ME']
ZONES=dict(AT='IT-NORD',CH='IT-NORD',SI='IT-NORD',GR='IT-SUD',ME='IT-CSUD')
REFERENCE='https://www.tilburgsciencehub.com/topics/analyze/causal-inference/rdd/regression-discontinuity-in-time-rdit/'
THRESHOLDS=dict(condition_number=1e8,negative_prediction_MW=-1.0,
 endpoint_range_padding_scales=1.0,turning_neighborhood_days=3,
 turning_min_effective_hours=24,ci_width_ratio_to_p1=3.0,
 numerical_atol=1e-6,numerical_rtol=1e-9,magnitude_relative_range=.5,magnitude_floor_MW=25.0)

def model_id(code, p, family='FE', h=90, year=2026):
    prefix = 'V3_2_' + ('PLACEBO2025_' if year == 2025 else '')
    return f'{prefix}{family}_P{p}_{code}' + (f'_BW{h}' if h != 90 else '')

def configurations(p):
    for code in CODES:
        yield code, p, 'BASIC', 90, 2026
        for h in [30, 60, 90, 180]:
            yield code, p, 'FE', h, 2026
        yield code, p, 'FE', 90, 2025

def scalar_table(models):
    keys = ['model_id', 'corridor', 'destination_zone', 'cutoff', 'cutoff_year', 'bandwidth_days',
            'polynomial_order', 'family', 'tau', 'se', 'ci_low', 'ci_high', 'p_value', 'N', 'N_pre',
            'N_post', 'kernel', 'HAC', 'hour_FE', 'weekday_FE', 'weather_controls', 'rank',
            'number_of_regressors', 'condition_number', 'weighted_R2', 'sample_hash', 'weight_hash',
            'code_hash', 'status', 'covariance_status', 'functional_form_status']
    return pd.DataFrame([{k: m[k] for k in keys} for m in models])

def stability(ms):
    values = np.array([m['tau'] for m in ms])
    same_sign = bool(np.all(values > 0) or np.all(values < 0))
    scale = max(abs(values[0]), THRESHOLDS['magnitude_floor_MW'])
    relative_range = float(np.ptp(values) / scale)
    category = ('FUNCTIONAL_FORM_SENSITIVE_SIGN' if not same_sign else
                'SIGN_STABLE_MAGNITUDE_SENSITIVE' if relative_range > THRESHOLDS['magnitude_relative_range'] else
                'SIGN_STABLE_WITHIN_COARSE_RANGE_THRESHOLD')
    return dict(sign_stable=same_sign, relative_range=relative_range, classification=category,
                cubic_se_ratio=ms[2]['se'] / ms[0]['se'])

def sample(d, code, h, year):
    q = d[d.corridor_id.eq(code+'_IT')].copy()
    cut = pd.Timestamp(f'{year}-01-01', tz='Europe/Rome').tz_convert('UTC')
    mid = q.timestamp_utc + pd.to_timedelta(q.delivery_interval_minutes / 2, unit='m')
    q['r'] = (mid-cut).dt.total_seconds()/86400
    q['post'] = q.timestamp_utc.ge(cut).astype(float)
    q = q[q.r.abs().le(h)].sort_values('timestamp_utc').copy()
    q['z'] = q.r/h
    q['triangular'] = 1-q.r.abs()/h
    # Endpoints have zero weight. There are no exact endpoints on this midpoint grid.
    assert not q.triangular.eq(0).any()
    assert q.z.abs().le(1).all() and not q.timestamp_utc.duplicated().any()
    assert q.scheduled_export_proxy.notna().all(), 'Missing frozen outcome'
    assert np.allclose(q.timestamp_utc.diff().dropna().dt.total_seconds()/3600, 1)
    assert np.isclose(((cut-cut).total_seconds()/86400)/h, 0)
    return q


def terms(q, p, scaled=True):
    q = q.copy()
    prefix = 'z' if scaled else 'r'
    names = ['post']
    for k in range(1, p+1):
        q[f'{prefix}{k}'] = q[prefix]**k
        q[f'post_{prefix}{k}'] = q.post*q[f'{prefix}{k}']
        names += [f'{prefix}{k}', f'post_{prefix}{k}']
    return q, names


def curve(m, side, grid):
    """Prediction at omitted FE categories, from saved beta and full covariance."""
    x = np.zeros((len(grid), len(m['columns'])))
    x[:, m['columns'].index('const')] = 1
    x[:, m['columns'].index('post')] = side
    for k in range(1, m['polynomial_order']+1):
        x[:, m['columns'].index(f'z{k}')] = grid**k
        x[:, m['columns'].index(f'post_z{k}')] = side*grid**k
    pred = x @ np.array(m['beta'])
    var = np.einsum('ij,jk,ik->i', x, np.array(m['covariance']), x)
    assert var.min() >= -1e-6
    se = np.sqrt(np.maximum(var, 0))
    return pred, se


def shape_qa(m, f, p1):
    core = ['const', 'post'] + [n for k in range(1, m['polynomial_order']+1) for n in (f'z{k}', f'post_z{k}')]
    ix = [f.names.index(n) for n in core]
    nuisance = f.x@f.beta - f.x[:, ix]@f.beta[ix]
    adjusted = f.y-nuisance
    scale = max(float(np.std(adjusted)), float(np.quantile(adjusted,.75)-np.quantile(adjusted,.25)), 1.)
    flags, ends, turns, lines = [], [], [], []
    all_pred = f.x@f.beta
    negative_N = int((all_pred < THRESHOLDS['negative_prediction_MW']).sum())
    if negative_N:
        flags.append('NEGATIVE_FITTED_HOURLY_OUTCOME')
    for side in [0, 1]:
        grid = np.linspace(-1,0,241) if side == 0 else np.linspace(0,1,241)
        pred, se = curve(m, side, grid)
        yside = adjusted[f.sample.post.to_numpy() == side]
        endpoint = float(pred[0] if side == 0 else pred[-1])
        endpoint_se = float(se[0] if side == 0 else se[-1])
        extreme = endpoint < yside.min()-scale or endpoint > yside.max()+scale
        if extreme:
            flags.append('EXTREME_ENDPOINT_FITTED_VALUE')
        ends.append(dict(side=side, fitted=endpoint, se=endpoint_se, extreme=bool(extreme),
                         adjusted_observed_min=float(yside.min()), adjusted_observed_max=float(yside.max())))
        coefficients = [m['beta'][m['columns'].index('const')] + side*m['tau']]
        for k in range(1,m['polynomial_order']+1):
            coefficients.append(m['beta'][m['columns'].index(f'z{k}')] + side*m['beta'][m['columns'].index(f'post_z{k}')])
        roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(coefficients))
        inside = sorted(float(t.real) for t in roots if abs(t.imag)<1e-9 and grid[0] < t.real < grid[-1])
        if len(inside) == 2:
            flags.append('CUBIC_TWO_TURNING_POINTS_ON_ONE_SIDE')
        for root in inside:
            r = root*m['bandwidth_days']
            near = (f.sample.r-r).abs().le(THRESHOLDS['turning_neighborhood_days']) & f.sample.post.eq(side)
            w = f.weights[near.to_numpy()]
            ess = float(w.sum()**2/(w@w)) if len(w) else 0.
            weak = ess < THRESHOLDS['turning_min_effective_hours']
            edge = abs(root) > .9
            if weak:
                flags.append('TURNING_POINT_LOW_LOCAL_SUPPORT')
            if edge:
                flags.append('TURNING_POINT_IN_LOW_WEIGHT_ENDPOINT_REGION')
            turns.append(dict(side=side, r_days=r, z=root, local_N=len(w), local_effective_N=ess,
                              mean_weight=float(w.mean()) if len(w) else 0., weak_support=weak, endpoint_region=edge))
        for r,z,y,s in zip(grid*m['bandwidth_days'],grid,pred,se):
            lines.append(dict(model_id=m['model_id'], post=side, r_days=r, z=z, fitted=y, se=s,
                              ci_low=y-1.96*s, ci_high=y+1.96*s))
    ratio = m['se']/p1['se'] if p1 else 1.
    if ratio > THRESHOLDS['ci_width_ratio_to_p1']:
        flags.append('CI_EXPANSION_VS_LINEAR')
    if m['condition_number'] > THRESHOLDS['condition_number']:
        flags.append('CONDITION_NUMBER_EXPLOSION')
    q = pd.DataFrame(dict(r_days=f.sample.r.to_numpy(), raw_outcome=f.y, adjusted_outcome=adjusted))
    q['bin_day'] = np.floor(q.r_days).astype(int)
    bins = q.groupby('bin_day',as_index=False).agg(r_days=('r_days','mean'), raw_mean=('raw_outcome','mean'),
             adjusted_mean=('adjusted_outcome','mean'), N=('raw_outcome','size'))
    bins['model_id'] = m['model_id']
    return dict(flags=sorted(set(flags)), endpoint_diagnostics=ends, turning_points=turns,
                hourly_negative_fitted_N=negative_N, ci_width_ratio_to_p1=ratio,
                adjusted_outcome_scale=scale), lines, bins


def fit_model(data, code, p, family, h, year, existing):
    q = sample(data, code, h, year)
    q, names = terms(q,p)
    fe = () if family == 'BASIC' else ('hour','weekday')
    f = engine.fit(q,'scheduled_export_proxy',names,fe,weight='triangular')
    raw, raw_names = terms(q,p,False)
    rf = engine.fit(raw,'scheduled_export_proxy',raw_names,fe,weight='triangular')
    cov = f.covariance(168)
    est = f.contrast(cov,{'post':1})
    tau_raw = float(rf.beta[rf.names.index('post')])
    tau_diff = est['coefficient']-tau_raw
    pred_diff = float(np.max(np.abs(f.x@f.beta-rf.x@rf.beta)))
    assert np.isclose(est['coefficient'],tau_raw,atol=1e-6,rtol=1e-9), 'RAW_SCALED_TAU_IMPLEMENTATION_ERROR'
    assert np.allclose(f.x@f.beta,rf.x@rf.beta,atol=1e-5,rtol=1e-8), 'RAW_SCALED_PREDICTION_ERROR'
    # Independent QR solution with reversed columns detects coefficient instability.
    aw = (f.x*np.sqrt(f.weights)[:,None])[:,::-1]
    qr, rr = np.linalg.qr(aw, mode='reduced')
    beta_qr = np.linalg.solve(rr, qr.T@(f.y*np.sqrt(f.weights)))[::-1]
    assert np.allclose(f.beta,beta_qr,atol=1e-6,rtol=1e-9), 'QR_COEFFICIENT_INSTABILITY'
    eig = np.linalg.eigvalsh(cov)
    assert np.isfinite(cov).all() and eig.min() >= -1e-8*max(1,eig.max()), 'INVALID_COVARIANCE'
    sample_hash = hashlib.sha256(pd.DatetimeIndex(f.sample.timestamp_utc).as_unit('ns').asi8.tobytes()).hexdigest()
    weight_hash = hashlib.sha256(f.weights.astype('<f8').tobytes()).hexdigest()
    mid = model_id(code,p,family,h,year)
    m = dict(model_id=mid,corridor=code,destination_zone=ZONES[code],cutoff=f'{year}-01-01T00:00:00+01:00',
             cutoff_year=year,bandwidth=h,bandwidth_days=h,polynomial_order=p,family=family,FE=list(fe),
             hour_FE=bool(fe),weekday_FE=bool(fe),weather_controls=False,kernel='triangular',HAC=168,
             tau=est['coefficient'],se=est['se'],SE=est['se'],ci_low=est['ci_low'],ci_high=est['ci_high'],
             CI=[est['ci_low'],est['ci_high']],p_value=est['p_value'],N=len(f.y),N_pre=f.audit['n_pre'],N_post=f.audit['n_post'],
             rank=f.audit['rank'],number_of_regressors=f.audit['k'],condition_number=f.audit['condition_number'],
             weighted_R2=f.audit['r2'],sample_hash=sample_hash,weight_hash=weight_hash,
             code_hash=sha(Path(__file__)),estimator_hash=sha(ENGINE),columns=f.names,beta=f.beta.tolist(),
             covariance=cov.tolist(),covariance_status='PASS',covariance_min_eigenvalue=float(eig.min()),
             reference_categories=f.audit['references'],raw_condition_number=rf.audit['condition_number'],
             raw_scaled_tau_difference=tau_diff,raw_scaled_max_fitted_difference=pred_diff,
             qr_max_coefficient_difference=float(np.max(abs(f.beta-beta_qr))),status='ESTIMATED',
             functional_form_status='NO_THRESHOLD_FLAG')
    p1 = existing.get(model_id(code,1,family,h,year))
    if p1:
        assert m['sample_hash']==p1['sample_hash'] and m['weight_hash']==p1['weight_hash'], 'POLYNOMIAL_SAMPLE_OR_WEIGHT_MISMATCH'
    shape, lines, bins = shape_qa(m,f,p1)
    m['shape_diagnostics'] = shape
    if shape['flags']:
        m['status'] = 'ESTIMATED_FUNCTIONALLY_UNSTABLE'
        m['functional_form_status'] = 'HIGH_ORDER_FUNCTIONAL_FORM_SENSITIVE' if p>1 else 'LINEAR_SHAPE_WARNING'
    # Compare cutoff fitted limits to the formal machine tau.
    left,_ = curve(m,0,np.array([0.]))
    right,_ = curve(m,1,np.array([0.]))
    assert np.isclose(right[0]-left[0],m['tau'],atol=1e-10,rtol=1e-12)
    m['cutoff_prediction_tau_difference'] = float(right[0]-left[0]-m['tau'])
    if code=='ME' and family=='FE' and h==90 and year==2026:
        # An explicit lag loop independently checks the FFT HAC post variance.
        influence = f.influence()[:,f.names.index('post')]
        elapsed = ((pd.DatetimeIndex(f.sample.timestamp_utc)-f.sample.timestamp_utc.iloc[0]).total_seconds()/3600).astype(int)
        scores = np.zeros(elapsed[-1]+1);scores[elapsed]=influence
        direct = scores@scores
        for lag in range(1,169):
            direct += 2*(1-lag/169)*(scores[lag:]@scores[:-lag])
        assert np.isclose(direct,m['se']**2,atol=1e-8,rtol=1e-10)
        m['independent_HAC_variance_difference'] = float(direct-m['se']**2)
    return m,lines,bins


def baseline_check(m,frozen):
    if m['polynomial_order']!=1 or m['bandwidth_days']!=90:
        return None
    code=m['corridor']
    old_id=(f'V3_PLACEBO2025_{code}' if m['cutoff_year']==2025 else f'V3_{m["family"]}_{code}')
    old=frozen[old_id]
    sample_ok=(m['N']==old['N']==4320 and m['N_pre']==old['pre_N']==2160 and
               m['N_post']==old['post_N']==2160 and m['sample_hash']==old['sample_hash'])
    coefficient_ok=np.allclose([m['tau'],m['se']],[old['estimate'],old['SE']],atol=1e-8,rtol=1e-10)
    coefficient_errors=[]
    for c in old['coefficients']:
        name={'r':'z1','post_r':'post_z1'}.get(c['term'],c['term'])
        scale=90 if c['term'] in ['r','post_r'] else 1
        coefficient_errors.append(m['beta'][m['columns'].index(name)]/scale-c['coefficient'])
    assert max(abs(np.array(coefficient_errors)))<1e-7, 'FULL_LINEAR_COEFFICIENT_MISMATCH'
    result=dict(model_id=m['model_id'],frozen_model_id=old_id,sample_match=sample_ok,
                tau_difference=m['tau']-old['estimate'],se_difference=m['se']-old['SE'],
                max_full_coefficient_difference=max(abs(np.array(coefficient_errors))),status='PASS' if sample_ok and coefficient_ok else 'FAIL')
    if not sample_ok:
        pass  # Caller records the error in the isolated run receipt; no global output.
        raise RuntimeError('STOP_SAMPLE_MISMATCH')
    assert coefficient_ok, 'STOP_LINEAR_BASELINE_MISMATCH'
    return result



def prepare_panel(frame):
    """Original estimate.panel validation, with an explicit freshly built frame."""
    columns=['timestamp_utc','corridor_id','scheduled_export_proxy','commercial_net_MW',
             'delivery_interval_minutes','destination_bidding_zone']
    d=frame[columns].copy();d['timestamp_utc']=pd.to_datetime(d.timestamp_utc,utc=True)
    assert d.delivery_interval_minutes.eq(60).all()
    assert np.allclose(d.scheduled_export_proxy,d.commercial_net_MW.clip(lower=0),equal_nan=True)
    for code,zone in ZONES.items():
        assert d.loc[d.corridor_id.eq(code+'_IT'),'destination_bidding_zone'].eq(zone).all()
    t=d.timestamp_utc.dt.tz_convert('Europe/Rome')
    d['hour'],d['weekday']=t.dt.hour,t.dt.weekday
    return d


def transform_to_elapsed_days(model):
    """Exact linear change of basis z=r/h for beta and full covariance.

    This does not refit or alter the stored scaled specification. It supplies the
    transformation required to compare polynomial P1 to the original V3 basis.
    """
    factors=np.ones(len(model['columns']));names=[]
    for i,name in enumerate(model['columns']):
        if name.startswith('post_z'):
            k=int(name[6:]);factors[i]=model['bandwidth_days']**(-k);names.append('post_r' if k==1 else f'post_r{k}')
        elif name.startswith('z') and name[1:].isdigit():
            k=int(name[1:]);factors[i]=model['bandwidth_days']**(-k);names.append('r' if k==1 else f'r{k}')
        else:names.append(name)
    beta=np.asarray(model['beta'])*factors
    covariance=np.asarray(model['covariance'])*factors[:,None]*factors[None,:]
    return names,beta,covariance


def not_estimated_models():
    return [dict(model_id=f'V3_2_PLACEBO2024_FE_P{p}_{code}',corridor=code,
                 polynomial_order=p,cutoff='2024-01-01',status='INSUFFICIENT_PRE_SUPPORT',
                 detail='No 2023 pre-cutoff observations; not estimated')
            for code in CODES for p in [1,2,3]]


def estimate_polynomial(frame,main_results,output_dir,*,source_hashes,parent_verification,order_validator=None):
    """Run the genuine P1 validation → P2 → P3 schedule (90 fitted models).

    The caller must provide the independently produced main verification receipt.
    Every polynomial sample is checked against fresh V3 results before fitting.
    No frozen expected values are treated as model inputs. PR5 performs the full
    comparison to the immutable polynomial reference afterwards.
    In authorized runs, order_validator independently checks each completed order
    against immutable references before any higher order may be estimated.
    """
    if parent_verification.get('status')!='PASS':raise ValueError('MAIN_BASELINE_VERIFICATION_REQUIRED')
    output=Path(output_dir)
    if output.exists() and any(output.iterdir()):raise ValueError('POLYNOMIAL_OUTPUT_NOT_EMPTY')
    output.mkdir(parents=True,exist_ok=True)
    (output/'tables').mkdir()
    data=prepare_panel(frame)
    frozen={m['model_id']:m for m in main_results['models']}
    for code in CODES:
        q=sample(data,code,90,2026);expected=frozen[f'V3_BASIC_{code}']
        counts=[len(q),int(q.post.eq(0).sum()),int(q.post.eq(1).sum())]
        if counts!=[4320,2160,2160] or canonical_sample_hash(q.timestamp_utc)!=expected['sample_hash']:
            raise ValueError('STOP_SAMPLE_MISMATCH:'+code)
    obj=dict(patch='V3.2_POLYNOMIAL_RDIT',reference_url=REFERENCE,
             design=dict(orders=[1,2,3],bandwidth_grid=[30,60,90,180],main_bandwidth=90,
                         kernel='triangular',HAC=168,outcome='positive part of scheduled commercial net flow (MW)',
                         midpoint='delivery start UTC + delivery_interval_minutes / 2',
                         running_variable='elapsed midpoint days / bandwidth',timezone='Europe/Rome',
                         weather=False,synthetic=False,QA_thresholds=THRESHOLDS),
             input_sha256=source_hashes['panel'],baseline_results_sha256=source_hashes['main_results'],
             models=[],stages={},baseline_checks=[],not_estimated=not_estimated_models(),
             numerical_replication='AUTHORIZED FULL NUMERICAL REPLICATION: NOT YET VERIFIED')
    samples={};members=[]
    def persist_samples():
        # Persist genuine sample membership before an independent order gate.
        pd.DataFrame(members).to_csv(output/'model_sample_index.csv',index=False)
        pd.concat([pd.DataFrame({'sample_hash_utc_ns':key,'timestamp_utc':ts}) for key,ts in samples.items()],ignore_index=True).to_parquet(output/'sample_timestamps.parquet',index=False)
    for p in [1,2,3]:
        if p>1 and obj['stages'].get(str(p-1))!='PASS':raise RuntimeError('PRIOR_POLYNOMIAL_STAGE_REQUIRED')
        existing={m['model_id']:m for m in obj['models']}
        lines=[];bins=[]
        for config in configurations(p):
            m,l,b=fit_model(data,*config,existing)
            check=baseline_check(m,frozen)
            if check:obj['baseline_checks'].append(check)
            q=sample(data,config[0],config[3],config[4]);key=m['sample_hash']
            samples.setdefault(key,pd.DatetimeIndex(q.timestamp_utc).as_unit('ns'))
            members.append(dict(model_id=m['model_id'],sample_hash_utc_ns=key))
            obj['models'].append(m);lines.extend(l);bins.append(b)
        if p==1 and (len(obj['baseline_checks'])!=15 or any(c['status']!='PASS' for c in obj['baseline_checks'])):
            raise RuntimeError('P1_LINEAR_VALIDATION_FAILED')
        pd.DataFrame(lines).to_csv(output/f'tables/plot_curves_P{p}.csv',index=False)
        pd.concat(bins,ignore_index=True).to_csv(output/f'tables/daily_bins_P{p}.csv',index=False)
        persist_samples()
        # Save a resumable diagnostic snapshot, but do not mark this order PASS
        # or enter the next order before its independent frozen-reference gate.
        write_json(output/'results.json',obj)
        if order_validator is not None:
            check=order_validator(p,obj,output)
            if not isinstance(check,dict) or check.get('status')!='PASS':
                raise RuntimeError('INDEPENDENT_POLYNOMIAL_ORDER_VALIDATION_FAILED:'+str(p))
            obj.setdefault('independent_order_validation',{})[str(p)]=check
        obj['stages'][str(p)]='PASS'
        write_json(output/'results.json',obj)
    write_json(output/'numerical_validation.json',validate_fitted_models(obj))
    scalar_table(obj['models']).to_csv(output/'model_catalog.csv',index=False)
    pd.DataFrame(obj['not_estimated']).to_csv(output/'not_estimated.csv',index=False)
    pd.DataFrame(obj['baseline_checks']).to_csv(output/'linear_validation.csv',index=False)
    persist_samples()
    return clean(obj)


def known_polynomials():
    t=pd.date_range('2025-12-01T23:00:00Z',periods=60*24,freq='h')
    cutoff=pd.Timestamp('2025-12-31T23:00:00Z')
    r=(t+pd.Timedelta(minutes=30)-cutoff).total_seconds()/86400
    z=np.asarray(r/30);post=np.asarray(t>=cutoff,dtype=float)
    local=t.tz_convert('Europe/Rome')
    base=pd.DataFrame(dict(timestamp_utc=t,r=r,z=z,post=post,hour=local.hour,weekday=local.weekday,triangular=1-np.abs(z)))
    checks=[]
    for p in [1,2,3]:
        y=500-125*post+30*z-40*post*z
        if p>=2:y=y+80*z*z-130*post*z*z
        if p==3:y=y+50*z*z*z+90*post*z*z*z
        for fe in [(),('hour','weekday')]:
            q=base.copy();q['y']=y+(0.4*q.hour+2*q.weekday if fe else 0)
            q,names=terms(q,p)
            f=engine.fit(q,'y',names,fe,weight='triangular')
            tau=f.beta[f.names.index('post')]
            assert np.isclose(tau,-125,atol=1e-9)
            assert np.max(abs(f.residuals))<1e-9
            checks.append(dict(check=f'known_P{p}_{"FE" if fe else "BASIC"}',status='PASS',error=float(tau+125)))
    # Missing time blocks must not be treated as adjacent observations in HAC.
    full=pd.date_range('2025-01-01',periods=600,freq='h',tz='UTC')
    keep=np.ones(600,dtype=bool);keep[90:155]=False;keep[350:399]=False
    times=full[keep];x=np.arange(600,dtype=float)
    scores=np.column_stack([np.sin(x/13)+.4*np.cos(x/31),np.cos(x/23)])[keep]
    actual=engine.hac(times,scores,168)
    regular=np.zeros((600,2));regular[keep]=scores
    direct=regular.T@regular
    for lag in range(1,169):
        cross=regular[lag:].T@regular[:-lag]
        direct+=(1-lag/169)*(cross+cross.T)
    assert np.allclose(actual,direct,atol=1e-8,rtol=1e-10)
    compressed=engine.hac(pd.date_range(times[0],periods=len(times),freq='h'),scores,168)
    assert not np.allclose(actual,compressed,atol=1e-3)
    checks.append(dict(check='elapsed_HAC_with_missing_blocks',status='PASS',error=float(np.max(abs(actual-direct)))))
    return checks


def validate_fitted_models(obj):
    """Numerical portion of V3.2 validate.main, without presentation dependencies."""
    ms=obj['models'];models={m['model_id']:m for m in ms}
    assert len(ms)==len(models)==90
    assert obj['stages']=={'1':'PASS','2':'PASS','3':'PASS'}
    assert len(obj['baseline_checks'])==15 and all(c['status']=='PASS' for c in obj['baseline_checks'])
    for m in ms:
        p=m['polynomial_order'];h=m['bandwidth_days']
        assert m['rank']==m['number_of_regressors']==2+2*p+(29 if m['family']=='FE' else 0)
        assert m['N']==h*48 and m['N_pre']==m['N_post']==h*24
        assert m['HAC']==168 and m['kernel']=='triangular' and not m['weather_controls']
        assert m['code_hash']==sha(Path(__file__))
        assert m['estimator_hash']==sha(ENGINE)
        assert np.isfinite([m[k] for k in ['tau','se','ci_low','ci_high','p_value','condition_number','weighted_R2']]).all()
        assert m['se']>0 and 0<=m['p_value']<=1
        assert np.isclose(m['ci_low'],m['tau']-1.96*m['se'])
        assert np.isclose(m['ci_high'],m['tau']+1.96*m['se'])
        base=models[model_id(m['corridor'],1,m['family'],h,m['cutoff_year'])]
        assert m['sample_hash']==base['sample_hash'] and m['weight_hash']==base['weight_hash']
        beta=np.array(m['beta']);cov=np.array(m['covariance']);i=m['columns'].index('post')
        assert np.isclose(beta[i],m['tau']) and np.isclose(cov[i,i],m['se']**2)
        assert cov.shape==(len(beta),len(beta)) and np.allclose(cov,cov.T)
        for k in range(1,p+1):assert f'z{k}' in m['columns'] and f'post_z{k}' in m['columns']
    return {'status':'PASS','models':90,'check':'contract_sample_weights_rank_covariance'}
