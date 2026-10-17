"""Complete V3 execution grid, ported from empirical/weather/compute modules.

Historical source names are provenance only. Runtime uses explicit DataFrames;
no expected numerical results are read or used to fit any model. The internal
ports preserve full coefficients, joint covariance blocks and sidecar generation.
"""
from pathlib import Path
import pandas as pd
from ._engine_context import EngineContext, Y, ACTUALS, CFG
from ._assembly import canonical_id

SOURCE_MODULES = {
    'no_weather':'v3_formal_cbam/src/reused/empirical.py::main/native_series/gaps',
    'actual_weather':'v3_formal_cbam/src/reused/weather_models.py::run',
    'synthetic_weather':'v3_formal_cbam/src/reused/weather_synthetic.py::run',
    'assembly':'v3_formal_cbam/src/compute.py::assemble',
    'basic':'v3_formal_cbam/src/rdit.py::basic_models/plot_data',
}

def estimate_main(panel, weather, synthetic, weights, synthetic_summary, output_dir,
                  *, source_hashes, annual_times, weights_default):
    """Fit all inherited formal branches and save fresh results under output_dir.

    annual_times must be the InputCatalog-verified frozen sample_ME_F timestamps.
    The pipeline is responsible for validating I/B/A boundaries before this call.
    Returns a result whose numerical replication status is NOT_YET_VERIFIED.
    """
    from . import _empirical, _weather_models, _weather_synthetic, _assembly
    context=EngineContext(Path(output_dir),panel,weather,synthetic,weights,
                          synthetic_summary,source_hashes,pd.DatetimeIndex(annual_times),weights_default)
    _empirical.run(context)
    _weather_models.run(context)
    _weather_synthetic.run(context)
    result=_assembly.assemble(context)
    versions=weights.loc[weights.target.eq('ME_IT'),'version'].unique()
    expected=set(scheduled_model_ids(versions))
    actual={m['model_id'] for m in result['models']}
    if actual!=expected or len(result['models'])!=len(expected):
        raise RuntimeError('FORMAL_SCHEDULE_MISMATCH:'+str({'missing':sorted(expected-actual),'extra':sorted(actual-expected)}))
    return result


def scheduled_model_ids(weight_versions):
    """Enumerate the executed frozen grid without fitting or reading expected data.

    Mirrors empirical.native_series/gaps and the frozen local-weather branches.
    Annual weather remains recorded as blocked by the execution coverage gates.
    This is an inventory check; it is never a source of fitted values or samples.
    """
    versions=sorted(weight_versions)
    if len(versions)!=16 or len(set(versions))!=16:raise ValueError('EXPECTED_16_WEIGHT_VERSIONS')
    ids=[]
    def add(series,model,version='actual',outcome=Y,bw=0,donut=0,cutoff=2026,
            fe=('hour','weekday'),support='full',spec='main'):
        old='__'.join(map(str,['V2F',model,series,version,outcome,bw,donut,cutoff,'+'.join(fe),support,spec]))
        meta=dict(model_id=old,series=series,model=model,weight_version=version,
                  outcome=outcome,bandwidth_days=bw,donut_days=donut,cutoff=cutoff,
                  support=support,spec=spec)
        ids.extend(canonical_id(meta,lag) for lag in CFG['hac_elapsed_hours'])
    def native(series,version='actual'):
        for outcome in [Y,'commercial_net_MW']:
            for bw in CFG['bandwidth_days']:add(series,'R1',version,outcome,bw=bw)
            for donut in CFG['donut_days']:
                if donut:add(series,'R1',version,outcome,bw=90,donut=donut,spec='donut')
            for fe,spec in [(('hour','weekday','month'),'main'),(('hour','weekday','week'),'week_FE'),(('hour','weekday','month'),'include_leap_day')]:
                for model in ['F1','F2']:add(series,model,version,outcome,fe=fe,spec=spec)
            add(series,'local_spread_appendix',version,outcome,bw=90,spec='frozen_appendix')
        if version=='actual' or 'train2024' in version:
            for bw in CFG['bandwidth_days']:add(series,'R1_placebo',version,bw=bw,cutoff=2025)
        for support in ['minmax','p5_p95','p10_p90']:add(series,'F2',version,fe=('hour','weekday','month'),support=support)
        add(series,'F2_piecewise',version,fe=('hour','weekday','month'),spec='fixed_pre_knots')
    for corridor in ACTUALS:native(corridor[:2]+'_actual')
    for version in versions:native('ME_synthetic',version)
    for version in versions:
        for outcome in [Y,'commercial_net_MW']:
            for bw in CFG['bandwidth_days']:add('G_ME','R2',version,outcome,bw=bw)
            for donut in CFG['donut_days']:
                if donut:add('G_ME','R2',version,outcome,bw=90,donut=donut,spec='donut')
        if 'train2024' in version:
            for bw in CFG['bandwidth_days']:add('G_ME','R2_placebo',version,bw=bw,cutoff=2025)
        if version in ['GY050_final','GY050_train2024']:
            for fe,spec in [(('hour','weekday','month'),'main'),(('hour','weekday','week'),'week_FE'),(('hour','weekday','month'),'include_leap_day')]:
                for outcome in [Y,'commercial_net_MW']:add('G_ME','F1',version,outcome,fe=fe,spec=spec)
    for series in [c[:2]+'_actual' for c in ACTUALS]+['ME_synthetic','G_ME']:
        for variant in ['FULL_WEATHER_QUADRATIC','WEATHER_LINEAR_ONLY']:
            for arm in ['A','B','C']:
                if variant=='FULL_WEATHER_QUADRATIC':
                    stem='V3_GAP_WEATHER_ME' if series=='G_ME' else 'V3_SYNTHETIC_WEATHER_ME' if series=='ME_synthetic' else 'V3_WEATHER_'+series[:2]
                    mid=stem+('' if arm=='C' else '_'+arm)
                else:
                    family='R1_WEATHER' if series in ['ME_synthetic','G_ME'] else 'R1__full'
                    mid=f'V3__{series}__{family}__{variant}__{arm}'
                ids.extend([mid,mid+'_HAC24'])
    for corridor in ACTUALS:ids.extend([f'V3_BASIC_{corridor[:2]}',f'V3_BASIC_{corridor[:2]}_HAC24'])
    if len(ids)!=len(set(ids)):raise ValueError('DUPLICATE_SCHEDULED_MODEL_ID')
    return ids
