"""PR4 frozen-weight arithmetic; artificial tests need no authorized payloads."""
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cbam_power import synthetic as s



def rows_fixture():
    rows = []
    for version in s.ALL_VERSIONS:
        mapping = {'GR_IT': .25, 'AT_IT': .75}
        if '_loo_GR_IT' in version:
            mapping = {'GR_IT': 0., 'AT_IT': 1.}
        elif '_loo_AT_IT' in version:
            mapping = {'GR_IT': 1., 'AT_IT': 0.}
        for donor, weight in mapping.items():
            rows.append(dict(target='ME_IT', version=version, donor=donor, weight=weight, reason=None,
                             trained_max_local_year=2024 if version.endswith('train2024') else 2025))
    return pd.DataFrame(rows)


def panel_fixture():
    index = pd.date_range('2026-01-01', periods=3, freq='h', tz='UTC')
    rows = []
    signed = {'ME_IT': [500., 700., 50.], 'GR_IT': [-100., 1000., np.nan], 'AT_IT': [300., 1000., 40.]}
    for corridor, values in signed.items():
        for t, value in zip(index, values):
            rows.append(dict(timestamp_utc=t, corridor_id=corridor,
                             commercial_net_MW=value, scheduled_export_proxy=max(value, 0),
                             physical_net_MW=value, raw_price_spread=20., spread_center=10., spread_c=10.))
    return pd.DataFrame(rows)


@pytest.mark.public
def test_full_version_inventory_and_training_identity():
    mapping, rows = s.validate_weight_rows(rows_fixture())
    assert len(mapping) == 16 and len(rows) == 32
    assert set(mapping) == set(s.ALL_VERSIONS)
    assert list(mapping)[:2] == s.VERSIONS


@pytest.mark.parametrize('change', ['missing', 'unknown', 'duplicate', 'donor', 'negative', 'sum', 'future', 'loo'])
@pytest.mark.public
def test_invalid_frozen_weights_rejected(change):
    rows = rows_fixture()
    if change == 'missing': rows = rows.iloc[:-1]
    elif change == 'unknown': rows.loc[0, 'version'] = 'REOPTIMIZED'
    elif change == 'duplicate': rows = pd.concat([rows, rows.iloc[[0]]])
    elif change == 'donor': rows.loc[0, 'donor'] = 'FR_IT'
    elif change == 'negative': rows.loc[0, 'weight'] = -.1
    elif change == 'sum': rows.loc[0, 'weight'] = .1
    elif change == 'future': rows.loc[0, 'trained_max_local_year'] = 2026
    elif change == 'loo':
        rows.loc[rows.version == 'GY025_final_loo_AT_IT', 'weight'] = .5
    with pytest.raises(ValueError):
        s.validate_weight_rows(rows)


@pytest.mark.public
def test_positive_donors_before_weighting_and_gap_sign_no_clipping():
    out = s.construct_paths(panel_fixture(), {'DEMO': {'GR_IT': .25, 'AT_IT': .75}})
    # Hour 0: .25*max(-100,0)+.75*300 = 225, not max(200,0).
    assert out[s.Y].iloc[0] == 225.
    assert out.commercial_net_MW.iloc[0] == 200.
    assert out.G_ME.iloc[0] == 275.
    assert out[s.Y].iloc[1] == 1000.  # No 600 MW clipping.
    assert out.G_ME.iloc[1] == -300.
    assert np.isnan(out[s.Y].iloc[2])


@pytest.mark.public
def test_zero_weight_missing_donor_stays_missing():
    out = s.construct_paths(panel_fixture(), {'DEMO_LOO': {'GR_IT': 0., 'AT_IT': 1.}})
    assert out[s.Y].iloc[0] == 300.
    assert np.isnan(out[s.Y].iloc[2])


@pytest.mark.public
def test_duplicate_input_key_rejected():
    panel = panel_fixture()
    with pytest.raises(ValueError, match='DUPLICATE'):
        s.construct_paths(pd.concat([panel, panel.iloc[[0]]]), {'DEMO': {'GR_IT': .25, 'AT_IT': .75}})


@pytest.mark.public
def test_missing_corridor_and_bad_positive_proxy_rejected():
    panel = panel_fixture()
    with pytest.raises(ValueError, match='MISSING_SYNTHETIC_CORRIDOR'):
        s.construct_paths(panel[panel.corridor_id != 'AT_IT'], {'DEMO': {'GR_IT': .25, 'AT_IT': .75}})
    panel.loc[panel.corridor_id == 'GR_IT', s.Y] = 1.
    with pytest.raises(AssertionError):
        s.construct_paths(panel, {'DEMO': {'GR_IT': .25, 'AT_IT': .75}})


@pytest.mark.public
def test_fixed_sum_no_hourly_renormalization():
    frame = pd.DataFrame({'GR_IT': [10., 10.], 'AT_IT': [30., np.nan]})
    actual = s.strict_weighted_sum(frame, {'GR_IT': .25, 'AT_IT': .75})
    assert actual.iloc[0] == 25.
    assert np.isnan(actual.iloc[1])
    with pytest.raises(ValueError, match='fixed donor'):
        s.strict_weighted_sum(frame, {'GR_IT': 1.})


@pytest.mark.public
def test_independent_fixed_objective_calculation():
    # Energy discrepancy 4-(.25*2+.75*6) = -1; two flow discrepancies 2, 4.
    features = [('e', 'energy', [4., 2., 6.]), ('m1', 'monthly_path', [2., 0., 0.]),
                ('m2', 'monthly_path', [4., 0., 0.])]
    loss, details = s.fixed_weight_objective(features, [.25, .75], .5)
    assert loss == pytest.approx(.5*1 + .25*4 + .25*16, abs=1e-14)
    assert sum(r['weighted_squared_discrepancy'] for r in details) == pytest.approx(5.5)
    pure_generation, _ = s.fixed_weight_objective(features, [.25, .75], 1.)
    assert pure_generation == 1.


@pytest.mark.public
def test_prediction_error_opposite_to_gap():
    actual, predicted = np.array([2., 6.]), np.array([5., 9.])
    result = s.metrics(actual, predicted, 2.)
    assert result['bias'] == 3. and result['rmse'] == 3. and result['nrmse'] == 1.5
    assert np.mean(actual-predicted) == -result['bias']


@pytest.mark.public
def test_microsecond_and_nanosecond_hash_representation_explicit():
    us = pd.date_range('2024-01-01', periods=3, freq='h', tz='UTC').as_unit('us')
    ns = us.as_unit('ns')
    assert s.sample_hash(us) == s.sample_hash(ns)
    assert s.frozen_sample_hash(us) == s.frozen_sample_hash(ns)
    assert s.sample_hash(us) != s.frozen_sample_hash(us)


def feature_fixture():
    times = pd.DatetimeIndex([pd.Timestamp(year=y, month=m, day=15, hour=h, tz='Europe/Rome').tz_convert('UTC')
                              for y in [2024, 2025] for m in range(1, 13) for h in range(24)])
    n = len(times)
    x = np.arange(n, dtype=float) % 53
    wide = pd.DataFrame({'ME_IT': x+2, 'GR_IT': x*2, 'AT_IT': x*3+5}, index=times)
    rows = [dict(country=c, year=y, feature=f, share=value, status='FULL_WINDOW_VERIFIED')
            for y in [2024, 2025] for c,value in [('ME',.2),('GR',.3),('AT',.6)]
            for f in ['hydro']]
    return pd.DataFrame(rows), wide


@pytest.mark.public
def test_feature_scaling_preserves_training_scale_and_blocks():
    mix, wide = feature_fixture()
    train, scales, _, _ = s.build_predictors(mix, wide, 'ME_IT', s.DONORS, [2024])
    final, _, diagnostics, _ = s.build_predictors(mix, wide, 'ME_IT', s.DONORS, [2024, 2025], scales)
    assert {b for _, b, _ in train} == {'energy', 'monthly_path', 'hour_shape', 'moments', 'quantiles', 'zero_top'}
    expected_scale = np.std([.2, .3, .6], ddof=0)
    energy = next(v for n, _, v in train if n == 'energy:2024:hydro')
    np.testing.assert_allclose(energy, np.array([.2,.3,.6])/expected_scale)
    assert next(r for r in diagnostics if r['feature']=='energy:2025:hydro')['scale'] == expected_scale


@pytest.mark.public
def test_shared_constant_then_new_variation_uses_unit_scale():
    mix, wide = feature_fixture()
    mix.loc[mix.year == 2024, 'share'] = 0.
    _, scales, _, _ = s.build_predictors(mix, wide, 'ME_IT', s.DONORS, [2024])
    _, _, diagnostics, _ = s.build_predictors(mix, wide, 'ME_IT', s.DONORS, [2024,2025], scales)
    constant = next(r for r in diagnostics if r['feature'] == 'energy:2024:hydro')
    varying = next(r for r in diagnostics if r['feature'] == 'energy:2025:hydro')
    assert constant['omitted'] and constant['scale'] == 0.
    assert not varying['omitted'] and varying['scale'] == 1.


@pytest.mark.public
def test_missing_energy_month_or_training_month_blocks_features():
    mix, wide = feature_fixture()
    mix.loc[0, 'status'] = 'BLOCKED_ANNUAL_MIX'
    with pytest.raises(ValueError, match='BLOCKED_ANNUAL_MIX'):
        s.build_predictors(mix, wide, 'ME_IT', s.DONORS, [2024])
    with pytest.raises(ValueError, match='BLOCKED_PRE_MONTH'):
        s.outcome_features(wide.iloc[:24], [2024], 10.)


@pytest.mark.public
def test_sparse_eurostat_keeps_missing_and_published_zero_distinct():
    payload = {'id':['time','siec','unit'], 'size':[2,1,1], 'dimension':{
        'time':{'category':{'index':{'2024-01':0,'2024-02':1}}},
        'siec':{'category':{'index':{'RA100':0}}}, 'unit':{'category':{'index':{'GWH':0}}}},
        'value':{'0':0.}}
    frame, _ = s.decode_eurostat(payload)
    assert len(frame) == 1 and frame.iloc[0].value == 0.
    assert '2024-02' not in set(frame.time)


@pytest.mark.public
def test_no_optimizer_in_default_module():
    # There is no callable optimization implementation to be accidentally reached.
    assert not hasattr(s, 'fit_weights') and not hasattr(s, 'simplex_least_squares')
    with patch.object(s, 'fit_weights', create=True, side_effect=AssertionError('optimizer called')):
        s.construct_paths(panel_fixture(), {'DEMO': {'GR_IT': .25, 'AT_IT': .75}})
        s.fixed_weight_objective([('e', 'energy', [2., 1., 3.])], [.5,.5], 1.)


@pytest.mark.authorized
@pytest.mark.skipif(not os.environ.get('CBAM_INPUT_ROOT'), reason='authorized frozen weights require CBAM_INPUT_ROOT')
def test_authorized_full_precision_weights_and_feature_freeze():
    from cbam_power.inputs import InputCatalog
    catalog = InputCatalog()
    mapping, rows = s.load_frozen_weights(catalog)
    assert len(mapping) == 16 and len(rows) == 32
    with catalog.open_input('v2_weights') as handle:
        raw = pd.read_csv(handle, float_precision='round_trip')
    for version, donors in mapping.items():
        for donor, weight in donors.items():
            assert weight == raw[(raw.target=='ME_IT') & (raw.version==version) & (raw.donor==donor)].weight.iloc[0]
    assert raw[raw.target=='CH_IT'].weight.isna().all()
    with catalog.open_input('v2_weight_feature_freeze') as handle:
        assert json.load(handle) == s.FROZEN_FEATURES


class GenerationCatalog:
    """Artificial monthly GWh source, never an expected research panel."""
    def __init__(self, *, missing=False, residual=5.):
        codes = ['RA100', 'N9000', 'C0000', 'G3000', 'RA300', 'RA400', 'TOTAL']
        times = [f'2024-{month:02}' for month in range(1, 13)]
        values = [10., 0., 0., 5., 0., 0., 15.+residual]
        self.payload = {'id':['time','siec','unit'], 'size':[12,7,1], 'dimension':{
            'time':{'category':{'index':{t:i for i,t in enumerate(times)}}},
            'siec':{'category':{'index':{c:i for i,c in enumerate(codes)}}},
            'unit':{'category':{'index':{'GWH':0}}}},
            'value':{str(month*7+tech):value for month in range(12) for tech,value in enumerate(values)}}
        if missing:
            del self.payload['value']['1']  # Explicit zero nuclear in January becomes missing.
        self.assets = {f'generation_{c}':{'sha256':'a'*64} for c in ['CH','ME','FR','AT','SI','EL']}

    @contextmanager
    def open_input(self, aid):
        assert aid in self.assets
        yield io.BytesIO(json.dumps(self.payload).encode())


@pytest.mark.public
def test_annual_generation_uses_12_month_sums_and_sparse_missing():
    mix, coverage, native = s.build_energy_mix(GenerationCatalog())
    z = mix[(mix.country=='ME') & (mix.year==2024)].set_index('feature')
    assert z.loc['hydro','generation_gwh'] == 120.
    assert z.loc['hydro','share'] == .5
    assert z.loc['gas','share'] == .25
    assert z.loc['other','share'] == .25
    assert z.loc['nuclear','generation_gwh'] == 0.
    assert (mix[mix.year==2025].status == 'BLOCKED_ANNUAL_MIX').all()
    missing, _, _ = s.build_energy_mix(GenerationCatalog(missing=True))
    z = missing[(missing.country=='ME') & (missing.year==2024)].set_index('feature')
    assert np.isnan(z.loc['nuclear','generation_gwh'])
    assert z.share.isna().all()


@pytest.mark.public
def test_generation_preserves_original_rounding_only_exception():
    tiny, _, _ = s.build_energy_mix(GenerationCatalog(residual=-.001))
    z = tiny[(tiny.country=='ME') & (tiny.year==2024)].set_index('feature')
    assert z.loc['other','generation_gwh'] == 0.
    assert z.status.eq('FULL_WINDOW_VERIFIED').all()
    invalid, _, _ = s.build_energy_mix(GenerationCatalog(residual=-.1))
    assert invalid.status.eq('BLOCKED_ANNUAL_MIX').all()


class SavedWeightCatalog:
    def __init__(self, rows):
        # Original CSV contains blocked CH string reasons, while ME reasons are all null.
        blocked=rows.iloc[[0]].assign(target='CH_IT',reason='BLOCKED_ANNUAL_MIX',weight=np.nan)
        self.data=pd.concat([rows,blocked],ignore_index=True).to_csv(index=False).encode()
    @contextmanager
    def open_input(self, aid, *, purpose):
        assert aid=='v2_weights' and purpose=='verify'
        yield io.BytesIO(self.data)


def saved_artificial_synthetic(tmp_path):
    directory=tmp_path.resolve()
    weights,rows=s.validate_weight_rows(rows_fixture())
    paths=s.construct_paths(panel_fixture(),weights)
    paths.to_parquet(directory/'all_versions.parquet',index=False)
    paths[paths.weight_version.isin(s.VERSIONS)].to_parquet(directory/'main.parquet',index=False)
    rows.to_csv(directory/'weights.csv',index=False)
    pd.DataFrame({'feature':['ARTIFICIAL']}).to_csv(directory/'energy_mix.csv',index=False)
    pd.DataFrame({'feature':['ARTIFICIAL']}).to_csv(directory/'objective_features.csv',index=False)
    payload={'weights_reestimated':False,'weights':weights,
             'objective_evaluation':[{'weight_version':v,'objective':0.} for v in s.ALL_VERSIONS]}
    (directory/'summary.json').write_text(json.dumps(payload))
    return directory,SavedWeightCatalog(rows)


@pytest.mark.public
def test_saved_synthetic_verification_never_reconstructs_or_optimizes(tmp_path):
    directory,catalog=saved_artificial_synthetic(tmp_path)
    with patch.object(s,'build_synthetic',side_effect=AssertionError('reconstruction invoked')) as builder, \
         patch.object(s,'construct_paths',side_effect=AssertionError('prediction invoked')) as predictor, \
         patch.object(s,'fit_weights',create=True,side_effect=AssertionError('optimizer invoked')) as optimizer, \
         patch.object(s,'validate_frozen',return_value=[{'status':'PASS','scope':'mock verifier dispatch only'}]) as verifier:
        checks=s.validate_saved(directory,catalog)
    builder.assert_not_called();predictor.assert_not_called();optimizer.assert_not_called()
    verifier.assert_called_once()
    reconstructed_argument=verifier.call_args.args[0]
    assert len(reconstructed_argument.weights)==16
    assert len(reconstructed_argument.all_versions)==48
    assert checks[0]['scope']=='mock verifier dispatch only'


@pytest.mark.public
@pytest.mark.parametrize('change',['weights','summary','main','objective_versions','missing'])
def test_saved_synthetic_cannot_replace_frozen_expected_or_skip_artifacts(tmp_path,change):
    directory,catalog=saved_artificial_synthetic(tmp_path)
    if change=='weights':
        rows=pd.read_csv(directory/'weights.csv')
        rows.loc[rows.version=='GY050_train2024','weight']=.5
        rows.to_csv(directory/'weights.csv',index=False)
    elif change=='summary':
        payload=json.loads((directory/'summary.json').read_text());payload['weights_reestimated']=True
        (directory/'summary.json').write_text(json.dumps(payload))
    elif change=='main':
        frame=pd.read_parquet(directory/'main.parquet');frame=frame.iloc[1:]
        frame.to_parquet(directory/'main.parquet',index=False)
    elif change=='objective_versions':
        payload=json.loads((directory/'summary.json').read_text());payload['objective_evaluation'].pop()
        (directory/'summary.json').write_text(json.dumps(payload))
    else:(directory/'objective_features.csv').unlink()
    with patch.object(s,'validate_frozen',side_effect=AssertionError('should fail before verifier')):
        with pytest.raises(ValueError):s.validate_saved(directory,catalog)
