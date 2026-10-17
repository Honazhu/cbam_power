"""Public/artificial mathematics and opt-in authorized frozen-weather checks.

Public: pytest tests/test_weather.py -m public
Authorized gate (no skips accepted): CBAM_WEATHER_AUTHORIZED=1 pytest tests/test_weather.py
Use PYTHONDONTWRITEBYTECODE=1; real outputs go only to CBAM_OUTPUT_ROOT/pr3.
"""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tomllib
import uuid
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import Geod
from shapely.geometry import box

from cbam_power.inputs import InputCatalog, InputError, REPO
from cbam_power.weather import (FIELDS, GRID_DIMS, KEYS, WeatherError, compare_panel,
    geodesic_area, grid_indices, merge_records, native_overlap_audit, open_native, run_weather_validation,
    select_expver, select_grid, validate_native, validate_weights, weighted_fields)


@pytest.fixture
def semantics():
    # Artificial metadata and physical constants, independently stated from
    # ERA5 definitions. No private payload or frozen result is needed here.
    result = {'variables': {}}
    for var, unit, param, scale, offset, shift, output in [
        ('t2m', 'K', 167, 1., -273.15, 0, 'temp_c'),
        ('u10', 'm s**-1', 165, 1., 0., 0, None),
        ('v10', 'm s**-1', 166, 1., 0., 0, None),
        ('ssrd', 'J m**-2', 169, 1/3600, 0., -1, 'ssrd_processed'),
        ('tp', 'm', 228, 1000., 0., -1, 'tp_processed'),
    ]:
        result['variables'][var] = dict(scale=scale, offset=offset,
            delivery_start_offset_hours=shift, output=output,
            native_metadata=dict(units=unit, GRIB_paramId=str(param), GRIB_stepUnits='1',
                GRIB_gridType='regular_ll', GRIB_stepType='accum' if shift else 'instant',
                GRIB_dataType='fc' if shift else 'an',
                GRIB_iDirectionIncrementInDegrees=.25, GRIB_jDirectionIncrementInDegrees=.25))
    return result


@pytest.fixture
def native(semantics):
    data = xr.Dataset(coords={
        'valid_time': pd.date_range('2026-01-01', periods=2, freq='h').to_numpy(),
        'latitude': [1.25, 1.0], 'longitude': [0., .25],
        'expver': ('valid_time', ['0001', '0005']),
    }, attrs={'history': 'Artificial reanalysis oper fixture'})
    for var in ['t2m', 'u10', 'v10']:
        data[var] = (GRID_DIMS, np.ones((2, 2, 2), dtype='float32'))
        data[var].attrs.update(semantics['variables'][var]['native_metadata'])
    return data


@pytest.fixture
def mask():
    return pd.DataFrame(dict(latitude=[1.25, 1.0], longitude=[0., .25],
                             intersection_area_m2=[1., 3.], area_weight=[.25, .75]))


@pytest.mark.public
def test_environment_versions_and_lock():
    assert sys.version_info[:2] == (3, 12)
    assert platform.python_version() == (REPO/'.python-version').read_text().strip()
    assert sys.prefix != sys.base_prefix
    project = tomllib.loads((REPO/'pyproject.toml').read_text())
    lock = tomllib.loads((REPO/'uv.lock').read_text())
    from packaging.specifiers import SpecifierSet
    wanted = SpecifierSet(project['project']['requires-python'])
    locked = SpecifierSet(lock['requires-python'])  # uv canonicalizes to ==3.12.*
    assert platform.python_version() in wanted and platform.python_version() in locked
    for version in ['3.11.9', '3.12.0', '3.12.14', '3.12.99', '3.13.0']:
        assert (version in wanted) == (version in locked)
    packages = {p['name']: p['version'] for p in lock['package']}
    for name in ['numpy', 'pandas', 'scipy', 'pyarrow', 'lxml', 'openpyxl', 'xarray',
                 'netcdf4', 'h5netcdf', 'h5py', 'shapely', 'pyproj', 'pytest']:
        assert importlib.metadata.version(name) == packages[name]
    assert 'cdsapi' not in packages


@pytest.mark.public
def test_two_backends_and_parquet_roundtrip(native, semantics, tmp_path):
    path = tmp_path/'artificial.nc'
    native.to_netcdf(path, engine='netcdf4')
    with open_native(path, backend='netcdf4') as first, open_native(path, backend='h5netcdf') as other:
        validate_native(first, semantics)
        validate_native(other, semantics)
        xr.testing.assert_equal(first.load(), other.load())
    frame = pd.DataFrame({'timestamp_utc': pd.date_range('2026-01-01', periods=2, tz='UTC'),
                          'value': [1., np.nan]})
    frame.to_parquet(tmp_path/'artificial.parquet', index=False)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(tmp_path/'artificial.parquet'))


@pytest.mark.public
def test_installed_import_from_other_working_directory(tmp_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-B', '-c',
        'from cbam_power.weather import FIELDS; assert len(FIELDS)==4'],
        cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.public
def test_independent_spatial_temperature_and_accumulation_math(semantics):
    grids = {'t2m': np.array([[273.15, 293.15]]),
             'ssrd': np.array([[7200., 3600.]]), 'tp': np.array([[.002, .006]])}
    out = weighted_fields(grids, [.25, .75], semantics)
    assert out['temp_c'][0][0] == pytest.approx(15., abs=1e-12)
    assert out['ssrd_processed'][0][0] == pytest.approx(1.25)
    assert out['tp_processed'][0][0] == pytest.approx(5.)
    validity = pd.Timestamp('2026-01-01T01:00Z')
    assert validity+pd.Timedelta(hours=out['temp_c'][1]) == pd.Timestamp('2026-01-01T01:00Z')
    for name in ['ssrd_processed', 'tp_processed']:
        assert validity+pd.Timedelta(hours=out[name][1]) == pd.Timestamp('2026-01-01T00:00Z')


@pytest.mark.public
def test_wind_magnitude_before_mean_counterexample(semantics):
    out = weighted_fields({'u10': np.array([[3., -3.]]), 'v10': np.array([[4., -4.]])}, [.5, .5], semantics)
    assert out['wind_ms'][0][0] == 5.
    assert np.hypot((3.-3.)/2, (4.-4.)/2) == 0.


@pytest.mark.public
def test_null_cell_propagates_without_renormalization(semantics):
    out = weighted_fields({'t2m': np.array([[273.15, np.nan]])}, [.25, .75], semantics)
    assert np.isnan(out['temp_c'][0][0])


@pytest.mark.public
def test_geodesic_area_independent_vertex_formula():
    expected, _ = Geod(ellps='WGS84').polygon_area_perimeter([0., .25, .25, 0.], [0., 0., .25, .25])
    assert geodesic_area(box(0, 0, .25, .25)) == pytest.approx(abs(expected), rel=1e-12)


@pytest.mark.public
def test_grid_cell_order_and_expver_priority(native, mask):
    native['t2m'].values[:] = np.arange(8).reshape(2, 2, 2)
    np.testing.assert_array_equal(select_grid(native.t2m, native, mask), [[0., 3.], [4., 7.]])
    values = xr.DataArray([[np.nan, 10.], [20., 99.]], dims=['expver', 'point'],
                          coords={'expver': ['0001', '0005']})
    np.testing.assert_array_equal(select_expver(values).values, [20., 10.])
    np.testing.assert_array_equal(select_expver(values.sel(expver=['0005', '0001'])).values, [20., 10.])


@pytest.mark.public
def test_native_fill_decoding(native, semantics, tmp_path):
    native.t2m.values[0, 0, 0] = np.nan
    path = tmp_path/'fill.nc'
    native.to_netcdf(path, engine='netcdf4', encoding={'t2m': {'_FillValue': -9999.}})
    with open_native(path) as decoded:
        validate_native(decoded, semantics)
        assert np.isnan(decoded.t2m.values[0, 0, 0])
        assert decoded.t2m.encoding['_FillValue'] == -9999.


@pytest.mark.public
@pytest.mark.parametrize('mutation,error', [
    ('variable', 'VARIABLE_IDENTITY'), ('units', 'VARIABLE_SEMANTICS'),
    ('step', 'VARIABLE_SEMANTICS'), ('duplicate_time', 'DUPLICATE_OR_INVALID_TIME'),
    ('subhour', 'HOURLY_TIME'), ('duplicate_lat', 'DUPLICATE_COORDINATE'),
    ('coordinate_order', 'COORDINATE_ORDER_OR_GRID'), ('coordinate_nan', 'COORDINATE'),
    ('expver', 'UNKNOWN_EXPVER'), ('stream', 'STREAM_NOT_OPER'),
])
def test_invalid_native_rejected(native, semantics, mutation, error):
    if mutation == 'variable':
        native = native.rename({'t2m': 'wrong_variable'})
    elif mutation in ['units', 'step']:
        native.t2m.attrs['units' if mutation == 'units' else 'GRIB_stepType'] = 'wrong'
    elif mutation == 'duplicate_time':
        native = native.assign_coords(valid_time=[native.valid_time.values[0]]*2)
    elif mutation == 'subhour':
        native = native.assign_coords(valid_time=native.valid_time.values+np.timedelta64(1, 'm'))
    elif mutation == 'duplicate_lat':
        native = native.assign_coords(latitude=[1., 1.])
    elif mutation == 'coordinate_order':
        native = native.assign_coords(latitude=[1., 1.25])
    elif mutation == 'coordinate_nan':
        native = native.assign_coords(latitude=[1., np.nan])
    elif mutation == 'expver':
        native = native.assign_coords(expver=('valid_time', ['0009', '0009']))
    elif mutation == 'stream':
        native.attrs['history'] = 'wrong stream'
    with pytest.raises(WeatherError, match=error):
        validate_native(native, semantics)


@pytest.mark.public
@pytest.mark.parametrize('mutation,error', [('missing', 'MISSING_MASK_GRID'),
    ('duplicate', 'DUPLICATE_GRID_CELL'), ('negative', 'AREA_WEIGHT'), ('sum', 'AREA_WEIGHT'),
    ('nan', 'MASK_NONFINITE'), ('resolution', 'MASK_GRID_RESOLUTION')])
def test_invalid_mask_rejected(native, mask, mutation, error):
    if mutation == 'missing':
        mask.loc[0, 'latitude'] = 9.
    elif mutation == 'duplicate':
        mask.loc[1, ['latitude', 'longitude']] = mask.loc[0, ['latitude', 'longitude']]
    elif mutation == 'negative':
        mask['area_weight'] = [-.5, 1.5]
    elif mutation == 'sum':
        mask['area_weight'] = [.1, .2]
    elif mutation == 'nan':
        mask.loc[0, 'area_weight'] = np.nan
    else:
        mask.loc[0, 'latitude'] = 1.1
    with pytest.raises(WeatherError, match=error):
        validate_weights(mask)
        grid_indices(native, mask)


@pytest.mark.public
def test_missing_unreadable_or_wrong_backend(tmp_path):
    path = tmp_path/'invalid.nc'
    with pytest.raises(WeatherError, match='NETCDF_DECODE'):
        with open_native(path):
            pass
    path.write_bytes(b'not a NetCDF')
    with pytest.raises(WeatherError, match='NETCDF_DECODE'):
        with open_native(path):
            pass
    with pytest.raises(WeatherError, match='NETCDF_BACKEND'):
        with open_native(path, backend='unknown'):
            pass


@pytest.mark.public
def test_modified_source_hash_rejected_before_decode(native, tmp_path):
    path = tmp_path/'probe.nc'
    native.to_netcdf(path, engine='netcdf4')
    # Tiny artificial catalog uses the real PR2 confinement/integrity reader.
    c = object.__new__(InputCatalog)
    c.input_root = tmp_path.resolve()
    c.assets = {'probe': dict(asset_id='probe', relative_path='probe.nc', format='netcdf',
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size,
        allowed_uses=['process'], materialization='private_copy', schema_ref='netcdf')}
    c.manifest = {'schemas': {'netcdf': {}}}
    c.resolve('probe')
    with path.open('r+b') as handle:
        handle.seek(-1, 2)
        old = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([old[0] ^ 1]))
    with pytest.raises(InputError, match='BLOCKED_INPUT_HASH'):
        c.resolve('probe')


@pytest.mark.public
def test_overlapping_records_select_first_and_reject_conflicts():
    base = dict(timestamp_utc=pd.Timestamp('2026-01-01T00:00Z'), geography='X', variable='temp_c')
    long = pd.DataFrame([dict(base, value=1., asset_id='first'), dict(base, value=1., asset_id='second')])
    _, selected, audit = merge_records(long)
    assert selected.asset_id.tolist() == ['first']
    assert audit['duplicate_keys'] == 1
    for value, reason in [(2., 'REVISION_CONFLICT'), (np.nan, 'REVISION_MISSING_CONFLICT')]:
        bad = long.copy()
        bad.loc[1, 'value'] = value
        with pytest.raises(WeatherError, match=reason):
            merge_records(bad)


@pytest.mark.public
def test_panel_wrong_alignment_missing_mask_and_strict_tolerance():
    frozen = pd.DataFrame(dict(timestamp_utc=pd.date_range('2026-01-01', periods=2, freq='h', tz='UTC'),
                              geography=['X', 'X'], **{f: [0., 0.] for f in FIELDS}))
    good = frozen.copy()
    good.timestamp_utc = good.timestamp_utc.astype('datetime64[us, UTC]')
    assert compare_panel(good, frozen, tolerance=1e-9)[0]['status'] == 'PASS'
    for mutation in ['time', 'null', 'boundary']:
        bad = good.copy()
        if mutation == 'time':
            bad.timestamp_utc += pd.Timedelta(hours=1)
        elif mutation == 'null':
            bad.loc[0, 'temp_c'] = np.nan
        else:
            bad.loc[0, 'temp_c'] = 1e-9
        report, differences = compare_panel(bad, frozen, tolerance=1e-9)
        assert report['status'] == 'FAIL' and differences
    with pytest.raises(WeatherError, match='DUPLICATE_PANEL_KEY'):
        compare_panel(pd.concat([good, good]), frozen, tolerance=1e-9)


@pytest.mark.public
def test_native_conflict_in_same_complete_mask_blocks(native, mask, tmp_path):
    path = tmp_path/'old.nc'
    native.to_netcdf(path, engine='netcdf4')
    prior = dict(asset_id='old', variables=list(native.data_vars),
                 **{axis: native[axis].values.copy() for axis in GRID_DIMS})
    class Catalog:
        def resolve(self, aid):
            assert aid == 'old'
            return path
    native.t2m.values[:] = 2.
    with pytest.raises(WeatherError, match='NATIVE_REVISION_CONFLICT'):
        native_overlap_audit(native, [prior], Catalog(), 'netcdf4', {'X': mask})


@pytest.mark.public
def test_different_regional_sources_never_form_grid_mosaic(native, mask, tmp_path):
    path = tmp_path/'old.nc'
    native.to_netcdf(path, engine='netcdf4')
    prior = dict(asset_id='old', variables=list(native.data_vars),
                 **{axis: native[axis].values.copy() for axis in GRID_DIMS})
    class Catalog:
        def resolve(self, aid):
            return path
    current = native.assign_coords(longitude=[.25, .5]).copy(deep=True)
    current.t2m.values[:] = 2.
    other_mask = mask.copy()
    other_mask.longitude += .25
    rows = native_overlap_audit(current, [prior], Catalog(), 'netcdf4', {'A': mask, 'B': other_mask})
    difference = next(r for r in rows if r['variable'] == 't2m')
    assert difference['status'] == 'HANDLED_DISJOINT_COMPLETE_MASKS'
    assert difference['max_abs_difference'] == 1.
    assert difference['common_complete_masks'] == []
    assert not difference['source_substitution_or_averaging']


@pytest.mark.public
def test_dst_weather_remains_utc_and_hour_shift():
    # Rome autumn DST has two local 02:00 instants, distinguished by UTC.
    t = pd.date_range('2025-10-26T00:00Z', periods=3, freq='h')
    assert t.tz_convert('Europe/Rome').hour.tolist() == [2, 2, 3]
    assert t.is_unique
    assert (t-pd.Timedelta(hours=1)).is_unique


@pytest.fixture(scope='session')
def authorized_result():
    if os.environ.get('CBAM_WEATHER_AUTHORIZED') != '1':
        pytest.skip('Authorized integration not requested; this is not G3 PASS')
    catalog = InputCatalog()  # Missing inputs are errors when authorized mode is requested.
    run = catalog.output_root/'pr3'/('pytest_weather_'+uuid.uuid4().hex[:12])
    import cbam_power.weather as weather
    original_reconstruct = weather.reconstruct
    original_open = catalog.open_input
    compute_reads = set()
    def computation_only(aid, *, purpose='process'):
        assert catalog.assets[aid]['classification'] == 'F', 'BENCHMARK_USED_AS_RECONSTRUCTION_INPUT'
        assert purpose == 'process', 'VERIFICATION_INPUT_DURING_RECONSTRUCTION'
        compute_reads.add(aid)
        return original_open(aid, purpose=purpose)
    def guarded(*args, **kwargs):
        with patch.object(catalog, 'open_input', side_effect=computation_only), \
             patch('socket.socket', side_effect=AssertionError('NETWORK_FORBIDDEN')), \
             patch('socket.create_connection', side_effect=AssertionError('NETWORK_FORBIDDEN')):
            return original_reconstruct(*args, **kwargs)
    with patch.object(weather, 'reconstruct', side_effect=guarded):
        result, comparisons, support, receipt = run_weather_validation(catalog, run)
    (run/'computation_input_ids.json').write_text(json.dumps(sorted(compute_reads), indent=2))
    return catalog, run, result, comparisons, support, receipt


@pytest.mark.authorized
@pytest.mark.frozen
def test_authorized_raw_50_lineage_58_mapping_22(authorized_result):
    c, run, result, _, _, receipt = authorized_result
    files = result.diagnostics['files']
    assert len(files) == 50
    assert len({f['sha256'] for f in files}) == 50
    assert all(set(f['expver']) <= {'1', '5'} for f in files)
    assert len(result.diagnostics['request_lineage']) == 58
    assert len({r['legacy_logical_path'] for r in result.diagnostics['request_lineage'] if r['legacy_path_broken']}) == 22
    assert receipt['status'] == 'PASS'
    assert c.output_root in run.parents
    assert all(f['sha256'] == c.assets[f['asset_id']]['sha256'] for f in files)
    audit = pd.read_csv(c.resolve('era5_frozen_file_audit', purpose='verify')).set_index('sha256')
    for file in files:
        expected = audit.loc[file['sha256']]
        assert file['valid_times'] == expected.valid_times
        assert pd.Timestamp(file['first_valid_time']) == pd.Timestamp(expected.first_valid_time)
        assert pd.Timestamp(file['last_valid_time']) == pd.Timestamp(expected.last_valid_time)
        assert set(file['variables']) == set(expected.variables.split(';'))


@pytest.mark.authorized
def test_authorized_all_50_native_values_equal_across_backends(authorized_result):
    catalog, run, result, _, _, _ = authorized_result
    records = []
    for file in result.diagnostics['files']:
        path = catalog.resolve(file['asset_id'])
        with open_native(path, backend='netcdf4') as left, open_native(path, backend='h5netcdf') as right:
            xr.testing.assert_equal(left.load(), right.load())
        records.append(dict(asset_id=file['asset_id'], status='EXACT_VALUES_COORDINATES_AND_NULLS'))
    (run/'backend_equivalence.json').write_text(json.dumps(records, indent=2))
    assert len(records) == 50


@pytest.mark.authorized
@pytest.mark.frozen
def test_authorized_eight_masks_and_spatial_coverage(authorized_result):
    _, _, result, _, _, _ = authorized_result
    masks = result.diagnostics['masks']
    assert len(masks) == 8
    assert all(m['status'] == 'PASS' for m in masks)
    assert {m['geography']: m['grid_cells'] for m in masks} == {
        'AT': 210, 'CH': 113, 'SI': 61, 'GR': 466, 'ME': 41,
        'IT-NORD': 286, 'IT-SUD': 95, 'IT-CSUD': 129}
    assert any(r['status'] == 'BLOCKED_SPATIAL_COVERAGE' for r in result.diagnostics['spatial'])
    assert not result.diagnostics['weights_replaced']


@pytest.mark.authorized
def test_authorized_compute_reads_exclude_benchmarks(authorized_result):
    catalog, run, _, _, _, _ = authorized_result
    ids = json.loads((run/'computation_input_ids.json').read_text())
    assert len([aid for aid in ids if catalog.assets[aid]['format'] == 'netcdf']) == 50
    assert all(catalog.assets[aid]['classification'] == 'F' for aid in ids)
    assert not any(aid in ids for aid in ['weather_v23_benchmark', 'weather_v2f_benchmark',
                                        'engine_weather_panel_hourly', 'v3_results'])


@pytest.mark.authorized
@pytest.mark.frozen
def test_authorized_frozen_keys_null_masks_values(authorized_result):
    _, _, result, comparisons, _, _ = authorized_result
    assert result.panel.shape == (41486, 6)
    assert result.panel.timestamp_utc.nunique() == 7708
    assert result.panel.groupby('geography').size().to_dict() == {
        'ME': 7708, 'IT-CSUD': 7708, 'AT': 4345, 'CH': 4345,
        'SI': 4345, 'GR': 4345, 'IT-NORD': 4345, 'IT-SUD': 4345}
    assert result.panel[FIELDS].isna().sum().to_dict() == {f: 16 for f in FIELDS}
    assert result.v23_panel.shape == (50250, 6)
    assert result.v23_panel[FIELDS].isna().sum().to_dict() == {f: 26352 for f in FIELDS}
    for report in comparisons.values():
        assert report['status'] == 'PASS' and report['keys_and_order_equal']
        assert report['missing_keys'] == report['extra_keys'] == 0
        assert all(r['null_mask_equal'] and r['max_abs_difference'] < 1e-9 for r in report['fields'].values())


@pytest.mark.authorized
@pytest.mark.frozen
def test_authorized_r1_and_blocked_annual_coverage(authorized_result):
    _, _, _, _, support, _ = authorized_result
    local = [r for r in support['bilateral'] if r['model'] == 'R1']
    assert len(local) == 5
    assert all(r['complete_hours'] == r['expected_hours'] == 4320 for r in local)
    annual = [r for r in support['bilateral'] if r['model'] == 'F']
    assert all(r['status'] == 'BLOCKED_WEATHER_PANEL' for r in annual)
    assert all(r['status'] == 'NOT_RUN' for r in support['synthetic_annual_status'])
    assert not support['model_estimation_executed']
    assert {r['year'] for r in support['by_year'] if r['model'] == 'F'} == {2024, 2025, 2026}


@pytest.mark.authorized
def test_authorized_no_output_overwrite_or_root_overlap(authorized_result):
    c, run, _, _, _, _ = authorized_result
    with pytest.raises(WeatherError, match='OUTPUT_ALREADY_EXISTS'):
        run_weather_validation(c, run)
    with pytest.raises(WeatherError, match='OUTPUT_ROOT'):
        run_weather_validation(c, c.input_root/'forbidden')
