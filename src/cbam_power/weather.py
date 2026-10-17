"""Offline ERA5 reconstruction using frozen inputs and scientific semantics.

Ported from V3 src/reused/weather_audit.py and V23 src/process_weather.py.
No legacy imports, estimators, acquisition, import-time I/O or baseline updates.
The V23 sample view and V2F/V3 native-coverage panel have different row universes.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sys
import tomllib

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod
from shapely.geometry import Polygon, box, shape
from shapely.geometry.polygon import orient

from .inputs import InputCatalog, REPO, disjoint, stream_sha

FIELDS = ['temp_c', 'wind_ms', 'ssrd_processed', 'tp_processed']
KEYS = ['timestamp_utc', 'geography']
GRID_DIMS = ['valid_time', 'latitude', 'longitude']
DESTINATIONS = {'AT': 'IT-NORD', 'CH': 'IT-NORD', 'SI': 'IT-NORD',
                'GR': 'IT-SUD', 'ME': 'IT-CSUD'}


class WeatherError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise WeatherError(code)


def json_input(catalog, aid, purpose='process'):
    with catalog.open_input(aid, purpose=purpose) as handle:
        return json.load(handle)


def csv_input(catalog, aid, purpose='process'):
    with catalog.open_input(aid, purpose=purpose) as handle:
        return pd.read_csv(handle)  # Frozen parser's default precision, not renormalized.


def geodesic_area(geometry):
    if geometry.is_empty:
        return 0.0
    if isinstance(geometry, Polygon):
        return abs(Geod(ellps='WGS84').geometry_area_perimeter(orient(geometry, sign=1))[0])
    return sum(geodesic_area(part) for part in geometry.geoms)


def validate_weights(mask):
    require(set(mask.columns) == {'latitude', 'longitude', 'intersection_area_m2', 'area_weight'},
            'BLOCKED_MASK_SCHEMA')
    require(len(mask) > 0 and np.isfinite(mask.to_numpy(dtype=float)).all(), 'BLOCKED_MASK_NONFINITE')
    require(not mask.duplicated(['latitude', 'longitude']).any(), 'BLOCKED_DUPLICATE_GRID_CELL')
    require(np.allclose(mask[['latitude', 'longitude']] * 4,
                        np.round(mask[['latitude', 'longitude']] * 4), atol=1e-10, rtol=0),
            'BLOCKED_MASK_GRID_RESOLUTION')
    w = mask.area_weight.to_numpy()
    require((w > 0).all() and (mask.intersection_area_m2 > 0).all()
            and np.isclose(w.sum(), 1, atol=1e-12, rtol=0), 'BLOCKED_AREA_WEIGHT')
    return w


def load_masks(catalog, units):
    obj = json_input(catalog, 'geography_masks')
    # GeoJSON has no embedded CRS member. Geographic lon/lat and the frozen
    # registry's WGS84 area method provide the CRS evidence; no CRS is invented.
    registry = csv_input(catalog, 'geography_registry').set_index('unit')
    geoms = {f['properties']['unit']: shape(f['geometry']) for f in obj['features']}
    masks, audit = {}, []
    for unit in units:
        require(unit in geoms and unit in registry.index, 'BLOCKED_GEOGRAPHY_ID')
        geom = geoms[unit]
        require(geom.is_valid and not geom.is_empty, 'BLOCKED_MASK_GEOMETRY')
        require('WGS84' in registry.loc[unit, 'area_method'], 'BLOCKED_CRS_EVIDENCE')
        aid = 'geography_' + unit.replace('-', '_') + '_weights'
        mask = csv_input(catalog, aid)
        w = validate_weights(mask)
        require(len(mask) == int(registry.loc[unit, 'grid_cells']), 'BLOCKED_MASK_GRID_COUNT')
        areas = np.array([geodesic_area(geom.intersection(box(
            r.longitude-.125, r.latitude-.125, r.longitude+.125, r.latitude+.125)))
            for r in mask.itertuples(index=False)])
        # Independent QA only. The recalculated values never replace input weights.
        require(np.allclose(areas, mask.intersection_area_m2, atol=.1, rtol=1e-10)
                and np.allclose(areas / areas.sum(), w, atol=1e-12, rtol=1e-10),
                'BLOCKED_FROZEN_MASK_CONFLICT')
        masks[unit] = mask
        audit.append(dict(geography=unit, grid_cells=len(mask), weight_sum=float(w.sum()),
                          max_area_error_m2=float(np.max(np.abs(areas-mask.intersection_area_m2))),
                          max_weight_error=float(np.max(np.abs(areas/areas.sum()-w))),
                          polygon_area_m2=geodesic_area(geom), intersection_area_m2=float(areas.sum()),
                          crs='WGS84 lon/lat; frozen registry evidence',
                          embedded_crs=obj.get('crs', 'NOT_EMBEDDED'),
                          mapping_status=registry.loc[unit, 'mapping_status'],
                          weights_asset_id=aid, weights_sha256=catalog.assets[aid]['sha256'], status='PASS'))
    return masks, audit


def expver_codes(values):
    result = []
    for value in np.asarray(values).ravel():
        if isinstance(value, bytes):
            value = value.decode('ascii')
        code = str(value).lstrip('0')
        require(code in {'1', '5'}, 'BLOCKED_UNKNOWN_EXPVER')
        result.append(code)
    return result


def select_expver(array):
    if 'expver' not in array.dims:
        if 'expver' in array.coords:
            expver_codes(array.expver.values)
        return array
    codes = expver_codes(array.expver.values)
    require(len(codes) == len(set(codes)), 'BLOCKED_DUPLICATE_EXPVER')
    levels = dict(zip(codes, array.expver.values))
    order = [levels[k] for k in ['1', '5'] if k in levels]
    result = array.sel(expver=order[0], drop=True)
    for level in order[1:]:
        result = result.fillna(array.sel(expver=level, drop=True))
    return result


def validate_native(dataset, semantics):
    require('oper' in dataset.attrs.get('history', ''), 'BLOCKED_STREAM_NOT_OPER')
    for name in GRID_DIMS:
        require(name in dataset.coords and dataset[name].dims == (name,), 'BLOCKED_COORDINATE:'+name)
    t = pd.DatetimeIndex(pd.to_datetime(dataset.valid_time.values, utc=True))
    require(len(t) > 0 and not t.hasnans and t.is_unique, 'BLOCKED_DUPLICATE_OR_INVALID_TIME')
    require(t.is_monotonic_increasing and (t == t.floor('h')).all(), 'BLOCKED_HOURLY_TIME')
    for name, step, limit in [('latitude', -.25, 90), ('longitude', .25, 180)]:
        values = np.asarray(dataset[name].values, dtype=float)
        require(len(values) > 0 and np.isfinite(values).all() and (np.abs(values) <= limit).all(),
                'BLOCKED_COORDINATE:'+name)
        require(len(np.unique(np.round(values, 6))) == len(values), 'BLOCKED_DUPLICATE_COORDINATE')
        require(np.allclose(np.diff(values), step, atol=1e-10, rtol=0), 'BLOCKED_COORDINATE_ORDER_OR_GRID')
    variables = set(dataset.data_vars)
    require(variables in [{'t2m', 'u10', 'v10'}, {'ssrd', 'tp'}], 'BLOCKED_VARIABLE_IDENTITY')
    for var in sorted(variables):
        array = dataset[var]
        reference = semantics['variables'][var]['native_metadata']
        for key in ['units', 'GRIB_stepType', 'GRIB_gridType', 'GRIB_dataType', 'GRIB_paramId', 'GRIB_stepUnits']:
            require(str(array.attrs.get(key)) == str(reference[key]), 'BLOCKED_VARIABLE_SEMANTICS:'+var+':'+key)
        for key in ['GRIB_iDirectionIncrementInDegrees', 'GRIB_jDirectionIncrementInDegrees']:
            require(array.attrs.get(key) == .25, 'BLOCKED_GRID_RESOLUTION')
        require(set(GRID_DIMS) <= set(array.dims), 'BLOCKED_VARIABLE_DIMENSIONS')
        for dim in array.dims:
            require(dim in GRID_DIMS+['expver'] or array.sizes[dim] == 1, 'BLOCKED_UNEXPECTED_DIMENSION')
    codes = sorted(set(expver_codes(dataset.expver.values))) if 'expver' in dataset.coords else []
    return t, dict(dimensions=dict(dataset.sizes), variables=sorted(variables),
                   coordinates={axis: dict(dimensions=list(dataset[axis].dims), dtype=str(dataset[axis].dtype),
                                           first=str(dataset[axis].values[0]), last=str(dataset[axis].values[-1]),
                                           units=dataset[axis].attrs.get('units', 'CF time encoding' if axis == 'valid_time' else 'UNVERIFIED'))
                                for axis in GRID_DIMS},
                   first_valid_time=str(t.min()), last_valid_time=str(t.max()), valid_times=len(t),
                   time_gap_count=int((np.diff(t.as_unit('ns').asi8) != 3600*10**9).sum()),
                   latitude_order='decreasing', longitude_order='increasing',
                   expver=codes or ['UNVERIFIED_NOT_EMBEDDED'],
                   expver_dimensions=list(dataset.expver.dims) if 'expver' in dataset.coords else [],
                   CF_time_units=dataset.valid_time.encoding.get('units', 'UNVERIFIED'),
                   CF_calendar=dataset.valid_time.encoding.get('calendar', 'UNVERIFIED'),
                   step_coordinate_present='step' in dataset.coords,
                   forecast_reference_time_present='forecast_reference_time' in dataset.coords,
                   accumulation_evidence='Native GRIB_stepType plus hash-bound frozen official semantics')


@contextmanager
def open_native(path, *, backend='netcdf4'):
    require(backend in {'netcdf4', 'h5netcdf'}, 'BLOCKED_NETCDF_BACKEND')
    try:
        with xr.open_dataset(path, engine=backend, decode_cf=True, mask_and_scale=True,
                             cache=False) as dataset:
            yield dataset
    except WeatherError:
        raise
    except (OSError, ValueError, ImportError, KeyError) as error:
        raise WeatherError('BLOCKED_NETCDF_DECODE:'+type(error).__name__) from error


def grid_indices(dataset, mask):
    lat = {round(float(x), 6): i for i, x in enumerate(dataset.latitude.values)}
    lon = {round(float(x), 6): i for i, x in enumerate(dataset.longitude.values)}
    require(all(round(float(x), 6) in lat for x in mask.latitude) and
            all(round(float(x), 6) in lon for x in mask.longitude), 'BLOCKED_MISSING_MASK_GRID')
    return ([lat[round(float(x), 6)] for x in mask.latitude],
            [lon[round(float(x), 6)] for x in mask.longitude])


def select_grid(array, dataset, mask):
    array = select_expver(array)
    for dim in list(array.dims):
        if dim not in GRID_DIMS:
            require(array.sizes[dim] == 1, 'BLOCKED_UNEXPECTED_DIMENSION')
            array = array.isel({dim: 0}, drop=True)
    rows, cols = grid_indices(dataset, mask)
    return array.transpose(*GRID_DIMS).values[:, rows, cols].astype(float)


def weighted_fields(grids, weights, semantics):
    weights = np.asarray(weights, dtype=float)
    require((weights > 0).all() and np.isclose(weights.sum(), 1, atol=1e-12, rtol=0), 'BLOCKED_AREA_WEIGHT')
    require(set(grids) <= {'t2m', 'u10', 'v10', 'ssrd', 'tp'}, 'BLOCKED_VARIABLE_IDENTITY')
    require(('u10' in grids) == ('v10' in grids), 'BLOCKED_WIND_COMPONENT')
    out = {}
    for var in ['t2m', 'ssrd', 'tp']:
        if var in grids:
            spec = semantics['variables'][var]
            out[spec['output']] = ((grids[var]*spec['scale']+spec['offset']) @ weights,
                                   int(spec['delivery_start_offset_hours']))
    if 'u10' in grids:
        out['wind_ms'] = (np.hypot(grids['u10'], grids['v10']) @ weights, 0)
    return out  # Matrix product propagates missing cells; no skipna/renormalization.


def merge_records(long, *, strict=False):
    keys = KEYS+['variable']
    duplicates = long[long.duplicated(keys, keep=False)]
    grouped = duplicates.groupby(keys).value
    bounds = grouped.agg(['min', 'max', 'count', 'size'])
    delta = bounds['max']-bounds['min']
    maxdiff = float(delta.max()) if delta.notna().any() else 0.0
    require(not ((bounds['count'] > 0) & (bounds['count'] != bounds['size'])).any(),
            'BLOCKED_REVISION_MISSING_CONFLICT')
    require(maxdiff == 0 if strict else maxdiff < 1e-10, 'BLOCKED_REVISION_CONFLICT')
    selected = long.drop_duplicates(keys, keep='first')
    panel = selected.pivot(index=KEYS, columns='variable', values='value').reset_index()
    return panel, selected, dict(duplicate_records=len(duplicates), duplicate_keys=len(bounds),
                                 max_abs_difference=maxdiff, selection='first in frozen request/content order')


def native_overlap_audit(current, previous, catalog, backend, masks):
    """Compare actual overlapping grid values, not just regional averages.

    Only overlapping slices of previous files are reread, with at most two files
    open. Prior metadata, rather than all prior data cubes, stays in memory.
    """
    rows = []
    for prior in previous:
        common = {axis: np.intersect1d(current[axis].values, prior[axis]) for axis in GRID_DIMS}
        variables = set(current.data_vars) & set(prior['variables'])
        if not variables or any(len(x) == 0 for x in common.values()):
            continue
        with open_native(catalog.resolve(prior['asset_id']), backend=backend) as old:
            for var in sorted(variables):
                left = select_expver(current[var].sel(common)).transpose(*GRID_DIMS).values
                right = select_expver(old[var].sel(common)).transpose(*GRID_DIMS).values
                same_nulls = np.array_equal(np.isnan(left), np.isnan(right))
                valid = ~np.isnan(left) & ~np.isnan(right)
                diff = float(np.max(np.abs(left[valid].astype(float)-right[valid].astype(float)))) if valid.any() else 0.
                shared = []
                for unit, mask in masks.items():
                    full = lambda d: (mask.latitude.isin(d.latitude.values).all()
                                      and mask.longitude.isin(d.longitude.values).all())
                    if full(current) and full(old):
                        shared.append(unit)
                status = 'IDENTICAL_OVERLAP'
                if not same_nulls or diff >= 1e-10:
                    # Frozen regional files are not a mosaic. A partial polygon
                    # is never used to supply cells for another request's full
                    # polygon. Audit different values without inventing a new
                    # global-grid revision rule that the frozen code did not use.
                    for unit in shared:
                        a = select_grid(current[var].sel(valid_time=common['valid_time']), current, masks[unit])
                        b = select_grid(old[var].sel(valid_time=common['valid_time']), old, masks[unit])
                        require(np.array_equal(np.isnan(a), np.isnan(b)), 'BLOCKED_NATIVE_REVISION_NULL_CONFLICT')
                        observed = ~np.isnan(a)
                        require(not observed.any() or float(np.max(np.abs(a[observed]-b[observed]))) < 1e-10,
                                'BLOCKED_NATIVE_REVISION_CONFLICT')
                    status = 'HANDLED_OUTSIDE_COMMON_MASKS' if shared else 'HANDLED_DISJOINT_COMPLETE_MASKS'
                rows.append(dict(previous_asset_id=prior['asset_id'], variable=var,
                                 overlapping_grid_values=int(left.size), max_abs_difference=diff,
                                 null_masks_equal=bool(same_nulls), common_complete_masks=shared,
                                 status=status, difference_cause='UNVERIFIED' if status != 'IDENTICAL_OVERLAP' else 'not applicable',
                                 source_substitution_or_averaging=False))
    return rows


def sample_times(catalog):
    result = {}
    for name, aid in [('R1', 'sample_ME_R1'), ('F', 'sample_ME_F')]:
        t = pd.DatetimeIndex(pd.to_datetime(csv_input(catalog, aid).timestamp_utc, utc=True)).as_unit('ns')
        require(t.is_unique and t.is_monotonic_increasing, 'BLOCKED_FROZEN_SAMPLE')
        result[name] = t
    return result


def v23_sample_view(records, catalog, samples, config):
    """Reproduce V23's explicit expected-sample shells, without baseline values.

    CH_R1 uses the same timestamp sample as ME_R1. Verify this independently
    against frozen upstream CH delivery timestamps and the frozen ±90-day rule.
    """
    upstream = csv_input(catalog, 'frozen_v2_panel')
    ch = pd.DatetimeIndex(pd.to_datetime(upstream.loc[upstream.corridor_id.eq('CH_IT'), 'timestamp_utc'], utc=True))
    cutoff = pd.Timestamp(config['time']['policy_cutoff_utc'])
    distance = (ch+pd.Timedelta(minutes=30)-cutoff) / pd.Timedelta(days=1)
    ch = ch[np.abs(distance) < 90].as_unit('ns')
    require(ch.equals(samples['R1']), 'BLOCKED_CH_R1_SAMPLE_ALIAS')
    requests = json_input(catalog, 'era5_requests_v23')['requests']
    groups = {}
    for request in requests:
        if request.get('superseded_by') or request['group'] == 'probe':
            continue
        groups.setdefault(request['group'], []).append(request)
    frames = []
    for group, requests in sorted(groups.items()):
        require(group in {'ME_R1', 'ME_F', 'CH_R1'}, 'BLOCKED_V23_GROUP')
        units = requests[0]['units']
        expected = samples['F' if group == 'ME_F' else 'R1']
        parts = []
        for request in requests:
            if request.get('downloaded'):
                rid = 'v23:'+request['id']
                for line in catalog.manifest['era5_lineage']:
                    if line['request_id'] == rid:
                        q = records[records.asset_id.eq(line['asset_id']) & records.geography.isin(units)]
                        parts.append(q[q.timestamp_utc.isin(expected)])
        require(bool(parts), 'BLOCKED_V23_GROUP_NO_SOURCES')
        panel, _, _ = merge_records(pd.concat(parts, ignore_index=True), strict=True)
        index = pd.MultiIndex.from_product([expected, units], names=KEYS)
        frame = panel.set_index(KEYS).reindex(index).reset_index()
        frames.append(frame[KEYS+FIELDS])
    combined = pd.concat(frames, ignore_index=True)
    require(not combined.groupby(KEYS)[FIELDS].nunique().gt(1).any().any(), 'BLOCKED_V23_GROUP_CONFLICT')
    # Exact original consolidation: first observed value across frozen groups,
    # retaining all-missing expected-sample rows. This is not raw-data imputation.
    return combined.groupby(KEYS)[FIELDS].first().reset_index().sort_values(KEYS).reset_index(drop=True)


@dataclass
class Reconstruction:
    panel: pd.DataFrame
    v23_panel: pd.DataFrame
    lineage: pd.DataFrame
    diagnostics: dict


def reconstruct(catalog, *, backend='netcdf4', progress=None):
    config = tomllib.loads((REPO / 'config/main.toml').read_text())
    semantics = json_input(catalog, 'weather_semantics')
    require(semantics['status'] == 'VERIFIED', 'BLOCKED_UNVERIFIED_SEMANTICS')
    require(catalog.assets['era5_semantics_document']['sha256'] == semantics['official_document']['sha256'],
            'BLOCKED_SEMANTICS_DOCUMENT')
    catalog.resolve('era5_semantics_document')
    masks, mask_audit = load_masks(catalog, config['weather']['geographies'])
    ids = list(dict.fromkeys(r['asset_id'] for r in catalog.manifest['era5_lineage']))
    require(len(ids) == catalog.manifest['era5_contract']['unique_content_files'], 'BLOCKED_ERA5_CLOSURE')
    records, files, spatial, prior_files, overlaps = [], [], [], [], []
    for number, aid in enumerate(ids, 1):
        path = catalog.resolve(aid)
        with open_native(path, backend=backend) as dataset:
            times, audit = validate_native(dataset, semantics)
            # One file at a time, once loaded for native audit and all geographies.
            dataset.load()
            counts = {}
            for var in audit['variables']:
                values = dataset[var].values
                require(not np.isinf(values).any(), 'BLOCKED_NATIVE_INFINITE')
                counts[var] = dict(native_values=int(values.size), nulls=int(np.isnan(values).sum()),
                                   dtype=str(values.dtype), units=dataset[var].attrs['units'],
                                   fill_value=str(dataset[var].encoding.get('_FillValue', 'NOT_EMBEDDED')),
                                   GRIB_stepType=dataset[var].attrs['GRIB_stepType'])
            overlap = native_overlap_audit(dataset, prior_files, catalog, backend, masks)
            overlaps.extend(dict(current_asset_id=aid, **r) for r in overlap)
            full = []
            for unit, mask in masks.items():
                covered = mask.latitude.round(6).isin(np.round(dataset.latitude.values, 6)) & mask.longitude.round(6).isin(np.round(dataset.longitude.values, 6))
                spatial.append(dict(asset_id=aid, geography=unit, mask_cells=len(mask),
                                    covered_cells=int(covered.sum()), area_share=float(mask.loc[covered, 'area_weight'].sum()),
                                    status='SPATIAL_MASK_COMPLETE' if covered.all() else 'BLOCKED_SPATIAL_COVERAGE'))
                if not covered.all():
                    continue  # A regional request cannot supply a different incomplete polygon.
                full.append(unit)
                grids = {var: select_grid(dataset[var], dataset, mask) for var in audit['variables']}
                for field, (values, shift) in weighted_fields(grids, mask.area_weight.to_numpy(), semantics).items():
                    records.append(pd.DataFrame(dict(timestamp_utc=times+pd.Timedelta(hours=shift),
                                                      geography=unit, variable=field, value=values, asset_id=aid)))
            require(bool(full), 'BLOCKED_FILE_NO_COMPLETE_MASK')
            files.append(dict(asset_id=aid, sha256=catalog.assets[aid]['sha256'], backend=backend,
                              fully_covered_geographies=full, native=counts, **audit))
            prior_files.append(dict(asset_id=aid, variables=audit['variables'],
                                    **{axis: dataset[axis].values.copy() for axis in GRID_DIMS}))
        if progress:
            progress(number, len(ids), aid)
    long = pd.concat(records, ignore_index=True)
    panel, lineage, duplication = merge_records(long)
    first = pd.Timestamp(config['time']['window_start_inclusive'])
    end = pd.Timestamp(config['time']['window_end_exclusive'])
    panel = panel[panel.timestamp_utc.ge(first) & panel.timestamp_utc.lt(end)].sort_values(KEYS).reset_index(drop=True)
    lineage = lineage[lineage.timestamp_utc.ge(first) & lineage.timestamp_utc.lt(end)]
    samples = sample_times(catalog)
    v23 = v23_sample_view(long, catalog, samples, config)
    return Reconstruction(panel, v23, lineage.drop(columns='value'),
                          dict(files=files, masks=mask_audit, spatial=spatial,
                               native_overlaps=overlaps, regional_overlaps=duplication,
                               request_lineage=catalog.manifest['era5_lineage'],
                               V23_sample_rule='Original group shells and first-observed consolidation; CH_R1 alias independently verified',
                               raw_missing_values_filled=False, weights_replaced=False))


def compare_panel(actual, frozen, *, tolerance):
    a, b = actual.copy(), frozen.copy()
    for frame in [a, b]:
        require(set(frame.columns) == set(KEYS+FIELDS), 'BLOCKED_PANEL_SCHEMA')
        frame['timestamp_utc'] = pd.to_datetime(frame.timestamp_utc, utc=True).astype('datetime64[ns, UTC]')
        require(not frame.duplicated(KEYS).any(), 'BLOCKED_DUPLICATE_PANEL_KEY')
    key_equal = pd.MultiIndex.from_frame(a[KEYS]).equals(pd.MultiIndex.from_frame(b[KEYS]))
    joined = b.merge(a, on=KEYS, how='outer', suffixes=('_frozen', '_new'), indicator=True, validate='one_to_one')
    report = dict(status='PASS', rows_new=len(a), rows_frozen=len(b), keys_and_order_equal=key_equal,
                  missing_keys=int(joined._merge.eq('left_only').sum()), extra_keys=int(joined._merge.eq('right_only').sum()),
                  tolerance_strict_abs_less_than=tolerance, fields={})
    differences = []
    if not key_equal:
        report['status'] = 'FAIL'
        differences.extend(joined.loc[joined._merge.ne('both'), KEYS+['_merge']].to_dict('records'))
    for field in FIELDS:
        x, y = joined[field+'_frozen'], joined[field+'_new']
        mask_equal = x.isna().equals(y.isna())
        both = x.notna() & y.notna()
        delta = (x-y).abs()
        finite = np.isfinite(x[both]).all() and np.isfinite(y[both]).all()
        maximum = float(delta[both].max()) if both.any() else 0.
        nonzero = both & x.ne(0)
        relative = float((delta[nonzero]/x[nonzero].abs()).max()) if nonzero.any() else 0.
        report['fields'][field] = dict(null_new=int(a[field].isna().sum()), null_frozen=int(b[field].isna().sum()),
                                       null_mask_equal=mask_equal, compared_values=int(both.sum()),
                                       max_abs_difference=maximum, max_relative_difference=relative)
        bad = x.isna().ne(y.isna()) | (both & delta.ge(tolerance))
        if not mask_equal or not finite or not maximum < tolerance:
            report['status'] = 'FAIL'
        for row in joined.loc[bad, KEYS+[field+'_frozen', field+'_new']].to_dict('records'):
            differences.append(dict(variable=field, **row))
    return report, differences


def coverage(panel, catalog):
    samples = sample_times(catalog)
    geography, bilateral, composition = [], [], []
    for model, times in samples.items():
        complete = {}
        for unit, group in panel.groupby('geography'):
            q = group.set_index('timestamp_utc').reindex(times)
            complete[unit] = q[FIELDS].notna().all(axis=1)
            geography.append(dict(model=model, geography=unit, expected_hours=len(times),
                                  complete_hours=int(complete[unit].sum())))
        for origin, destination in DESTINATIONS.items():
            valid = complete[origin] & complete[destination]
            bilateral.append(dict(series=origin+'_actual', model=model, origin=origin, destination=destination,
                                  expected_hours=len(times), complete_hours=int(valid.sum()),
                                  status='WEATHER_PANEL_COMPLETE' if valid.all() else 'BLOCKED_WEATHER_PANEL'))
            for year in sorted(set(times.tz_convert('Europe/Rome').year)):
                take = times.tz_convert('Europe/Rome').year == year
                composition.append(dict(series=origin+'_actual', model=model, year=int(year),
                                        expected_hours=int(take.sum()), complete_case_hours=int(valid.to_numpy()[take].sum()),
                                        missing_hours=int((~valid.to_numpy()[take]).sum())))
    # Compare coverage with registered diagnostics, without claiming models ran.
    existing = csv_input(catalog, 'engine_weather_model_status', 'verify')
    for row in bilateral:
        check = existing[existing.series.eq(row['series']) &
                         existing.model.isin(['R1'] if row['model'] == 'R1' else ['F1', 'F2'])]
        require(len(check) > 0, 'BLOCKED_MISSING_WEATHER_STATUS')
        if row['model'] == 'R1':
            require(check.status.eq('WEATHER_ADJUSTED_RUN').all() and
                    check.N.eq(row['complete_hours']).all() and row['complete_hours'] == row['expected_hours'],
                    'BLOCKED_R1_COVERAGE_CHANGED')
        else:
            require(check.status.eq(row['status']).all() and check.expected_N.eq(row['expected_hours']).all()
                    and check.complete_N.eq(row['complete_hours']).all(), 'BLOCKED_ANNUAL_COVERAGE_CHANGED')
    annual = csv_input(catalog, 'engine_weather_synthetic_annual_coverage', 'verify')
    for row in annual.itertuples():
        new = next(x for x in geography if x['model'] == 'F' and x['geography'] == row.geography)
        require((new['expected_hours'], new['complete_hours']) == (row.expected_hours, row.complete_hours),
                'BLOCKED_SYNTHETIC_ANNUAL_COVERAGE_CHANGED')
    units = ['ME', 'IT-CSUD', 'GR', 'IT-SUD', 'AT', 'IT-NORD']
    missing = [u for u in units if any(x['geography'] == u and x['model'] == 'F'
                                     and x['complete_hours'] != x['expected_hours'] for x in geography)]
    reason = ('Incomplete annual native bilateral weather: '+','.join(missing) if missing else
              'Annual synthetic weather implementation not executed by R1-only module')
    status = csv_input(catalog, 'engine_weather_synthetic_model_status', 'verify')
    annual_status = status[status.model.isin(['F1', 'F2'])]
    require(len(annual_status) == 3 and annual_status.status.eq('NOT_RUN').all()
            and annual_status.reason.eq(reason).all(), 'BLOCKED_SYNTHETIC_ANNUAL_STATUS_CHANGED')
    expected_composition = csv_input(catalog, 'engine_weather_sample_composition', 'verify')
    new_composition = pd.DataFrame(composition)
    columns = ['series', 'model', 'year', 'expected_hours', 'complete_case_hours', 'missing_hours']
    require(new_composition[columns].sort_values(columns[:3]).reset_index(drop=True).equals(
            expected_composition[columns].sort_values(columns[:3]).reset_index(drop=True)), 'BLOCKED_YEAR_COMPOSITION_CHANGED')
    return dict(geography=geography, bilateral=bilateral, by_year=composition,
                synthetic_annual_status=annual_status.fillna('').to_dict('records'),
                model_estimation_executed=False)


def environment_record():
    import netCDF4
    import h5py
    import pyproj
    import shapely
    buffer = io.StringIO()
    from contextlib import redirect_stdout
    with redirect_stdout(buffer):
        np.show_config()
    packages = {name: importlib.metadata.version(name) for name in
                ['numpy', 'pandas', 'scipy', 'pyarrow', 'xarray', 'netCDF4', 'h5netcdf',
                 'h5py', 'shapely', 'pyproj', 'lxml', 'openpyxl', 'pytest']}
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages,
                netcdf=netCDF4.__netcdf4libversion__, netcdf_HDF5=netCDF4.__hdf5libversion__,
                h5py_HDF5=h5py.version.hdf5_version, PROJ=pyproj.proj_version_str,
                GEOS=shapely.geos_version_string, numpy_build=buffer.getvalue(),
                optional_reporting='Not required or installed by PR3; no PDF/font capability claim')


def run_weather_validation(catalog, run_directory, *, backend='netcdf4', progress=None):
    directory = Path(run_directory).expanduser().absolute()
    require(not any(p.is_symlink() for p in [directory, *directory.parents]), 'BLOCKED_OUTPUT_SYMLINK')
    directory = directory.resolve()
    require(catalog.output_root in directory.parents, 'BLOCKED_OUTPUT_ROOT')
    disjoint(directory, catalog.input_root)
    disjoint(directory, REPO)
    require(not directory.exists(), 'BLOCKED_OUTPUT_ALREADY_EXISTS')
    directory.mkdir(parents=True)

    def write_json(name, obj):
        (directory/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))

    receipt = dict(scope='ERA5 weather only; no economic model estimation', backend=backend,
                   identities={name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in {
                                   'code/weather.py': Path(__file__),
                                   'code/inputs.py': Path(__file__).with_name('inputs.py'),
                                   **{name: REPO/name for name in ['config/main.toml', 'manifests/inputs.toml',
                                                                 'manifests/baselines.toml', 'uv.lock']}
                               }.items()},
                   status='STARTED')
    write_json('environment.json', environment_record())
    write_json('receipt.json', receipt)
    try:
        result = reconstruct(catalog, backend=backend, progress=progress)
        result.panel.to_parquet(directory/'weather_panel.parquet', index=False)
        result.v23_panel.to_parquet(directory/'v23_sample_view.parquet', index=False)
        result.lineage.to_parquet(directory/'selection_lineage.parquet', index=False)
        write_json('native_and_spatial_audit.json', result.diagnostics)
        comparisons, differences = {}, []
        tolerance = catalog.baselines['validation']['weather']['strict_abs_less_than']
        for aid, panel in [('weather_v23_benchmark', result.v23_panel),
                           ('weather_v2f_benchmark', result.panel), ('engine_weather_panel_hourly', result.panel)]:
            frozen = pd.read_parquet(catalog.resolve(aid, purpose='verify'))
            report, diff = compare_panel(panel, frozen, tolerance=tolerance)
            comparisons[aid] = report
            differences.extend(dict(reference=aid, **row) for row in diff)
        write_json('comparisons.json', comparisons)
        write_json('differences.json', differences)
        require(all(r['status'] == 'PASS' for r in comparisons.values()), 'BLOCKED_WEATHER_NUMERICAL_EQUIVALENCE')
        support = coverage(result.panel, catalog)
        write_json('coverage.json', support)
        receipt.update(status='PASS', decoded_contents=len(result.diagnostics['files']),
                       panel_shape=list(result.panel.shape), v23_shape=list(result.v23_panel.shape),
                       geography_hours=result.panel.groupby('geography').size().to_dict(),
                       null_counts=result.panel[FIELDS].isna().sum().to_dict(),
                       output_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in directory.glob('*.parquet')})
        write_json('receipt.json', receipt)
        return result, comparisons, support, receipt
    except Exception as error:
        receipt.update(status='FAIL', error=type(error).__name__+': '+str(error))
        write_json('receipt.json', receipt)
        raise
