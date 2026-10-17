"""Frozen-weight Montenegro synthetic reconstruction, with explicit inputs.

Source mapping: V3/src/reused/synthetic_me.py supplies prediction arithmetic,
validation metrics, UTC sample hashes and summary statistics; reused/
weight_learning.py supplies Eurostat and frozen feature transformations. The
fixed objective uses its documented row weighting, evaluated as in
V3.1/src/synthetic_method_summary.py::audit_frozen_objective. No optimizer is
included. Historical benchmark reads are isolated in validate_frozen().

The V2 hourly panel is an inherited frozen upstream input, not a claim that
all original provider observations were reconstructed by this package.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

DONORS = ["GR_IT", "AT_IT"]
Y = "scheduled_export_proxy"
VERSIONS = ["GY050_train2024", "GY050_final"]
ALL_VERSIONS = tuple(sorted(f"{tag}_{suffix}" for tag in ["GY025", "GY050", "GY075", "G"]
                           for suffix in ["train2024", "final", "final_loo_AT_IT", "final_loo_GR_IT"]))
COUNTRIES = {"CH": "Switzerland", "ME": "Montenegro", "FR": "France", "AT": "Austria", "SI": "Slovenia", "GR": "Greece"}
SOURCE_FUNCTIONS = {
    "construct_paths": "V3/src/reused/synthetic_me.py::build_synthetic",
    "summarize": "V3/src/reused/synthetic_me.py::build_synthetic",
    "build_predictors": "V3/src/reused/weight_learning.py::build_predictors",
    "build_energy_mix": "V3/src/reused/weight_learning.py::build_energy_mix",
    "fixed_weight_objective": "V3.1/src/synthetic_method_summary.py::audit_frozen_objective",
}


FROZEN_FEATURES = {
    "version": "weight_features_v1_frozen_before_V2_post_regressions",
    "energy": "equal-year mean of standardised seven-category feature mean",
    "outcome_blocks": ["monthly_path", "hour_shape", "moments", "quantiles", "zero_top"],
    "quantiles": [.05, .10, .25, .50, .75, .90, .95],
    "zero_top": "zero fraction; fraction >= target 2024 P95 (diagnostic, not sample selection)",
    "scaling": "cross-object population SD including target; energy scales from 2024; outcome block-specific scales from 2024; shared all-object constants alone omitted",
    "statistical_threshold": "UNCONFIRMED: DESCRIPTIVE_ONLY, never validated causal counterfactual",
    "final_choice": "GY lambda=.5 fixed ex ante for this V2; all .25/.75/G retained without 2026 selection",
    "hydro": "Eurostat RA100 all net hydro output including separately reported mixed/pumped; not natural-water availability",
    "coal": "Eurostat C0000 coal and manufactured gases; includes coal-derived gases, not falsely labelled pure coal/lignite",
    "other": "reported TOTAL minus six separately measured categories; exact exhaustive residual, not missing-input imputation",
}

def sample_hash(index):
    """Canonical UTC nanosecond timestamp hash; frozen legacy hash is separate."""
    idx = pd.DatetimeIndex(index).as_unit("ns")
    return hashlib.sha256(idx.asi8.tobytes()).hexdigest()

def frozen_sample_hash(index):
    """V21 CSV timestamps were parsed as microseconds in the frozen environment."""
    return hashlib.sha256(pd.DatetimeIndex(index).as_unit("us").asi8.tobytes()).hexdigest()

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def metrics(actual, prediction, training_sd):
    error = np.asarray(prediction) - np.asarray(actual)
    result = dict(n=len(error), rmse=float(np.sqrt(np.mean(error**2))),
                  mae=float(np.mean(abs(error))), bias=float(error.mean()),
                  nrmse=float(np.sqrt(np.mean(error**2)) / training_sd),
                  training_target_sd=float(training_sd))
    for name, values in [("actual", actual), ("predicted", prediction)]:
        values = np.asarray(values)
        result.update({name + "_mean": float(values.mean()),
                       name + "_sd": float(values.std(ddof=1)),
                       name + "_zero_share": float(np.mean(values == 0))})
        for q in [5, 50, 95]:
            result[name + "_p" + str(q)] = float(np.percentile(values, q))
    return result

def weighted_complete(matrix, weights):
    """No hourly renormalisation; required donor missing => whole hour missing."""
    matrix = np.asarray(matrix, float)
    result = np.full(matrix.shape[0], np.nan)
    valid = np.isfinite(matrix).all(axis=1)
    result[valid] = matrix[valid] @ np.asarray(weights, float)
    return result

def outcome_features(wide, years, top_reference):
    dates = pd.DatetimeIndex(wide.index).tz_convert("Europe/Rome")
    z = wide.loc[dates.year.isin(years)]
    d = pd.DatetimeIndex(z.index).tz_convert("Europe/Rome")
    # All target/donor features use the same complete-case learning hours.
    z = z.dropna()
    d = pd.DatetimeIndex(z.index).tz_convert("Europe/Rome")
    if len(z) == 0 or not set(years).issubset(set(d.year)):
        raise ValueError("BLOCKED_PRE_FLOW")
    blocks = []
    for year in years:
        for month in range(1, 13):
            part = z[(d.year == year) & (d.month == month)]
            if len(part) == 0:
                raise ValueError("BLOCKED_PRE_MONTH")
            blocks.append((f"monthly_path:{year}-{month:02}", "monthly_path", part.mean()))
    for hour in range(24):
        blocks.append((f"hour_shape:{hour:02}", "hour_shape", z[d.hour == hour].mean()))
    blocks += [("moments:mean", "moments", z.mean()), ("moments:sd", "moments", z.std(ddof=1))]
    for q in FROZEN_FEATURES["quantiles"]:
        blocks.append((f"quantiles:{q:.2f}", "quantiles", z.quantile(q)))
    blocks += [("zero_top:zero", "zero_top", (z == 0).mean()), ("zero_top:top", "zero_top", (z >= top_reference).mean())]
    return blocks, z

def build_predictors(mix, wide, target, donors, years, scaling=None):
    objects = [target, *donors]
    train = wide.loc[pd.DatetimeIndex(wide.index).tz_convert("Europe/Rome").year == 2024].dropna()
    if train.empty:
        raise ValueError("BLOCKED_TRAIN_DATA")
    top = float(train[target].quantile(.95))
    features = []
    for year in years:
        for feature in sorted(mix.feature.unique()):
            row = []
            for corridor in objects:
                selected = mix[(mix.country == corridor[:2]) & (mix.year == year) & (mix.feature == feature)]
                if len(selected) != 1 or selected.iloc[0].status != "FULL_WINDOW_VERIFIED":
                    raise ValueError("BLOCKED_ANNUAL_MIX:" + corridor + ":" + str(year))
                row.append(selected.iloc[0].share)
            features.append((f"energy:{year}:{feature}", "energy", pd.Series(row, index=objects)))
    yf, z = outcome_features(wide[objects], years, top)
    features += yf
    # Scales locked on training data and extended by named block pattern, not 2026.
    scales, diagnostics, out = {}, [], []
    for name, block, vector in features:
        a = vector.loc[objects].to_numpy(float)
        key = name.replace("2025", "2024") if block in ("energy", "monthly_path") else name
        raw_sd = float(np.std(a))
        scale = float(scaling[key]) if scaling is not None and key in scaling else raw_sd
        shared_constant = bool(np.max(a) - np.min(a) <= 1e-12)
        donor_constant_target_different = bool(np.max(a[1:]) - np.min(a[1:]) <= 1e-12 and not shared_constant)
        if shared_constant:
            # Preserve the zero sentinel: a newly varying 2025 predictor uses
            # deterministic unit scale, never a validation-year fitted scale.
            scales[key] = scale
            diagnostics.append(dict(feature=name, block=block, omitted=True, reason="SHARED_ALL_OBJECT_CONSTANT", scale=scale,
                                    target_value=a[0], donor_min=np.min(a[1:]), donor_max=np.max(a[1:]), supported=True))
            continue
        if not np.isfinite(scale) or scale < 1e-12:
            # The 2024 scale can be zero for a 2025 predictor newly varying. Use
            # deterministic unit scale, record it, never delete target mismatch.
            scale = 1.0
        scales[key] = scale
        diagnostics.append(dict(feature=name, block=block, omitted=False,
                                reason="DONOR_CONSTANT_TARGET_UNSUPPORTED" if donor_constant_target_different else "RETAINED",
                                scale=scale, target_value=a[0], donor_min=np.min(a[1:]), donor_max=np.max(a[1:]),
                                supported=bool(np.min(a[1:])-1e-10 <= a[0] <= np.max(a[1:])+1e-10)))
        out.append((name, block, a / scale))
    return out, scales, diagnostics, z

def strict_weighted_sum(frame, weights):
    """Missing any fixed-pool input => missing; zero weights do not hide data failures."""
    if set(frame.columns) != set(weights):
        raise ValueError("Missing or extra fixed donor")
    w = np.array([weights[c] for c in frame.columns])
    if np.any(w < -1e-12) or not np.isclose(w.sum(), 1, atol=1e-10):
        raise ValueError("Invalid simplex")
    out = pd.Series(frame.to_numpy() @ w, index=frame.index)
    out[frame.isna().any(axis=1)] = np.nan
    return out

def decode_eurostat(payload):
    j = payload
    dims, sizes = j["id"], j["size"]
    maps = [{int(v): k for k, v in j["dimension"][d]["category"]["index"].items()} for d in dims]
    rows = []
    for raw_index, value in j.get("value", {}).items():
        coords = np.unravel_index(int(raw_index), sizes)
        row = {dim: maps[i][int(coords[i])] for i, dim in enumerate(dims)}
        row.update(value=value, flag=j.get("status", {}).get(str(raw_index), ""))
        rows.append(row)
    return pd.DataFrame(rows, columns=dims + ["value", "flag"]), j

def build_energy_mix(catalog):
    """Use a SINGLE comparable net-generation series; no Ember gross/net splicing.

    Seven groups sum to TOTAL. Only explicit published zero is zero. Missing
    any of six component groups blocks 'other' and the annual matching vector.
    Eurostat C0000 aggregation includes manufactured gases; preserved visibly.
    """
    allrows, coverage, native = [], [], []
    codes = {"hydro": "RA100", "nuclear": "N9000", "coal_lignite_and_manufactured_gases": "C0000",
             "gas": "G3000", "wind": "RA300", "solar": "RA400"}
    for country in COUNTRIES:
        aid = "generation_" + ("EL" if country == "GR" else country)
        with catalog.open_input(aid) as handle:
            frame, meta = decode_eurostat(json.load(handle))
        frame = frame[frame.unit == "GWH"].copy()
        frame["country"] = country
        native.extend(frame.to_dict("records"))
        for year in (2024, 2025):
            f = frame[frame.time.str.startswith(str(year), na=False)]
            group_values = {}
            for feature, code in {**codes, "total": "TOTAL", "hydro_pure": "RA110", "hydro_mixed": "RA120", "hydro_pumped": "RA130"}.items():
                z = f[f.siec == code]
                complete = len(z) == 12 and z.time.nunique() == 12 and z.value.notna().all()
                value = float(z.value.sum()) if complete else np.nan
                group_values[feature] = value
                coverage.append(dict(country=country, year=year, feature=feature, siec=code, months=z.time.nunique(),
                                     status="FULL_WINDOW_VERIFIED" if complete else "BLOCKED", annual_gwh=value,
                                     zero_status="SOURCE_REPORTED_ZERO" if complete and value == 0 else "NOT_ZERO_OR_MISSING",
                                     missing_months=";".join(sorted(set(f"{year}-{m:02}" for m in range(1, 13))-set(z.time))),
                                     source="Eurostat nrg_cb_pem", source_updated=meta.get("updated", ""),
                                     source_sha256=catalog.assets[aid]["sha256"], flags=";".join(sorted(set(z.flag))),
                                     measurement="Net generation, GWh; annual sum of all 12 months"))
            required = [group_values[k] for k in [*codes, "total"]]
            ready = np.isfinite(required).all() and group_values["total"] > 0
            other = group_values["total"] - sum(group_values[k] for k in codes) if ready else np.nan
            if ready and other < -0.02:
                ready = False
            # Published rounding can produce at most 0.006 GWh in six components.
            if ready and -0.02 <= other < 0:
                other = 0.0
            values = {k: group_values[k] for k in codes}
            values["other"] = other
            denominator = sum(values.values()) if ready else np.nan
            for feature, value in values.items():
                allrows.append(dict(country=country, year=year, feature=feature, generation_gwh=value,
                                    share=value / denominator if ready else np.nan,
                                    total_generation_gwh=group_values["total"], status="FULL_WINDOW_VERIFIED" if ready else "BLOCKED_ANNUAL_MIX",
                                    source="Eurostat nrg_cb_pem", source_updated=meta.get("updated", ""),
                                    vintage_use="2026 publication used retrospectively, not real-time 2025 information",
                                    hydro_definition=FROZEN_FEATURES["hydro"], coal_definition=FROZEN_FEATURES["coal"]))
    return pd.DataFrame(allrows), pd.DataFrame(coverage), pd.DataFrame(native)



def load_frozen_weights(catalog):
    """Read complete precision from the registered CSV; never re-fit or round it."""
    with catalog.open_input('v2_weights') as handle:
        source = pd.read_csv(handle, float_precision='round_trip')
    return validate_weight_rows(source)


def validate_weight_rows(source):
    """Freeze the ME pool, all 16 versions, training years and donor identities."""
    rows = source[source.target == 'ME_IT'].copy()
    if (set(rows.version) != set(ALL_VERSIONS) or len(rows) != 32
            or rows.duplicated(['version', 'donor']).any()):
        raise ValueError('FROZEN_WEIGHT_VERSION_SET')
    result = {}
    for version in [*VERSIONS, *[v for v in rows.version.unique() if v not in VERSIONS]]:
        group = rows[rows.version == version]
        if set(group.donor) != set(DONORS):
            raise ValueError('FROZEN_DONOR_POOL')
        group = group.set_index('donor').reindex(DONORS)
        w = group.weight.to_numpy(float)
        if not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1, atol=1e-13):
            raise ValueError('FROZEN_WEIGHT_SIMPLEX')
        allowed_year = 2024 if version.endswith('train2024') else 2025
        if not (group.trained_max_local_year == allowed_year).all():
            raise ValueError('FROZEN_WEIGHT_TRAINING_YEAR')
        if '_loo_' in version and group.loc[version.split('_loo_')[1], 'weight'] != 0:
            raise ValueError('FROZEN_WEIGHT_LOO')
        result[version] = dict(zip(DONORS, map(float, w)))
    return result, rows


def construct_paths(panel, weights):
    """Reconstruct every supplied frozen version, without reading a benchmark.

    Values are positive per donor before weighting. A missing donor makes the
    hour missing even in a leave-one-out version with that donor's zero weight.
    This reproduces synthetic_me.weighted_complete, not dynamic renormalisation.
    Artificial fixtures may supply a single DEMO-named weight version.
    """
    panel = panel[panel.corridor_id.isin(['ME_IT', *DONORS])].copy()
    panel['timestamp_utc'] = pd.to_datetime(panel.timestamp_utc, utc=True)
    if panel.duplicated(['timestamp_utc', 'corridor_id']).any():
        raise ValueError('DUPLICATE_SYNTHETIC_INPUT_KEY')
    if set(panel.corridor_id) != {'ME_IT', *DONORS}:
        raise ValueError('MISSING_SYNTHETIC_CORRIDOR')
    np.testing.assert_allclose(panel[Y], panel.commercial_net_MW.clip(lower=0), equal_nan=True)
    cols = [Y, 'commercial_net_MW', 'physical_net_MW', 'raw_price_spread', 'spread_center', 'spread_c']
    wide = {c: panel.pivot(index='timestamp_utc', columns='corridor_id', values=c).sort_index()
            for c in cols}
    idx = wide[Y].index
    paths = []
    for version, mapping in weights.items():
        if set(mapping) != set(DONORS):
            raise ValueError('FROZEN_DONOR_POOL')
        w = np.asarray([mapping[d] for d in DONORS], float)
        if not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1, atol=1e-13):
            raise ValueError('FROZEN_WEIGHT_SIMPLEX')
        out = pd.DataFrame({'timestamp_utc': idx, 'actual_ME': wide[Y]['ME_IT'].to_numpy()})
        for col in cols:
            out[col] = weighted_complete(wide[col][DONORS], w)
        out['G_ME'] = out.actual_ME - out[Y]
        out['G_ME_commercial_net'] = wide['commercial_net_MW']['ME_IT'].to_numpy() - out.commercial_net_MW
        out['weight_version'] = version
        out['series'] = 'ME_synthetic'
        out['status'] = 'DESCRIPTIVE_ONLY_WEAK_PREFIT_WEAK_SUPPORT'
        out['weight_training_years'] = '2024' if version.endswith('train2024') else '2024;2025'
        out['sample_hash'] = sample_hash(idx)
        out['frozen_sample_hash'] = frozen_sample_hash(idx)
        paths.append(out)
    if not paths:
        raise ValueError('NO_SYNTHETIC_WEIGHT_VERSIONS')
    return pd.concat(paths, ignore_index=True)


def fixed_weight_objective(features, weights, lam):
    """Evaluate the exact historical scaled-feature objective at given weights.

    The row multipliers and accumulation order mirror weight_learning.fit_weights;
    only its optimization call is replaced with direct fixed-vector evaluation.
    """
    if not 0 <= lam <= 1:
        raise ValueError('INVALID_OBJECTIVE_LAMBDA')
    weights = np.asarray(weights, float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1, atol=1e-13):
        raise ValueError('FROZEN_WEIGHT_SIMPLEX')
    rows, details = [], []
    groups = sorted(set(block for _, block, _ in features if block != 'energy'))
    for name, block, values in features:
        count = sum(b == block for _, b, _ in features)
        factor = lam / count if block == 'energy' else (1 - lam) / len(groups) / count
        values = np.asarray(values, float)
        if len(values) != len(weights) + 1 or not np.isfinite(values).all():
            raise ValueError('BLOCKED_MISSING_PREDICTOR')
        if factor > 0:
            rows.append(np.sqrt(factor) * values)
        details.append(dict(feature=name, block=block, objective_row_weight=factor,
                            weighted_squared_discrepancy=float(factor * (values[0] - values[1:] @ weights)**2)))
    if not rows:
        raise ValueError('EMPTY_OBJECTIVE')
    arr = np.asarray(rows)
    return float(np.sum((arr[:, 0] - arr[:, 1:] @ weights)**2)), details


def evaluate_objectives(mix, panel, rows):
    """All 16 frozen objectives, including original leave-one-out predictor pools."""
    wide = panel[panel.corridor_id.isin(['ME_IT', *DONORS])].pivot(index='timestamp_utc', columns='corridor_id', values=Y).sort_index()
    records, features_out = [], []
    for version in ALL_VERSIONS:
        group = rows[rows.version == version].set_index('donor')
        donors = [d for d in DONORS if '_loo_' not in version or d != version.split('_loo_')[1]]
        train, scales, diagnostics, complete = build_predictors(mix, wide, 'ME_IT', donors, [2024])
        if version.endswith('train2024'):
            features = train
        else:
            features, _, diagnostics, complete = build_predictors(mix, wide, 'ME_IT', donors, [2024, 2025], scales)
        weight_vector = group.reindex(donors).weight.to_numpy(float)
        loss, contributions = fixed_weight_objective(features, weight_vector, float(group['lambda'].iloc[0]))
        records.append(dict(weight_version=version, objective=loss, weights_refitted=False,
                            operation='FIXED_WEIGHT_OBJECTIVE_EVALUATION',
                            sample_hash=sample_hash(complete.index), frozen_sample_hash=frozen_sample_hash(complete.index)))
        contributions = {r['feature']: r for r in contributions}
        features_out.extend(dict(weight_version=version, **row,
                                 **{k: v for k, v in contributions.get(row['feature'], {}).items() if k not in row})
                            for row in diagnostics)
    return pd.DataFrame(records), pd.DataFrame(features_out)


def summarize(paths, panel, weights, *, data_hash='UNVERIFIED', code_hash='UNVERIFIED'):
    """Fresh fixed-weight descriptive diagnostics from original synthetic_me code.

    No historical validation statistic is an input. The legacy convention for
    prediction error is synthetic minus actual, opposite to G_ME.
    """
    train_path = paths[paths.weight_version == 'GY050_train2024'].set_index('timestamp_utc')
    if train_path.empty:
        raise ValueError('MISSING_TRAINING_WEIGHT_VERSION')
    local = train_path.index.tz_convert('Europe/Rome')
    common = train_path[['actual_ME', Y]].notna().all(axis=1)
    training = train_path[(local.year == 2024) & common]
    sd = float(training.actual_ME.std(ddof=1))
    training_mean = float(training.actual_ME.mean())
    tl = training.index.tz_convert('Europe/Rome')
    monthhour = training.actual_ME.groupby([tl.month, tl.hour]).mean()
    if len(monthhour) != 288:
        raise ValueError('BLOCKED_TRAINING_MONTH_HOUR')
    wide = panel.pivot(index='timestamp_utc', columns='corridor_id', values=Y).sort_index()
    validation, monthly, support, daily, baselines, zero = [], [], [], [], [], []
    for year in [2024, 2025]:
        z = train_path[(local.year == year) & common]
        y, pred = z.actual_ME.to_numpy(), z[Y].to_numpy()
        if len(y) == 0:
            raise ValueError('BLOCKED_VALIDATION_DATA')
        row = dict(target='ME_IT', method='GY050_train2024', year=year,
                   model_id=f'ME_VALIDATION_GY050_TRAIN2024_EVALUATE_{year}',
                   data_hash=data_hash, code_version_hash=code_hash,
                   outcome=Y, weight_version='GY050_train2024', weather_version='NOT_APPLICABLE',
                   FE='NOT_APPLICABLE', HAC='NOT_APPLICABLE', bandwidth='NOT_APPLICABLE', support='FULL_YEAR',
                   evaluation='TRAINING' if year == 2024 else 'CROSS_YEAR_VALIDATION_PREVIOUSLY_SEEN',
                   training_year=2024, sample_hash=sample_hash(z.index),
                   frozen_sample_hash=frozen_sample_hash(z.index),
                   sample_hash_definition='UTC nanoseconds int64 bytes SHA256',
                   status='DESCRIPTIVE_ONLY_WEAK_PREFIT_WEAK_SUPPORT', **metrics(y, pred, sd))
        validation.append(row)
        xx = wide.loc[z.index, DONORS].to_numpy()
        if not np.isfinite(xx).all():
            raise ValueError('MISSING_SYNTHETIC_DONOR')
        lower, upper = xx.min(axis=1), xx.max(axis=1)
        distance = np.maximum.reduce([lower - y, np.zeros(len(y)), y - upper])
        support.append(dict(target='ME_IT', year=year, n=len(y), sample_hash=sample_hash(z.index),
                            outside_share=float(np.mean(distance > 1e-10)), mean_distance=float(distance.mean()),
                            median_distance=float(np.median(distance)), p90_distance=float(np.quantile(distance, .9)),
                            p95_distance=float(np.quantile(distance, .95)), max_distance=float(distance.max()),
                            convex_lb_rmse=float(np.sqrt(np.mean(distance**2))),
                            interpretation='Lower bound only; not an alternative synthetic series'))
        jointzero = np.all(xx == 0, axis=1)
        if not np.array_equal(pred == 0, jointzero):
            raise ValueError('SYNTHETIC_ZERO_STATE_MISMATCH')
        zero.append(dict(year=year, actual_zero_share=float(np.mean(y == 0)),
                         predicted_zero_share=float(np.mean(pred == 0)),
                         positive_weight_donors_joint_zero_share=float(jointzero.mean()),
                         identity_mismatches=0, all_frozen_weights_strictly_positive=True))
        localidx = z.index.tz_convert('Europe/Rome')
        dd = pd.DataFrame({'actual': y, 'prediction': pred, 'donor_min': lower, 'donor_max': upper}, index=localidx).resample('D').mean()
        dd['gap_prediction_minus_actual'] = dd.prediction - dd.actual
        dd['year'] = year
        dd['date'] = dd.index.strftime('%Y-%m-%d')
        dd['hours'] = pd.Series(1, index=localidx).resample('D').sum()
        if not dd.hours.isin([23, 24, 25]).all():
            raise ValueError('SYNTHETIC_DAILY_COVERAGE')
        daily.extend(dd.reset_index(drop=True).to_dict('records'))
        if year == 2025:
            for month in range(1, 13):
                mask = localidx.month == month
                monthly.append(dict(month=month, year=year, method='GY050_train2024',
                                    model_id=f'ME_VALIDATION_GY050_2025_MONTH_{month:02d}',
                                    sample_hash=sample_hash(z.index[mask]), **metrics(y[mask], pred[mask], sd)))
            forecasts = {'NAIVE_2024_MEAN': np.full(len(y), training_mean),
                         'NAIVE_2024_MONTH_HOUR': np.array([monthhour.loc[(m, h)] for m, h in zip(localidx.month, localidx.hour)]),
                         'GR_alone': xx[:, 0], 'AT_alone': xx[:, 1]}
            for method, values in forecasts.items():
                baselines.append(dict(method=method, training_year=2024, validation_year=2025,
                                      sample_hash=sample_hash(z.index), **metrics(y, values, sd)))
    dailyframe = pd.DataFrame(daily)
    card = dict(target='ME_formal', method='GY050_train2024', training_year=2024, validation_year=2025,
                status='DESCRIPTIVE_ONLY_WEAK_PREFIT_WEAK_SUPPORT',
                donor_1=DONORS[0], donor_1_weight=weights[VERSIONS[0]][DONORS[0]],
                donor_2=DONORS[1], donor_2_weight=weights[VERSIONS[0]][DONORS[1]],
                max_weight=max(weights[VERSIONS[0]].values()), weight_hhi=sum(w*w for w in weights[VERSIONS[0]].values()))
    for year in [2024, 2025]:
        vr = next(r for r in validation if r['year'] == year)
        dd = dailyframe[dailyframe.year == year]
        for c in ['rmse', 'nrmse', 'mae', 'bias']:
            card[f'{c}_{year}'] = vr[c]
        card[f'avg_daily_gap_{year}_mw'] = float(dd.gap_prediction_minus_actual.mean())
        card[f'avg_abs_daily_gap_{year}_mw'] = float(abs(dd.gap_prediction_minus_actual).mean())
        card[f'mean_daily_actual_{year}'] = float(dd.actual.mean())
        card[f'mean_daily_synth_{year}'] = float(dd.prediction.mean())
    naive = next(b for b in baselines if b['method'] == 'NAIVE_2024_MEAN')['rmse']
    card['naive_rmse_2025'] = naive
    card['naive_monthhour_rmse_2025'] = next(b for b in baselines if b['method'] == 'NAIVE_2024_MONTH_HOUR')['rmse']
    card['validation_r2_2025'] = 1 - card['rmse_2025']**2 / naive**2
    card['error_improvement_pct_2025'] = 100*(naive-card['rmse_2025'])/naive
    return clean(dict(status='RUN', model_fit=False, weights_reestimated=False,
                      accepted_target='ME_IT', donors=DONORS, weights=weights,
                      validation_weight_version='GY050_train2024', final_weight_version='GY050_final',
                      validation_previously_seen=True, validation=validation, monthly=monthly,
                      scorecard=card, baselines=baselines, convex_support=support, zero_states=zero, daily=daily,
                      sample_hash_definition='UTC nanosecond integers SHA256; legacy separately recorded as microsecond integers SHA256',
                      G_ME_definition='actual minus final-weight synthetic; opposite sign to validation bias',
                      metadata=dict(outcome='positive part of signed commercial scheduled net MW per donor before weighting',
                                    gap_F2='Requires explicit choice of spread; module does not create an undefined gap price',
                                    evidence_status=['DESCRIPTIVE_ONLY', 'WEAK_PREFIT', 'WEAK_SUPPORT'],
                                    formal_CH_branch='PERMANENTLY_DROPPED_NO_CH_RESULTS_INCLUDED'),
                      historical_refit_validation='NOT_RUN: this implementation evaluates frozen objectives without optimization',
                      optional_presentation_assets=['v23_donor_expansion_validation', 'v23_donor_expansion_weights']))


@dataclass
class SyntheticResult:
    all_versions: pd.DataFrame
    main: pd.DataFrame
    weights: dict
    weight_rows: pd.DataFrame
    payload: dict
    energy_mix: pd.DataFrame
    energy_coverage: pd.DataFrame
    generation_native: pd.DataFrame
    objectives: pd.DataFrame
    objective_features: pd.DataFrame


def build_synthetic(catalog):
    """Authorized reconstruction using only registered F inputs, without writes.

    New diagnostics are returned to the orchestrator. Verification is explicit
    and separate: no synthetic benchmark/result is used as a numerical input.
    """
    weights, rows = load_frozen_weights(catalog)
    with catalog.open_input('v2_weight_feature_freeze') as handle:
        definition = json.load(handle)
    if definition != FROZEN_FEATURES:
        raise ValueError('FROZEN_FEATURE_DEFINITION_CHANGED')
    with catalog.open_input('frozen_v2_panel') as handle:
        panel = pd.read_csv(handle)
    panel['timestamp_utc'] = pd.to_datetime(panel.timestamp_utc, utc=True)
    paths = construct_paths(panel, weights)
    mix, coverage, native = build_energy_mix(catalog)
    objectives, features = evaluate_objectives(mix, panel, rows)
    payload = summarize(paths, panel, weights, data_hash=catalog.assets['frozen_v2_panel']['sha256'],
                        code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    payload['sources'] = [dict(asset_id=aid, sha256=catalog.assets[aid]['sha256'])
                          for aid in ['frozen_v2_panel', 'v2_weights', 'v2_weight_feature_freeze',
                                      *['generation_' + ('EL' if c == 'GR' else c) for c in COUNTRIES]]]
    payload['objective_evaluation'] = clean(objectives.to_dict('records'))
    return SyntheticResult(paths, paths[paths.weight_version.isin(VERSIONS)].copy(), weights, rows,
                           payload, mix, coverage, native, objectives, features)


def validate_frozen(result, catalog):
    """Independent comparison only, never supplies values to reconstruction.

    Tolerances preserve synthetic_me and V3.1 fixed-objective validation.
    Expected timestamps are checked directly before comparing legacy hashes.
    """
    checks = []
    def csv(aid):
        with catalog.open_input(aid, purpose='verify') as handle:
            return pd.read_csv(handle)
    def check(label, left, right, atol=1e-8):
        np.testing.assert_allclose(left, right, rtol=1e-11, atol=atol, equal_nan=True, err_msg=label)
        delta = np.asarray(left, float)-np.asarray(right, float)
        checks.append(dict(check=label, status='PASS', max_absolute_difference=float(np.nanmax(abs(delta))) if delta.size else 0.0))
    old = csv('v2_synthetic_benchmark')
    old = old[old.target == 'ME_IT'].copy()
    old['timestamp_utc'] = pd.to_datetime(old.timestamp_utc, utc=True)
    keys = ['weight_version', 'timestamp_utc']
    left = result.all_versions.sort_values(keys).reset_index(drop=True)
    right = old.sort_values(keys).reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise ValueError('FROZEN_SYNTHETIC_SAMPLE_IDENTITY')
    for col in [Y, 'commercial_net_MW', 'physical_net_MW', 'raw_price_spread', 'spread_center', 'spread_c']:
        if not np.array_equal(left[col].isna(), right[col].isna()):
            raise ValueError('FROZEN_SYNTHETIC_MISSING_MASK')
        check('prediction/' + col, left[col], right[col])
    validation = csv('v21_validation')
    for row in result.payload['validation']:
        frozen = validation[(validation.model == 'GY050_train2024') & (validation.year == row['year']) & (validation.window == 'full_year')].iloc[0]
        for key in ['n', 'rmse', 'mae', 'bias', 'nrmse', 'training_target_sd']:
            check(f'validation/{row["year"]}/{key}', row[key], frozen[key])
        if row['frozen_sample_hash'] != frozen.sample_hash:
            raise ValueError('FROZEN_VALIDATION_SAMPLE_HASH')
    monthly = csv('v21_validation_monthly')
    for row in result.payload['monthly']:
        frozen = monthly[(monthly.model == 'GY050_train2024') & (monthly.year == 2025) & (monthly.month == row['month'])].iloc[0]
        for key in ['n', 'rmse', 'mae', 'bias', 'nrmse']:
            check(f'monthly/{row["month"]}/{key}', row[key], frozen[key])
    envelope = csv('v21_envelope')
    for row in result.payload['convex_support']:
        frozen = envelope[(envelope.year == row['year']) & (envelope.window == 'full_year')].iloc[0]
        check(f'envelope/{row["year"]}/outside', row['outside_share'], frozen.outside_fraction)
        check(f'envelope/{row["year"]}/bound', row['convex_lb_rmse'], frozen.rmse_lower_bound)
    for row in result.payload['baselines']:
        frozen = validation[(validation.model == row['method']) & (validation.year == 2025) & (validation.window == 'full_year')].iloc[0]
        for key in ['rmse', 'mae', 'bias', 'nrmse']:
            check(f'baseline/{row["method"]}/{key}', row[key], frozen[key])
    card = csv('v26_scorecard').query("target=='ME_formal'").iloc[0]
    for key, value in result.payload['scorecard'].items():
        if key in card.index and isinstance(value, (int, float)):
            check('scorecard/' + key, value, card[key])
    for row in result.objectives.itertuples():
        expected = result.weight_rows[result.weight_rows.version == row.weight_version].loss.iloc[0]
        if not abs(row.objective - expected) < 1e-10:
            raise ValueError('FROZEN_OBJECTIVE_MISMATCH:' + row.weight_version)
        checks.append(dict(check='objective/' + row.weight_version, status='PASS', max_absolute_difference=abs(row.objective - expected)))
    predictors = csv('v2_weight_predictors')
    predictors = predictors[predictors.target == 'ME_IT'].copy()
    for version, years in [('GY050_train2024', '2024'), ('GY050_final', '2024;2025')]:
        current = result.objective_features[result.objective_features.weight_version == version].set_index('feature').sort_index()
        expected = predictors[predictors.training_years == years].set_index('feature').sort_index()
        if not current.index.equals(expected.index) or not current.omitted.equals(expected.omitted):
            raise ValueError('FROZEN_FEATURE_IDENTITY')
        for col in ['scale', 'target_value', 'donor_min', 'donor_max']:
            check('feature/' + version + '/' + col, current[col], expected[col], atol=1e-10)
    return checks


def validate_saved(directory, catalog):
    """Verify saved synthetic artifacts without reconstructing or fitting them.

    The pipeline verifies run-receipt hashes before this function. Frozen source
    weights are re-read independently here, so a new output cannot become its
    own expected objective/weight reference. All paths are explicit and local.
    """
    directory = Path(directory)
    def source(name):
        path = directory / name
        if any(p.is_symlink() for p in [path, *path.parents]):
            raise ValueError('SYNTHETIC_OUTPUT_SYMLINK')
        if not path.is_file():
            raise ValueError('SYNTHETIC_SAVED_ARTIFACT_MISSING:' + name)
        return path
    paths = pd.read_parquet(source('all_versions.parquet'))
    main = pd.read_parquet(source('main.parquet'))
    rows = pd.read_csv(source('weights.csv'), float_precision='round_trip')
    weights, rows = validate_weight_rows(rows)
    with catalog.open_input('v2_weights', purpose='verify') as handle:
        _, frozen_rows = validate_weight_rows(pd.read_csv(handle, float_precision='round_trip'))
    keys = ['version', 'donor']
    saved = rows.sort_values(keys).reset_index(drop=True)
    frozen = frozen_rows.sort_values(keys).reset_index(drop=True)
    # The all-null ME reason column is inferred as float after CSV serialization,
    # while the complete frozen CSV also contains CH string reasons. Check exact
    # values and missingness; a dtype-only inference difference is not a redesign.
    if (list(saved.columns) != list(frozen.columns) or saved.shape != frozen.shape
            or not saved.isna().equals(frozen.isna())
            or any(saved.loc[saved[c].notna(), c].tolist() != frozen.loc[frozen[c].notna(), c].tolist() for c in saved)):
        raise ValueError('FROZEN_SAVED_WEIGHT_IDENTITY')
    expected_main = paths[paths.weight_version.isin(VERSIONS)]
    path_keys = ['weight_version', 'timestamp_utc']
    if not main.sort_values(path_keys).reset_index(drop=True).equals(expected_main.sort_values(path_keys).reset_index(drop=True)):
        raise ValueError('SYNTHETIC_MAIN_VERSION_PROJECTION_CHANGED')
    payload = json.loads(source('summary.json').read_text())
    if payload.get('weights_reestimated') is not False or payload.get('weights') != weights:
        raise ValueError('SYNTHETIC_SUMMARY_WEIGHT_IDENTITY')
    result = SyntheticResult(paths, main, weights, frozen_rows, payload,
        pd.read_csv(source('energy_mix.csv')), pd.DataFrame(), pd.DataFrame(),
        pd.DataFrame(payload['objective_evaluation']), pd.read_csv(source('objective_features.csv')))
    if set(result.objectives.weight_version) != set(ALL_VERSIONS) or len(result.objectives) != 16:
        raise ValueError('FROZEN_OBJECTIVE_VERSION_SET')
    return validate_frozen(result, catalog)
