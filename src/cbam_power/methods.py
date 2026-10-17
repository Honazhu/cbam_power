"""Explicit, in-memory adapters for a limited reusable-methods prototype.

EXPERIMENTAL REUSABLE API — ARTIFICIAL TESTS ONLY; OUTSIDE PR5/PR7 CERTIFICATION.
The authorized V3/V3.2 workflow and its frozen configuration do not call this API.
No data acquisition, input catalog, output directory, optimizer or model scheduler
is used here. Countries, calendar, policy and units are caller declarations.

Delegates: market.integrate_native; estimators.fit/hac/contrast;
weather.weighted_fields; synthetic.weighted_complete/fixed_weight_objective.
The new local-design adapter makes the existing midpoint/triangular construction
explicit. It does not generalize the coupled seasonal or polynomial schedulers.
"""
from dataclasses import dataclass
from numbers import Integral
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from . import estimators, market

PUBLIC_API = (
    "Corridor", "RDiTSpec", "validate_inputs", "aggregate_flows",
    "construct_rdit", "fit_rdit", "infer_hac", "aggregate_era5",
    "predict_synthetic", "evaluate_fixed_objective",
)
CERTIFICATION = "EXPERIMENTAL REUSABLE API — ARTIFICIAL TESTS ONLY; OUTSIDE PR5/PR7 CERTIFICATION"


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _identifier(value):
    _require(isinstance(value, str) and bool(value.strip()) and value == value.strip(),
             "INVALID_IDENTIFIER")
    _require(not any(ord(c) < 32 for c in value), "INVALID_IDENTIFIER")
    return value


def _integer(value, minimum, code):
    _require(isinstance(value, Integral) and not isinstance(value, bool) and value >= minimum, code)
    return int(value)


def _times(values, frequency_minutes):
    _integer(frequency_minutes, 1, "INVALID_FREQUENCY")
    try:
        t = pd.DatetimeIndex(values)
    except (ValueError, TypeError) as exc:
        raise ValueError("INVALID_TIMESTAMPS") from exc
    _require(len(t) > 0 and not t.hasnans and t.tz is not None, "EXPLICIT_AWARE_TIMESTAMPS_REQUIRED")
    t = t.tz_convert("UTC")
    _require(not t.has_duplicates and t.is_monotonic_increasing, "SORTED_UNIQUE_TIMESTAMPS_REQUIRED")
    ns = t.as_unit("ns").asi8
    _require(np.all(ns % (int(frequency_minutes) * 60 * 10**9) == 0), "INVALID_DELIVERY_GRID")
    return t


@dataclass(frozen=True, kw_only=True)
class Corridor:
    """Caller-defined directed corridor; positive means origin to destination."""
    id: str
    origin: str
    destination: str

    def __post_init__(self):
        for value in (self.id, self.origin, self.destination):
            _identifier(value)
        _require(self.origin != self.destination, "IDENTICAL_CORRIDOR_ENDPOINTS")


@dataclass(frozen=True, kw_only=True)
class RDiTSpec:
    """Explicit local-linear settings, independent of the frozen V3 model grid.

    Only triangular weighting and optional hour/weekday FE are supported. The
    inherited estimator requires at least 100 complete observations, full rank,
    observed-level FE references and normal 1.96-based inference. These method
    limits are retained, not exposed as silently adjustable research settings.
    """
    model_id: str
    corridor: Corridor
    outcome: str
    calendar_timezone: str
    cutoff: str
    bandwidth_days: float
    kernel: str
    fixed_effects: tuple[str, ...]
    hac_hours: int
    midpoint_minutes: float
    donut_days: float
    minimum_side: int

    def __post_init__(self):
        _identifier(self.model_id)
        _identifier(self.outcome)
        _require(isinstance(self.corridor, Corridor), "EXPLICIT_CORRIDOR_REQUIRED")
        _require(self.outcome not in {"timestamp_utc", "corridor_id", "post", "r", "post_r",
                                      "hour", "weekday", "triangular"}, "RESERVED_OUTCOME_NAME")
        try:
            ZoneInfo(self.calendar_timezone)
            cut = pd.Timestamp(self.cutoff)
        except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError("INVALID_CALENDAR_OR_CUTOFF") from exc
        _require(not pd.isna(cut) and cut.tzinfo is not None, "EXPLICIT_AWARE_CUTOFF_REQUIRED")
        _require(cut.tz_convert("UTC").value % (3600 * 10**9) == 0, "CUTOFF_MUST_BE_HOURLY")
        _require(np.isfinite(self.bandwidth_days) and self.bandwidth_days > 0, "INVALID_BANDWIDTH")
        _require(self.kernel == "triangular", "UNSUPPORTED_KERNEL")
        _require(isinstance(self.fixed_effects, tuple) and len(set(self.fixed_effects)) == len(self.fixed_effects)
                 and set(self.fixed_effects) <= {"hour", "weekday"}, "UNSUPPORTED_FIXED_EFFECTS")
        _integer(self.hac_hours, 0, "INVALID_HAC_LAG")
        _integer(self.minimum_side, 0, "INVALID_MINIMUM_SIDE")
        _require(np.isfinite(self.midpoint_minutes) and 0 <= self.midpoint_minutes < 60,
                 "INVALID_MIDPOINT")
        _require(np.isfinite(self.donut_days) and 0 <= self.donut_days < self.bandwidth_days,
                 "INVALID_DONUT")


def validate_inputs(frame, *, timestamp, numeric_columns, nullable_columns,
                    frequency_minutes, entity_column, entity_id):
    """Validate an in-memory table without filling, sorting or dropping rows.

    Missing intervals are permitted; duplicate or off-grid intervals are not.
    Supply both entity arguments as None for an explicitly ungrouped table.
    Numeric missingness must be declared column by column. This checks structure,
    not provider identity, permissions or completeness of an empirical dataset.
    """
    _require(isinstance(frame, pd.DataFrame) and frame.columns.is_unique, "INVALID_TABLE_SCHEMA")
    columns = tuple(numeric_columns)
    nullable = set(nullable_columns)
    _require(len(set(columns)) == len(columns) and nullable <= set(columns), "INVALID_COLUMN_CONTRACT")
    _require(timestamp in frame and all(c in frame for c in columns), "MISSING_REQUIRED_COLUMN")
    t = _times(frame[timestamp], frequency_minutes)
    _require((entity_column is None) == (entity_id is None), "INCOMPLETE_ENTITY_CONTRACT")
    if entity_column is not None:
        _identifier(entity_id)
        _require(entity_column in frame and frame[entity_column].notna().all()
                 and frame[entity_column].eq(entity_id).all(), "ENTITY_MISMATCH")
    for name in columns:
        _require(pd.api.types.is_numeric_dtype(frame[name])
                 and not pd.api.types.is_bool_dtype(frame[name]), "NONNUMERIC_COLUMN")
        arr = frame[name].to_numpy(dtype=float, na_value=np.nan)
        _require(not np.isinf(arr).any(), "INFINITE_INPUT")
        _require(name in nullable or not np.isnan(arr).any(), "UNDECLARED_MISSING_INPUT")
    return {"rows": len(frame), "timezone": "UTC", "frequency_minutes": int(frequency_minutes),
            "missing": {name: int(frame[name].isna().sum()) for name in columns},
            "missing_intervals": int((t[-1] - t[0]).total_seconds() / (60 * frequency_minutes)) + 1 - len(t)}


def aggregate_flows(timestamps, values, *, corridor, input_unit, positive_direction,
                    duration_minutes):
    """Duration-integrate signed MW/GW; take positive part AFTER hourly netting.

    Supply native duration(s) explicitly. No country-dependent frequency guessing
    occurs. Incomplete hours remain missing; no interval is extended over a gap.
    """
    _require(isinstance(corridor, Corridor), "EXPLICIT_CORRIDOR_REQUIRED")
    _require(input_unit in {"MW", "GW"}, "UNSUPPORTED_POWER_UNIT")
    _require(positive_direction in {"origin_to_destination", "destination_to_origin"},
             "EXPLICIT_FLOW_DIRECTION_REQUIRED")
    t = _times(timestamps, 15)
    values = np.asarray(values, dtype=float)
    _require(values.shape == (len(t),) and not np.isinf(values).any(), "INVALID_NATIVE_VALUES")
    scale = {"MW": 1.0, "GW": 1000.0}[input_unit]
    sign = 1.0 if positive_direction == "origin_to_destination" else -1.0
    hourly = market.integrate_native(t, values * scale * sign, duration_minutes, "signed_net_MW")
    result = hourly.reset_index().drop(columns=["weighted"])
    result = result.rename(columns={"integrated_MWh_if_power": "integrated_MWh"})
    result["positive_export_MW"] = result.signed_net_MW.clip(lower=0)
    result["corridor_id"] = corridor.id
    result["origin"] = corridor.origin
    result["destination"] = corridor.destination
    return result


def construct_rdit(frame, *, timestamp, spec):
    """Explicit local-design adapter; no historical calendar helper is invoked.

    UTC delivery-start defines Post. The explicitly supplied midpoint defines
    elapsed days and the open bandwidth window; missing outcomes stay missing
    until the unchanged estimator applies its declared complete-case rule.
    """
    _require(isinstance(spec, RDiTSpec), "EXPLICIT_RDIT_SPEC_REQUIRED")
    validate_inputs(frame, timestamp=timestamp, numeric_columns=(spec.outcome,),
                    nullable_columns=(spec.outcome,), frequency_minutes=60,
                    entity_column="corridor_id", entity_id=spec.corridor.id)
    d = frame[[timestamp, "corridor_id", spec.outcome]].copy()
    d = d.rename(columns={timestamp: "timestamp_utc"})
    t = _times(d.timestamp_utc, 60)
    d["timestamp_utc"] = t
    local = t.tz_convert(spec.calendar_timezone)
    d["hour"] = local.hour
    d["weekday"] = local.weekday
    cut = pd.Timestamp(spec.cutoff).tz_convert("UTC")
    d["r"] = (t + pd.Timedelta(minutes=spec.midpoint_minutes) - cut).total_seconds() / 86400
    d["post"] = (t >= cut).astype(float)
    d["post_r"] = d.post * d.r
    d = d.loc[d.r.abs().lt(spec.bandwidth_days) & d.r.abs().ge(spec.donut_days)].copy()
    d["triangular"] = 1.0 - d.r.abs() / spec.bandwidth_days
    return d


def fit_rdit(frame, *, timestamp, spec):
    """Delegate WLS, FE, rank and sample rules to the unchanged estimator."""
    design = construct_rdit(frame, timestamp=timestamp, spec=spec)
    fitted = estimators.fit(design, spec.outcome, ["post", "r", "post_r"],
                            fe=spec.fixed_effects, weight="triangular", minimum_side=spec.minimum_side)
    fitted.audit["prototype_spec"] = {"model_id": spec.model_id, "corridor_id": spec.corridor.id,
        "origin": spec.corridor.origin, "destination": spec.corridor.destination,
        "calendar_timezone": spec.calendar_timezone, "cutoff": pd.Timestamp(spec.cutoff).isoformat(),
        "bandwidth_days": spec.bandwidth_days, "kernel": spec.kernel, "hac_hours": spec.hac_hours,
        "midpoint_minutes": spec.midpoint_minutes, "donut_days": spec.donut_days,
        "certification": CERTIFICATION}
    return fitted


def infer_hac(fitted, *, lag_hours, contrasts):
    """Elapsed-hour Bartlett HAC with inherited n/(n-k) correction and contrasts.

    No missing-hour compression. Contrasts are explicit dictionaries of fitted
    coefficient names to scalars. Normal CI/p-value rules are inherited verbatim.
    """
    _require(isinstance(fitted, estimators.Fit), "FIT_REQUIRED")
    _integer(lag_hours, 0, "INVALID_HAC_LAG")
    declared = fitted.audit.get("prototype_spec", {}).get("hac_hours", lag_hours)
    _require(declared == lag_hours, "HAC_SPEC_MISMATCH")
    _require(isinstance(contrasts, dict) and bool(contrasts), "EXPLICIT_CONTRASTS_REQUIRED")
    _times(fitted.sample.timestamp_utc, 60)
    for label, terms in contrasts.items():
        _identifier(label)
        _require(isinstance(terms, dict) and bool(terms) and set(terms) <= set(fitted.names)
                 and np.isfinite(list(terms.values())).all(), "INVALID_CONTRAST")
    covariance = fitted.covariance(int(lag_hours))
    return {"covariance": covariance, "coefficient_order": tuple(fitted.names),
            "contrasts": {label: fitted.contrast(covariance, terms) for label, terms in contrasts.items()},
            "lag_hours": int(lag_hours), "certification": CERTIFICATION}


def aggregate_era5(timestamps, grids, *, region_id, cell_ids, area_weights,
                   raw_units, accumulated_shift_hours):
    """Apply frozen ERA5 transformations to caller-selected cells and weights.

    This is aggregation only, not polygon construction, raw decoding or revision
    selection. Cell IDs must be unique and match the weight map exactly. Only
    t2m/u10/v10/ssrd/tp are supported. The declared hourly accumulation alignment
    must be -1; instantaneous variables remain at validity time. No renormalizing
    around missing cells and no interpolation. Output uses the union of timestamps.
    """
    from .weather import weighted_fields
    _identifier(region_id)
    t = _times(timestamps, 60)
    ids = tuple(cell_ids)
    _require(bool(ids) and len(set(ids)) == len(ids), "DUPLICATE_OR_EMPTY_GRID_CELLS")
    for cell in ids:
        _identifier(cell)
    _require(isinstance(area_weights, dict) and set(area_weights) == set(ids), "MASK_CELL_MISMATCH")
    w = np.array([area_weights[cell] for cell in ids], dtype=float)
    _require(np.isfinite(w).all() and (w > 0).all() and np.isclose(w.sum(), 1, atol=1e-12, rtol=0),
             "INVALID_AREA_WEIGHTS")
    units = {"t2m": "K", "u10": "m s-1", "v10": "m s-1", "ssrd": "J m-2", "tp": "m"}
    _require(isinstance(grids, dict) and bool(grids) and set(grids) <= set(units), "UNKNOWN_ERA5_VARIABLE")
    _require(isinstance(raw_units, dict) and set(raw_units) == set(grids)
             and all(raw_units[k] == units[k] for k in grids), "ERA5_UNIT_MISMATCH")
    _require(accumulated_shift_hours == -1 and not isinstance(accumulated_shift_hours, bool),
             "UNSUPPORTED_ACCUMULATED_ALIGNMENT")
    arrays = {key: np.asarray(value, float) for key, value in grids.items()}
    _require(all(a.shape == (len(t), len(ids)) and not np.isinf(a).any() for a in arrays.values()),
             "INVALID_GRID_SHAPE_OR_VALUE")
    semantics = {"variables": {
        "t2m": {"output": "temp_c", "scale": 1, "offset": -273.15, "delivery_start_offset_hours": 0},
        "ssrd": {"output": "ssrd_processed", "scale": 1/3600, "offset": 0,
                 "delivery_start_offset_hours": accumulated_shift_hours},
        "tp": {"output": "tp_processed", "scale": 1000, "offset": 0,
               "delivery_start_offset_hours": accumulated_shift_hours}}}
    fields = weighted_fields(arrays, w, semantics)
    result = pd.concat([pd.Series(values, index=t + pd.Timedelta(hours=shift), name=name)
                        for name, (values, shift) in fields.items()], axis=1, sort=True).sort_index()
    result.index.name = "timestamp_utc"
    result = result.reset_index()
    result["geography"] = region_id
    return result


def _donors(target_id, donor_ids, weights):
    _identifier(target_id)
    ids = tuple(donor_ids)
    _require(bool(ids) and len(set(ids)) == len(ids) and target_id not in ids, "INVALID_DONOR_POOL")
    for donor in ids:
        _identifier(donor)
    _require(isinstance(weights, dict) and set(weights) == set(ids), "WEIGHT_DONOR_MISMATCH")
    w = np.array([weights[d] for d in ids], dtype=float)
    _require(np.isfinite(w).all() and (w >= 0).all() and np.isclose(w.sum(), 1, atol=1e-13, rtol=0),
             "INVALID_WEIGHT_SIMPLEX")
    return ids, w


def predict_synthetic(frame, *, timestamp, target_id, donor_ids, weights, input_kind):
    """Fixed-weight positive-export prediction; gap = actual minus synthetic.

    input_kind is signed_net_MW (take each positive part) or positive_export_MW
    (reject negative values). No donor optimization, hourly renormalization or
    capacity clipping. Missing any declared donor makes prediction missing even
    when that donor's fixed weight is zero. Target absence preserves donor output.
    """
    from .synthetic import weighted_complete
    ids, w = _donors(target_id, donor_ids, weights)
    _require(input_kind in {"signed_net_MW", "positive_export_MW"}, "UNSUPPORTED_SYNTHETIC_INPUT")
    columns = (target_id, *ids)
    validate_inputs(frame, timestamp=timestamp, numeric_columns=columns, nullable_columns=columns,
                    frequency_minutes=60, entity_column=None, entity_id=None)
    data = frame[list(columns)].to_numpy(float)
    if input_kind == "signed_net_MW":
        data = np.maximum(data, 0)
    else:
        _require(not np.any(data < 0), "NEGATIVE_POSITIVE_EXPORT")
    predicted = weighted_complete(data[:, 1:], w)
    return pd.DataFrame({"timestamp_utc": _times(frame[timestamp], 60), "target_id": target_id,
                         "actual_MW": data[:, 0], "synthetic_MW": predicted,
                         "gap_MW": data[:, 0] - predicted})


def evaluate_fixed_objective(features, *, target_id, donor_ids, weights, balance):
    """Evaluate supplied, ALREADY SCALED features; never fit or choose weights.

    features columns: feature, block, target_id, each donor ID. 'energy' identifies
    the energy block; other explicit block labels share the outcome allocation.
    Feature scaling, training-window selection and frozen version management are
    caller responsibilities and are not claimed as generalized by this adapter.
    """
    from .synthetic import fixed_weight_objective
    ids, w = _donors(target_id, donor_ids, weights)
    _require(isinstance(features, pd.DataFrame) and features.columns.is_unique
             and {"feature", "block", target_id, *ids} <= set(features.columns), "INVALID_FEATURE_SCHEMA")
    _require(len(features) > 0 and not features.feature.duplicated().any(), "DUPLICATE_OR_EMPTY_FEATURES")
    for value in (*features.feature, *features.block):
        _identifier(value)
    _require(np.isfinite(balance) and 0 <= balance <= 1, "INVALID_OBJECTIVE_BALANCE")
    rows = [(row.feature, row.block, np.array([row[target_id], *[row[d] for d in ids]], dtype=float))
            for _, row in features.iterrows()]
    return fixed_weight_objective(rows, w, balance)
