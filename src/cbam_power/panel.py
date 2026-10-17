"""V3 panel construction without legacy imports or expected-result inputs.

Sources: V3/data_processing.{reconstruct_core,build_panel}, V3/variables.add_variables,
V3/reused/native_data.{old_panel,main}, V3/reused/design.calendar. AT/CH/ME inherit
the frozen V1 values; SI/GR are parsed from the registered frozen market archives.
Weather is an explicit already reconstructed stage input. This module never fits.
"""
import numpy as np
import pandas as pd

from .market import MarketReader, PARTNERS, ZONES, TZ

ACTUALS = tuple(PARTNERS)
WEATHER_FIELDS = ("temp_c", "wind_ms", "ssrd_processed", "tp_processed")
CORE_COLUMNS = ["timestamp_utc", "timestamp_local", "corridor_id", "scheduled_export_proxy",
                "commercial_net_MW", "physical_net_MW", "physical_export_proxy", "commercial_net_MWh",
                "origin_price", "destination_price", "raw_price_spread", "spread_center", "spread_c",
                "year", "month", "hour", "weekday", "italian_zone", "destination_bidding_zone",
                "origin_bidding_zone", "spread_center_source", "data_status"]
POLICY_CUTOFF = "2026-01-01T00:00:00+01:00"
SOURCE_FUNCTIONS = {
    "reconstruct_core": "V3/src/data_processing.py::reconstruct_core; reused/native_data.py::main",
    "assemble_panel": "V3/src/variables.py::add_variables; reused/design.py::calendar; data_processing.py::build_panel",
    "build_panel": "V3/src/data_processing.py::build_panel",
}


def _timestamps(values):
    # An explicit UTC offset is required; naive input is not silently assigned UTC.
    t = pd.DatetimeIndex(pd.to_datetime(values))
    if t.tz is None:
        raise ValueError("Panel timestamps require explicit UTC/timezone")
    t = t.tz_convert("UTC")
    if not t.equals(t.floor("h")):
        raise ValueError("Panel must use hourly UTC delivery starts")
    return t


def _metadata(d, cid, grid):
    d = d.copy()
    local = grid.tz_convert(TZ)
    d["corridor_id"] = cid
    d["timestamp_utc"] = grid
    d["timestamp_local"] = [t.isoformat() for t in local]
    for var, value in [("year", local.year), ("month", local.month), ("hour", local.hour),
                       ("weekday", local.weekday)]:
        d[var] = value
    d["italian_zone"] = ZONES[cid]
    d["data_status"] = "DESCRIPTIVE_ONLY"
    d["commercial_net_MWh"] = d.commercial_net_MW
    d["destination_bidding_zone"] = ZONES[cid]
    d["origin_bidding_zone"] = cid[:2]
    d["spread_center_source"] = ("V1 fixed pre2026 value preserved" if cid in ("AT_IT", "CH_IT", "ME_IT")
                                 else "fixed 2024–2025 observed hourly native spread mean")
    return d.reset_index(drop=True)[CORE_COLUMNS]


def reconstruct_core(catalog, reader=None):
    """Reconstruct frozen 22-column variable contract from F inputs only.

    Old reconstruct_core filled metadata columns from the V2 reference. Here the
    same explicit native_data.main definitions produce those columns; no V3
    expected weather/flow/price/parameter value enters construction.
    """
    with catalog.open_input("frozen_v1_panel", purpose="process") as handle:
        old = pd.read_csv(handle)
    old["timestamp_utc"] = pd.to_datetime(old.timestamp_utc, utc=True)
    old = old.rename(columns={"it_price": "destination_price",
                              "scheduled_net_origin_to_it": "commercial_net_MW",
                              "physical_net_origin_to_it": "physical_net_MW",
                              "spread_center_pre2026": "spread_center"})
    if old.duplicated(["corridor_id", "timestamp_utc"]).any():
        raise ValueError("Duplicate frozen V1 corridor/hour")
    grid = pd.DatetimeIndex(sorted(old.timestamp_utc.unique()))
    if not grid.equals(pd.date_range(grid.min(), grid.max(), freq="h")):
        raise ValueError("Frozen V1 master grid has holes")
    # The schema is pinned in the manifest, never inferred from expected V3 data.
    schema = catalog.manifest["schemas"][catalog.assets["frozen_v2_panel"]["schema_ref"]]
    if schema["columns"] != CORE_COLUMNS:
        raise ValueError("Frozen V2 column contract changed")
    reader = reader or MarketReader(catalog)
    flows = reader.build_flows()
    prices = reader.gme_prices()
    frames, reconciliation = [], []
    for cid in ACTUALS:
        if cid in ("AT_IT", "CH_IT", "ME_IT"):
            d = old[old.corridor_id.eq(cid)].set_index("timestamp_utc").reindex(grid).copy()
            if not np.isclose(d.scheduled_export_proxy, d.commercial_net_MW.clip(lower=0), equal_nan=True).all():
                raise ValueError("Frozen V1 outcome differs from positive hourly net")
        else:
            d = pd.DataFrame(index=grid)
            for column in ["commercial_net_MW", "physical_net_MW"]:
                d[column] = flows[cid][column].reindex(grid)
            d["origin_price"] = reader.new_prices(cid).reindex(grid)
            d["destination_price"] = prices[ZONES[cid].replace("IT-", "")].reindex(grid)
            d["raw_price_spread"] = d.destination_price - d.origin_price
            d["scheduled_export_proxy"] = d.commercial_net_MW.clip(lower=0)
            d["physical_export_proxy"] = d.physical_net_MW.clip(lower=0)
            pre = grid.tz_convert(TZ).year < 2026
            d["spread_center"] = float(d.loc[pre, "raw_price_spread"].mean())
            d["spread_c"] = d.raw_price_spread - d.spread_center
        for variable in ["commercial_net_MW", "physical_net_MW"]:
            delta = d[variable] - flows[cid][variable].reindex(grid)
            reconciliation.append(dict(corridor_id=cid, variable=variable,
                maximum_abs_difference=float(delta.abs().max()),
                inherited_V1=cid in ("AT_IT", "CH_IT", "ME_IT"),
                mirror_overwrites_original=False))
        frames.append(_metadata(d, cid, grid))
    result = pd.concat(frames, ignore_index=True)
    result.attrs["lineage"] = {"inherited_v1_corridors": ["AT_IT", "CH_IT", "ME_IT"],
                              "native_reparsed_corridors": ["SI_IT", "GR_IT"],
                              "v1_asset": "frozen_v1_panel", "schema_asset": "frozen_v2_panel"}
    result.attrs["mirror_reconciliation"] = reconciliation
    result.attrs["market_sources"] = reader.diagnostics["sources"]
    result.attrs["market_diagnostic_counts"] = {key: len(value) for key, value in reader.diagnostics.items()}
    return result


def assemble_panel(core, weather, cutoff=POLICY_CUTOFF, *, strict_corridors=True):
    """Apply frozen variables/calendar and bilateral merge, with no fitting.

    Artificial/demo minimum core columns: timestamp_utc, corridor_id,
    commercial_net_MW, physical_net_MW, origin_price, destination_price,
    spread_center. Additional frozen variable columns, if supplied, are preserved.
    This function never estimates a new spread center or fills absent weather.
    """
    if weather is None:
        raise ValueError("Missing completed weather stage")
    required = {"timestamp_utc", "corridor_id", "commercial_net_MW", "physical_net_MW",
                "origin_price", "destination_price", "spread_center"}
    if not required.issubset(core.columns):
        raise ValueError("Missing core panel columns: " + ", ".join(sorted(required - set(core.columns))))
    d = core.copy()
    lineage = dict(core.attrs)
    d["timestamp_utc"] = _timestamps(d.timestamp_utc)
    if d.duplicated(["corridor_id", "timestamp_utc"]).any():
        raise ValueError("Duplicate corridor x UTC delivery hour")
    actuals = set(d.corridor_id.unique())
    if not actuals.issubset(ACTUALS) or (strict_corridors and actuals != set(ACTUALS)):
        raise ValueError("Formal actual corridor set must be AT/CH/SI/GR/ME; FR excluded")
    local = d.timestamp_utc.dt.tz_convert(TZ)
    defaults = {
        "timestamp_local": local.map(lambda t: t.isoformat()),
        "scheduled_export_proxy": d.commercial_net_MW.clip(lower=0),
        "physical_export_proxy": d.physical_net_MW.clip(lower=0),
        "commercial_net_MWh": d.commercial_net_MW,
        "raw_price_spread": d.destination_price - d.origin_price,
        "spread_c": d.destination_price - d.origin_price - d.spread_center,
        "italian_zone": d.corridor_id.map(ZONES),
        "destination_bidding_zone": d.corridor_id.map(ZONES),
        "origin_bidding_zone": d.corridor_id.str[:2],
        "spread_center_source": "explicit input center",
        "data_status": "DESCRIPTIVE_ONLY",
    }
    for column, values in defaults.items():
        if column not in d:
            d[column] = values
    for column in ["year", "month", "hour", "weekday"]:
        d[column] = getattr(local.dt, column)
    d = d[CORE_COLUMNS]
    for column in ["day", "week"]:
        d[column] = local.dt.day if column == "day" else local.dt.isocalendar().week.astype(int)
    cut = pd.Timestamp(cutoff)
    if cut.tzinfo is None:
        raise ValueError("Policy cutoff must have explicit timezone")
    cut = cut.tz_convert("UTC")
    d["N_commercial"] = d.commercial_net_MW
    d["Y_positive"] = d.N_commercial.clip(lower=0)
    d["Y_signed"] = d.N_commercial
    d["S_native"] = d.destination_price - d.origin_price
    d["Post"] = d.timestamp_utc.ge(cut).astype(int)
    d["r_days"] = (d.timestamp_utc + pd.Timedelta(minutes=30) - cut).dt.total_seconds() / 86400
    d["delivery_interval_minutes"] = 60
    missing_weather_columns = {"timestamp_utc", "geography", *WEATHER_FIELDS} - set(weather.columns)
    if missing_weather_columns:
        raise ValueError("Incomplete weather schema: " + ", ".join(sorted(missing_weather_columns)))
    w = weather[["timestamp_utc", "geography", *WEATHER_FIELDS]].copy()
    w["timestamp_utc"] = _timestamps(w.timestamp_utc)
    if w.duplicated(["timestamp_utc", "geography"]).any():
        raise ValueError("Duplicate weather geography/hour")
    if not set(w.geography).issubset({*ZONES.values(), *[cid[:2] for cid in ACTUALS]}):
        raise ValueError("Unexpected weather geography")
    for side, column in [("origin", "origin_bidding_zone"), ("destination", "destination_bidding_zone")]:
        renamed = w.rename(columns={"geography": column, **{c: f"{side}_weather_{c}" for c in WEATHER_FIELDS}})
        d = d.merge(renamed, on=["timestamp_utc", column], how="left", validate="many_to_one")
    d = d.sort_values(["corridor_id", "timestamp_utc"]).reset_index(drop=True)
    columns = [f"{side}_weather_{c}" for side in ("origin", "destination") for c in WEATHER_FIELDS]
    d["weather_complete"] = d[columns].notna().all(axis=1)
    if not np.array_equal(d.Y_positive, d.scheduled_export_proxy, equal_nan=True):
        raise ValueError("Positive export proxy identity changed")
    if not np.allclose(d.S_native, d.raw_price_spread, atol=1e-10, equal_nan=True):
        raise ValueError("Native price spread identity changed")
    if not d.destination_bidding_zone.eq(d.corridor_id.map(ZONES)).all():
        raise ValueError("Frozen bidding-zone mapping changed")
    for column in ["N_commercial", "Y_positive", "Y_signed", "S_native", "origin_price", "destination_price"]:
        if d[column].isna().any():
            raise ValueError("Missing formal core variable (never imputed): " + column)
    d.attrs.update(lineage)
    d.attrs["weather_status"] = "LOCAL_AVAILABLE_ANNUAL_COVERAGE_LIMITATION_PRESERVED"
    return d


def build_panel(catalog, weather, config=None):
    """Authorized entry point; caller must supply a valid weather stage artifact."""
    core = reconstruct_core(catalog)
    cutoff = POLICY_CUTOFF if config is None else config["time"]["policy_cutoff_local"]
    panel = assemble_panel(core, weather, cutoff)
    if config is not None:
        expected = pd.date_range(config["time"]["window_start_inclusive"],
                                 config["time"]["window_end_exclusive"], freq="h", inclusive="left")
        for cid, group in panel.groupby("corridor_id"):
            if not pd.DatetimeIndex(group.timestamp_utc).equals(expected):
                raise ValueError("Frozen UTC grid mismatch: " + cid)
    return panel
