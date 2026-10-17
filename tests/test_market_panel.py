"""Artificial market/panel tests; no private files or network required."""
from contextlib import contextmanager
from io import BytesIO
import json
from unittest.mock import patch
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from cbam_power.market import (MarketReader, PARTNERS, ZONES, gme_selection, henex_selection,
                               infer_minutes, integrate_native, parse_gme_archive, parse_henex_bytes)
from cbam_power.panel import CORE_COLUMNS, assemble_panel, reconstruct_core

pytestmark = pytest.mark.public


def quarter_grid(start="2025-01-01T00:00Z", n=4):
    return pd.date_range(start, periods=n, freq="15min")


def fixture_panel(times=None):
    times = pd.date_range("2025-12-31T22:00Z", periods=3, freq="h") if times is None else times
    frames = [pd.DataFrame(dict(timestamp_utc=times, corridor_id=cid,
                   commercial_net_MW=np.arange(len(times)) * 10 - 10,
                   physical_net_MW=np.arange(len(times)) * 12 - 12,
                   origin_price=40., destination_price=55., spread_center=7.25)) for cid in PARTNERS]
    weather = pd.concat([pd.DataFrame(dict(timestamp_utc=times, geography=geo,
                          temp_c=5., wind_ms=4., ssrd_processed=100., tp_processed=.1))
                         for geo in sorted({*ZONES.values(), *[cid[:2] for cid in PARTNERS]})], ignore_index=True)
    return pd.concat(frames, ignore_index=True), weather


def gme_zip(day="2025-01-01", quarter=False, both=False, missing=False):
    start = pd.Timestamp(day, tz="Europe/Rome")
    stop = (start.tz_localize(None) + pd.Timedelta(days=1)).tz_localize("Europe/Rome")
    blob = BytesIO()
    with ZipFile(blob, "w") as archive:
        for native15 in ([False, True] if both else [quarter]):
            step = 15 if native15 else 60
            count = int((stop - start).total_seconds() // (60 * step)) - int(missing)
            element = "Prezzi15" if native15 else "Prezzi"
            position = "Periodo" if native15 else "Ora"
            rows = [f"<{element}><Data>{start.strftime('%Y%m%d')}</Data><{position}>{i}</{position}>"
                    f"<NORD>{i},5</NORD><CSUD>25,25</CSUD><SUD>20</SUD></{element}>"
                    for i in range(1, count + 1)]
            archive.writestr(start.strftime("%Y%m%d") + "MGP" + element + ".xml", "<NewDataSet>" + "".join(rows) + "</NewDataSet>")
    blob.seek(0)
    with ZipFile(blob) as archive:
        members = gme_selection(archive.namelist())
    blob.seek(0)
    return blob, members


def henex_blob(day, step, missing=False):
    date = pd.Timestamp(day)
    start, stop = date.tz_localize("Europe/Rome"), (date + pd.Timedelta(days=1)).tz_localize("Europe/Rome")
    n = int((stop - start).total_seconds() / 60 / step) - int(missing)
    frame = pd.DataFrame([[date, *range(1, n + 1)], ["Market Clearing Price", *([None] * n)],
                          ["Greece Mainland" + (" 15min" if step == 15 else ""), *range(n)]])
    blob = BytesIO()
    with pd.ExcelWriter(blob, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="SPOT_Summary (SELL)", index=False, header=False)
    return blob.getvalue()


class MemoryCatalog:
    """Test-only asset API with artificial bytes; never an authorized fallback."""
    def __init__(self, payloads, selections=()):
        self.payloads = payloads
        self.assets = {aid: {"sha256": "0" * 64} for aid in payloads}
        self.manifest = dict(stages=[dict(stage_id="market", inputs=list(payloads))],
                             archive_selections=[], price_mirror_selection=list(selections))

    @contextmanager
    def open_input(self, asset_id, purpose):
        assert purpose == "process"
        yield BytesIO(json.dumps(self.payloads[asset_id]).encode())


def test_signed_hourly_net_before_positive_and_energy():
    h = integrate_native(quarter_grid(), [-400, 200, 200, 200], 15, "commercial_net_MW")
    assert h.commercial_net_MW.iloc[0] == 50
    assert h.integrated_MWh_if_power.iloc[0] == 50
    assert h.commercial_net_MW.clip(lower=0).iloc[0] != np.clip([-400, 200, 200, 200], 0, None).mean()


@pytest.mark.parametrize("values", [[10, np.nan, 30, 40], [10, np.inf, 30, 40]])
def test_missing_subinterval_is_not_filled(values):
    h = integrate_native(quarter_grid(), values, 15)
    assert np.isnan(h.value.iloc[0])
    assert h.valid_minutes.iloc[0] == 45


def test_missing_slot_not_extended():
    t = quarter_grid(n=8).delete(2)
    assert np.array_equal(infer_minutes(t), [15] * 7)
    h = integrate_native(t, np.ones(7), infer_minutes(t))
    assert np.isnan(h.value.iloc[0])
    assert h.value.iloc[1] == 1


@pytest.mark.parametrize("kind", ["duplicate", "overlap", "straddle", "unsupported", "naive"])
def test_bad_native_intervals_rejected(kind):
    t, values, duration = quarter_grid(), [1] * 4, 15
    if kind == "duplicate": t = pd.DatetimeIndex([t[0], t[0], t[2], t[3]])
    if kind == "overlap": duration = 30
    if kind == "straddle": t = t + pd.Timedelta(minutes=5)
    if kind == "unsupported": duration = 10
    if kind == "naive": t = t.tz_localize(None)
    with pytest.raises(ValueError): integrate_native(t, values, duration)


def test_flow_scale_and_italy_direction():
    payloads = {}
    for ep in ["cbet", "cbpf"]:
        for year in [2024, 2025, 2026]:
            t = quarter_grid(f"{year}-01-01T00:00Z", 8)
            payloads[f"flow_{ep}_it_{year}"] = dict(unix_seconds=(t.as_unit("s").asi8).tolist(),
                countries=[dict(name=country, data=[-.4, .2, .2, .2] * 2) for country in PARTNERS.values()])
    flows = MarketReader(MemoryCatalog(payloads)).build_flows()
    assert set(flows) == set(PARTNERS)
    for cid in PARTNERS:
        np.testing.assert_array_equal(flows[cid]["commercial_net_MW"], [50.] * 6)


@pytest.mark.parametrize("day,hours", [("2025-03-30", 23), ("2025-10-26", 25)])
def test_gme_dst_and_native15_precedence(day, hours):
    blob, selected = gme_zip(day, both=True)
    d, checks = parse_gme_archive(blob, selected)
    assert len(selected) == 1 and selected[0].endswith("Prezzi15.xml")
    assert len(d) == hours * 4
    assert checks[0]["complete_day"]
    assert d.timestamp_utc.is_unique
    h = integrate_native(d.timestamp_utc, d.NORD, d.duration_minutes)
    assert len(h) == hours
    assert h.value.iloc[0] == 3


def test_gme_incomplete_day_rejected():
    blob, selected = gme_zip(missing=True)
    with pytest.raises(ValueError, match="Incomplete GME"): parse_gme_archive(blob, selected)


def test_gme_unpinned_member_rejected():
    blob, _ = gme_zip(both=True)
    with pytest.raises(ValueError, match="pinned member"): parse_gme_archive(blob, ["20250101MGPPrezzi.xml"])


@pytest.mark.parametrize("day,step,hours", [("2025-03-30", 60, 23), ("2025-10-26", 15, 25)])
def test_henex_delivery_day_and_mcp(day, step, hours):
    d, audit = parse_henex_bytes(henex_blob(day, step), "artificial_v02.xlsx")
    assert len(d) == hours * 60 // step
    assert audit["revision"] == 2
    assert d.timestamp_utc.iloc[0] == pd.Timestamp(day, tz="Europe/Rome").tz_convert("UTC")
    assert d.value.iloc[-1] == len(d) - 1


def test_henex_incomplete_day_rejected():
    with pytest.raises(ValueError, match="period completeness"):
        parse_henex_bytes(henex_blob("2025-01-01", 60, missing=True), "artificial.xlsx")


def test_henex_revision_is_numeric_and_unambiguous():
    names = [f"2025/DAM/20250101_EL-DAM_ResultsSummary_EN_v{n}.xlsx" for n in [1, 2, 10]]
    assert henex_selection(names) == [names[2]]
    with pytest.raises(ValueError, match="Ambiguous"):
        henex_selection(names + ["other/DAM/20250101_EL-DAM_ResultsSummary_EN_v10.xlsx"])


@pytest.mark.parametrize("country,expected", [("SI", 33), ("GR", 22)])
def test_frozen_source_priority(country, expected):
    times = pd.date_range("2024-01-01T00:00Z", periods=2, freq="h")
    aid = f"price_mirror_{country}_2024"
    payloads = {aid: dict(unix_seconds=times.as_unit("s").asi8.tolist(), price=[33, 33], unit="EUR/MWh"),
                aid + "_metadata": dict(status="SUCCESS")}
    cat = MemoryCatalog(payloads, [dict(country=country, selected=True, asset_id=aid)])
    native = pd.DataFrame(dict(timestamp_utc=quarter_grid("2024-01-01T00:00Z", 8), value=22., duration_minutes=15))
    reader = MarketReader(cat)
    with patch.object(reader, "smard_native", return_value=native), patch.object(reader, "henex_native", return_value=native):
        result = reader.new_prices(country + "_IT")
    np.testing.assert_array_equal(result, [expected, expected])


def test_panel_40_columns_cutoff_spread_and_center():
    core, weather = fixture_panel()
    d = assemble_panel(core, weather)
    assert d.shape == (15, 40)
    assert set(d.corridor_id) == set(PARTNERS)
    assert d.groupby("corridor_id").Post.apply(list).iloc[0] == [0, 1, 1]
    np.testing.assert_array_equal(d.raw_price_spread, [15] * 15)
    np.testing.assert_array_equal(d.spread_c, [7.75] * 15)
    np.testing.assert_allclose(d[d.corridor_id.eq("ME_IT")].r_days, [-.5 / 24, .5 / 24, 1.5 / 24])
    assert d.weather_complete.all()
    assert d.origin_weather_wind_ms.eq(4).all()


def test_dst_repeated_hour_retains_distinct_utc_keys():
    times = pd.date_range("2025-10-25T22:00Z", periods=25, freq="h")
    core, weather = fixture_panel(times)
    d = assemble_panel(core, weather)
    me = d[d.corridor_id.eq("ME_IT")]
    assert len(me) == 25
    assert me.hour.eq(2).sum() == 2
    assert not d.duplicated(["corridor_id", "timestamp_utc"]).any()


def test_weather_missing_preserved_without_row_deletion():
    core, weather = fixture_panel()
    weather = weather[~((weather.geography == "ME") & (weather.timestamp_utc == weather.timestamp_utc.min()))]
    d = assemble_panel(core, weather)
    assert len(d) == 15
    assert d.origin_weather_temp_c.isna().sum() == 1
    assert (~d.weather_complete).sum() == 1


@pytest.mark.parametrize("kind", ["missing_weather", "duplicate_weather", "duplicate_panel", "FR", "missing_core", "wrong_zone", "wrong_proxy", "naive"])
def test_panel_invalid_inputs_fail(kind):
    core, weather = fixture_panel()
    if kind == "missing_weather": weather = None
    if kind == "duplicate_weather": weather = pd.concat([weather, weather.iloc[[0]]])
    if kind == "duplicate_panel": core = pd.concat([core, core.iloc[[0]]])
    if kind == "FR": core.loc[0, "corridor_id"] = "FR_IT"
    if kind == "missing_core": core.loc[0, "origin_price"] = np.nan
    if kind == "wrong_zone": core["destination_bidding_zone"] = "IT-SUD"
    if kind == "wrong_proxy": core["scheduled_export_proxy"] = 999
    if kind == "naive": core["timestamp_utc"] = core.timestamp_utc.dt.tz_localize(None)
    with pytest.raises(ValueError): assemble_panel(core, weather)


def test_authorized_construction_inherits_v1_but_rebuilds_si_gr():
    """Independent miniature hybrid lineage; a newer mirror cannot overwrite V1."""
    times = pd.date_range("2025-12-31T22:00Z", periods=3, freq="h")
    old = pd.concat([pd.DataFrame(dict(timestamp_utc=times, corridor_id=cid,
        origin_price=20., it_price=30., raw_price_spread=10., scheduled_export_proxy=[10., 20., 30.],
        scheduled_net_origin_to_it=[10., 20., 30.], physical_net_origin_to_it=[11., 22., 33.],
        physical_export_proxy=[11., 22., 33.], spread_center_pre2026=2.5, spread_c=7.5))
        for cid in ["AT_IT", "CH_IT", "ME_IT"]], ignore_index=True)

    class FrozenInput:
        manifest = {"schemas": {"v2": {"columns": CORE_COLUMNS}}}
        assets = {"frozen_v2_panel": {"schema_ref": "v2"}}

        @contextmanager
        def open_input(self, asset_id, purpose):
            assert asset_id == "frozen_v1_panel" and purpose == "process"
            yield BytesIO(old.to_csv(index=False).encode())

    flow = {cid: {"commercial_net_MW": pd.Series([99., 99., 99.], index=times),
                  "physical_net_MW": pd.Series([88., 88., 88.], index=times)} for cid in PARTNERS}
    reader = SimpleNamespace(build_flows=lambda: flow,
        gme_prices=lambda: {z: pd.Series([30., 50., 100.], index=times) for z in ["NORD", "SUD", "CSUD"]},
        new_prices=lambda cid: pd.Series([10., 40., 80.], index=times), diagnostics={"sources": [], "resolution": []})
    core = reconstruct_core(FrozenInput(), reader)
    for cid in ["AT_IT", "CH_IT", "ME_IT"]:
        inherited = core[core.corridor_id == cid]
        np.testing.assert_array_equal(inherited.commercial_net_MW, [10., 20., 30.])
        np.testing.assert_array_equal(inherited.spread_center, [2.5] * 3)
    for cid in ["SI_IT", "GR_IT"]:
        rebuilt = core[core.corridor_id == cid]
        np.testing.assert_array_equal(rebuilt.commercial_net_MW, [99.] * 3)
        # Only the first observation is before 2026 in Europe/Rome.
        np.testing.assert_array_equal(rebuilt.raw_price_spread, [20., 10., 20.])
        np.testing.assert_array_equal(rebuilt.spread_center, [20.] * 3)
        np.testing.assert_array_equal(rebuilt.spread_c, [0., -10., 0.])
