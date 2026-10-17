"""Offline market parsers migrated from V3 ``reused/native_data.py`` and
``reused/official_prices.py``. Scientific interval arithmetic and source priority
are preserved; filesystem discovery, global ROOT rebinding and writes are removed.

All authorized inputs are explicit PR2 manifest assets. Parsers never download,
choose a latest file, read a benchmark, or modify an input.
"""
from datetime import datetime
from io import BytesIO
import json
from pathlib import PurePosixPath
import re
from zipfile import ZipFile

from lxml import etree
import numpy as np
import pandas as pd

TZ = "Europe/Rome"
PARTNERS = {"AT_IT": "Austria", "CH_IT": "Switzerland", "SI_IT": "Slovenia",
            "GR_IT": "Greece", "ME_IT": "Montenegro"}
ZONES = {"AT_IT": "IT-NORD", "CH_IT": "IT-NORD", "SI_IT": "IT-NORD",
         "GR_IT": "IT-SUD", "ME_IT": "IT-CSUD"}
SOURCE_FUNCTIONS = {
    "infer_minutes": "V3/src/reused/native_data.py::infer_minutes",
    "integrate_native": "V3/src/reused/native_data.py::integrate_native",
    "parse_gme_archive": "V3/src/reused/native_data.py::gme_prices",
    "parse_henex_bytes": "V3/src/reused/official_prices.py::parse_henex_bytes",
    "MarketReader": "V3/src/reused/native_data.py::{build_flows,new_prices}; official_prices.py::{smard_native,henex_native}",
}


def _utc_index(t):
    t = pd.DatetimeIndex(t)
    if t.tz is None:
        raise ValueError("Native timestamps must have explicit timezone")
    return t.tz_convert("UTC")


def infer_minutes(t):
    """Frozen nominal within-day interval; missing slots are never extended."""
    t = _utc_index(t)
    if t.has_duplicates or not t.is_monotonic_increasing:
        raise ValueError("Duplicate/nonmonotone native UTC")
    prev = pd.Series(t, index=t).diff().dt.total_seconds() / 60
    dates = pd.Series(t.tz_convert(TZ).date, index=t)
    good = prev.where(prev.isin([15, 30, 60]) & dates.eq(dates.shift()))
    if good.notna().sum() == 0:
        raise ValueError("Native frequency not identified")
    dur = good.groupby(dates).transform("min").fillna(good.mode().iloc[0])
    gap = (pd.Series(t, index=t).shift(-1) - pd.Series(t, index=t)).dt.total_seconds() / 60
    dur = dur.where(gap.isna() | gap.ge(dur), gap)
    return dur.to_numpy()


def integrate_native(t, values, minutes, name="value"):
    """Frozen duration integration: full 60 valid minutes required per hour.

    The returned hourly mean MW and hourly integral MWh have the same numerical
    value because the output interval is one hour. Positive parts are deliberately
    absent here: apply them to the resulting hourly signed commercial flow.
    """
    t = _utc_index(t)
    x = np.asarray(values, float)
    m = np.broadcast_to(np.asarray(minutes, float), len(t))
    if len(x) != len(t) or t.has_duplicates or not t.is_monotonic_increasing:
        raise ValueError("Invalid native alignment")
    if np.any(~np.isin(m, [15, 30, 60])):
        raise ValueError("Unsupported duration")
    end = t + pd.to_timedelta(m, unit="m")
    if len(t) > 1 and np.any(end[:-1] > t[1:]):
        raise ValueError("Overlapping native intervals")
    if np.any(end > t.floor("h") + pd.Timedelta(hours=1)):
        raise ValueError("Unexpected straddling intervals")
    valid = np.isfinite(x)
    d = pd.DataFrame({"timestamp_utc": t.floor("h"), "valid_minutes": np.where(valid, m, 0),
                      "intervals": 1, "observed_intervals": valid.astype(int),
                      "weighted": np.where(valid, x * m, 0)})
    h = d.groupby("timestamp_utc").sum()
    if h.valid_minutes.gt(60).any():
        raise ValueError("Duplicate/overlap exceeds full hour")
    h["missing_subinterval"] = ~np.isclose(h.valid_minutes, 60)
    h[name] = (h.weighted / 60).where(~h.missing_subinterval)
    h["integrated_MWh_if_power"] = h[name]
    return h


def native_series(t, values, source_id, variable, minutes=None, scale=1):
    """Pure counterpart of old native_series; native/intermediate writes removed."""
    t = _utc_index(t)
    m = infer_minutes(t) if minutes is None else np.asarray(minutes)
    h = integrate_native(t, np.asarray(values, float) * scale, m, variable)
    resolution = pd.DataFrame({"date": t.tz_convert(TZ).date, "duration_minutes": m}).groupby("date").agg(
        native_minutes_min=("duration_minutes", "min"),
        native_minutes_max=("duration_minutes", "max"), native_rows=("duration_minutes", "size")).reset_index()
    resolution["source_id"] = source_id
    resolution["variable"] = variable
    return h[variable], resolution


def _join_unique(parts, label):
    if not parts:
        raise ValueError(f"Missing selected input: {label}")
    joined = pd.concat(parts).sort_index()
    if joined.index.has_duplicates:
        raise ValueError(f"Overlapping selected file vintages: {label}")
    return joined


def gme_selection(names):
    """Frozen archive rule: native Prezzi15 takes precedence over Prezzi."""
    names = set(names)
    return sorted(n for n in names if n.endswith("Prezzi15.xml") or
                  (n.endswith("Prezzi.xml") and n.replace("Prezzi.xml", "Prezzi15.xml") not in names))


def parse_gme_archive(file, selected_members, source="GME"):
    """Parse explicitly pinned members and verify the frozen native15 rule."""
    records, checks = [], []
    with ZipFile(file) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP member names")
        if list(selected_members) != gme_selection(names):
            raise ValueError("GME pinned member selection disagrees with frozen rule")
        for member in selected_members:
            q = member.endswith("Prezzi15.xml")
            parser = etree.XMLParser(resolve_entities=False, no_network=True)
            root = etree.fromstring(archive.read(member), parser=parser)
            rows = root.findall("Prezzi15" if q else "Prezzi")
            if not rows:
                raise ValueError("Empty selected GME member: " + member)
            day = pd.Timestamp(rows[0].findtext("Data"), tz=TZ)
            if day.year not in (2024, 2025, 2026):
                raise ValueError("Unexpected frozen GME year")
            minutes = 15 if q else 60
            nextday = (day.tz_localize(None) + pd.Timedelta(days=1)).tz_localize(TZ)
            expected = int((nextday - day).total_seconds() / 60 / minutes)
            positions = [int(r.findtext("Periodo" if q else "Ora")) for r in rows]
            complete = sorted(positions) == list(range(1, expected + 1))
            checks.append(dict(source=source, member=member, date=str(day.date()), native_minutes=minutes,
                               expected=expected, observed=len(rows), complete_day=complete))
            if not complete:
                raise ValueError("Incomplete GME delivery day: " + member)
            for pos, row in zip(positions, rows):
                entry = dict(timestamp_utc=day.tz_convert("UTC") + pd.Timedelta(minutes=minutes * (pos - 1)),
                             duration_minutes=minutes, source_member=member)
                for zone in ["NORD", "CSUD", "SUD"]:
                    raw = row.findtext(zone)
                    entry[zone] = float(raw.replace(",", ".")) if raw else np.nan
                records.append(entry)
    return pd.DataFrame(records), checks


def parse_henex_bytes(blob, source):
    """Direct migration of official_prices.parse_henex_bytes (MCP, not index)."""
    d = pd.read_excel(BytesIO(blob), sheet_name="SPOT_Summary (SELL)", header=None)
    rows = d.index[d.iloc[:, 0].astype(str).str.strip().eq("Market Clearing Price")]
    if len(rows) != 1:
        raise ValueError("Ambiguous HEnEx MCP heading " + source)
    row = int(rows[0]) + 1
    label = str(d.iloc[row, 0]).strip()
    if not label.startswith("Greece Mainland"):
        raise ValueError("Unexpected price zone " + label)
    step = 15 if "15min" in label else 60
    date_rows = [i for i in range(min(4, len(d))) if isinstance(d.iloc[i, 0], (pd.Timestamp, datetime))]
    if len(date_rows) != 1:
        raise ValueError("HEnEx delivery date not identified " + source)
    ir = date_rows[0]
    date = pd.Timestamp(d.iloc[ir, 0]).normalize()
    start = date.tz_localize(TZ)
    stop = (date + pd.Timedelta(days=1)).tz_localize(TZ)
    n = int((stop - start).total_seconds() / 60 / step)
    periods = pd.to_numeric(d.iloc[ir, 1:], errors="coerce")
    actual = periods.dropna().astype(int).tolist()
    if actual != list(range(1, n + 1)):
        raise ValueError("HEnEx DST/period completeness " + source)
    cols = periods.index[periods.notna()]
    values = pd.to_numeric(d.loc[row, cols], errors="coerce").to_numpy(float)
    t = pd.date_range(start.tz_convert("UTC"), periods=n, freq=f"{step}min")
    match = re.search(r"_v(\d+)", source)
    version = int(match.group(1)) if match else 1
    native = pd.DataFrame(dict(timestamp_utc=t, value=values, duration_minutes=step,
                               source_file=source, version=version))
    audit = dict(source=source, delivery_date=str(date.date()),
                 timezone="CET/CEST=Europe/Rome; starts01:00 Eastern", price_row=label,
                 native_minutes=step, observed_periods=len(actual), expected_periods=n,
                 missing_prices=int(np.isnan(values).sum()), unit="EUR/MWh", revision=version)
    return native, audit


def henex_selection(names):
    """Frozen highest numeric revision within each registered annual archive."""
    grouped = {}
    for name in names:
        if "/DAM/" in name and re.search(r"\d{8}_EL-DAM_ResultsSummary_EN_v\d+\.xlsx$", name):
            grouped.setdefault(PurePosixPath(name).name[:8], []).append(name)
    result = []
    for _, choices in sorted(grouped.items()):
        revisions = [int(re.search(r"_v(\d+)", x).group(1)) for x in choices]
        if len(revisions) != len(set(revisions)):
            raise ValueError("Ambiguous HEnEx same-day revision")
        result.append(choices[int(np.argmax(revisions))])
    return result


class MarketReader:
    """Read explicit registered market assets; retain diagnostics in memory."""
    def __init__(self, catalog):
        self.catalog = catalog
        stages = [s for s in catalog.manifest["stages"] if s["stage_id"] == "market"]
        if len(stages) != 1:
            raise ValueError("Missing/ambiguous market stage")
        self.asset_ids = tuple(stages[0]["inputs"])
        self.archives = {a["asset_id"]: a["selected_members"]
                         for a in catalog.manifest["archive_selections"]}
        self.diagnostics = {"resolution": [], "sources": [], "gme_delivery": [],
                            "henex_delivery": [], "price_overlap": []}

    def _open(self, asset_id):
        if asset_id not in self.asset_ids:
            raise ValueError("Asset not registered for market stage: " + asset_id)
        self.diagnostics["sources"].append(dict(asset_id=asset_id,
            sha256=self.catalog.assets[asset_id]["sha256"], role="registered_raw_input"))
        return self.catalog.open_input(asset_id, purpose="process")

    def _json(self, asset_id):
        with self._open(asset_id) as handle:
            return json.load(handle)

    def _series(self, t, values, asset_id, variable, minutes=None, scale=1):
        series, resolution = native_series(t, values, asset_id, variable, minutes, scale)
        self.diagnostics["resolution"].append(resolution)
        self.diagnostics["sources"].append(dict(asset_id=asset_id, variable=variable,
                                                sha256=self.catalog.assets[asset_id]["sha256"], scale=scale))
        return series

    def build_flows(self):
        output = {cid: {} for cid in PARTNERS}
        for ep, variable in [("cbet", "commercial_net_MW"), ("cbpf", "physical_net_MW")]:
            parts = {cid: [] for cid in PARTNERS}
            for year in (2024, 2025, 2026):
                aid = f"flow_{ep}_it_{year}"
                payload = self._json(aid)
                t = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
                for cid, partner in PARTNERS.items():
                    items = [z for z in payload["countries"] if z["name"] == partner]
                    if len(items) != 1:
                        raise ValueError("Missing/ambiguous country boundary: " + aid)
                    parts[cid].append(self._series(t, items[0]["data"], aid, variable, scale=1000))
            for cid in PARTNERS:
                output[cid][variable] = _join_unique(parts[cid], f"{ep}:{cid}")
        return output

    def gme_prices(self):
        pieces = []
        for year in (2024, 2025, 2026):
            aid = f"gme_prices_{year}"
            with self._open(aid) as file:
                d, checks = parse_gme_archive(file, self.archives[aid], aid)
            pieces.append(d)
            self.diagnostics["gme_delivery"].extend(checks)
        d = pd.concat(pieces).sort_values("timestamp_utc")
        if d.timestamp_utc.duplicated().any():
            raise ValueError("Duplicate GME delivery intervals")
        result = {}
        for zone in ("NORD", "CSUD", "SUD"):
            result[zone], resolution = native_series(d.timestamp_utc, d[zone], "gme_selected_archives",
                                                     zone, d.duration_minutes)
            self.diagnostics["resolution"].append(resolution)
        return result

    def smard_native(self, country):
        pieces = []
        ids = [a for a in self.asset_ids if re.fullmatch(rf"smard_{country}_\d+", a)]
        for aid in ids:
            j = self._json(aid)
            if "series" not in j:
                raise ValueError("Selected SMARD payload lacks series: " + aid)
            d = pd.DataFrame(j["series"], columns=["unix_ms", "value"])
            d["timestamp_utc"] = pd.to_datetime(d.unix_ms, unit="ms", utc=True)
            d["value"] = pd.to_numeric(d.value, errors="coerce")
            d["source_file"] = aid
            d["source_vintage_ms"] = j.get("meta_data", {}).get("created")
            pieces.append(d)
        if not pieces:
            raise ValueError("Missing selected SMARD sources")
        d = pd.concat(pieces).sort_values("timestamp_utc")
        if d.timestamp_utc.duplicated().any():
            raise ValueError("SMARD weekly overlap/revision requires explicit resolution")
        cutoff = pd.Timestamp("2025-10-01", tz=TZ).tz_convert("UTC")
        pre = d[d.timestamp_utc < cutoff]
        if pre.groupby(pre.timestamp_utc.dt.floor("h")).value.nunique().gt(1).any():
            raise ValueError("Unexpected pre-SDAC within-hour prices")
        d["duration_minutes"] = 15
        d["market_mtu_minutes"] = np.where(d.timestamp_utc < cutoff, 60, 15)
        return d

    def henex_native(self):
        pieces = []
        for year in (2024, 2025):
            aid = f"henex_annual_{year}"
            with self._open(aid) as file, ZipFile(file) as archive:
                if self.archives[aid] != henex_selection(archive.namelist()):
                    raise ValueError("HEnEx pinned revision selection disagrees with frozen rule")
                for member in self.archives[aid]:
                    d, audit = parse_henex_bytes(archive.read(member), member)
                    pieces.append(d)
                    audit["asset_id"] = aid
                    self.diagnostics["henex_delivery"].append(audit)
        for aid in self.asset_ids:
            if not re.fullmatch(r"henex_results_\d{8}", aid):
                continue
            with self._open(aid) as file:
                d, audit = parse_henex_bytes(file.read(), aid + ".xlsx")
            pieces.append(d)
            self.diagnostics["henex_delivery"].append(audit)
        if not pieces:
            raise ValueError("Missing selected HEnEx sources")
        d = pd.concat(pieces).sort_values("timestamp_utc")
        if d.timestamp_utc.duplicated().any():
            raise ValueError("HEnEx overlapping selected delivery versions")
        return d

    def new_prices(self, cid):
        if cid not in ("SI_IT", "GR_IT"):
            raise ValueError("Only SI/GR prices are reconstructed; AT/CH/ME inherit V1")
        country, parts = cid[:2], []
        for selection in self.catalog.manifest["price_mirror_selection"]:
            if selection["country"] != country or not selection["selected"]:
                continue
            aid = selection["asset_id"]
            metadata = self._json(aid + "_metadata")
            if metadata["status"] != "SUCCESS":
                raise ValueError("Pinned price mirror is not SUCCESS")
            j = self._json(aid)
            if j.get("unit", "").replace(" ", "") != "EUR/MWh":
                raise ValueError("Unexpected price unit")
            t = pd.to_datetime(j["unix_seconds"], unit="s", utc=True)
            parts.append(self._series(t, j["price"], aid, "origin_price"))
        mirror = _join_unique(parts, country + " mirror")
        native = self.smard_native(country) if country == "SI" else self.henex_native()
        official, resolution = native_series(native.timestamp_utc, native.value, country + "_official",
                                              "origin_price", native.duration_minutes)
        self.diagnostics["resolution"].append(resolution)
        overlap = pd.concat([mirror.rename("mirror"), official.rename("official")], axis=1).dropna()
        self.diagnostics["price_overlap"].append(overlap.assign(country=country,
                                                               difference=overlap.official - overlap.mirror))
        # Frozen SI mirror first; GR HEnEx first. No interpolation of absent prices.
        return mirror.combine_first(official) if country == "SI" else official.combine_first(mirror)
