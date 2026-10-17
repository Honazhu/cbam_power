"""Artificial-only tests for the limited explicit reusable-methods prototype."""
from dataclasses import replace
import inspect
import socket
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from cbam_power import estimators, methods as m

pytestmark = pytest.mark.public


def specification(case="A", **changes):
    settings = {
        "A": dict(corridor=m.Corridor(id="ARTIFICIAL_ME_IT", origin="ME", destination="IT"),
                  calendar_timezone="Europe/Rome", cutoff="2026-01-01T00:00:00+01:00",
                  bandwidth_days=10.0, hac_hours=24),
        "B": dict(corridor=m.Corridor(id="ARTIFICIAL_CA_US", origin="CA", destination="US"),
                  calendar_timezone="America/Toronto", cutoff="2024-07-01T00:00:00-04:00",
                  bandwidth_days=8.0, hac_hours=7),
    }[case]
    return replace(m.RDiTSpec(model_id=f"ARTIFICIAL_{case}", outcome="positive_export_MW",
        kernel="triangular", fixed_effects=("hour", "weekday"), midpoint_minutes=30.0,
        donut_days=0.0, minimum_side=24, **settings), **changes)


def artificial_hourly(spec, n=720, noise=False):
    cut = pd.Timestamp(spec.cutoff).tz_convert("UTC")
    times = pd.date_range(cut - pd.Timedelta(hours=n//2), periods=n, freq="h")
    local = times.tz_convert(spec.calendar_timezone)
    r = (times-cut).total_seconds()/86400 + spec.midpoint_minutes/1440
    post = (times>=cut).astype(float)
    jump = 4.0 if spec.corridor.origin == "ME" else -6.0
    y = 40 + jump*post + .2*r - .03*post*r + .17*local.hour + .31*local.weekday
    if noise:
        y += .4*np.sin(np.arange(n)*.73)
    return pd.DataFrame({"timestamp_utc":times, "corridor_id":spec.corridor.id,
                         spec.outcome: y}), jump


@pytest.mark.parametrize("case", ["A", "B"])
def test_cross_region_explicit_end_to_end(case):
    spec = specification(case)
    original, jump = artificial_hourly(spec)
    start = original.timestamp_utc.iloc[0]
    native = pd.date_range(start, periods=4*len(original), freq="15min")
    signed = np.repeat(original[spec.outcome].to_numpy(),4) + np.tile([3.,-3.,2.,-2.],len(original))
    unit, direction = ("MW", "origin_to_destination") if case=="A" else ("GW", "destination_to_origin")
    values = signed if case=="A" else -signed/1000
    with mock.patch.object(socket, "create_connection", side_effect=AssertionError("NETWORK_FORBIDDEN")), \
         mock.patch("cbam_power.inputs.InputCatalog.__init__", side_effect=AssertionError("REAL_INPUTS_FORBIDDEN")), \
         mock.patch("scipy.optimize.minimize", side_effect=AssertionError("OPTIMIZATION_FORBIDDEN")):
        aggregated = m.aggregate_flows(native, values, corridor=spec.corridor, input_unit=unit,
            positive_direction=direction, duration_minutes=15)
        proof = m.validate_inputs(aggregated, timestamp="timestamp_utc",
            numeric_columns=("signed_net_MW", "positive_export_MW", "integrated_MWh"), nullable_columns=(),
            frequency_minutes=60, entity_column="corridor_id", entity_id=spec.corridor.id)
        assert proof["rows"]==720 and proof["missing_intervals"]==0
        np.testing.assert_allclose(aggregated.positive_export_MW, original[spec.outcome], atol=1e-13)
        np.testing.assert_allclose(aggregated.integrated_MWh, original[spec.outcome], atol=1e-13)
        fitted = m.fit_rdit(aggregated, timestamp="timestamp_utc", spec=spec)
        inferred = m.infer_hac(fitted, lag_hours=spec.hac_hours, contrasts={"jump":{"post":1.}})
        assert inferred["contrasts"]["jump"]["coefficient"]==pytest.approx(jump, abs=1e-10)
        np.testing.assert_allclose(fitted.beta[:4], [40,jump,.2,-.03], atol=1e-10)
        assert fitted.audit["references"]=={"hour":"0", "weekday":"0"}
        assert len(fitted.y)==int(spec.bandwidth_days*48)
        assert inferred["covariance"].shape==(33,33)
        assert fitted.audit["prototype_spec"]["calendar_timezone"]==spec.calendar_timezone
        donors = ("ARTIFICIAL_GR_IT", "ARTIFICIAL_AT_IT") if case=="A" else ("ARTIFICIAL_MX_US", "ARTIFICIAL_BR_US", "ARTIFICIAL_AR_US")
        weights = dict(zip(donors, [.3,.7] if case=="A" else [.2,.3,.5]))
        donor_values = np.array([10.,30.] if case=="A" else [5.,15.,25.])
        wide = pd.DataFrame({"time":original.timestamp_utc, spec.corridor.id:original[spec.outcome]})
        for donor, value in zip(donors, donor_values):
            wide[donor]=value
        prediction=m.predict_synthetic(wide,timestamp="time",target_id=spec.corridor.id,
            donor_ids=donors,weights=weights,input_kind="positive_export_MW")
        expected = 24.0 if case=="A" else 18.0
        np.testing.assert_allclose(prediction.synthetic_MW, expected)
        np.testing.assert_allclose(prediction.gap_MW, original[spec.outcome]-expected)
        geometry=m.aggregate_era5(original.timestamp_utc.iloc[:2],
            {"t2m":np.array([[273.15,283.15],[275.15,285.15]])},
            region_id=spec.corridor.origin, cell_ids=("west", "east"),
            area_weights={"west":.25,"east":.75}, raw_units={"t2m":"K"}, accumulated_shift_hours=-1)
        np.testing.assert_allclose(geometry.temp_c, [7.5,9.5], atol=1e-13)
        assert geometry.geography.eq(spec.corridor.origin).all()


def test_parameters_have_no_scientific_defaults():
    for cls in (m.Corridor,m.RDiTSpec):
        assert all(p.default is inspect.Parameter.empty for p in inspect.signature(cls).parameters.values())
    for name in m.PUBLIC_API:
        if callable(getattr(m,name)):
            assert all(p.default is inspect.Parameter.empty for p in inspect.signature(getattr(m,name)).parameters.values())
    with pytest.raises(TypeError):
        m.RDiTSpec()


def test_hac_independent_weighted_normal_equation_and_elapsed_pairs():
    spec=specification("B",fixed_effects=(),bandwidth_days=4,hac_hours=3)
    data,_=artificial_hourly(spec,n=220,noise=True)
    data=data.drop(index=[35,50,87]).reset_index(drop=True)
    fitted=m.fit_rdit(data,timestamp="timestamp_utc",spec=spec)
    times=fitted.sample.timestamp_utc
    r=(times-pd.Timestamp(spec.cutoff)).dt.total_seconds().to_numpy()/86400 + .5/24
    post=(times>=pd.Timestamp(spec.cutoff)).to_numpy(float)
    x=np.column_stack([np.ones(len(r)),post,r,post*r])
    w=1-abs(r)/4
    y=fitted.sample[spec.outcome].to_numpy()
    bread=np.linalg.inv(x.T@(w[:,None]*x))
    beta=np.linalg.solve(x.T@(w[:,None]*x),x.T@(w*y))
    score=(x*(w*(y-x@beta))[:,None])@bread*np.sqrt(len(y)/(len(y)-4))
    expected=np.zeros((4,4))
    for i,left in enumerate(times):
        for j,right in enumerate(times):
            lag=abs((left-right).total_seconds()/3600)
            if lag<=3:
                expected+=(1-lag/4)*np.outer(score[i],score[j])
    inference=m.infer_hac(fitted,lag_hours=3,contrasts={"jump":{"post":1},"slope_after":{"r":1,"post_r":1}})
    np.testing.assert_allclose(fitted.beta,beta,atol=1e-10,rtol=0)
    np.testing.assert_allclose(inference["covariance"],expected,atol=1e-12,rtol=1e-9)
    compressed=estimators.hac(pd.date_range(times.iloc[0],periods=len(times),freq="h"),score,3)
    assert not np.allclose(compressed,expected,atol=1e-10,rtol=1e-5)
    result=inference["contrasts"]["slope_after"]
    a=np.array([0.,0.,1.,1.]);se=np.sqrt(a@expected@a)
    assert result["coefficient"]==pytest.approx(a@beta)
    assert result["ci_low"]==pytest.approx(a@beta-1.96*se)
    assert 0<=result["p_value"]<=1


def test_hac_delegate_preserves_joint_covariance_of_distinct_samples():
    spec=specification("B",fixed_effects=(),bandwidth_days=4,hac_hours=2)
    d,_=artificial_hourly(spec,n=220,noise=True)
    a=m.fit_rdit(d,timestamp="timestamp_utc",spec=spec)
    b=m.fit_rdit(d.drop(index=[20,30,40]),timestamp="timestamp_utc",spec=replace(spec,model_id="ARTIFICIAL_B_SECOND"))
    cov=estimators.joint_covariance([a,b],2)
    np.testing.assert_allclose(cov[:4,:4],m.infer_hac(a,lag_hours=2,contrasts={"jump":{"post":1}})["covariance"],atol=1e-14)
    np.testing.assert_allclose(cov[4:,4:],b.covariance(2),atol=1e-14)
    # Independent cross-model covariance, using each model's actual UTC support.
    cross=np.zeros((4,4)); ia,ib=a.influence(),b.influence()
    for i,t1 in enumerate(a.sample.timestamp_utc):
        for j,t2 in enumerate(b.sample.timestamp_utc):
            delta=abs((t1-t2).total_seconds()/3600)
            if delta<=2:cross+=(1-delta/3)*np.outer(ia[i],ib[j])
    np.testing.assert_allclose(cov[:4,4:],cross,atol=1e-13)


def test_dst_midpoint_post_and_calendar_are_independent_explicit_parameters():
    spec=specification("B",cutoff="2024-11-03T01:00:00-05:00",bandwidth_days=2,midpoint_minutes=15)
    times=pd.date_range("2024-11-03T04:00Z",periods=5,freq="h")
    d=pd.DataFrame({"time":times,"corridor_id":spec.corridor.id,spec.outcome:np.arange(5.)})
    design=m.construct_rdit(d,timestamp="time",spec=spec)
    assert design.hour.tolist()==[0,1,1,2,3]
    assert design.post.tolist()==[0,0,1,1,1]
    np.testing.assert_allclose(design.r,[-1.75/24,-.75/24,.25/24,1.25/24,2.25/24])
    np.testing.assert_allclose(design.triangular,1-abs(design.r)/2)


def test_flow_net_before_positive_missing_and_reverse_sign():
    corridor=specification().corridor
    t=pd.date_range("2020-02-01T00:00Z",periods=8,freq="15min")
    h=m.aggregate_flows(t,[4,-2,-2,-4,1,1,np.nan,1],corridor=corridor,input_unit="MW",
                        positive_direction="origin_to_destination",duration_minutes=15)
    assert h.signed_net_MW.iloc[0]==-1 and h.positive_export_MW.iloc[0]==0
    assert pd.isna(h.signed_net_MW.iloc[1]) and h.valid_minutes.iloc[1]==45
    reversed_=m.aggregate_flows(t[:4],[.004,-.002,-.002,-.004],corridor=corridor,input_unit="GW",
                        positive_direction="destination_to_origin",duration_minutes=15)
    assert reversed_.signed_net_MW.iloc[0]==1 and reversed_.integrated_MWh.iloc[0]==1


def test_weather_manual_wind_order_units_time_shift_and_missing_cells():
    times=pd.date_range("2019-02-03T01:00Z",periods=2,freq="h")
    grids={"t2m":np.array([[273.15,293.15],[np.nan,280.15]]),
           "u10":np.array([[10.,-10.],[3.,0.]]),"v10":np.array([[0.,0.],[4.,5.]]),
           "ssrd":np.array([[3600.,7200.],[10800.,14400.]]),
           "tp":np.array([[.001,.003],[.002,.004]])}
    q=m.aggregate_era5(times,grids,region_id="ARTIFICIAL_REGION",cell_ids=("left","right"),
        area_weights={"left":.25,"right":.75},
        raw_units={"t2m":"K","u10":"m s-1","v10":"m s-1","ssrd":"J m-2","tp":"m"},
        accumulated_shift_hours=-1).set_index("timestamp_utc")
    assert q.index.tolist()==list(pd.date_range("2019-02-03T00:00Z",periods=3,freq="h"))
    assert q.loc[times[0],"wind_ms"]==10 # magnitude first; magnitude of averaged u would be 5.
    assert q.loc[times[1],"wind_ms"]==5
    assert q.loc[times[0],"temp_c"]==15 and pd.isna(q.loc[times[1],"temp_c"])
    assert q.loc[times[0]-pd.Timedelta(hours=1),"ssrd_processed"]==1.75
    assert q.loc[times[0]-pd.Timedelta(hours=1),"tp_processed"]==2.5
    assert pd.isna(q.loc[times[1],"ssrd_processed"])


def synthetic_fixture():
    return pd.DataFrame({"time":pd.date_range("2017-05-04T00:00Z",periods=4,freq="h"),
                         "TARGET":[20.,20.,20.,np.nan],"LEFT":[-10.,10.,1000.,5.],"RIGHT":[10.,np.nan,1000.,5.]})


def test_fixed_donors_positive_before_weighting_missing_zero_weight_and_no_clipping():
    d=synthetic_fixture()
    r=m.predict_synthetic(d,timestamp="time",target_id="TARGET",donor_ids=("LEFT","RIGHT"),
        weights={"LEFT":.5,"RIGHT":.5},input_kind="signed_net_MW")
    assert r.synthetic_MW.iloc[0]==5 and r.gap_MW.iloc[0]==15
    assert pd.isna(r.synthetic_MW.iloc[1])
    assert r.synthetic_MW.iloc[2]==1000 and r.gap_MW.iloc[2]==-980
    assert r.synthetic_MW.iloc[3]==5 and pd.isna(r.gap_MW.iloc[3])
    zero=m.predict_synthetic(d,timestamp="time",target_id="TARGET",donor_ids=("LEFT","RIGHT"),
        weights={"LEFT":1.,"RIGHT":0.},input_kind="signed_net_MW")
    assert pd.isna(zero.synthetic_MW.iloc[1])


def test_fixed_objective_manual_block_weights_no_scaling_or_optimizer():
    features=pd.DataFrame({"feature":["solar","hydro","january","hour0"],
        "block":["energy","energy","monthly","hourly"],
        "TARGET":[2.,4.,3.,5.],"LEFT":[0.,0.,0.,0.],"RIGHT":[4.,4.,4.,4.]})
    objective,details=m.evaluate_fixed_objective(features,target_id="TARGET",donor_ids=("LEFT","RIGHT"),
        weights={"LEFT":.75,"RIGHT":.25},balance=.4)
    # prediction 1 in every supplied feature; energy weights .2 each, others .3 each.
    assert objective==pytest.approx(.2*1+.2*9+.3*4+.3*16)
    assert [r["objective_row_weight"] for r in details]==[.2,.2,.3,.3]


@pytest.mark.parametrize("change,code",[
    ({"kernel":"uniform"},"UNSUPPORTED_KERNEL"),({"cutoff":"2024-01-01"},"EXPLICIT_AWARE_CUTOFF"),
    ({"calendar_timezone":"UNKNOWN"},"INVALID_CALENDAR"),({"bandwidth_days":0},"INVALID_BANDWIDTH"),
    ({"fixed_effects":("country",)},"UNSUPPORTED_FIXED"),({"hac_hours":-1},"INVALID_HAC"),
    ({"hac_hours":1.5},"INVALID_HAC"),({"midpoint_minutes":60},"INVALID_MIDPOINT"),
    ({"donut_days":10},"INVALID_DONUT"),({"outcome":"post"},"RESERVED_OUTCOME"),
])
def test_invalid_scientific_contract_rejected(change,code):
    with pytest.raises(ValueError,match=code):specification(**change)


@pytest.mark.parametrize("issue",["naive","duplicate","unordered","off_grid","entity","nullable_entity","infinite","undeclared_null"])
def test_validation_negative_inputs(issue):
    d=pd.DataFrame({"t":pd.date_range("2023-01-01T00:00Z",periods=4,freq="h"),"id":"TEST","x":np.arange(4.)})
    if issue=="naive":d["t"]=d.t.dt.tz_localize(None)
    if issue=="duplicate":d.loc[1,"t"]=d.loc[0,"t"]
    if issue=="unordered":d=d.iloc[::-1]
    if issue=="off_grid":d["t"]=d.t+pd.Timedelta(minutes=1)
    if issue=="entity":d.loc[0,"id"]="OTHER"
    if issue=="nullable_entity":
        d["id"]=d.id.astype("string")
        d.loc[0,"id"]=pd.NA
    if issue=="infinite":d.loc[0,"x"]=np.inf
    if issue=="undeclared_null":d.loc[0,"x"]=np.nan
    with pytest.raises(ValueError):
        m.validate_inputs(d,timestamp="t",numeric_columns=("x",),nullable_columns=(),
            frequency_minutes=60,entity_column="id",entity_id="TEST")


@pytest.mark.parametrize("change",[
    {"input_unit":"MWh"},{"positive_direction":"reporter_positive"},{"duration_minutes":20},
])
def test_invalid_market_contract(change):
    args=dict(corridor=specification().corridor,input_unit="MW",positive_direction="origin_to_destination",duration_minutes=15)
    args.update(change)
    with pytest.raises(ValueError):m.aggregate_flows(pd.date_range("2020-01-01T00:00Z",periods=4,freq="15min"),np.ones(4),**args)


@pytest.mark.parametrize("change",[
    {"cell_ids":("a","a")},{"area_weights":{"a":1.}}, {"area_weights":{"a":.4,"b":.4}},
    {"area_weights":{"a":-.1,"b":1.1}}, {"raw_units":{"t2m":"C"}},
    {"accumulated_shift_hours":0}, {"grids":{"t2m":np.ones((2,1))}},
    {"grids":{"unknown":np.ones((2,2))}},
])
def test_invalid_weather_contract(change):
    args=dict(grids={"t2m":np.full((2,2),273.15)},region_id="ARTIFICIAL",cell_ids=("a","b"),
        area_weights={"a":.5,"b":.5},raw_units={"t2m":"K"},accumulated_shift_hours=-1)
    args.update(change)
    with pytest.raises(ValueError):m.aggregate_era5(pd.date_range("2020-01-01T00:00Z",periods=2,freq="h"),**args)


@pytest.mark.parametrize("change",[
    {"donor_ids":("LEFT","LEFT")},{"donor_ids":("TARGET","LEFT")},
    {"weights":{"LEFT":1.}}, {"weights":{"LEFT":.4,"RIGHT":.4}},
    {"weights":{"LEFT":-.1,"RIGHT":1.1}},{"input_kind":"positive_export_MW"},
])
def test_invalid_synthetic_contract(change):
    args=dict(timestamp="time",target_id="TARGET",donor_ids=("LEFT","RIGHT"),
              weights={"LEFT":.5,"RIGHT":.5},input_kind="signed_net_MW")
    args.update(change)
    with pytest.raises(ValueError):m.predict_synthetic(synthetic_fixture(),**args)


def test_inference_rejects_undeclared_lag_and_unknown_terms():
    spec=specification();d,_=artificial_hourly(spec);f=m.fit_rdit(d,timestamp="timestamp_utc",spec=spec)
    with pytest.raises(ValueError,match="HAC_SPEC_MISMATCH"):
        m.infer_hac(f,lag_hours=23,contrasts={"jump":{"post":1}})
    with pytest.raises(ValueError,match="INVALID_CONTRAST"):
        m.infer_hac(f,lag_hours=24,contrasts={"jump":{"unknown":1}})


def test_missing_outcomes_not_filled_and_input_frame_unmodified():
    spec=specification();d,_=artificial_hourly(spec);d.loc[360,spec.outcome]=np.nan
    before=d.copy(deep=True)
    f=m.fit_rdit(d,timestamp="timestamp_utc",spec=spec)
    assert len(f.y)==479 and d.timestamp_utc.iloc[360] not in set(f.sample.timestamp_utc)
    pd.testing.assert_frame_equal(d,before)


def test_short_weekday_design_keeps_inherited_rank_block():
    spec=specification("B",bandwidth_days=5)
    d,_=artificial_hourly(spec)
    with pytest.raises(estimators.NotEstimable,match="BLOCKED_RANK"):
        m.fit_rdit(d,timestamp="timestamp_utc",spec=spec)


def test_rejects_corridor_relabeling_and_preserves_small_sample_status():
    spec=specification("B");d,_=artificial_hourly(spec)
    wrong=d.copy();wrong["corridor_id"]="ARTIFICIAL_UNRELATED"
    with pytest.raises(ValueError,match="ENTITY_MISMATCH"):
        m.fit_rdit(wrong,timestamp="timestamp_utc",spec=spec)
    with pytest.raises(estimators.NotEstimable,match="INSUFFICIENT_SUPPORT"):
        m.fit_rdit(d.iloc[330:390],timestamp="timestamp_utc",spec=spec)


def test_documented_external_artificial_case(monkeypatch):
    """Execute the maintained example with private input/network/optimizer sentinels."""
    import socket
    from pathlib import Path
    from markdown_it import MarkdownIt
    from cbam_power import inputs
    from scipy import optimize

    def forbidden(*args, **kwargs):
        raise AssertionError("DOCUMENTED_CASE_EXTERNAL_ACCESS")

    for name in ("create_connection", "socket"):
        monkeypatch.setattr(socket, name, forbidden)
    monkeypatch.setattr(inputs, "InputCatalog", forbidden)
    monkeypatch.setattr(optimize, "minimize", forbidden)
    source = (Path(__file__).resolve().parents[1]/"docs/methodology.md").read_text()
    examples = [t.content for t in MarkdownIt().parse(source)
                if t.type == "fence" and t.info == "python"
                and 'ARTIFICIAL_CA_US' in t.content]
    assert len(examples) == 1
    namespace = {"__name__": "documented_artificial_example"}
    exec(compile(examples[0], "<documented-artificial-case>", "exec"), namespace)
    assert namespace["input_check"]["missing_intervals"] == 0
    assert len(namespace["fitted"].sample) == 384
