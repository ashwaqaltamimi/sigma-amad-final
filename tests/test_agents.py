# tests/test_agents.py
import pandas as pd
import pytest
from sigma_agents import AgentReport, Blackboard


def _bb(anoms=2, n=10):
    """يبني Blackboard تجريبياً ببيانات مُصطنعة بسيطة (لا Streamlit)."""
    a = pd.DataFrame({
        "amount":      [9000.0, 61000.0],
        "fraud_type":  ["تجزئة عمليات (Structuring)", "انحراف سلوكي (Behavioral Deviation)"],
        "risk_score":  [85.0, 92.0],
        "date":        pd.to_datetime(["2025-11-15", "2025-12-01"]),
    })
    ctx = {"n": n, "anoms": anoms, "exposure": 70000.0,
           "struct": 1, "velocity": 0, "roundtrip": 0, "behavioral": 1}
    flow = {"inflow": 50000.0, "outflow": 70000.0, "net": -20000.0}
    sama = {"ctr_threshold": 60000,
            "alert_for": lambda t: f"تنبيه ساما: {t}",
            "str_deadline": lambda d, days=3: pd.Timestamp(d).date()}
    return Blackboard(ctx=ctx, anomalies=a, forecast=None, flow=flow,
                      sama=sama, amount_col="amount", date_col="date",
                      currency="ريال")


def test_blackboard_holds_inputs_and_posts_lines():
    bb = _bb()
    assert bb.ctx["anoms"] == 2
    line = bb.post("محمد", "اختبار")
    assert line == "محمد: اختبار"
    assert bb.lines == ["محمد: اختبار"]
    assert bb.reports == []


from sigma_agents import agent_detection


def test_detection_reports_real_counts():
    bb = _bb(anoms=2, n=10)
    rep = agent_detection(bb)
    assert rep.persona == "محمد"
    assert rep.key_number == "2 / 10"
    assert rep.confidence == 88.5            # متوسط (85+92)/2
    assert rep.handoff_to == "سارة"          # الكشف الإحصائي يسلّم للتحليل الشبكي
    assert any("محمد:" in ln for ln in bb.lines)
    assert rep.evidence                      # سلسلة أدلة بأعلى العمليات خطراً
    assert "خطر 92" in " ".join(rep.evidence)
    assert bb.reports[-1] is rep


def test_detection_empty_is_safe():
    bb = _bb(anoms=0, n=10)
    bb.anomalies = bb.anomalies.iloc[0:0]    # لا عمليات شاذة
    bb.ctx.update(struct=0, velocity=0, roundtrip=0, behavioral=0)
    rep = agent_detection(bb)
    assert rep.confidence == 0.0
    assert "لم يُرصد" in rep.decided


from sigma_agents import agent_network, build_str_records


def _network_result():
    """ناتج تحليل شبكي مُصطنع بنمطين (يحاكي بنية sigma_network)."""
    from sigma_network import NetworkFinding
    cycle = NetworkFinding(
        pattern="cycle", label="دوران شبكي (Closed Loop)",
        accounts=["SA-A", "SA-B", "SA-C"], txn_indices=[0, 1],
        total_amount=280_000.0, window_hours=5.0,
        evidence=["حلقة مغلقة: SA-A → SA-B → SA-C → SA-A خلال 5.0 ساعة"],
        severity=95.0, sama_ref="المادة 12")
    mule = NetworkFinding(
        pattern="mule", label="شبكة تجميع وتمرير (Mule Network)",
        accounts=["SA-HUB", "SA-S1"], txn_indices=[1],
        total_amount=68_750.0, window_hours=49.0,
        evidence=["حساب تجميع SA-HUB"], severity=90.0, sama_ref="المادة 14")
    return {"findings": [cycle, mule], "n_edges": 12, "n_accounts": 9,
            "counts": {"cycle": 1, "mule": 1, "passthrough": 0}}


def test_network_agent_honest_when_no_counterparty():
    bb = _bb()                               # network=None في الـ fixture
    rep = agent_network(bb)
    assert rep.persona == "سارة"
    assert rep.key_number == "—"
    assert "غير مفعّل" in rep.decided        # صدق: لا عمود طرف مقابل → لا ادعاء
    assert rep.handoff_to == "نورة"


def test_network_agent_reports_findings_with_evidence():
    bb = _bb()
    bb.network = _network_result()
    rep = agent_network(bb)
    assert rep.key_number == "2 نمط · 5 حساب"
    assert "حلقة مغلقة" in rep.decided
    assert any("حلقة مغلقة" in e for e in rep.evidence)
    assert any("سارة:" in ln for ln in bb.lines)


from sigma_agents import agent_compliance


def test_compliance_counts_str_and_ctr():
    bb = _bb(anoms=2, n=10)            # حالة تجزئة واحدة + مبلغ 61000 ≥ عتبة CTR
    rep = agent_compliance(bb)
    assert rep.persona == "نورة"
    assert rep.key_number == "STR 1 · CTR 1"
    # لا ثقة مختلقة: نورة حتمية 100% في مخرجها التنظيمي، فلا نعرض رقم ثقة زائفاً
    assert rep.confidence is None
    assert rep.handoff_to == "ريم"
    assert any("تنبيه ساما:" in ln for ln in bb.lines)   # جملة ساما الثابتة تُعرَض


def test_compliance_counts_network_cases_as_str():
    bb = _bb(anoms=2, n=10)
    bb.network = _network_result()
    rep = agent_compliance(bb)
    # حالة تجزئة + حلقة + شبكة تجميع = 3 حالات STR (بلاغ لكل حالة لا لكل عملية)
    assert rep.key_number == "STR 3 · CTR 1"
    assert "مسودات البلاغ جاهزة" in rep.decided
    assert "فوري" in rep.decided               # الإبلاغ فوري وفق المادة 15
    assert len(rep.evidence) == 3


def test_str_records_fields_are_computed():
    bb = _bb(anoms=2, n=10)
    bb.network = _network_result()
    records = build_str_records(bb)
    assert len(records) == 3
    struct = [r for r in records if "Structuring" in r["pattern"]][0]
    assert struct["txn_count"] == 1
    assert struct["total_amount"] == 9000.0
    assert struct["deadline"] != "—"
    mule = [r for r in records if "Mule" in r["pattern"]][0]
    assert mule["total_amount"] == 68_750.0
    assert mule["sama_ref"] == "المادة 14"


def test_compliance_empty_is_safe():
    bb = _bb(anoms=0, n=10)
    bb.anomalies = bb.anomalies.iloc[0:0]
    bb.ctx["struct"] = 0
    rep = agent_compliance(bb)
    assert "لا بلاغات" in rep.decided
    assert rep.key_number == "STR 0 · CTR 0"


import numpy as np
from sigma_agents import agent_forecast


def test_forecast_none_is_safe():
    bb = _bb()                         # forecast=None في الـ fixture
    rep = agent_forecast(bb)
    assert rep.persona == "فهد"
    assert rep.confidence is None
    assert "غير كافية" in rep.decided


def test_forecast_real_shows_ci_not_misleading_pct():
    bb = _bb()
    fdf = pd.DataFrame({"forecast": [1000.0, 1100.0, 900.0]})
    bb.forecast = {"forecast": fdf, "slope": -5.0, "sigma": 100.0,
                   "history": pd.Series([1, 2, 3])}
    rep = agent_forecast(bb)
    assert rep.confidence is None            # عدم اليقين بنطاق ثقة لا بنسبة مضلّلة
    assert "ريال/يوم" in rep.key_number
    assert "هابط" in rep.decided             # slope<0
    assert "نطاق ثقة 95%" in rep.decided


from sigma_agents import agent_reporting


def test_reporting_summarizes_exposure():
    bb = _bb(anoms=2, n=10)
    rep = agent_reporting(bb)
    assert rep.persona == "ريم"
    assert rep.handoff_to == "القائدة"
    assert "70,000" in rep.key_number        # exposure
    assert "ملخص تنفيذي" in rep.decided


from sigma_agents import run_agent_pipeline


def test_pipeline_runs_all_in_order_with_mission_confidence():
    bb = _bb(anoms=2, n=10)
    result = run_agent_pipeline(bb)
    ids = [r.agent_id for r in result["reports"]]
    assert ids == ["mohammed", "sara", "noura", "fahad", "reem"]
    assert result["final"]["confidence"] == 80.0     # (10-2)/10*100
    assert "plan" in result and result["plan"]
    assert result["final"]["status"] == "Mission Approved"


def test_pipeline_empty_data_is_safe():
    bb = _bb(anoms=0, n=5)
    bb.anomalies = bb.anomalies.iloc[0:0]
    bb.ctx["struct"] = 0
    result = run_agent_pipeline(bb)
    assert len(result["reports"]) == 5
    assert result["final"]["confidence"] == 100.0     # (5-0)/5*100
