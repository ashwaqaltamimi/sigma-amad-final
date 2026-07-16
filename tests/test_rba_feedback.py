# tests/test_rba_feedback.py
# اختبارات النهج القائم على المخاطر (RBA) وحلقة التغذية الراجعة
import pandas as pd
import pytest

from sigma_rba import (rate_customers, tier_map, velocity_thresholds,
                       boost_anomaly_scores, RBA_SCORE_BOOST)
from sigma_feedback import (build_review_cases, apply_feedback, case_key,
                            feedback_log, VERDICT_FP, VERDICT_TP,
                            load_feedback_history, history_to_decisions,
                            append_feedback_cycle, learning_curve,
                            HISTORY_COLUMNS)


def _profiles():
    return pd.DataFrame([
        # حساب تجميع نمطي: طالب + حساب شهرين ← عالي (20) ؟ لا: 20 فقط = منخفض
        {"account_id": "SA-NEW", "occupation": "طالب", "residency": "مقيم",
         "fatf_jurisdiction": 0, "pep": 0, "account_age_months": 2},
        # مستورد غير مقيم بولاية FATF وحساب حديث ← عالي (35+25+20+10=90)
        {"account_id": "SA-HOT", "occupation": "مستورد ومصدّر", "residency": "غير مقيم",
         "fatf_jurisdiction": 1, "pep": 0, "account_age_months": 3},
        # موظف حكومي سعودي قديم ← منخفض (0)
        {"account_id": "SA-SAFE", "occupation": "موظف حكومي", "residency": "سعودي",
         "fatf_jurisdiction": 0, "pep": 0, "account_age_months": 120},
        # PEP ← متوسط (40)
        {"account_id": "SA-PEP", "occupation": "مسؤول حكومي رفيع", "residency": "سعودي",
         "fatf_jurisdiction": 0, "pep": 1, "account_age_months": 60},
    ])


# ── التصنيف ──────────────────────────────────────────────────────────

def test_rating_scores_and_tiers_are_declared_sums():
    r = rate_customers(_profiles()).set_index("account_id")
    assert r.loc["SA-HOT", "rba_score"] == 90 and r.loc["SA-HOT", "rba_tier"] == "عالي"
    assert r.loc["SA-SAFE", "rba_score"] == 0 and r.loc["SA-SAFE", "rba_tier"] == "منخفض"
    assert r.loc["SA-PEP", "rba_score"] == 40 and r.loc["SA-PEP", "rba_tier"] == "متوسط"
    assert "FATF" in r.loc["SA-HOT", "rba_reasons"]        # الأسباب مقروءة
    assert r.loc["SA-SAFE", "rba_reasons"] == "لا عوامل خطورة مرتفعة"


def test_rating_returns_none_without_required_columns():
    assert rate_customers(pd.DataFrame({"account_id": ["x"]})) is None
    assert rate_customers(None) is None


def test_velocity_threshold_tightens_with_tier():
    tiers = tier_map(rate_customers(_profiles()))
    df = pd.DataFrame({"account_id": ["SA-HOT", "SA-PEP", "SA-SAFE", "SA-UNKNOWN"]})
    thr = velocity_thresholds(df, "account_id", tiers)
    assert list(thr) == [3, 4, 5, 5]      # عالي=3 · متوسط=4 · منخفض/مجهول=5


def test_boost_raises_only_high_tier_anomalies():
    tiers = {"SA-HOT": "عالي", "SA-SAFE": "منخفض"}
    df = pd.DataFrame({
        "account_id": ["SA-HOT", "SA-HOT", "SA-SAFE"],
        "is_anomaly": [True, False, True],
        "risk_score": [80.0, 50.0, 95.0],
    })
    out = boost_anomaly_scores(df, "account_id", tiers)
    assert out.loc[0, "risk_score"] == 80.0 + RBA_SCORE_BOOST   # شاذ + عالي
    assert out.loc[1, "risk_score"] == 50.0                     # غير شاذ: لا رفع
    assert out.loc[2, "risk_score"] == 95.0                     # منخفض: لا رفع
    assert list(out["rba_tier"]) == ["عالي", "عالي", "منخفض"]


# ── حلقة التغذية الراجعة ─────────────────────────────────────────────

def _detections():
    return pd.DataFrame({
        "account_id": ["SA-1", "SA-1", "SA-2", "SA-3"],
        "fraud_type": ["انحراف سلوكي (Behavioral Deviation)"] * 2 +
                      ["تجزئة عمليات (Structuring)", "تكرار مشبوه (Velocity)"],
        "is_anomaly": [True, True, True, True],
        "risk_score": [70.0, 65.0, 90.0, 85.0],
        "amount": [1000.0, 2000.0, 9500.0, 400.0],
    })


def test_review_cases_group_by_account_and_type():
    df = _detections()
    cases = build_review_cases(df, "account_id", "amount")
    assert len(cases) == 3                                  # SA-1 حالتاه مجمّعتان
    top = cases[0]
    assert top["account"] == "SA-2" and top["max_risk"] == 90.0
    sa1 = [c for c in cases if c["account"] == "SA-1"][0]
    assert sa1["txn_count"] == 2 and sa1["total_amount"] == 3000.0


def test_false_positive_verdict_closes_case_and_true_confirms():
    df = _detections()
    decisions = {
        case_key("SA-1", "انحراف سلوكي (Behavioral Deviation)"): VERDICT_FP,
        case_key("SA-2", "تجزئة عمليات (Structuring)"): VERDICT_TP,
    }
    out, closed, confirmed = apply_feedback(df, decisions, "account_id")
    assert closed == 1 and confirmed == 1
    assert not out.loc[0, "is_anomaly"] and not out.loc[1, "is_anomaly"]
    assert "أُقفلت" in out.loc[0, "review_status"]
    assert out.loc[2, "is_anomaly"] and "مؤكدة" in out.loc[2, "review_status"]
    assert out.loc[3, "is_anomaly"]                          # حالة لم تُراجع: كما هي


def test_feedback_improves_precision_measurably():
    """قرار «إنذار كاذب» واحد يرفع الدقة المقاسة — جوهر الحلقة."""
    df = _detections()
    truth = pd.Series([False, False, True, True])            # SA-1 بريء فعلاً
    def precision(frame):
        tp = int((frame["is_anomaly"] & truth).sum())
        fp = int((frame["is_anomaly"] & ~truth).sum())
        return tp / (tp + fp)
    before = precision(df)
    out, _, _ = apply_feedback(
        df, {case_key("SA-1", "انحراف سلوكي (Behavioral Deviation)"): VERDICT_FP},
        "account_id")
    assert before == 0.5 and precision(out) == 1.0


def test_feedback_log_is_exportable_audit_trail():
    log = feedback_log({case_key("SA-1", "تكرار مشبوه (Velocity)"): VERDICT_FP})
    assert list(log.columns) == ["الحساب", "نمط الاشتباه", "قرار المراجع"]
    assert log.iloc[0]["الحساب"] == "SA-1"


def test_apply_feedback_preserves_prefeedback_baseline():
    df = _detections()
    out, _, _ = apply_feedback(
        df, {case_key("SA-1", "انحراف سلوكي (Behavioral Deviation)"): VERDICT_FP},
        "account_id")
    assert list(out["is_anomaly_prefeedback"]) == [True, True, True, True]
    assert list(out["is_anomaly"]) == [False, False, True, True]


# ── الذاكرة التراكمية عبر الجلسات ────────────────────────────────────

def test_load_feedback_history_missing_file_is_honest_empty():
    hist = load_feedback_history(None)
    assert list(hist.columns) == HISTORY_COLUMNS and len(hist) == 0
    hist2 = load_feedback_history("/no/such/path/feedback_memory.csv")
    assert len(hist2) == 0


def test_history_to_decisions_latest_cycle_wins():
    hist = pd.DataFrame([
        {"cycle": 1, "timestamp": "t1", "account": "SA-1",
         "fraud_type": "F", "verdict": VERDICT_FP, "reviewer": "r"},
        {"cycle": 2, "timestamp": "t2", "account": "SA-1",
         "fraud_type": "F", "verdict": VERDICT_TP, "reviewer": "r"},
    ])
    decisions = history_to_decisions(hist)
    assert decisions[case_key("SA-1", "F")] == VERDICT_TP     # الأحدث يفوز


def test_append_feedback_cycle_writes_file_and_numbers_cycles(tmp_path):
    path = str(tmp_path / "feedback_memory.csv")
    empty = load_feedback_history(path)

    hist1, n1 = append_feedback_cycle(
        path, empty, {case_key("SA-1", "F"): VERDICT_FP}, "مراجع", "2026-07-10 09:00")
    assert n1 == 1 and list(hist1["cycle"]) == [1]
    assert pd.read_csv(path)["cycle"].tolist() == [1]          # كُتب فعلاً على القرص

    hist2, n2 = append_feedback_cycle(
        path, hist1, {case_key("SA-2", "G"): VERDICT_TP}, "مراجع", "2026-07-12 10:00")
    assert n2 == 1 and sorted(hist2["cycle"].unique().tolist()) == [1, 2]


def test_append_feedback_cycle_skips_unchanged_decisions(tmp_path):
    path = str(tmp_path / "feedback_memory.csv")
    hist1, _ = append_feedback_cycle(
        path, load_feedback_history(path),
        {case_key("SA-1", "F"): VERDICT_FP}, "مراجع", "t1")
    # نفس القرار بالضبط مرة أخرى — لا دورة جديدة ولا صف مكرّر
    hist2, n2 = append_feedback_cycle(
        path, hist1, {case_key("SA-1", "F"): VERDICT_FP}, "مراجع", "t2")
    assert n2 == 0 and len(hist2) == len(hist1)


def test_learning_curve_shows_cumulative_precision_gain():
    df = pd.DataFrame({
        "account_id": ["SA-1", "SA-1", "SA-2", "SA-3"],
        "fraud_type": ["انحراف سلوكي (Behavioral Deviation)"] * 2 +
                      ["تجزئة عمليات (Structuring)", "تكرار مشبوه (Velocity)"],
        "is_anomaly": [True, True, True, True],
        "is_anomaly_prefeedback": [True, True, True, True],
    })
    truth = pd.Series([False, False, True, True])   # SA-1 بريء، SA-2/SA-3 حقيقيان
    hist = pd.DataFrame([
        {"cycle": 1, "timestamp": "t1", "account": "SA-1",
         "fraud_type": "انحراف سلوكي (Behavioral Deviation)",
         "verdict": VERDICT_FP, "reviewer": "r"},
    ])
    curve = learning_curve(df, "account_id", truth, hist)
    assert list(curve["cycle"]) == [0, 1]
    assert curve.iloc[0]["precision"] == 50.0        # الدورة صفر: 2 صحيح / 4 = 50%
    assert curve.iloc[1]["precision"] == 100.0        # بعد الدورة 1: 2 صحيح / 2 = 100%
    assert curve.iloc[1]["applied_total"] == 1


def test_learning_curve_baseline_only_without_history():
    df = pd.DataFrame({
        "account_id": ["SA-2"], "fraud_type": ["تجزئة عمليات (Structuring)"],
        "is_anomaly": [True], "is_anomaly_prefeedback": [True],
    })
    truth = pd.Series([True])
    curve = learning_curve(df, "account_id", truth, load_feedback_history(None))
    assert len(curve) == 1 and curve.iloc[0]["cycle"] == 0 and curve.iloc[0]["precision"] == 100.0
