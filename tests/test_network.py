# tests/test_network.py
# اختبارات محرك تحليل الشبكات — بيانات مُصطنعة صغيرة، كل نمط على حدة
import pandas as pd
import pytest

from sigma_network import (
    detect_counterparty_column, run_network_analysis, apply_network_flags,
    NetworkFinding,
)


def _df(rows):
    """rows: (date, amount, account, counterparty, type)"""
    df = pd.DataFrame(rows, columns=["date", "amount", "account_id",
                                     "counterparty_id", "transaction_type"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _run(df):
    return run_network_analysis(df, "account_id", "counterparty_id",
                                "amount", "date", cat_col="transaction_type")


# ── كشف عمود الطرف المقابل ────────────────────────────────────────────────

def test_detects_counterparty_column_ar_and_en():
    assert detect_counterparty_column(
        pd.DataFrame(columns=["date", "counterparty_id"])) == "counterparty_id"
    assert detect_counterparty_column(
        pd.DataFrame(columns=["تاريخ", "الطرف المقابل"])) == "الطرف المقابل"
    assert detect_counterparty_column(pd.DataFrame(columns=["date", "amount"])) is None


# ── الحلقات المغلقة ──────────────────────────────────────────────────────

def test_cycle_detected_with_evidence():
    df = _df([
        ("2025-12-15 08:00", 95000, "SA-A", "SA-B", "تحويل داخلي"),
        ("2025-12-15 09:00", 94500, "SA-B", "SA-C", "تحويل داخلي"),
        ("2025-12-15 10:00", 94800, "SA-C", "SA-A", "تحويل داخلي"),
    ])
    res = _run(df)
    assert res["counts"]["cycle"] == 1
    f = res["findings"][0]
    assert f.pattern == "cycle"
    assert set(f.accounts) == {"SA-A", "SA-B", "SA-C"}
    assert sorted(f.txn_indices) == [0, 1, 2]
    assert len(f.evidence) == 4               # سطر ملخص + 3 عمليات
    assert "حلقة مغلقة" in f.evidence[0]


def test_cycle_rejected_when_amounts_differ():
    df = _df([
        ("2025-12-15 08:00", 95000, "SA-A", "SA-B", "تحويل داخلي"),
        ("2025-12-15 09:00", 20000, "SA-B", "SA-C", "تحويل داخلي"),   # ليس نفس المال
        ("2025-12-15 10:00", 94800, "SA-C", "SA-A", "تحويل داخلي"),
    ])
    assert _run(df)["counts"]["cycle"] == 0


def test_cycle_rejected_outside_time_window():
    df = _df([
        ("2025-12-01 08:00", 95000, "SA-A", "SA-B", "تحويل داخلي"),
        ("2025-12-10 09:00", 94500, "SA-B", "SA-C", "تحويل داخلي"),   # بعد أيام
        ("2025-12-20 10:00", 94800, "SA-C", "SA-A", "تحويل داخلي"),
    ])
    assert _run(df)["counts"]["cycle"] == 0


# ── شبكات التجميع ────────────────────────────────────────────────────────

def _mule_rows():
    rows = [(f"2025-12-18 {9 + i}:00", 6000, f"SA-S{i}", "SA-HUB", "تحويل داخلي")
            for i in range(5)]                       # 5 مرسلين × 6,000 = 30,000
    rows.append(("2025-12-19 10:00", 28000, "SA-HUB", "EXT-77", "حوالة دولية"))
    return rows


def test_mule_hub_detected_with_pass_ratio():
    res = _run(_df(_mule_rows()))
    assert res["counts"]["mule"] == 1
    f = [x for x in res["findings"] if x.pattern == "mule"][0]
    assert f.accounts[0] == "SA-HUB"
    assert f.total_amount == 30000
    assert len(f.txn_indices) == 6            # 5 واردة + 1 صادرة
    assert "حساب تجميع" in f.evidence[0]


def test_mule_rejected_without_outflow():
    rows = _mule_rows()[:-1]                  # تجميع بلا تمرير → ليس شبكة بغال
    assert _run(_df(rows))["counts"]["mule"] == 0


def test_mule_rejected_with_few_senders():
    rows = [(f"2025-12-18 {9 + i}:00", 12000, f"SA-S{i}", "SA-HUB", "تحويل داخلي")
            for i in range(3)]                # 3 مرسلين فقط < الحد الأدنى 4
    rows.append(("2025-12-19 10:00", 30000, "SA-HUB", "EXT-77", "حوالة دولية"))
    assert _run(_df(rows))["counts"]["mule"] == 0


# ── التمرير السريع ───────────────────────────────────────────────────────

def test_passthrough_detected():
    df = _df([
        ("2025-12-18 09:00", 20000, "SA-X", "SA-Y", "تحويل داخلي"),
        ("2025-12-18 15:00", 19000, "SA-Y", "EXT-01", "حوالة دولية"),
    ])
    res = _run(df)
    assert res["counts"]["passthrough"] == 1
    f = [x for x in res["findings"] if x.pattern == "passthrough"][0]
    assert f.accounts == ["SA-Y"]
    assert "حساب عبور" in f.evidence[0]


def test_passthrough_excludes_mule_accounts():
    res = _run(_df(_mule_rows()))             # الحساب المُجمِّع لا يُبلَّغ مرتين
    assert res["counts"]["mule"] == 1
    assert res["counts"]["passthrough"] == 0


# ── حالات الأمان والصدق ──────────────────────────────────────────────────

def test_returns_none_without_counterparty_column():
    df = _df([("2025-12-15 08:00", 95000, "SA-A", "SA-B", "تحويل داخلي")])
    assert run_network_analysis(df, "account_id", None, "amount", "date") is None


def test_cash_and_atm_are_not_edges():
    df = _df([
        ("2025-12-15 08:00", 9000, "SA-A", "CASH-DESK", "إيداع نقدي"),
        ("2025-12-15 09:00", 9000, "SA-A", "ATM-NETWORK", "سحب نقدي"),
    ])
    assert run_network_analysis(df, "account_id", "counterparty_id",
                                "amount", "date",
                                cat_col="transaction_type") is None


def test_apply_network_flags_marks_rows_and_keeps_hybrid_copy():
    df = _df([
        ("2025-12-15 08:00", 95000, "SA-A", "SA-B", "تحويل داخلي"),
        ("2025-12-15 09:00", 94500, "SA-B", "SA-C", "تحويل داخلي"),
        ("2025-12-15 10:00", 94800, "SA-C", "SA-A", "تحويل داخلي"),
    ])
    df["is_anomaly"] = [False, False, True]
    df["risk_score"] = [10.0, 20.0, 99.0]
    df["fraud_type"] = ["—", "—", "تجزئة عمليات (Structuring)"]
    out = apply_network_flags(df, _run(df)["findings"])

    assert out["is_anomaly"].all()                      # الحلقة رفعت الجميع
    assert list(out["is_anomaly_hybrid"]) == [False, False, True]
    assert out.loc[0, "risk_score"] == 95.0             # درجة النمط القاعدية
    assert out.loc[2, "risk_score"] == 99.0             # لا تخفيض لدرجة أعلى
    assert out.loc[0, "fraud_type"] == "دوران شبكي (Closed Loop)"
    assert out.loc[2, "fraud_type"] == "تجزئة عمليات (Structuring)"   # لا كتابة فوق الأدق
    assert (out["network_pattern"] == "دوران شبكي (Closed Loop)").all()
