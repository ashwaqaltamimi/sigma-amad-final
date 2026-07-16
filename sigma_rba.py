# sigma_rba.py
# ============================================================================
#  النهج القائم على المخاطر — Risk-Based Approach (RBA)
#  منطق نقي بلا Streamlit (قابل للاختبار باستقلال).
#
#  الأساس النظامي: المادة 5 من نظام مكافحة غسل الأموال (م/20 · 1439هـ) —
#  «تحديد مخاطر غسل الأموال وتقييمها وفهمها... واتخاذ تدابير العناية الواجبة
#  على أساس المخاطر» — وقواعد ساما التي توجب تصنيف العملاء وتشديد الرقابة
#  على الفئات الأعلى خطراً.
#
#  المبدأ الحاكم محفوظ: التصنيف بجدول عوامل مُعلن الأوزان (لا صندوق أسود)،
#  وكل درجة تحمل «أسباباً» مقروءة تُعرض للمدقق. التصنيف يعني رقابة أشد —
#  لا اتهاماً: في بيانات الديمو عملاء بملفات عالية المخاطر وسلوك نظيف تماماً.
# ============================================================================
from typing import Optional

import pandas as pd

# ── جدول عوامل الخطورة — أوزان مُعلنة قابلة للدفاع (مستمدة من منهجية FATF) ──
W_FATF        = 35   # ارتباط بولاية قضائية عالية المخاطر وفق قوائم FATF
W_PEP         = 40   # شخصية سياسية معرّضة (PEP) — عناية واجبة معززة إلزامية
W_CASH_OCC    = 25   # نشاط كثيف التعامل النقدي
W_NEW_ACCOUNT = 20   # حساب حديث (< 6 أشهر) — لم يتكوّن له نمط سلوكي بعد
W_NONRESIDENT = 10   # غير مقيم — صعوبة تتبع أعلى

CASH_INTENSIVE_OCC = ("تاجر تجزئة نقدي", "تجارة ذهب ومجوهرات", "مطاعم ومقاهٍ",
                      "تاجر جملة", "صرافة", "مستورد ومصدّر")
NEW_ACCOUNT_MONTHS = 6

TIER_HIGH, TIER_MED = 60, 30           # عتبتا التصنيف (عالي ≥60 · متوسط ≥30)
TIERS = ("عالي", "متوسط", "منخفض")

# عتبة قاعدة التكرار (Velocity) تتشدد تلقائياً مع ارتفاع الخطورة:
# العميل عالي المخاطر يُراجع عند 3 عمليات/24س بدل 5 — هذا جوهر الـ RBA.
VELOCITY_BY_TIER = {"عالي": 3, "متوسط": 4, "منخفض": 5}
VELOCITY_DEFAULT = 5

# معامل رفع مُعلن لدرجة خطر العمليات المشبوهة من عملاء عاليي المخاطر
RBA_SCORE_BOOST = 10.0


def detect_profiles_columns(df) -> bool:
    """يتحقق أن إطار الملفات يحمل الأعمدة المطلوبة للتصنيف."""
    need = {"account_id", "occupation", "fatf_jurisdiction", "pep",
            "account_age_months"}
    return need.issubset(set(map(str, df.columns)))


def rate_customers(profiles: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    يُصنّف كل عميل (منخفض/متوسط/عالي) بجمع أوزان عوامله المُعلنة.
    يعيد DataFrame: account_id · rba_score · rba_tier · rba_reasons —
    الأسباب نص مقروء يُعرض للمدقق (سلسلة أدلة التصنيف نفسه).
    """
    if profiles is None or not detect_profiles_columns(profiles):
        return None

    out = []
    for _, r in profiles.iterrows():
        score, reasons = 0, []
        if int(r.get("fatf_jurisdiction", 0)):
            score += W_FATF
            reasons.append(f"ولاية قضائية عالية المخاطر وفق FATF (+{W_FATF})")
        if int(r.get("pep", 0)):
            score += W_PEP
            reasons.append(f"شخصية سياسية معرّضة PEP (+{W_PEP})")
        if str(r.get("occupation", "")) in CASH_INTENSIVE_OCC:
            score += W_CASH_OCC
            reasons.append(f"نشاط كثيف النقد: {r['occupation']} (+{W_CASH_OCC})")
        if float(r.get("account_age_months", 999)) < NEW_ACCOUNT_MONTHS:
            score += W_NEW_ACCOUNT
            reasons.append(f"حساب حديث ({int(r['account_age_months'])} شهر) (+{W_NEW_ACCOUNT})")
        if str(r.get("residency", "")) == "غير مقيم":
            score += W_NONRESIDENT
            reasons.append(f"غير مقيم (+{W_NONRESIDENT})")

        tier = ("عالي" if score >= TIER_HIGH
                else "متوسط" if score >= TIER_MED else "منخفض")
        out.append({"account_id": str(r["account_id"]), "rba_score": score,
                    "rba_tier": tier,
                    "rba_reasons": " · ".join(reasons) or "لا عوامل خطورة مرتفعة"})
    return pd.DataFrame(out)


def tier_map(ratings: Optional[pd.DataFrame]) -> dict:
    """قاموس {حساب: تصنيف} — فارغ إذا لا تصنيفات (بيانات بلا ملفات KYC)."""
    if ratings is None or not len(ratings):
        return {}
    return dict(zip(ratings["account_id"].astype(str), ratings["rba_tier"]))


def velocity_thresholds(df, account_col, tiers: dict) -> pd.Series:
    """
    عتبة التكرار لكل صف حسب تصنيف صاحب الحساب — العميل الأعلى خطراً
    يُراجَع عند عدد عمليات أقل. الحسابات بلا تصنيف تأخذ العتبة الافتراضية.
    """
    if not account_col or account_col not in df.columns or not tiers:
        return pd.Series(VELOCITY_DEFAULT, index=df.index)
    return (df[account_col].astype(str)
            .map(lambda a: VELOCITY_BY_TIER.get(tiers.get(a), VELOCITY_DEFAULT)))


def boost_anomaly_scores(out_df, account_col, tiers: dict):
    """
    يرفع درجة خطر العمليات المشبوهة لعملاء التصنيف «عالي» بمعامل RBA
    المُعلن (+10، سقف 100) ويكتب عمود rba_tier للعرض في سجل التدقيق.
    لا يمسّ عمليات العملاء الآخرين ولا العمليات السليمة.
    """
    out = out_df.copy()
    if not account_col or account_col not in out.columns or not tiers:
        out["rba_tier"] = "—"
        return out
    out["rba_tier"] = out[account_col].astype(str).map(
        lambda a: tiers.get(a, "—"))
    hot = out["is_anomaly"] & (out["rba_tier"] == "عالي")
    out.loc[hot, "risk_score"] = (out.loc[hot, "risk_score"]
                                  + RBA_SCORE_BOOST).clip(upper=100.0)
    return out
