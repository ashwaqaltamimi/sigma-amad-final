# sigma_feedback.py
# ============================================================================
#  حلقة التغذية الراجعة — Analyst Feedback Loop عبر الجلسات (Cross-Session)
#  منطق نقي بلا Streamlit (قابل للاختبار باستقلال).
#
#  الفكرة: مراجع الامتثال البشري يفحص كل «حالة تنبيه» (حساب + نمط اشتباه)
#  ويحكم: مؤكدة أو إنذار كاذب. الحكم يتغذى فوراً في النظام:
#    - «إنذار كاذب»  ← تُقفَل الحالة وتُستبعد عملياتها من التنبيهات،
#                       فترتفع الدقة (Precision) المقاسة حيّاً.
#    - «مؤكدة»       ← تُوسَم الحالة موثّقةً بمراجعة بشرية.
#
#  الحلقة تُغلَق فعلياً عبر الجلسات لا داخل جلسة واحدة فقط: القرارات
#  تُحفَظ في سجل تراكمي على القرص (كل صف = دورة مراجعة)، فتُطبَّق تلقائياً
#  من أول لحظة تحميل في أي جلسة لاحقة — النظام «يبدأ أذكى» من الدورة
#  السابقة دون أي تدخل. هذا مقصور على بيانات Demo فقط (انظر sigma_banking:
#  FEEDBACK_MEMORY_PATH) — لا صلة له بملف عمليات حقيقي مرفوع من مستخدم،
#  الذي يبقى داخل الجلسة فقط دون أي تخزين على القرص (المبدأ الحاكم محفوظ).
#
#  المبدأ الحاكم محفوظ أيضاً في طبيعة «التعلّم» نفسها: تعلّم خاضع للإشراف
#  بصيغته الأمينة — قرارات بشرية صريحة تُطبَّق كقواعد كبت/تأكيد تراكمية
#  قابلة للتدقيق وسجلها قابل للتصدير، لا «إعادة تدريب» خفية لا يمكن
#  تفسيرها أمام مراجع ساما.
# ============================================================================
from typing import Dict, List, Optional

import pandas as pd

VERDICT_FP  = "إنذار كاذب"
VERDICT_TP  = "مؤكدة"
REVIEW_NONE = "لم تُراجَع"

CLOSED_LABEL    = "أُقفلت بمراجعة بشرية (إنذار كاذب)"
CONFIRMED_LABEL = "مؤكدة بمراجعة بشرية"

# أعمدة سجل الذاكرة التراكمية على القرص — صف واحد لكل قرار في كل دورة
HISTORY_COLUMNS = ["cycle", "timestamp", "account", "fraud_type", "verdict", "reviewer"]


def case_key(account: str, fraud_type: str) -> str:
    """مفتاح حالة التنبيه: حساب + نمط اشتباه."""
    return f"{account}||{fraud_type}"


def build_review_cases(anomalies, account_col, amount_col) -> List[dict]:
    """
    يجمع العمليات المشبوهة إلى «حالات» قابلة للمراجعة (حساب × نمط) —
    وحدة العمل الفعلية لموظف الامتثال، لا العملية المفردة.
    """
    if anomalies is None or not len(anomalies) or not account_col \
            or account_col not in anomalies.columns:
        return []
    cases = []
    grp = anomalies.groupby([anomalies[account_col].astype(str), "fraud_type"])
    for (acct, ftype), g in grp:
        cases.append({
            "key": case_key(acct, ftype), "account": acct, "fraud_type": ftype,
            "txn_count": len(g),
            "total_amount": float(g[amount_col].abs().sum()),
            "max_risk": float(g["risk_score"].max()),
        })
    cases.sort(key=lambda c: c["max_risk"], reverse=True)
    return cases


def apply_feedback(df, decisions: Dict[str, str], account_col):
    """
    يُطبّق قرارات المراجعة على إطار النتائج:
      - الحالات المحكومة «إنذار كاذب»: is_anomaly=False + وسم الإقفال —
        تخرج من التنبيهات والبلاغات والمقاييس فوراً.
      - الحالات المحكومة «مؤكدة»: تبقى مع وسم التوثيق البشري.
    يحفظ `is_anomaly_prefeedback` (نسخة قبل أي تطبيق) — أساس صادق لمنحنى
    التعلّم (الدورة صفر = بلا أي تغذية راجعة إطلاقاً)، بنفس نمط
    is_anomaly_if / is_anomaly_hybrid في sigma_banking.
    يُعيد (df، عدد المُقفلة، عدد المؤكدة).
    """
    out = df.copy()
    out["is_anomaly_prefeedback"] = out["is_anomaly"].copy()
    out["review_status"] = REVIEW_NONE
    closed = confirmed = 0
    if not decisions or not account_col or account_col not in out.columns:
        return out, closed, confirmed

    keys = out[account_col].astype(str) + "||" + out["fraud_type"].astype(str)
    for key, verdict in decisions.items():
        mask = (keys == key) & out["is_anomaly"]
        n = int(mask.sum())
        if not n:
            continue
        if verdict == VERDICT_FP:
            out.loc[mask, "is_anomaly"] = False
            out.loc[mask, "review_status"] = CLOSED_LABEL
            closed += 1
        elif verdict == VERDICT_TP:
            out.loc[mask, "review_status"] = CONFIRMED_LABEL
            confirmed += 1
    return out, closed, confirmed


def feedback_log(decisions: Dict[str, str]) -> pd.DataFrame:
    """سجل قرارات المراجعة للتصدير — أثر تدقيقي كامل (من/ماذا/الحكم)."""
    rows = []
    for key, verdict in sorted(decisions.items()):
        acct, ftype = key.split("||", 1)
        rows.append({"الحساب": acct, "نمط الاشتباه": ftype, "قرار المراجع": verdict})
    return pd.DataFrame(rows)


# ============================================================================
#  الذاكرة التراكمية على القرص — الحلقة تُغلَق عبر الجلسات
# ============================================================================

def load_feedback_history(path: Optional[str]) -> pd.DataFrame:
    """
    يقرأ سجل التغذية الراجعة التراكمي من القرص — إطاراً فارغاً بصدق إن
    غاب المسار أو الملف (أول تشغيل قط، أو وضع غير Demo).
    """
    if not path:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    try:
        return pd.read_csv(path, dtype={"cycle": int})
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def history_to_decisions(history: pd.DataFrame) -> Dict[str, str]:
    """آخر قرار مسجَّل لكل حالة عبر كامل السجل التراكمي (الدورة الأحدث تفوز)."""
    decisions = {}
    if history is None or not len(history):
        return decisions
    for _, r in history.sort_values("cycle").iterrows():
        decisions[case_key(str(r["account"]), str(r["fraud_type"]))] = r["verdict"]
    return decisions


def append_feedback_cycle(path: Optional[str], history: pd.DataFrame,
                          new_decisions: Dict[str, str], reviewer: str,
                          timestamp: str):
    """
    يضيف دورة مراجعة جديدة إلى الذاكرة التراكمية — فقط القرارات المتغيّرة
    فعلياً عمّا هو مسجَّل بالفعل (لا صف مكرّر بلا تغيير حقيقي، ولا دورة
    فارغة). يكتب الملف إن توفّر مسار. يُعيد (السجل المحدَّث، عدد الصفوف
    المضافة).
    """
    prev = history_to_decisions(history)
    changed = {k: v for k, v in new_decisions.items() if prev.get(k) != v}
    if not changed:
        return history, 0

    next_cycle = int(history["cycle"].max()) + 1 if len(history) else 1
    rows = [{"cycle": next_cycle, "timestamp": timestamp,
             "account": k.split("||", 1)[0], "fraud_type": k.split("||", 1)[1],
             "verdict": v, "reviewer": reviewer}
            for k, v in changed.items()]
    new_rows = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    updated = pd.concat([history, new_rows], ignore_index=True)
    if path:
        updated.to_csv(path, index=False, encoding="utf-8-sig")
    return updated, len(new_rows)


def learning_curve(df_full, account_col, truth_mask, history: pd.DataFrame) -> pd.DataFrame:
    """
    يعيد أداء الكشف (دقة/استدعاء/F1) عند كل دورة مراجعة تراكمية، مقارنةً
    بحقيقة أرضية معروفة truth_mask — الدورة صفر = الأداء الخام بلا أي
    تغذية راجعة إطلاقاً (df_full["is_anomaly_prefeedback"] إن توفّر،
    وإلا is_anomaly كما هو). يُستخدَم فقط حين تتوفر حقيقة أرضية حقيقية
    (بيانات Demo) — لا رقم أداء بلا أساس.
    """
    base_col = ("is_anomaly_prefeedback" if "is_anomaly_prefeedback" in df_full.columns
                else "is_anomaly")
    keys_full = (df_full[account_col].astype(str) + "||"
                 + df_full["fraud_type"].astype(str))

    def _metrics(pred):
        pred = pred.astype(bool)
        tp = int((pred & truth_mask).sum()); fp = int((pred & ~truth_mask).sum())
        fn = int((~pred & truth_mask).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return round(p * 100, 1), round(r * 100, 1), round(f1 * 100, 1)

    p0, r0, f0 = _metrics(df_full[base_col])
    rows = [{"cycle": 0, "applied_total": 0, "precision": p0, "recall": r0, "f1": f0}]

    if history is not None and len(history):
        cum_fp, applied = set(), 0
        for cyc in sorted(history["cycle"].unique()):
            batch = history[history["cycle"] == cyc]
            applied += len(batch)
            cum_fp |= {case_key(str(r["account"]), str(r["fraud_type"]))
                       for _, r in batch.iterrows() if r["verdict"] == VERDICT_FP}
            pred = df_full[base_col] & (~keys_full.isin(cum_fp))
            p, r, f1 = _metrics(pred)
            rows.append({"cycle": int(cyc), "applied_total": applied,
                        "precision": p, "recall": r, "f1": f1})
    return pd.DataFrame(rows)
