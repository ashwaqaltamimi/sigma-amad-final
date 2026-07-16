# sigma_network.py
# ============================================================================
#  محرك تحليل شبكات غسيل الأموال — Graph Link Analysis
#  منطق نقي بلا Streamlit (قابل للاختبار باستقلال، مثل sigma_agents).
#
#  يبني رسماً موجّهاً من عمود «الطرف المقابل» ويكشف ثلاثة أنماط معيارية
#  لا يستطيع الكشف الإحصائي (صفاً-بصف) رؤيتها لأنها علاقات بين حسابات:
#    1. دوران شبكي  (Closed Loop)   — حلقة مغلقة A→B→C→A خلال نافذة قصيرة.
#    2. شبكة تجميع  (Mule Network)  — مرسلون متعددون → حساب تجميع → تمرير للخارج.
#    3. تمرير سريع  (Pass-through)  — دخول مبلغ ثم خروج معظمه خلال ساعات.
#
#  المبدأ الحاكم محفوظ: كل اكتشاف يحمل «سلسلة أدلة» — قائمة العمليات الفعلية
#  (فهرس، تاريخ، مبلغ) التي بُني عليها القرار، ودرجة الخطورة قاعدية مُعلنة
#  (ثابتة لكل نمط، موثّقة أدناه) — لا أرقام مُختلقة.
# ============================================================================
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

# ── عتبات الكشف الشبكي — حسابية وقابلة للدفاع ────────────────────────────────
CYCLE_WINDOW_H     = 72     # نافذة الحلقة المغلقة (ساعة)
CYCLE_AMT_TOL      = 0.25   # تقارب مبالغ الحلقة ±25%
CYCLE_MAX_LEN      = 4      # أقصى طول حلقة (قفزات)
MULE_MIN_SENDERS   = 4      # حد أدنى لعدد المرسلين المختلفين لحساب التجميع
MULE_WINDOW_DAYS   = 7      # نافذة التجميع (يوم)
MULE_MIN_TOTAL     = 20_000 # حد أدنى لإجمالي المبالغ المجمّعة (ريال)
MULE_OUT_RATIO     = 0.50   # نسبة التمرير الدنيا من المُجمّع
PASS_WINDOW_H      = 72     # نافذة التمرير السريع (ساعة)
PASS_MIN_TOTAL     = 15_000 # حد أدنى لمبلغ التمرير (ريال)
PASS_OUT_RATIO     = 0.70   # نسبة الخروج الدنيا من الداخل

# درجات خطورة قاعدية مُعلنة لكل نمط (rule-based severity — ليست "ثقة" مُختلقة):
# الحلقة المغلقة أخطر الأنماط (تدوير متعمّد)، ثم التجميع، ثم التمرير.
SEVERITY = {"cycle": 95.0, "mule": 90.0, "passthrough": 82.0}

PATTERN_LABELS = {
    "cycle":       "دوران شبكي (Closed Loop)",
    "mule":        "شبكة تجميع وتمرير (Mule Network)",
    "passthrough": "تمرير سريع (Rapid Pass-through)",
}

PATTERN_SAMA = {
    "cycle":       "المادة 15 · نظام مكافحة غسل الأموال (م/20) — إبلاغ فوري عن تدوير الأموال",
    "mule":        "المادة 15 · إبلاغ فوري للتحريات المالية + عناية واجبة معززة (EDD)",
    "passthrough": "القسم 7 · قواعد ساما — مراقبة العمليات غير الاعتيادية (حساب عبور)",
}


@dataclass
class NetworkFinding:
    pattern: str                 # cycle | mule | passthrough
    label: str                   # الاسم العربي للنمط
    accounts: List[str]          # الحسابات المتورطة (مرتّبة حسب دور كل حساب)
    txn_indices: List[int]       # فهارس العمليات الداعمة في DataFrame الأصلي
    total_amount: float          # إجمالي المبالغ المتورطة
    window_hours: float          # المدى الزمني للنمط
    evidence: List[str] = field(default_factory=list)  # سلسلة الأدلة (سطور مقروءة)
    severity: float = 0.0        # درجة الخطورة القاعدية للنمط
    sama_ref: str = ""           # المرجع التنظيمي


# ── أدوات مساعدة ─────────────────────────────────────────────────────────────

def detect_counterparty_column(df) -> Optional[str]:
    """يكتشف عمود الطرف المقابل بمرونة (عربي/إنجليزي) — أو يعيد None بصدق."""
    for col in df.columns:
        cl = str(col).lower()
        if any(w in cl for w in ["counterparty", "beneficiary", "receiver",
                                 "to_account", "طرف", "مستفيد", "مستلم"]):
            return col
    return None


def _transfer_edges(df, account_col, cp_col, amount_col, date_col, cat_col=None):
    """
    يستخرج حواف التحويل (مرسل → مستقبل) من صفوف التحويلات/الحوالات فقط.
    الإيداعات والسحوبات وأطراف الكاش (CASH/ATM) ليست حوافاً بين حسابات.
    """
    frame = df
    if cat_col and cat_col in df.columns:
        is_tr = df[cat_col].astype(str).str.contains(
            "تحويل|حوالة|سريع|transfer|sarie", case=False, na=False)
        frame = df[is_tr]
    edges = []
    for idx, row in frame.iterrows():
        snd, rcv = str(row[account_col]), str(row[cp_col])
        if not snd or not rcv or rcv.upper() in ("CASH-DESK", "ATM-NETWORK", "NAN"):
            continue
        edges.append({"idx": idx, "sender": snd, "receiver": rcv,
                      "amount": abs(float(row[amount_col])), "ts": row[date_col]})
    return edges


def _evidence_line(e) -> str:
    return f"[{e['ts']:%Y-%m-%d %H:%M}] {e['sender']} ← {e['amount']:,.0f} ريال → {e['receiver']}"


# ── الكاشف 1: الحلقات المغلقة (Closed Loops) ────────────────────────────────

def find_cycles(edges) -> List[NetworkFinding]:
    """
    DFS على الرسم الموجّه لاكتشاف الحلقات البسيطة بطول 2..CYCLE_MAX_LEN.
    تُقبل الحلقة فقط إذا وقعت عملياتها داخل نافذة CYCLE_WINDOW_H
    وكانت مبالغها متقاربة (±CYCLE_AMT_TOL) — أي تدوير فعلي لنفس الأموال.
    """
    adj = {}
    for e in edges:
        adj.setdefault(e["sender"], set()).add(e["receiver"])

    cycles_seen, findings = set(), []

    def dfs(start, node, path):
        for nxt in sorted(adj.get(node, ())):
            if nxt == start and len(path) >= 2:
                key = frozenset(path)
                if key not in cycles_seen:
                    cycles_seen.add(key)
                    _validate_cycle(list(path), edges, findings)
            elif nxt not in path and len(path) < CYCLE_MAX_LEN:
                dfs(start, nxt, path + [nxt])

    for start in sorted(adj):
        dfs(start, start, [start])
    return findings


def _validate_cycle(accounts, edges, findings):
    """يتحقق زمنياً ومالياً من حلقة مرشحة ويبني سلسلة الأدلة إن صحّت."""
    hops = list(zip(accounts, accounts[1:] + accounts[:1]))
    hop_edges = []
    for snd, rcv in hops:
        cand = [e for e in edges if e["sender"] == snd and e["receiver"] == rcv]
        if not cand:
            return
        hop_edges.extend(cand)

    ts = [e["ts"] for e in hop_edges]
    window_h = (max(ts) - min(ts)).total_seconds() / 3600
    amts = [e["amount"] for e in hop_edges]
    if window_h > CYCLE_WINDOW_H or min(amts) < max(amts) * (1 - CYCLE_AMT_TOL):
        return

    hop_edges.sort(key=lambda e: e["ts"])
    total = sum(amts)
    findings.append(NetworkFinding(
        pattern="cycle", label=PATTERN_LABELS["cycle"],
        accounts=accounts, txn_indices=[e["idx"] for e in hop_edges],
        total_amount=total, window_hours=round(window_h, 1),
        evidence=[f"حلقة مغلقة: {' → '.join(accounts + [accounts[0]])} "
                  f"خلال {window_h:.1f} ساعة"]
                 + [_evidence_line(e) for e in hop_edges],
        severity=SEVERITY["cycle"], sama_ref=PATTERN_SAMA["cycle"]))


# ── الكاشف 2: شبكات التجميع والتمرير (Mule Networks) ────────────────────────

def find_mule_hubs(edges) -> List[NetworkFinding]:
    """
    حساب تجميع = يستقبل من MULE_MIN_SENDERS+ مرسلين مختلفين خلال
    MULE_WINDOW_DAYS بإجمالي ≥ MULE_MIN_TOTAL، ثم يُمرّر ≥ MULE_OUT_RATIO
    من المُجمّع خلال PASS_WINDOW_H من آخر وارد. الشرطان معاً (تجميع + تمرير)
    هما ما يميّز شبكة البغال عن حساب نشِط بريء — دفاعاً عن الدقة (Precision).
    """
    inflows = {}
    for e in edges:
        inflows.setdefault(e["receiver"], []).append(e)

    findings = []
    for hub in sorted(inflows):
        ins = sorted(inflows[hub], key=lambda e: e["ts"])
        senders = {e["sender"] for e in ins}
        if len(senders) < MULE_MIN_SENDERS:
            continue
        span_days = (ins[-1]["ts"] - ins[0]["ts"]).total_seconds() / 86400
        total_in = sum(e["amount"] for e in ins)
        if span_days > MULE_WINDOW_DAYS or total_in < MULE_MIN_TOTAL:
            continue
        # التمرير: خوارج الحساب المُجمِّع بعد بدء التجميع وحتى 72 ساعة من آخر وارد
        deadline = ins[-1]["ts"] + pd.Timedelta(hours=PASS_WINDOW_H)
        outs = sorted((e for e in edges
                       if e["sender"] == hub and ins[0]["ts"] <= e["ts"] <= deadline),
                      key=lambda e: e["ts"])
        total_out = sum(e["amount"] for e in outs)
        if total_out < MULE_OUT_RATIO * total_in:
            continue

        window_h = ((outs[-1]["ts"] if outs else ins[-1]["ts"]) - ins[0]["ts"]
                    ).total_seconds() / 3600
        findings.append(NetworkFinding(
            pattern="mule", label=PATTERN_LABELS["mule"],
            accounts=[hub] + sorted(senders),
            txn_indices=[e["idx"] for e in ins + outs],
            total_amount=total_in, window_hours=round(window_h, 1),
            evidence=[f"حساب تجميع {hub}: استقبل {total_in:,.0f} ريال من "
                      f"{len(senders)} مرسلاً خلال {span_days:.1f} يوم، "
                      f"ثم مرّر {total_out:,.0f} ريال "
                      f"({total_out / total_in * 100:.0f}%) للخارج"]
                     + [_evidence_line(e) for e in ins + outs],
            severity=SEVERITY["mule"], sama_ref=PATTERN_SAMA["mule"]))
    return findings


# ── الكاشف 3: التمرير السريع (Rapid Pass-through) ───────────────────────────

def find_passthrough(edges, exclude_accounts=()) -> List[NetworkFinding]:
    """
    حساب عبور = دخل عليه ≥ PASS_MIN_TOTAL ثم خرج ≥ PASS_OUT_RATIO منه
    خلال PASS_WINDOW_H. الحسابات المكتشفة في نمط أعمق (mule) تُستبعد
    لتفادي ازدواج التنبيه على نفس السلوك.
    """
    inflows, outflows = {}, {}
    for e in edges:
        inflows.setdefault(e["receiver"], []).append(e)
        outflows.setdefault(e["sender"], []).append(e)

    findings = []
    for acct in sorted(set(inflows) & set(outflows) - set(exclude_accounts)):
        ins = sorted(inflows[acct], key=lambda e: e["ts"])
        total_in = sum(e["amount"] for e in ins)
        if total_in < PASS_MIN_TOTAL:
            continue
        deadline = ins[-1]["ts"] + pd.Timedelta(hours=PASS_WINDOW_H)
        outs = sorted((e for e in outflows[acct]
                       if ins[0]["ts"] <= e["ts"] <= deadline),
                      key=lambda e: e["ts"])
        total_out = sum(e["amount"] for e in outs)
        if total_out < PASS_OUT_RATIO * total_in:
            continue

        window_h = (outs[-1]["ts"] - ins[0]["ts"]).total_seconds() / 3600
        findings.append(NetworkFinding(
            pattern="passthrough", label=PATTERN_LABELS["passthrough"],
            accounts=[acct], txn_indices=[e["idx"] for e in ins + outs],
            total_amount=total_in, window_hours=round(window_h, 1),
            evidence=[f"حساب عبور {acct}: دخل {total_in:,.0f} ريال وخرج "
                      f"{total_out:,.0f} ريال ({total_out / total_in * 100:.0f}%) "
                      f"خلال {window_h:.1f} ساعة"]
                     + [_evidence_line(e) for e in ins + outs],
            severity=SEVERITY["passthrough"], sama_ref=PATTERN_SAMA["passthrough"]))
    return findings


# ── نقطة الدخول ──────────────────────────────────────────────────────────────

def run_network_analysis(df, account_col, cp_col, amount_col, date_col,
                         cat_col=None) -> Optional[dict]:
    """
    يبني الرسم الشبكي ويشغّل الكواشف الثلاثة بالترتيب (الأعمق أولاً).
    يعيد None بصدق إذا غاب أي عمود مطلوب — لا تحليل بلا بيانات علاقات.
    """
    for col in (account_col, cp_col, amount_col, date_col):
        if not col or col not in df.columns:
            return None

    edges = _transfer_edges(df, account_col, cp_col, amount_col, date_col, cat_col)
    if not edges:
        return None

    cycles = find_cycles(edges)
    mules = find_mule_hubs(edges)
    covered = {a for f in cycles + mules for a in f.accounts}
    passthru = find_passthrough(edges, exclude_accounts=covered)

    findings = cycles + mules + passthru
    accounts = {e["sender"] for e in edges} | {e["receiver"] for e in edges}
    return {"findings": findings, "n_edges": len(edges),
            "n_accounts": len(accounts),
            "counts": {"cycle": len(cycles), "mule": len(mules),
                       "passthrough": len(passthru)},
            # أسماء الأعمدة تُحفظ ليستطيع العارض البصري إعادة بناء الحواف
            "account_col": account_col, "cp_col": cp_col}


def apply_network_flags(df, findings):
    """
    يدمج الاكتشافات الشبكية في نتائج الكاشف الهجين:
      - يحفظ نسخة `is_anomaly_hybrid` قبل الدمج (للمقارنة الصادقة في لوحة الأداء)
      - يرفع is_anomaly للعمليات الداعمة ويمنحها درجة الخطورة القاعدية للنمط
      - يضبط fraud_type للنمط الشبكي دون الكتابة فوق تصنيف قاعدي أدق (تجزئة/تكرار)
    """
    out = df.copy()
    out["is_anomaly_hybrid"] = out["is_anomaly"].copy()
    out["network_pattern"] = ""
    generic = {"—", "انحراف سلوكي (Behavioral Deviation)",
               "دوران مشبوه (Round-trip Proxy)"}
    for f in findings:
        idx = [i for i in f.txn_indices if i in out.index]
        out.loc[idx, "is_anomaly"] = True
        out.loc[idx, "network_pattern"] = f.label
        low = out.loc[idx, "risk_score"] < f.severity
        out.loc[[i for i, v in low.items() if v], "risk_score"] = f.severity
        for i in idx:
            if out.at[i, "fraud_type"] in generic:
                out.at[i, "fraud_type"] = f.label
    return out
