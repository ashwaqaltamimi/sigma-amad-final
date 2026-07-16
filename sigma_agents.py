# sigma_agents.py
# ============================================================================
#  طبقة الوكلاء الحتمية — منطق نقي بلا Streamlit (قابل للاختبار باستقلال)
#  كل رقم يُنتجه الوكلاء مقروء من قيم محسوبة مسبقاً — لا إعادة حساب ولا اختلاق.
# ============================================================================
from dataclasses import dataclass, field
from typing import Optional, List, Any


@dataclass
class AgentReport:
    agent_id: str
    persona: str
    role: str
    icon: str
    perceived: str
    decided: str
    key_number: str
    confidence: Optional[float]
    rule: str
    handoff_to: Optional[str]
    lines: List[str]
    evidence: List[str] = field(default_factory=list)   # سلسلة الأدلة القابلة للفحص


@dataclass
class Blackboard:
    ctx: dict
    anomalies: Any            # pandas DataFrame
    forecast: Optional[dict]
    flow: dict
    sama: dict                # {"ctr_threshold", "alert_for", "str_deadline"}
    amount_col: str
    date_col: str
    currency: str
    network: Optional[dict] = None   # ناتج run_network_analysis أو None بصدق
    lines: List[str] = field(default_factory=list)
    reports: List[AgentReport] = field(default_factory=list)

    def post(self, who: str, text: str) -> str:
        line = f"{who}: {text}"
        self.lines.append(line)
        return line


def agent_detection(bb: Blackboard) -> AgentReport:
    ctx, a = bb.ctx, bb.anomalies
    n = ctx["n"]
    # محمد يُبلّغ عمّا كشفته طبقتاه (الإحصاء + القواعد) فقط — الأنماط الشبكية
    # تخص سارة؛ كل وكيل يتحدث عن اكتشافاته هو (لا ازدواج في الأرقام)
    own = (ctx["struct"] + ctx["velocity"] + ctx["roundtrip"] + ctx["behavioral"])
    conf = round(float(a["risk_score"].mean()), 1) if (own > 0 and len(a)) else 0.0
    breakdown = (f"{ctx['struct']} تجزئة · {ctx['velocity']} تكرار · "
                 f"{ctx['roundtrip']} دوران · {ctx['behavioral']} سلوكي")
    decided = (f"رصد {own} عملية مشبوهة ({breakdown}) — والتحليل الشبكي مع سارة"
               if own > 0 else "لم يُرصد شذوذ إحصائي/قاعدي في هذه البيانات")
    lines = [bb.post("محمد", f"فحص {n:,} عملية → رصد {own} مشبوهة"),
             bb.post("محمد", f"التوزيع: {breakdown}")]
    # سلسلة الأدلة: أعلى 5 عمليات خطراً بأرقامها الفعلية من سجل التدقيق
    evidence = []
    if own > 0 and len(a):
        top = a.sort_values("risk_score", ascending=False).head(5)
        for _, r in top.iterrows():
            evidence.append(
                f"[{r[bb.date_col]:%Y-%m-%d %H:%M}] {r[bb.amount_col]:,.0f} ريال "
                f"· خطر {r['risk_score']:.0f} · {r['fraud_type']}")
    rep = AgentReport(
        agent_id="mohammed", persona="محمد",
        role="محلل مكافحة الاحتيال", icon="🕵️",
        perceived=f"فحص {n:,} عملية", decided=decided,
        key_number=f"{own} / {n:,}", confidence=conf,
        rule="IsolationForest + قواعد AML (تجزئة/تكرار/دوران)",
        handoff_to="سارة", lines=lines, evidence=evidence)
    bb.reports.append(rep)
    return rep


def agent_network(bb: Blackboard) -> AgentReport:
    """سارة — محللة شبكات غسيل الأموال: تقرأ ناتج التحليل الشبكي المحسوب مسبقاً."""
    net = bb.network
    base = dict(agent_id="sara", persona="سارة",
                role="محللة شبكات غسيل الأموال", icon="🕸️", handoff_to="نورة",
                rule="Graph Link Analysis — حلقات مغلقة / تجميع / تمرير (DFS)")
    if not net:
        decided = ("التحليل الشبكي غير مفعّل — لا يتوفر عمود «الطرف المقابل» "
                   "في هذه البيانات")
        rep = AgentReport(**base, perceived="البحث عن عمود الطرف المقابل",
                          decided=decided, key_number="—", confidence=None,
                          lines=[bb.post("سارة", decided)])
        bb.reports.append(rep)
        return rep

    findings, counts = net["findings"], net["counts"]
    perceived = f"رسم شبكي: {net['n_accounts']} حساباً · {net['n_edges']} تحويلاً"
    if findings:
        accounts = sorted({acc for f in findings for acc in f.accounts})
        total = sum(f.total_amount for f in findings)
        breakdown = (f"{counts['cycle']} حلقة مغلقة · {counts['mule']} شبكة تجميع · "
                     f"{counts['passthrough']} تمرير سريع")
        decided = (f"رصدت {len(findings)} نمطاً شبكياً ({breakdown}) "
                   f"يتورط فيه {len(accounts)} حساباً بمبالغ {total:,.0f} {bb.currency}")
        key_number = f"{len(findings)} نمط · {len(accounts)} حساب"
    else:
        decided = "فحصت الرسم الشبكي كاملاً — لا حلقات ولا شبكات تجميع ولا تمرير"
        key_number = "0 نمط شبكي"
    lines = [bb.post("سارة", f"بنيت الرسم: {perceived}"),
             bb.post("سارة", decided)]
    evidence = [ln for f in findings for ln in f.evidence]
    rep = AgentReport(**base, perceived=perceived, decided=decided,
                      key_number=key_number, confidence=None,
                      lines=lines, evidence=evidence)
    bb.reports.append(rep)
    return rep


def build_str_records(bb: Blackboard) -> List[dict]:
    """
    يبني سجلات مسودة بلاغ STR (حقول مستوحاة من نموذج goAML) — بلاغ واحد
    لكل حالة (حساب تجزئة أو نمط شبكي)، كل حقل مقروء من البيانات الفعلية.
    """
    records = []
    a = bb.anomalies
    # حالات التجزئة: بلاغ لكل حساب مُجزِّئ (لا لكل عملية)
    if len(a) and "fraud_type" in a.columns:
        struct_rows = a[a["fraud_type"] == "تجزئة عمليات (Structuring)"]
        acct_col = next((c for c in ("account_id", "الحساب", "account")
                         if c in struct_rows.columns), None)
        if acct_col and len(struct_rows):
            groups = struct_rows.groupby(acct_col)
        elif len(struct_rows):
            # لا عمود حساب في البيانات → حالة واحدة تجمع كل عمليات التجزئة
            groups = [("غير محدد", struct_rows)]
        else:
            groups = []
        for acct, g in groups:
            records.append({
                "pattern": "تجزئة عمليات (Structuring)",
                "accounts": [str(acct)],
                "txn_count": len(g),
                "total_amount": float(g[bb.amount_col].abs().sum()),
                "period": f"{g[bb.date_col].min():%Y-%m-%d} — {g[bb.date_col].max():%Y-%m-%d}",
                "sama_ref": "المادة 15 · نظام مكافحة غسل الأموال (م/20 · 1439هـ)",
                "deadline": str(bb.sama["str_deadline"](g[bb.date_col].max(), 3)),
                "evidence": [f"[{r[bb.date_col]:%Y-%m-%d %H:%M}] "
                             f"{r[bb.amount_col]:,.0f} ريال"
                             for _, r in g.iterrows()],
            })
    # الأنماط الشبكية: الحلقات وشبكات التجميع تستوجب STR
    if bb.network:
        for f in bb.network["findings"]:
            if f.pattern not in ("cycle", "mule"):
                continue
            # موعد البلاغ يُحسب من تاريخ آخر عملية داعمة في الحالة نفسها
            case_rows = (a.loc[a.index.intersection(f.txn_indices)]
                         if len(a) else a)
            last_ts = case_rows[bb.date_col].max() if len(case_rows) else None
            records.append({
                "pattern": f.label,
                "accounts": list(f.accounts),
                "txn_count": len(f.txn_indices),
                "total_amount": float(f.total_amount),
                "period": f"{f.window_hours} ساعة",
                "sama_ref": f.sama_ref,
                "deadline": str(bb.sama["str_deadline"](last_ts, 3)) if last_ts is not None else "—",
                "evidence": list(f.evidence),
            })
    return records


def agent_compliance(bb: Blackboard) -> AgentReport:
    ctx, a = bb.ctx, bb.anomalies
    # بلاغ STR واحد لكل حالة (حساب تجزئة أو نمط شبكي) — لا لكل عملية،
    # مطابقةً لممارسة الإبلاغ الفعلية لدى وحدة التحريات المالية
    str_records = build_str_records(bb)
    str_count = len(str_records)
    ctr_thr = bb.sama["ctr_threshold"]
    ctr_count = int((a[bb.amount_col].abs() >= ctr_thr).sum()) if len(a) else 0

    deadline = min((r["deadline"] for r in str_records if r["deadline"] != "—"),
                   default=None)

    decided = (f"يستوجب {str_count} بلاغ STR فوري و{ctr_count} عملية نقدية تحت المراقبة — "
               f"مسودات البلاغ جاهزة"
               if (str_count or ctr_count) else "لا بلاغات STR/CTR مطلوبة")
    dl_txt = f" · أقرب موعد STR: {deadline}" if deadline else ""
    lines = [bb.post("نورة", "قرأت قوائم محمد وسارة → ربطتها بضوابط ساما"),
             bb.post("نورة", f"STR: {str_count} حالة · CTR: {ctr_count}{dl_txt}")]
    # حلقة التغذية الراجعة: نورة تعترف بالتعلّم التراكمي عبر الجلسات، لا
    # بقرارات هذه الجلسة فقط — هذا هو الفرق بين "كبت مؤقت" و"حلقة تعلّم حقيقية"
    fb_closed, cycles = ctx.get("fb_closed", 0), ctx.get("fb_cycles_total", 0)
    suppressed_total = ctx.get("fb_suppressed_total", 0)
    if cycles:
        lines.append(bb.post("نورة", f"الذاكرة التراكمية: {cycles} دورة مراجعة عبر "
                                     f"جلسات سابقة أقفلت {suppressed_total} إنذاراً كاذباً "
                                     f"— النظام بدأ هذه الجلسة أذكى من الأولى"))
    elif fb_closed:
        lines.append(bb.post("نورة", f"تغذية راجعة: {fb_closed} حالة أقفلها "
                                     f"المراجع البشري كإنذار كاذب — الدقة تتحسن"))
    # تعرض جملة ساما الثابتة للنمط الأغلب بين العمليات المشبوهة (نص ثابت لا مُولّد)
    if len(a):
        top_type = a["fraud_type"].mode().iloc[0]
        lines.append(bb.post("نورة", bb.sama["alert_for"](top_type)))
    # سلسلة الأدلة: ملخص كل حالة STR بمرجعها التنظيمي وموعدها
    evidence = [f"حالة STR: {r['pattern']} · {', '.join(r['accounts'][:4])} · "
                f"{r['total_amount']:,.0f} ريال · {r['sama_ref']} · "
                f"الموعد {r['deadline']}"
                for r in str_records]
    rep = AgentReport(
        agent_id="noura", persona="نورة",
        role="أخصائية الامتثال والـ AML", icon="🛡️",
        perceived=f"مخرجات محمد وسارة ({ctx['anoms']} عملية مشبوهة)", decided=decided,
        key_number=f"STR {str_count} · CTR {ctr_count}", confidence=None,
        rule="ربط كل نمط بمرجعه النظامي (م/20 + قواعد ساما) + مسودة STR منظمة الحقول",
        handoff_to="ريم", lines=lines, evidence=evidence)
    bb.reports.append(rep)
    return rep


def agent_forecast(bb: Blackboard) -> AgentReport:
    fc, flow, cur = bb.forecast, bb.flow, bb.currency
    base = dict(agent_id="fahad", persona="فهد",
                role="محلل السيولة والتنبؤ", icon="💰",
                perceived="التدفق النقدي اليومي", handoff_to="ريم",
                rule="forecast_cashflow (اتجاه + موسمية + نطاق ثقة 95%)")
    if fc is None:
        decided = "بيانات غير كافية للتنبؤ (يلزم 14 يوماً على الأقل)"
        rep = AgentReport(**base, decided=decided, key_number="—",
                          confidence=None, lines=[bb.post("فهد", decided)])
        bb.reports.append(rep)
        return rep

    fdf, slope, sigma = fc["forecast"], fc["slope"], fc["sigma"]
    fc_mean = float(fdf["forecast"].mean())
    trend = "صاعد ↑" if slope > 0 else "هابط ↓"
    # عدم اليقين يُعبَّر عنه بنطاق ثقة 95% (±1.96σ) — التمثيل الصادق المستخدم في
    # تبويب التحليل — لا بنسبة ثقة مضلّلة تنهار إلى الصفر مع تقلّب التدفق اليومي.
    band = 1.96 * sigma
    low, high = fc_mean - band, fc_mean + band
    net = flow["net"]
    alert = "⚠️ ضغط سيولة متوقّع" if (slope < 0 or net < 0) else "السيولة ضمن النطاق"
    decided = (f"متوسط متوقّع {fc_mean:,.0f} {cur}/يوم · الاتجاه {trend} · "
               f"نطاق ثقة 95%: [{low:,.0f} — {high:,.0f}] · {alert}")
    lines = [bb.post("فهد", f"تنبؤ 14 يوماً: ~{fc_mean:,.0f} {cur}/يوم ({trend})"),
             bb.post("فهد", f"نطاق ثقة 95%: {low:,.0f} — {high:,.0f} {cur} · {alert}")]
    rep = AgentReport(**base, decided=decided,
                      key_number=f"{fc_mean:,.0f} {cur}/يوم",
                      confidence=None, lines=lines)
    bb.reports.append(rep)
    return rep


def agent_reporting(bb: Blackboard) -> AgentReport:
    ctx, cur = bb.ctx, bb.currency
    summary = (f"ملخص تنفيذي: {ctx['anoms']} عملية مشبوهة من {ctx['n']:,} "
               f"بحجم تعرّض {ctx['exposure']:,.0f} {cur}. التقرير جاهز (PDF/Excel).")
    lines = [bb.post("ريم", "جمعت مخرجات الفريق → جهّزت الملخص التنفيذي"),
             bb.post("ريم", "PDF + Excel جاهزان للرفع المؤسسي")]
    rep = AgentReport(
        agent_id="reem", persona="ريم",
        role="منسّقة التقارير التنظيمية", icon="📄",
        perceived="مخرجات محمد وسارة ونورة وفهد", decided=summary,
        key_number=f"تعرّض {ctx['exposure']:,.0f} {cur}", confidence=None,
        rule="تجميع المخرجات + build_pdf/build_excel",
        handoff_to="القائدة", lines=lines)
    bb.reports.append(rep)
    return rep


def run_agent_pipeline(bb: Blackboard) -> dict:
    """خالد — الوكيل القائد: ينسّق المنفّذين بالترتيب ويُجمّع النتيجة للقائدة البشرية."""
    plan = ("سأشغّل محمد للكشف الإحصائي، ثم سارة للتحليل الشبكي، "
            "ثم نورة للامتثال، وفهد للسيولة، وأختم بريم لتجهيز التقرير.")
    bb.post("خالد", "بدء المهمة — تنسيق الفريق")
    for fn in (agent_detection, agent_network, agent_compliance,
               agent_forecast, agent_reporting):
        fn(bb)
    n, anoms = bb.ctx["n"], bb.ctx["anoms"]
    mission_conf = round((n - anoms) / n * 100, 1) if n else 0.0
    bb.post("خالد", f"اكتملت المهمة — نسبة الامتثال {mission_conf}% — "
                    f"بانتظار اعتماد القائدة")
    return {
        "plan": plan,
        "reports": bb.reports,
        "final": {"status": "Mission Approved", "confidence": mission_conf,
                  "message": "Digital Workforce Finished Analysis — "
                             "Ready For Human Compliance Review"},
    }
