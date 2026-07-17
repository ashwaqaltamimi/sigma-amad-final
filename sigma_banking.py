# ============================================================================
#  Sigma AI  —  مركز العمليات المالية الذكي  |  النواة المصرفية
#  Intelligent Financial Operations Center — Banking Core
#  هاكاثون امد 2026 — مصرف الإنماء × أكاديمية طويق
#
#  فلسفة البناء: كل رقم في هذه المنصة ناتج عن حساب حقيقي قابل للتفسير
#  والدفاع عنه. لا توجد أرقام تجميلية ولا ادعاءات لا يدعمها الكود.
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from datetime import datetime
import io
import math
import os
import time
from sigma_agents import Blackboard, run_agent_pipeline, build_str_records
from sigma_network import (detect_counterparty_column, run_network_analysis,
                           apply_network_flags)
from sigma_rba import (rate_customers, tier_map, velocity_thresholds,
                       boost_anomaly_scores, VELOCITY_BY_TIER, RBA_SCORE_BOOST)
from sigma_feedback import (build_review_cases, apply_feedback, feedback_log,
                            VERDICT_FP, VERDICT_TP, REVIEW_NONE,
                            load_feedback_history, history_to_decisions,
                            append_feedback_cycle, learning_curve,
                            case_key as _fb_case_key, HISTORY_COLUMNS as _FB_HISTORY_COLUMNS)

# مسار الذاكرة التراكمية لحلقة التغذية الراجعة — بيانات Demo فقط عمداً.
# ملف عمليات حقيقي مرفوع من مستخدم لا يُخزَّن على القرص أبداً (المبدأ الحاكم)؛
# هذا الملف يحمل فقط قرارات مراجعة على حسابات Demo التوضيحية لإثبات أن
# الحلقة تُغلَق عبر الجلسات، لا محتوى عمليات حقيقياً.
FEEDBACK_MEMORY_PATH = os.path.join(os.path.dirname(__file__), "feedback_memory.csv")

# البذرة الأولى الوحيدة — دورة مراجعة سابقة صادقة تحقّقنا منها آلياً:
# SA-1038 (دوران مشبوه) إنذار كاذب حقيقي (ليس ضمن حسابات SA-90xx المزروعة).
# تُستخدَم لإعادة بناء feedback_memory.csv عند «إعادة التعيين» في محطة
# المراجعة، بدل مسحها بالكامل — حتى لا تُفقَد لحظة العرض «النظام بدأ أذكى».
_SEED_FEEDBACK_DECISIONS = {
    _fb_case_key("SA-1038", "دوران مشبوه (Round-trip Proxy)"): VERDICT_FP,
}
_SEED_FEEDBACK_REVIEWER = "أشواق التميمي — مراجِعة الامتثال"
_SEED_FEEDBACK_TIMESTAMP = "2026-07-12 10:30"


def _reset_feedback_memory_to_seed():
    """يعيد الذاكرة التراكمية إلى حالة البذرة الأولى النظيفة (دورة 1 فقط)."""
    empty = pd.DataFrame(columns=_FB_HISTORY_COLUMNS)
    return append_feedback_cycle(FEEDBACK_MEMORY_PATH, empty,
                                 _SEED_FEEDBACK_DECISIONS, _SEED_FEEDBACK_REVIEWER,
                                 _SEED_FEEDBACK_TIMESTAMP)

# ----------------------------------------------------------------------------
#  0) إعداد الصفحة
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sigma AI | Banking Intelligence",
    page_icon="Σ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# حل دائم لوضوح جميع الـ captions على الخلفية الداكنة
st.markdown("""
<style>
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] * {
    color: #44607E !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
#  1) إعدادات النواة المصرفية  (Bank Profile)
#  ----------------------------------------------------------------------------
#  سيقما منصة مصرفية حصرياً — بقرار نهائي من صاحبة المشروع. هذا الكائن
#  يجمع مصطلحات وإعدادات القطاع المصرفي في مكان واحد بدل نثرها في الكود.
# ============================================================================
class BankProfile:
    """إعدادات النواة المصرفية: المصطلحات والحساسيات ونصوص الواجهة."""
    def __init__(self, key, name_ar, name_en, contamination,
                 anomaly_metric, protected_metric, anomaly_alert,
                 advisor_role, currency="ريال"):
        self.key = key
        self.name_ar = name_ar
        self.name_en = name_en
        self.contamination = contamination          # حساسية كشف الشذوذ
        self.anomaly_metric = anomaly_metric         # تسمية مؤشر الشذوذ
        self.protected_metric = protected_metric     # تسمية المبلغ المعرّض
        self.anomaly_alert = anomaly_alert           # نص التنبيه
        self.advisor_role = advisor_role             # دور المستشار في الـ AI
        self.currency = currency


BANK = BankProfile(
    key="banking",
    name_ar="القطاع المصرفي وحوكمة الخزينة",
    name_en="Banking & Treasury Governance",
    contamination=0.02,
    anomaly_metric="عمليات تستوجب المراجعة",
    protected_metric="إجمالي قيمة العمليات المشبوهة",
    anomaly_alert=(
        "رصد رادار سيجما عمليات خارج النطاق السلوكي المعياري. "
        "هذه العمليات تستوجب مراجعة الامتثال للتحقق من مطابقتها "
        "لضوابط مكافحة غسل الأموال وعمليات التجزئة (Structuring)."
    ),
    advisor_role=(
        "مستشار مالي تنفيذي متخصص في القطاع المصرفي السعودي، "
        "خبير في إدارة الأصول والخصوم (ALM)، السيولة، والامتثال "
        "لضوابط البنك المركزي السعودي (SAMA) ومعيار IFRS 9."
    ),
)

# ============================================================================
#  2) الهوية البصرية  —  خلفية بيضاء + أخضر الإنماء المؤسسي + ذهبي رصين
#  ابتعدنا عن النيون البنفسجي المبتذل نحو طابع مصرفي رصين وموثوق.
# ============================================================================
def inject_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

    /* تثبيت مخطط الألوان على الفاتح — يمنع المتصفح/النظام الليلي من
       قلب ألوان النصوص والحقول الأصلية فتختفي على الخلفية البيضاء */
    :root { color-scheme: light only; }
    html, body { color-scheme: light only !important; }

    :root {
        --bg:        #FFFFFF;
        --bg-2:      #EEF3F9;
        --surface:   #F6F9FC;
        --line:      rgba(40,70,110,0.18);
        --ink:       #0E1C30;
        --ink-dim:   #44607E;
        --accent:    #1E7D58;   /* أخضر الإنماء المؤسسي */
        --accent-2:  #8A6D0F;   /* ذهبي رصين */
        --danger:    #C0392B;
        --ok:        #1E7D58;
    }

    .stApp {
        background:
          radial-gradient(1200px 600px at 80% -10%, rgba(46,143,110,0.10), transparent 60%),
          radial-gradient(900px 500px at -10% 10%, rgba(201,162,39,0.06), transparent 55%),
          var(--bg) !important;
        color: var(--ink) !important;
    }
    html, body, [class*="css"], h1,h2,h3,h4,p,div,span,label,input,button {
        font-family: 'IBM Plex Sans Arabic', sans-serif !important;
    }
    /* استثناء أيقونات Material — لا يُحذف.
       القاعدة أعلاه تفرض الخط العربي على كل <span> بـ !important، وستريمليت
       الحديث يرسم أيقوناته (سهم الموسّع، أيقونة الرفع) عبر <span> يحمل اسم
       الرباط نصًّا ويعتمد على خط Material لتحويله إلى أيقونة. فرض الخط العربي
       يقتل التحويل، فتُطبع الكلمة نفسها (upload / keyboard_arrow_right) فوق
       العنوان العربي وتتداخل معه. الاستثناء يعيد لكل أيقونة خطّها.
       تحقّق حي: قبل الاستثناء كان خط الأيقونة "IBM Plex Sans Arabic" وبعده
       "Material Symbols Rounded". */
    [data-testid="stIconMaterial"],
    [data-testid="stExpanderToggleIcon"],
    span.material-symbols-rounded, span.material-symbols-outlined,
    span.material-icons,
    [class*="material-symbols"], [class*="material-icons"] {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                     'Material Icons' !important;
        font-weight: normal !important;
        font-style: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        direction: ltr !important;
        font-feature-settings: 'liga' !important;
    }

    /* إخفاء السايدبار بالكامل */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    button[kind="header"] { display: none !important; }

    /* إخفاء شريط ستريمليت العلوي الأبيض بالكامل */
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0 !important;
        background: transparent !important;
    }

    /* توسيع المحتوى الرئيسي ورفعه للأعلى بعد إزالة الشريط */
    .block-container {
        max-width: 1200px !important;
        padding: 1.2rem 2.5rem 6rem !important;
    }

    /* الرفع — upload zone مخصص */
    [data-testid="stFileUploader"] {
        background: #F6F9FC !important;
        border: 2px dashed rgba(46,143,110,0.5) !important;
        border-radius: 16px !important;
        padding: 8px !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(46,143,110,0.9) !important;
    }
    /* الصندوق الداخلي ونصّه — لا يتبع ثيم المتصفح الليلي */
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] > div {
        background: #F6F9FC !important;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] * {
        color: var(--ink) !important;
    }
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: var(--ink-dim) !important;
    }
    [data-testid="stFileUploader"] button[kind="secondary"],
    [data-testid="baseButton-secondary"] {
        background: #FFFFFF !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
    }

    /* ════ أمان الوضع الليلي: نصوص المكوّنات الأصلية تبقى داكنة على الأبيض ════ */
    [data-baseweb="tab"], [data-baseweb="tab"] p,
    [role="tab"], [role="tab"] p {
        color: var(--ink-dim) !important;
    }
    [role="tab"][aria-selected="true"], [role="tab"][aria-selected="true"] p {
        color: var(--accent) !important;
    }
    /* القوائم المنسدلة (selectbox / multiselect) */
    [data-baseweb="select"] *, [data-baseweb="popover"] li,
    [data-baseweb="menu"] li, [role="option"] {
        color: var(--ink) !important;
    }
    [data-baseweb="select"] > div, [data-baseweb="popover"] [role="listbox"] {
        background: #FFFFFF !important;
    }
    /* الموسّعات وحالات st.status */
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] p,
    [data-testid="stStatusWidget"] *, details summary {
        color: var(--ink) !important;
    }
    [data-testid="stExpander"] details {
        background: #FFFFFF !important; border: 1px solid var(--line) !important;
    }
    /* تنبيهات st.info/success/warning — نص داكن مقروء (هذه لا تحمل ألواناً
       مضمّنة، فالفرض هنا آمن ولا يمسّ بطاقات HTML الملوّنة) */
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] * {
        color: var(--ink) !important;
    }

    /* ── البطاقات المضيئة (KPI Cards) ── */
    [data-testid="stMetric"] {
        position: relative;
        background:
          radial-gradient(120% 80% at 50% -10%, rgba(46,143,110,0.16), transparent 60%),
          linear-gradient(160deg, var(--surface), var(--bg-2));
        border: 1px solid rgba(46,143,110,0.22);
        border-radius: 18px;
        padding: 24px 22px !important;
        box-shadow: 0 10px 32px rgba(20,45,80,0.10),
                    0 0 30px rgba(46,143,110,0.10),
                    inset 0 1px 0 rgba(255,255,255,0.05);
        overflow: hidden;
        transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    }
    /* شريط إضاءة علوي */
    [data-testid="stMetric"]::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        opacity: .9;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(46,143,110,0.6);
        box-shadow: 0 14px 40px rgba(20,45,80,0.13),
                    0 0 44px rgba(46,143,110,0.22);
    }
    /* إضاءات لونية متمايزة لكل بطاقة في صف المؤشرات */
    [data-testid="stHorizontalBlock"] > div:nth-of-type(1) [data-testid="stMetric"]::before {
        background: linear-gradient(90deg, #1E7D58, #45c89a); }
    [data-testid="stHorizontalBlock"] > div:nth-of-type(2) [data-testid="stMetric"]::before {
        background: linear-gradient(90deg, #8A6D0F, #f0cf5a); }
    [data-testid="stHorizontalBlock"] > div:nth-of-type(3) [data-testid="stMetric"]::before {
        background: linear-gradient(90deg, #C0392B, #f08a87); }
    [data-testid="stHorizontalBlock"] > div:nth-of-type(4) [data-testid="stMetric"]::before {
        background: linear-gradient(90deg, #2E5F8A, #6fa8d8); }
    [data-testid="stMetricValue"] {
        color: #0E1C30 !important; font-weight: 800 !important;
        font-size: 2.1rem !important;
        text-shadow: 0 2px 12px rgba(20,45,80,0.12);
    }
    [data-testid="stMetricLabel"] p {
        color: var(--ink-dim) !important; font-size:.92rem !important; font-weight: 500 !important;
    }

    [data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {
        background: var(--bg-2); border: 1px solid var(--line);
        border-radius: 16px; padding: 12px;
    }

    /* العنوان الرئيسي */
    .sigma-header { text-align: right; padding: 8px 0 4px; border-bottom: 1px solid var(--line); margin-bottom: 18px; }
    .sigma-header .title { font-size: 1.9rem; font-weight: 700; color: var(--ink); }
    .sigma-header .title .sym { color: var(--accent); font-family:'IBM Plex Mono' !important; }
    .sigma-header .sub { color: var(--ink-dim); font-size: 1rem; margin-top: 2px; }

    .section-head { font-size: 1.5rem; font-weight: 700; color: #0E1C30;
        margin: 14px 0 4px; padding-right: 14px; line-height: 1.4;
        border-right: 4px solid var(--accent);
        text-shadow: 0 1px 8px rgba(46,143,110,0.25); }
    .section-sub  { color: var(--accent); font-size: .82rem; margin: 0 0 16px; padding-right: 14px;
        font-family:'IBM Plex Mono' !important; letter-spacing:.4px; opacity: .85; font-weight: 500; }

    .pill { display:inline-block; padding:4px 12px; border-radius:40px; font-size:.8rem; font-weight:600; }
    .pill-live { background: rgba(46,143,110,0.12); color: var(--accent); border:1px solid rgba(46,143,110,0.4); }
    .pill-sim  { background: rgba(201,162,39,0.10); color: var(--accent-2); border:1px solid rgba(201,162,39,0.4); }

    .stButton button, .stDownloadButton button {
        background: linear-gradient(135deg, var(--accent), #15603F) !important;
        color: #fff !important; border: none !important; border-radius: 10px !important;
        font-weight: 600 !important; padding: 10px 22px !important;
    }
    .stButton button:hover, .stDownloadButton button:hover { filter: brightness(1.08); }

    section[data-testid="stSidebar"] { background: var(--bg-2) !important; border-left: 1px solid var(--line); }
    #MainMenu, footer { visibility: hidden; }

    /* ── شريط محادثة المساعد — متناسق مع الثيم الداكن ── */
    /* إزالة الخط الأبيض/الحد العلوي الافتراضي فوق الشريط السفلي */
    [data-testid="stBottom"], [data-testid="stBottomBlockContainer"],
    [data-testid="stBottom"] > div {
        background: var(--bg) !important;
        border-top: none !important;
        box-shadow: none !important;
    }
    [data-testid="stBottomBlockContainer"] {
        padding-bottom: 18px !important;
        background: linear-gradient(to top, var(--bg) 80%, rgba(10,22,40,0)) !important;
    }
    [data-testid="stChatInput"] {
        background: var(--surface) !important;
        border: 1.5px solid rgba(46,143,110,0.45) !important;
        border-radius: 16px !important;
        max-width: 1120px; margin: 0 auto;
        box-shadow: 0 6px 24px rgba(20,45,80,0.10),
                    0 0 28px rgba(46,143,110,0.12);
        transition: border-color .2s ease, box-shadow .2s ease;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(46,143,110,0.85) !important;
        box-shadow: 0 6px 28px rgba(20,45,80,0.12), 0 0 36px rgba(46,143,110,0.25);
    }
    [data-testid="stChatInput"] > div { background: transparent !important; border: none !important; }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: var(--ink) !important;
        font-family: 'IBM Plex Sans Arabic' !important;
        font-size: .95rem !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--ink-dim) !important; opacity: .65;
    }
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, var(--accent), #15603F) !important;
        border-radius: 12px !important; border: none !important;
    }
    [data-testid="stChatInput"] button:hover { filter: brightness(1.1); }
    [data-testid="stChatInput"] button svg { fill: #fff !important; }

    /* ── رسائل المحادثة ── */
    [data-testid="stChatMessage"] {
        background: var(--bg-2) !important;
        border: 1px solid var(--line) !important;
        border-radius: 14px !important;
    }

    /* ════════════════════════════════════════════════════════════
       إمكانية الوصول (Accessibility) — معيار WCAG
       ════════════════════════════════════════════════════════════ */

    /* مؤشر اليد على كل العناصر القابلة للنقر */
    .stButton button, .stDownloadButton button,
    [data-testid="stChatInput"] button,
    [role="tab"], [data-testid="stFileUploader"] button,
    summary, [data-testid="stExpander"] summary {
        cursor: pointer !important;
    }

    /* حلقات تركيز واضحة للتنقّل بلوحة المفاتيح (3px أخضر) */
    .stButton button:focus-visible,
    .stDownloadButton button:focus-visible,
    [data-testid="stChatInput"] textarea:focus-visible,
    [role="tab"]:focus-visible,
    [data-baseweb="select"]:focus-within,
    input:focus-visible, summary:focus-visible {
        outline: 3px solid var(--accent) !important;
        outline-offset: 2px !important;
        border-radius: 8px;
    }

    /* تمييز التبويب النشط بوضوح (لون + ثقل) */
    [role="tab"][aria-selected="true"] {
        color: var(--accent) !important;
        font-weight: 700 !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: var(--accent) !important;
    }

    /* احترام تفضيل تقليل الحركة — يوقف كل الانتقالات والتحوّلات */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            transition: none !important;
            animation: none !important;
            scroll-behavior: auto !important;
        }
        [data-testid="stMetric"]:hover { transform: none !important; }
    }
    </style>
    """, unsafe_allow_html=True)


# ميزانية لونية موحّدة للرسوم البيانية
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#0E1C30", family="IBM Plex Sans Arabic"),
    margin=dict(t=50, b=30, l=10, r=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
)


# ============================================================================
#  3) محرك البيانات — قراءة وتنظيف وربط الأعمدة بشكل دفاعي
# ============================================================================
@st.cache_data(show_spinner=False)
def load_data(file_bytes, filename):
    """يقرأ CSV أو Excel ويعيد DataFrame خام."""
    bio = io.BytesIO(file_bytes)
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(bio)
    return pd.read_csv(bio)


@st.cache_data(show_spinner=False)
def load_demo():
    """يحمّل ملف البيانات التجريبية المرفق مع المشروع."""
    import os
    demo_path = os.path.join(os.path.dirname(__file__), "banking_demo.csv")
    return pd.read_csv(demo_path)


@st.cache_data(show_spinner=False)
def load_customer_profiles():
    """يحمّل ملفات KYC التجريبية (customer_profiles.csv) — أو None بصدق."""
    import os
    path = os.path.join(os.path.dirname(__file__), "customer_profiles.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# كشف أعمدة التاريخ/المبلغ/الفئة تلقائياً (عربي/إنجليزي) بمرونة
def detect_columns(df):
    """يكتشف أعمدة التاريخ/المبلغ/الفئة بمرونة، ويعيد اقتراحات (لا يفرضها)."""
    date_col = amount_col = cat_col = None
    for col in df.columns:
        cl = str(col).lower()
        if date_col is None and any(w in cl for w in ["date", "تاريخ", "time", "يوم"]):
            date_col = col
        if amount_col is None and any(w in cl for w in
                ["amount", "sar", "price", "value", "مبلغ", "قيمة", "total", "debit", "credit"]):
            amount_col = col
        if cat_col is None and any(w in cl for w in
                ["channel", "category", "type", "dept", "قناة", "فئة", "نوع", "قسم", "وصف"]):
            cat_col = col
    return date_col, amount_col, cat_col


def validate_upload(df, date_col_guess, amount_col_guess):
    """
    يتحقق من جودة الملف المرفوع ويُعيد قائمة بالمشاكل.
    لا يصمت عند أي خطأ — يُبلّغ بنص عربي واضح يذكر اسم العمود الناقص.
    """
    issues = []
    if date_col_guess is None:
        issues.append(
            "❌ **عمود التاريخ مفقود** — لم يُعثر على عمود باسم "
            "`date` أو `تاريخ` أو `time` أو `يوم`. "
            "أضف عموداً يحمل أحد هذه الأسماء."
        )
    if amount_col_guess is None:
        issues.append(
            "❌ **عمود المبلغ مفقود** — لم يُعثر على عمود باسم "
            "`amount` أو `مبلغ` أو `SAR` أو `value` أو `total`. "
            "أضف عموداً رقمياً يحمل أحد هذه الأسماء."
        )
    if len(df) < 30:
        issues.append(
            f"⚠️ **بيانات قليلة جداً** — الملف يحتوي على **{len(df)} صف** فقط. "
            f"يُوصى بـ 30 صفاً على الأقل للحصول على تحليل إحصائي موثوق."
        )
    return issues


def prepare(df, date_col, amount_col, cat_col):
    """تنظيف صارم: يعيد (df_clean, تقرير_جودة). يرفع ValueError برسالة واضحة عند الفشل."""
    df = df.copy()

    # ── تحليل التاريخ بذكاء: يتعامل مع dd/mm/yyyy و yyyy-mm-dd معاً ──
    parsed_default = pd.to_datetime(df[date_col], errors="coerce")
    parsed_dayfirst = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    # نختار التحليل الذي ينجح في صفوف أكثر (dayfirst يكشف dd/mm/yyyy)
    if parsed_dayfirst.notna().sum() > parsed_default.notna().sum():
        df[date_col] = parsed_dayfirst
    else:
        df[date_col] = parsed_default
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")

    n0 = len(df)

    # ── كشف الصفوف المكررة (لا يصمت النظام عنها) ──────────────────
    dup_count = int(df.duplicated().sum())
    if dup_count:
        df = df.drop_duplicates().reset_index(drop=True)

    df = df.dropna(subset=[date_col, amount_col])
    dropped = n0 - len(df) - dup_count
    if df.empty:
        raise ValueError(
            "تعذّر استخراج أي صف صالح. تأكد من اختيار عمود التاريخ "
            "وعمود المبلغ الصحيحين من القائمة الجانبية."
        )
    df = df.sort_values(date_col).reset_index(drop=True)
    df[cat_col] = df[cat_col].astype(str).fillna("غير مصنّف")
    quality = {"total": n0, "kept": len(df), "dropped": max(dropped, 0),
               "duplicates": dup_count,
               "span_days": (df[date_col].max() - df[date_col].min()).days}
    return df, quality


# ============================================================================
#  4) محرك كشف الشذوذ متعدد الأبعاد  (Isolation Forest)
#  نُغذّيه بأربع سمات سلوكية لا بقيمة المبلغ وحدها — هذا هو الفرق بين
#  "نظام قواعد" بدائي و"ذكاء سلوكي" حقيقي.
# ============================================================================

# عتبات تصنيف الاحتيال — حسابية وقابلة للدفاع
_STRUCT_LOW   = 8_500    # ريال — بداية نطاق التجزئة
_STRUCT_HIGH  = 10_000   # ريال — نهاية نطاق التجزئة (حد التقرير المعياري)
_STRUCT_TOL   = 0.10     # تسامح ±10% لاعتبار المبالغ متقاربة
_STRUCT_MIN_N = 3        # حد أدنى 3 عمليات متقاربة في 48 ساعة
_VEL_MIN_N    = 5        # حد أدنى 5 عمليات شاذة من نفس الحساب في 24 ساعة
_VEL_HOURS    = 24       # نافذة زمنية لكشف الكثافة
_RT_TOL       = 0.20     # تسامح ±20% لاعتبار مبالغ الدوران متقاربة
_RT_MIN_N     = 2        # حد أدنى 2 تحويل في نفس اليوم لاعتباره مؤشر دوران


def _roundtrip_mask(frame, amount_col, date_col, account_col, cat_col=None):
    """
    مؤشر دوران أولي (Round-trip Proxy) — بدون تحليل شبكات كامل:
    يرصد حساباً له عمليتا تحويل أو أكثر في نفس اليوم بمبالغ متقاربة (±20%).
    هذا نمط تدوير الأموال المباشر — مؤشر أولي يستوجب التحقق، لا حكماً قاطعاً.
    يُعيد Series منطقية بنفس فهرس frame.
    """
    mask = pd.Series(False, index=frame.index)
    if not account_col or account_col not in frame.columns:
        return mask
    day = frame[date_col].dt.date
    if cat_col and cat_col in frame.columns:
        is_tr = frame[cat_col].astype(str).str.contains("تحويل|حوالة", na=False)
    else:
        is_tr = pd.Series(True, index=frame.index)
    for (_acct, _day), g in frame.groupby([account_col, day]):
        tr = g[is_tr.loc[g.index]]
        if len(tr) >= _RT_MIN_N:
            amts = tr[amount_col].abs()
            if amts.max() > 0 and amts.min() >= amts.max() * (1 - _RT_TOL):
                mask.loc[tr.index] = True
    return mask


def _classify_fraud_type(out_df, amount_col, date_col, account_col,
                         rt_mask=None, risk_tiers=None):
    """
    يُصنّف كل عملية شاذة إلى واحد من أربعة أنماط احتيال بناءً على قواعد حسابية:

    1. تجزئة عمليات (Structuring)
       القاعدة: المبلغ في [8,500 – 10,000) ريال
               + 3 عمليات متقاربة (±10%) في نافذة 48 ساعة
       المبرر: تجنّب حد التقرير الإلزامي (10,000 ريال)

    2. تكرار مشبوه (Velocity)
       القاعدة: نفس الحساب لديه 5+ عمليات شاذة في 24 ساعة
       المبرر: نمط التحويل السريع المتكرر

    3. دوران مشبوه (Round-trip Proxy)
       القاعدة: حساب له 2+ تحويل في نفس اليوم بمبالغ متقاربة (±20%)
       المبرر: مؤشر أولي على تدوير الأموال — يستوجب التحقق لا الحكم القاطع

    4. انحراف سلوكي (Behavioral Deviation)
       القاعدة: لا يتطابق مع أي نمط أعلاه
       المبرر: خروج إحصائي عن النطاق السلوكي المعتاد
    """
    anom = out_df[out_df["is_anomaly"]].copy()
    anom_idx_set = set(anom.index)
    results = {}
    if rt_mask is None:
        rt_mask = pd.Series(False, index=out_df.index)

    for idx, row in anom.iterrows():
        amt = abs(row[amount_col])
        dt  = row[date_col]

        # ── قاعدة 1: التجزئة ─────────────────────────────────────
        if _STRUCT_LOW <= amt < _STRUCT_HIGH:
            window = out_df[
                (out_df[date_col] >= dt - pd.Timedelta(hours=48)) &
                (out_df[date_col] <= dt + pd.Timedelta(hours=48)) &
                ((out_df[amount_col] - row[amount_col]).abs() <= amt * _STRUCT_TOL)
            ]
            if len(window) >= _STRUCT_MIN_N:
                results[idx] = "تجزئة عمليات (Structuring)"
                continue

        # ── قاعدة 2: الكثافة ─────────────────────────────────────
        # العتبة متكيفة مع تصنيف RBA لصاحب الحساب (عالي=3 · متوسط=4 · افتراضي=5)
        if account_col and account_col in out_df.columns:
            acct = row.get(account_col)
            if pd.notna(acct):
                vel_thr = VELOCITY_BY_TIER.get(
                    (risk_tiers or {}).get(str(acct)), _VEL_MIN_N)
                same_acct_total = out_df[
                    (out_df[account_col] == acct) &
                    (out_df[date_col] >= dt - pd.Timedelta(hours=_VEL_HOURS)) &
                    (out_df[date_col] <= dt + pd.Timedelta(hours=_VEL_HOURS))
                ]
                if len(same_acct_total) >= vel_thr:
                    results[idx] = "تكرار مشبوه (Velocity)"
                    continue

        # ── قاعدة 3: الدوران المشبوه (مؤشر أولي) ────────────────
        if bool(rt_mask.get(idx, False)):
            results[idx] = "دوران مشبوه (Round-trip Proxy)"
            continue

        # ── قاعدة 4: الانحراف السلوكي (افتراضي) ─────────────────
        results[idx] = "انحراف سلوكي (Behavioral Deviation)"

    return results


def detect_anomalies(df, amount_col, date_col, contamination, account_col=None,
                     cat_col=None, risk_tiers=None):
    """
    يكشف الشذوذ بـ IsolationForest ويُصنّف كل عملية شاذة إلى نوع احتيال محدد.
    المخرجات الإضافية: is_anomaly، risk_score، fraud_type، rba_tier
    risk_tiers: قاموس {حساب: تصنيف RBA} — يجعل عتبة قاعدة التكرار متكيفة
    (العميل عالي المخاطر يُراجَع عند 3 عمليات/24س بدل 5) ويرفع درجة خطر
    عملياته المشبوهة بمعامل مُعلن. None = سلوك افتراضي بلا RBA.
    """
    feats = pd.DataFrame(index=df.index)
    feats["amount"] = df[amount_col]
    feats["dow"] = df[date_col].dt.dayofweek
    roll = df[amount_col].rolling(7, min_periods=1).mean()
    feats["dev_from_trend"] = df[amount_col] - roll
    std = df[amount_col].std() or 1.0
    feats["abs_z"] = (df[amount_col] - df[amount_col].mean()).abs() / std

    # ── ميزة 5: كثافة الحساب (Velocity) ──────────────────────────────
    # عدد العمليات من نفس الحساب في آخر 24 ساعة
    # قاعدة حسابية واضحة: كلما ارتفع العدد = احتمال velocity أعلى
    if account_col and account_col in df.columns:
        vel = [
            ((df[account_col] == row[account_col]) &
             (df[date_col] >= row[date_col] - pd.Timedelta(hours=24)) &
             (df[date_col] <= row[date_col])).sum()
            for _, row in df.iterrows()
        ]
        feats["velocity_24h"] = vel
    else:
        feats["velocity_24h"] = 0

    # ── ميزة 6: تقارب المبالغ (Structuring Proximity) ─────────────────
    # عدد العمليات بمبلغ متقارب (±10%) في آخر 48 ساعة
    # قاعدة حسابية واضحة: تكرار المبالغ المتقاربة = إشارة تجزئة
    struct_prox = [
        ((df[date_col] >= row[date_col] - pd.Timedelta(hours=48)) &
         (df[date_col] <= row[date_col]) &
         ((df[amount_col] - row[amount_col]).abs() <= abs(row[amount_col]) * 0.10)).sum()
        for _, row in df.iterrows()
    ]
    feats["struct_proximity"] = struct_prox

    feats = feats.fillna(0)

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    out = df.copy()
    out["is_anomaly"] = model.fit_predict(feats) == -1
    out["is_anomaly_if"] = out["is_anomaly"].copy()   # كشف IF المنفصل (للمقارنة)
    raw = -model.score_samples(feats)
    out["risk_score"] = ((raw - raw.min()) / (np.ptp(raw) + 1e-9) * 100).round(1)

    # ── الكاشف الهجين: ضمّ قواعد AML الصريحة عالية الثقة ──────────────
    # IsolationForest يكشف الانحراف الإحصائي العام، لكن أنماط التجزئة
    # والتكرار مصمّمة لتكون "تحت الرادار" الإحصائي. لذا نضمّ قاعدتين
    # صريحتين — وهما من علامات AML المعيارية لدى ساما — لرفع الاستدعاء:
    amt_abs = df[amount_col].abs().reset_index(drop=True)
    rule_struct = ((amt_abs >= _STRUCT_LOW) & (amt_abs < _STRUCT_HIGH) &
                   (feats["struct_proximity"].reset_index(drop=True) >= _STRUCT_MIN_N))
    # عتبة التكرار متكيفة مع تصنيف RBA — النهج القائم على المخاطر (المادة 5)
    vel_thr = velocity_thresholds(df, account_col, risk_tiers or {})
    rule_vel = (feats["velocity_24h"].reset_index(drop=True)
                >= vel_thr.reset_index(drop=True))

    # قاعدة 3: مؤشر الدوران المشبوه (Round-trip Proxy)
    rt_mask  = _roundtrip_mask(df, amount_col, date_col, account_col, cat_col)
    rule_rt  = rt_mask.reset_index(drop=True)

    rule_flag = (rule_struct | rule_vel | rule_rt).to_numpy()

    newly = rule_flag & (~out["is_anomaly"].to_numpy())
    out.loc[rule_flag, "is_anomaly"] = True
    # العمليات المُضافة بالقاعدة عالية الثقة → درجة خطر لا تقل عن 80
    out.loc[newly & (out["risk_score"] < 80), "risk_score"] = 80.0

    # تصنيف نوع الاحتيال لكل عملية شاذة (يمرّر قناع الدوران والعتبات المتكيفة)
    out["fraud_type"] = "—"
    if out["is_anomaly"].any():
        fraud_map = _classify_fraud_type(out, amount_col, date_col, account_col,
                                         rt_mask=rt_mask,
                                         risk_tiers=risk_tiers)
        for idx, ftype in fraud_map.items():
            out.at[idx, "fraud_type"] = ftype

    # طبقة RBA: رفع مُعلن (+10) لدرجة خطر العمليات المشبوهة من عملاء
    # التصنيف «عالي» + كتابة عمود rba_tier لسجل التدقيق
    out = boost_anomaly_scores(out, account_col, risk_tiers or {})

    return out


# ============================================================================
#  5) محرك التنبؤ بالتدفق النقدي  (اتجاه خطي + موسمية أسبوعية + نطاق ثقة)
#  نموذج إحصائي شفّاف: كل مكوّن مفسّر — الاتجاه، النمط الأسبوعي، عدم اليقين.
# ============================================================================
def forecast_cashflow(df, amount_col, date_col, horizon=14):
    s = df.set_index(date_col)[amount_col].resample("D").sum().ffill()
    if len(s) < 14:
        return None
    y = s.values.astype(float)
    t = np.arange(len(y))

    # الاتجاه العام عبر المربعات الصغرى
    A = np.vstack([t, np.ones_like(t)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    trend = slope * t + intercept
    resid = y - trend

    # الملف الموسمي الأسبوعي من البواقي
    dow = s.index.dayofweek.values
    seasonal = np.array([resid[dow == d].mean() if (dow == d).any() else 0.0 for d in range(7)])

    # الإسقاط المستقبلي
    ft = np.arange(len(y), len(y) + horizon)
    fdates = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    fdow = fdates.dayofweek.values
    fc = slope * ft + intercept + seasonal[fdow]

    sigma = resid.std()
    fdf = pd.DataFrame({
        "date": fdates,
        "forecast": fc,
        "lower": fc - 1.96 * sigma,
        "upper": fc + 1.96 * sigma,
    })
    return {"history": s, "forecast": fdf, "slope": slope, "sigma": sigma}


# كلمات تصنيف اتجاه التدفق من نوع العملية
_INFLOW_WORDS  = ["إيداع", "deposit", "credit", "وارد", "دائن", "راتب", "تحصيل"]
_OUTFLOW_WORDS = ["سحب", "حوالة", "تحويل", "withdraw", "debit", "صادر",
                  "مدين", "pos", "نقاط بيع", "دفع", "سداد"]


def classify_flow(cat_value):
    """
    يُصنّف اتجاه التدفق من نوع العملية بقاعدة كلمات واضحة:
      'in'  = تدفق داخل (إيداع/وارد)
      'out' = تدفق خارج (سحب/حوالة/دفع)
    الافتراضي 'out' (تحفّظي: معظم العمليات المصرفية صرف).
    """
    v = str(cat_value).lower()
    if any(w in v for w in _INFLOW_WORDS):
        return "in"
    if any(w in v for w in _OUTFLOW_WORDS):
        return "out"
    return "out"


# تقسيم التدفق النقدي إلى داخل/خارج وحساب الصافي من نوع العملية
def split_cashflow(df, amount_col, cat_col, date_col):
    """
    يقسّم التدفق إلى داخل/خارج بناءً على transaction_type.
    يُعيد dict فيه السلاسل الزمنية اليومية + الإجماليات + الصافي.
    """
    direction = df[cat_col].map(classify_flow)
    amt = df[amount_col].abs()
    inflow_s  = amt.where(direction == "in",  0.0)
    outflow_s = amt.where(direction == "out", 0.0)

    daily_in  = inflow_s.groupby(df[date_col].dt.date).sum()
    daily_out = outflow_s.groupby(df[date_col].dt.date).sum()
    total_in  = float(inflow_s.sum())
    total_out = float(outflow_s.sum())
    return {
        "daily_in": daily_in, "daily_out": daily_out,
        "inflow": total_in, "outflow": total_out,
        "net": total_in - total_out,
    }


# ============================================================================
#  6) المساعد المالي — محرك محلي (افتراضي) + خيار ربط OpenAI
# ============================================================================
# كلمات مفتاحية لكل فئة — بحث بسيط بـ .lower() لا regex
_ADVISOR_KEYWORDS = {
    "A": ["شذوذ", "احتيال", "مشبوه", "خطر", "تجزئة", "structuring",
          "غير اعتيادي", "شاذ", "مخاطر", "velocity", "تكرار"],
    "B": ["تنبؤ", "توقع", "سيولة", "تدفق", "الأسبوع", "القادم",
          "forecast", "مستقبل", "كاش", "صافي", "داخل", "خارج"],
    "C": ["ساما", "sama", "امتثال", "str", "غسيل", "aml", "تقرير",
          "kyc", "اعرف عميلك", "بلاغ", "تنظيم"],
}


def local_advisor(query, ctx, sector):
    """
    مستشار محلي بنظام فئات رباعي — بحث بسيط بـ .lower() عن الكلمات المفتاحية.
    كل رد مبني على الأرقام الفعلية المحسوبة في ctx (لا توليد LLM).
      A — مخاطر وشذوذ  | B — تنبؤ وسيولة | C — امتثال وساما | D — عام
    """
    q = query.strip().lower()
    def fmt(x): return f"{x:,.0f}"

    def has(cat):
        return any(kw in q for kw in _ADVISOR_KEYWORDS[cat])

    n, anoms, exposure = ctx["n"], ctx["anoms"], ctx["exposure"]
    trend_txt, top_cat, fc_mean = ctx["trend_txt"], ctx["top_cat"], ctx["fc_mean"]
    struct = ctx.get("struct", 0); velocity = ctx.get("velocity", 0)
    roundtrip = ctx.get("roundtrip", 0); behavioral = ctx.get("behavioral", 0)
    inflow = ctx.get("inflow"); outflow = ctx.get("outflow"); net = ctx.get("net")

    # ── الفئة A — مخاطر وشذوذ ────────────────────────────────────
    if has("A"):
        return (f"رصد المحرك **{anoms}** عملية مشبوهة من أصل {fmt(n)} عملية، "
                f"بحجم تعرّض مالي **{fmt(exposure)} {sector.currency}**. "
                f"التوزيع: **{struct}** تجزئة عمليات (Structuring)، "
                f"**{velocity}** تكرار مشبوه (Velocity)، "
                f"**{roundtrip}** دوران مشبوه (Round-trip Proxy)، "
                f"**{behavioral}** انحراف سلوكي. "
                f"أعلى تركّز في فئة «{top_cat}». "
                f"التوصية: ابدأ بالعمليات الأعلى درجة خطر في سجل التدقيق.")

    # ── الفئة B — تنبؤ وسيولة ────────────────────────────────────
    if has("B"):
        base = (f"يشير نموذج التنبؤ إلى أن التدفق النقدي **{trend_txt}**، "
                f"بمتوسط متوقع **{fmt(fc_mean)} {sector.currency}** يومياً للفترة القادمة. "
                f"النموذج يفصل الاتجاه عن النمط الأسبوعي ويعرض نطاق ثقة 95%.")
        if net is not None:
            base += (f" أما الفترة المُحلّلة: التدفق الداخل **{fmt(inflow)}**، "
                     f"الخارج **{fmt(outflow)}**، والصافي **{fmt(net)} {sector.currency}**.")
        return base

    # ── الفئة C — امتثال وساما ───────────────────────────────────
    if has("C"):
        return (f"من منظور الامتثال لساما: رُصدت **{struct}** عملية تجزئة "
                f"(تستوجب رفع بلاغ STR فوراً وفق المادة 15 من نظام مكافحة غسل الأموال)، "
                f"و**{velocity}** عملية تكرار مشبوه (تستوجب مراجعة العمليات غير الاعتيادية)، "
                f"و**{behavioral}** انحراف سلوكي (يُوصى بالتحقق من هوية العميل وفق KYC). "
                f"الإجراء: راجع سجل التدقيق — كل عملية مرفقة بالمرجع التنظيمي وموعد الإبلاغ.")

    # ── الفئة D — عام ────────────────────────────────────────────
    return ("سيجما متخصص في تحليل البيانات المالية المصرفية. يمكنني مساعدتك في: "
            "تحليل الشذوذ، توقع التدفق النقدي، أو متطلبات الامتثال لساما.")


def openai_advisor(query, history, ctx, sector, api_key):
    """يربط بـ OpenAI إن توفّر مفتاح؛ يرفع استثناءً عند الفشل ليُلتقط ويُستبدل بالمحلي."""
    from openai import OpenAI
    sys = (f"أنت {sector.advisor_role} "
           f"بيانات الجلسة الحالية: {ctx['n']} عملية، {ctx['anoms']} عملية شاذة، "
           f"حجم تعرّض {ctx['exposure']:,.0f} {sector.currency}، "
           f"التدفق النقدي {ctx['trend_txt']}، أعلى فئة خطر «{ctx['top_cat']}». "
           f"أجب بإيجاز مهني بالعربية مستنداً إلى هذه الأرقام فقط، دون مبالغة.")
    client = OpenAI(api_key=api_key)
    msgs = [{"role": "system", "content": sys}] + history + [{"role": "user", "content": query}]
    r = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0.4)
    return r.choices[0].message.content


# ============================================================================
#  7) مولّد تقرير PDF تنفيذي — reportlab + arabic_reshaper + python-bidi
#  قرار المكتبة موثّق في CONTEXT.md: fpdf2 لا يدعم تشكيل العربية ولا RTL،
#  لذا انتقلنا إلى reportlab مع خط Amiri + إعادة تشكيل الحروف وترتيب BiDi.
# ============================================================================
def reshape_arabic(text):
    """
    يحل مشكلتين دفعة واحدة لأي نص عربي قبل طباعته في PDF:
      1) Reshaping: ربط الحروف العربية (ligatures) بدل فصلها.
      2) BiDi: ترتيب النص من اليمين لليسار بصرياً.
    إن لم تتوفر المكتبات يُعيد النص كما هو (فشل آمن).
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(str(text)))
    except Exception:
        return str(text)


def build_pdf(ctx, sector, quality, audit_rows=None, cashflow=None):
    """
    يبني تقرير امتثال PDF عربي/إنجليزي باستخدام reportlab.
    audit_rows: قائمة tuples (تاريخ، مبلغ، تصنيف، درجة خطر، إجراء ساما).
    cashflow:   dict(inflow, outflow, net) — أو None.
    """
    try:
        import os
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        return None

    base = os.path.dirname(__file__)
    reg  = os.path.join(base, "fonts", "Amiri-Regular.ttf")
    bold = os.path.join(base, "fonts", "Amiri-Bold.ttf")
    if not os.path.exists(reg):
        return None
    try:
        pdfmetrics.registerFont(TTFont("Amiri", reg))
        pdfmetrics.registerFont(TTFont("Amiri-Bold", bold if os.path.exists(bold) else reg))
    except Exception:
        return None

    # ألوان الهوية — خلفية بيضاء ونصوص داكنة (تباين طباعي واضح للتقارير الرسمية)
    BG    = (1.0, 1.0, 1.0)                    # أبيض — خلفية الصفحة
    SURF  = (243/255, 247/255, 251/255)        # رمادي فاتح مزرق — خلفية الصفوف
    GREEN = (30/255, 125/255, 88/255)          # أخضر الإنماء الداكن (نص/ترويسة)
    GOLD  = (138/255, 109/255, 15/255)         # ذهبي داكن — عناوين الأقسام
    INK   = (14/255, 28/255, 48/255)           # كحلي داكن — النص الأساسي
    DIM   = (68/255, 96/255, 126/255)          # رمادي مزرق — النص الثانوي
    RED   = (176/255, 46/255, 34/255)          # أحمر داكن — درجات الخطر

    W, H = A4
    buf  = io.BytesIO()
    c    = canvas.Canvas(buf, pagesize=A4)

    def ar(t):  return reshape_arabic(t)
    def fill(rgb): c.setFillColorRGB(*rgb)

    def page_bg():
        fill(BG); c.rect(0, 0, W, H, fill=1, stroke=0)

    def rtext(x, y, txt, size=11, color=INK, font="Amiri", arabic=True):
        """نص محاذى لليمين عند x."""
        fill(color); c.setFont(font, size)
        c.drawRightString(x, y, ar(txt) if arabic else str(txt))

    def ltext(x, y, txt, size=11, color=INK, font="Amiri", arabic=False):
        fill(color); c.setFont(font, size)
        c.drawString(x, y, ar(txt) if arabic else str(txt))

    page_bg()
    M_R = W - 50   # هامش يمين
    M_L = 50       # هامش يسار
    y   = H - 60

    # ── الترويسة ثنائية اللغة ─────────────────────────────────────
    rtext(M_R, y, "Sigma AI | تقرير الامتثال المالي", 20, GREEN, "Amiri-Bold")
    y -= 22
    ltext(M_L, y, "Financial Compliance Report", 12, DIM, "Amiri-Bold")
    rtext(M_R, y, datetime.now().strftime("%Y-%m-%d  %H:%M"), 11, DIM, "Amiri")
    y -= 12
    fill(GOLD); c.setLineWidth(1.2); c.line(M_L, y, M_R, y)
    y -= 28

    # ── الملخص التنفيذي ───────────────────────────────────────────
    rtext(M_R, y, "الملخص التنفيذي", 15, GOLD, "Amiri-Bold")
    y -= 24
    _comp_pct = (ctx['n'] - ctx['anoms']) / max(ctx['n'], 1) * 100
    summary = [
        ("إجمالي العمليات المُحلّلة",  f"{ctx['n']:,}"),
        ("العمليات المشبوهة المرصودة", f"{ctx['anoms']:,}"),
        ("إجمالي قيمة العمليات المشبوهة", f"{ctx['exposure']:,.0f} ريال"),
        ("نسبة الامتثال المقدّرة",      f"{_comp_pct:.1f}%"),
        ("عمليات مصنّفة تجزئة (Structuring)", f"{ctx.get('struct', 0)}"),
        ("نطاق البيانات الزمني",        f"{quality['span_days']} يوماً"),
        ("اتجاه التدفق النقدي",         "صاعد" if ctx['slope'] > 0 else "هابط"),
    ]
    for label, val in summary:
        fill(SURF); c.roundRect(M_L, y - 6, M_R - M_L, 22, 4, fill=1, stroke=0)
        rtext(M_R - 10, y, label, 11, DIM, "Amiri")
        ltext(M_L + 12, y, val, 12, INK, "Amiri-Bold", arabic=True)
        y -= 28
    y -= 8

    # ── جدول سجل التدقيق ──────────────────────────────────────────
    if audit_rows:
        rtext(M_R, y, "سجل التدقيق وكشف الاحتيال", 15, GOLD, "Amiri-Bold")
        y -= 22
        headers = ["التاريخ", "المبلغ", "التصنيف", "درجة الخطر", "إجراء ساما"]
        col_x   = [M_R, M_R - 95, M_R - 165, M_R - 300, M_R - 360]  # حواف يمنى للأعمدة
        fill(GREEN); c.rect(M_L, y - 4, M_R - M_L, 20, fill=1, stroke=0)
        for hx, htxt in zip(col_x, headers):
            rtext(hx, y, htxt, 9.5, (1, 1, 1), "Amiri-Bold")
        y -= 24
        for i, (d, amt, ftype, risk, action) in enumerate(audit_rows[:18]):
            if y < 90:
                c.showPage(); page_bg(); y = H - 60
            if i % 2 == 0:
                fill(SURF); c.rect(M_L, y - 4, M_R - M_L, 18, fill=1, stroke=0)
            rtext(col_x[0], y, str(d),     8.5, INK, "Amiri")
            rtext(col_x[1], y, str(amt),   8.5, INK, "Amiri")
            rtext(col_x[2], y, str(ftype), 8.5, GOLD, "Amiri")
            rtext(col_x[3], y, str(risk),  8.5, RED,  "Amiri")
            rtext(col_x[4], y, str(action),8.5, DIM,  "Amiri")
            y -= 19
        y -= 14

    # ── قسم التدفق النقدي ─────────────────────────────────────────
    if cashflow:
        if y < 140:
            c.showPage(); page_bg(); y = H - 60
        rtext(M_R, y, "تحليل التدفق النقدي للفترة", 15, GOLD, "Amiri-Bold")
        y -= 26
        net_color = GREEN if cashflow["net"] >= 0 else RED
        cf_rows = [
            ("التدفق الداخل",  f"{cashflow['inflow']:,.0f} ريال",  GREEN),
            ("التدفق الخارج",  f"{cashflow['outflow']:,.0f} ريال", RED),
            ("صافي التدفق",    f"{cashflow['net']:,.0f} ريال",     net_color),
        ]
        for label, val, clr in cf_rows:
            fill(SURF); c.roundRect(M_L, y - 6, M_R - M_L, 24, 4, fill=1, stroke=0)
            rtext(M_R - 10, y, label, 12, DIM, "Amiri")
            ltext(M_L + 12, y, val, 13, clr, "Amiri-Bold", arabic=True)
            y -= 30

    # ── التذييل ───────────────────────────────────────────────────
    fill(GOLD); c.setLineWidth(0.8); c.line(M_L, 50, M_R, 50)
    rtext(M_R, 36, "أُنتج بواسطة Sigma AI — للاستخدام الداخلي فقط", 8.5, DIM, "Amiri")
    ltext(M_L, 36, "Sigma AI Report", 8.5, DIM, "Amiri-Bold")

    c.showPage()
    c.save()
    return buf.getvalue()


# ============================================================================
#  7a) تصدير سجل التدقيق إلى Excel — ورقتان (سجل + ملخص)
#  يُنتَج في الذاكرة (BytesIO) دون كتابة أي ملف على القرص.
# ============================================================================
def build_excel(audit_df, summary):
    """
    يبني ملف .xlsx بورقتين:
      1) «سجل التدقيق»  — الصفوف الشاذة بكل أعمدتها التنظيمية.
      2) «ملخص الامتثال» — مؤشرات تنفيذية + التاريخ والوقت.
    يُعيد bytes أو None إن لم تتوفّر openpyxl.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
    except ImportError:
        return None

    wb = Workbook()
    # ترويسة خضراء بنص أبيض على جسم أبيض — تباين واضح للطباعة والعرض
    HEAD_FILL = PatternFill("solid", fgColor="1E7D58")
    HEAD_FONT = Font(color="FFFFFF", bold=True, name="Calibri")
    RTL = Alignment(horizontal="right", vertical="center")

    # ── الورقة 1: سجل التدقيق ──────────────────────────────────────
    ws1 = wb.active
    ws1.title = "سجل التدقيق"
    ws1.sheet_view.rightToLeft = True
    if audit_df is not None and len(audit_df):
        headers = list(audit_df.columns)
        ws1.append(headers)
        for j, _ in enumerate(headers, 1):
            cell = ws1.cell(row=1, column=j)
            cell.fill = HEAD_FILL; cell.font = HEAD_FONT; cell.alignment = RTL
        for _, row in audit_df.iterrows():
            ws1.append([str(v) for v in row.tolist()])
        for col in ws1.columns:
            width = max((len(str(c.value)) for c in col if c.value), default=10)
            ws1.column_dimensions[col[0].column_letter].width = min(width + 4, 45)
    else:
        ws1.append(["لا توجد عمليات مشبوهة في هذه البيانات"])

    # ── الورقة 2: ملخص الامتثال ────────────────────────────────────
    ws2 = wb.create_sheet("ملخص الامتثال")
    ws2.sheet_view.rightToLeft = True
    ws2.append(["المؤشر", "القيمة"])
    for j in (1, 2):
        c = ws2.cell(row=1, column=j)
        c.fill = HEAD_FILL; c.font = HEAD_FONT; c.alignment = RTL
    for k, v in summary.items():
        ws2.append([k, str(v)])
    ws2.column_dimensions["A"].width = 28
    ws2.column_dimensions["B"].width = 28

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ============================================================================
#  7b) وحدة الامتثال — AML Compliance Module
#  المراجع النظامية (مُتحقَّق منها):
#   - نظام مكافحة غسل الأموال، المرسوم الملكي م/20 لعام 1439هـ وتعديلاته
#     (المادة 15: إبلاغ الإدارة العامة للتحريات المالية «فوراً» عند الاشتباه).
#   - قواعد ودليل ساما لمكافحة غسل الأموال وتمويل الإرهاب
#     (القسم 7: مراقبة العمليات والأنشطة · القسم 8: الإبلاغ عن الاشتباه).
#  ملاحظة صدق: النظام يوجب الإبلاغ فوراً — «3 أيام العمل» المحسوبة في المنصة
#  هي سقف تصعيد داخلي متحفّظ تتبناه المنصة، لا مهلة نظامية.
# ============================================================================

# عتبة المراقبة النقدية — متسقة مع حد الإفصاح النقدي الوطني (60,000 ريال)
CTR_THRESHOLD    = 60_000          # ريال سعودي
STRUCTURING_LOW  = CTR_THRESHOLD * 0.85   # 51,000 ريال — بداية نطاق الخطر

# خريطة المراجع التنظيمية: نوع الشذوذ ← المرجع + الإجراء + سقف التصعيد الداخلي
SAMA_RULES = {
    "تجزئة":         {"ref": "المادة 15 · نظام مكافحة غسل الأموال (م/20 · 1439هـ)",
                      "action": "إبلاغ فوري للإدارة العامة للتحريات المالية (SAFIU)",
                      "deadline_days": 3, "authority": "SAFIU"},
    "نطاق التجزئة":  {"ref": "القسم 7 · قواعد ساما — مراقبة العمليات والأنشطة",
                      "action": "مراجعة داخلية فورية + تقييم رفع بلاغ اشتباه",
                      "deadline_days": 1, "authority": "إدارة الامتثال"},
    "قيمة متطرفة":   {"ref": "عتبة المراقبة النقدية 60,000 ريال (حد الإفصاح الوطني)",
                      "action": "عناية معززة وتوثيق مصدر الأموال",
                      "deadline_days": 1, "authority": "إدارة الامتثال"},
    "توقيت شاذ":     {"ref": "متطلبات العناية الواجبة المعززة (EDD) — النظام ولائحته",
                      "action": "مراجعة ملف العميل الكامل",
                      "deadline_days": 5, "authority": "إدارة الامتثال"},
    "انحراف سلوكي":  {"ref": "القسم 7 · قواعد ساما — الرقابة المستمرة على السلوك",
                      "action": "مراجعة داخلية",
                      "deadline_days": 7, "authority": "إدارة المخاطر"},
}


# الجمل التنظيمية الثابتة المرتبطة بنوع الاحتيال — نصوص ثابتة لا تُولّد بـ AI
SAMA_ALERTS = {
    "تجزئة عمليات (Structuring)":
        "⚠️ يستوجب رفع بلاغ اشتباه STR فوراً وفق المادة 15 من نظام مكافحة "
        "غسل الأموال (م/20) — الإدارة العامة للتحريات المالية",
    "تكرار مشبوه (Velocity)":
        "⚠️ يستوجب المراجعة وفق القسم 7 من قواعد ساما — مراقبة العمليات "
        "غير الاعتيادية",
    "دوران مشبوه (Round-trip Proxy)":
        "⚠️ مؤشر دوران مشبوه — يستوجب التحقق من مصدر الأموال وفق متطلبات "
        "المراقبة المستمرة (القسم 7 · قواعد ساما)",
    "انحراف سلوكي (Behavioral Deviation)":
        "⚠️ يُوصى بالتحقق من هوية العميل وفق متطلبات اعرف عميلك (KYC) — ساما",
    "دوران شبكي (Closed Loop)":
        "⚠️ حلقة تدوير أموال مغلقة مؤكدة شبكياً — يستوجب رفع STR فوراً وفق "
        "المادة 15 من نظام مكافحة غسل الأموال (م/20)",
    "شبكة تجميع وتمرير (Mule Network)":
        "⚠️ شبكة حسابات تجميع وتمرير (Money Mules) — يستوجب رفع STR فوراً "
        "للحسابات كافة (المادة 15) + عناية واجبة معززة",
    "تمرير سريع (Rapid Pass-through)":
        "⚠️ حساب عبور يُمرّر الأموال خلال ساعات — يستوجب المراجعة وفق القسم 7 "
        "من قواعد ساما — مراقبة العمليات",
}


def sama_alert_for(fraud_type):
    """يُعيد الجملة التنظيمية الثابتة لنوع الاحتيال (نص ثابت غير مولّد)."""
    return SAMA_ALERTS.get(fraud_type, SAMA_ALERTS["انحراف سلوكي (Behavioral Deviation)"])


def calc_str_deadline(detection_date, deadline_days=3):
    """
    يحسب «سقف التصعيد الداخلي» لبلاغ STR: N أيام عمل من تاريخ الاكتشاف،
    متجاهلاً الجمعة (4) والسبت (5) — عطلة نهاية الأسبوع السعودية.
    ملاحظة نظامية: المادة 15 من نظام مكافحة غسل الأموال توجب الإبلاغ فوراً؛
    هذا التاريخ سقف داخلي متحفّظ تتعقّبه المنصة، وليس المهلة النظامية.
    """
    d, added = pd.Timestamp(detection_date), 0
    while added < deadline_days:
        d += pd.Timedelta(days=1)
        if d.dayofweek not in (4, 5):
            added += 1
    return d.date()


def build_str_text(records):
    """
    يبني نص مسودة بلاغ STR بحقول منظمة مستوحاة من معيار goAML الدولي
    (UNODC) — كل حقل مقروء من سجلات build_str_records المحسوبة.
    """
    lines = [
        "=" * 62,
        "مسودة بلاغ عمليات مشبوهة (STR) — منصة سيجما",
        "Suspicious Transaction Report Draft — structured fields (goAML-inspired)",
        f"تاريخ الإنشاء: {datetime.now():%Y-%m-%d %H:%M}",
        "الجهة المستقبلة: الإدارة العامة للتحريات المالية (SAFIU)",
        "الأساس النظامي: المادة 15 · نظام مكافحة غسل الأموال (م/20 · 1439هـ) — الإبلاغ فوري",
        "=" * 62, "",
    ]
    for i, r in enumerate(records, 1):
        lines += [
            f"── الحالة {i} من {len(records)} " + "─" * 38,
            f"نمط الاشتباه   : {r['pattern']}",
            f"الحسابات       : {', '.join(r['accounts'])}",
            f"عدد العمليات   : {r['txn_count']}",
            f"إجمالي المبالغ : {r['total_amount']:,.0f} ريال سعودي",
            f"النطاق الزمني  : {r['period']}",
            f"المرجع التنظيمي: {r['sama_ref']}",
            f"الإبلاغ        : فوري · سقف التصعيد الداخلي: {r['deadline']}",
            "سلسلة الأدلة   :",
        ]
        lines += [f"  • {ev}" for ev in r["evidence"]]
        lines.append("")
    lines += ["-" * 62,
              "هذه مسودة أولية أنشأتها منصة سيجما من بيانات العمليات الفعلية.",
              "تُراجَع وتُعتمَد من مسؤول الإبلاغ عن غسيل الأموال (MLRO) قبل الرفع."]
    return "\n".join(lines)


def sama_ref(tag_string):
    """يُعيد قاموس مرجع ساما للنوع الأول المُكتشَف في سلسلة التصنيف."""
    for key in SAMA_RULES:
        if key in tag_string:
            return SAMA_RULES[key]
    return SAMA_RULES["انحراف سلوكي"]


def classify_anomaly(df_anomalies, df_full, amount_col, date_col):
    """
    يُصنّف كل عملية شاذة بناءً على سماتها الحسابية ويربطها بلوائح ساما:
      - نطاق التجزئة : 51,000–59,999 ريال (أسفل عتبة المراقبة النقدية) ← القسم 7 · قواعد ساما
      - قيمة متطرفة  : z-score > 3                                   ← عتبة المراقبة النقدية 60,000
      - تجزئة محتملة : 3+ عمليات متقاربة ±5% في الشهر ذاته          ← المادة 15 · نظام م/20
      - توقيت شاذ    : جمعة أو سبت                                   ← المادة 4
      - انحراف سلوكي : الحالة الافتراضية                              ← القسم 7 · قواعد ساما
    """
    mean_a = df_full[amount_col].mean()
    std_a  = df_full[amount_col].std() or 1.0

    tags = []
    for idx, row in df_anomalies.iterrows():
        reasons = []
        amt = abs(row[amount_col])

        # 1. نطاق التجزئة — المنطقة الخطرة أسفل عتبة المراقبة النقدية (القسم 7)
        if STRUCTURING_LOW <= amt < CTR_THRESHOLD:
            reasons.append("🟡 نطاق التجزئة")

        # 2. قيمة متطرفة — z-score > 3 (عتبة المراقبة النقدية)
        z = abs((row[amount_col] - mean_a) / std_a)
        if z > 3:
            reasons.append("🔴 قيمة متطرفة")

        # 3. تجزئة محتملة — 3+ عمليات بنفس المبلغ ±5% في الشهر ذاته (المادة 15)
        tol = amt * 0.05
        same_month = df_full[
            (df_full[date_col].dt.year  == row[date_col].year) &
            (df_full[date_col].dt.month == row[date_col].month) &
            ((df_full[amount_col] - row[amount_col]).abs() <= tol)
        ]
        if len(same_month) >= 3:
            reasons.append("🟠 تجزئة محتملة")

        # 4. توقيت شاذ — جمعة أو سبت (عناية واجبة معززة EDD)
        if row[date_col].dayofweek in (4, 5):
            reasons.append("🔵 توقيت شاذ")

        # 5. انحراف سلوكي — الحالة الافتراضية (القسم 7)
        if not reasons:
            reasons.append("⚪ انحراف سلوكي")

        tags.append(" · ".join(reasons))

    return tags


# ============================================================================
#  7c) العارض البصري لشبكات غسيل الأموال — Plotly Network Graph (UI فقط)
#  يرسم كل اكتشاف شبكي من عملياته الفعلية: العقد حسابات، الأسهم تحويلات
#  حقيقية بمبالغها. المواضع حتمية (دائرة/نجمة/خط) — لا عشوائية بين تشغيلين.
# ============================================================================
_NET_INK, _NET_RED, _NET_GOLD, _NET_BLUE = "#0E1C30", "#C0392B", "#8A6D0F", "#2E5F8A"


def _finding_edges(finding, txns, account_col, cp_col, amount_col):
    """يجمع حواف الاكتشاف من صفوف العمليات الداعمة: (مرسل، مستقبل) → مبلغ وعدد."""
    agg = {}
    for _, r in txns.iterrows():
        key = (str(r[account_col]), str(r[cp_col]))
        cur = agg.setdefault(key, {"amount": 0.0, "n": 0})
        cur["amount"] += abs(float(r[amount_col]))
        cur["n"] += 1
    return agg


def _finding_positions(finding, edge_keys):
    """مواضع حتمية للعقد حسب نمط الاكتشاف — دائرة للحلقة، نجمة للتجميع، خط للتمرير."""
    pos = {}
    if finding.pattern == "cycle":
        n = len(finding.accounts)
        for i, acc in enumerate(finding.accounts):
            ang = math.pi / 2 - 2 * math.pi * i / n     # نبدأ من الأعلى وباتجاه العقارب
            pos[acc] = (math.cos(ang), math.sin(ang))
    else:
        hub = finding.accounts[0]
        senders = sorted({s for s, r in edge_keys if r == hub and s != hub})
        receivers = sorted({r for s, r in edge_keys if s == hub and r != hub})
        pos[hub] = (0.0, 0.0)
        # المرسلون: قوس واسع على اليمين — نصف قطر كبير يمنع تزاحم التسميات
        for i, s in enumerate(senders):
            ang = (math.pi * 5 / 12) - (math.pi * 5 / 6) * i / max(len(senders) - 1, 1) \
                  if len(senders) > 1 else 0.0
            pos[s] = (1.55 * math.cos(ang), 1.35 * math.sin(ang))
        for i, r in enumerate(receivers):                # المستقبلون يسار الحساب المحوري
            y = 0.55 - 1.1 * i / max(len(receivers) - 1, 1) if len(receivers) > 1 else 0.0
            pos[r] = (-1.55, y)
    # أي عقدة لم تُموضَع (احتياط) — تُصفّ أسفل الرسم
    stray = [n for n in {a for k in edge_keys for a in k} if n not in pos]
    for i, n_ in enumerate(stray):
        pos[n_] = (-1.0 + 2.0 * i / max(len(stray) - 1, 1) if len(stray) > 1 else 0.0,
                   -1.7)
    return pos


def build_network_figure(finding, txns, account_col, cp_col, amount_col):
    """
    يبني رسم Plotly لاكتشاف شبكي واحد — كل سهم تحويل فعلي بمبلغه المجموع.
    قواعد وضوح صارمة (لعرضٍ نظيف أمام اللجنة):
      - تسميات الأسهم أرقام فقط (لا نص عربي داخل SVG — يتجنب انعكاس BiDi)
        وكل تسمية داخل صندوق أبيض معزول يمنع التداخل البصري.
      - اسم الحساب في صندوق أسفل/فوق العقدة حسب موقعها — لا يلمس الأسهم.
      - العنوان العربي خارج الرسم (HTML يدعم RTL) — انظر render_network_findings.
    """
    edges = _finding_edges(finding, txns, account_col, cp_col, amount_col)
    if not edges:
        return None
    pos = _finding_positions(finding, edges.keys())
    hub = finding.accounts[0]
    edge_color = _NET_RED if finding.pattern == "cycle" else _NET_GOLD
    nodes = sorted({a for k in edges for a in k})
    cx = sum(pos[n][0] for n in nodes) / len(nodes)
    cy = sum(pos[n][1] for n in nodes) / len(nodes)

    fig = go.Figure()
    annotations = []
    for (s, r), info in sorted(edges.items()):
        (x0, y0), (x1, y1) = pos[s], pos[r]
        annotations.append(dict(
            x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=2.0,
            arrowcolor=edge_color, standoff=24, startstandoff=24, opacity=0.8))
        # تسمية المبلغ: عند 42% من المسار (قرب المرسل حيث الفراغ أكبر)
        # مع إزاحة عمودية بعيداً عن مركز الرسم — داخل صندوق أبيض معزول
        lx, ly = x0 + 0.42 * (x1 - x0), y0 + 0.42 * (y1 - y0)
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy) or 1.0
        px, py = -dy / norm, dx / norm                   # متجه عمودي على السهم
        if (lx + px * 0.01 - cx) ** 2 + (ly + py * 0.01 - cy) ** 2 < \
           (lx - px * 0.01 - cx) ** 2 + (ly - py * 0.01 - cy) ** 2:
            px, py = -px, -py                            # اختر الجهة الأبعد عن المركز
        label = f"{info['amount']:,.0f}"
        if info["n"] > 1:
            label += f"  ×{info['n']}"
        annotations.append(dict(
            x=lx + px * 0.17, y=ly + py * 0.17, xref="x", yref="y",
            showarrow=False, text=label,
            font=dict(size=10, color=_NET_BLUE, family="IBM Plex Mono"),
            bgcolor="rgba(255,255,255,0.95)", bordercolor="#D5DEE8",
            borderwidth=1, borderpad=3))

    # اسم كل حساب في صندوق معزول — فوق العقدة إن كانت بأعلى الرسم وتحتها إن بأسفله
    for n_ in nodes:
        x, y = pos[n_]
        above = y >= cy
        annotations.append(dict(
            x=x, y=y + (0.22 if above else -0.22), xref="x", yref="y",
            showarrow=False, text=n_, yanchor="bottom" if above else "top",
            font=dict(size=11, color=_NET_INK, family="IBM Plex Mono"),
            bgcolor="rgba(255,255,255,0.95)", bordercolor="#D5DEE8",
            borderwidth=1, borderpad=3))

    # العقد: المحور بالأحمر، حسابات النمط بالذهبي، الأطراف الخارجية بالأزرق
    node_color = [(_NET_RED if n == hub or finding.pattern == "cycle"
                   else _NET_GOLD if n in finding.accounts else _NET_BLUE)
                  for n in nodes]
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
        mode="markers",
        marker=dict(size=30, color="#FFFFFF", opacity=1,
                    line=dict(width=3, color=node_color)),
        hovertext=[f"{n} — {'محور النمط' if n == hub else 'حساب متورط' if n in finding.accounts else 'طرف خارجي'}"
                   for n in nodes],
        hoverinfo="text", showlegend=False))

    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    fig.update_layout(
        annotations=annotations, height=430,
        xaxis=dict(visible=False, range=[min(xs) - 0.85, max(xs) + 0.85]),
        yaxis=dict(visible=False, range=[min(ys) - 0.6, max(ys) + 0.6]),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def render_network_findings(net_res, source_df, amount_col, key_prefix):
    """يعرض رسوم الاكتشافات الشبكية من net_res — يُستدعى من الامتثال وغرفة الوكلاء."""
    acc, cp = net_res.get("account_col"), net_res.get("cp_col")
    for i, f in enumerate(net_res["findings"]):
        txns = source_df.loc[source_df.index.intersection(f.txn_indices)]
        fig = build_network_figure(f, txns, acc, cp, amount_col)
        if fig:
            # العنوان العربي خارج الرسم — HTML يعرض RTL صحيحاً (بعكس SVG)
            st.markdown(
                f"<div dir='rtl' style='margin:14px 0 0;padding:10px 16px;"
                f"background:linear-gradient(160deg,#F6F9FC,#EEF3F9);"
                f"border:1px solid rgba(46,143,110,0.35);border-radius:12px 12px 0 0;"
                f"border-bottom:none;font-size:.95rem;color:#0E1C30;'>"
                f"<b style='color:#C0392B;'>🕸️ {f.label}</b>"
                f"<span style='color:#44607E;font-size:.85rem;'> — إجمالي "
                f"<b style='color:#8A6D0F;'>{f.total_amount:,.0f} ريال</b> خلال "
                f"<b>{f.window_hours} ساعة</b> · {len(f.txn_indices)} عملية · "
                f"المبالغ على الأسهم بالريال السعودي</span></div>",
                unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True,
                            key=f"{key_prefix}_{i}", config={"displayModeBar": False})


# ============================================================================
#  7.5) غرفة عمليات الوكلاء — Mission Control Room (UI فقط، لا منطق)
#  تقرأ النتائج المحسوبة مسبقاً في page_tool وتعرض الفريق الحتمي حيّاً.
# ============================================================================
_AGENT_ROOM_CSS = """
<style>
.sigma-room { direction: rtl; }
@keyframes agentIn {
  0%   { opacity:0; transform: translateY(14px) scale(.96); filter: blur(6px); }
  100% { opacity:1; transform: translateY(0)    scale(1);   filter: blur(0); }
}
@keyframes softPulse { 0%,100%{opacity:.55;} 50%{opacity:1;} }
@keyframes beamFlow  { to { stroke-dashoffset: 0; } }
.agent-card {
  position: relative; overflow: hidden; border-radius: 18px;
  padding: 18px 20px; margin: 10px 0;
  background: linear-gradient(160deg, rgba(248,251,254,0.92), rgba(238,244,250,0.92));
  border: 1px solid rgba(46,143,110,.35);
  box-shadow: 0 12px 34px rgba(20,45,80,0.11), inset 0 1px 0 rgba(255,255,255,.04);
  backdrop-filter: blur(10px);
  animation: agentIn .6s cubic-bezier(.22,.61,.36,1) both;
}
.agent-card::before {
  content:""; position:absolute; top:0; right:0; left:0; height:2px;
  background: linear-gradient(90deg, #1E7D58, #8A6D0F);
}
.agent-head { display:flex; align-items:center; gap:12px; margin-bottom:8px; }
.agent-avatar {
  width:42px; height:42px; border-radius:50%; display:grid; place-items:center;
  font-size:1.3rem; background:rgba(46,143,110,.16);
  border:1px solid rgba(46,143,110,.5); box-shadow:0 0 18px rgba(46,143,110,.25);
}
.agent-name { font-weight:800; color:#0E1C30; font-size:1.05rem; }
.agent-role { color:#5A7795; font-size:.8rem; }
.agent-num  { color:#8A6D0F; font-weight:800; font-family:'IBM Plex Mono'; font-size:1.1rem; }
.agent-line { color:#44607E; font-size:.9rem; margin:3px 0; }
.agent-badge {
  display:inline-block; padding:2px 10px; border-radius:999px; font-size:.72rem;
  color:#1E7D58; border:1px solid rgba(46,143,110,.5); background:rgba(46,143,110,.1);
  animation: softPulse 1.6s ease-in-out infinite;
}
.bb-panel {
  border-radius:16px; padding:14px 18px; margin:8px 0;
  background:rgba(241,246,251,0.92); border:1px solid rgba(60,110,158,.3);
  font-family:'IBM Plex Mono'; color:#1E6B50; min-height:60px;
}
.bb-line { padding:3px 0; border-bottom:1px dashed rgba(20,45,80,0.10); }
.leader-banner {
  border-radius:18px; padding:18px 22px; margin:12px 0; text-align:center;
  background: radial-gradient(120% 90% at 50% -10%, rgba(46,143,110,.22), transparent 60%),
              linear-gradient(160deg,#F6F9FC,#EEF3F9);
  border:1px solid rgba(201,162,39,.5);
  box-shadow:0 10px 32px rgba(20,45,80,0.12), 0 0 36px rgba(46,143,110,.18);
}
@media (prefers-reduced-motion: reduce) {
  .agent-card,.agent-badge { animation: none !important; }
}
</style>
"""


def _agent_card_html(rep):
    """يبني HTML بطاقة وكيل واحدة من AgentReport."""
    conf = f"{rep.confidence:.0f}%" if rep.confidence is not None else "—"
    lines = "".join(f"<div class='agent-line'>• {ln.split(': ',1)[-1]}</div>"
                    for ln in rep.lines)
    return (
        f"<div class='agent-card'>"
        f"<div class='agent-head'>"
        f"<div class='agent-avatar'>{rep.icon}</div>"
        f"<div><div class='agent-name'>{rep.persona}</div>"
        f"<div class='agent-role'>{rep.role}</div></div>"
        f"<div style='margin-inline-start:auto;text-align:left;'>"
        f"<div class='agent-num'>{rep.key_number}</div>"
        f"<div class='agent-role'>ثقة {conf}</div></div></div>"
        f"<div class='agent-line'><b>أدرك:</b> {rep.perceived}</div>"
        f"<div class='agent-line'><b>قرّر:</b> {rep.decided}</div>"
        f"{lines}"
        f"<div class='agent-badge'>↦ تسليم إلى {rep.handoff_to}</div>"
        f"</div>")


def render_agent_room(ctx, anomalies, fc, flow, sector, amount_col, date_col,
                      net_res=None):
    """يعرض غرفة عمليات الوكلاء: قائد + 5 منفّذين، حيّاً عبر اللوحة المشتركة."""
    st.markdown(_AGENT_ROOM_CSS, unsafe_allow_html=True)
    st.markdown(
        "<div class='sigma-room'><div class='section-head'>غرفة عمليات الوكلاء</div>"
        "<div class='section-sub'>Mission Control · قوة عاملة رقمية تُسانِد فريق الامتثال</div></div>",
        unsafe_allow_html=True)

    run = st.button("▶ تشغيل الفريق", use_container_width=True, key="run_agents")
    if run:
        st.session_state["agents_ran"] = True
    if not st.session_state.get("agents_ran"):
        st.info("اضغط «تشغيل الفريق» لبدء المهمة — القائدة تعطي الأمر، وخالد ينسّق الفريق.")
        return
    # المسرحة البصرية (الإيقاع الزمني) تُعرض عند الضغط فقط؛ في أي إعادة تشغيل
    # (تنزيل STR، رسالة شات، تطبيق قرار مراجعة) تبقى الغرفة كما هي بلا تأخير.
    animate = run
    if st.button("↻ إعادة تشغيل الغرفة", key="reset_agents"):
        st.session_state["agents_ran"] = False
        st.rerun()

    # بناء اللوحة وتشغيل المنطق (حتمي، فوري) ثم العرض المتسلسل (مسرحة بصرية)
    sama = {"ctr_threshold": CTR_THRESHOLD,
            "alert_for": sama_alert_for, "str_deadline": calc_str_deadline}
    bb = Blackboard(ctx=ctx, anomalies=anomalies, forecast=fc, flow=flow,
                    sama=sama, amount_col=amount_col, date_col=date_col,
                    currency=sector.currency, network=net_res)
    result = run_agent_pipeline(bb)

    # بطاقة القائد خالد (خطة التنسيق)
    with st.status("🧠 خالد — الوكيل القائد ينسّق الفريق…", expanded=True) as s:
        st.markdown(f"<div class='agent-line'>{result['plan']}</div>",
                    unsafe_allow_html=True)
        if animate:
            time.sleep(0.5)
        s.update(label="🧠 خالد — وزّع المهام على الفريق ✓", state="complete")

    # بطاقات المنفّذين تباعاً — حالة عمل (إيقاع) ثم البطاقة الدائمة المرئية
    for rep in result["reports"]:
        with st.status(f"{rep.icon} {rep.persona} — {rep.role}…", expanded=False) as s:
            if animate:
                time.sleep(0.6)
            s.update(label=f"{rep.icon} {rep.persona} — اكتمل ✓ (تسليم إلى {rep.handoff_to})",
                     state="complete")
        st.markdown(_agent_card_html(rep), unsafe_allow_html=True)
        # لحظة سارة: رسم الشبكة الحي — اللجنة ترى حلقة الغسيل بعينها
        if rep.agent_id == "sara" and net_res and net_res["findings"]:
            render_network_findings(net_res, anomalies, amount_col,
                                    key_prefix="netfig_room")
        # سلسلة الأدلة: العمليات الفعلية التي بُني عليها قرار الوكيل — قابلة للفحص
        if rep.evidence:
            with st.expander(f"🔎 سلسلة الأدلة — {rep.persona} ({len(rep.evidence)} بند)"):
                for ev in rep.evidence:
                    st.markdown(f"<div class='bb-line' style='direction:rtl;"
                                f"font-family:IBM Plex Mono;font-size:.8rem;"
                                f"color:#44607E;'>{ev}</div>",
                                unsafe_allow_html=True)

    # مسودة بلاغ STR بحقول منظمة (goAML-inspired) — ناتج عمل نورة، جاهزة للتنزيل
    str_records = build_str_records(bb)
    if str_records:
        st.download_button(
            f"📄 تنزيل مسودة بلاغ STR — {len(str_records)} حالة (حقول منظمة بمعيار goAML)",
            build_str_text(str_records), file_name="sigma_str_draft.txt",
            mime="text/plain", use_container_width=True)

    # اللوحة المشتركة — كل ما كتبه الفريق على اللوحة
    st.markdown(
        "<div class='bb-panel'><b>🗒️ اللوحة المشتركة</b>" +
        "".join(f"<div class='bb-line'>{ln}</div>" for ln in bb.lines) +
        "</div>", unsafe_allow_html=True)

    # اعتماد القائدة البشرية
    f = result["final"]
    st.markdown(
        f"<div class='leader-banner'><div style='font-size:1.3rem;font-weight:800;"
        f"color:#0E1C30;'>✅ Mission Approved — اعتمدت القائدة النتيجة</div>"
        f"<div style='color:#8A6D0F;font-weight:700;margin-top:6px;'>"
        f"نسبة الامتثال {f['confidence']:.0f}% · {f['message']}</div></div>",
        unsafe_allow_html=True)


# ============================================================================
#  8) الواجهة الرئيسية
# ============================================================================
def _goto(page):
    """تبديل الصفحة وإعادة تشغيل الواجهة."""
    st.session_state["page"] = page
    st.rerun()


# ============================================================================
#  صفحة الهبوط  (Landing Page)
# ============================================================================
def page_landing():
    sector = BANK
    _card = (
        "<div dir='rtl' style='background:linear-gradient(135deg,#F6F9FC,#EEF3F9);"
        "border:1px solid {border};border-radius:16px;padding:24px 22px;margin-top:4px;'>"
        "<div style='font-size:2.5rem;font-weight:800;color:{color};'>{num}</div>"
        "<div style='color:#0E1C30;font-weight:600;margin:8px 0 4px;font-size:1rem;'>{title}</div>"
        "<div style='color:#44607E;font-size:.78rem;'>{src}</div></div>"
    )
    _step = (
        "<div dir='rtl' style='background:#F6F9FC;border-right:3px solid {accent};"
        "border-radius:12px;padding:20px 18px;'>"
        "<div style='font-size:1.6rem;font-weight:800;color:{accent};margin-bottom:8px;'>{n}</div>"
        "<div style='color:#0E1C30;font-weight:600;font-size:.95rem;margin-bottom:6px;'>{title}</div>"
        "<div style='color:#44607E;font-size:.82rem;line-height:1.7;'>{desc}</div></div>"
    )

    # ── Hero ──────────────────────────────────────────────────────────
    st.markdown("""
    <div dir="rtl" style="padding:56px 0 20px;">
      <div style="font-size:.82rem;font-weight:600;letter-spacing:1.5px;
                  color:#1E7D58;margin-bottom:16px;">
        Sigma AI &nbsp;·&nbsp; هاكاثون امد 2026 &nbsp;·&nbsp; مصرف الإنماء × أكاديمية طويق
      </div>
      <div style="font-size:3.2rem;font-weight:800;color:#0E1C30;line-height:1.3;">
        التقصير لا يبدأ بسوء نية — بل بالزحام<br>
        <span style="color:#C0392B;">والغرامة تصل إلى 5 ملايين ريال عن كل مخالفة</span>
      </div>
      <div style="font-size:1.15rem;color:#44607E;margin-top:20px;line-height:2;max-width:760px;">
        بنك سعودي واحد يرفع
        <strong style="color:#0E1C30;">33 ألف بلاغ اشتباه شهرياً</strong>
        — ولا يُثبَت منها إلا
        <strong style="color:#C0392B;">أقل من 3%</strong>،
        فيغرق المحقق في الزحام وتمرّ الشبكة الحقيقية. وعند ثبوت التقصير، تُجيز
        <strong style="color:#0E1C30;">المادة 25</strong>
        للبنك المركزي الغرامة وعقوبات تطال المسؤولين أنفسهم.
        <br>
        سيقما يكشف الشبكة ويربطها بمادتها النظامية — و<strong style="color:#1E7D58;">كنزه
        الحقيقي ذاكرة تحفظ كل قرار مراجعة، فيبدأ كل يوم أذكى من أمسه.</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── بطاقات الأرقام ────────────────────────────────────────────────
    # الترتيب من اليمين لليسار (RTL): أول بطاقة في العمود الأيمن
    # الذاكرة أولاً (العمود الأيمن) — هي الكنز الفارق، لا الشبكة ولا الطبقات.
    v1, v2, v3 = st.columns(3)
    with v3:
        st.markdown(_card.format(
            border="rgba(201,162,39,0.4)", color="#8A6D0F",
            num="78.7% ← 82.2%", title="ذاكرة تصمد عبر الجلسات — كنز سيقما",
            src="قرار مراجعة واحد · F1 يصل 88.1% قبل لمس أي زر"), unsafe_allow_html=True)
    with v2:
        st.markdown(_card.format(
            border="rgba(46,143,110,0.5)", color="#1E7D58",
            num="12% ← 86%", title="أثر الطبقات الثلاث على دقة الكشف",
            src="إحصائي ← قواعد AML + تصنيف مخاطر ← تحليل شبكي"), unsafe_allow_html=True)
    with v1:
        st.markdown(_card.format(
            border="rgba(217,83,79,0.4)", color="#C0392B",
            num="68,750 ريال", title="شبكة غسيل خفية كشفها سيقما",
            src="6 مرسلين ← حساب تجميع ← تمرير خارجي في 49 ساعة"), unsafe_allow_html=True)

    # ── كيف يعمل سيجما ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    # الترتيب من اليمين لليسار (RTL): البطاقة ① في العمود الأيمن
    s1, s2, s3 = st.columns(3)
    with s3:
        st.markdown(_step.format(accent="#C0392B", n="01",
            title="كشف الشذوذ السلوكي",
            desc="يفحص كل عملية إحصائيًا وبقواعد مكافحة الغسيل، "
                 "ويشدّد الرقابة كلما ارتفع خطر العميل"), unsafe_allow_html=True)
    with s2:
        st.markdown(_step.format(accent="#8A6D0F", n="02",
            title="التحليل الشبكي",
            desc="يرسم العلاقات بين الحسابات فيكشف شبكات الغسيل الخفية "
                 "التي تغيب عن فحص العملية الواحدة"), unsafe_allow_html=True)
    with s1:
        st.markdown(_step.format(accent="#1E7D58", n="03",
            title="الذاكرة الذهبية — قاعدة معرفة تكبر مع البنك",
            desc="كل قرار مراجعة يُحفظ ويُطبَّق تلقائيًا في كل تشغيل قادم، "
                 "فتُقفل الإنذارات الكاذبة وتبقى خبرة المحققين داخل البنك "
                 "بعد انتقالهم — أصل يزداد قيمة، لا نظام يَبلى"),
            unsafe_allow_html=True)

    # ── زر الانتقال للأداة ───────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        if st.button("ابدأ التحليل الآن  ←", use_container_width=True):
            _goto("tool")
        st.markdown("""
        <div dir='rtl' style='text-align:center;color:#44607E;font-size:.82rem;margin-top:8px;'>
          ارفع ملفك أو جرّب البيانات التجريبية فوراً
        </div>""", unsafe_allow_html=True)

    # ── تذييل ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div dir='rtl' style='margin-top:60px;padding-top:20px;
         border-top:1px solid rgba(40,70,110,0.18);
         color:#44607E;font-size:.8rem;text-align:center;'>
      {sector.name_ar} &nbsp;·&nbsp; جميع الأرقام المعروضة ناتجة عن حسابات حقيقية قابلة للتدقيق
      &nbsp;·&nbsp; Sigma AI © 2026
    </div>""", unsafe_allow_html=True)


# ============================================================================
#  شريط الحالة العلوي — أول ما تراه اللجنة بعد رفع الملف
# ============================================================================
def render_status_bar(ctx, sector):
    st.markdown(
        f"<div dir='rtl' style='display:flex;flex-wrap:wrap;gap:14px;"
        f"background:linear-gradient(135deg,rgba(46,143,110,0.12),rgba(238,244,250,0.85));"
        f"border:1px solid rgba(46,143,110,0.35);border-radius:14px;"
        f"padding:14px 20px;margin-bottom:6px;box-shadow:0 6px 24px rgba(20,45,80,0.09);'>"
        f"<span style='color:#0E1C30;font-weight:600;font-size:.95rem;'>"
        f"✅ تم تحليل <b style='color:#1E7D58;'>{ctx['n']:,}</b> عملية</span>"
        f"<span style='color:#2E5F8A;'>|</span>"
        f"<span style='color:#0E1C30;font-weight:600;font-size:.95rem;'>"
        f"⚠️ <b style='color:#C0392B;'>{ctx['anoms']:,}</b> عملية مشبوهة</span>"
        f"<span style='color:#2E5F8A;'>|</span>"
        f"<span style='color:#0E1C30;font-weight:600;font-size:.95rem;'>"
        f"💰 قيمة العمليات المشبوهة: "
        f"<b style='color:#8A6D0F;'>{ctx['exposure']:,.0f}</b> {sector.currency}</span>"
        f"</div>", unsafe_allow_html=True)


# ============================================================================
#  قياس أداء النموذج مقابل الحقيقة الأرضية (Precision / Recall / F1)
#  يعمل فقط حين تتوفّر حسابات احتيال معروفة (بيانات Demo) — وإلا يُعيد None
#  فلا يظهر للملفات الحقيقية. هذا يصون المبدأ الحاكم: لا رقم بلا أساس.
# ============================================================================
def compute_performance(df, account_col, amount_col=None):
    """
    يقارن كشف النموذج (is_anomaly) بالحقيقة الأرضية في بيانات Demo:
    الحسابات المزروعة SA-90xx تمثّل احتيالاً معروفاً (39 عملية عبر 4 سيناريوهات:
    تجزئة، تكرار، دوران، وشبكة تجميع). يُعيد dict بمقاييس الطبقات الثلاث
    (IF وحده / الهجين / الهجين+الشبكي) أو None إن لم تتوفّر حقيقة أرضية.
    """
    if not account_col or account_col not in df.columns:
        return None
    truth = df[account_col].astype(str).str.match(r"SA-90\d\d")
    if truth.sum() == 0:
        return None   # لا حقيقة أرضية → لا تعرض اللوحة (ملف حقيقي)

    def _metrics(pred):
        pred = pred.astype(bool)
        tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": p, "recall": r, "f1": f}

    hybrid = _metrics(df["is_anomaly"])
    # مقاييس IsolationForest وحده (إن توفّر العمود المنفصل)
    if_only = (_metrics(df["is_anomaly_if"])
               if "is_anomaly_if" in df.columns else None)
    # مقاييس الكاشف الهجين قبل الطبقة الشبكية (تُحفظ في apply_network_flags)
    hybrid_only = (_metrics(df["is_anomaly_hybrid"])
                   if "is_anomaly_hybrid" in df.columns else None)

    # متوسط مبلغ الاحتيال الفعلي (لتقدير التعرّض لكل عملية فائتة)
    avg_fraud_amount = 0.0
    if amount_col and amount_col in df.columns:
        truth_amount = df.loc[truth, amount_col].abs()
        avg_fraud_amount = float(truth_amount.mean()) if len(truth_amount) else 0.0

    out = dict(hybrid)
    out.update({"n_truth": int(truth.sum()),
                "if_only": if_only,
                "hybrid_only": hybrid_only,
                "avg_fraud_amount": avg_fraud_amount})
    return out


# ============================================================================
#  لوحة الامتثال المستقلة — 4 مؤشرات KPI محسوبة من البيانات الفعلية
# ============================================================================
def render_rba_panel(ratings, anomalies, account_col):
    """لوحة النهج القائم على المخاطر: توزيع التصنيفات + العملاء الأعلى خطراً بأسبابهم."""
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-head' style='font-size:1.15rem;'>"
                "النهج القائم على المخاطر (RBA)</div>"
                "<div class='section-sub'>Risk-Based Approach · المادة 5 — "
                "العناية الواجبة على أساس المخاطر</div>", unsafe_allow_html=True)
    if ratings is None:
        st.caption("💡 تصنيف العملاء يتطلب ملفات KYC (customer_profiles.csv) — "
                   "متوفر في وضع Demo. عتبات الرقابة تتشدد تلقائياً مع التصنيف.")
        return

    counts = ratings["rba_tier"].value_counts()
    r1, r2, r3 = st.columns(3)
    _rcard = ("<div dir='rtl' style='background:linear-gradient(160deg,#F6F9FC,#EEF3F9);"
              "border:1px solid {b};border-radius:16px;padding:18px 20px;text-align:center;'>"
              "<div style='font-size:2rem;font-weight:800;color:{c};'>{v}</div>"
              "<div style='color:#44607E;font-size:.8rem;'>{k}</div>"
              "<div style='color:#44607E;font-size:.7rem;margin-top:4px;'>{s}</div></div>")
    with r1:
        st.markdown(_rcard.format(b="rgba(192,57,43,.5)", c="#C0392B",
                    v=int(counts.get("عالي", 0)), k="عملاء عاليو المخاطر",
                    s="عتبة التكرار: 3 عمليات/24س + رفع درجة الخطر"),
                    unsafe_allow_html=True)
    with r2:
        st.markdown(_rcard.format(b="rgba(201,162,39,.5)", c="#8A6D0F",
                    v=int(counts.get("متوسط", 0)), k="عملاء متوسطو المخاطر",
                    s="عتبة التكرار: 4 عمليات/24س"), unsafe_allow_html=True)
    with r3:
        st.markdown(_rcard.format(b="rgba(46,143,110,.5)", c="#1E7D58",
                    v=int(counts.get("منخفض", 0)), k="عملاء منخفضو المخاطر",
                    s="العتبة الافتراضية: 5 عمليات/24س"), unsafe_allow_html=True)

    high = ratings[ratings["rba_tier"] == "عالي"].sort_values(
        "rba_score", ascending=False)
    if len(high):
        with st.expander(f"ملفات العملاء عاليي المخاطر ({len(high)}) — "
                         "الدرجة وأسبابها المعلنة"):
            show = high.rename(columns={"account_id": "الحساب",
                                        "rba_score": "الدرجة",
                                        "rba_reasons": "عوامل التصنيف (أوزان معلنة)"})
            st.dataframe(show[["الحساب", "الدرجة", "عوامل التصنيف (أوزان معلنة)"]],
                         use_container_width=True, hide_index=True)
            st.caption("⚖️ التصنيف العالي يعني رقابة أشد — لا اتهاماً: في هذه البيانات "
                       "عملاء بملفات عالية المخاطر وسلوك نظيف تماماً لم يُنذر عنهم النظام.")


def render_learning_memory_panel(history, df_full, account_col, truth_mask):
    """
    الذاكرة التراكمية: أثبات أن الحلقة تُغلَق عبر الجلسات لا داخل جلسة واحدة.
    تعرض عدد دورات المراجعة السابقة (من أي جلسة تشغيل، حتى منذ أيام) ومنحنى
    الدقة التراكمي — تظهر فقط حين تتوفر حقيقة أرضية حقيقية (بيانات Demo)،
    وتختفي بصدق للملفات الحقيقية (لا رقم أداء بلا أساس).
    """
    if truth_mask is None:
        return
    curve = learning_curve(df_full, account_col, truth_mask, history)
    if len(curve) < 2:
        st.caption("💡 لا دورات مراجعة سابقة محفوظة بعد — أوّل قرار تُطبّقينه أدناه "
                   "يبدأ الذاكرة التراكمية التي تصمد عبر إعادة تشغيل التطبيق.")
        return

    base, latest = curve.iloc[0], curve.iloc[-1]
    n_cycles = int(curve["cycle"].max())
    st.markdown(
        f"<div dir='rtl' style='background:linear-gradient(135deg,#F6F9FC,#EEF3F9);"
        f"border:1px solid rgba(46,143,110,.4);border-radius:14px;padding:16px 20px;"
        f"margin-bottom:12px;'>"
        f"<b style='color:#1E7D58;'>🧠 الذاكرة التراكمية عبر الجلسات</b>"
        f"<span style='color:#44607E;font-size:.85rem;'> — {n_cycles} دورة مراجعة "
        f"سابقة محفوظة على القرص، {int(latest['applied_total'])} قراراً تراكمياً</span>"
        f"<div style='margin-top:8px;font-size:1.05rem;color:#0E1C30;'>"
        f"الدقة ارتفعت من <b style='color:#C0392B;'>{base['precision']:.0f}%</b> "
        f"(بلا أي تغذية راجعة) إلى <b style='color:#1E7D58;'>{latest['precision']:.0f}%</b> "
        f"عبر {n_cycles} دورة تراكمية — دون المساس بالاستدعاء "
        f"({latest['recall']:.0f}%).</div></div>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["cycle"], y=curve["precision"], mode="lines+markers",
                             name="الدقة", line=dict(color="#1E7D58", width=3, shape="hv"),
                             marker=dict(size=9)))
    fig.add_trace(go.Scatter(x=curve["cycle"], y=curve["recall"], mode="lines+markers",
                             name="الاستدعاء", line=dict(color="#2E5F8A", width=2,
                                                         dash="dot", shape="hv"),
                             marker=dict(size=7)))
    fig.update_layout(title="منحنى التعلّم — الأداء عبر دورات المراجعة التراكمية",
                      xaxis_title="دورة المراجعة", yaxis=dict(title="%", range=[0, 105]),
                      height=280, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


def render_feedback_panel(anomalies, account_col, amount_col, ctx, df_full=None,
                          truth_mask=None, feedback_history=None, use_demo=False):
    """محطة مراجعة البلاغات: قرارات المراجع البشري تتغذى في النظام فوراً وتُحفَظ عبر الجلسات."""
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-head' style='font-size:1.15rem;'>"
                "محطة المراجعة البشرية — حلقة التغذية الراجعة</div>"
                "<div class='section-sub'>Analyst Feedback Loop · "
                "قرار المراجع يُقفل الإنذار الكاذب ويرفع الدقة — ويصمد عبر الجلسات</div>",
                unsafe_allow_html=True)

    if feedback_history is None:
        feedback_history = load_feedback_history(None)
    render_learning_memory_panel(feedback_history, df_full, account_col, truth_mask)

    decisions = st.session_state.setdefault("fb_decisions", {})
    if ctx.get("fb_closed") or ctx.get("fb_confirmed"):
        st.success(f"✅ في هذه الدورة: أُقفلت {ctx['fb_closed']} حالة كإنذار كاذب "
                   f"وتأكدت {ctx['fb_confirmed']} — كل الأرقام في هذه الصفحة "
                   f"محدّثة بقرارات المراجع.")

    cases = build_review_cases(anomalies, account_col, amount_col)
    if not cases and not decisions:
        st.info("لا حالات تنبيه بانتظار المراجعة.")
        return

    with st.form("fb_form"):
        st.caption("راجع كل حالة (حساب × نمط) واحكم — ثم اضغط «تطبيق قرارات المراجعة»:")
        new_decisions = dict(decisions)
        for c in cases[:12]:      # أعلى 12 حالة خطراً — وحدة عمل واقعية للمراجع
            cols = st.columns([2.6, 1.4])
            with cols[0]:
                st.markdown(
                    f"<div dir='rtl' style='padding:6px 2px;font-size:.88rem;'>"
                    f"<b>{c['account']}</b> · {c['fraud_type']}<br>"
                    f"<span style='color:#44607E;font-size:.78rem;'>"
                    f"{c['txn_count']} عملية · {c['total_amount']:,.0f} ريال · "
                    f"أعلى خطر {c['max_risk']:.0f}</span></div>",
                    unsafe_allow_html=True)
            with cols[1]:
                choice = st.selectbox(
                    "قرار المراجع", [REVIEW_NONE, VERDICT_TP, VERDICT_FP],
                    index=[REVIEW_NONE, VERDICT_TP, VERDICT_FP].index(
                        decisions.get(c["key"], REVIEW_NONE)),
                    key=f"fbsel_{c['key']}", label_visibility="collapsed")
                if choice == REVIEW_NONE:
                    new_decisions.pop(c["key"], None)
                else:
                    new_decisions[c["key"]] = choice
        applied = st.form_submit_button("🧠 تطبيق قرارات المراجعة",
                                        use_container_width=True)
    if applied:
        st.session_state["fb_decisions"] = new_decisions
        if use_demo:
            _, n_new = append_feedback_cycle(
                FEEDBACK_MEMORY_PATH, feedback_history, new_decisions,
                reviewer="مراجع الامتثال", timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"))
            if n_new:
                st.toast(f"🧠 دورة مراجعة جديدة محفوظة في الذاكرة التراكمية "
                         f"({n_new} قراراً) — ستبدأ الجلسة القادمة بهذا التعلّم.")
        st.rerun()

    if decisions:
        log = feedback_log(decisions)
        st.download_button("📋 تصدير سجل قرارات المراجعة (أثر تدقيقي)",
                           log.to_csv(index=False).encode("utf-8-sig"),
                           file_name="sigma_review_log.csv", mime="text/csv",
                           use_container_width=True)

    if use_demo and len(feedback_history):
        st.download_button("🧠 تصدير الذاكرة التراكمية الكاملة (كل الدورات)",
                           feedback_history.to_csv(index=False).encode("utf-8-sig"),
                           file_name="sigma_feedback_memory.csv", mime="text/csv",
                           use_container_width=True)
        with st.expander("⚠️ إعادة تعيين الذاكرة إلى الدورة 1 الأصلية (لأغراض التمرين)"):
            st.caption("يمسح أي دورات مراجعة إضافية أُضيفت أثناء التمرين ويعيد الذاكرة "
                       "إلى حالتها الأصلية (دورة 1 فقط — حالة SA-1038) — استخدميه بين "
                       "جلسات التمرين، لا أثناء العرض الفعلي.")
            confirm = st.checkbox("أؤكد رغبتي في إعادة التعيين للدورة 1 الأصلية")
            if confirm and st.button("↩️ إعادة التعيين الآن"):
                _reset_feedback_memory_to_seed()
                st.session_state.pop("fb_decisions", None)
                st.rerun()


def render_compliance_dashboard(df, anomalies, amount_col, date_col,
                                account_col, cat_col, ctx, sector, net_res=None,
                                ratings=None, feedback_history=None,
                                truth_mask=None, use_demo=False):
    st.markdown("""
    <div dir='rtl' style='margin:8px 0 18px;'>
      <div class='section-head'>لوحة الامتثال التنفيذية</div>
      <div class='section-sub'>Compliance Dashboard · مؤشرات محسوبة من البيانات الفعلية</div>
    </div>""", unsafe_allow_html=True)

    # ── المؤشر 1: عداد STR الشهري (آخر 30 يوماً من البيانات) ──────────
    if not anomalies.empty:
        last_date = df[date_col].max()
        win_start = last_date - pd.Timedelta(days=30)
        recent    = anomalies[anomalies[date_col] >= win_start]
        str_month = int((recent["fraud_type"] == "تجزئة عمليات (Structuring)").sum())
    else:
        str_month = 0
    str_color = "#C0392B" if str_month > 0 else "#1E7D58"

    # ── المؤشر 2: نسبة الامتثال ──────────────────────────────────────
    pct = (ctx["n"] - ctx["anoms"]) / max(ctx["n"], 1) * 100
    if pct >= 95:   pct_color, pct_hex = "#1E7D58", "rgba(46,143,110,0.9)"
    elif pct >= 90: pct_color, pct_hex = "#8A6D0F", "rgba(201,162,39,0.9)"
    else:           pct_color, pct_hex = "#C0392B", "rgba(192,57,43,0.9)"

    k1, k2 = st.columns([1, 1.4])
    with k1:
        st.markdown(
            f"<div dir='rtl' style='position:relative;overflow:hidden;"
            f"background:radial-gradient(120% 80% at 50% -10%, rgba(192,57,43,0.16), transparent 60%),"
            f"linear-gradient(160deg,#F6F9FC,#EEF3F9);border:1px solid {str_color};"
            f"border-radius:18px;padding:24px 22px;height:150px;"
            f"box-shadow:0 10px 32px rgba(20,45,80,0.10),0 0 30px rgba(192,57,43,0.14);'>"
            f"<div style='position:absolute;top:0;left:0;right:0;height:3px;background:{str_color};'></div>"
            f"<div style='color:#44607E;font-size:.85rem;margin-bottom:8px;'>"
            f"بلاغات STR — آخر 30 يوماً</div>"
            f"<div style='font-size:3rem;font-weight:800;color:{str_color};'>{str_month}</div>"
            f"<div style='color:#44607E;font-size:.78rem;'>عملية تستوجب رفع STR فوراً — المادة 15</div>"
            f"</div>", unsafe_allow_html=True)
    with k2:
        st.markdown(
            f"<div dir='rtl' style='background:linear-gradient(160deg,#F6F9FC,#EEF3F9);"
            f"border:1px solid {pct_color};border-radius:18px;padding:24px 22px;height:150px;"
            f"box-shadow:0 10px 32px rgba(20,45,80,0.10);'>"
            f"<div style='color:#44607E;font-size:.85rem;margin-bottom:10px;'>نسبة الامتثال</div>"
            f"<div style='font-size:2.4rem;font-weight:800;color:{pct_color};margin-bottom:12px;'>"
            f"{pct:.1f}%</div>"
            f"<div style='background:rgba(20,45,80,0.10);border-radius:20px;height:14px;width:100%;'>"
            f"<div style='background:{pct_hex};height:14px;border-radius:20px;width:{pct:.1f}%;'></div>"
            f"</div>"
            f"<div style='color:#44607E;font-size:.8rem;margin-top:8px;'>"
            f"{pct:.0f}% من العمليات ضمن النطاق الطبيعي</div>"
            f"</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2 = st.columns([1.3, 1])

    # ── المؤشر 3: الحسابات الأعلى خطراً ──────────────────────────────
    with t1:
        st.markdown("<div class='section-head' style='font-size:1.15rem;'>"
                    "الحسابات الأعلى خطراً</div>"
                    "<div class='section-sub'>Top 5 High-risk Accounts</div>",
                    unsafe_allow_html=True)
        if (not anomalies.empty) and account_col and account_col in anomalies.columns:
            grp = anomalies.groupby(account_col).agg(
                alerts=("risk_score", "size"),
                avg_risk=("risk_score", "mean"),
            ).reset_index()
            # التصنيف الأغلب لكل حساب
            top_type = (anomalies.groupby(account_col)["fraud_type"]
                        .agg(lambda s: s.mode().iloc[0]))
            grp["التصنيف"] = grp[account_col].map(top_type)
            grp = grp.sort_values("avg_risk", ascending=False).head(5)
            grp["avg_risk"] = grp["avg_risk"].round(1)
            grp = grp.rename(columns={account_col: "الحساب",
                                      "alerts": "عدد التنبيهات",
                                      "avg_risk": "متوسط درجة الخطر"})
            grp = grp[["الحساب", "عدد التنبيهات", "متوسط درجة الخطر", "التصنيف"]]
            st.dataframe(grp, use_container_width=True, hide_index=True)
        else:
            st.info("لا يوجد عمود حساب (account_id) في البيانات لحساب الحسابات الأعلى خطراً.")

    # ── المؤشر 4: توزيع أنواع الاشتباه (Donut) ───────────────────────
    with t2:
        st.markdown("<div class='section-head' style='font-size:1.15rem;'>"
                    "توزيع أنواع الاشتباه</div>"
                    "<div class='section-sub'>Suspicion Type Distribution</div>",
                    unsafe_allow_html=True)
        labels = ["تجزئة عمليات", "تكرار مشبوه", "دوران مشبوه", "انحراف سلوكي",
                  "دوران شبكي", "شبكة تجميع", "تمرير سريع"]
        values = [ctx.get("struct", 0), ctx.get("velocity", 0),
                  ctx.get("roundtrip", 0), ctx.get("behavioral", 0),
                  ctx.get("net_cycle", 0), ctx.get("net_mule", 0),
                  ctx.get("net_pass", 0)]
        colors = ["#8A6D0F", "#C0392B", "#146C94", "#1E7D58",
                  "#B03A2E", "#C9A227", "#2E5F8A"]
        if sum(values) > 0:
            fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55,
                                   marker=dict(colors=colors),
                                   textinfo="label+percent",
                                   textfont=dict(size=12, family="IBM Plex Sans Arabic")))
            fig.update_layout(showlegend=False, height=300, **PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("لم تُرصد أنماط اشتباه في البيانات الحالية.")

    # ── النهج القائم على المخاطر: تصنيف العملاء وعتباته المتكيفة ──────
    render_rba_panel(ratings, anomalies, account_col)

    # ── خريطة الشبكة: الأنماط الشبكية المكتشفة بسلاسل أدلتها ──────────
    if net_res is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-head' style='font-size:1.15rem;'>"
                    "تحليل شبكات غسيل الأموال</div>"
                    "<div class='section-sub'>Graph Link Analysis · "
                    f"{net_res['n_accounts']} حساباً · {net_res['n_edges']} تحويلاً</div>",
                    unsafe_allow_html=True)
        if net_res["findings"]:
            render_network_findings(net_res, df, amount_col, key_prefix="netfig_comp")
            for f in net_res["findings"]:
                sev_color = "#C0392B" if f.severity >= 90 else "#8A6D0F"
                st.markdown(
                    f"<div dir='rtl' style='background:rgba(192,57,43,0.05);"
                    f"border:1px solid {sev_color};border-right:4px solid {sev_color};"
                    f"border-radius:12px;padding:14px 16px;margin:8px 0;'>"
                    f"<b style='color:{sev_color};'>🕸️ {f.label}</b>"
                    f"<span style='color:#44607E;font-size:.82rem;'> · خطورة قاعدية "
                    f"{f.severity:.0f} · {f.total_amount:,.0f} ريال · "
                    f"{f.window_hours} ساعة · {len(f.txn_indices)} عملية</span>"
                    f"<div style='color:#44607E;font-size:.84rem;margin-top:6px;'>"
                    f"الحسابات: {' · '.join(f.accounts)}</div>"
                    f"<div style='color:#8A6D0F;font-size:.8rem;margin-top:4px;'>"
                    f"{f.sama_ref}</div></div>",
                    unsafe_allow_html=True)
                with st.expander(f"سلسلة الأدلة — {f.label}"):
                    for ev in f.evidence:
                        st.markdown(f"<div dir='rtl' style='font-family:IBM Plex Mono;"
                                    f"font-size:.8rem;color:#44607E;padding:2px 0;'>"
                                    f"{ev}</div>", unsafe_allow_html=True)
        else:
            st.success("فُحص الرسم الشبكي كاملاً — لا حلقات مغلقة ولا شبكات تجميع ولا تمرير سريع.")
    else:
        st.caption("💡 التحليل الشبكي يتطلب عمود «الطرف المقابل» (counterparty) في البيانات "
                   "— أضفه لتفعيل كشف الحلقات المغلقة وشبكات التجميع.")

    # ── لوحة أداء النموذج (تظهر فقط عند توفّر حقيقة أرضية معروفة) ──────
    perf = compute_performance(df, account_col, amount_col)
    if perf:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-head' style='font-size:1.15rem;'>"
                    "أداء النموذج مقابل الحقيقة الأرضية</div>"
                    "<div class='section-sub'>Model Performance · Precision / Recall / F1 "
                    "— measured on planted ground-truth</div>", unsafe_allow_html=True)
        st.caption(f"يُقاس على بيانات Demo ذات الحقيقة الأرضية المعروفة "
                   f"({perf['n_truth']} عملية احتيال مزروعة فعلياً). "
                   f"كل مقياس محسوب بمقارنة كشف النموذج بالاحتيال الحقيقي — لا رقم مفترض.")

        p1, p2, p3 = st.columns(3)
        _pcard = (
            "<div dir='rtl' style='position:relative;overflow:hidden;text-align:center;"
            "background:radial-gradient(120% 80% at 50% -10%, rgba({g},0.18), transparent 60%),"
            "linear-gradient(160deg,#F6F9FC,#EEF3F9);border:1px solid {b};"
            "border-radius:18px;padding:22px 18px;"
            "box-shadow:0 10px 32px rgba(20,45,80,0.10),0 0 30px rgba({g},0.15);'>"
            "<div style='position:absolute;top:0;left:0;right:0;height:3px;background:{c};'></div>"
            "<div style='color:#44607E;font-size:.82rem;margin-bottom:6px;'>{label}</div>"
            "<div style='font-size:2.6rem;font-weight:800;color:{c};'>{val}</div>"
            "<div style='color:#44607E;font-size:.74rem;margin-top:4px;'>{sub}</div></div>"
        )
        with p1:
            st.markdown(_pcard.format(
                g="46,143,110", b="rgba(46,143,110,0.5)", c="#1E7D58",
                label="الدقة (Precision)", val=f"{perf['precision']*100:.0f}%",
                sub="من العمليات المُنذَر بها كانت احتيالاً فعلياً"),
                unsafe_allow_html=True)
        with p2:
            st.markdown(_pcard.format(
                g="201,162,39", b="rgba(201,162,39,0.5)", c="#8A6D0F",
                label="الاستدعاء (Recall)", val=f"{perf['recall']*100:.0f}%",
                sub="من الاحتيال الحقيقي تم كشفه"),
                unsafe_allow_html=True)
        with p3:
            st.markdown(_pcard.format(
                g="60,110,158", b="rgba(60,110,158,0.5)", c="#2E5F8A",
                label="مقياس F1", val=f"{perf['f1']*100:.0f}%",
                sub="التوازن بين الدقة والاستدعاء"),
                unsafe_allow_html=True)

        st.markdown(
            f"<div dir='rtl' style='background:#F6F9FC;border:1px solid rgba(40,70,110,0.18);"
            f"border-radius:12px;padding:12px 16px;margin-top:12px;font-size:.82rem;color:#44607E;'>"
            f"<b style='color:#0E1C30;'>مصفوفة الالتباس (Confusion Matrix):</b> "
            f"كشف صحيح <b style='color:#1E7D58;'>{perf['tp']}</b> · "
            f"إنذار كاذب <b style='color:#C0392B;'>{perf['fp']}</b> · "
            f"احتيال فائت <b style='color:#C0392B;'>{perf['fn']}</b> · "
            f"سليم صحيح <b style='color:#1E7D58;'>{perf['tn']}</b></div>",
            unsafe_allow_html=True)

        # ── المقارنة المرئية: الطبقات الثلاث — إحصائي / قواعد / شبكي ──
        ifm, hbm = perf.get("if_only"), perf.get("hybrid_only")
        if ifm:
            st.markdown("<div dir='rtl' style='margin-top:16px;color:#0E1C30;"
                        "font-weight:600;font-size:.95rem;'>"
                        "أثر القرار المعماري — كل طبقة كشف تُضاف فوق سابقتها</div>",
                        unsafe_allow_html=True)
            cols = {
                "المقياس": ["الدقة Precision", "الاستدعاء Recall", "مقياس F1"],
                "IsolationForest وحده": [f"{ifm['precision']*100:.0f}%",
                                         f"{ifm['recall']*100:.0f}%",
                                         f"{ifm['f1']*100:.0f}%"],
            }
            if hbm:
                cols["+ قواعد AML (هجين)"] = [f"{hbm['precision']*100:.0f}%",
                                              f"{hbm['recall']*100:.0f}%",
                                              f"{hbm['f1']*100:.0f}%"]
                final_label = "+ التحليل الشبكي"
            else:
                final_label = "الكاشف الهجين"
            cols[final_label] = [f"{perf['precision']*100:.0f}%",
                                 f"{perf['recall']*100:.0f}%",
                                 f"{perf['f1']*100:.0f}%"]
            st.dataframe(pd.DataFrame(cols), use_container_width=True, hide_index=True)

        # ── تفسير عربي بالتعرّض المالي ───────────────────────────────
        z = perf.get("avg_fraud_amount", 0.0)
        st.markdown(
            f"<div dir='rtl' style='background:rgba(46,143,110,0.07);"
            f"border:1px solid rgba(46,143,110,0.3);border-radius:12px;"
            f"padding:14px 16px;margin-top:12px;font-size:.88rem;color:#44607E;line-height:1.9;'>"
            f"النموذج يكتشف <b style='color:#1E7D58;'>{perf['tp']}</b> "
            f"من أصل <b style='color:#0E1C30;'>{perf['n_truth']}</b> عملية محتالة فعلية. "
            f"كل عملية فائتة (<b style='color:#C0392B;'>{perf['fn']}</b>) تمثّل تعرّضاً مالياً "
            f"محتملاً بمتوسط <b style='color:#8A6D0F;'>{z:,.0f} ريال</b> — "
            f"وهذا ما يجعل رفع الاستدعاء أولويةً تنفيذية لا مجرد رقم تقني.</div>",
            unsafe_allow_html=True)

    # ── محطة المراجعة البشرية + الذاكرة التراكمية نُقِلت إلى تبويب مستقل
    #    «🧠 الذاكرة والتعلّم» (لم تعد مدفونة في قاع هذه اللوحة). ──────────
    st.info("🧠 محطة المراجعة البشرية والذاكرة التراكمية عبر الجلسات "
            "أصبحت في تبويب مستقل: **الذاكرة والتعلّم**.")

    # ── تنبيه ساما المرجعي ───────────────────────────────────────────
    st.markdown("""
    <div dir='rtl' style='background:rgba(201,162,39,0.08);border:1px solid rgba(201,162,39,0.3);
         border-radius:12px;padding:14px 16px;margin-top:14px;font-size:.85rem;color:#44607E;
         line-height:1.8;'>
      <strong style='color:#8A6D0F;'>⚠ تنبيه امتثال ساما:</strong>
      وفق المادة 15 من نظام مكافحة غسل الأموال (م/20 · 1439هـ)، يُلزَم البنك بإبلاغ
      الإدارة العامة للتحريات المالية <strong style='color:#0E1C30;'>(SAFIU)</strong>
      <strong style='color:#0E1C30;'>فوراً</strong> عند نشوء الاشتباه — وتتعقّب
      سيجما سقفاً داخلياً متحفظاً (3 أيام عمل) لضمان عدم تجاوز أي حالة.
    </div>""", unsafe_allow_html=True)


# ============================================================================
#  صفحة الأداة  (Tool Page)
# ============================================================================
def page_tool():
    sector = BANK
    api_key = st.session_state.get("openai_key", "")

    # ── شريط علوي: العنوان + زر الرجوع ──────────────────────────────
    top_l, top_r = st.columns([1, 5])
    with top_l:
        if st.button("← العودة"):
            _goto("landing")
    with top_r:
        st.markdown("""
        <div dir='rtl' style='padding:12px 0 4px;'>
          <span style='font-size:1.4rem;font-weight:700;color:#0E1C30;'>
            <span style='color:#1E7D58;font-family:IBM Plex Mono;'>Σ</span>
            &nbsp;منصة سيجما — تحليل العمليات المالية
          </span>
          <span style='margin-right:12px;' class='pill pill-sim'>
            القطاع المصرفي
          </span>
        </div>""", unsafe_allow_html=True)
    st.divider()

    # ── منطقة الرفع ──────────────────────────────────────────────────
    ua, ub = st.columns([1.3, 1])
    with ua:
        st.markdown("""<div dir='rtl' style='margin-bottom:10px;'>
          <div style='font-size:1.05rem;font-weight:700;color:#0E1C30;margin-bottom:4px;'>
            ارفع ملف العمليات المالية</div>
          <div style='color:#44607E;font-size:.85rem;'>
            CSV أو Excel · عمود تاريخ + مبلغ + فئة (الكشف تلقائي)</div>
        </div>""", unsafe_allow_html=True)
        uploaded = st.file_uploader("رفع الملف", type=["csv", "xlsx", "xls"],
                                    label_visibility="collapsed")
        if uploaded:
            st.session_state["use_demo"] = False
    with ub:
        st.markdown("""<div dir='rtl' style='margin-bottom:10px;'>
          <div style='font-size:1.05rem;font-weight:700;color:#0E1C30;margin-bottom:4px;'>
            أو جرّب فوراً</div>
          <div style='color:#44607E;font-size:.85rem;'>
            514 عملية مصرفية واقعية — تتضمن 4 سيناريوهات غسيل أموال مزروعة</div>
        </div>""", unsafe_allow_html=True)
        if st.button("🎬 تشغيل Demo", use_container_width=True):
            st.session_state["use_demo"] = True
            st.session_state.pop("chat", None)
        if st.session_state.get("use_demo") and not uploaded:
            st.markdown("<span class='pill pill-live'>● وضع Demo نشط — 514 عملية</span>",
                        unsafe_allow_html=True)

    use_demo = st.session_state.get("use_demo", False) and not uploaded

    if not use_demo and uploaded is None:
        return   # انتظار المستخدم

    st.divider()

    # ── تحميل البيانات ───────────────────────────────────────────────
    try:
        if use_demo:
            raw = load_demo()
            st.caption("📋 وضع Demo — 514 عملية مصرفية تتضمن 4 سيناريوهات غسيل أموال مزروعة "
                       "(تجزئة · تكرار · دوران · شبكة تجميع)")
        else:
            raw = load_data(uploaded.getvalue(), uploaded.name)
    except Exception as e:
        st.error(f"تعذّر قراءة الملف: {e}"); return

    d_guess, a_guess, c_guess = detect_columns(raw)
    cols = list(raw.columns)
    date_col   = d_guess or cols[0]
    amount_col = a_guess or cols[0]
    cat_col    = c_guess or cols[0]

    # ── التحقق من جودة الملف — لا يصمت النظام عند الخطأ ─────────────
    issues = validate_upload(raw, d_guess, a_guess)
    if issues:
        for iss in issues:
            if iss.startswith("❌"):
                st.error(iss)
            else:
                st.warning(iss)
        if any(i.startswith("❌") for i in issues):
            return   # لا تكمل إذا كانت أعمدة أساسية مفقودة

    with st.expander("⚙️ ربط الأعمدة يدوياً (اختياري — الكشف تلقائي)"):
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            date_col = st.selectbox("عمود التاريخ", cols,
                                    index=cols.index(date_col) if date_col in cols else 0)
        with mc2:
            amount_col = st.selectbox("عمود المبلغ", cols,
                                      index=cols.index(amount_col) if amount_col in cols else 0)
        with mc3:
            cat_col = st.selectbox("عمود الفئة/القناة", cols,
                                   index=cols.index(cat_col) if cat_col in cols else 0)

    # كشف عمود الحساب إن وُجد (للتصنيف الاحتيالي)
    account_col = next((c for c in cols
                        if any(w in str(c).lower()
                               for w in ["account","حساب","acct","account_id"])), None)
    # الحساب نفسه قد يلتقط counterparty_id — استبعده ثم اكشف عمود الطرف المقابل
    if account_col and "counterparty" in str(account_col).lower():
        account_col = next((c for c in cols if c != account_col and
                            any(w in str(c).lower()
                                for w in ["account", "حساب", "acct"])), None)
    counterparty_col = detect_counterparty_column(raw)

    # ── تنظيف ────────────────────────────────────────────────────────
    try:
        df, quality = prepare(raw, date_col, amount_col, cat_col)
    except ValueError as e:
        st.error(str(e)); return

    if quality["dropped"]:
        st.caption(f"تنظيف البيانات: حُفظ {quality['kept']:,} صف، "
                   f"استُبعد {quality['dropped']:,} صف غير صالح.")
    if quality.get("duplicates"):
        st.warning(f"⚠️ رُصد **{quality['duplicates']:,} صف مكرّر** تماماً "
                   f"وأُزيل تلقائياً لتفادي ازدواج التحليل. "
                   f"يُنصح بمراجعة مصدر البيانات لتفادي تكرار القيد.")

    # ── طبقة RBA: تصنيف العملاء من ملفات KYC (Demo فقط — أو None بصدق) ──
    ratings, risk_tiers = None, {}
    if use_demo and account_col:
        profiles = load_customer_profiles()
        ratings = rate_customers(profiles)
        risk_tiers = tier_map(ratings)

    # ── التحليلات ────────────────────────────────────────────────────
    df = detect_anomalies(df, amount_col, date_col, sector.contamination,
                          account_col=account_col, cat_col=cat_col,
                          risk_tiers=risk_tiers)

    # الطبقة الشبكية: تعمل فقط عند توفّر عمودي الحساب والطرف المقابل (بصدق)
    net_res = None
    if account_col and counterparty_col:
        net_res = run_network_analysis(df, account_col, counterparty_col,
                                       amount_col, date_col, cat_col=cat_col)
    if net_res and net_res["findings"]:
        df = apply_network_flags(df, net_res["findings"])

    # ── حلقة التغذية الراجعة: تُغلَق عبر الجلسات لا داخل جلسة واحدة ─────
    # أول تحميل في أي جلسة (حتى بعد إغلاق المتصفح): تُقرأ الذاكرة التراكمية
    # من القرص وتُطبَّق تلقائياً — النظام «يبدأ أذكى» من كل دورة مراجعة
    # سابقة دون أي تدخل. لا صلة لهذا بملف عمليات حقيقي (Demo فقط، بصدق).
    feedback_history = (load_feedback_history(FEEDBACK_MEMORY_PATH) if use_demo
                        else load_feedback_history(None))
    if "fb_decisions" not in st.session_state:
        st.session_state["fb_decisions"] = (history_to_decisions(feedback_history)
                                            if use_demo else {})
    fb_decisions = st.session_state["fb_decisions"]
    df, fb_closed, fb_confirmed = apply_feedback(df, fb_decisions, account_col)
    # حقيقة أرضية لمنحنى التعلّم — Demo فقط، وإلا None بصدق (لا رقم بلا أساس)
    truth_mask = (df[account_col].astype(str).str.match(r"SA-90\d\d")
                  if use_demo and account_col else None)

    anomalies = df[df["is_anomaly"]].sort_values("risk_score", ascending=False)
    exposure  = float(anomalies[amount_col].abs().sum())

    fc        = forecast_cashflow(df, amount_col, date_col, horizon=14)
    slope     = fc["slope"] if fc else 0.0
    fc_mean   = float(fc["forecast"]["forecast"].mean()) if fc else float(df[amount_col].mean())
    trend_txt = "في اتجاه صاعد" if slope > 0 else "في اتجاه هابط"
    top_cat   = (anomalies[cat_col].mode().iloc[0]
                 if not anomalies.empty else df[cat_col].mode().iloc[0])

    # تفكيك أنواع الاحتيال + التدفق داخل/خارج لإثراء السياق
    ft_counts = anomalies["fraud_type"].value_counts() if not anomalies.empty else {}
    flow = split_cashflow(df, amount_col, cat_col, date_col)

    ctx = {"n": len(df), "anoms": len(anomalies), "exposure": exposure,
           "slope": slope, "fc_mean": fc_mean, "trend_txt": trend_txt, "top_cat": top_cat,
           "struct":     int(ft_counts.get("تجزئة عمليات (Structuring)", 0)),
           "velocity":   int(ft_counts.get("تكرار مشبوه (Velocity)", 0)),
           "roundtrip":  int(ft_counts.get("دوران مشبوه (Round-trip Proxy)", 0)),
           "behavioral": int(ft_counts.get("انحراف سلوكي (Behavioral Deviation)", 0)),
           "net_cycle":  int(ft_counts.get("دوران شبكي (Closed Loop)", 0)),
           "net_mule":   int(ft_counts.get("شبكة تجميع وتمرير (Mule Network)", 0)),
           "net_pass":   int(ft_counts.get("تمرير سريع (Rapid Pass-through)", 0)),
           "rba_high":   int((ratings["rba_tier"] == "عالي").sum()) if ratings is not None else 0,
           "rba_med":    int((ratings["rba_tier"] == "متوسط").sum()) if ratings is not None else 0,
           "rba_low":    int((ratings["rba_tier"] == "منخفض").sum()) if ratings is not None else 0,
           "fb_closed": fb_closed, "fb_confirmed": fb_confirmed,
           "fb_cycles_total": int(feedback_history["cycle"].nunique()) if len(feedback_history) else 0,
           "fb_suppressed_total": int((feedback_history["verdict"] == VERDICT_FP).sum()) if len(feedback_history) else 0,
           "inflow": flow["inflow"], "outflow": flow["outflow"], "net": flow["net"]}

    # status bar + tabs
    render_status_bar(ctx, sector)

    # قاموس مبسّط لغير المصرفيين — المحكّمون ليسوا جميعاً من خلفية مصرفية
    with st.expander("📖 اشرح لي المصطلحات ببساطة (لغير المصرفيين)"):
        st.markdown("""
<div dir='rtl' style='line-height:2;color:#44607E;font-size:.9rem;'>
<b style='color:#0E1C30;'>غسيل الأموال:</b> إدخال أموال قذرة (من جريمة) إلى النظام المصرفي بحيث تبدو نظيفة — عبر تفتيتها أو تدويرها أو تمريرها بين حسابات.<br>
<b style='color:#0E1C30;'>بلاغ الاشتباه (STR):</b> تقرير رسمي يرفعه البنك <b>فوراً</b> للإدارة العامة للتحريات المالية عند الاشتباه بعملية — مثل بلاغ للشرطة لكن ماليّاً.<br>
<b style='color:#0E1C30;'>التجزئة (Structuring):</b> تقسيم مبلغ كبير إلى دفعات صغيرة تحت عتبة الرقابة حتى لا يلفت الانتباه — مثل تهريب حمولة شاحنة في حقائب صغيرة.<br>
<b style='color:#0E1C30;'>التكرار المشبوه (Velocity):</b> عمليات كثيرة جداً في وقت قصير جداً من حساب واحد.<br>
<b style='color:#0E1C30;'>الدوران المغلق (Closed Loop):</b> نفس المال يلفّ بين ثلاثة حسابات ويرجع لصاحبه — هدفه تضييع أثر المصدر.<br>
<b style='color:#0E1C30;'>شبكة التجميع (Mule Network):</b> عدة أشخاص («بغال أموال») يحوّلون مبالغ صغيرة لحساب واحد يجمعها ثم يرسلها للخارج.<br>
<b style='color:#0E1C30;'>الدقة (Precision):</b> من كل 100 إنذار أطلقها النظام — كم كان صحيحاً؟<br>
<b style='color:#0E1C30;'>الاستدعاء (Recall):</b> من كل 100 عملية غسيل حقيقية — كم أمسك النظام؟<br>
<b style='color:#0E1C30;'>F1:</b> رقم واحد يوازن الاثنين معاً — كلما ارتفع كان الكاشف أفضل.<br>
<b style='color:#0E1C30;'>النهج القائم على المخاطر (RBA):</b> ليس كل العملاء سواء — عميل بحساب جديد ونشاط نقدي كثيف يستحق رقابة أشد من موظف براتب ثابت؛ فتتشدد عتبات المراقبة تلقائياً حسب تصنيفه.<br>
<b style='color:#0E1C30;'>حلقة التغذية الراجعة:</b> عندما يحكم المراجع البشري أن إنذاراً «كاذب»، يتغذى النظام بقراره فوراً — يُقفل الإنذار وترتفع الدقة، ويبقى القرار مسجلاً للتدقيق.
</div>""", unsafe_allow_html=True)

    _tab_agents, _tab_compliance, _tab_memory, _tab_analysis = st.tabs(
        ["🤖 غرفة الوكلاء", "🛡️ لوحة الامتثال",
         "🧠 الذاكرة والتعلّم", "📊 التحليل التفصيلي"])
    with _tab_agents:
        render_agent_room(ctx, anomalies, fc, flow, sector, amount_col, date_col,
                          net_res)
    with _tab_compliance:
        render_compliance_dashboard(df, anomalies, amount_col, date_col,
                                    account_col, cat_col, ctx, sector,
                                    net_res=net_res, ratings=ratings,
                                    feedback_history=feedback_history,
                                    truth_mask=truth_mask, use_demo=use_demo)
    with _tab_memory:
        # الذاكرة عبر الجلسات + محطة المراجعة البشرية — التمايز الجوهري، تبويب مستقل
        # يظهر فور فتحه بدل أن يُدفَن في قاع لوحة الامتثال.
        render_feedback_panel(anomalies, account_col, amount_col, ctx, df_full=df,
                              truth_mask=truth_mask,
                              feedback_history=feedback_history, use_demo=use_demo)
    with _tab_analysis:
        # ── المؤشرات القيادية ────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("العمليات المُحلّلة",    f"{len(df):,}")
        m2.metric(sector.anomaly_metric,   f"{len(anomalies):,}")
        m3.metric(f"{sector.protected_metric} ({sector.currency})", f"{exposure:,.0f}")
        m4.metric("اتجاه التدفق النقدي",   "صاعد ↑" if slope > 0 else "هابط ↓")
        st.divider()

        # ── التحليل السلوكي ──────────────────────────────────────────────
        st.markdown("<div class='section-head'>التحليل السلوكي للتدفقات</div>"
                    "<div class='section-sub'>Behavioral Flow Analysis</div>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.3])
        with c1:
            by_cat = (df.groupby(cat_col)[amount_col].sum().abs()
                      .sort_values(ascending=False).head(8))
            fig = go.Figure(go.Bar(x=by_cat.values, y=by_cat.index,
                                   orientation="h", marker_color="#1E7D58"))
            fig.update_layout(title="توزيع القيمة حسب الفئة/القناة", **PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            normal = df[~df["is_anomaly"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=normal[date_col], y=normal[amount_col],
                                     mode="markers", name="عمليات اعتيادية",
                                     marker=dict(color="#2E5F8A", size=5, opacity=0.6)))
            if not anomalies.empty:
                fig.add_trace(go.Scatter(x=anomalies[date_col], y=anomalies[amount_col],
                                         mode="markers", name="عمليات شاذة",
                                         marker=dict(color="#C0392B", size=11, symbol="x",
                                                     line=dict(width=1, color="#fff"))))
            fig.update_layout(title="خريطة الشذوذ الزمنية", **PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        # ── التنبؤ ───────────────────────────────────────────────────────
        st.divider()
        st.markdown("<div class='section-head'>التنبؤ بالتدفق النقدي</div>"
                    "<div class='section-sub'>Cash-flow Forecast · trend + weekly seasonality + 95% CI</div>",
                    unsafe_allow_html=True)
        if fc:
            hist_s = fc["history"]; fdf = fc["forecast"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["upper"], mode="lines",
                                     line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["lower"], mode="lines",
                                     fill="tonexty", fillcolor="rgba(201,162,39,0.15)",
                                     line=dict(width=0), name="نطاق الثقة 95%"))
            fig.add_trace(go.Scatter(x=hist_s.index[-45:], y=hist_s.values[-45:],
                                     mode="lines", name="السلوك التاريخي",
                                     line=dict(color="#2E5F8A", width=2)))
            fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["forecast"],
                                     mode="lines+markers", name="التنبؤ",
                                     line=dict(color="#8A6D0F", width=3)))
            fig.update_layout(title="مسار التدفق النقدي المتوقع — 14 يوماً", **PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("النموذج يفصل ثلاثة مكوّنات: الاتجاه العام، "
                       "النمط الأسبوعي المتكرّر، ونطاق عدم اليقين 95%.")
        else:
            st.warning("نطاق البيانات قصير للتنبؤ (يلزم 14 يوماً على الأقل).")

        # ── تقسيم التدفق النقدي: داخل / خارج / صافي ───────────────────────
        st.divider()
        st.markdown("<div class='section-head'>التدفق النقدي: الداخل والخارج</div>"
                    "<div class='section-sub'>Cash-flow Direction · Inflow vs Outflow</div>",
                    unsafe_allow_html=True)

        # التحقق من وجود عمود نوع العملية للتصنيف
        has_type_col = cat_col and cat_col in df.columns and df[cat_col].nunique() > 1
        if not has_type_col:
            st.warning("لتصنيف التدفق داخل/خارج، حدّد **عمود نوع العملية** "
                       "من «ربط الأعمدة يدوياً» أعلاه.")
        else:
            net = flow["net"]
            net_color = "#1E7D58" if net >= 0 else "#C0392B"
            net_label = "موجب — فائض" if net >= 0 else "سالب — عجز"

            # مؤشر صافي التدفق
            mcf1, mcf2, mcf3 = st.columns(3)
            mcf1.metric("إجمالي الداخل", f"{flow['inflow']:,.0f} {sector.currency}")
            mcf2.metric("إجمالي الخارج", f"{flow['outflow']:,.0f} {sector.currency}")
            _glow = "46,143,110" if net >= 0 else "192,57,43"
            mcf3.markdown(
                f"<div dir='rtl' style='position:relative;overflow:hidden;"
                f"background:radial-gradient(120% 80% at 50% -10%, rgba({_glow},0.18), transparent 60%),"
                f"linear-gradient(160deg,#F6F9FC,#EEF3F9);"
                f"border:1px solid {net_color};border-radius:18px;padding:24px 22px;"
                f"box-shadow:0 10px 32px rgba(20,45,80,0.10), 0 0 30px rgba({_glow},0.18);'>"
                f"<div style='position:absolute;top:0;left:0;right:0;height:3px;background:{net_color};'></div>"
                f"<div style='color:#44607E;font-size:.92rem;font-weight:500;margin-bottom:6px;'>"
                f"صافي التدفق ({net_label})</div>"
                f"<div style='font-size:2.1rem;font-weight:800;color:{net_color};"
                f"text-shadow:0 2px 12px rgba(20,45,80,0.12);'>"
                f"{net:,.0f} {sector.currency}</div></div>", unsafe_allow_html=True)

            # رسمان منفصلان: داخل (أخضر) وخارج (أحمر)
            di, do = flow["daily_in"], flow["daily_out"]
            cf_in, cf_out = st.columns(2)
            with cf_in:
                fig = go.Figure(go.Bar(x=list(di.index), y=di.values,
                                       marker_color="#1E7D58", name="الداخل"))
                fig.update_layout(title="التدفق الداخل اليومي", **PLOT_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            with cf_out:
                fig = go.Figure(go.Bar(x=list(do.index), y=do.values,
                                       marker_color="#C0392B", name="الخارج"))
                fig.update_layout(title="التدفق الخارج اليومي", **PLOT_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        # مؤشرات الامتثال (STR/CTR/نسبة الامتثال) تُعرض في تبويب «لوحة الامتثال»
        # عبر render_compliance_dashboard — لا نكرّرها هنا لتفادي ازدواج بصيغة
        # مختلفة. نحسب تصنيف المحفزات التنظيمية مرة واحدة فقط لإعادة استخدامه
        # أدناه في جدول التدقيق وتصدير Excel (بدل استدعاء classify_anomaly ثلاث مرات).
        reg_tags = classify_anomaly(anomalies, df, amount_col, date_col) if not anomalies.empty else []

        # ── سجل التدقيق ──────────────────────────────────────────────────
        st.divider()
        st.markdown("<div class='section-head'>سجل التدقيق وكشف الثغرات</div>"
                    "<div class='section-sub'>Audit & Anomaly Log — ranked by risk score · SAMA Article Reference</div>",
                    unsafe_allow_html=True)
        if not anomalies.empty:
            st.error(sector.anomaly_alert)

            # إحصاء أنواع الاحتيال للعرض السريع
            ft_counts = anomalies["fraud_type"].value_counts()
            fc1, fc2, fc3, fc4 = st.columns(4)
            _ft_card = ("<div dir='rtl' style='background:#F6F9FC;border:1px solid {b};"
                        "border-radius:12px;padding:14px 16px;'>"
                        "<div style='font-size:1.5rem;font-weight:800;color:{c};'>{n}</div>"
                        "<div style='color:#0E1C30;font-size:.85rem;font-weight:600;'>{t}</div></div>")
            with fc1:
                n = ft_counts.get("تجزئة عمليات (Structuring)", 0)
                st.markdown(_ft_card.format(b="rgba(201,162,39,0.4)", c="#8A6D0F",
                    n=n, t="تجزئة عمليات"), unsafe_allow_html=True)
            with fc2:
                n = ft_counts.get("تكرار مشبوه (Velocity)", 0)
                st.markdown(_ft_card.format(b="rgba(217,83,79,0.4)", c="#C0392B",
                    n=n, t="تكرار مشبوه"), unsafe_allow_html=True)
            with fc3:
                n = ft_counts.get("دوران مشبوه (Round-trip Proxy)", 0)
                st.markdown(_ft_card.format(b="rgba(20,108,148,0.4)", c="#146C94",
                    n=n, t="دوران مشبوه"), unsafe_allow_html=True)
            with fc4:
                n = ft_counts.get("انحراف سلوكي (Behavioral Deviation)", 0)
                st.markdown(_ft_card.format(b="rgba(46,143,110,0.4)", c="#1E7D58",
                    n=n, t="انحراف سلوكي"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # جدول التدقيق الموسّع — مع جملة ساما الثابتة لكل عملية
            show = anomalies[[date_col, cat_col, amount_col, "fraud_type", "risk_score"]].copy()
            show["تنبيه ساما"]     = [sama_alert_for(ft) for ft in anomalies["fraud_type"]]
            show["المرجع التنظيمي"] = [sama_ref(t)["ref"]    for t in reg_tags]
            show["موعد الإبلاغ"]   = [
                str(calc_str_deadline(r[date_col], sama_ref(t)["deadline_days"]))
                for (_, r), t in zip(anomalies.iterrows(), reg_tags)
            ]
            show.rename(columns={
                date_col:    "التاريخ",
                cat_col:     "الفئة/القناة",
                amount_col:  "المبلغ (ريال)",
                "fraud_type":"نوع الاحتيال",
                "risk_score":"درجة الخطر",
            }, inplace=True)
            show = show[["التاريخ", "الفئة/القناة", "المبلغ (ريال)",
                         "نوع الاحتيال", "تنبيه ساما",
                         "المرجع التنظيمي", "موعد الإبلاغ", "درجة الخطر"]]
            # تمييز كل صف شاذ بخلفية حمراء فاتحة (#FDECEA) لإبرازه فوراً
            styled = show.style.set_properties(**{
                "background-color": "#FDECEA",
                "color": "#7A1F1A",
            })
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # عرض جمل ساما الثابتة الثلاث صراحةً (مرجع دائم للمحكّم)
            st.markdown("""
            <div dir='rtl' style='background:rgba(192,57,43,0.06);
                 border:1px solid rgba(192,57,43,0.3);border-radius:12px;
                 padding:14px 18px;margin-top:12px;font-size:.85rem;line-height:2;color:#44607E;'>
              <strong style='color:#C0392B;'>الإجراءات التنظيمية وفق ساما حسب نوع الاحتيال:</strong><br>
              • <strong>تجزئة عمليات:</strong> يستوجب رفع بلاغ STR فوراً وفق المادة 15 من نظام مكافحة غسل الأموال<br>
              • <strong>تكرار مشبوه:</strong> يستوجب المراجعة وفق ضوابط مراقبة العمليات غير الاعتيادية<br>
              • <strong>دوران مشبوه (مؤشر أولي):</strong> يستوجب التحقق من مصدر الأموال (القسم 7 · قواعد ساما)<br>
              • <strong>انحراف سلوكي:</strong> يُوصى بالتحقق من هوية العميل وفق متطلبات اعرف عميلك (KYC)
            </div>""", unsafe_allow_html=True)
        else:
            st.success("لم تُرصد عمليات خارج النطاق السلوكي المعياري. الوضع مستقر وفق لوائح ساما.")

        # ── المستشار المالي ──────────────────────────────────────────────
        st.divider()
        st.markdown("<div class='section-head'>المستشار المالي الذكي</div>"
                    "<div class='section-sub'>AI Financial Advisor</div>", unsafe_allow_html=True)

        with st.expander("🔑 ربط OpenAI لتحليل أعمق (اختياري)"):
            _key_input = st.text_input("OpenAI API Key", type="password",
                                       value=st.session_state.get("openai_key", ""),
                                       help="يبقى في الجلسة فقط ولا يُحفظ.")
            if _key_input:
                st.session_state["openai_key"] = _key_input
                api_key = _key_input
                st.markdown("<span class='pill pill-live'>● OpenAI متصل</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<span class='pill pill-sim'>● المحرك المحلي نشط</span>",
                            unsafe_allow_html=True)

        if "chat" not in st.session_state:
            st.session_state.chat = []   # تبدأ فارغة — الترحيب سطر مدمج لا فقاعة

        # سطر ترحيب مدمج وأنيق (يظهر فقط قبل بدء المحادثة)
        if not st.session_state.chat:
            st.markdown(
                f"<div dir='rtl' style='display:flex;align-items:center;gap:10px;"
                f"background:rgba(46,143,110,0.07);border:1px solid rgba(46,143,110,0.25);"
                f"border-radius:12px;padding:10px 16px;color:#44607E;font-size:.88rem;'>"
                f"<span style='font-size:1.1rem;'>💬</span>"
                f"<span>حُلّلت <b style='color:#0E1C30;'>{len(df):,}</b> عملية · "
                f"رُصدت <b style='color:#C0392B;'>{len(anomalies)}</b> مشبوهة · "
                f"التدفق {trend_txt}. اسألني عن الشذوذ، التنبؤ، أو متطلبات ساما.</span>"
                f"</div>", unsafe_allow_html=True)

        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        if q := st.chat_input("اسأل سيجما…"):
            st.session_state.chat.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                with st.spinner("يصيغ التحليل…"):
                    answer = None
                    if api_key:
                        try:
                            h = [{"role": m["role"], "content": m["content"]}
                                 for m in st.session_state.chat[-6:]]
                            answer = openai_advisor(q, h, ctx, sector, api_key)
                        except Exception:
                            st.caption("تعذّر الاتصال بـ OpenAI — التحويل للمحرك المحلي.")
                    if answer is None:
                        answer = local_advisor(q, ctx, sector)
                    st.write(answer)
            st.session_state.chat.append({"role": "assistant", "content": answer})

        # ── التقرير التنفيذي ─────────────────────────────────────────────
        st.divider()
        st.markdown("<div class='section-head'>التقرير التنفيذي والتصدير</div>"
                    "<div class='section-sub'>Executive Report (PDF) · Audit Export (Excel)</div>",
                    unsafe_allow_html=True)

        short_ft = {"تجزئة عمليات (Structuring)": "تجزئة عمليات",
                    "تكرار مشبوه (Velocity)": "تكرار مشبوه",
                    "دوران مشبوه (Round-trip Proxy)": "دوران مشبوه",
                    "انحراف سلوكي (Behavioral Deviation)": "انحراف سلوكي"}
        short_action = {"تجزئة عمليات (Structuring)": "رفع STR",
                        "تكرار مشبوه (Velocity)": "مراجعة",
                        "دوران مشبوه (Round-trip Proxy)": "تحقق المصدر",
                        "انحراف سلوكي (Behavioral Deviation)": "تحقق KYC"}

        # تجهيز صفوف PDF + جدول Excel الكامل
        audit_rows = None
        excel_audit = None
        if not anomalies.empty:
            audit_rows = [
                (str(r[date_col].date()), f"{abs(r[amount_col]):,.0f}",
                 short_ft.get(r["fraud_type"], "انحراف"),
                 f"{r['risk_score']:.0f}",
                 short_action.get(r["fraud_type"], "مراجعة"))
                for _, r in anomalies.head(18).iterrows()
            ]
            excel_audit = pd.DataFrame({
                "التاريخ":         [str(r[date_col].date()) for _, r in anomalies.iterrows()],
                "الحساب":          [str(r[account_col]) if account_col else "—"
                                    for _, r in anomalies.iterrows()],
                "المبلغ (ريال)":   [f"{abs(r[amount_col]):,.0f}" for _, r in anomalies.iterrows()],
                "النوع":           [str(r[cat_col]) for _, r in anomalies.iterrows()],
                "التصنيف":         list(anomalies["fraud_type"]),
                "درجة الخطر":      [f"{r['risk_score']:.0f}" for _, r in anomalies.iterrows()],
                "تنبيه ساما":      [sama_alert_for(ft) for ft in anomalies["fraud_type"]],
                "الموعد النهائي STR": [
                    str(calc_str_deadline(r[date_col], sama_ref(t)["deadline_days"]))
                    for (_, r), t in zip(anomalies.iterrows(), reg_tags)],
            })

        _comp_pct = (len(df) - len(anomalies)) / max(len(df), 1) * 100
        excel_summary = {
            "إجمالي العمليات":   f"{len(df):,}",
            "العمليات المشبوهة": f"{len(anomalies):,}",
            "نسبة الامتثال":     f"{_comp_pct:.1f}%",
            "بلاغات STR المطلوبة (تجزئة)": f"{ctx.get('struct', 0)}",
            "تاريخ التقرير":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        cf_summary = {"inflow": flow["inflow"], "outflow": flow["outflow"], "net": flow["net"]}

        rep_a, rep_b = st.columns(2)
        with rep_a:
            pdf_bytes = build_pdf(ctx, sector, quality,
                                  audit_rows=audit_rows, cashflow=cf_summary)
            if pdf_bytes:
                st.download_button("📄 تحميل تقرير الامتثال (PDF عربي)", data=pdf_bytes,
                                   file_name=f"Sigma_{sector.key}_compliance_report.pdf",
                                   mime="application/pdf", use_container_width=True)
            else:
                st.caption("لتفعيل PDF: pip install reportlab arabic-reshaper python-bidi")
        with rep_b:
            xlsx_bytes = build_excel(excel_audit, excel_summary)
            if xlsx_bytes:
                st.download_button("📥 تصدير Excel (سجل التدقيق)", data=xlsx_bytes,
                                   file_name=f"Sigma_{sector.key}_audit_log.xlsx",
                                   mime=("application/vnd.openxmlformats-officedocument"
                                         ".spreadsheetml.sheet"),
                                   use_container_width=True)
            else:
                st.caption("لتفعيل Excel: pip install openpyxl")


# ============================================================================
#  نقطة الدخول الرئيسية — يوجّه بين الصفحتين
# ============================================================================
def main():
    inject_styles()
    page = st.session_state.get("page", "landing")
    if page == "tool":
        page_tool()
    else:
        page_landing()


if __name__ == "__main__":
    main()
