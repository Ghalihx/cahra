import hashlib
import io
import json

import pandas as pd
import streamlit as st

from modules.preprocessing import preprocess
from modules.absa import analyze_absa
from modules.context import detect_context
from modules.rule_engine import (
    rule_based_reasoning,
    rule_based_dataset_reasoning,
)
from modules.llm import llm_reasoning
from modules.hybrid import hybrid_reasoning
from modules.xai import (
    explain_results,
    explain_single_pipeline,
    explain_dataset_decisions,
)


# ============================================================
# KONFIGURASI
# ============================================================

st.set_page_config(
    page_title="CAHRA",
    page_icon="🌊",
    layout="wide",
)


# ============================================================
# CUSTOM CSS — TEMA LAUT / BAHARI
# ============================================================

st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root palette ── */
:root {
    --ocean-dark:   #023E8A;
    --ocean-mid:    #0077B6;
    --ocean-light:  #00B4D8;
    --ocean-pale:   #ADE8F4;
    --sand:         #F8F9FA;
    --card-bg:      #FFFFFF;
    --text-dark:    #1A1A2E;
    --text-muted:   #6C757D;
    --success:      #2DC653;
    --warning-col:  #F4A261;
    --danger:       #E63946;
    --radius:       14px;
    --shadow:       0 4px 18px rgba(0,119,182,0.10);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #023E8A 0%, #0077B6 60%, #00B4D8 100%);
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 6px 10px;
    margin: 2px 0;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.18);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr {
    border-color: rgba(255,255,255,0.25) !important;
}
[data-testid="stSidebar"] .stCheckbox label {
    font-weight: 500;
}

/* ── Tombol Primary ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0077B6 0%, #00B4D8 100%);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.55rem 1.4rem;
    box-shadow: 0 4px 14px rgba(0,180,216,0.35);
    transition: all 0.25s ease;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(0,119,182,0.45);
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0);
}

/* ── Tombol Secondary / Download ── */
.stDownloadButton > button {
    border: 2px solid var(--ocean-light);
    color: var(--ocean-mid);
    border-radius: 10px;
    font-weight: 500;
    background: transparent;
    transition: all 0.2s;
}
.stDownloadButton > button:hover {
    background: var(--ocean-light);
    color: #fff;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: var(--card-bg);
    border: 1px solid var(--ocean-pale);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    box-shadow: var(--shadow);
}
[data-testid="stMetric"] label {
    color: var(--ocean-mid) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: var(--text-dark) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 2px solid var(--ocean-pale);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-weight: 500;
    font-size: 0.85rem;
    color: var(--text-muted);
    padding: 8px 14px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #EBF5FB;
    color: var(--ocean-mid) !important;
    border-bottom: 3px solid var(--ocean-mid);
    font-weight: 700;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--ocean-pale);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
}

/* ── Info / Warning / Error / Success ── */
[data-testid="stAlert"] {
    border-radius: var(--radius);
    font-size: 0.9rem;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
}

/* ── Code block ── */
.stCode {
    border-radius: var(--radius);
}

/* ── Text Input / Text Area ── */
.stTextInput input, .stTextArea textarea {
    border-radius: 10px;
    border: 1.5px solid var(--ocean-pale);
    font-size: 0.95rem;
    transition: border-color 0.2s;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--ocean-mid);
    box-shadow: 0 0 0 3px rgba(0,119,182,0.15);
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--ocean-light);
    border-radius: var(--radius);
    background: #F0FAFF;
    transition: background 0.2s;
}
[data-testid="stFileUploader"]:hover {
    background: #E0F4FF;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    border-radius: 10px;
    border: 1.5px solid var(--ocean-pale);
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--ocean-mid) !important;
}

/* ── Divider ── */
hr {
    border-color: var(--ocean-pale) !important;
}

/* ── Hero card ── */
.hero-card {
    background: linear-gradient(135deg, #023E8A 0%, #0077B6 55%, #00B4D8 100%);
    border-radius: 20px;
    padding: 2.5rem 2.5rem;
    color: white;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(0,119,182,0.25);
    position: relative;
    overflow: hidden;
}
.hero-card::before {
    content: '🌊';
    font-size: 9rem;
    position: absolute;
    right: 2rem;
    top: -1rem;
    opacity: 0.12;
    line-height: 1;
}
.hero-card h1 {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.02em;
}
.hero-card p {
    font-size: 1.05rem;
    opacity: 0.88;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.20);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* ── Section card ── */
.section-card {
    background: var(--card-bg);
    border-radius: var(--radius);
    border: 1px solid var(--ocean-pale);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow);
}

/* ── Recommendation list ── */
.rec-item {
    background: #EBF5FB;
    border-left: 4px solid var(--ocean-mid);
    border-radius: 0 8px 8px 0;
    padding: 0.55rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
    color: var(--text-dark);
}

/* ── Priority badge ── */
.badge-high {
    display: inline-block;
    background: #FFE5E7;
    color: var(--danger);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 700;
}
.badge-med {
    display: inline-block;
    background: #FFF3E0;
    color: #E65100;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 700;
}
.badge-low {
    display: inline-block;
    background: #E8F5E9;
    color: #2E7D32;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 700;
}

/* ── Pipeline steps ── */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 5px 0;
    font-size: 0.87rem;
}
.pipeline-num {
    background: rgba(255,255,255,0.22);
    border-radius: 50%;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.75rem;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER DATASET
# ============================================================

@st.cache_data(show_spinner=False)
def load_uploaded_reviews(file_bytes, filename):
    """Membaca CSV yang di-upload user saat aplikasi berjalan.

    Dataset tidak disimpan permanen ke source code/server. File hanya dipakai
    selama sesi Streamlit aktif. Kolom wajib bernama `komentar`
    (huruf besar/kecil dan spasi di nama kolom ditoleransi).
    """
    last_error = None

    for encoding in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding=encoding,
                sep=",",
                engine="python",
            )
            break
        except Exception as exc:
            last_error = exc
    else:
        raise ValueError(
            f"CSV '{filename}' tidak dapat dibaca. Detail: {last_error}"
        )

    # Rapikan nama kolom dan toleransi `Komentar`, ` komentar `, dll.
    normalized = {str(col).strip().lower(): col for col in df.columns}
    if "komentar" not in normalized:
        raise ValueError(
            "Dataset wajib memiliki kolom bernama 'komentar'. "
            f"Kolom yang ditemukan: {list(df.columns)}"
        )

    source_col = normalized["komentar"]
    df = df[[source_col]].copy().rename(columns={source_col: "komentar"})
    df["komentar"] = df["komentar"].fillna("").astype(str)
    return df


def reset_dataset_results():
    """Hapus hasil analisis lama saat user mengganti dataset upload."""
    keys = [
        "review_result_df",
        "absa_df",
        "context_df",
        "summary_aspect",
        "context_summary",
        "dataset_rule_result",
        "dataset_llm_result",
    ]
    for key in keys:
        st.session_state.pop(key, None)


@st.cache_data(show_spinner=False)
def analyze_dataset(comments):
    review_rows = []
    absa_rows = []
    context_rows = []

    for idx, original_text in enumerate(comments, start=1):
        original_text = str(original_text)
        clean_text = preprocess(original_text)
        absa_results = analyze_absa(clean_text)
        contexts = detect_context(original_text)

        review_rows.append({
            "No": idx,
            "Komentar Asli": original_text,
            "Hasil Preprocessing": clean_text,
            "Jumlah Aspek": len(absa_results),
            "Jumlah Konteks": len(contexts),
        })

        for item in absa_results:
            absa_rows.append({
                "No": idx,
                "Komentar Asli": original_text,
                "Hasil Preprocessing": clean_text,
                "Aspek": item["Aspek"],
                "Sentimen": item["Sentimen"],
                "Evidence": item["Evidence"],
            })

        for item in contexts:
            context_rows.append({
                "No": idx,
                "Komentar Asli": original_text,
                "Jenis": item["Jenis"],
                "Konteks": item["Konteks"],
            })

    review_df = pd.DataFrame(review_rows)

    absa_df = pd.DataFrame(
        absa_rows,
        columns=[
            "No",
            "Komentar Asli",
            "Hasil Preprocessing",
            "Aspek",
            "Sentimen",
            "Evidence",
        ],
    )

    context_df = pd.DataFrame(
        context_rows,
        columns=[
            "No",
            "Komentar Asli",
            "Jenis",
            "Konteks",
        ],
    )

    return review_df, absa_df, context_df


def summarize_absa(absa_df):
    if absa_df.empty:
        return pd.DataFrame(
            columns=["Aspek", "Positif", "Negatif", "Netral", "Total"]
        )

    summary = (
        absa_df.groupby(["Aspek", "Sentimen"])
        .size()
        .unstack(fill_value=0)
    )

    for sentiment in ["Positif", "Negatif", "Netral"]:
        if sentiment not in summary.columns:
            summary[sentiment] = 0

    summary = summary[["Positif", "Negatif", "Netral"]]
    summary["Total"] = summary.sum(axis=1)
    return summary.reset_index()


def summarize_context(context_df):
    if context_df.empty:
        return pd.DataFrame(columns=["Jenis", "Konteks", "Jumlah"])

    return (
        context_df.groupby(["Jenis", "Konteks"])
        .size()
        .reset_index(name="Jumlah")
        .sort_values("Jumlah", ascending=False)
        .reset_index(drop=True)
    )


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def build_dataset_summary_text(total_reviews, summary_aspect, context_summary):
    return (
        f"Jumlah ulasan: {total_reviews}\n\n"
        "Distribusi sentimen per aspek:\n"
        + summary_aspect.to_string(index=False)
        + "\n\nDistribusi konteks:\n"
        + context_summary.to_string(index=False)
    )


def render_llm_output(llm_result):
    status = llm_result.get("status")

    if status == "disabled":
        st.info(llm_result.get("message", "LLM belum diaktifkan."))
        return

    if status == "error":
        st.error(llm_result.get("message", "LLM gagal."))
        return

    if status != "ok":
        st.info("Belum ada hasil LLM.")
        return

    output = llm_result.get("output") or {}

    st.success(
        f"✅ LLM aktif: **{llm_result.get('provider')}** / `{llm_result.get('model')}`"
    )

    if output.get("ringkasan"):
        st.markdown("**📋 Ringkasan reasoning**")
        st.markdown(
            f"<div class='section-card'>{output['ringkasan']}</div>",
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**⚠️ Isu utama**")
        issues = output.get("isu_utama") or []
        if issues:
            for item in issues:
                st.markdown(
                    f"<div class='rec-item' style='border-left-color:#E63946'>🔴 {item}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Tidak ada isu yang dilaporkan LLM.")

    with c2:
        st.markdown("**💪 Kekuatan utama**")
        strengths = output.get("kekuatan_utama") or []
        if strengths:
            for item in strengths:
                st.markdown(
                    f"<div class='rec-item' style='border-left-color:#2DC653'>🟢 {item}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Tidak ada kekuatan yang dilaporkan LLM.")

    st.markdown("**💡 Rekomendasi LLM**")
    recs = output.get("rekomendasi") or []
    if recs:
        for item in recs:
            st.markdown(
                f"<div class='rec-item'>💡 {item}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Tidak ada rekomendasi LLM.")

    if output.get("keterbatasan"):
        st.caption(f"⚠️ Keterbatasan: {output['keterbatasan']}")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <div style='font-size:2.8rem; line-height:1;'>🌊</div>
        <div style='font-size:1.5rem; font-weight:800; letter-spacing:-0.02em; margin-top:6px;'>CAHRA</div>
        <div style='font-size:0.75rem; opacity:0.8; font-weight:400; line-height:1.4;'>
            Context-Aware Hybrid<br>Reasoning Framework
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

pipeline_steps = [
    ("📥", "Data Collection"),
    ("🔧", "Preprocessing"),
    ("🔍", "Aspect Detection"),
    ("💬", "Aspect Sentiment"),
    ("🌐", "Context Extraction"),
    ("🤖", "LLM Reasoning"),
    ("📏", "Rule-Based Reasoning"),
    ("⚡", "Hybrid Reasoning"),
    ("🔎", "Explainable AI"),
    ("📊", "Decision Support"),
]

steps_html = "<div style='padding: 0.2rem 0;'>"
for i, (icon, label) in enumerate(pipeline_steps, 1):
    steps_html += f"""
    <div class='pipeline-step'>
        <div class='pipeline-num'>{i}</div>
        <span>{icon} {label}</span>
    </div>
    """
steps_html += "</div>"

st.sidebar.markdown(
    f"<div style='font-size:0.78rem; font-weight:700; text-transform:uppercase; "
    f"letter-spacing:0.08em; opacity:0.75; margin-bottom:6px;'>Pipeline Penelitian</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(steps_html, unsafe_allow_html=True)

st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "Mode Analisis",
    ["📊 Analisis Dataset", "📝 Analisis Satu Ulasan"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.78rem; font-weight:700; text-transform:uppercase; "
    "letter-spacing:0.08em; opacity:0.75; margin-bottom:4px;'>🤖 LLM Reasoning</div>",
    unsafe_allow_html=True,
)
use_llm = st.sidebar.checkbox(
    "Aktifkan Google Gemini",
    value=False,
    help=(
        "Tanpa Gemini, CAHRA tetap berjalan dengan Rule Engine. "
        "Jika diaktifkan, Hybrid Reasoning menggabungkan keduanya."
    ),
)

gemini_api_key = ""
gemini_model = "gemini-3.7-flash"

if use_llm:
    gemini_api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="API key hanya digunakan pada sesi aplikasi ini.",
    )
    gemini_model = st.sidebar.text_input(
        "Model Gemini",
        value="gemini-3.7-flash",
        help="Jika model sedang 503/high demand, CAHRA akan retry lalu mencoba model fallback otomatis.",
    )

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.caption("© 2024 Penelitian Wisata Bahari")


# ============================================================
# HERO HEADER
# ============================================================

if mode == "📊 Analisis Dataset":
    hero_subtitle = "Analisis Dataset Ulasan Wisata Bahari"
    hero_desc = "Upload dataset CSV → preprocessing → ABSA → context extraction → reasoning → decision support"
else:
    hero_subtitle = "Analisis Satu Ulasan Wisata Bahari"
    hero_desc = "Masukkan satu ulasan dan jalankan seluruh pipeline secara real-time"

st.markdown(
    f"""
    <div class='hero-card'>
        <div class='hero-badge'>Prototype Penelitian</div>
        <h1>🌊 CAHRA</h1>
        <p><strong>{hero_subtitle}</strong></p>
        <p style='opacity:0.75; font-size:0.88rem; margin-top:6px;'>{hero_desc}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODE 1 - DATASET
# ============================================================

if mode == "📊 Analisis Dataset":

    st.markdown(
        "<p style='color:#6C757D; margin-bottom:1rem;'>"
        "Upload dataset CSV yang ingin dianalisis. Dataset <strong>tidak perlu</strong> "
        "ditaruh atau diganti di source code — hanya digunakan selama sesi ini."
        "</p>",
        unsafe_allow_html=True,
    )

    uploaded_dataset = st.file_uploader(
        "📤 Upload dataset CSV",
        type=["csv"],
        help="CSV wajib memiliki kolom bernama 'komentar'.",
        key="dataset_uploader",
    )

    if uploaded_dataset is None:
        st.info(
            "📂 Silakan upload file CSV terlebih dahulu. "
            "Kolom wajib: **`komentar`**."
        )
        st.stop()

    file_bytes = uploaded_dataset.getvalue()
    dataset_id = hashlib.sha256(file_bytes).hexdigest()

    # Kalau user mengganti file, jangan tampilkan hasil dataset sebelumnya.
    if st.session_state.get("active_dataset_id") != dataset_id:
        reset_dataset_results()
        st.session_state["active_dataset_id"] = dataset_id

    try:
        reviews_df = load_uploaded_reviews(
            file_bytes,
            uploaded_dataset.name,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    total_data = len(reviews_df)
    komentar_kosong = int(
        reviews_df["komentar"].astype(str).str.strip().eq("").sum()
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("📋 Jumlah Ulasan", f"{total_data:,}")
    c2.metric("📁 Kolom Dataset", "komentar")
    c3.metric("⚠️ Komentar Kosong", f"{komentar_kosong:,}")

    st.caption(
        f"🗂️ Dataset aktif: **{uploaded_dataset.name}** · "
        f"{uploaded_dataset.size / 1024:.1f} KB · hanya untuk sesi ini"
    )

    with st.expander("👀 Lihat contoh data mentah", expanded=False):
        st.dataframe(
            reviews_df.head(50),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "🚀 Jalankan Pipeline Dataset",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner(
            f"⏳ Menganalisis {total_data:,} komentar: preprocessing → ABSA → context → rules..."
        ):
            review_result_df, absa_df, context_df = analyze_dataset(
                tuple(reviews_df["komentar"].tolist())
            )
            summary_aspect = summarize_absa(absa_df)
            context_summary = summarize_context(context_df)
            rule_result = rule_based_dataset_reasoning(
                summary_aspect.to_dict("records"),
                context_summary.to_dict("records"),
            )

        st.session_state["review_result_df"] = review_result_df
        st.session_state["absa_df"] = absa_df
        st.session_state["context_df"] = context_df
        st.session_state["summary_aspect"] = summary_aspect
        st.session_state["context_summary"] = context_summary
        st.session_state["dataset_rule_result"] = rule_result
        st.session_state.pop("dataset_llm_result", None)

    if "review_result_df" in st.session_state:
        review_result_df = st.session_state["review_result_df"]
        absa_df = st.session_state["absa_df"]
        context_df = st.session_state["context_df"]
        summary_aspect = st.session_state["summary_aspect"]
        context_summary = st.session_state["context_summary"]
        rule_result = st.session_state["dataset_rule_result"]
        llm_result = st.session_state.get(
            "dataset_llm_result",
            {
                "status": "disabled",
                "message": "LLM belum dijalankan untuk ringkasan dataset.",
                "output": None,
            },
        )

        hybrid_result = hybrid_reasoning(llm_result, rule_result)
        xai_rows = explain_dataset_decisions(rule_result, hybrid_result)
        decision_df = pd.DataFrame(rule_result.get("decisions", []))
        xai_df = pd.DataFrame(xai_rows)

        st.success("✅ Pipeline utama dataset selesai dijalankan.")

        total_aspek = len(absa_df)
        komentar_terdeteksi_aspek = (
            absa_df["No"].nunique() if not absa_df.empty else 0
        )
        komentar_terdeteksi_konteks = (
            context_df["No"].nunique() if not context_df.empty else 0
        )

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📝 Total Ulasan", f"{len(review_result_df):,}")
        m2.metric("🔍 Temuan Aspek", f"{total_aspek:,}")
        m3.metric("✅ Ulasan dengan Aspek", f"{komentar_terdeteksi_aspek:,}")
        m4.metric("🌐 Ulasan dengan Konteks", f"{komentar_terdeteksi_konteks:,}")

        st.markdown("<br>", unsafe_allow_html=True)

        tabs = st.tabs([
            "1️⃣ Preprocessing",
            "2️⃣ ABSA",
            "3️⃣ Context",
            "4️⃣ Rule Engine",
            "5️⃣ LLM",
            "6️⃣ Hybrid",
            "7️⃣ XAI",
            "8️⃣ Decision",
            "9️⃣ Ringkasan",
        ])

        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------
        with tabs[0]:
            st.markdown("### 🔧 Hasil Preprocessing Dataset")
            st.markdown(
                "<p style='color:#6C757D; font-size:0.88rem;'>"
                "Setiap ulasan melalui tahap pembersihan teks, normalisasi, dan tokenisasi.</p>",
                unsafe_allow_html=True,
            )
            st.dataframe(
                review_result_df,
                use_container_width=True,
                hide_index=True,
                height=500,
            )
            st.download_button(
                "⬇️ Download hasil preprocessing",
                dataframe_to_csv_bytes(review_result_df),
                file_name="hasil_preprocessing_dataset.csv",
                mime="text/csv",
            )

        # ----------------------------------------------------
        # ABSA
        # ----------------------------------------------------
        with tabs[1]:
            st.markdown("### 💬 Aspect-Based Sentiment Analysis")
            st.markdown(
                "<p style='color:#6C757D; font-size:0.88rem;'>"
                "Setiap aspek dalam ulasan dideteksi dan diklasifikasikan sentimennya.</p>",
                unsafe_allow_html=True,
            )
            if absa_df.empty:
                st.warning("⚠️ Tidak ada aspek yang terdeteksi pada dataset.")
            else:
                aspek_options = ["Semua"] + sorted(
                    absa_df["Aspek"].dropna().unique().tolist()
                )
                sentimen_options = ["Semua", "Positif", "Negatif", "Netral"]
                f1, f2 = st.columns(2)
                aspek_filter = f1.selectbox(
                    "🔎 Filter aspek", aspek_options, key="dataset_aspect_filter"
                )
                sentimen_filter = f2.selectbox(
                    "🎭 Filter sentimen", sentimen_options, key="dataset_sentiment_filter"
                )
                filtered = absa_df.copy()
                if aspek_filter != "Semua":
                    filtered = filtered[filtered["Aspek"] == aspek_filter]
                if sentimen_filter != "Semua":
                    filtered = filtered[filtered["Sentimen"] == sentimen_filter]

                st.dataframe(
                    filtered,
                    use_container_width=True,
                    hide_index=True,
                    height=500,
                )
                st.download_button(
                    "⬇️ Download hasil ABSA",
                    dataframe_to_csv_bytes(absa_df),
                    file_name="hasil_absa_dataset.csv",
                    mime="text/csv",
                )

        # ----------------------------------------------------
        # CONTEXT
        # ----------------------------------------------------
        with tabs[2]:
            st.markdown("### 🌐 Context Extraction")
            st.markdown(
                "<p style='color:#6C757D; font-size:0.88rem;'>"
                "Konteks seperti weekend, cuaca, atau musim tertentu dideteksi dari ulasan.</p>",
                unsafe_allow_html=True,
            )
            if context_df.empty:
                st.info("ℹ️ Tidak ditemukan konteks khusus pada dataset.")
            else:
                st.dataframe(
                    context_df,
                    use_container_width=True,
                    hide_index=True,
                    height=500,
                )
                st.download_button(
                    "⬇️ Download hasil context extraction",
                    dataframe_to_csv_bytes(context_df),
                    file_name="hasil_context_dataset.csv",
                    mime="text/csv",
                )

        # ----------------------------------------------------
        # RULE ENGINE
        # ----------------------------------------------------
        with tabs[3]:
            st.markdown("### 📏 Rule-Based Reasoning")
            st.markdown(
                "<p style='color:#6C757D; font-size:0.88rem;'>"
                "Rule engine menentukan prioritas berdasarkan rasio sentimen negatif "
                "per aspek dan konteks seperti Weekend/Holiday atau Weather.</p>",
                unsafe_allow_html=True,
            )
            if decision_df.empty:
                st.info("ℹ️ Belum ada keputusan rule-based.")
            else:
                st.dataframe(
                    decision_df,
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "⬇️ Download hasil Rule Engine",
                    dataframe_to_csv_bytes(decision_df),
                    file_name="hasil_rule_engine_dataset.csv",
                    mime="text/csv",
                )

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------
        with tabs[4]:
            st.markdown("### 🤖 LLM Reasoning (Google Gemini)")
            st.markdown(
                "<p style='color:#6C757D; font-size:0.88rem;'>"
                "Untuk dataset besar, Gemini membaca ringkasan agregat ABSA/context, "
                "bukan mengirim seluruh komentar satu per satu.</p>",
                unsafe_allow_html=True,
            )

            if not use_llm:
                st.info(
                    "💡 Aktifkan **Google Gemini** di sidebar untuk menjalankan LLM Reasoning."
                )
            elif not gemini_api_key:
                st.warning("🔑 Masukkan **Gemini API Key** di sidebar.")
            else:
                if st.button(
                    "🤖 Jalankan LLM pada Ringkasan Dataset",
                    type="primary",
                    key="run_dataset_llm",
                ):
                    summary_text = build_dataset_summary_text(
                        len(review_result_df),
                        summary_aspect,
                        context_summary,
                    )
                    with st.spinner("🔄 Gemini sedang melakukan reasoning..."):
                        st.session_state["dataset_llm_result"] = llm_reasoning(
                            "",
                            api_key=gemini_api_key,
                            model=gemini_model,
                            dataset_summary=summary_text,
                        )
                    st.rerun()

            render_llm_output(llm_result)

        # ----------------------------------------------------
        # HYBRID
        # ----------------------------------------------------
        with tabs[5]:
            st.markdown("### ⚡ Hybrid Reasoning")
            h1, h2, h3 = st.columns(3)
            h1.metric("🔀 Mode", hybrid_result.get("mode", "-"))
            h2.metric("🎯 Confidence", hybrid_result.get("confidence", "-"))
            h3.metric(
                "🤖 LLM Digunakan",
                "Ya" if hybrid_result.get("llm_used") else "Tidak",
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='section-card'>{hybrid_result.get('explanation', '')}</div>",
                unsafe_allow_html=True,
            )

            if hybrid_result.get("summary"):
                st.markdown("**📋 Ringkasan Hybrid**")
                st.markdown(
                    f"<div class='section-card'>{hybrid_result['summary']}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("**💡 Rekomendasi gabungan**")
            recs = hybrid_result.get("recommendations", [])
            if recs:
                for i, rec in enumerate(recs, 1):
                    st.markdown(
                        f"<div class='rec-item'>{i}. {rec}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("ℹ️ Belum ada rekomendasi hybrid.")

        # ----------------------------------------------------
        # XAI
        # ----------------------------------------------------
        with tabs[6]:
            st.markdown("### 🔎 Explainable AI (XAI)")
            st.markdown(
                "<p style='color:#6C757D; font-size:0.88rem;'>"
                "Bagian ini menunjukkan dasar keputusan: jumlah temuan negatif, "
                "rasio negatif, konteks penguat, dan rekomendasi yang dihasilkan.</p>",
                unsafe_allow_html=True,
            )
            if xai_df.empty:
                st.info("ℹ️ Belum ada penjelasan XAI.")
            else:
                st.dataframe(
                    xai_df,
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "⬇️ Download XAI",
                    dataframe_to_csv_bytes(xai_df),
                    file_name="hasil_xai_dataset.csv",
                    mime="text/csv",
                )

        # ----------------------------------------------------
        # DECISION SUPPORT
        # ----------------------------------------------------
        with tabs[7]:
            st.markdown("### 📊 Decision Support")
            if decision_df.empty:
                st.info("ℹ️ Belum ada keputusan yang dapat ditampilkan.")
            else:
                priority_df = decision_df[
                    [
                        "Aspek",
                        "Prioritas",
                        "Rasio Negatif",
                        "Konteks Penguat",
                        "Rekomendasi",
                    ]
                ].copy()
                st.dataframe(
                    priority_df,
                    use_container_width=True,
                    hide_index=True,
                )

                high = priority_df[priority_df["Prioritas"] == "Tinggi"]
                if not high.empty:
                    st.error("🚨 **Prioritas utama pengelola:**")
                    for _, row in high.iterrows():
                        st.markdown(
                            f"<div class='rec-item' style='border-left-color:#E63946;'>"
                            f"<span class='badge-high'>TINGGI</span>&nbsp; "
                            f"<strong>{row['Aspek']}</strong>: {row['Rekomendasi']}</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.success("✅ Tidak ada aspek yang masuk prioritas Tinggi menurut rule engine.")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**💡 Rekomendasi Hybrid Final**")
                for rec in hybrid_result.get("recommendations", []):
                    st.markdown(
                        f"<div class='rec-item'>💡 {rec}</div>",
                        unsafe_allow_html=True,
                    )

        # ----------------------------------------------------
        # RINGKASAN
        # ----------------------------------------------------
        with tabs[8]:
            st.markdown("### 📈 Ringkasan Hasil Dataset")
            st.markdown("**📊 Distribusi sentimen per aspek**")
            st.dataframe(
                summary_aspect,
                use_container_width=True,
                hide_index=True,
            )

            if not summary_aspect.empty:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**📉 Grafik sentimen per aspek**")
                chart_data = summary_aspect.set_index("Aspek")[["Positif", "Negatif", "Netral"]]
                st.bar_chart(
                    chart_data,
                    color=["#2DC653", "#E63946", "#ADB5BD"],
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**🌐 Distribusi konteks**")
            st.dataframe(
                context_summary,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# MODE 2 - SATU ULASAN
# ============================================================

else:
    st.markdown(
        "<p style='color:#6C757D; margin-bottom:1rem;'>"
        "Masukkan satu ulasan wisata bahari dan jalankan seluruh pipeline analisis secara real-time.</p>",
        unsafe_allow_html=True,
    )

    review = st.text_area(
        "✍️ Masukkan ulasan wisata:",
        value="Pantainya indah tetapi banyak sampah saat weekend",
        height=130,
        placeholder="Contoh: Pantainya sangat bersih dan pemandangan bawah lautnya menakjubkan!",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Jalankan Pipeline Lengkap", type="primary", use_container_width=True):
        if not review.strip():
            st.warning("⚠️ Silakan masukkan ulasan terlebih dahulu.")
            st.stop()

        clean_text = preprocess(review)
        results = analyze_absa(clean_text)
        contexts = detect_context(review)
        rule_result = rule_based_reasoning(results, contexts)

        if use_llm and gemini_api_key:
            with st.spinner("🔄 Gemini sedang melakukan reasoning..."):
                llm_result = llm_reasoning(
                    review,
                    results,
                    contexts,
                    api_key=gemini_api_key,
                    model=gemini_model,
                )
        else:
            llm_result = {
                "status": "disabled",
                "provider": "Google Gemini",
                "model": gemini_model,
                "message": (
                    "LLM tidak aktif. Hybrid menggunakan Rule-Based fallback."
                ),
                "output": None,
            }

        hybrid_result = hybrid_reasoning(llm_result, rule_result)
        xai_rows = explain_single_pipeline(
            review,
            rule_result,
            llm_result,
            hybrid_result,
        )

        st.success("✅ Pipeline lengkap selesai dijalankan.")

        single_tabs = st.tabs([
            "1️⃣ Preprocessing",
            "2️⃣ ABSA",
            "3️⃣ Context",
            "4️⃣ Rule Engine",
            "5️⃣ LLM",
            "6️⃣ Hybrid",
            "7️⃣ XAI",
            "8️⃣ Decision",
        ])

        with single_tabs[0]:
            st.markdown("### 🔧 Hasil Preprocessing")
            st.markdown(
                "<p style='color:#6C757D; font-size:0.88rem;'>"
                "Teks asli setelah melewati tahap pembersihan dan normalisasi:</p>",
                unsafe_allow_html=True,
            )
            st.code(clean_text, language="text")

        with single_tabs[1]:
            st.markdown("### 💬 Aspect-Based Sentiment Analysis")
            if results:
                st.dataframe(
                    pd.DataFrame(results),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning("⚠️ Tidak ditemukan aspek pada ulasan ini.")

        with single_tabs[2]:
            st.markdown("### 🌐 Context Extraction")
            if contexts:
                st.dataframe(
                    pd.DataFrame(contexts),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("ℹ️ Tidak ditemukan konteks khusus.")

        with single_tabs[3]:
            st.markdown("### 📏 Rule-Based Reasoning")
            rule_df = pd.DataFrame(rule_result.get("decisions", []))
            if rule_df.empty:
                st.info("ℹ️ Belum ada aspek untuk diproses rule engine.")
            else:
                st.dataframe(
                    rule_df,
                    use_container_width=True,
                    hide_index=True,
                )

        with single_tabs[4]:
            st.markdown("### 🤖 LLM Reasoning")
            render_llm_output(llm_result)

        with single_tabs[5]:
            st.markdown("### ⚡ Hybrid Reasoning")
            hc1, hc2 = st.columns(2)
            hc1.metric("🔀 Mode", hybrid_result.get("mode", "-"))
            hc2.metric("🎯 Confidence", hybrid_result.get("confidence", "-"))
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='section-card'>{hybrid_result.get('explanation', '')}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("**💡 Rekomendasi**")
            for rec in hybrid_result.get("recommendations", []):
                st.markdown(
                    f"<div class='rec-item'>💡 {rec}</div>",
                    unsafe_allow_html=True,
                )

        with single_tabs[6]:
            st.markdown("### 🔎 Explainable AI")
            if xai_rows:
                st.dataframe(
                    pd.DataFrame(xai_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("ℹ️ Belum ada penjelasan XAI.")

        with single_tabs[7]:
            st.markdown("### 📊 Decision Support")
            recs = hybrid_result.get("recommendations", [])
            if recs:
                for i, rec in enumerate(recs, 1):
                    st.markdown(
                        f"<div class='rec-item'>{i}. {rec}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("ℹ️ Belum ada rekomendasi decision support.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.caption(
                "⚠️ Decision Support adalah rekomendasi penelitian, bukan keputusan otomatis final."
            )
