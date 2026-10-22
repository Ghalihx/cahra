import hashlib
import io
import json
import os

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


def _load_local_env():
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except Exception:
            pass

_load_local_env()


# ============================================================
# KONFIGURASI
# ============================================================

st.set_page_config(
    page_title="CAHRA",
    page_icon="🌊",
    layout="wide",
)

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
    cols_to_keep = [source_col]
    label_source = None
    for lbl in ["sentimen_aktual", "label", "sentimen", "actual"]:
        if lbl in normalized:
            label_source = normalized[lbl]
            cols_to_keep.append(label_source)
            break

    df = df[cols_to_keep].copy()
    rename_dict = {source_col: "komentar"}
    if label_source:
        rename_dict[label_source] = "sentimen_aktual"
    df = df.rename(columns=rename_dict)
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
        "dataset_eval_data",
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


@st.cache_data(show_spinner=False)
def get_evaluation_metrics(comments_tuple, labels_tuple=None):
    from evaluate import predict_sentiment
    from modules.absa import detect_sentiment
    from modules.preprocessing import preprocess
    from sklearn.metrics import confusion_matrix, classification_report

    y_true = []
    y_pred = []
    errors = []

    has_custom_labels = (
        labels_tuple is not None
        and len(labels_tuple) == len(comments_tuple)
        and any(str(l).strip() for l in labels_tuple if l is not None)
    )

    for idx, comment in enumerate(comments_tuple, start=1):
        clean = preprocess(str(comment))
        pred = predict_sentiment(clean)

        if has_custom_labels and labels_tuple[idx - 1] and str(labels_tuple[idx - 1]).strip():
            actual = str(labels_tuple[idx - 1]).strip().capitalize()
            label_col_name = "Sentimen Aktual"
        else:
            actual = detect_sentiment(clean)
            label_col_name = "Sentimen Acuan (Baseline)"

        y_true.append(actual)
        y_pred.append(pred)

        if actual != pred:
            errors.append({
                "No": idx,
                "Komentar": comment,
                label_col_name: actual,
                "Sentimen Prediksi": pred,
            })

    labels = ["Positif", "Negatif", "Netral"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Aktual {l}" for l in labels],
        columns=[f"Prediksi {l}" for l in labels],
    )
    rep_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    rep_df = pd.DataFrame(rep_dict).transpose()

    return {
        "cm_df": cm_df,
        "cm": cm,
        "rep_df": rep_df,
        "errors": errors,
        "total": len(comments_tuple),
        "acc": rep_dict.get("accuracy", 0.0),
        "macro_f1": rep_dict.get("macro avg", {}).get("f1-score", 0.0),
        "has_custom_labels": has_custom_labels,
    }


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
        f"LLM aktif: {llm_result.get('provider')} / {llm_result.get('model')}"
    )

    if output.get("ringkasan"):
        st.write("**Ringkasan reasoning**")
        st.write(output["ringkasan"])

    c1, c2 = st.columns(2)

    with c1:
        st.write("**Isu utama**")
        issues = output.get("isu_utama") or []
        if issues:
            for item in issues:
                st.write(f"- {item}")
        else:
            st.caption("Tidak ada isu yang dilaporkan LLM.")

    with c2:
        st.write("**Kekuatan utama**")
        strengths = output.get("kekuatan_utama") or []
        if strengths:
            for item in strengths:
                st.write(f"- {item}")
        else:
            st.caption("Tidak ada kekuatan yang dilaporkan LLM.")

    st.write("**Rekomendasi LLM**")
    recs = output.get("rekomendasi") or []
    if recs:
        for item in recs:
            st.write(f"- {item}")
    else:
        st.caption("Tidak ada rekomendasi LLM.")

    if output.get("keterbatasan"):
        st.caption(f"Keterbatasan: {output['keterbatasan']}")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌊 CAHRA")
st.sidebar.write("Context-Aware Hybrid Reasoning Framework")
st.sidebar.markdown("---")
st.sidebar.write("**Pipeline Penelitian:**")
st.sidebar.write("1. Data Collection")
st.sidebar.write("2. Preprocessing")
st.sidebar.write("3. Aspect Detection")
st.sidebar.write("4. Aspect Sentiment")
st.sidebar.write("5. Context Extraction")
st.sidebar.write("6. LLM Reasoning")
st.sidebar.write("7. Rule-Based Reasoning")
st.sidebar.write("8. Hybrid Reasoning")
st.sidebar.write("9. Explainable AI")
st.sidebar.write("10. Decision Support")
st.sidebar.write("11. Model Evaluation (Confusion Matrix)")

st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "Mode Analisis",
    [
        "📊 Analisis Dataset",
        "📝 Analisis Satu Ulasan",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 LLM Reasoning")
use_llm = st.sidebar.checkbox(
    "Aktifkan Google Gemini",
    value=False,
    help=(
        "Tanpa Gemini, CAHRA tetap berjalan dengan Rule Engine. "
        "Jika diaktifkan, Hybrid Reasoning menggabungkan keduanya."
    ),
)

# Cek ketersediaan default API Key dari secrets atau env
default_api_key = os.getenv("GEMINI_API_KEY", "")
if not default_api_key:
    try:
        default_api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

default_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
if not default_model or default_model == "gemini-3.7-flash":
    default_model = "gemini-3.8-flash"

gemini_api_key = default_api_key
gemini_model = default_model

if use_llm:
    gemini_api_key = st.sidebar.text_input(
        "Gemini API Key",
        value=default_api_key,
        type="password",
        help="API key terisi otomatis jika ada di .env / secrets, atau masukkan manual.",
    )
    gemini_model = st.sidebar.text_input(
        "Model Gemini",
        value=default_model,
        help="Jika model sedang 503/high demand, CAHRA akan retry lalu mencoba model fallback otomatis.",
    )


# ============================================================
# HEADER
# ============================================================

st.title("🌊 CAHRA")
st.subheader("Context-Aware Aspect-Based Sentiment Analysis")
st.caption("Prototype Penelitian Wisata Bahari Berkelanjutan")
st.divider()


# ============================================================
# MODE 1 - DATASET
# ============================================================

if mode == "📊 Analisis Dataset":
    st.header("📊 Analisis Dataset Ulasan Wisata Bahari")

    st.write(
        "Upload dataset CSV yang ingin dianalisis. Dataset **tidak perlu** "
        "ditaruh atau diganti di source code."
    )

    uploaded_dataset = st.file_uploader(
        "📤 Upload dataset CSV",
        type=["csv"],
        help="CSV wajib memiliki kolom bernama 'komentar'.",
        key="dataset_uploader",
    )

    if uploaded_dataset is None:
        st.info(
            "Silakan upload file CSV terlebih dahulu. "
            "Kolom wajib: `komentar`."
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
    c1.metric("Jumlah Ulasan", f"{total_data:,}")
    c2.metric("Kolom Dataset", "komentar")
    c3.metric("Komentar Kosong", f"{komentar_kosong:,}")

    st.caption(
        f"Dataset aktif: {uploaded_dataset.name} • "
        f"{uploaded_dataset.size / 1024:.1f} KB • hanya untuk sesi ini"
    )

    with st.expander("👀 Lihat contoh data mentah", expanded=False):
        st.dataframe(
            reviews_df.head(50),
            use_container_width=True,
            hide_index=True,
        )

    if st.button(
        "🚀 Jalankan Pipeline Dataset",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner(
            f"Menganalisis {total_data:,} komentar: preprocessing → ABSA → context → rules..."
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

        st.success("Pipeline utama dataset selesai.")

        total_aspek = len(absa_df)
        komentar_terdeteksi_aspek = (
            absa_df["No"].nunique() if not absa_df.empty else 0
        )
        komentar_terdeteksi_konteks = (
            context_df["No"].nunique() if not context_df.empty else 0
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Ulasan", f"{len(review_result_df):,}")
        m2.metric("Temuan Aspek", f"{total_aspek:,}")
        m3.metric("Ulasan dengan Aspek", f"{komentar_terdeteksi_aspek:,}")
        m4.metric("Ulasan dengan Konteks", f"{komentar_terdeteksi_konteks:,}")

        tabs = st.tabs([
            "1. Preprocessing",
            "2. ABSA",
            "3. Context",
            "4. Rule Engine",
            "5. LLM",
            "6. Hybrid",
            "7. XAI",
            "8. Decision Support",
            "9. Ringkasan",
            "10. Evaluasi Model (Confusion Matrix)",
        ])

        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------
        with tabs[0]:
            st.subheader("1. Hasil Preprocessing Dataset")
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
            st.subheader("2. Aspect-Based Sentiment Analysis")
            if absa_df.empty:
                st.warning("Tidak ada aspek yang terdeteksi pada dataset.")
            else:
                pos_count = int((absa_df["Sentimen"] == "Positif").sum())
                neg_count = int((absa_df["Sentimen"] == "Negatif").sum())
                net_count = int((absa_df["Sentimen"] == "Netral").sum())
                total_temuan = len(absa_df)

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Temuan Aspek", f"{total_temuan:,}")
                s2.metric("Sentimen Positif", f"{pos_count:,}", f"{(pos_count/total_temuan)*100:.1f}%" if total_temuan else None)
                s3.metric("Sentimen Negatif", f"{neg_count:,}", f"{(neg_count/total_temuan)*100:.1f}%" if total_temuan else None)
                s4.metric("Sentimen Netral", f"{net_count:,}", f"{(net_count/total_temuan)*100:.1f}%" if total_temuan else None)

                st.markdown("---")

                aspek_options = ["Semua"] + sorted(
                    absa_df["Aspek"].dropna().unique().tolist()
                )
                sentimen_options = ["Semua", "Positif", "Negatif", "Netral"]
                f1, f2 = st.columns(2)
                aspek_filter = f1.selectbox(
                    "Filter aspek", aspek_options, key="dataset_aspect_filter"
                )
                sentimen_filter = f2.selectbox(
                    "Filter sentimen", sentimen_options, key="dataset_sentiment_filter"
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
            st.subheader("3. Context Extraction")
            if context_df.empty:
                st.info("Tidak ditemukan konteks khusus pada dataset.")
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
            st.subheader("4. Rule-Based Reasoning")
            st.write(
                "Rule engine menentukan prioritas berdasarkan rasio sentimen negatif "
                "per aspek dan konteks seperti Weekend/Holiday atau Weather."
            )
            if decision_df.empty:
                st.info("Belum ada keputusan rule-based.")
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
            st.subheader("5. LLM Reasoning (Google Gemini)")
            st.caption(
                "Untuk dataset besar, Gemini membaca ringkasan agregat ABSA/context, "
                "bukan mengirim 16 ribu komentar satu per satu."
            )

            if not use_llm:
                st.info(
                    "Aktifkan 'Google Gemini' di sidebar untuk menjalankan LLM Reasoning."
                )
            elif not gemini_api_key:
                st.warning("Masukkan Gemini API Key di sidebar.")
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
                    with st.spinner("Gemini sedang melakukan reasoning..."):
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
            st.subheader("6. Hybrid Reasoning")
            h1, h2, h3 = st.columns(3)
            h1.metric("Mode", hybrid_result.get("mode", "-"))
            h2.metric("Confidence", hybrid_result.get("confidence", "-"))
            h3.metric(
                "LLM Digunakan",
                "Ya" if hybrid_result.get("llm_used") else "Tidak",
            )

            st.write(hybrid_result.get("explanation", ""))
            if hybrid_result.get("summary"):
                st.write("**Ringkasan Hybrid**")
                st.write(hybrid_result["summary"])

            st.write("**Rekomendasi gabungan**")
            recs = hybrid_result.get("recommendations", [])
            if recs:
                for i, rec in enumerate(recs, 1):
                    st.write(f"{i}. {rec}")
            else:
                st.info("Belum ada rekomendasi hybrid.")

        # ----------------------------------------------------
        # XAI
        # ----------------------------------------------------
        with tabs[6]:
            st.subheader("7. Explainable AI (XAI)")
            st.write(
                "Bagian ini menunjukkan dasar keputusan: jumlah temuan negatif, "
                "rasio negatif, konteks penguat, dan rekomendasi yang dihasilkan."
            )
            if xai_df.empty:
                st.info("Belum ada penjelasan XAI.")
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
            st.subheader("8. Decision Support")
            if decision_df.empty:
                st.info("Belum ada keputusan yang dapat ditampilkan.")
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
                    st.error("Prioritas utama pengelola")
                    for _, row in high.iterrows():
                        st.write(f"- **{row['Aspek']}**: {row['Rekomendasi']}")
                else:
                    st.success("Tidak ada aspek yang masuk prioritas Tinggi menurut rule engine.")

                st.write("**Rekomendasi Hybrid Final**")
                for rec in hybrid_result.get("recommendations", []):
                    st.write(f"- {rec}")

        # ----------------------------------------------------
        # RINGKASAN
        # ----------------------------------------------------
        with tabs[8]:
            st.subheader("9. Ringkasan Hasil Dataset")
            st.write("**Distribusi sentimen per aspek**")
            st.dataframe(
                summary_aspect,
                use_container_width=True,
                hide_index=True,
            )

            if not summary_aspect.empty:
                chart_data = summary_aspect.set_index("Aspek")[[
                    "Positif", "Negatif", "Netral"
                ]]
                st.bar_chart(chart_data)

            st.write("**Distribusi konteks**")
            st.dataframe(
                context_summary,
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # 10. EVALUASI MODEL (CONFUSION MATRIX)
        # ----------------------------------------------------
        with tabs[9]:
            st.subheader("10. Evaluasi Model (Confusion Matrix)")

            labels_tuple = (
                tuple(reviews_df["sentimen_aktual"].tolist())
                if "sentimen_aktual" in reviews_df.columns
                else None
            )

            with st.spinner("Menghitung Confusion Matrix untuk dataset..."):
                eval_data = get_evaluation_metrics(
                    tuple(reviews_df["komentar"].tolist()),
                    labels_tuple=labels_tuple,
                )

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Total Data", f"{eval_data['total']:,}")
            e2.metric("Akurasi (Accuracy)", f"{eval_data['acc'] * 100:.1f}%")
            e3.metric("F1-Score (Macro Avg)", f"{eval_data['macro_f1'] * 100:.1f}%")
            e4.metric("Prediksi Koreksi / Berbeda", f"{len(eval_data['errors']):,}")

            c_cm1, c_cm2 = st.columns([1, 1])
            with c_cm1:
                st.write("**Tabel Confusion Matrix**")
                st.dataframe(eval_data["cm_df"], use_container_width=True)
                st.download_button(
                    "⬇️ Download Tabel CSV",
                    dataframe_to_csv_bytes(eval_data["cm_df"].reset_index()),
                    file_name="tabel_confusion_matrix.csv",
                    mime="text/csv",
                    key="dl_cm_csv_tab",
                )
            with c_cm2:
                st.write("**Grafik Confusion Matrix**")
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from sklearn.metrics import ConfusionMatrixDisplay

                labels = ["Positif", "Negatif", "Netral"]
                fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
                disp = ConfusionMatrixDisplay(confusion_matrix=eval_data["cm"], display_labels=labels)
                disp.plot(cmap="Blues", ax=ax, colorbar=True, values_format="d")
                ax.set_title("Confusion Matrix", fontsize=10, pad=10)
                plt.tight_layout()
                st.pyplot(fig)

                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300)
                plt.close(fig)
                st.download_button(
                    "⬇️ Download Grafik PNG",
                    buf.getvalue(),
                    file_name="confusion_matrix.png",
                    mime="image/png",
                    key="dl_cm_png_tab",
                )

            st.markdown("---")
            st.write("**Classification Report**")
            filtered_rep = eval_data["rep_df"].loc[
                ["Positif", "Negatif", "Netral", "accuracy", "macro avg", "weighted avg"],
                ["precision", "recall", "f1-score", "support"],
            ].round(4)
            st.dataframe(filtered_rep, use_container_width=True)
            st.download_button(
                "⬇️ Download Metrik CSV",
                dataframe_to_csv_bytes(filtered_rep.reset_index()),
                file_name="classification_report.csv",
                mime="text/csv",
                key="dl_rep_csv_tab",
            )

            st.markdown("---")
            st.write("**Detail Koreksi / Perbedaan Prediksi**")
            if eval_data["errors"]:
                err_df = pd.DataFrame(eval_data["errors"])
                st.dataframe(err_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download Detail CSV",
                    dataframe_to_csv_bytes(err_df),
                    file_name="detail_prediksi.csv",
                    mime="text/csv",
                    key="dl_err_csv_tab",
                )
            else:
                st.success("Semua prediksi sesuai.")


# ============================================================
# MODE 2 - SATU ULASAN
# ============================================================

else:
    st.header("📝 Analisis Satu Ulasan Wisata Bahari")

    review = st.text_area(
        "Masukkan ulasan wisata:",
        value="Pantainya indah tetapi banyak sampah saat weekend",
        height=130,
    )

    if st.button("🔍 Jalankan Pipeline Lengkap", type="primary"):
        if not review.strip():
            st.warning("Silakan masukkan ulasan.")
            st.stop()

        clean_text = preprocess(review)
        results = analyze_absa(clean_text)
        contexts = detect_context(review)
        rule_result = rule_based_reasoning(results, contexts)

        if use_llm and gemini_api_key:
            with st.spinner("Gemini sedang melakukan reasoning..."):
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

        st.success("Pipeline lengkap selesai dijalankan.")

        single_tabs = st.tabs([
            "1. Preprocessing",
            "2. ABSA",
            "3. Context",
            "4. Rule Engine",
            "5. LLM",
            "6. Hybrid",
            "7. XAI",
            "8. Decision Support",
        ])

        with single_tabs[0]:
            st.subheader("1. Hasil Preprocessing")
            st.code(clean_text)

        with single_tabs[1]:
            st.subheader("2. Aspect-Based Sentiment Analysis")
            if results:
                res_df = pd.DataFrame(results)
                pos_single = int((res_df["Sentimen"] == "Positif").sum()) if "Sentimen" in res_df.columns else 0
                neg_single = int((res_df["Sentimen"] == "Negatif").sum()) if "Sentimen" in res_df.columns else 0
                net_single = int((res_df["Sentimen"] == "Netral").sum()) if "Sentimen" in res_df.columns else 0

                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Positif", f"{pos_single}")
                sc2.metric("Negatif", f"{neg_single}")
                sc3.metric("Netral", f"{net_single}")

                st.dataframe(
                    res_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning("Tidak ditemukan aspek.")

        with single_tabs[2]:
            st.subheader("3. Context Extraction")
            if contexts:
                st.dataframe(
                    pd.DataFrame(contexts),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Tidak ditemukan konteks khusus.")

        with single_tabs[3]:
            st.subheader("4. Rule-Based Reasoning")
            rule_df = pd.DataFrame(rule_result.get("decisions", []))
            if rule_df.empty:
                st.info("Belum ada aspek untuk diproses rule engine.")
            else:
                st.dataframe(
                    rule_df,
                    use_container_width=True,
                    hide_index=True,
                )

        with single_tabs[4]:
            st.subheader("5. LLM Reasoning")
            render_llm_output(llm_result)

        with single_tabs[5]:
            st.subheader("6. Hybrid Reasoning")
            st.write(f"**Mode:** {hybrid_result.get('mode')}")
            st.write(f"**Confidence:** {hybrid_result.get('confidence')}")
            st.write(hybrid_result.get("explanation", ""))
            for rec in hybrid_result.get("recommendations", []):
                st.write(f"- {rec}")

        with single_tabs[6]:
            st.subheader("7. Explainable AI")
            if xai_rows:
                st.dataframe(
                    pd.DataFrame(xai_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Belum ada penjelasan XAI.")

        with single_tabs[7]:
            st.subheader("8. Decision Support")
            recs = hybrid_result.get("recommendations", [])
            if recs:
                for i, rec in enumerate(recs, 1):
                    st.write(f"{i}. {rec}")
            else:
                st.info("Belum ada rekomendasi decision support.")

            st.caption(
                "Decision Support adalah rekomendasi penelitian, bukan keputusan otomatis final."
            )


