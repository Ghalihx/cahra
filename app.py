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

st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "Mode Analisis",
    [
        "📊 Analisis Dataset",
        "📝 Analisis Satu Ulasan",
        "🎯 Evaluasi Model (Confusion Matrix)",
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


# ============================================================
# MODE 2 - SATU ULASAN
# ============================================================

elif mode == "📝 Analisis Satu Ulasan":
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
                st.dataframe(
                    pd.DataFrame(results),
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


# ============================================================
# MODE 3 - EVALUASI MODEL (CONFUSION MATRIX)
# ============================================================

else:
    st.header("🎯 Evaluasi Model & Confusion Matrix")
    st.write(
        "Modul ini digunakan untuk menghitung evaluasi empiris performa model "
        "menggunakan **Confusion Matrix**, **Accuracy**, **Precision**, **Recall**, dan **F1-Score** "
        "untuk kebutuhan publikasi artikel ilmiah atau skripsi."
    )

    st.markdown("---")

    col_opt1, col_opt2 = st.columns([1, 1])
    with col_opt1:
        use_sample = st.checkbox(
            "Gunakan data uji sampel bawaan (20 ulasan berlabel)",
            value=False,
            help="Centang untuk melihat contoh hasil evaluasi tanpa perlu upload file sendiri.",
        )

    eval_file = None
    if not use_sample:
        eval_file = st.file_uploader(
            "📤 Upload Dataset Uji Berlabel (CSV)",
            type=["csv"],
            help="CSV wajib memiliki kolom 'komentar' dan kolom 'sentimen_aktual' (Positif, Negatif, Netral).",
            key="eval_file_uploader",
        )

    test_df = None
    if use_sample:
        sample_path = "data/data_uji_contoh.csv"
        if os.path.exists(sample_path):
            test_df = pd.read_csv(sample_path)
            st.success("Menggunakan data uji contoh bawaan (20 ulasan).")
        else:
            st.error("File sampel data/data_uji_contoh.csv tidak ditemukan.")
    elif eval_file is not None:
        try:
            test_df = pd.read_csv(eval_file)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

    if test_df is not None:
        col_map = {str(c).strip().lower(): c for c in test_df.columns}
        text_col = col_map.get("komentar") or col_map.get("review") or col_map.get("ulasan")
        label_col = (
            col_map.get("sentimen_aktual")
            or col_map.get("label")
            or col_map.get("sentimen")
            or col_map.get("actual")
        )

        if not text_col or not label_col:
            st.error(
                "Dataset wajib memiliki kolom teks ulasan ('komentar') dan kolom label aktual ('sentimen_aktual'). "
                f"Kolom yang ditemukan: {list(test_df.columns)}"
            )
            st.stop()

        with st.expander("👀 Pratinjau Dataset Uji", expanded=False):
            st.dataframe(test_df[[text_col, label_col]], use_container_width=True, hide_index=True)

        if st.button("🚀 Jalankan Evaluasi Model", type="primary", use_container_width=True):
            with st.spinner("Menjalankan pipeline prediksi pada data uji..."):
                from evaluate import predict_sentiment
                from sklearn.metrics import (
                    confusion_matrix,
                    classification_report,
                    ConfusionMatrixDisplay,
                )
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                y_true = []
                y_pred = []
                errors = []

                for idx, row in test_df.iterrows():
                    actual = str(row[label_col]).strip().capitalize()
                    pred = predict_sentiment(str(row[text_col]))
                    y_true.append(actual)
                    y_pred.append(pred)

                    if actual != pred:
                        errors.append({
                            "No": idx + 1,
                            "Komentar": row[text_col],
                            "Aktual": actual,
                            "Prediksi": pred,
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

                st.session_state["eval_results"] = {
                    "cm_df": cm_df,
                    "cm": cm,
                    "rep_df": rep_df,
                    "errors": errors,
                    "total": len(test_df),
                    "acc": rep_dict.get("accuracy", 0.0),
                    "macro_f1": rep_dict.get("macro avg", {}).get("f1-score", 0.0),
                }

        if "eval_results" in st.session_state:
            res = st.session_state["eval_results"]
            cm_df = res["cm_df"]
            cm = res["cm"]
            rep_df = res["rep_df"]
            errors = res["errors"]

            st.success("Evaluasi model selesai dihitung.")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Data Uji", f"{res['total']:,}")
            k2.metric("Akurasi (Accuracy)", f"{res['acc'] * 100:.2f}%")
            k3.metric("F1-Score (Macro Avg)", f"{res['macro_f1'] * 100:.2f}%")
            k4.metric("Jumlah Prediksi Salah", f"{len(errors):,}")

            eval_tabs = st.tabs([
                "📊 Confusion Matrix",
                "📈 Classification Report",
                "🔍 Analisis Error (Pembahasan)",
            ])

            with eval_tabs[0]:
                st.subheader("1. Confusion Matrix")
                c_cm1, c_cm2 = st.columns([1, 1])
                with c_cm1:
                    st.write("**Tabel Nilai Confusion Matrix**")
                    st.dataframe(cm_df, use_container_width=True)
                    st.download_button(
                        "⬇️ Download Tabel CSV",
                        dataframe_to_csv_bytes(cm_df.reset_index()),
                        file_name="tabel_confusion_matrix.csv",
                        mime="text/csv",
                    )
                with c_cm2:
                    st.write("**Grafik Heatmap (Siap Pakai untuk Artikel)**")
                    import matplotlib.pyplot as plt
                    from sklearn.metrics import ConfusionMatrixDisplay
                    labels = ["Positif", "Negatif", "Netral"]
                    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
                    disp.plot(cmap="Blues", ax=ax, colorbar=True, values_format="d")
                    ax.set_title("Confusion Matrix CAHRA", fontsize=11, pad=10)
                    plt.tight_layout()
                    st.pyplot(fig)

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=300)
                    plt.close(fig)
                    st.download_button(
                        "⬇️ Download Gambar Heatmap (300 DPI)",
                        buf.getvalue(),
                        file_name="confusion_matrix_300dpi.png",
                        mime="image/png",
                    )

            with eval_tabs[1]:
                st.subheader("2. Classification Report (Metrik Kinerja)")
                st.write(
                    "Tabel di bawah ini dapat langsung disalin ke bab **Hasil dan Pembahasan** di artikel Anda."
                )
                filtered_rep = rep_df.loc[
                    ["Positif", "Negatif", "Netral", "accuracy", "macro avg", "weighted avg"],
                    ["precision", "recall", "f1-score", "support"],
                ].round(4)
                st.dataframe(filtered_rep, use_container_width=True)
                st.download_button(
                    "⬇️ Download Metrik CSV",
                    dataframe_to_csv_bytes(filtered_rep.reset_index()),
                    file_name="metrik_evaluasi.csv",
                    mime="text/csv",
                )

            with eval_tabs[2]:
                st.subheader("3. Analisis Kesalahan (Error Analysis)")
                st.write(
                    "Ulasan berikut adalah kasus di mana prediksi model berbeda dari label aktual manusia. "
                    "Gunakan daftar ini sebagai bahan diskusi di bab **Pembahasan**."
                )
                if errors:
                    err_df = pd.DataFrame(errors)
                    st.dataframe(err_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇️ Download Data Error CSV",
                        dataframe_to_csv_bytes(err_df),
                        file_name="analisis_error.csv",
                        mime="text/csv",
                    )
                else:
                    st.success("Luar biasa! Tidak ada kesalahan klasifikasi pada data uji ini.")
    else:
        st.info("Silakan pilih 'Gunakan data uji sampel' di atas atau upload file CSV pengujian Anda.")

