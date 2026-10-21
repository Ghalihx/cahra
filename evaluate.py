"""Script Evaluasi Model CAHRA untuk Kebutuhan Artikel Ilmiah / Skripsi.

Menghasilkan:
1. Tabel Confusion Matrix (Positif, Negatif, Netral)
2. Metrik Akurasi, Presisi, Recall, F1-Score
3. Grafik Confusion Matrix (evaluasi_confusion_matrix.png) resolusi tinggi (300 DPI)
4. Rekap Analisis Error (analisis_error.csv) untuk bab Pembahasan
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from modules.preprocessing import preprocess
from modules.absa import analyze_absa, detect_sentiment
from modules.rule_engine import rule_based_reasoning


def predict_sentiment(text: str) -> str:
    """Memprediksi sentimen keseluruhan ulasan dengan pipeline CAHRA."""
    clean_text = preprocess(str(text))
    absa_results = analyze_absa(clean_text)

    if absa_results:
        rule_res = rule_based_reasoning(absa_results)
        decisions = rule_res.get("decisions", [])
        sentiments = [d.get("Sentimen Rule", "Netral") for d in decisions]
        
        pos_count = sentiments.count("Positif")
        neg_count = sentiments.count("Negatif")
        
        if neg_count > pos_count:
            return "Negatif"
        elif pos_count > neg_count:
            return "Positif"
        else:
            return "Netral"
    
    # Fallback jika tidak ada aspek khusus yang terdeteksi
    return detect_sentiment(clean_text)


def evaluate(file_path: str, output_dir: str = "."):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' tidak ditemukan.")

    df = pd.read_csv(file_path)
    
    # Normalisasi nama kolom
    col_map = {str(c).strip().lower(): c for c in df.columns}
    text_col = col_map.get("komentar") or col_map.get("review") or col_map.get("ulasan")
    label_col = (
        col_map.get("sentimen_aktual")
        or col_map.get("label")
        or col_map.get("sentimen")
        or col_map.get("actual")
    )

    if not text_col or not label_col:
        raise ValueError(
            f"File CSV wajib memiliki kolom teks ('komentar') dan label aktual ('sentimen_aktual'). "
            f"Kolom yang ada: {list(df.columns)}"
        )

    print(f"Membaca dataset uji: {file_path} ({len(df)} baris)")
    
    y_true = []
    y_pred = []
    errors = []

    for idx, row in df.iterrows():
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
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Aktual {l}" for l in labels],
        columns=[f"Prediksi {l}" for l in labels],
    )

    print("\n" + "=" * 55)
    print("1. TABEL CONFUSION MATRIX")
    print("=" * 55)
    print(cm_df.to_string())

    # 2. Classification Report
    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()

    print("\n" + "=" * 55)
    print("2. METRIK EVALUASI MODEL (CLASSIFICATION REPORT)")
    print("=" * 55)
    print(
        report_df.loc[
            labels + ["accuracy", "macro avg", "weighted avg"],
            ["precision", "recall", "f1-score", "support"],
        ].round(4).to_string()
    )

    # 3. Simpan Gambar Confusion Matrix (Resolusi tinggi 300 DPI untuk artikel)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, colorbar=True, values_format="d")
    ax.set_title("Confusion Matrix Evaluasi Model CAHRA", fontsize=12, pad=12)
    ax.set_xlabel("Prediksi Model", fontsize=10)
    ax.set_ylabel("Kelas Aktual (Ground Truth)", fontsize=10)
    plt.tight_layout()

    img_path = os.path.join(output_dir, "evaluasi_confusion_matrix.png")
    fig.savefig(img_path, dpi=300)
    plt.close(fig)
    print(f"\n[OK] Grafik Confusion Matrix disimpan di: {img_path}")

    # 4. Simpan Analisis Error untuk Pembahasan Artikel
    error_path = os.path.join(output_dir, "analisis_error.csv")
    error_df = pd.DataFrame(errors)
    error_df.to_csv(error_path, index=False)
    print(f"[OK] Rekap kesalahan ({len(errors)} ulasan) disimpan di: {error_path}")

    return cm_df, report_df, error_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi Model CAHRA")
    parser.add_argument(
        "--file",
        type=str,
        default="data/data_uji_contoh.csv",
        help="Path ke file dataset CSV berlabel",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Folder tempat menyimpan grafik dan laporan",
    )
    args = parser.parse_args()

    evaluate(args.file, args.output_dir)
