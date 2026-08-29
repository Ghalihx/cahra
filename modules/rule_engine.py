"""Rule-Based Reasoning untuk CAHRA."""

from __future__ import annotations

import re
from typing import Any


RECOMMENDATIONS = {
    "Kebersihan": "Perkuat pengelolaan sampah, jadwal pembersihan, dan fasilitas tempat sampah.",
    "Keindahan": "Pertahankan kualitas lanskap dan cegah aktivitas yang menurunkan daya tarik visual.",
    "Fasilitas": "Prioritaskan pemeliharaan dan penambahan fasilitas dasar wisata sesuai kebutuhan pengunjung.",
    "Aksesibilitas": "Evaluasi kondisi jalan, akses transportasi, petunjuk arah, dan titik kemacetan.",
    "Keamanan": "Perkuat informasi keselamatan, pengawasan, rambu bahaya, dan mitigasi risiko pantai.",
    "Pelayanan": "Tingkatkan standar pelayanan, keramahan petugas, serta informasi bagi wisatawan.",
    "Harga": "Evaluasi transparansi dan kewajaran harga tiket, parkir, serta biaya layanan.",
    "Keramaian": "Terapkan pengelolaan kapasitas pengunjung dan pengaturan arus saat periode ramai.",
}

NEGATED_POSITIVE = {
    "tidak bersih": "Negatif",
    "tidak bagus": "Negatif",
    "tidak nyaman": "Negatif",
    "tidak aman": "Negatif",
    "tidak ramah": "Negatif",
    "tidak menarik": "Negatif",
    "tidak mudah": "Negatif",
    "kurang bersih": "Negatif",
    "kurang bagus": "Negatif",
    "kurang nyaman": "Negatif",
    "kurang aman": "Negatif",
    "kurang ramah": "Negatif",
}

NEGATED_NEGATIVE = {
    "tidak kotor": "Positif",
    "tidak bau": "Positif",
    "tidak mahal": "Positif",
    "tidak macet": "Positif",
    "tidak bahaya": "Positif",
    "tidak rusak": "Positif",
    "tidak sesak": "Positif",
}


def _context_names(contexts: list[dict] | None) -> set[str]:
    return {
        str(item.get("Konteks", ""))
        for item in (contexts or [])
        if item.get("Konteks")
    }


def _correct_sentiment(evidence: str, initial: str) -> tuple[str, str | None]:
    ev = str(evidence).lower()

    for phrase, sentiment in NEGATED_POSITIVE.items():
        if phrase in ev:
            return sentiment, f"Negasi terdeteksi pada frasa '{phrase}'."

    for phrase, sentiment in NEGATED_NEGATIVE.items():
        if phrase in ev:
            return sentiment, f"Negasi membalik makna pada frasa '{phrase}'."

    return initial, None


def rule_based_reasoning(
    absa_results: list[dict],
    contexts: list[dict] | None = None,
) -> dict[str, Any]:
    """Reasoning untuk satu ulasan."""

    contexts = contexts or []
    context_names = _context_names(contexts)
    decisions = []

    for item in absa_results:
        aspect = item.get("Aspek", "Tidak diketahui")
        evidence = str(item.get("Evidence", ""))
        initial = item.get("Sentimen", "Netral")
        final_sentiment, correction_reason = _correct_sentiment(evidence, initial)

        priority = "Rendah"
        reasons = []

        if correction_reason:
            reasons.append(correction_reason)

        if final_sentiment == "Negatif":
            priority = "Sedang"
            reasons.append(f"Aspek {aspect} memiliki sentimen negatif.")

            if "Weekend/Holiday" in context_names and aspect in {
                "Kebersihan", "Keramaian", "Fasilitas", "Pelayanan"
            }:
                priority = "Tinggi"
                reasons.append("Konteks akhir pekan/libur dapat meningkatkan tekanan pengunjung.")

            if "Weather" in context_names and aspect in {
                "Keamanan", "Aksesibilitas"
            }:
                priority = "Tinggi"
                reasons.append("Konteks cuaca meningkatkan relevansi risiko keselamatan/akses.")

        elif final_sentiment == "Positif":
            reasons.append(f"Aspek {aspect} merupakan kekuatan yang perlu dipertahankan.")
        else:
            reasons.append(f"Aspek {aspect} belum menunjukkan arah sentimen yang kuat.")

        decisions.append({
            "Aspek": aspect,
            "Sentimen Awal": initial,
            "Sentimen Rule": final_sentiment,
            "Prioritas": priority,
            "Evidence": evidence,
            "Rekomendasi": RECOMMENDATIONS.get(
                aspect,
                "Lakukan evaluasi lebih lanjut pada aspek ini."
            ),
            "Alasan Rule": " ".join(reasons),
        })

    negatives = [d for d in decisions if d["Sentimen Rule"] == "Negatif"]
    positives = [d for d in decisions if d["Sentimen Rule"] == "Positif"]

    return {
        "status": "ok",
        "decisions": decisions,
        "negative_aspects": [d["Aspek"] for d in negatives],
        "positive_aspects": [d["Aspek"] for d in positives],
        "contexts": sorted(context_names),
        "recommendations": [d["Rekomendasi"] for d in negatives],
    }


def rule_based_dataset_reasoning(
    summary_rows: list[dict],
    context_rows: list[dict] | None = None,
) -> dict[str, Any]:
    """Reasoning rule-based pada ringkasan agregat dataset."""

    context_rows = context_rows or []
    context_counts = {
        str(row.get("Konteks")): int(row.get("Jumlah", 0))
        for row in context_rows
    }

    decisions = []

    for row in summary_rows:
        aspect = str(row.get("Aspek", "Tidak diketahui"))
        positive = int(row.get("Positif", 0))
        negative = int(row.get("Negatif", 0))
        neutral = int(row.get("Netral", 0))
        total = max(int(row.get("Total", positive + negative + neutral)), 0)
        neg_ratio = (negative / total) if total else 0.0

        if neg_ratio >= 0.40:
            priority = "Tinggi"
        elif neg_ratio >= 0.20:
            priority = "Sedang"
        else:
            priority = "Rendah"

        boosts = []
        if context_counts.get("Weekend/Holiday", 0) > 0 and aspect in {
            "Kebersihan", "Keramaian", "Fasilitas", "Pelayanan"
        }:
            boosts.append("Weekend/Holiday")
        if context_counts.get("Weather", 0) > 0 and aspect in {
            "Keamanan", "Aksesibilitas"
        }:
            boosts.append("Weather")

        if boosts and priority == "Sedang":
            priority = "Tinggi"

        decisions.append({
            "Aspek": aspect,
            "Positif": positive,
            "Negatif": negative,
            "Netral": neutral,
            "Total": total,
            "Rasio Negatif": round(neg_ratio, 4),
            "Prioritas": priority,
            "Konteks Penguat": ", ".join(boosts) if boosts else "-",
            "Rekomendasi": RECOMMENDATIONS.get(
                aspect,
                "Lakukan evaluasi lebih lanjut pada aspek ini."
            ),
        })

    order = {"Tinggi": 0, "Sedang": 1, "Rendah": 2}
    decisions.sort(
        key=lambda x: (order.get(x["Prioritas"], 9), -x["Rasio Negatif"], -x["Total"])
    )

    return {
        "status": "ok",
        "decisions": decisions,
        "context_counts": context_counts,
        "recommendations": [
            d["Rekomendasi"]
            for d in decisions
            if d["Prioritas"] in {"Tinggi", "Sedang"}
        ],
    }
