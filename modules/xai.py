"""Explainable AI (XAI) untuk CAHRA."""

from __future__ import annotations

from typing import Any


def explain_results(results):
    """Kompatibel dengan tampilan prototype app-2.py."""
    explanations = []

    for item in results:
        sentiment = item["Sentimen"]
        if sentiment == "Positif":
            icon = "🟢"
        elif sentiment == "Negatif":
            icon = "🔴"
        else:
            icon = "⚪"

        explanations.append({
            "icon": icon,
            "Aspek": item["Aspek"],
            "Sentimen": sentiment,
            "Evidence": item.get("Evidence", ""),
        })

    return explanations


def explain_single_pipeline(
    text: str,
    rule_result: dict[str, Any],
    llm_result: dict[str, Any] | None,
    hybrid_result: dict[str, Any],
) -> list[dict[str, str]]:
    rows = []

    for decision in rule_result.get("decisions", []):
        rows.append({
            "Tahap": "Rule-Based",
            "Aspek": str(decision.get("Aspek", "")),
            "Keputusan": str(decision.get("Sentimen Rule", "")),
            "Bukti/Alasan": (
                f"Evidence: '{decision.get('Evidence', '')}'. "
                f"{decision.get('Alasan Rule', '')}"
            ).strip(),
        })

    if llm_result and llm_result.get("status") == "ok":
        out = llm_result.get("output", {})
        rows.append({
            "Tahap": "LLM",
            "Aspek": "Keseluruhan ulasan",
            "Keputusan": "Reasoning tersedia",
            "Bukti/Alasan": str(out.get("ringkasan", "")),
        })

    rows.append({
        "Tahap": "Hybrid",
        "Aspek": "Decision Support",
        "Keputusan": hybrid_result.get("mode", ""),
        "Bukti/Alasan": (
            f"Confidence: {hybrid_result.get('confidence', '')}. "
            f"{hybrid_result.get('explanation', '')}"
        ).strip(),
    })

    return rows


def explain_dataset_decisions(
    rule_result: dict[str, Any],
    hybrid_result: dict[str, Any],
) -> list[dict[str, str]]:
    rows = []

    for decision in rule_result.get("decisions", []):
        rows.append({
            "Aspek": str(decision.get("Aspek", "")),
            "Prioritas": str(decision.get("Prioritas", "")),
            "Penjelasan": (
                f"{decision.get('Negatif', 0)} dari {decision.get('Total', 0)} temuan "
                f"bernilai negatif (rasio {decision.get('Rasio Negatif', 0):.1%}). "
                f"Konteks penguat: {decision.get('Konteks Penguat', '-')}"
            ),
            "Rekomendasi": str(decision.get("Rekomendasi", "")),
        })

    if hybrid_result.get("llm_used"):
        rows.append({
            "Aspek": "Hybrid",
            "Prioritas": hybrid_result.get("confidence", ""),
            "Penjelasan": "Ringkasan rule engine digabung dengan reasoning Gemini.",
            "Rekomendasi": "; ".join(hybrid_result.get("recommendations", [])),
        })

    return rows
