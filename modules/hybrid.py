"""Hybrid Reasoning: menggabungkan Rule Engine dan LLM."""

from __future__ import annotations

from typing import Any


def hybrid_reasoning(
    llm_result: dict[str, Any] | None,
    rule_result: dict[str, Any] | None,
) -> dict[str, Any]:
    llm_result = llm_result or {}
    rule_result = rule_result or {}

    rule_recs = list(dict.fromkeys(rule_result.get("recommendations", []) or []))
    llm_ok = llm_result.get("status") == "ok" and isinstance(
        llm_result.get("output"), dict
    )

    llm_recs = []
    llm_summary = ""
    if llm_ok:
        output = llm_result.get("output", {})
        llm_recs = list(dict.fromkeys(output.get("rekomendasi", []) or []))
        llm_summary = str(output.get("ringkasan", ""))

    combined = []
    for rec in rule_recs + llm_recs:
        rec = str(rec).strip()
        if rec and rec not in combined:
            combined.append(rec)

    if llm_ok and rule_result.get("status") == "ok":
        mode = "LLM + Rule-Based"
        confidence = "Tinggi"
        explanation = (
            "Keputusan menggabungkan aturan transparan CAHRA dengan reasoning Gemini."
        )
    elif rule_result.get("status") == "ok":
        mode = "Rule-Based fallback"
        confidence = "Sedang"
        explanation = (
            "LLM tidak aktif/gagal, sehingga keputusan menggunakan rule engine yang tetap dapat diaudit."
        )
    elif llm_ok:
        mode = "LLM only"
        confidence = "Sedang"
        explanation = "Rule engine tidak tersedia; keputusan berasal dari LLM."
    else:
        mode = "Tidak tersedia"
        confidence = "Rendah"
        explanation = "Belum ada hasil LLM maupun rule engine."

    return {
        "status": "ok" if combined or llm_summary else "empty",
        "mode": mode,
        "confidence": confidence,
        "summary": llm_summary,
        "recommendations": combined,
        "explanation": explanation,
        "llm_used": llm_ok,
        "rule_used": rule_result.get("status") == "ok",
    }
