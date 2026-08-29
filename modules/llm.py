"""LLM Reasoning untuk CAHRA.

Google Gemini bersifat opsional. Modul ini memiliki retry dengan exponential
backoff untuk error 503/429/500 dan fallback model jika model utama sedang
padat. Pipeline CAHRA tetap dapat berjalan dengan Rule Engine saat LLM gagal.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any


def _safe_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(x in msg for x in [
        "503", "unavailable", "high demand", "overloaded",
        "429", "resource_exhausted", "500", "internal",
    ])


def llm_reasoning(
    text: str,
    absa_results: list[dict] | None = None,
    contexts: list[dict] | None = None,
    *,
    api_key: str | None = None,
    model: str = "gemini-3.7-flash",
    dataset_summary: str | None = None,
) -> dict[str, Any]:
    if not api_key or not str(api_key).strip():
        return {
            "status": "disabled",
            "provider": "Google Gemini",
            "model": model,
            "message": "LLM belum dijalankan karena Gemini API Key belum diisi.",
            "output": None,
        }

    try:
        from google import genai
    except Exception as exc:
        return {
            "status": "error",
            "provider": "Google Gemini",
            "model": model,
            "message": f"Package google-genai tidak tersedia: {exc}",
            "output": None,
        }

    absa_results = absa_results or []
    contexts = contexts or []

    if dataset_summary:
        subject = (
            "Berikut ringkasan agregat dataset ulasan wisata bahari:\n\n"
            f"{dataset_summary}"
        )
    else:
        subject = (
            f"ULASAN:\n{text}\n\n"
            f"HASIL ABSA AWAL:\n{json.dumps(absa_results, ensure_ascii=False)}\n\n"
            f"KONTEKS:\n{json.dumps(contexts, ensure_ascii=False)}"
        )

    prompt = f"""
Anda adalah komponen LLM Reasoning pada CAHRA
(Context-Aware Hybrid Reasoning Framework) untuk wisata bahari berkelanjutan.

Tugas Anda:
1. Telaah hasil ABSA/context yang sudah diberikan; jangan mengarang bukti.
2. Identifikasi isu utama dan kekuatan utama.
3. Berikan reasoning singkat yang memperhatikan konteks.
4. Berikan rekomendasi operasional untuk pengelola destinasi.
5. Jika data tidak cukup, nyatakan keterbatasannya.

{subject}

Kembalikan JSON VALID saja dengan struktur berikut:
{{
  "ringkasan": "...",
  "isu_utama": ["..."],
  "kekuatan_utama": ["..."],
  "rekomendasi": ["..."],
  "alasan": ["..."],
  "keterbatasan": "..."
}}
""".strip()

    client = genai.Client(api_key=str(api_key).strip())

    # Model utama dulu; bila 503 terus, coba fallback yang lebih ringan.
    candidate_models = []
    for m in [model, "gemini-3.5-flash-lite", "gemini-3.5-flash"]:
        if m and m not in candidate_models:
            candidate_models.append(m)

    last_error = None

    for candidate in candidate_models:
        delays = [0, 2, 5, 10]
        for attempt, delay in enumerate(delays, start=1):
            if delay:
                time.sleep(delay)
            try:
                response = client.models.generate_content(
                    model=candidate,
                    contents=prompt,
                )
                raw_text = (response.text or "").strip()
                parsed = _safe_json(raw_text)

                if parsed is None:
                    parsed = {
                        "ringkasan": raw_text,
                        "isu_utama": [],
                        "kekuatan_utama": [],
                        "rekomendasi": [],
                        "alasan": [],
                        "keterbatasan": (
                            "Respons model tidak mengikuti format JSON; teks asli dipertahankan."
                        ),
                    }

                note = "LLM Reasoning berhasil."
                if candidate != model:
                    note += f" Model utama sedang tidak tersedia; otomatis memakai fallback {candidate}."

                return {
                    "status": "ok",
                    "provider": "Google Gemini",
                    "model": candidate,
                    "requested_model": model,
                    "message": note,
                    "output": parsed,
                }

            except Exception as exc:
                last_error = exc
                if not _is_retryable(exc):
                    break

    return {
        "status": "error",
        "provider": "Google Gemini",
        "model": model,
        "message": (
            "LLM gagal setelah retry dan fallback. "
            f"Error terakhir: {last_error}. "
            "Rule Engine/Hybrid fallback tetap dapat digunakan."
        ),
        "output": None,
    }
