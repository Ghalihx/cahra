CAHRA - Context-Aware Hybrid Reasoning Framework
================================================

VERSI FINAL STREAMLIT - DATASET DI-UPLOAD SAAT APLIKASI DIBUKA

STRUKTUR
CAHRA/
├── app.py
├── data/
│   └── README.txt
├── modules/
│   ├── preprocessing.py
│   ├── absa.py
│   ├── context.py
│   ├── llm.py
│   ├── rule_engine.py
│   ├── hybrid.py
│   └── xai.py
├── requirements.txt
└── .gitignore

DATASET
- Dataset TIDAK perlu disimpan di source code.
- Pada Streamlit pilih "Analisis Dataset" lalu upload CSV.
- Kolom wajib: komentar
- Nama file bebas.
- Dataset hanya dipakai selama sesi aplikasi aktif.
- Saat file diganti, hasil analisis dataset sebelumnya otomatis di-reset.

PIPELINE
1. Data Collection / Upload Dataset
2. Preprocessing
3. Aspect Detection / ABSA
4. Aspect Sentiment
5. Context Extraction
6. LLM Reasoning (Google Gemini, opsional)
7. Rule-Based Reasoning
8. Hybrid Reasoning
9. Explainable AI (XAI)
10. Decision Support

MENJALANKAN DI WINDOWS
1. py -3.11 -m venv .venv
2. .venv\Scripts\activate
3. python -m pip install --upgrade pip
4. pip install -r requirements.txt
5. python -m streamlit run app.py
6. Buka http://localhost:8501

DEPLOY STREAMLIT COMMUNITY CLOUD
1. Upload folder proyek ini ke GitHub (tanpa .venv dan __pycache__).
2. Deploy app.py pada Streamlit Community Cloud.
3. Buka aplikasi hasil deploy.
4. Pilih "Analisis Dataset" dan upload CSV dari browser.
5. Tidak perlu mengubah reviews.csv lewat GitHub/source code.
