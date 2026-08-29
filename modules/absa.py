import re


# ============================================================
# KNOWLEDGE BASE ASPEK
# ============================================================

aspect_keywords = {
    "Kebersihan": [
        "sampah", "kotor", "bersih",
        "bau", "limbah"
    ],
    "Keindahan": [
        "indah", "cantik", "pemandangan",
        "bagus", "menarik"
    ],
    "Fasilitas": [
        "toilet", "parkir", "gazebo",
        "mushola", "fasilitas"
    ],
    "Aksesibilitas": [
        "jalan", "akses", "macet",
        "transportasi"
    ],
    "Keamanan": [
        "aman", "bahaya", "ombak",
        "penjaga", "pelampung"
    ],
    "Pelayanan": [
        "ramah", "pelayanan",
        "petugas", "pemandu"
    ],
    "Harga": [
        "mahal", "murah",
        "harga", "tiket", "biaya"
    ],
    "Keramaian": [
        "ramai", "sepi",
        "padat", "sesak"
    ]
}


# ============================================================
# KAMUS SENTIMEN
# ============================================================

positive_words = [
    "indah", "cantik", "bagus",
    "bersih", "nyaman", "aman",
    "ramah", "murah", "puas",
    "menarik", "mudah",
    "senyum", "senang", "suka", "cinta",
    "tertawa", "keren", "kagum", "terima kasih"
]

negative_words = [
    "sampah", "kotor", "bau",
    "mahal", "macet", "bahaya",
    "rusak", "sesak", "kecewa",
    "sedih", "menangis", "marah", "takut",
    "buruk", "jijik"
]


# ============================================================
# PEMISAH KLAUSA
# ============================================================

def split_clauses(text):
    clauses = re.split(
        r"\b(?:tetapi|tapi|namun|sedangkan)\b",
        text
    )

    return [
        clause.strip()
        for clause in clauses
        if clause.strip()
    ]


# ============================================================
# DETEKSI ASPEK
# ============================================================

def detect_aspects(clause):
    detected = []

    for aspect, keywords in aspect_keywords.items():
        if any(
            keyword in clause
            for keyword in keywords
        ):
            detected.append(aspect)

    return detected


# ============================================================
# SENTIMEN PER KLAUSA
# ============================================================

def detect_sentiment(clause):
    positive_score = sum(
        1
        for word in positive_words
        if word in clause
    )

    negative_score = sum(
        1
        for word in negative_words
        if word in clause
    )

    if positive_score > negative_score:
        return "Positif"
    elif negative_score > positive_score:
        return "Negatif"
    else:
        return "Netral"


# ============================================================
# ABSA ENGINE
# ============================================================

def analyze_absa(clean_text):
    """Analisis aspek + sentimen pada teks yang sudah dipreprocess."""
    clauses = split_clauses(clean_text)
    results = []

    for clause in clauses:
        aspects = detect_aspects(clause)
        sentiment = detect_sentiment(clause)

        for aspect in aspects:
            results.append({
                "Aspek": aspect,
                "Sentimen": sentiment,
                "Evidence": clause
            })

    return results
