import html
import re
import unicodedata

import emoji


# ============================================================
# NORMALISASI SLANG
# ============================================================

SLANG = {
    # negasi — maknanya dipertahankan
    "gk": "tidak", "gak": "tidak", "ga": "tidak",
    "nggak": "tidak", "ngga": "tidak", "enggak": "tidak",
    "engga": "tidak", "tdk": "tidak", "bkn": "bukan",
    "blm": "belum", "belom": "belum",

    # konjungsi penting — tidak dibuang
    "tp": "tapi", "tpi": "tapi",

    # slang umum yang jelas
    "bgt": "banget", "bgtt": "banget",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah",
    "yg": "yang", "dgn": "dengan", "dg": "dengan",
    "dr": "dari", "krn": "karena", "karna": "karena",
    "klo": "kalau", "kalo": "kalau", "org": "orang",
    "sm": "sama", "sma": "sama", "smua": "semua",
    "jd": "jadi", "jdi": "jadi", "brp": "berapa",
    "bbrp": "beberapa", "kmrn": "kemarin", "kmren": "kemarin",
    "kemaren": "kemarin", "skrg": "sekarang", "skrng": "sekarang",
    "bgs": "bagus", "bgus": "bagus", "jln": "jalan",
    "pgn": "ingin", "pngn": "ingin", "pengen": "ingin",
    "msh": "masih", "hrs": "harus", "bs": "bisa", "bsa": "bisa",
    "dlu": "dulu", "dl": "dulu", "trs": "terus", "trus": "terus",
    "kyk": "seperti", "gmn": "bagaimana", "gimana": "bagaimana",
    "gimna": "bagaimana", "knp": "kenapa", "dmn": "di mana",
    "dimana": "di mana", "disana": "di sana", "disitu": "di situ",
    "kesana": "ke sana", "kesitu": "ke situ", "nyampe": "sampai",
    "sampe": "sampai", "ampe": "sampai", "mantul": "mantap",
    "rekomen": "rekomendasi", "recommend": "rekomendasi",
    "recommended": "rekomendasi",
}


# ============================================================
# EMOJI / EMOTICON -> KATA BAHASA INDONESIA
# ============================================================

EMOJI_MAP = {
    "🙂": "senyum",
    "😊": "senang", "😀": "senang", "😃": "senang",
    "😄": "senang", "😁": "senang",

    "😍": "suka", "🥰": "suka", "😘": "suka",

    "❤️": "cinta", "❤": "cinta", "💕": "cinta", "💖": "cinta",

    "😂": "tertawa", "🤣": "tertawa",

    "😭": "menangis",
    "😢": "sedih", "😞": "sedih", "😔": "sedih",
    "☹️": "sedih", "🙁": "sedih",

    "😡": "marah", "😠": "marah", "🤬": "marah",

    "😱": "kaget",
    "😨": "takut", "😰": "takut",

    "👍": "bagus",
    "👍🏻": "bagus", "👍🏼": "bagus", "👍🏽": "bagus",
    "👍🏾": "bagus", "👍🏿": "bagus",
    "👎": "buruk",

    "🔥": "keren",
    "✨": "bagus", "⭐": "bagus", "🌟": "bagus",

    "🙏": "terima kasih",

    "🤩": "kagum", "😎": "keren", "👌": "bagus",

    "🤮": "jijik", "🤢": "jijik",
    "💔": "kecewa",
}


EMOTICON_MAP = {
    ":-)": "senyum", ":)": "senyum",
    ";-)": "senyum", ";)": "senyum",
    ":-(": "sedih", ":(": "sedih",
    ":'(": "menangis",
    ":-D": "tertawa", ":D": "tertawa",
    ":-P": "bercanda", ":P": "bercanda",
}


# ============================================================
# REGEX DASAR
# ============================================================

URL_PATTERN = re.compile(
    r"(?:https?://\S+|www\.\S+|\b(?:t\.co|bit\.ly|tinyurl\.com)/\S+)",
    flags=re.IGNORECASE,
)

HTML_PATTERN = re.compile(r"<[^>]+>")


# ============================================================
# EMOJI -> KATA, TANPA PENGULANGAN DARI EMOJI BERULANG
# ============================================================

def convert_emoji(text: str) -> str:
    """Konversi emoji/emoticon menjadi kata bermakna.

    Pengulangan yang berasal dari emoji/emoticon yang sama akan diringkas:
        😍😍😍 -> suka
        😭😭   -> menangis
        👍👍🔥🔥 -> bagus keren
        :) :) -> senyum

    Kata asli pengguna tidak dideduplikasi oleh fungsi ini.
    Contoh "bagus bagus 👍" tetap dapat menjadi "bagus bagus bagus",
    karena dua "bagus" pertama memang berasal dari teks asli.
    """

    text = "" if text is None else str(text)

    # Gunakan token sementara supaya yang dideduplikasi hanya hasil emoji,
    # bukan kata asli yang kebetulan sama (mis. "bagus").
    def token_for(meaning: str) -> str:
        return f"__cahraemoji_{meaning.replace(' ', '_')}__"

    # Emoticon teks diproses lebih dulu agar :) tidak terganggu proses lain.
    for emoticon, meaning in sorted(
        EMOTICON_MAP.items(), key=lambda item: -len(item[0])
    ):
        text = text.replace(emoticon, f" {token_for(meaning)} ")

    # Emoji Unicode. Urutkan yang terpanjang dulu agar sequence seperti ❤️
    # dan varian skin-tone diproses sebagai satu unit.
    for char, meaning in sorted(
        EMOJI_MAP.items(), key=lambda item: -len(item[0])
    ):
        text = text.replace(char, f" {token_for(meaning)} ")

    # Rapikan spasi sebelum deduplikasi token.
    text = re.sub(r"\s+", " ", text)

    # Deduplikasi HANYA token emoji yang berurutan dan memiliki makna sama.
    # 😊😀😃 -> ketiganya token "senang" -> satu token "senang".
    text = re.sub(
        r"(__cahraemoji_[a-zA-Z0-9_]+__)(?:\s+\1)+",
        r"\1",
        text,
    )

    # Emoji yang belum masuk kamus tidak langsung dibuang.
    # Contoh :beaming_face_with_smiling_eyes:
    # menjadi "beaming face with smiling eyes".
    text = emoji.demojize(text, language="en")
    text = re.sub(
        r":([a-zA-Z0-9_+\-&]+):",
        lambda m: " " + m.group(1).replace("_", " ") + " ",
        text,
    )

    # Token sementara -> kata Indonesia.
    text = re.sub(
        r"__cahraemoji_([a-zA-Z0-9_]+)__",
        lambda m: m.group(1).replace("_", " "),
        text,
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text


# Alias agar tetap kompatibel dengan CAHRA Complete v2.
def _emoji_to_words(text: str) -> str:
    return convert_emoji(text)


# ============================================================
# NORMALISASI KARAKTER BERULANG
# ============================================================

def normalize_repeated_chars(text: str) -> str:
    # Tawa yang sangat panjang
    text = re.sub(r"\b(?:wk){3,}\b", "wkwk", text)
    text = re.sub(r"\b(?:ha){3,}\b", "haha", text)
    text = re.sub(r"\b(?:he){3,}\b", "hehe", text)

    # Karakter berulang >= 3: baguuuus -> bagus
    text = re.sub(r"([a-z])\1{2,}", r"\1", text)

    # Tanda baca berulang
    text = re.sub(r"!{3,}", "!!", text)
    text = re.sub(r"\?{3,}", "??", text)
    text = re.sub(r"\.{4,}", "...", text)
    return text


# Alias kompatibilitas.
def _normalize_repeated_chars(text: str) -> str:
    return normalize_repeated_chars(text)


# ============================================================
# NORMALISASI SLANG
# ============================================================

def normalize_slang(text: str) -> str:
    pattern = re.compile(r"\b[\w]+\b", flags=re.UNICODE)
    return pattern.sub(lambda m: SLANG.get(m.group(0), m.group(0)), text)


# Alias kompatibilitas.
def _normalize_slang(text: str) -> str:
    return normalize_slang(text)


# ============================================================
# PREPROCESSING UTAMA CAHRA
# ============================================================

def preprocess(text):
    """Preprocessing CAHRA yang konsisten dengan pipeline penelitian.

    Tahapan:
    - Unicode normalization
    - decode HTML entity
    - hapus URL
    - hapus HTML/tag
    - hapus mention
    - hashtag dipertahankan sebagai kata
    - lowercase
    - emoji/emoticon -> kata/informasi
    - emoji berulang dengan arti sama -> satu kata
    - normalisasi karakter berulang
    - normalisasi slang yang jelas
    - normalisasi zero-width character dan spasi

    TIDAK melakukan stopword removal.
    Negasi/konjungsi seperti "tidak", "bukan", "kurang", "tapi",
    "tetapi", "namun", dan "meskipun" tetap dipertahankan.
    """

    text = "" if text is None else str(text)

    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)

    # URL dan HTML
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)

    # Mention dibuang, hashtag dipertahankan sebagai kata
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(?=\w)", "", text)

    # Lowercase sebelum slang normalization
    text = text.lower()

    # Emoji -> kata Indonesia; pengulangan emoji diringkas
    text = convert_emoji(text)

    # Normalisasi karakter dan slang
    text = normalize_repeated_chars(text)
    text = normalize_slang(text)

    # Zero-width characters + spasi
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Alias nama dari pipeline Colab sebelumnya.
def clean_comment(text):
    return preprocess(text)
