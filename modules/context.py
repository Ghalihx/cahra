def detect_context(original_text):
    contexts = []
    text = str(original_text).lower()

    # Linguistic Context
    if any(
        word in text
        for word in [
            "tetapi", "tapi",
            "namun", "sedangkan"
        ]
    ):
        contexts.append({
            "Jenis": "Linguistic",
            "Konteks": "Contrast"
        })

    if any(
        word in text
        for word in [
            "tidak", "bukan", "kurang"
        ]
    ):
        contexts.append({
            "Jenis": "Linguistic",
            "Konteks": "Negation"
        })

    # Situational Context
    if any(
        word in text
        for word in [
            "weekend", "sabtu",
            "minggu", "libur"
        ]
    ):
        contexts.append({
            "Jenis": "Situational",
            "Konteks": "Weekend/Holiday"
        })

    if any(
        word in text
        for word in [
            "hujan", "cerah",
            "mendung"
        ]
    ):
        contexts.append({
            "Jenis": "Situational",
            "Konteks": "Weather"
        })

    return contexts
