import os
import json

def save_document(text, translation, filename):
    os.makedirs("data", exist_ok=True)
    data = {
        "original": text,
        "translation": translation
    }
    with open(f"data/{filename}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
