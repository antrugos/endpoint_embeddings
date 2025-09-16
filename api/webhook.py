import os
import requests
import json
import numpy as np
from fastapi import FastAPI, Request
from sklearn.metrics.pairwise import cosine_similarity
app = FastAPI()

# ============================
# 1. Configuración
# ============================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face API Token
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

HF_API_URL = "https://api-inference.huggingface.co/models/Antrugos/namuywam-es-embeddings"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# En Vercel, __file__ apunta a /var/task/api/webhook.py → subimos al root
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# ============================
# 2. Funciones auxiliares
# ============================
def normalize(vecs: np.ndarray) -> np.ndarray:
    """Normaliza embeddings fila por fila"""
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

def send_message(chat_id, text):
    """Envía mensaje a Telegram"""
    url = f"{BASE_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error al enviar mensaje a Telegram: {e}")

def query_hf(text: str):
    """Devuelve embedding de Hugging Face API para un texto."""
    payload = {"inputs": text}
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        data = response.json()
        try:
            # Caso 1: API devuelve [{"embedding": [...]}]
            if isinstance(data, list) and isinstance(data[0], dict) and "embedding" in data[0]:
                return np.array(data[0]["embedding"], dtype=np.float32)
            # Caso 2: API devuelve [[...]]
            elif isinstance(data, list) and isinstance(data[0], list):
                return np.array(data[0], dtype=np.float32)
        except Exception as e:
            print("Error parsing embedding:", e)
            return None
    else:
        print("Error HF:", response.text)
        return None

def detect_direction(text: str) -> str:
    """Heurística simple para decidir idioma origen"""
    if any(ch in text for ch in ["ѳ", "ѫ", "ѯ", "ts", "tik", "pak"]):
        return "nmw-es"
    else:
        return "es-nmw"

# ============================
# 3. load embeddings
# ============================
gum_embeddings = normalize(np.load(os.path.join(BASE_DIR, "data/gum_embeddings.npy")))
es_embeddings = normalize(np.load(os.path.join(BASE_DIR, "data/es_embeddings.npy")))

with open(os.path.join(BASE_DIR, "data/gum_texts.json"), encoding="utf-8") as f:
    gum_texts = json.load(f)

with open(os.path.join(BASE_DIR, "data/es_texts.json"), encoding="utf-8") as f:
    es_texts = json.load(f)

# ============================
# 4. Endpoint webhook
# ============================
@app.post("/api/webhook")
async def webhook(request: Request):
    data = await request.json()
    print("Datos recibidos:", data)

    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        user_message = data["message"]["text"]

        direction = detect_direction(user_message)

        emb = query_hf(user_message)
        if emb is None:
            send_message(chat_id, "⚠️ No se pudo obtener embedding desde Hugging Face.")
            return {"status": "error"}
        
        if direction == "es-nmw":
            sims = cosine_similarity([emb], gum_embeddings)[0]
            idx = int(np.argmax(sims))
            send_message(chat_id, f"Traducción ES→NMW: {gum_texts[idx]}")
        else:
            sims = cosine_similarity([emb], es_embeddings)[0]
            idx = int(np.argmax(sims))
            send_message(chat_id, f"Traducción NMW→ES: {es_texts[idx]}")

    return {"status": "ok"}
