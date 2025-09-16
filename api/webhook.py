import os
import requests
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

gum_texts = [...]  # lista de gum de tu dataset
es_texts = [...]   # lista de español
gum_embeddings = np.load("gum_embeddings.npy")
es_embeddings = np.load("es_embeddings.npy")

# ============================
# 2. Funciones auxiliares
# ============================
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
        if isinstance(data, list) and "embedding" in data[0]:
            return data[0]["embedding"]
        return data  # fallback
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
# 3. Endpoint webhook
# ============================
@app.post("/api/webhook")
async def webhook(request: Request):
    data = await request.json()
    print("Datos recibidos:", data)

    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        user_message = data["message"]["text"]

        direction = detect_direction(user_message)

        if direction == "es-nmw":
            emb = query_hf(user_message)
            if emb:
                sims = cosine_similarity([emb], gum_embeddings)[0]
                idx = np.argmax(sims)
                send_message(chat_id, f"Traducción ES→NMW: {gum_texts[idx]}")
            else:
                send_message(chat_id, "Error al obtener embedding.")
        else:
            emb = query_hf(user_message)
            if emb:
                sims = cosine_similarity([emb], es_embeddings)[0]
                idx = np.argmax(sims)
                send_message(chat_id, f"Traducción NMW→ES: {es_texts[idx]}")
            else:
                send_message(chat_id, "Error al obtener embedding.")

    return {"status": "ok"}
