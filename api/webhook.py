import os
import requests
from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

app = FastAPI()

# ============================
# 1. Configuración
# ============================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# Cargar dataset paralelo (sube a tu repo o colócalo en /api/data)
df = pd.read_csv("diccionario_Es_Nam_clean.csv", sep=";")

# Cargar embeddings desde Hugging Face
model = SentenceTransformer("Antrugos/namuywam-es-embeddings")

# Crear índices FAISS
gum_embeddings = model.encode(df["gum"].tolist(), convert_to_numpy=True, normalize_embeddings=True)
es_embeddings = model.encode(df["es"].tolist(), convert_to_numpy=True, normalize_embeddings=True)

dim = gum_embeddings.shape[1]
index_gum = faiss.IndexFlatIP(dim)  # gum → es
index_es = faiss.IndexFlatIP(dim)   # es → gum
index_gum.add(es_embeddings)
index_es.add(gum_embeddings)

# ============================
# 2. Funciones auxiliares
# ============================
def send_message(chat_id, text):
    url = f"{BASE_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error al enviar mensaje a Telegram: {e}")

def buscar_es_a_gum(query, top_k=1):
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index_es.search(query_emb, k=top_k)
    return df.iloc[indices[0][0]]["gum"]

def buscar_gum_a_es(query, top_k=1):
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index_gum.search(query_emb, k=top_k)
    return df.iloc[indices[0][0]]["es"]

# Detección básica: si el texto contiene caracteres propios de Namuy-Wam
def detect_direction(text: str) -> str:
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
            translated = buscar_es_a_gum(user_message)
            send_message(chat_id, f"Traducción ES→NMW: {translated}")
        else:
            translated = buscar_gum_a_es(user_message)
            send_message(chat_id, f"Traducción NMW→ES: {translated}")

    return {"status": "ok"}
