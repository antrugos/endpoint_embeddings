import os
import requests
from fastapi import FastAPI, Request

app = FastAPI()

# ============================
# 1. Configuración
# ============================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face API Token
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

HF_API_URL = "https://api-inference.huggingface.co/models/Antrugos/namuywam-es-embeddings"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

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

def query_hf(texts):
    """
    Envía texto a Hugging Face Inference API.
    `texts` debe ser una lista [texto1, texto2].
    """
    payload = {"inputs": texts}
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()
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
            # Comparar entrada con lista Namuy-Wam
            # En este caso mandamos pares [es, gum] a HF para similitud
            result = query_hf([user_message, "Traducción al Namuy-Wam"])
            if result:
                send_message(chat_id, f"Traducción ES→NMW: {result}")
            else:
                send_message(chat_id, "Error al traducir con Hugging Face.")
        else:
            result = query_hf([user_message, "Traducción al Español"])
            if result:
                send_message(chat_id, f"Traducción NMW→ES: {result}")
            else:
                send_message(chat_id, "Error al traducir con Hugging Face.")

    return {"status": "ok"}
