import os
import requests
from dotenv import load_dotenv

# Cargar variables del .env
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
url = "https://api-inference.huggingface.co/models/antrugos/namuywam-es-embeddings"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}
payload = {"inputs": "hola mundo"}

res = requests.post(url, headers=headers, json=payload)

print("Status:", res.status_code)
print("Response:", res.text[:500])
