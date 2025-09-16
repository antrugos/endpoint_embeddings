import os
import requests
import json
import numpy as np
from http.server import BaseHTTPRequestHandler
from fastapi import FastAPI, Request
from urllib.parse import parse_qs

# ============================
# 1. Configuración
# ============================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face API Token
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

HF_API_URL = "https://jecwd9rddkv39uoi.us-east-1.aws.endpoints.huggingface.cloud"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Leer el contenido de la solicitud
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            # Parsear el JSON del webhook de Telegram
            data = json.loads(post_data.decode('utf-8'))
            
            # Verificar si es un mensaje de texto
            if 'message' in data and 'text' in data['message']:
                chat_id = data['message']['chat']['id']
                text = data['message']['text']
                
                # Procesar el mensaje
                response = self.process_message(text, chat_id)
                
                # Responder con éxito
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode())
            else:
                # Si no es un mensaje de texto, responder con éxito pero sin procesar
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ignored"}).encode())
                
        except Exception as e:
            print(f"Error: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def process_message(self, text, chat_id):
        """Procesa el mensaje y envía la respuesta traducida"""
        try:
            # Comando de ayuda
            if text.lower() in ['/start', '/help']:
                help_message = """¡Hola! 👋
                
Soy un bot traductor español ↔ namuy-wam

Simplemente envía una palabra en español y te doy la traducción en namuy-wam.

Ejemplo:
• Envías: hola
• Respondo: ka watirru"""
                
                self.send_telegram_message(chat_id, help_message)
                return
            
            # Obtener traducción de Hugging Face
            translation = self.get_translation_from_hf(text)
            
            if translation:
                # Formatear respuesta
                response_text = f"🔄 Traducción:\n\n📝 Español: {text}\n🌿 Namuy-wam: {translation}"
            else:
                response_text = f"❌ No pude encontrar una traducción para '{text}'. Intenta con otra palabra."
            
            # Enviar respuesta por Telegram
            self.send_telegram_message(chat_id, response_text)
            
        except Exception as e:
            error_message = f"❌ Ocurrió un error al procesar tu mensaje: {str(e)}"
            self.send_telegram_message(chat_id, error_message)
    
    def get_translation_from_hf(self, text):
        """Obtiene la traducción desde la API de Hugging Face"""
        try:
            headers = {
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            }
            
            # Payload para el modelo de embeddings
            payload = {
                "inputs": text,
                "parameters": {
                    "task": "translation",
                    "source_lang": "es",
                    "target_lang": "namuy-wam"
                }
            }
            
            # Realizar la petición a Hugging Face
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Adapta esto según la estructura de respuesta de tu modelo
                # Esto es un ejemplo genérico, ajusta según tu modelo específico
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', result[0].get('translation_text', None))
                elif isinstance(result, dict):
                    return result.get('generated_text', result.get('translation_text', None))
                else:
                    return str(result)
            else:
                print(f"Error en HF API: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("Timeout en la petición a Hugging Face")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error en la petición a Hugging Face: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado en HF: {e}")
            return None
    
    def send_telegram_message(self, chat_id, text):
        """Envía un mensaje por Telegram"""
        try:
            url = f"{TELEGRAM_API_URL}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code != 200:
                print(f"Error enviando mensaje por Telegram: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error enviando mensaje por Telegram: {e}")
    
    def do_GET(self):
        # Endpoint de salud para verificar que la API funciona
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {
            "status": "active",
            "message": "Bot traductor español-namuy-wam funcionando"
        }
        self.wfile.write(json.dumps(response).encode())
