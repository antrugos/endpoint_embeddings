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

def handler(request):
    """Función principal para manejar requests en Vercel"""
    
    # Configurar CORS
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type': 'application/json'
    }
    
    # Manejar preflight OPTIONS
    if request.method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({'status': 'ok'})
        }
    
    # GET para verificar que la API funciona
    if request.method == 'GET':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'status': 'active',
                'message': 'Bot traductor español-namuy-wam funcionando'
            })
        }
    
    # POST para manejar webhooks de Telegram
    if request.method == 'POST':
        try:
            # Obtener el body del request
            if hasattr(request, 'body'):
                body = request.body
                if isinstance(body, bytes):
                    body = body.decode('utf-8')
            else:
                body = request.get_json()
                body = json.dumps(body) if isinstance(body, dict) else str(body)
            
            # Parsear el JSON del webhook de Telegram
            if isinstance(body, str):
                data = json.loads(body)
            else:
                data = body
            
            # Verificar si es un mensaje de texto
            if 'message' in data and 'text' in data['message']:
                chat_id = data['message']['chat']['id']
                text = data['message']['text']
                
                # Procesar el mensaje
                process_message(text, chat_id)
                
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({'status': 'processed'})
                }
            else:
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({'status': 'ignored'})
                }
                
        except Exception as e:
            print(f"Error: {e}")
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps({'error': str(e)})
            }
    
    # Método no permitido
    return {
        'statusCode': 405,
        'headers': headers,
        'body': json.dumps({'error': 'Method not allowed'})
    }

def process_message(text, chat_id):
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
            
            send_telegram_message(chat_id, help_message)
            return
        
        # Obtener traducción de Hugging Face
        translation = get_translation_from_hf(text)
        
        if translation:
            # Formatear respuesta
            response_text = f"🔄 Traducción:\n\n📝 Español: {text}\n🌿 Namuy-wam: {translation}"
        else:
            response_text = f"❌ No pude encontrar una traducción para '{text}'. Intenta con otra palabra."
        
        # Enviar respuesta por Telegram
        send_telegram_message(chat_id, response_text)
        
    except Exception as e:
        error_message = f"❌ Ocurrió un error al procesar tu mensaje: {str(e)}"
        send_telegram_message(chat_id, error_message)

def get_translation_from_hf(text):
    """Obtiene la traducción desde la API de Hugging Face"""
    try:
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Payload para el modelo de embeddings
        # Ajusta esto según tu modelo específico
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
        
        print(f"HF Response Status: {response.status_code}")
        print(f"HF Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Adapta esto según la estructura de respuesta de tu modelo
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

def send_telegram_message(chat_id, text):
    """Envía un mensaje por Telegram"""
    try:
        url = f"{TELEGRAM_API_URL}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        print(f"Telegram Response Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error enviando mensaje por Telegram: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error enviando mensaje por Telegram: {e}")

# Función por defecto que exporta Vercel
def lambda_handler(event, context):
    """Handler para AWS Lambda / Vercel"""
    
    class MockRequest:
        def __init__(self, event):
            self.method = event.get('httpMethod', event.get('requestContext', {}).get('http', {}).get('method', 'GET'))
            self.body = event.get('body', '{}')
            if isinstance(self.body, str):
                try:
                    self.json_body = json.loads(self.body)
                except:
                    self.json_body = {}
            else:
                self.json_body = self.body
        
        def get_json(self):
            return self.json_body
    
    mock_request = MockRequest(event)
    result = handler(mock_request)
    
    return {
        'statusCode': result['statusCode'],
        'headers': result['headers'],
        'body': result['body']
    }