import requests
import time
import os

from index import ChatBot

# —————————————————————————————————————————
# Configuración de Telegram
# —————————————————————————————————————————
TOKEN = os.getenv("BOT_TOKEN", "8186125106:AAG1K9UjX9T5GJIIg2c4-64U3b-DziyFlt0")
URL = f"https://api.telegram.org/bot{TOKEN}/"
offset = None  # para no reprocesar updates viejos

# Diccionario para rastrear la última actividad de cada usuario
ultima_actividad = {}
# Diccionario para rastrear si ya se envió el mensaje de inactividad
mensaje_inactividad_enviado = {}

def main():
    global offset
    # Inicializar el chatbot mejorado (carga FAQs, entrena modelos, etc.)
    print("Inicializando ChatBot…")
    chatbot = ChatBot(json_path="answers_FAQ.json")
    print("Chatbot IngeLean listo. Esperando mensajes…")

    while True:
        params = {"timeout": 60, "offset": offset}
        try:
            resp = requests.get(URL + "getUpdates", params=params, timeout=65).json()
        except Exception as e:
            print("Error en getUpdates:", e)
            time.sleep(5)
            continue

        ahora = time.time()

        # Verificar inactividad de usuarios
        for chat_id, timestamp in list(ultima_actividad.items()):
            if ahora - timestamp > 60 and not mensaje_inactividad_enviado.get(chat_id, False):
                payload = {
                    "chat_id": chat_id,
                    "text": "¿Sigues ahí? Si deseas finalizar la conversación, solo escríbeme 'adiós' o continúa preguntando."
                }
                try:
                    requests.post(URL + "sendMessage", data=payload)
                    mensaje_inactividad_enviado[chat_id] = True
                except Exception as e:
                    print("Error en sendMessage (inactividad):", e)

        for update in resp.get("result", []):
            offset = update["update_id"] + 1

            msg = update.get("message")
            if not msg or "text" not in msg:
                continue  # ignorar stickers, fotos, etc.

            chat_id = msg["chat"]["id"]
            texto = msg["text"].strip()

            # Actualizar última actividad
            ultima_actividad[chat_id] = time.time()
            mensaje_inactividad_enviado[chat_id] = False  # Reiniciar bandera al recibir mensaje

            # /start
            if texto.lower() == "/start":
                nombre = msg["from"].get("first_name", "")
                reply = (
                    f"¡Hola {nombre}! 🤖\n"
                    "Soy tu bot de FAQs de INGE-LEAN.\n"
                    "Escríbeme cualquier consulta y te responderé."
                )
            elif texto.lower() in ["adiós", "adios", "bye", "fin"]:
                reply = "¡Hasta luego! Si necesitas algo más, no dudes en escribirme."
            else:
                # Usamos el método de tu clase para predecir y generar respuesta
                reply = chatbot.predecir_intencion(texto_usuario=texto)

            # Enviar respuesta
            payload = {"chat_id": chat_id, "text": reply}
            try:
                requests.post(URL + "sendMessage", data=payload)
            except Exception as e:
                print("Error en sendMessage:", e)

        time.sleep(1)


if __name__ == "__main__":
    main()
