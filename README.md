# 🤖 ChatBot HeaLink - INGE-LEAN

Un chatbot inteligente desarrollado para INGE-LEAN que proporciona respuestas automáticas sobre servicios de automatización industrial, desarrollo de hardware y software, ciberseguridad y más.

## 📋 Descripción

Este chatbot utiliza técnicas avanzadas de procesamiento de lenguaje natural (NLP) y machine learning para responder consultas sobre los servicios de INGE-LEAN, una empresa especializada en soluciones para la Industria 4.0.

### 🎯 Características Principales

- **Procesamiento de Lenguaje Natural Avanzado**: Utiliza spaCy y NLTK para análisis semántico
- **Múltiples Modelos de ML**: Ensemble de Random Forest, SVM y búsqueda semántica
- **Integración con Telegram**: Bot completamente funcional en Telegram
- **Respuestas Inteligentes**: Sistema de confianza adaptativo para respuestas precisas
- **Soporte en Español**: Optimizado para el idioma español

## 🚀 Servicios Cubiertos

El chatbot puede responder consultas sobre:

- **Registro de Clientes**: Proceso de registro y acceso a servicios
- **Automatización Industrial**: Procesos, tecnologías y beneficios
- **Desarrollo de Hardware**: PCBs, sistemas embebidos, circuitos electrónicos
- **Desarrollo de Software**: Aplicaciones a medida, visión por computador
- **Casos de Éxito**: Resultados y testimonios de proyectos anteriores
- **Tarjetas NFC**: Tecnología de transferencia de datos por proximidad
- **Ciberseguridad**: Protección de sistemas industriales y datos

## 📦 Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio**
   ```bash
   git clone <url-del-repositorio>
   cd chatbot_healink
   ```

2. **Crear entorno virtual**
   ```bash
   python -m venv venv
   ```

3. **Activar el entorno virtual**
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **Linux/Mac:**
   ```bash
   source venv/bin/activate
   ```

4. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

5. **Descargar modelo de spaCy**
   ```bash
   python -m spacy download es_core_news_md
   ```

## ⚙️ Configuración

### Configuración del Bot de Telegram

1. **Obtener Token de Bot**
   - Contacta a @BotFather en Telegram
   - Crea un nuevo bot con `/newbot`
   - Guarda el token proporcionado

2. **Configurar Variables de Entorno**
   
   Crea un archivo `.env` en la raíz del proyecto:
   ```env
   BOT_TOKEN=tu_token_aqui
   ```

   O modifica directamente el token en `bot.py`:
   ```python
   TOKEN = "tu_token_aqui"
   ```

## 🎮 Uso

### Ejecutar el Bot de Telegram

```bash
python bot.py
```

El bot estará disponible en Telegram y responderá automáticamente a las consultas de los usuarios.

### Uso Programático

```python
from index import ChatBot

# Inicializar el chatbot
chatbot = ChatBot(json_path="answers_FAQ.json")

# Hacer una consulta
respuesta = chatbot.predecir_intencion("¿Cómo me registro en INGE-LEAN?")
print(respuesta)
```

## 🏗️ Arquitectura del Sistema

### Componentes Principales

1. **`bot.py`**: Servidor principal del bot de Telegram
2. **`index.py`**: Clase ChatBot con toda la lógica de NLP y ML
3. **`answers_FAQ.json`**: Base de conocimientos con preguntas y respuestas
4. **`requirements.txt`**: Dependencias del proyecto

### Tecnologías Utilizadas

- **Procesamiento de Texto**: spaCy, NLTK, regex
- **Machine Learning**: scikit-learn, sentence-transformers
- **Modelos**: Random Forest, SVM, embeddings semánticos
- **API**: Telegram Bot API, requests
- **Lenguaje**: Python 3.8+

### Algoritmos Implementados

1. **Preprocesamiento de Texto**:
   - Normalización (minúsculas, acentos)
   - Lematización con spaCy
   - Expansión de sinónimos con WordNet

2. **Extracción de Características**:
   - TF-IDF con n-gramas
   - Embeddings semánticos (SentenceTransformers)

3. **Clasificación**:
   - Ensemble de múltiples modelos
   - Voting ponderado por confianza
   - Búsqueda semántica por similitud de coseno

## 📊 Estructura de Datos

### Formato de FAQs (answers_FAQ.json)

```json
[
    {
        "intent": "nombre_del_intent",
        "questions": [
            "Pregunta 1",
            "Pregunta 2",
            "..."
        ],
        "answer": "Respuesta completa para este intent"
    }
]
```

## 🔧 Personalización

### Añadir Nuevas Preguntas y Respuestas

1. Edita el archivo `answers_FAQ.json`
2. Añade nuevas entradas siguiendo el formato existente
3. Reinicia el bot para cargar los cambios

### Ajustar Modelos

Los parámetros de los modelos se pueden modificar en `index.py`:

- **TF-IDF**: `max_features`, `ngram_range`
- **Random Forest**: `n_estimators`, `max_depth`
- **SVM**: `kernel`, `class_weight`
- **Umbral de confianza**: `umbral_base`

## 📈 Rendimiento

El sistema utiliza múltiples técnicas para optimizar la precisión:

- **Augmentación de datos**: Generación automática de variaciones de preguntas
- **Ensemble learning**: Combinación de múltiples modelos
- **Umbral adaptativo**: Ajuste dinámico de la confianza mínima
- **Búsqueda semántica**: Respaldado por embeddings de alta calidad

## 🛠️ Mantenimiento

### Logs y Monitoreo

El bot incluye logs detallados para monitorear:
- Predicciones realizadas
- Niveles de confianza
- Errores de comunicación con Telegram

### Actualizaciones

Para actualizar el sistema:
1. Actualizar `answers_FAQ.json` con nuevas preguntas
2. Ajustar parámetros de modelos si es necesario
3. Reiniciar el bot

## 📝 Licencia

Este proyecto está desarrollado para INGE-LEAN. Todos los derechos reservados.

**Desarrollado con ❤️ para INGE-LEAN** 